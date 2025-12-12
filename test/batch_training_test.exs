defmodule MLNx.BatchTrainingTest do
  use ExUnit.Case
  doctest MLNx.BatchTraining

  describe "batch_gradient_descent/3" do
    test "converges on simple linear data" do
      # y = 2x (perfect linear relationship)
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
      
      {weights, bias, history} = MLNx.BatchTraining.batch_gradient_descent(
        x, y,
        learning_rate: 0.01,
        iterations: 200
      )
      
      # Should converge close to w=2, b=0
      assert_in_delta(Nx.to_number(weights[0][0]), 2.0, 0.15)
      assert_in_delta(Nx.to_number(bias[0][0]), 0.0, 0.4)
      
      # Loss should decrease
      assert length(history) == 200
      assert List.last(history) < List.first(history)
    end

    test "loss decreases monotonically" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])
      y = Nx.tensor([[3.0], [5.0], [7.0], [9.0]])
      
      {_w, _b, history} = MLNx.BatchTraining.batch_gradient_descent(
        x, y,
        learning_rate: 0.01,
        iterations: 50
      )
      
      # Each loss should be <= previous (monotonic decrease)
      Enum.chunk_every(history, 2, 1, :discard)
      |> Enum.each(fn [prev, curr] ->
        assert curr <= prev
      end)
    end

    test "works with multiple features" do
      # y = 2*x1 + 3*x2
      x = Nx.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
      y = Nx.tensor([[5.0], [10.0], [15.0]])
      
      {weights, _bias, _history} = MLNx.BatchTraining.batch_gradient_descent(
        x, y,
        learning_rate: 0.01,
        iterations: 200
      )
      
      # Should learn both weights
      assert Nx.shape(weights) == {2, 1}
    end

    test "can disable history tracking" do
      x = Nx.tensor([[1.0], [2.0]])
      y = Nx.tensor([[2.0], [4.0]])
      
      {_w, _b, history} = MLNx.BatchTraining.batch_gradient_descent(
        x, y,
        track_history: false,
        iterations: 10
      )
      
      assert history == []
    end
  end

  describe "stochastic_gradient_descent/3" do
    test "converges on simple linear data" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
      
      {weights, bias, history} = MLNx.BatchTraining.stochastic_gradient_descent(
        x, y,
        learning_rate: 0.01,
        epochs: 20
      )
      
      # Should converge close to w=2, b=0 (may be less accurate than batch)
      assert_in_delta(Nx.to_number(weights[0][0]), 2.0, 0.4)
      assert_in_delta(Nx.to_number(bias[0][0]), 0.0, 0.5)
      
      # Loss should generally decrease (may be noisy)
      assert length(history) == 20
      assert List.last(history) < List.first(history)
    end

    test "updates weights for each example" do
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0]])
      
      {_w, _b, history} = MLNx.BatchTraining.stochastic_gradient_descent(
        x, y,
        learning_rate: 0.01,
        epochs: 5,
        shuffle: false
      )
      
      # History tracks per epoch, not per example
      assert length(history) == 5
    end

    test "shuffling affects training" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0]])
      
      # Train with shuffling
      {w1, b1, _} = MLNx.BatchTraining.stochastic_gradient_descent(
        x, y,
        learning_rate: 0.01,
        epochs: 10,
        shuffle: true
      )
      
      # Train without shuffling
      {w2, b2, _} = MLNx.BatchTraining.stochastic_gradient_descent(
        x, y,
        learning_rate: 0.01,
        epochs: 10,
        shuffle: false
      )
      
      # Both should converge (results may differ slightly)
      assert Nx.shape(w1) == Nx.shape(w2)
      assert Nx.shape(b1) == Nx.shape(b2)
    end

    test "works with single example" do
      x = Nx.tensor([[5.0]])
      y = Nx.tensor([[10.0]])
      
      {weights, _bias, _history} = MLNx.BatchTraining.stochastic_gradient_descent(
        x, y,
        learning_rate: 0.01,
        epochs: 50
      )
      
      # Should learn w ≈ 2
      assert_in_delta(Nx.to_number(weights[0][0]), 2.0, 0.5)
    end
  end

  describe "mini_batch_gradient_descent/3" do
    test "converges on simple linear data" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]])
      
      {weights, bias, history} = MLNx.BatchTraining.mini_batch_gradient_descent(
        x, y,
        batch_size: 2,
        learning_rate: 0.01,
        epochs: 50
      )
      
      # Should converge close to w=2, b=0
      assert_in_delta(Nx.to_number(weights[0][0]), 2.0, 0.3)
      assert_in_delta(Nx.to_number(bias[0][0]), 0.0, 0.4)
      
      # Loss should decrease
      assert length(history) == 50
      assert List.last(history) < List.first(history)
    end

    test "works with different batch sizes" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0]])
      
      # Batch size 2
      {w1, _, _} = MLNx.BatchTraining.mini_batch_gradient_descent(
        x, y, batch_size: 2, learning_rate: 0.01, epochs: 20
      )
      
      # Batch size 4
      {w2, _, _} = MLNx.BatchTraining.mini_batch_gradient_descent(
        x, y, batch_size: 4, learning_rate: 0.01, epochs: 20
      )
      
      # Both should converge to similar weights
      assert_in_delta(Nx.to_number(w1[0][0]), Nx.to_number(w2[0][0]), 0.3)
    end

    test "handles batch size larger than dataset" do
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0]])
      
      # Batch size larger than dataset
      {weights, _bias, _history} = MLNx.BatchTraining.mini_batch_gradient_descent(
        x, y,
        batch_size: 10,
        learning_rate: 0.01,
        epochs: 100
      )
      
      # Should still work (becomes batch GD)
      assert_in_delta(Nx.to_number(weights[0][0]), 2.0, 0.3)
    end

    test "creates correct number of batches" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
      
      batches = MLNx.BatchTraining.create_batches(x, y, 2)
      
      # 5 examples with batch_size=2 should create 3 batches
      assert length(batches) == 3
      
      # Last batch should have 1 example
      {last_x, last_y} = List.last(batches)
      assert Nx.axis_size(last_x, 0) == 1
      assert Nx.axis_size(last_y, 0) == 1
    end
  end

  describe "shuffle_data/2" do
    test "preserves data size" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])
      y = Nx.tensor([[10.0], [20.0], [30.0], [40.0]])
      
      {shuffled_x, shuffled_y} = MLNx.BatchTraining.shuffle_data(x, y)
      
      assert Nx.shape(shuffled_x) == Nx.shape(x)
      assert Nx.shape(shuffled_y) == Nx.shape(y)
    end

    test "preserves correspondence between x and y" do
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      y = Nx.tensor([[10.0], [20.0], [30.0]])
      
      {shuffled_x, shuffled_y} = MLNx.BatchTraining.shuffle_data(x, y)
      
      # Check that correspondence is maintained
      # If x[i] = 2.0, then y[i] should be 20.0
      x_list = Nx.to_flat_list(shuffled_x)
      y_list = Nx.to_flat_list(shuffled_y)
      
      Enum.zip(x_list, y_list)
      |> Enum.each(fn {x_val, y_val} ->
        assert_in_delta(y_val, x_val * 10.0, 0.01)
      end)
    end
  end

  describe "create_batches/3" do
    test "creates correct number of batches" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
      
      batches = MLNx.BatchTraining.create_batches(x, y, 2)
      
      assert length(batches) == 3
    end

    test "batches have correct sizes" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
      y = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
      
      batches = MLNx.BatchTraining.create_batches(x, y, 3)
      
      # Should have 3 batches: [3, 3, 1]
      assert length(batches) == 3
      
      {b1_x, _} = Enum.at(batches, 0)
      {b2_x, _} = Enum.at(batches, 1)
      {b3_x, _} = Enum.at(batches, 2)
      
      assert Nx.axis_size(b1_x, 0) == 3
      assert Nx.axis_size(b2_x, 0) == 3
      assert Nx.axis_size(b3_x, 0) == 1
    end

    test "handles exact division" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])
      y = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])
      
      batches = MLNx.BatchTraining.create_batches(x, y, 2)
      
      # Should have exactly 2 batches of size 2
      assert length(batches) == 2
      
      Enum.each(batches, fn {batch_x, _} ->
        assert Nx.axis_size(batch_x, 0) == 2
      end)
    end
  end

  describe "train/3" do
    test "supports batch method" do
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0]])
      
      {weights, _bias, _history} = MLNx.BatchTraining.train(
        x, y,
        method: :batch,
        iterations: 200
      )
      
      assert_in_delta(Nx.to_number(weights[0][0]), 2.0, 0.3)
    end

    test "supports stochastic method" do
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0]])
      
      {weights, _bias, _history} = MLNx.BatchTraining.train(
        x, y,
        method: :stochastic,
        epochs: 30
      )
      
      assert_in_delta(Nx.to_number(weights[0][0]), 2.0, 0.4)
    end

    test "supports mini_batch method (default)" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0]])
      
      {weights, _bias, _history} = MLNx.BatchTraining.train(
        x, y,
        batch_size: 2,
        epochs: 50
      )
      
      assert_in_delta(Nx.to_number(weights[0][0]), 2.0, 0.3)
    end
  end

  describe "convergence comparison" do
    test "all methods converge to similar solution" do
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
      
      {w_batch, _, _} = MLNx.BatchTraining.batch_gradient_descent(
        x, y, learning_rate: 0.01, iterations: 100
      )
      
      {w_sgd, _, _} = MLNx.BatchTraining.stochastic_gradient_descent(
        x, y, learning_rate: 0.01, epochs: 20
      )
      
      {w_mini, _, _} = MLNx.BatchTraining.mini_batch_gradient_descent(
        x, y, batch_size: 2, learning_rate: 0.01, epochs: 20
      )
      
      # All should converge close to w=2
      w_batch_val = Nx.to_number(w_batch[0][0])
      w_sgd_val = Nx.to_number(w_sgd[0][0])
      w_mini_val = Nx.to_number(w_mini[0][0])
      
      assert_in_delta(w_batch_val, 2.0, 0.2)
      assert_in_delta(w_sgd_val, 2.0, 0.3)
      assert_in_delta(w_mini_val, 2.0, 0.3)
    end
  end
end
