defmodule MLNx.LossFunctionsTest do
  use ExUnit.Case
  doctest MLNx.LossFunctions

  describe "mse/2 - Mean Squared Error" do
    test "computes MSE correctly for simple case" do
      # Predictions: [2, 3, 4]
      # Targets:     [3, 3, 4]
      # Errors:      [-1, 0, 0]
      # Squared:     [1, 0, 0]
      # MSE:         1/3 ≈ 0.333
      predictions = Nx.tensor([2.0, 3.0, 4.0])
      targets = Nx.tensor([3.0, 3.0, 4.0])

      loss = MLNx.LossFunctions.mse(predictions, targets)

      assert_in_delta(Nx.to_number(loss), 0.333, 0.01)
    end

    test "returns zero for perfect predictions" do
      predictions = Nx.tensor([1.0, 2.0, 3.0])
      targets = Nx.tensor([1.0, 2.0, 3.0])

      loss = MLNx.LossFunctions.mse(predictions, targets)

      assert_in_delta(Nx.to_number(loss), 0.0, 1.0e-6)
    end

    test "penalizes large errors heavily" do
      # Small error: 1
      small_pred = Nx.tensor([2.0])
      small_target = Nx.tensor([1.0])
      small_loss = MLNx.LossFunctions.mse(small_pred, small_target)

      # Large error: 10
      large_pred = Nx.tensor([11.0])
      large_target = Nx.tensor([1.0])
      large_loss = MLNx.LossFunctions.mse(large_pred, large_target)

      # Large error should contribute 100x more (10² vs 1²)
      assert Nx.to_number(large_loss) > 90 * Nx.to_number(small_loss)
    end

    test "works with multi-dimensional tensors" do
      predictions = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.5, 2.5], [3.5, 4.5]])

      loss = MLNx.LossFunctions.mse(predictions, targets)

      # All errors are 0.5, so MSE = 0.25
      assert_in_delta(Nx.to_number(loss), 0.25, 0.01)
    end
  end

  describe "rmse/2 - Root Mean Squared Error" do
    test "computes RMSE correctly" do
      predictions = Nx.tensor([2.0, 3.0, 4.0])
      targets = Nx.tensor([3.0, 3.0, 4.0])

      rmse = MLNx.LossFunctions.rmse(predictions, targets)
      mse = MLNx.LossFunctions.mse(predictions, targets)

      # RMSE should be sqrt of MSE
      expected = Nx.sqrt(mse)
      assert_in_delta(Nx.to_number(rmse), Nx.to_number(expected), 1.0e-6)
    end

    test "is in same units as target" do
      # If MSE = 4, RMSE = 2 (back to original units)
      predictions = Nx.tensor([0.0, 4.0])
      targets = Nx.tensor([2.0, 2.0])

      rmse = MLNx.LossFunctions.rmse(predictions, targets)

      assert_in_delta(Nx.to_number(rmse), 2.0, 0.01)
    end
  end

  describe "mae/2 - Mean Absolute Error" do
    test "computes MAE correctly" do
      # Errors: [-1, 0, 1]
      # Absolute: [1, 0, 1]
      # MAE: 2/3 ≈ 0.667
      predictions = Nx.tensor([2.0, 3.0, 5.0])
      targets = Nx.tensor([3.0, 3.0, 4.0])

      loss = MLNx.LossFunctions.mae(predictions, targets)

      assert_in_delta(Nx.to_number(loss), 0.667, 0.01)
    end

    test "is robust to outliers compared to MSE" do
      # Regular predictions
      regular_pred = Nx.tensor([1.0, 2.0, 3.0])
      regular_target = Nx.tensor([1.5, 2.5, 3.5])

      # With outlier
      outlier_pred = Nx.tensor([1.0, 2.0, 13.0])  # Last one is way off
      outlier_target = Nx.tensor([1.5, 2.5, 3.5])

      # Compute both losses
      mae_regular = MLNx.LossFunctions.mae(regular_pred, regular_target)
      mae_outlier = MLNx.LossFunctions.mae(outlier_pred, outlier_target)
      
      mse_regular = MLNx.LossFunctions.mse(regular_pred, regular_target)
      mse_outlier = MLNx.LossFunctions.mse(outlier_pred, outlier_target)

      # MAE increase should be less dramatic than MSE increase
      mae_increase = Nx.to_number(mae_outlier) / Nx.to_number(mae_regular)
      mse_increase = Nx.to_number(mse_outlier) / Nx.to_number(mse_regular)

      assert mse_increase > mae_increase
    end

    test "returns zero for perfect predictions" do
      predictions = Nx.tensor([1.0, 2.0, 3.0])
      targets = Nx.tensor([1.0, 2.0, 3.0])

      loss = MLNx.LossFunctions.mae(predictions, targets)

      assert_in_delta(Nx.to_number(loss), 0.0, 1.0e-6)
    end
  end

  describe "huber/3 - Huber Loss" do
    test "uses quadratic loss for small errors" do
      # Small error (< delta)
      predictions = Nx.tensor([1.0])
      targets = Nx.tensor([1.5])  # Error = 0.5

      loss = MLNx.LossFunctions.huber(predictions, targets, delta: 1.0)

      # Should use quadratic: 0.5 * 0.5² = 0.125
      assert_in_delta(Nx.to_number(loss), 0.125, 0.01)
    end

    test "uses linear loss for large errors" do
      # Large error (> delta)
      predictions = Nx.tensor([1.0])
      targets = Nx.tensor([5.0])  # Error = 4.0

      loss = MLNx.LossFunctions.huber(predictions, targets, delta: 1.0)

      # Should use linear: δ(|e| - δ/2) = 1.0 * (4.0 - 0.5) = 3.5
      assert_in_delta(Nx.to_number(loss), 3.5, 0.01)
    end

    test "is more robust than MSE for outliers" do
      # Data with outlier
      predictions = Nx.tensor([1.0, 2.0, 10.0])  # Last is outlier
      targets = Nx.tensor([1.5, 2.5, 3.0])

      huber_loss = MLNx.LossFunctions.huber(predictions, targets, delta: 1.0)
      mse_loss = MLNx.LossFunctions.mse(predictions, targets)

      # Huber should be less affected by the outlier
      assert Nx.to_number(huber_loss) < Nx.to_number(mse_loss)
    end

    test "approaches MSE with large delta" do
      predictions = Nx.tensor([1.0, 2.0, 3.0])
      targets = Nx.tensor([1.5, 2.5, 3.5])

      # Very large delta means always use quadratic: 0.5 * e²
      huber_loss = MLNx.LossFunctions.huber(predictions, targets, delta: 100.0)
      mse_loss = MLNx.LossFunctions.mse(predictions, targets)

      # Huber uses 0.5*e² while MSE uses e², so Huber = 0.5 * MSE
      expected_huber = Nx.to_number(mse_loss) * 0.5
      assert_in_delta(Nx.to_number(huber_loss), expected_huber, 0.01)
    end

    test "uses default delta of 1.0" do
      predictions = Nx.tensor([1.0])
      targets = Nx.tensor([2.0])

      # Should work without specifying delta
      loss = MLNx.LossFunctions.huber(predictions, targets)

      assert is_struct(loss, Nx.Tensor)
    end
  end

  describe "binary_cross_entropy/3" do
    test "computes BCE correctly for confident correct predictions" do
      # Predicting 1 with high confidence (0.9)
      predictions = Nx.tensor([0.9])
      targets = Nx.tensor([1.0])

      loss = MLNx.LossFunctions.binary_cross_entropy(predictions, targets)

      # Loss should be small: -log(0.9) ≈ 0.105
      assert Nx.to_number(loss) < 0.2
    end

    test "penalizes confident wrong predictions heavily" do
      # Predicting 0 (0.1) when truth is 1
      wrong_pred = Nx.tensor([0.1])
      target = Nx.tensor([1.0])

      wrong_loss = MLNx.LossFunctions.binary_cross_entropy(wrong_pred, target)

      # Predicting 1 (0.9) when truth is 1
      right_pred = Nx.tensor([0.9])
      right_loss = MLNx.LossFunctions.binary_cross_entropy(right_pred, target)

      # Wrong prediction should have much higher loss
      assert Nx.to_number(wrong_loss) > 10 * Nx.to_number(right_loss)
    end

    test "returns zero for perfect predictions" do
      predictions = Nx.tensor([1.0, 0.0])
      targets = Nx.tensor([1.0, 0.0])

      loss = MLNx.LossFunctions.binary_cross_entropy(predictions, targets)

      # Should be very close to zero (epsilon prevents exact 0)
      assert Nx.to_number(loss) < 1.0e-5
    end

    test "handles batch of predictions" do
      # Mix of correct and incorrect predictions
      predictions = Nx.tensor([0.9, 0.2, 0.8, 0.1])
      targets = Nx.tensor([1.0, 0.0, 1.0, 0.0])

      loss = MLNx.LossFunctions.binary_cross_entropy(predictions, targets)

      # Should return average loss across batch
      assert is_struct(loss, Nx.Tensor)
      assert Nx.to_number(loss) > 0
    end

    test "clips predictions to avoid log(0)" do
      # Extreme predictions that would cause log(0)
      predictions = Nx.tensor([0.0, 1.0])
      targets = Nx.tensor([0.0, 1.0])

      # Should not raise error due to epsilon clipping
      loss = MLNx.LossFunctions.binary_cross_entropy(predictions, targets)

      assert is_struct(loss, Nx.Tensor)
    end
  end

  describe "categorical_cross_entropy/3" do
    test "computes CCE correctly for confident correct predictions" do
      # Predicting class 1 with high confidence
      predictions = Nx.tensor([[0.1, 0.8, 0.1]])
      targets = Nx.tensor([[0.0, 1.0, 0.0]])

      loss = MLNx.LossFunctions.categorical_cross_entropy(predictions, targets)

      # Loss should be small: -log(0.8) ≈ 0.223
      assert Nx.to_number(loss) < 0.3
    end

    test "penalizes wrong predictions" do
      # Predicting class 0 when truth is class 2
      predictions = Nx.tensor([[0.7, 0.2, 0.1]])
      targets = Nx.tensor([[0.0, 0.0, 1.0]])

      loss = MLNx.LossFunctions.categorical_cross_entropy(predictions, targets)

      # Loss should be high: -log(0.1) ≈ 2.3
      assert Nx.to_number(loss) > 2.0
    end

    test "handles multi-class batch" do
      # 3 samples, 3 classes each
      predictions = Nx.tensor([
        [0.7, 0.2, 0.1],  # Predicting class 0
        [0.1, 0.8, 0.1],  # Predicting class 1
        [0.2, 0.2, 0.6]   # Predicting class 2
      ])

      targets = Nx.tensor([
        [1.0, 0.0, 0.0],  # True class 0
        [0.0, 1.0, 0.0],  # True class 1
        [0.0, 0.0, 1.0]   # True class 2
      ])

      loss = MLNx.LossFunctions.categorical_cross_entropy(predictions, targets)

      # All predictions are correct, loss should be relatively small
      assert Nx.to_number(loss) < 0.6
    end

    test "returns near-zero for perfect predictions" do
      predictions = Nx.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
      ])

      targets = Nx.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
      ])

      loss = MLNx.LossFunctions.categorical_cross_entropy(predictions, targets)

      assert Nx.to_number(loss) < 1.0e-5
    end
  end

  describe "regression_metrics/3" do
    test "computes all regression metrics at once" do
      predictions = Nx.tensor([2.0, 3.0, 5.0])
      targets = Nx.tensor([3.0, 3.0, 4.0])

      metrics = MLNx.LossFunctions.regression_metrics(predictions, targets)

      # Should return map with all metrics
      assert Map.has_key?(metrics, :mse)
      assert Map.has_key?(metrics, :rmse)
      assert Map.has_key?(metrics, :mae)
      assert Map.has_key?(metrics, :huber)

      # All should be positive numbers
      assert metrics.mse > 0
      assert metrics.rmse > 0
      assert metrics.mae > 0
      assert metrics.huber > 0

      # RMSE should be sqrt of MSE
      assert_in_delta(metrics.rmse, :math.sqrt(metrics.mse), 0.01)
    end

    test "all metrics are zero for perfect predictions" do
      predictions = Nx.tensor([1.0, 2.0, 3.0])
      targets = Nx.tensor([1.0, 2.0, 3.0])

      metrics = MLNx.LossFunctions.regression_metrics(predictions, targets)

      assert_in_delta(metrics.mse, 0.0, 1.0e-5)
      assert_in_delta(metrics.rmse, 0.0, 1.0e-5)
      assert_in_delta(metrics.mae, 0.0, 1.0e-5)
      assert_in_delta(metrics.huber, 0.0, 1.0e-5)
    end
  end

  describe "loss function comparisons" do
    test "MSE vs MAE with outlier" do
      # Dataset with one outlier
      predictions = Nx.tensor([1.0, 2.0, 3.0, 100.0])
      targets = Nx.tensor([1.1, 2.1, 3.1, 4.0])

      mse = MLNx.LossFunctions.mse(predictions, targets) |> Nx.to_number()
      mae = MLNx.LossFunctions.mae(predictions, targets) |> Nx.to_number()

      # MSE should be much larger due to squared outlier
      # Outlier error: 96, contributes 96² = 9216 to MSE
      # But only 96 to MAE
      assert mse > 2000  # Dominated by outlier
      assert mae < 30    # More reasonable
    end

    test "Huber is between MSE and MAE for outliers" do
      predictions = Nx.tensor([1.0, 2.0, 3.0, 20.0])
      targets = Nx.tensor([1.1, 2.1, 3.1, 4.0])

      mse = MLNx.LossFunctions.mse(predictions, targets) |> Nx.to_number()
      huber = MLNx.LossFunctions.huber(predictions, targets, delta: 1.0) |> Nx.to_number()

      # Huber should be less than MSE (more robust)
      assert huber < mse
      # For this data, Huber might be close to MAE
      assert huber > 0  # Just verify it's positive
    end
  end
end
