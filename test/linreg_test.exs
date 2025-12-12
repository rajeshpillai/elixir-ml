defmodule MLNx.LinRegTest do
  use ExUnit.Case
  doctest MLNx.LinReg

  describe "predict/3" do
    test "predicts values correctly" do
      # Simple case: y = 2x + 3
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      w = Nx.tensor([2.0])
      b = Nx.tensor(3.0)

      y_hat = MLNx.LinReg.predict(x, w, b)
      expected = Nx.tensor([5.0, 7.0, 9.0])

      assert_all_close(y_hat, expected)
    end

    test "handles multiple features" do
      # y = 2*x1 + 3*x2 + 1
      x = Nx.tensor([[1.0, 1.0], [2.0, 2.0]])
      w = Nx.tensor([2.0, 3.0])
      b = Nx.tensor(1.0)

      y_hat = MLNx.LinReg.predict(x, w, b)
      expected = Nx.tensor([6.0, 11.0])

      assert_all_close(y_hat, expected)
    end
  end

  describe "mse/2" do
    test "calculates mean squared error correctly" do
      y_hat = Nx.tensor([1.0, 2.0, 3.0])
      y = Nx.tensor([1.5, 2.5, 3.5])

      loss = MLNx.LinReg.mse(y_hat, y)
      # MSE = mean((0.5)^2 + (0.5)^2 + (0.5)^2) = 0.25
      expected = Nx.tensor(0.25)

      assert_all_close(loss, expected)
    end

    test "returns zero for perfect predictions" do
      y_hat = Nx.tensor([1.0, 2.0, 3.0])
      y = Nx.tensor([1.0, 2.0, 3.0])

      loss = MLNx.LinReg.mse(y_hat, y)
      expected = Nx.tensor(0.0)

      assert_all_close(loss, expected)
    end
  end

  describe "train/3" do
    test "learns simple linear relationship" do
      # Generate data: y = 3x + 2 with some noise
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      y = Nx.tensor([5.0, 8.0, 11.0, 14.0, 17.0])

      {w, b} = MLNx.LinReg.train(x, y, iters: 1000, lr: 0.01)

      # Check if learned parameters are close to true values (w=3, b=2)
      assert_in_delta(Nx.to_number(w[0]), 3.0, 0.1)
      assert_in_delta(Nx.to_number(b), 2.0, 0.1)
    end

    test "learns with multiple features" do
      # Generate data: y = 2*x1 + 3*x2 + 1
      x = Nx.tensor([
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 3.0]
      ])
      y = Nx.tensor([6.0, 8.0, 9.0, 11.0, 16.0])

      {w, b} = MLNx.LinReg.train(x, y, iters: 2000, lr: 0.01)

      # Check predictions on training data
      y_pred = MLNx.LinReg.predict(x, w, b)
      loss = MLNx.LinReg.mse(y_pred, y)

      # Loss should be very small after training
      assert Nx.to_number(loss) < 0.1
    end

    test "uses default hyperparameters" do
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      y = Nx.tensor([2.0, 4.0, 6.0])

      # Should not raise error with default params
      {w, b} = MLNx.LinReg.train(x, y)

      assert is_struct(w, Nx.Tensor)
      assert is_struct(b, Nx.Tensor)
    end
  end

  # Helper function to compare tensors with tolerance
  defp assert_all_close(actual, expected, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    rtol = Keyword.get(opts, :rtol, 1.0e-5)

    diff = Nx.abs(Nx.subtract(actual, expected))
    threshold = Nx.add(atol, Nx.multiply(rtol, Nx.abs(expected)))

    assert Nx.all(Nx.less_equal(diff, threshold)) |> Nx.to_number() == 1,
           "Tensors not close enough.\nActual: #{inspect(actual)}\nExpected: #{inspect(expected)}"
  end
end
