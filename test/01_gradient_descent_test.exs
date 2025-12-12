defmodule MLNx.GradientDescentTest do
  use ExUnit.Case
  doctest MLNx.GradientDescent

  describe "step/3" do
    test "performs single gradient descent step correctly" do
      # Given parameters and gradient
      params = Nx.tensor([1.0, 2.0])
      gradient = Nx.tensor([0.5, -0.3])
      lr = 0.1

      # When we take one step
      new_params = MLNx.GradientDescent.step(params, gradient, lr)

      # Then: θ_new = θ_old - α * ∇f
      # [1.0, 2.0] - 0.1 * [0.5, -0.3] = [0.95, 2.03]
      expected = Nx.tensor([0.95, 2.03])
      assert_all_close(new_params, expected)
    end

    test "moves in opposite direction of gradient" do
      # Gradient points up, so step should move down
      params = Nx.tensor(5.0)
      gradient = Nx.tensor(2.0)  # Positive gradient (upward slope)
      lr = 0.1

      new_params = MLNx.GradientDescent.step(params, gradient, lr)

      # Should move down (decrease)
      assert Nx.to_number(new_params) < Nx.to_number(params)
    end
  end

  describe "gradient_norm/1" do
    test "computes L1 norm of gradient" do
      gradient = Nx.tensor([3.0, -4.0])
      norm = MLNx.GradientDescent.gradient_norm(gradient)

      # L1 norm = |3| + |-4| = 7
      assert_all_close(norm, Nx.tensor(7.0))
    end

    test "returns zero for zero gradient" do
      gradient = Nx.tensor([0.0, 0.0, 0.0])
      norm = MLNx.GradientDescent.gradient_norm(gradient)

      assert_all_close(norm, Nx.tensor(0.0))
    end
  end

  describe "minimize_quadratic/3" do
    test "finds minimum of quadratic function" do
      # Minimize f(x) = (x - 3)²
      # Minimum is at x = 3
      initial_x = Nx.tensor(10.0)
      target = Nx.tensor(3.0)

      result = MLNx.GradientDescent.minimize_quadratic(
        initial_x,
        target,
        learning_rate: 0.1,
        max_iters: 1000
      )

      # Should converge to target
      assert_in_delta(Nx.to_number(result), 3.0, 0.01)
    end

    test "converges from negative starting point" do
      # Start at x = -5, target x = 2
      initial_x = Nx.tensor(-5.0)
      target = Nx.tensor(2.0)

      result = MLNx.GradientDescent.minimize_quadratic(
        initial_x,
        target,
        learning_rate: 0.1,
        max_iters: 1000
      )

      assert_in_delta(Nx.to_number(result), 2.0, 0.01)
    end

    test "tracks history when requested" do
      initial_x = Nx.tensor(5.0)
      target = Nx.tensor(0.0)

      {_final_x, history} = MLNx.GradientDescent.minimize_quadratic(
        initial_x,
        target,
        learning_rate: 0.1,
        max_iters: 100,
        track_history: true
      )

      # History should be a list of {params, gradient_norm} tuples
      assert is_list(history)
      assert length(history) > 0

      # First entry should be initial state
      {first_params, _first_grad_norm} = hd(history)
      assert_in_delta(Nx.to_number(first_params), 5.0, 0.01)

      # Gradient norm should decrease over time (approaching minimum)
      gradient_norms = Enum.map(history, fn {_, norm} -> norm end)
      assert List.last(gradient_norms) < hd(gradient_norms)
    end
  end

  describe "optimize/3 with custom function" do
    test "minimizes simple linear function" do
      # Minimize f(x) = 2x, gradient is constant 2
      # Minimum is at x → -∞, but we'll see it moving in negative direction
      initial_x = Nx.tensor(10.0)
      gradient_fn = fn _x -> Nx.tensor(2.0) end

      result = MLNx.GradientDescent.optimize(
        initial_x,
        gradient_fn,
        learning_rate: 0.1,
        max_iters: 10  # Just a few steps
      )

      # Should move in negative direction
      assert Nx.to_number(result) < 10.0
    end

    test "stops when gradient is near zero" do
      # Function with zero gradient at x = 5
      # f(x) = (x - 5)², gradient = 2(x - 5)
      initial_x = Nx.tensor(8.0)
      gradient_fn = fn x ->
        Nx.multiply(2.0, Nx.subtract(x, 5.0))
      end

      result = MLNx.GradientDescent.optimize(
        initial_x,
        gradient_fn,
        learning_rate: 0.1,
        max_iters: 1000,
        tolerance: 1.0e-4
      )

      # Should converge to x = 5
      assert_in_delta(Nx.to_number(result), 5.0, 0.01)
    end

    test "respects maximum iterations" do
      # Use very small learning rate so it won't converge quickly
      initial_x = Nx.tensor(100.0)
      gradient_fn = fn x -> Nx.multiply(2.0, x) end

      {_result, history} = MLNx.GradientDescent.optimize(
        initial_x,
        gradient_fn,
        learning_rate: 0.001,
        max_iters: 50,
        track_history: true
      )

      # Should stop at exactly 50 iterations
      assert length(history) == 50
    end
  end

  describe "learning rate effects" do
    test "small learning rate converges slowly" do
      initial_x = Nx.tensor(10.0)
      target = Nx.tensor(0.0)

      {_, history_slow} = MLNx.GradientDescent.minimize_quadratic(
        initial_x,
        target,
        learning_rate: 0.01,  # Small LR
        max_iters: 100,
        track_history: true
      )

      {_, history_fast} = MLNx.GradientDescent.minimize_quadratic(
        initial_x,
        target,
        learning_rate: 0.1,   # Larger LR
        max_iters: 100,
        track_history: true
      )

      # With larger LR, gradient norm should decrease faster
      # (fewer iterations to reach same gradient norm)
      slow_final_norm = history_slow |> List.last() |> elem(1)
      fast_final_norm = history_fast |> List.last() |> elem(1)

      assert fast_final_norm < slow_final_norm
    end

    test "large learning rate can overshoot" do
      # With very large learning rate, might not converge well
      initial_x = Nx.tensor(1.0)
      target = Nx.tensor(0.0)

      # Reasonable learning rate
      result_good = MLNx.GradientDescent.minimize_quadratic(
        initial_x,
        target,
        learning_rate: 0.1,
        max_iters: 100
      )

      # Too large learning rate (> 1.0 for this problem)
      result_bad = MLNx.GradientDescent.minimize_quadratic(
        initial_x,
        target,
        learning_rate: 1.5,
        max_iters: 100
      )

      # Good LR should get closer to target
      good_error = abs(Nx.to_number(result_good) - 0.0)
      bad_error = abs(Nx.to_number(result_bad) - 0.0)

      assert good_error < bad_error
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
