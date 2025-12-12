defmodule MLNx.GradientDescent do
  @moduledoc """
  Gradient Descent - The Foundation of Machine Learning Optimization

  ## What is Gradient Descent?

  Gradient descent is an iterative optimization algorithm used to find the minimum
  of a function. In machine learning, we use it to find the parameters (weights)
  that minimize our loss function.

  ## The Intuition

  Imagine you're standing on a mountain in thick fog and want to reach the valley.
  You can't see far, but you can feel which direction slopes downward. Gradient
  descent works the same way:

  1. Start at a random position (random weights)
  2. Look around and find the steepest downward slope (compute gradient)
  3. Take a step in that direction (update weights)
  4. Repeat until you reach the bottom (convergence)

  ## The Mathematics

  For a function f(θ), the gradient descent update rule is:

      θ_new = θ_old - α * ∇f(θ_old)

  Where:
  - θ (theta) = parameters we're optimizing
  - α (alpha) = learning rate (step size)
  - ∇f (nabla f) = gradient (vector of partial derivatives)

  ## Why Does It Work?

  The gradient ∇f points in the direction of steepest ASCENT. By moving in the
  negative direction (-∇f), we move toward the minimum. The learning rate α
  controls how big our steps are.

  ## Key Concepts

  ### Learning Rate (α)
  - Too small: slow convergence, many iterations needed
  - Too large: might overshoot minimum, diverge
  - Just right: efficient convergence

  ### Convergence
  We stop when:
  - Gradient is close to zero (we're at a minimum)
  - Loss stops decreasing significantly
  - Maximum iterations reached

  ### Local vs Global Minima
  - Convex functions (like linear regression): one global minimum
  - Non-convex functions (like neural networks): many local minima
  """

  import Nx.Defn

  @doc """
  Performs gradient descent to minimize a function.

  ## Parameters

  - `initial_params` - Starting point for optimization (tensor)
  - `gradient_fn` - Function that computes gradient at given params
  - `opts` - Keyword list of options:
    - `:learning_rate` - Step size (default: 0.01)
    - `:max_iters` - Maximum iterations (default: 1000)
    - `:tolerance` - Convergence threshold (default: 1.0e-6)
    - `:track_history` - Whether to save parameter history (default: false)

  ## Returns

  If `track_history: false` (default):
    Final optimized parameters

  If `track_history: true`:
    `{final_params, history}` where history is list of {params, gradient_norm}

  ## Example

      # Minimize f(x) = x^2, which has minimum at x = 0
      gradient_fn = fn x -> 2 * x end
      
      result = MLNx.GradientDescent.optimize(
        Nx.tensor(5.0),
        gradient_fn,
        learning_rate: 0.1,
        max_iters: 100
      )
      
      # result ≈ 0.0
  """
  def optimize(initial_params, gradient_fn, opts \\ []) do
    lr = Keyword.get(opts, :learning_rate, 0.01)
    max_iters = Keyword.get(opts, :max_iters, 1000)
    tolerance = Keyword.get(opts, :tolerance, 1.0e-6)
    track_history = Keyword.get(opts, :track_history, false)

    if track_history do
      optimize_with_history(initial_params, gradient_fn, lr, max_iters, tolerance)
    else
      optimize_simple(initial_params, gradient_fn, lr, max_iters, tolerance)
    end
  end

  # Simple optimization without tracking history
  defp optimize_simple(params, gradient_fn, lr, max_iters, tolerance) do
    Enum.reduce_while(1..max_iters, params, fn iter, current_params ->
      # Compute gradient at current position
      # The gradient tells us which direction increases the function
      grad = gradient_fn.(current_params)

      # Check convergence: if gradient is very small, we're at a minimum
      # ||∇f|| ≈ 0 means we found a critical point
      grad_norm = grad |> Nx.abs() |> Nx.sum() |> Nx.to_number()

      if grad_norm < tolerance do
        # Converged! Gradient is essentially zero
        {:halt, current_params}
      else
        # Update rule: θ_new = θ_old - α * ∇f(θ_old)
        # We subtract because we want to go DOWN the gradient (minimize)
        new_params = Nx.subtract(current_params, Nx.multiply(lr, grad))

        {:cont, new_params}
      end
    end)
  end

  # Optimization with history tracking for visualization
  defp optimize_with_history(params, gradient_fn, lr, max_iters, tolerance) do
    initial_state = {params, []}

    {final_params, history} =
      Enum.reduce_while(1..max_iters, initial_state, fn iter, {current_params, history} ->
        grad = gradient_fn.(current_params)
        grad_norm = grad |> Nx.abs() |> Nx.sum() |> Nx.to_number()

        # Record this step in history
        new_history = [{current_params, grad_norm} | history]

        if grad_norm < tolerance do
          {:halt, {current_params, Enum.reverse(new_history)}}
        else
          new_params = Nx.subtract(current_params, Nx.multiply(lr, grad))
          {:cont, {new_params, new_history}}
        end
      end)

    {final_params, history}
  end

  @doc """
  Performs one step of gradient descent.

  This is useful when you want manual control over the optimization loop.

  ## Mathematical Formula

      θ_new = θ_old - α * ∇f(θ_old)

  ## Parameters

  - `params` - Current parameters (tensor)
  - `gradient` - Gradient at current parameters (tensor)
  - `learning_rate` - Step size (scalar or tensor)

  ## Example

      params = Nx.tensor([1.0, 2.0])
      gradient = Nx.tensor([0.5, -0.3])
      
      new_params = MLNx.GradientDescent.step(params, gradient, 0.1)
      # new_params = [1.0 - 0.1*0.5, 2.0 - 0.1*(-0.3)]
      #             = [0.95, 2.03]
  """
  defn step(params, gradient, learning_rate) do
    # θ_new = θ_old - α * ∇f(θ_old)
    # This is the core update rule of gradient descent
    Nx.subtract(params, Nx.multiply(learning_rate, gradient))
  end

  @doc """
  Computes the norm (magnitude) of the gradient.

  The gradient norm tells us how "steep" the function is at the current point.
  A small norm means we're near a critical point (minimum, maximum, or saddle).

  ## Mathematical Formula

      ||∇f|| = √(∂f/∂θ₁)² + (∂f/∂θ₂)² + ... + (∂f/∂θₙ)²

  For simplicity, we use L1 norm (sum of absolute values) instead of L2 (Euclidean):

      ||∇f||₁ = |∂f/∂θ₁| + |∂f/∂θ₂| + ... + |∂f/∂θₙ|

  ## Parameters

  - `gradient` - Gradient tensor

  ## Returns

  Scalar tensor representing the gradient magnitude
  """
  defn gradient_norm(gradient) do
    # L1 norm: sum of absolute values
    # This tells us the total "magnitude" of change across all parameters
    gradient
    |> Nx.abs()
    |> Nx.sum()
  end

  @doc """
  Demonstrates gradient descent on a simple quadratic function.

  This is a teaching function that shows GD optimizing f(x) = (x - target)²

  The minimum is at x = target, and the gradient is ∇f(x) = 2(x - target)

  ## Parameters

  - `initial_x` - Starting point
  - `target` - Where the minimum is located
  - `opts` - Options (learning_rate, max_iters, etc.)

  ## Example

      # Find minimum of f(x) = (x - 3)²
      # The minimum is at x = 3
      result = MLNx.GradientDescent.minimize_quadratic(
        Nx.tensor(10.0),  # Start at x = 10
        Nx.tensor(3.0),   # Target is x = 3
        learning_rate: 0.1,
        track_history: true
      )
      
      {final_x, history} = result
      # final_x ≈ 3.0
  """
  def minimize_quadratic(initial_x, target, opts \\ []) do
    # Gradient of f(x) = (x - target)² is:
    # ∇f(x) = 2(x - target)
    gradient_fn = fn x ->
      Nx.multiply(2.0, Nx.subtract(x, target))
    end

    optimize(initial_x, gradient_fn, opts)
  end
end
