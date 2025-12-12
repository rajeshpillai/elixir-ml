# Gradient Descent Visualization Demo
# Run with: mix run examples/gradient_descent_demo.exs

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("GRADIENT DESCENT: The Foundation of Machine Learning")
IO.puts(String.duplicate("=", 70) <> "\n")

IO.puts("""
CONCEPT: Gradient descent is an optimization algorithm that finds the minimum
of a function by iteratively moving in the direction of steepest descent.

Think of it like walking down a mountain in fog - you can't see the bottom,
but you can feel which way is downward, so you take small steps in that
direction until you reach the valley.
""")

# ============================================================================
# Example 1: Simple Quadratic Function
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 1: Minimizing f(x) = (x - 3)²")
IO.puts(String.duplicate("-", 70))

IO.puts("""
This is a simple bowl-shaped function with its minimum at x = 3.

Mathematical details:
  f(x) = (x - 3)²
  f'(x) = 2(x - 3)  ← This is the gradient
  
The gradient tells us:
  - If x > 3: gradient is positive → move left (decrease x)
  - If x < 3: gradient is negative → move right (increase x)
  - If x = 3: gradient is zero → we're at the minimum!
""")

initial_x = Nx.tensor(10.0)
target = Nx.tensor(3.0)

IO.puts("Starting at x = #{Nx.to_number(initial_x)}")
IO.puts("Target minimum at x = #{Nx.to_number(target)}\n")

# Run with history tracking to see the journey
{final_x, history} = MLNx.GradientDescent.minimize_quadratic(
  initial_x,
  target,
  learning_rate: 0.1,
  max_iters: 50,
  track_history: true
)

IO.puts("Optimization Journey:")
IO.puts(String.duplicate("-", 70))
IO.puts("Iter |      x      | Gradient Norm | Distance to Min")
IO.puts(String.duplicate("-", 70))

# Show first 10 iterations
history
|> Enum.take(10)
|> Enum.with_index(1)
|> Enum.each(fn {{params, grad_norm}, iter} ->
  x_val = Nx.to_number(params)
  distance = abs(x_val - 3.0)
  IO.puts(:io_lib.format("~4w | ~11.6f | ~13.6f | ~15.6f", [iter, x_val, grad_norm, distance]))
end)

if length(history) > 10 do
  IO.puts("  ... (#{length(history) - 10} more iterations)")
end

IO.puts(String.duplicate("-", 70))
IO.puts("Final x = #{Nx.to_number(final_x)} (target: 3.0)")
IO.puts("Converged in #{length(history)} iterations")
IO.puts("Final error: #{abs(Nx.to_number(final_x) - 3.0)}")

# ============================================================================
# Example 2: Effect of Learning Rate
# ============================================================================

IO.puts("\n\n" <> String.duplicate("-", 70))
IO.puts("Example 2: Impact of Learning Rate")
IO.puts(String.duplicate("-", 70))

IO.puts("""
The learning rate (α) controls how big our steps are.

  θ_new = θ_old - α * gradient

Let's see how different learning rates affect convergence:
""")

learning_rates = [0.01, 0.1, 0.5, 0.9]
initial_x = Nx.tensor(10.0)
target = Nx.tensor(0.0)

IO.puts("Starting from x = 10.0, targeting x = 0.0\n")
IO.puts("Learning Rate | Iterations | Final x    | Final Error")
IO.puts(String.duplicate("-", 70))

Enum.each(learning_rates, fn lr ->
  {final_x, history} = MLNx.GradientDescent.minimize_quadratic(
    initial_x,
    target,
    learning_rate: lr,
    max_iters: 100,
    tolerance: 1.0e-6,
    track_history: true
  )

  final_val = Nx.to_number(final_x)
  error = abs(final_val - 0.0)

  IO.puts(:io_lib.format("~13.2f | ~10w | ~10.6f | ~11.6f", [lr, length(history), final_val, error]))
end)

IO.puts("""

OBSERVATIONS:
  • Small LR (0.01): Slow but steady convergence
  • Medium LR (0.1): Good balance of speed and stability
  • Large LR (0.5, 0.9): Faster initially, but might oscillate
  • Too large LR (>1.0): Can diverge or oscillate wildly!
""")

# ============================================================================
# Example 3: Multi-dimensional Optimization
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 3: 2D Optimization")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Real ML problems optimize many parameters simultaneously.
Let's minimize f(x, y) = (x - 2)² + (y + 1)²

This is a bowl with minimum at (2, -1).

Gradient: ∇f = [2(x - 2), 2(y + 1)]
""")

initial_params = Nx.tensor([5.0, 3.0])
target = Nx.tensor([2.0, -1.0])

gradient_fn = fn params ->
  Nx.multiply(2.0, Nx.subtract(params, target))
end

{final_params, history} = MLNx.GradientDescent.optimize(
  initial_params,
  gradient_fn,
  learning_rate: 0.1,
  max_iters: 50,
  track_history: true
)

IO.puts("Starting at: [#{Nx.to_number(initial_params[0])}, #{Nx.to_number(initial_params[1])}]")
IO.puts("Target: [2.0, -1.0]\n")

IO.puts("Iter |      x      |      y      | Gradient Norm")
IO.puts(String.duplicate("-", 70))

history
|> Enum.take(10)
|> Enum.with_index(1)
|> Enum.each(fn {{params, grad_norm}, iter} ->
  x = Nx.to_number(params[0])
  y = Nx.to_number(params[1])
  IO.puts(:io_lib.format("~4w | ~11.6f | ~11.6f | ~13.6f", [iter, x, y, grad_norm]))
end)

IO.puts(String.duplicate("-", 70))
final_x = Nx.to_number(final_params[0])
final_y = Nx.to_number(final_params[1])
IO.puts("Final position: [#{final_x}, #{final_y}]")
IO.puts("Converged in #{length(history)} iterations")

# ============================================================================
# Example 4: Convergence Criteria
# ============================================================================

IO.puts("\n\n" <> String.duplicate("-", 70))
IO.puts("Example 4: Understanding Convergence")
IO.puts(String.duplicate("-", 70))

IO.puts("""
We stop gradient descent when:
  1. Gradient norm < tolerance (we're at a minimum)
  2. Maximum iterations reached

Let's see the difference:
""")

initial_x = Nx.tensor(10.0)
target = Nx.tensor(0.0)

# Tight tolerance - will converge
{final_tight, history_tight} = MLNx.GradientDescent.minimize_quadratic(
  initial_x,
  target,
  learning_rate: 0.1,
  max_iters: 1000,
  tolerance: 1.0e-8,
  track_history: true
)

# Loose tolerance - will stop early
{final_loose, history_loose} = MLNx.GradientDescent.minimize_quadratic(
  initial_x,
  target,
  learning_rate: 0.1,
  max_iters: 1000,
  tolerance: 1.0,
  track_history: true
)

IO.puts("Tight tolerance (1e-8):")
IO.puts("  Iterations: #{length(history_tight)}")
IO.puts("  Final x: #{Nx.to_number(final_tight)}")
IO.puts("  Final gradient norm: #{elem(List.last(history_tight), 1)}")

IO.puts("\nLoose tolerance (1.0):")
IO.puts("  Iterations: #{length(history_loose)}")
IO.puts("  Final x: #{Nx.to_number(final_loose)}")
IO.puts("  Final gradient norm: #{elem(List.last(history_loose), 1)}")

IO.puts("""

INSIGHT: Tighter tolerance means more iterations but better accuracy.
In practice, we balance computational cost with desired precision.
""")

# ============================================================================
# Summary
# ============================================================================

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("KEY TAKEAWAYS")
IO.puts(String.duplicate("=", 70))

IO.puts("""
1. GRADIENT DESCENT UPDATE RULE:
   θ_new = θ_old - α * ∇f(θ_old)
   
   We move OPPOSITE to the gradient (that's why we subtract)

2. LEARNING RATE (α):
   • Too small → slow convergence
   • Too large → instability, divergence
   • Just right → efficient optimization

3. CONVERGENCE:
   • Stop when gradient ≈ 0 (we're at a minimum)
   • Or when max iterations reached
   
4. GRADIENT NORM:
   • Measures how steep the function is
   • Small norm → near a critical point
   • Large norm → far from minimum

5. REAL ML APPLICATIONS:
   • Linear regression: minimize MSE loss
   • Neural networks: minimize cross-entropy loss
   • Any differentiable function can be optimized!

NEXT STEPS:
  - Learn about different loss functions
  - Explore stochastic gradient descent (SGD)
  - Study advanced optimizers (Adam, RMSprop)
""")

IO.puts(String.duplicate("=", 70))
IO.puts("Demo complete! 🎓")
IO.puts(String.duplicate("=", 70) <> "\n")
