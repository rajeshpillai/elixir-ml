# Batch Training Demo
# Run with: mix run examples/05_batch_training_demo.exs

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("BATCH TRAINING: Efficient Gradient Descent Variants")
IO.puts(String.duplicate("=", 70) <> "\n")

IO.puts("""
CONCEPT: Different ways to compute gradients and update weights.

Three variants:
  1. Batch GD: Use ALL examples → Accurate but slow
  2. Stochastic GD: Use ONE example → Fast but noisy
  3. Mini-Batch GD: Use SMALL batches → Best balance

Why it matters: Efficiency on large datasets!
""")

# Generate sample data: y = 2x + 1
IO.puts("Generating sample data: y = 2x + 1")
x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
y = Nx.tensor([[3.0], [5.0], [7.0], [9.0], [11.0], [13.0], [15.0], [17.0]])

IO.puts("Dataset: #{Nx.axis_size(x, 0)} examples\n")

# ============================================================================
# Example 1: Batch Gradient Descent
# ============================================================================

IO.puts(String.duplicate("-", 70))
IO.puts("Example 1: Batch Gradient Descent - Using All Data")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Batch GD computes gradients using ALL training examples:

  For each iteration:
    1. Compute predictions for ALL examples
    2. Compute error for ALL examples
    3. Average gradients across ALL examples
    4. Update weights ONCE

Characteristics:
  - Most accurate gradient estimate
  - Smooth convergence
  - Slow for large datasets (processes all data each iteration)
""")

{w_batch, b_batch, history_batch} = MLNx.BatchTraining.batch_gradient_descent(
  x, y,
  learning_rate: 0.01,
  iterations: 100
)

IO.puts("\nResults after 100 iterations:")
IO.puts("  Learned weight: #{Float.round(Nx.to_number(w_batch[0][0]), 3)}")
IO.puts("  Learned bias:   #{Float.round(Nx.to_number(b_batch[0][0]), 3)}")
IO.puts("  Target: w=2.0, b=1.0")
IO.puts("\nLoss progression:")
IO.puts("  Initial loss: #{Float.round(List.first(history_batch), 3)}")
IO.puts("  Final loss:   #{Float.round(List.last(history_batch), 3)}")
IO.puts("  Reduction:    #{Float.round((1 - List.last(history_batch)/List.first(history_batch)) * 100, 1)}%")

# ============================================================================
# Example 2: Stochastic Gradient Descent (SGD)
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 2: Stochastic Gradient Descent - One Example at a Time")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Stochastic GD uses ONE random example per update:

  For each epoch:
    For each example:
      1. Compute prediction for THIS example
      2. Compute error for THIS example
      3. Compute gradient from THIS example
      4. Update weights IMMEDIATELY

Characteristics:
  - Fast updates (no need to process all data)
  - Noisy convergence (high variance in gradients)
  - Good for very large datasets
  - Can escape local minima due to noise
""")

{w_sgd, b_sgd, history_sgd} = MLNx.BatchTraining.stochastic_gradient_descent(
  x, y,
  learning_rate: 0.01,
  epochs: 20
)

IO.puts("\nResults after 20 epochs:")
IO.puts("  Learned weight: #{Float.round(Nx.to_number(w_sgd[0][0]), 3)}")
IO.puts("  Learned bias:   #{Float.round(Nx.to_number(b_sgd[0][0]), 3)}")
IO.puts("  Target: w=2.0, b=1.0")
IO.puts("\nLoss progression (per epoch):")
IO.puts("  Initial loss: #{Float.round(List.first(history_sgd), 3)}")
IO.puts("  Final loss:   #{Float.round(List.last(history_sgd), 3)}")
IO.puts("  Note: SGD loss may fluctuate due to noise")

# ============================================================================
# Example 3: Mini-Batch Gradient Descent
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 3: Mini-Batch Gradient Descent - Best of Both Worlds")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Mini-Batch GD uses SMALL batches of examples:

  For each epoch:
    Divide data into batches
    For each batch:
      1. Compute predictions for batch
      2. Compute errors for batch
      3. Average gradients across batch
      4. Update weights

Characteristics:
  - Balance between accuracy and speed
  - More stable than SGD, faster than Batch
  - Efficient use of hardware (vectorization)
  - MOST COMMONLY USED in practice!
""")

{w_mini, b_mini, history_mini} = MLNx.BatchTraining.mini_batch_gradient_descent(
  x, y,
  batch_size: 2,
  learning_rate: 0.01,
  epochs: 20
)

IO.puts("\nResults after 20 epochs (batch_size=2):")
IO.puts("  Learned weight: #{Float.round(Nx.to_number(w_mini[0][0]), 3)}")
IO.puts("  Learned bias:   #{Float.round(Nx.to_number(b_mini[0][0]), 3)}")
IO.puts("  Target: w=2.0, b=1.0")
IO.puts("\nLoss progression:")
IO.puts("  Initial loss: #{Float.round(List.first(history_mini), 3)}")
IO.puts("  Final loss:   #{Float.round(List.last(history_mini), 3)}")

# ============================================================================
# Example 4: Convergence Comparison
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 4: Comparing Convergence - Which is Best?")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Let's compare all three methods on the same problem:
""")

IO.puts("\nFinal Results Comparison:")
IO.puts(String.duplicate("-", 70))
IO.puts("Method          | Weight | Bias  | Final Loss | Iterations/Epochs")
IO.puts(String.duplicate("-", 70))

IO.puts("Batch GD        | #{String.pad_leading("#{Float.round(Nx.to_number(w_batch[0][0]), 2)}", 6)} | #{String.pad_leading("#{Float.round(Nx.to_number(b_batch[0][0]), 2)}", 5)} | #{String.pad_leading("#{Float.round(List.last(history_batch), 4)}", 10)} | 100")
IO.puts("Stochastic GD   | #{String.pad_leading("#{Float.round(Nx.to_number(w_sgd[0][0]), 2)}", 6)} | #{String.pad_leading("#{Float.round(Nx.to_number(b_sgd[0][0]), 2)}", 5)} | #{String.pad_leading("#{Float.round(List.last(history_sgd), 4)}", 10)} | 20")
IO.puts("Mini-Batch GD   | #{String.pad_leading("#{Float.round(Nx.to_number(w_mini[0][0]), 2)}", 6)} | #{String.pad_leading("#{Float.round(Nx.to_number(b_mini[0][0]), 2)}", 5)} | #{String.pad_leading("#{Float.round(List.last(history_mini), 4)}", 10)} | 20")
IO.puts("Target          |   2.00 |  1.00 |          - | -")

IO.puts("""

OBSERVATIONS:
  - All three methods converge to similar solutions
  - Batch GD: Smoothest convergence, most accurate
  - Stochastic GD: Fastest updates, noisier convergence
  - Mini-Batch GD: Good balance, most practical
""")

# ============================================================================
# Example 5: Batch Size Impact
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 5: Impact of Batch Size")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Batch size is a key hyperparameter. Let's see how it affects training:
""")

batch_sizes = [1, 2, 4, 8]

IO.puts("\nBatch Size | Final Weight | Final Bias | Final Loss")
IO.puts(String.duplicate("-", 70))

Enum.each(batch_sizes, fn bs ->
  {w, b, hist} = MLNx.BatchTraining.mini_batch_gradient_descent(
    x, y,
    batch_size: bs,
    learning_rate: 0.01,
    epochs: 20
  )
  
  IO.puts("#{String.pad_leading("#{bs}", 10)} | #{String.pad_leading("#{Float.round(Nx.to_number(w[0][0]), 3)}", 12)} | #{String.pad_leading("#{Float.round(Nx.to_number(b[0][0]), 3)}", 10)} | #{Float.round(List.last(hist), 4)}")
end)

IO.puts("""

OBSERVATIONS:
  - Batch size = 1: Same as Stochastic GD (noisiest)
  - Batch size = 8: Same as Batch GD (smoothest, dataset has 8 examples)
  - Batch sizes 2-4: Good balance

RULE OF THUMB:
  - Small datasets: Use batch GD (or large batch size)
  - Large datasets: Use mini-batch GD with batch_size 32-256
  - Very large datasets: Smaller batches (16-64) for faster iterations
""")

# ============================================================================
# Example 6: When to Use Each Method
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 6: Decision Guide - Which Method to Choose?")
IO.puts(String.duplicate("-", 70))

IO.puts("""
BATCH GRADIENT DESCENT
  ✓ Use when:
    - Small dataset (< 10,000 examples)
    - Need smooth, deterministic convergence
    - Have enough memory
    - Want most accurate gradient estimates
  
  ✗ Avoid when:
    - Large dataset (slow!)
    - Limited memory
    - Need fast iterations

STOCHASTIC GRADIENT DESCENT (SGD)
  ✓ Use when:
    - Very large dataset (millions of examples)
    - Online learning (streaming data)
    - Want to escape local minima
    - Limited memory
  
  ✗ Avoid when:
    - Need stable convergence
    - Small dataset (too noisy)
    - Sensitive to hyperparameters

MINI-BATCH GRADIENT DESCENT
  ✓ Use when:
    - Medium to large datasets
    - Want balance of speed and stability
    - Using GPUs (vectorization benefits)
    - Deep learning (MOST COMMON CHOICE!)
  
  ✗ Avoid when:
    - Dataset is tiny (< 100 examples, use batch)
    - Extreme memory constraints (use SGD)

PRACTICAL RECOMMENDATIONS:
  - Start with mini-batch GD, batch_size = 32
  - If too slow: decrease batch size
  - If too noisy: increase batch size
  - Common batch sizes: 16, 32, 64, 128, 256
""")

# ============================================================================
# Summary
# ============================================================================

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("KEY TAKEAWAYS")
IO.puts(String.duplicate("=", 70))

IO.puts("""
1. THREE GRADIENT DESCENT VARIANTS
   - Batch: All data → Accurate but slow
   - Stochastic: One example → Fast but noisy
   - Mini-Batch: Small batches → Best balance

2. BATCH GRADIENT DESCENT
   - Uses entire dataset per iteration
   - Smooth, deterministic convergence
   - Slow for large datasets
   - Good for small datasets

3. STOCHASTIC GRADIENT DESCENT (SGD)
   - Uses one random example per update
   - Fast updates, many iterations
   - Noisy convergence path
   - Good for very large datasets

4. MINI-BATCH GRADIENT DESCENT
   - Uses small batches (typically 32-256)
   - Balance between accuracy and speed
   - Most commonly used in practice
   - Efficient on GPUs

5. CHOOSING BATCH SIZE
   - Small batch (1-16): Noisy but fast
   - Medium batch (32-128): Good balance
   - Large batch (256+): Smooth but slow
   - Rule of thumb: Start with 32

6. PRACTICAL TIPS
   - Mini-batch GD is the default choice
   - Batch size is a hyperparameter to tune
   - Larger batches need larger learning rates
   - Use powers of 2 for GPU efficiency

NEXT STEPS:
  - Learn about learning rate scheduling
  - Explore advanced optimizers (Adam, RMSprop)
  - Understand momentum and acceleration
""")

IO.puts(String.duplicate("=", 70))
IO.puts("Demo complete! 🎓")
IO.puts(String.duplicate("=", 70) <> "\n")
