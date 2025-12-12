# Regularization Demo
# Run with: mix run examples/03_regularization_demo.exs

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("REGULARIZATION: Preventing Overfitting")
IO.puts(String.duplicate("=", 70) <> "\n")

IO.puts("""
CONCEPT: Regularization adds a penalty for large weights to prevent
overfitting. It encourages simpler models that generalize better.

Formula: Total Loss = Data Loss + λ * Regularization Penalty

Where λ (lambda) controls the strength of regularization.
""")

# ============================================================================
# Example 1: Understanding L2 Regularization (Ridge)
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 1: L2 Regularization (Ridge) - Shrinking Weights")
IO.puts(String.duplicate("-", 70))

IO.puts("""
L2 Regularization penalizes the sum of squared weights:

  L2 Penalty = λ * Σw²

Effect: Shrinks ALL weights toward zero (but not to exactly zero)
""")

weights = Nx.tensor([5.0, -3.0, 2.0, -1.5])

IO.puts("\nOriginal weights: #{inspect(Nx.to_list(weights))}")
IO.puts("\nL2 Penalty for different λ values:")
IO.puts(String.duplicate("-", 70))

lambdas = [0.0, 0.01, 0.1, 1.0]

Enum.each(lambdas, fn lambda ->
  penalty = MLNx.Regularization.l2_penalty(weights, lambda) |> Nx.to_number()
  IO.puts("  λ = #{:io_lib.format("~4.2f", [lambda])} → Penalty = #{Float.round(penalty, 3)}")
end)

IO.puts("""

OBSERVATION:
  - λ = 0: No penalty (no regularization)
  - As λ increases, penalty increases
  - Large weights (5, -3) contribute more due to squaring

EFFECT ON TRAINING:
  - Model will prefer smaller weights
  - All features kept, but weights shrunk
  - Prevents overfitting by limiting model complexity
""")

# ============================================================================
# Example 2: Understanding L1 Regularization (Lasso)
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 2: L1 Regularization (Lasso) - Feature Selection")
IO.puts(String.duplicate("-", 70))

IO.puts("""
L1 Regularization penalizes the sum of absolute weights:

  L1 Penalty = λ * Σ|w|

Effect: Forces some weights to EXACTLY ZERO (feature selection!)
""")

IO.puts("\nOriginal weights: #{inspect(Nx.to_list(weights))}")
IO.puts("\nL1 Penalty for different λ values:")
IO.puts(String.duplicate("-", 70))

Enum.each(lambdas, fn lambda ->
  penalty = MLNx.Regularization.l1_penalty(weights, lambda) |> Nx.to_number()
  IO.puts("  λ = #{:io_lib.format("~4.2f", [lambda])} → Penalty = #{Float.round(penalty, 3)}")
end)

IO.puts("""

OBSERVATION:
  - L1 penalty is LINEAR (not squared)
  - All weights contribute equally to penalty
  - Penalty = λ * (|5| + |-3| + |2| + |-1.5|) = λ * 11.5

EFFECT ON TRAINING:
  - Small weights get pushed to exactly zero
  - Automatic feature selection
  - Sparse solutions (many weights = 0)
""")

# ============================================================================
# Example 3: L1 vs L2 Comparison
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 3: L1 vs L2 - Side by Side Comparison")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Let's compare how L1 and L2 penalize different weight values:
""")

test_weights = [
  Nx.tensor([1.0]),
  Nx.tensor([2.0]),
  Nx.tensor([5.0]),
  Nx.tensor([10.0])
]

lambda = 0.1

IO.puts("\nWeight | L1 Penalty | L2 Penalty | L2/L1 Ratio")
IO.puts(String.duplicate("-", 70))

Enum.each(test_weights, fn w ->
  weight_val = Nx.to_number(w[0])
  l1 = MLNx.Regularization.l1_penalty(w, lambda) |> Nx.to_number()
  l2 = MLNx.Regularization.l2_penalty(w, lambda) |> Nx.to_number()
  ratio = l2 / l1
  
  IO.puts(:io_lib.format("~6.1f | ~10.3f | ~10.3f | ~11.2f", [weight_val, l1, l2, ratio]))
end)

IO.puts("""

OBSERVATION:
  - For small weights: L1 and L2 are similar
  - For large weights: L2 penalty grows MUCH faster (quadratic)
  - L2/L1 ratio increases with weight magnitude

LESSON:
  - L2 heavily penalizes large weights (use when all features matter)
  - L1 treats all weights equally (use for feature selection)
""")

# ============================================================================
# Example 4: Elastic Net - Best of Both Worlds
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 4: Elastic Net - Combining L1 and L2")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Elastic Net combines L1 and L2:

  Elastic Net = λ * [α * Σ|w| + (1-α) * Σw²]

Where α controls the mix:
  - α = 0: Pure L2 (Ridge)
  - α = 1: Pure L1 (Lasso)
  - α = 0.5: Equal mix
""")

weights = Nx.tensor([3.0, -2.0, 1.0])
lambda = 0.1

IO.puts("\nWeights: #{inspect(Nx.to_list(weights))}")
IO.puts("λ = #{lambda}")
IO.puts("\nα Value | Penalty Type | Penalty Value")
IO.puts(String.duplicate("-", 70))

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

Enum.each(alphas, fn alpha ->
  penalty = MLNx.Regularization.elastic_net_penalty(weights, lambda, alpha) |> Nx.to_number()
  
  type = cond do
    alpha == 0.0 -> "Pure L2 (Ridge)"
    alpha == 1.0 -> "Pure L1 (Lasso)"
    true -> "Mixed"
  end
  
  IO.puts(:io_lib.format("~7.2f | ~16s | ~13.3f", [alpha, type, penalty]))
end)

IO.puts("""

OBSERVATION:
  - α = 0.0: Same as L2 regularization
  - α = 1.0: Same as L1 regularization
  - 0 < α < 1: Smooth transition between L1 and L2

WHEN TO USE ELASTIC NET:
  - Want some feature selection (L1) but not too aggressive
  - Features are correlated (L1 alone is unstable)
  - Want the benefits of both L1 and L2
""")

# ============================================================================
# Example 5: Effect on Total Loss
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 5: Regularization Effect on Total Loss")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Regularization adds a penalty to the data loss:

  Total Loss = MSE + Regularization Penalty

Let's see how this affects the total loss:
""")

# Simulate predictions and targets
predictions = Nx.tensor([2.0, 3.0, 4.0, 5.0])
targets = Nx.tensor([2.5, 3.5, 4.5, 5.5])
weights = Nx.tensor([2.0, -3.0])

# Compute MSE (data loss)
mse = MLNx.LossFunctions.mse(predictions, targets) |> Nx.to_number()

IO.puts("\nData Loss (MSE): #{Float.round(mse, 3)}")
IO.puts("Weights: #{inspect(Nx.to_list(weights))}")
IO.puts("\nλ Value | Reg Type | Reg Penalty | Total Loss")
IO.puts(String.duplicate("-", 70))

lambdas = [0.0, 0.01, 0.1, 0.5]

Enum.each(lambdas, fn lambda ->
  l2_loss = MLNx.Regularization.regularized_loss(
    predictions, targets, weights,
    reg_type: :l2, lambda: lambda
  ) |> Nx.to_number()
  
  l2_penalty = MLNx.Regularization.l2_penalty(weights, lambda) |> Nx.to_number()
  
  IO.puts(:io_lib.format("~7.2f | ~8s | ~11.3f | ~10.3f", 
    [lambda, "L2", l2_penalty, l2_loss]))
end)

IO.puts("""

OBSERVATION:
  - λ = 0: Total loss = MSE (no regularization)
  - As λ increases, total loss increases
  - Model must balance: fit data well AND keep weights small

TRAINING EFFECT:
  Without regularization: Model minimizes only MSE
  With regularization: Model trades off data fit vs weight magnitude
  Result: Simpler model that generalizes better!
""")

# ============================================================================
# Example 6: Choosing Lambda
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 6: Choosing the Right λ")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Lambda (λ) controls the regularization strength:

  - λ too small: Underfitting (model too simple)
  - λ too large: Overfitting (model too complex)
  - λ just right: Good generalization

How to choose λ:
  1. Try multiple values: [0.001, 0.01, 0.1, 1.0, 10.0]
  2. Use validation set to evaluate each
  3. Pick λ with best validation performance
  4. This is called hyperparameter tuning!
""")

weights = Nx.tensor([4.0, -3.0, 2.0, -1.0])

penalties = MLNx.Regularization.compare_lambdas(
  weights,
  reg_type: :l2,
  lambdas: [0.001, 0.01, 0.1, 1.0, 10.0]
)

IO.puts("\nL2 Penalty for different λ values:")
IO.puts("Weights: #{inspect(Nx.to_list(weights))}")
IO.puts(String.duplicate("-", 70))

penalties
|> Enum.sort_by(fn {lambda, _} -> lambda end)
|> Enum.each(fn {lambda, penalty} ->
  IO.puts("  λ = #{:io_lib.format("~6.3f", [lambda])} → Penalty = #{Float.round(penalty, 3)}")
end)

# ============================================================================
# Summary
# ============================================================================

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("KEY TAKEAWAYS")
IO.puts(String.duplicate("=", 70))

IO.puts("""
1. REGULARIZATION PREVENTS OVERFITTING
   Adds penalty for large weights to encourage simpler models

2. L2 REGULARIZATION (RIDGE)
   Formula: λ * Σw²
   - Shrinks all weights toward zero
   - Keeps all features
   - Penalizes large weights heavily (squared)

3. L1 REGULARIZATION (LASSO)
   Formula: λ * Σ|w|
   - Forces some weights to exactly zero
   - Automatic feature selection
   - Linear penalty (treats all weights equally)

4. ELASTIC NET
   Formula: λ * [α*Σ|w| + (1-α)*Σw²]
   - Combines L1 and L2
   - α controls the mix
   - Gets benefits of both

5. CHOOSING LAMBDA (λ)
   - Controls regularization strength
   - Tune using validation set
   - Balance: data fit vs model simplicity

6. WHEN TO USE EACH:
   - L2: When all features are relevant
   - L1: When you want feature selection
   - Elastic Net: When you want both benefits

NEXT STEPS:
  - Learn about feature normalization
  - Explore cross-validation for tuning λ
  - Build regularized models for real problems
""")

IO.puts(String.duplicate("=", 70))
IO.puts("Demo complete! 🎓")
IO.puts(String.duplicate("=", 70) <> "\n")
