# Loss Functions Comparison Demo
# Run with: mix run examples/loss_functions_demo.exs

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("LOSS FUNCTIONS: Measuring Prediction Error")
IO.puts(String.duplicate("=", 70) <> "\n")

IO.puts("""
CONCEPT: A loss function measures how wrong our predictions are.
Different loss functions have different properties and use cases.

The goal of machine learning is to MINIMIZE the loss function!
""")

# ============================================================================
# Example 1: Comparing MSE vs MAE
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 1: MSE vs MAE - Understanding the Difference")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Let's compare how MSE and MAE handle errors:

MSE (Mean Squared Error):
  - Formula: (1/n) Σ(ŷ - y)²
  - Squares errors → penalizes large errors heavily
  - Sensitive to outliers

MAE (Mean Absolute Error):
  - Formula: (1/n) Σ|ŷ - y|
  - Linear penalty → treats all errors equally
  - Robust to outliers
""")

# Regular predictions (no outliers)
regular_pred = Nx.tensor([2.0, 3.0, 4.0, 5.0])
regular_target = Nx.tensor([2.5, 3.5, 4.5, 5.5])

# With one outlier
outlier_pred = Nx.tensor([2.0, 3.0, 4.0, 15.0])  # Last one way off!
outlier_target = Nx.tensor([2.5, 3.5, 4.5, 5.5])

IO.puts("\nRegular predictions (all errors ≈ 0.5):")
IO.puts("  Predictions: #{inspect(Nx.to_list(regular_pred))}")
IO.puts("  Targets:     #{inspect(Nx.to_list(regular_target))}")

mse_regular = MLNx.LossFunctions.mse(regular_pred, regular_target) |> Nx.to_number()
mae_regular = MLNx.LossFunctions.mae(regular_pred, regular_target) |> Nx.to_number()

IO.puts("  MSE: #{Float.round(mse_regular, 3)}")
IO.puts("  MAE: #{Float.round(mae_regular, 3)}")

IO.puts("\nWith outlier (last prediction is 15 instead of 5.5):")
IO.puts("  Predictions: #{inspect(Nx.to_list(outlier_pred))}")
IO.puts("  Targets:     #{inspect(Nx.to_list(outlier_target))}")

mse_outlier = MLNx.LossFunctions.mse(outlier_pred, outlier_target) |> Nx.to_number()
mae_outlier = MLNx.LossFunctions.mae(outlier_pred, outlier_target) |> Nx.to_number()

IO.puts("  MSE: #{Float.round(mse_outlier, 3)}")
IO.puts("  MAE: #{Float.round(mae_outlier, 3)}")

mse_increase = mse_outlier / mse_regular
mae_increase = mae_outlier / mse_regular

IO.puts("""

OBSERVATION:
  MSE increased by #{Float.round(mse_increase, 1)}x (very sensitive!)
  MAE increased by #{Float.round(mae_increase, 1)}x (more robust)

  The outlier error (9.5) contributes:
    - To MSE: 9.5² = 90.25 (huge!)
    - To MAE: |9.5| = 9.5 (linear)

LESSON: Use MAE when you have outliers in your data!
""")

# ============================================================================
# Example 2: Huber Loss - Best of Both Worlds
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 2: Huber Loss - Combining MSE and MAE")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Huber Loss uses:
  - Quadratic (MSE) for small errors → smooth gradients
  - Linear (MAE) for large errors → robust to outliers

Formula:
  Huber(e) = { 0.5*e²           if |e| ≤ δ
             { δ(|e| - δ/2)     if |e| > δ

Let's see it in action with δ = 1.0:
""")

# Create data with mix of small and large errors
predictions = Nx.tensor([1.0, 2.0, 3.0, 10.0])
targets = Nx.tensor([1.2, 2.3, 3.1, 4.0])

errors = Nx.subtract(predictions, targets) |> Nx.to_list()

IO.puts("\nPredictions: #{inspect(Nx.to_list(predictions))}")
IO.puts("Targets:     #{inspect(Nx.to_list(targets))}")
IO.puts("Errors:      #{inspect(Enum.map(errors, &Float.round(&1, 1)))}")

mse = MLNx.LossFunctions.mse(predictions, targets) |> Nx.to_number()
mae = MLNx.LossFunctions.mae(predictions, targets) |> Nx.to_number()
huber = MLNx.LossFunctions.huber(predictions, targets, delta: 1.0) |> Nx.to_number()

IO.puts("\nLoss values:")
IO.puts("  MSE:   #{Float.round(mse, 3)} (dominated by large error)")
IO.puts("  MAE:   #{Float.round(mae, 3)} (treats all errors equally)")
IO.puts("  Huber: #{Float.round(huber, 3)} (balanced!)")

IO.puts("""

OBSERVATION:
  Huber loss is between MAE and MSE
  - Uses MSE for small errors (smooth optimization)
  - Uses MAE for large errors (robust to outliers)

LESSON: Huber is great when you want smooth gradients but also robustness!
""")

# ============================================================================
# Example 3: Binary Cross-Entropy for Classification
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 3: Binary Cross-Entropy - Classification Loss")
IO.puts(String.duplicate("-", 70))

IO.puts("""
For binary classification (spam/not spam, cat/dog, etc.):

Formula: BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

Where:
  - y ∈ {0, 1} = true label
  - ŷ ∈ (0, 1) = predicted probability

Let's classify emails as spam (1) or not spam (0):
""")

# Email classification examples
emails = [
  {"Obvious spam", 0.95, 1.0},
  {"Likely spam", 0.75, 1.0},
  {"Uncertain", 0.55, 1.0},
  {"Likely not spam", 0.25, 0.0},
  {"Obvious not spam", 0.05, 0.0},
  {"Wrong prediction!", 0.90, 0.0}  # Confident but wrong!
]

IO.puts("\nEmail Classification Results:")
IO.puts(String.duplicate("-", 70))
IO.puts("Description              | Predicted | True | Loss")
IO.puts(String.duplicate("-", 70))

Enum.each(emails, fn {desc, pred, true_label} ->
  pred_tensor = Nx.tensor([pred])
  true_tensor = Nx.tensor([true_label])
  loss = MLNx.LossFunctions.binary_cross_entropy(pred_tensor, true_tensor) |> Nx.to_number()
  
  IO.puts(:io_lib.format("~-24s | ~9.2f | ~4.1f | ~6.3f", [desc, pred, true_label, loss]))
end)

IO.puts("""

OBSERVATION:
  - Confident correct predictions: LOW loss
  - Uncertain predictions: MEDIUM loss
  - Confident wrong predictions: HIGH loss (2.3+)

LESSON: Cross-entropy heavily penalizes confident mistakes!
""")

# ============================================================================
# Example 4: Categorical Cross-Entropy for Multi-class
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 4: Categorical Cross-Entropy - Multi-class Classification")
IO.puts(String.duplicate("-", 70))

IO.puts("""
For multi-class problems (classify images as cat/dog/bird):

Formula: CCE = -Σ(y * log(ŷ))

Where:
  - y = one-hot encoded true label (e.g., [0, 1, 0] for class 2)
  - ŷ = predicted probability distribution (from softmax)

Let's classify animals:
""")

# Animal classification examples
examples = [
  {"Confident cat", [0.8, 0.1, 0.1], [1.0, 0.0, 0.0]},
  {"Confident dog", [0.1, 0.8, 0.1], [0.0, 1.0, 0.0]},
  {"Confident bird", [0.1, 0.1, 0.8], [0.0, 0.0, 1.0]},
  {"Uncertain", [0.4, 0.3, 0.3], [1.0, 0.0, 0.0]},
  {"Wrong!", [0.7, 0.2, 0.1], [0.0, 0.0, 1.0]}
]

IO.puts("\nAnimal Classification (Cat/Dog/Bird):")
IO.puts(String.duplicate("-", 70))
IO.puts("Description      | Predicted      | True Label | Loss")
IO.puts(String.duplicate("-", 70))

Enum.each(examples, fn {desc, pred, true_label} ->
  pred_tensor = Nx.tensor([pred])
  true_tensor = Nx.tensor([true_label])
  loss = MLNx.LossFunctions.categorical_cross_entropy(pred_tensor, true_tensor) |> Nx.to_number()
  
  pred_str = Enum.map(pred, &Float.round(&1, 1)) |> inspect()
  true_str = Enum.find_index(true_label, &(&1 == 1.0))
  
  IO.puts(:io_lib.format("~-16s | ~14s | Class ~w    | ~5.3f", [desc, pred_str, true_str, loss]))
end)

IO.puts("""

OBSERVATION:
  - Confident correct: loss ≈ 0.22
  - Uncertain: loss ≈ 0.92
  - Wrong prediction: loss ≈ 1.90

LESSON: Model should output high probability for the correct class!
""")

# ============================================================================
# Example 5: Choosing the Right Loss Function
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 5: Decision Guide - Which Loss Function?")
IO.puts(String.duplicate("-", 70))

IO.puts("""
REGRESSION (predicting continuous values):

  1. MSE (Mean Squared Error)
     ✓ Use when: Large errors are particularly bad
     ✓ Use when: No outliers in data
     ✗ Avoid when: Data has outliers
     Example: Predicting house prices in a uniform market

  2. MAE (Mean Absolute Error)
     ✓ Use when: All errors should be weighted equally
     ✓ Use when: Data has outliers
     ✓ Use when: Want interpretable error metric
     Example: Predicting delivery times (outliers common)

  3. Huber Loss
     ✓ Use when: Want smooth gradients AND robustness
     ✓ Use when: Some outliers but want MSE-like behavior for most data
     Example: Robust regression in noisy environments

  4. RMSE (Root Mean Squared Error)
     ✓ Use when: Want MSE but in original units (for reporting)
     Example: Reporting prediction error to stakeholders

CLASSIFICATION (predicting categories):

  5. Binary Cross-Entropy
     ✓ Use when: Two classes (binary classification)
     ✓ Use when: Output is probability (after sigmoid)
     Example: Spam detection, fraud detection

  6. Categorical Cross-Entropy
     ✓ Use when: Multiple classes (3+)
     ✓ Use when: Classes are mutually exclusive
     ✓ Use when: Output is probability distribution (after softmax)
     Example: Image classification, text categorization
""")

# ============================================================================
# Example 6: Practical Comparison
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 6: Side-by-Side Comparison")
IO.puts(String.duplicate("-", 70))

# Create test data
test_pred = Nx.tensor([2.0, 3.5, 5.0, 7.0, 20.0])  # Last one is outlier
test_target = Nx.tensor([2.1, 3.6, 4.9, 7.2, 8.0])

metrics = MLNx.LossFunctions.regression_metrics(test_pred, test_target)

IO.puts("\nTest Data:")
IO.puts("  Predictions: #{inspect(Nx.to_list(test_pred))}")
IO.puts("  Targets:     #{inspect(Nx.to_list(test_target))}")
IO.puts("  Errors:      #{inspect(Nx.subtract(test_pred, test_target) |> Nx.to_list() |> Enum.map(&Float.round(&1, 1)))}")

IO.puts("\nAll Regression Metrics:")
IO.puts("  MSE:   #{Float.round(metrics.mse, 3)} (very high due to outlier)")
IO.puts("  RMSE:  #{Float.round(metrics.rmse, 3)}")
IO.puts("  MAE:   #{Float.round(metrics.mae, 3)} (more reasonable)")
IO.puts("  Huber: #{Float.round(metrics.huber, 3)} (balanced)")

# ============================================================================
# Summary
# ============================================================================

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("KEY TAKEAWAYS")
IO.puts(String.duplicate("=", 70))

IO.puts("""
1. LOSS FUNCTIONS MEASURE ERROR
   Different functions have different properties and use cases

2. MSE vs MAE:
   MSE: Penalizes large errors heavily (quadratic)
   MAE: Treats all errors equally (linear)

3. HUBER LOSS:
   Best of both worlds - smooth gradients + robustness

4. CROSS-ENTROPY:
   For classification problems
   Binary: 2 classes
   Categorical: 3+ classes

5. CHOOSING THE RIGHT LOSS:
   - Consider your data (outliers?)
   - Consider your problem (regression vs classification)
   - Consider what errors matter most

6. IN PRACTICE:
   Loss function + Gradient Descent = Machine Learning!
   
   Training loop:
     1. Predict: ŷ = model(X)
     2. Compute loss: L = loss_fn(ŷ, y)
     3. Compute gradients: ∇L
     4. Update: θ = θ - α∇L
     5. Repeat!

NEXT STEPS:
  - Learn about regularization (preventing overfitting)
  - Explore different optimizers (Adam, RMSprop)
  - Build classification models with cross-entropy
""")

IO.puts(String.duplicate("=", 70))
IO.puts("Demo complete! 🎓")
IO.puts(String.duplicate("=", 70) <> "\n")
