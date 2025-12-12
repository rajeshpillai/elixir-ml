# ML Learning Journey - Commit 2: Loss Functions

## 🎓 What You Learned

**Loss Functions** measure how wrong our predictions are. Different loss functions have different properties - some penalize large errors heavily, others are robust to outliers, and some are designed specifically for classification.

### Core Concept

```
The goal of ML: Minimize Loss
  1. Make predictions: ŷ = model(X)
  2. Compute loss: L = loss_fn(ŷ, y)
  3. Use gradient descent to minimize L
```

---

## 📁 Files Created

### 1. [loss_functions.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/loss_functions.ex)

**5 Loss Functions** with extensive documentation:

**Regression Losses:**
- `mse/2` - Mean Squared Error (L2 loss)
- `rmse/2` - Root Mean Squared Error
- `mae/2` - Mean Absolute Error (L1 loss)
- `huber/3` - Robust loss combining MSE and MAE

**Classification Losses:**
- `binary_cross_entropy/3` - For binary classification
- `categorical_cross_entropy/3` - For multi-class classification

**Utility:**
- `regression_metrics/3` - Compute all regression metrics at once

### 2. [loss_functions_test.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/test/loss_functions_test.exs)

**27 Comprehensive Tests** covering:
- Basic loss computations
- Edge cases (perfect predictions, outliers)
- Comparative behavior (MSE vs MAE vs Huber)
- Classification losses
- Robustness properties

### 3. [loss_functions_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/loss_functions_demo.exs)

**6 Interactive Examples:**
1. MSE vs MAE with outliers
2. Huber loss demonstration
3. Binary cross-entropy for classification
4. Categorical cross-entropy for multi-class
5. Decision guide for choosing loss functions
6. Side-by-side comparison

---

## ✅ Test Results

```
Running ExUnit with seed: 0, max_cases: 8

...........................
Finished in 2.7 seconds (0.00s async, 2.7s sync)
27 tests, 0 failures
```

All tests pass! Key validations:
- ✓ MSE penalizes large errors heavily
- ✓ MAE is robust to outliers
- ✓ Huber balances both approaches
- ✓ Cross-entropy for classification works correctly
- ✓ All edge cases handled properly

---

## 🎬 Demo Highlights

### Example 1: MSE vs MAE with Outliers

**Regular data** (all errors ≈ 0.5):
- MSE: 0.25
- MAE: 0.5

**With outlier** (error = 9.5):
- MSE: 22.75 (increased 91x!)
- MAE: 2.75 (increased 11x)

**Insight**: MSE is very sensitive to outliers because it squares errors. MAE treats all errors equally.

### Example 2: Huber Loss

For data with mix of small and large errors:
- MSE: 9.035 (dominated by outlier)
- MAE: 1.65 (treats all equally)
- Huber: 1.393 (balanced!)

**Insight**: Huber uses quadratic for small errors (smooth gradients) and linear for large errors (robust to outliers).

### Example 3: Binary Cross-Entropy

Email spam classification:

| Prediction | True Label | Loss |
|------------|------------|------|
| 0.95 (spam) | 1 (spam) | 0.051 (good!) |
| 0.55 (uncertain) | 1 (spam) | 0.598 (medium) |
| 0.90 (spam) | 0 (not spam) | 2.303 (terrible!) |

**Insight**: Cross-entropy heavily penalizes confident wrong predictions!

---

## 🔑 Key Takeaways

### 1. Loss Function Formulas

**MSE (L2)**:
```
MSE = (1/n) Σ(ŷ - y)²
```
- Squares errors → large errors contribute much more
- Sensitive to outliers

**MAE (L1)**:
```
MAE = (1/n) Σ|ŷ - y|
```
- Linear penalty → all errors weighted equally
- Robust to outliers

**Huber**:
```
Huber(e) = { 0.5*e²        if |e| ≤ δ
           { δ(|e| - δ/2)  if |e| > δ
```
- Best of both worlds!

**Binary Cross-Entropy**:
```
BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```
- For binary classification
- Penalizes confident mistakes heavily

### 2. When to Use Each Loss

| Loss Function | Use When |
|---------------|----------|
| MSE | Large errors are particularly bad, no outliers |
| MAE | Data has outliers, want interpretable error |
| Huber | Want smooth gradients + robustness |
| RMSE | Want MSE but in original units (reporting) |
| Binary Cross-Entropy | Binary classification (2 classes) |
| Categorical Cross-Entropy | Multi-class classification (3+ classes) |

### 3. Outlier Sensitivity

For outlier with error = 9.5:
- **MSE contribution**: 9.5² = 90.25 (huge!)
- **MAE contribution**: |9.5| = 9.5 (linear)
- **Result**: MSE increases 91x, MAE only 11x

### 4. Implementation Details

Fixed defn compatibility by wrapping:
```elixir
def huber(predictions, targets, opts \\ []) do
  delta = Keyword.get(opts, :delta, 1.0)
  huber_impl(predictions, targets, delta)
end

defnp huber_impl(predictions, targets, delta) do
  # Implementation using only Nx operations
end
```

This pattern allows optional parameters while keeping numerical code in defn.

---

## 🚀 Git Commit

```bash
git log --oneline -1
# 448256e Commit 2: Loss Functions - Measuring Prediction Error
```

Commit includes:
- 3 files changed
- 1,191 insertions
- Educational commit message

---

## 📚 Connection to Previous Lessons

**Commit 1 (Gradient Descent)** + **Commit 2 (Loss Functions)** = Machine Learning!

```
Training Loop:
  1. Predict: ŷ = model(X)
  2. Compute loss: L = loss_fn(ŷ, y)  ← This commit!
  3. Compute gradients: ∇L
  4. Update: θ = θ - α∇L  ← Commit 1!
  5. Repeat
```

---

## 🎯 Learning Objectives Achieved

✅ Understand what loss functions measure  
✅ Know when to use MSE vs MAE vs Huber  
✅ Understand cross-entropy for classification  
✅ Recognize outlier sensitivity in different losses  
✅ Implement loss functions with proper Nx/defn patterns  
✅ Choose appropriate loss for different problems  

**You now know how to measure prediction error and choose the right metric!** 🎉

---

## 📊 Quick Reference

### Regression Problems

```elixir
# Standard case
loss = MLNx.LossFunctions.mse(predictions, targets)

# With outliers
loss = MLNx.LossFunctions.mae(predictions, targets)

# Balanced approach
loss = MLNx.LossFunctions.huber(predictions, targets, delta: 1.0)

# All metrics at once
metrics = MLNx.LossFunctions.regression_metrics(predictions, targets)
```

### Classification Problems

```elixir
# Binary classification (spam/not spam)
loss = MLNx.LossFunctions.binary_cross_entropy(predictions, targets)

# Multi-class (cat/dog/bird)
loss = MLNx.LossFunctions.categorical_cross_entropy(predictions, targets)
```

---

## 🔜 What's Next?

**Commit 3: Regularization** will cover:
- L1 Regularization (Lasso) - sparse solutions
- L2 Regularization (Ridge) - small weights
- Elastic Net - combination of both
- Preventing overfitting

You'll learn how to keep models from memorizing training data!
