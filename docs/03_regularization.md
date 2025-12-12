# ML Learning Journey - Commit 3: Regularization

## 🎓 What You Learned

**Regularization** prevents overfitting by adding a penalty for large weights to the loss function. This encourages simpler models that generalize better to new data.

### Core Concept

```
Total Loss = Data Loss + λ * Regularization Penalty

Without regularization: Model memorizes training data
With regularization: Model learns generalizable patterns
```

---

## 📁 Files Created

### 1. [regularization.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/regularization.ex)

**3 Regularization Types** with extensive documentation:

- `l2_penalty/2` - L2 (Ridge): λ * Σw²
- `l1_penalty/2` - L1 (Lasso): λ * Σ|w|
- `elastic_net_penalty/3` - Combines L1 and L2
- `regularized_loss/4` - Adds penalty to MSE loss
- `compare_lambdas/2` - Compare different λ values

### 2. [regularization_test.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/test/regularization_test.exs)

**18 Comprehensive Tests** covering:
- L2 penalty computation and properties
- L1 penalty and sparsity
- Elastic Net mixing
- Regularized loss calculation
- L1 vs L2 comparison

### 3. [03_regularization_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/03_regularization_demo.exs)

**6 Interactive Examples:**
1. L2 regularization (weight shrinkage)
2. L1 regularization (feature selection)
3. L1 vs L2 side-by-side comparison
4. Elastic Net mixing
5. Effect on total loss
6. Choosing lambda (λ)

---

## ✅ Test Results

```
Running ExUnit with seed: 0, max_cases: 8

..................
Finished in 0.9 seconds (0.00s async, 0.9s sync)
18 tests, 0 failures
```

All tests pass! Key validations:
- ✓ L2 penalizes large weights heavily (squared)
- ✓ L1 treats all weights equally (linear)
- ✓ Elastic Net smoothly mixes L1 and L2
- ✓ Regularization increases total loss
- ✓ Lambda controls penalty strength

---

## 🎬 Demo Highlights

### Example 1: L2 Regularization

For weights [5.0, -3.0, 2.0, -1.5]:

| λ | L2 Penalty |
|---|------------|
| 0.00 | 0.0 |
| 0.01 | 0.403 |
| 0.10 | 4.025 |
| 1.00 | 40.25 |

**Insight**: L2 penalty = λ * (5² + 3² + 2² + 1.5²) = λ * 40.25. Large weights contribute much more due to squaring!

### Example 2: L1 Regularization

For same weights:

| λ | L1 Penalty |
|---|------------|
| 0.00 | 0.0 |
| 0.01 | 0.115 |
| 0.10 | 1.15 |
| 1.00 | 11.5 |

**Insight**: L1 penalty = λ * (|5| + |3| + |2| + |1.5|) = λ * 11.5. Linear penalty - all weights contribute equally!

### Example 3: L1 vs L2 Comparison

For different weight magnitudes:

| Weight | L1 Penalty | L2 Penalty | L2/L1 Ratio |
|--------|------------|------------|-------------|
| 1.0 | 0.100 | 0.100 | 1.00 |
| 2.0 | 0.200 | 0.400 | 2.00 |
| 5.0 | 0.500 | 2.500 | 5.00 |
| 10.0 | 1.000 | 10.000 | 10.00 |

**Insight**: L2 penalty grows quadratically! For weight=10, L2 is 10x larger than L1.

---

## 🔑 Key Takeaways

### 1. Why Regularization?

**Problem**: Overfitting
- Model fits training data perfectly
- But performs poorly on new data
- Learns noise instead of signal

**Solution**: Regularization
- Penalizes large weights
- Encourages simpler models
- Better generalization

### 2. L2 Regularization (Ridge)

```
L2 = λ * Σw²
```

**Properties**:
- Shrinks ALL weights toward zero
- Keeps all features
- Penalizes large weights heavily (squared)

**Effect**: weights [10, -8, 15] → [2, -1.5, 3]

### 3. L1 Regularization (Lasso)

```
L1 = λ * Σ|w|
```

**Properties**:
- Forces some weights to EXACTLY ZERO
- Automatic feature selection!
- Linear penalty

**Effect**: weights [10, -8, 0.5, -0.3] → [8, -6, 0, 0]

### 4. Why L1 Creates Sparsity

L1 has "corners" at zero during optimization. Weights get pushed to these corners and stick there.

L2 is smooth everywhere, so weights rarely hit exactly zero.

### 5. Elastic Net

```
Elastic Net = λ * [α*Σ|w| + (1-α)*Σw²]
```

- α = 0: Pure L2
- α = 1: Pure L1
- 0 < α < 1: Mix of both

**Best of both worlds!**

### 6. Choosing Lambda (λ)

- λ = 0: No regularization (may overfit)
- Small λ: Mild regularization
- Large λ: Strong regularization (may underfit)

**Solution**: Tune using validation data!

---

## 🚀 Git Commit

```bash
git log --oneline -1
# 8e04d9e Commit 3: Regularization - Preventing Overfitting
```

Commit includes:
- 3 files changed
- 973 insertions
- Educational commit message

---

## 📚 Connection to Previous Lessons

**Complete Training Loop**:

```
1. Predict: ŷ = model(X)
2. Compute loss: L = MSE(ŷ, y) + λ*reg(w)  ← Regularization added!
3. Compute gradients: ∇L  ← Commit 1
4. Update: θ = θ - α∇L
5. Repeat
```

Regularization affects the gradients, pulling weights toward zero!

---

## 🎯 Learning Objectives Achieved

✅ Understand what regularization is and why we need it  
✅ Know L2 shrinks all weights (keeps all features)  
✅ Know L1 forces sparsity (automatic feature selection)  
✅ Understand Elastic Net combines both  
✅ Know how to choose regularization strength (λ)  
✅ Understand when to use each type  

**You now know how to prevent overfitting!** 🎉

---

## 📊 Quick Reference

### When to Use Each

| Regularization | Use When |
|----------------|----------|
| L2 (Ridge) | All features are relevant, want to shrink weights |
| L1 (Lasso) | Want automatic feature selection, sparse solutions |
| Elastic Net | Want both benefits, features are correlated |

### Code Examples

```elixir
# L2 regularization
penalty = MLNx.Regularization.l2_penalty(weights, 0.1)

# L1 regularization
penalty = MLNx.Regularization.l1_penalty(weights, 0.1)

# Elastic Net (50% L1, 50% L2)
penalty = MLNx.Regularization.elastic_net_penalty(weights, 0.1, 0.5)

# Regularized loss
loss = MLNx.Regularization.regularized_loss(
  predictions, targets, weights,
  reg_type: :l2, lambda: 0.1
)
```

---

## 🔜 What's Next?

**Commit 4: Feature Normalization** will cover:
- Min-max scaling
- Standardization (z-score)
- Why normalization helps gradient descent
- When to normalize features

You'll learn how to prepare data for better model performance!
