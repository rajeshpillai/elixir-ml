# ML Learning Journey - Commit 4: Feature Normalization

## 🎓 What You Learned

**Feature Normalization** scales features to similar ranges, improving gradient descent convergence and model performance. Without normalization, features on different scales cause uneven gradient updates and slow, unstable training.

### Core Concept

```
Problem: Features on different scales (e.g., age: 20-80, salary: 20k-200k)
Solution: Normalize all features to similar ranges
Result: Faster, more stable gradient descent!
```

---

## 📁 Files Created

### 1. [normalization.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/normalization.ex)

**2 Normalization Techniques** with comprehensive documentation:

**Min-Max Scaling:**
- `min_max_scale/1` - Scale to [0, 1]
- `min_max_scale/3` - Scale to custom range
- `inverse_min_max_scale/3` - Reverse transformation

**Standardization:**
- `standardize/1` - Z-score normalization (mean=0, std=1)
- `inverse_standardize/2` - Reverse transformation

**Utilities:**
- `compute_stats/1` - Calculate mean and std
- `normalize/2` - Generic normalization with options

### 2. [normalization_test.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/test/normalization_test.exs)

**38 Comprehensive Tests** (15 doctests + 23 regular tests):
- Min-max scaling (default and custom ranges)
- Standardization (z-score)
- Inverse transformations
- Edge cases (single values, all same values)
- 2D tensor support (per-feature normalization)

### 3. [04_normalization_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/04_normalization_demo.exs)

**6 Interactive Examples:**
1. Min-max scaling to [0, 1]
2. Custom range scaling ([-1, 1], [0, 10])
3. Standardization (z-score)
4. Min-max vs standardization comparison
5. Impact on gradient descent
6. Inverse transformations

---

## ✅ Test Results

```
Running ExUnit with seed: 648621, max_cases: 8

......................................
Finished in 5.6 seconds (0.00s async, 5.6s sync)
15 doctests, 23 tests, 0 failures
```

All 38 tests pass! Key validations:
- ✓ Min-max scales to [0, 1] correctly
- ✓ Custom range scaling works
- ✓ Standardization achieves mean=0, std=1
- ✓ Inverse transformations are perfect roundtrips
- ✓ 2D tensors normalized per-feature

---

## 🎬 Demo Highlights

### Example 1: Min-Max Scaling

Original data: [10, 20, 30, 40, 50]

| Original | Scaled [0, 1] |
|----------|---------------|
| 10 | 0.0 |
| 20 | 0.25 |
| 30 | 0.5 |
| 40 | 0.75 |
| 50 | 1.0 |

**Formula**: (x - min) / (max - min) = (30 - 10) / (50 - 10) = 0.5 ✓

### Example 2: Custom Range Scaling

Original data: [0, 25, 50, 75, 100]

| Range | Scaled Values |
|-------|---------------|
| [0, 1] | [0.0, 0.25, 0.5, 0.75, 1.0] |
| [-1, 1] | [-1.0, -0.5, 0.0, 0.5, 1.0] |
| [0, 10] | [0.0, 2.5, 5.0, 7.5, 10.0] |

**Insight**: Same data, different ranges - choose based on your algorithm's needs!

### Example 3: Standardization

Original data: [10, 20, 30, 40, 50]
- Mean: 30.0 → New mean: 0.0
- Std: 14.14 → New std: 1.0

Standardized: [-1.414, -0.707, 0.0, 0.707, 1.414]

**Insight**: Values tell you how many standard deviations from the mean!

### Example 4: Min-Max vs Standardization

Data with outlier: [5, 10, 15, 20, 100]

| Value | Min-Max [0,1] | Standardized |
|-------|---------------|--------------|
| 5 | 0.0 | -0.707 |
| 10 | 0.053 | -0.566 |
| 15 | 0.105 | -0.424 |
| 20 | 0.158 | -0.283 |
| 100 | 1.0 | 1.98 |

**Insight**: 
- Min-Max: Outlier squashes other values near 0 (sensitive!)
- Standardization: Outlier preserved as 1.98 std devs (robust!)

### Example 5: Gradient Descent Impact

**Before normalization:**
- Bedrooms: [2, 3, 4, 2, 5] (range: 2-5)
- Square feet: [800, 1500, 2200, 900, 3000] (range: 800-3000)

Problem: VERY different scales → uneven gradients!

**After normalization:**
- Both features in [0, 1] → SAME scale!
- Same learning rate works for all features
- Fast, stable convergence!

---

## 🔑 Key Takeaways

### 1. Why Normalize?

**Problem**: Features on different scales
- Large-scale features dominate gradient updates
- Small-scale features update too slowly
- Zig-zag path, slow convergence

**Solution**: Normalization
- All features on similar scales
- Equal contribution to learning
- Fast, direct path to minimum

### 2. Min-Max Scaling

```
Formula: (x - min) / (max - min)
```

**Properties**:
- Scales to specific range (default [0, 1])
- Preserves relationships between values
- Sensitive to outliers
- Bounded output

**Use when**:
- Known bounds
- No outliers
- Need specific range (e.g., neural networks)

### 3. Standardization (Z-score)

```
Formula: (x - mean) / std
```

**Properties**:
- Centers data: mean = 0, std = 1
- More robust to outliers
- Unbounded output
- Preserves distribution shape

**Use when**:
- Unknown bounds
- Outliers present
- Features follow normal distribution

### 4. Inverse Transformations

**Critical for predictions**:
1. Normalize training data, **save statistics**
2. Train model on normalized data
3. Normalize new inputs using **SAME statistics**
4. Get predictions (normalized)
5. **Inverse transform** to original scale

Perfect roundtrip: no information lost!

### 5. Practical Tips

- Normalize **AFTER** splitting train/test
- Use **train statistics** for both train and test
- For 2D data, normalize **per-feature** (column-wise)
- Always save normalization statistics!

---

## 🚀 Git Commit

```bash
git log --oneline -1
# [commit hash] Commit 4: Feature Normalization - Scaling for Better Learning
```

Commit includes:
- 3 files changed
- ~600 insertions
- Educational commit message

---

## 📚 Connection to Previous Lessons

**Complete Training Pipeline**:

```
1. Normalize features: X_norm = normalize(X)  ← Commit 4!
2. Predict: ŷ = model(X_norm)
3. Compute loss: L = loss_fn(ŷ, y) + λ*reg(w)  ← Commits 2 & 3
4. Compute gradients: ∇L  ← Commit 1
5. Update: θ = θ - α∇L
6. Repeat
```

Normalization happens **before** training and dramatically improves convergence!

---

## 🎯 Learning Objectives Achieved

✅ Understand why normalization is essential  
✅ Know min-max scaling (bounded, outlier-sensitive)  
✅ Know standardization (unbounded, outlier-robust)  
✅ Understand when to use each technique  
✅ Know how to use inverse transformations  
✅ Understand practical workflow for training  

**You now know how to prepare data for optimal learning!** 🎉

---

## 📊 Quick Reference

### When to Use Each

| Method | Use When |
|--------|----------|
| Min-Max | Known bounds, no outliers, need specific range |
| Standardization | Unknown bounds, outliers present, normal distribution |

### Code Examples

```elixir
# Min-max scaling to [0, 1]
{scaled, stats} = MLNx.Normalization.min_max_scale(data)

# Custom range scaling
{scaled, stats} = MLNx.Normalization.min_max_scale(data, -1, 1)

# Standardization
{standardized, stats} = MLNx.Normalization.standardize(data)

# Inverse transformation
original = MLNx.Normalization.inverse_min_max_scale(scaled, stats)

# Generic normalization
{normalized, stats} = MLNx.Normalization.normalize(data, method: :standardize)
```

---

## 🔜 What's Next?

**Commit 5: Batch Gradient Descent** will cover:
- Batch vs stochastic vs mini-batch
- Implementing batch training
- Convergence comparison
- When to use each variant

You'll learn how to efficiently train on large datasets!
