# ML Learning Journey - Commit 5: Batch Gradient Descent

## 🎓 What You Learned

**Batch Training** provides three strategies for computing gradients: batch (all data), stochastic (one example), and mini-batch (small batches). This enables efficient training on datasets of any size.

### Core Concept

```
Question: How many examples to use per gradient update?

Batch GD: ALL examples → Accurate but slow
Stochastic GD: ONE example → Fast but noisy  
Mini-Batch GD: SMALL batches → Best balance (RECOMMENDED!)
```

---

## 📁 Files Created

### 1. [batch_training.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/batch_training.ex)

**3 Gradient Descent Variants:**
- `batch_gradient_descent/3` - Uses entire dataset
- `stochastic_gradient_descent/3` - Uses one example at a time
- `mini_batch_gradient_descent/3` - Uses small batches

**Utilities:**
- `shuffle_data/2` - Shuffle features and targets together
- `create_batches/3` - Split data into mini-batches
- `train/3` - Generic training with method selection

### 2. [batch_training_test.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/test/batch_training_test.exs)

**27 Comprehensive Tests** (6 doctests + 21 regular):
- Batch GD convergence
- Stochastic GD convergence
- Mini-batch GD convergence
- Batch creation and shuffling
- Edge cases (single example, large batch size)
- Convergence comparison

### 3. [05_batch_training_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/05_batch_training_demo.exs)

**6 Interactive Examples:**
1. Batch gradient descent
2. Stochastic gradient descent
3. Mini-batch gradient descent
4. Convergence comparison
5. Batch size impact
6. Decision guide (when to use each)

---

## ✅ Test Results

```
Running ExUnit with seed: 902619, max_cases: 8

...........................\nFinished in 10.7 seconds (0.00s async, 10.7s sync)
6 doctests, 21 tests, 0 failures
```

All 27 tests pass! Key validations:
- ✓ All three variants converge to similar solutions
- ✓ Batch GD has smooth convergence
- ✓ SGD has noisy but fast convergence
- ✓ Mini-batch balances both
- ✓ Batch creation works correctly

---

## 🎬 Demo Highlights

### Convergence Comparison

Training on y = 2x + 1:

| Method | Weight | Bias | Final Loss | Iterations |
|--------|--------|------|------------|------------|
| Batch GD | 2.09 | 0.49 | 0.054 | 100 |
| Stochastic GD | 2.08 | 0.50 | 0.055 | 20 epochs |
| Mini-Batch GD | 2.10 | 0.46 | 0.061 | 20 epochs |
| **Target** | **2.00** | **1.00** | **-** | **-** |

**Insight**: All three converge to similar solutions!

### Batch Size Impact

| Batch Size | Final Weight | Final Loss |
|------------|--------------|------------|
| 1 (SGD) | 2.071 | 0.0426 |
| 2 | 2.092 | 0.0577 |
| 4 | 2.100 | 0.0686 |
| 8 (Batch) | 2.102 | 0.0748 |

**Insight**: Smaller batches → noisier but faster convergence

---

## 🔑 Key Takeaways

### 1. Batch Gradient Descent

```
gradient = (1/n) * Σ ∇L(x_i)  for ALL examples
```

**Properties**:
- Most accurate gradient
- Smooth convergence
- Slow for large datasets

**Use when**: Small datasets (< 10k examples)

### 2. Stochastic Gradient Descent

```
gradient = ∇L(x_i)  for ONE random example
```

**Properties**:
- Fast updates
- Noisy convergence
- Can escape local minima

**Use when**: Very large datasets, online learning

### 3. Mini-Batch Gradient Descent

```
gradient = (1/B) * Σ ∇L(x_i)  for B examples in batch
```

**Properties**:
- Balance of speed and accuracy
- GPU-efficient
- **MOST COMMONLY USED!**

**Use when**: Most cases (default choice)

### 4. Choosing Batch Size

- **Small (1-16)**: Fast, noisy
- **Medium (32-128)**: Balanced (RECOMMENDED)
- **Large (256+)**: Smooth, slow

**Rule of thumb**: Start with 32

### 5. Practical Tips

- Mini-batch GD is the default choice
- Use powers of 2 for GPU efficiency
- Larger batches may need larger learning rates
- Tune batch size as a hyperparameter

---

## 🚀 Git Commit

```bash
git log --oneline -1
# [hash] Commit 5: Batch Gradient Descent - Efficient Training Strategies
```

Commit includes:
- 3 files changed
- ~800 insertions
- Educational commit message

---

## 📚 Connection to Previous Lessons

**Complete Training Loop**:

```
1. Normalize: X = normalize(X)  ← Commit 4
2. For each epoch:
     Batches = create_batches(X, y)  ← Commit 5!
     For each batch:
       ŷ = model(X_batch)
       L = loss(ŷ, y) + λ*reg(w)  ← Commits 2 & 3
       ∇L = compute_gradients()  ← Commit 1
       θ = θ - α∇L
```

Batch training makes gradient descent practical for large datasets!

---

## 🎯 Learning Objectives Achieved

✅ Understand three gradient descent variants  
✅ Know when to use each variant  
✅ Understand batch size tradeoffs  
✅ Know mini-batch GD is the default choice  
✅ Can implement efficient training strategies  

**You now know how to train efficiently on any dataset size!** 🎉

---

## 📊 Quick Reference

### Decision Guide

| Dataset Size | Method | Batch Size |
|--------------|--------|------------|
| < 10k | Batch GD | Full dataset |
| 10k - 1M | Mini-Batch GD | 32-256 |
| > 1M | Mini-Batch GD | 16-64 |
| Streaming | Stochastic GD | 1 |

### Code Examples

```elixir
# Batch gradient descent
{w, b, history} = MLNx.BatchTraining.batch_gradient_descent(
  x, y, learning_rate: 0.01, iterations: 100
)

# Stochastic gradient descent
{w, b, history} = MLNx.BatchTraining.stochastic_gradient_descent(
  x, y, learning_rate: 0.01, epochs: 20
)

# Mini-batch gradient descent (RECOMMENDED)
{w, b, history} = MLNx.BatchTraining.mini_batch_gradient_descent(
  x, y, batch_size: 32, learning_rate: 0.01, epochs: 20
)

# Generic training
{w, b, history} = MLNx.BatchTraining.train(
  x, y, method: :mini_batch, batch_size: 32, epochs: 20
)
```

---

## 🔜 What's Next?

**Commit 6: Learning Rate Scheduling** will cover:
- Fixed learning rate
- Step decay
- Exponential decay
- Adaptive learning rates

You'll learn how to optimize training speed and convergence!
