# ML Learning Journey - Commit 1: Gradient Descent

## 🎓 What You Learned

**Gradient Descent** is the foundation of all machine learning optimization. It's an iterative algorithm that finds the minimum of a function by following the steepest downward slope.

### Core Concept

```
θ_new = θ_old - α * ∇f(θ_old)
```

- **θ (theta)**: Parameters we're optimizing
- **α (alpha)**: Learning rate (step size)
- **∇f (nabla f)**: Gradient (direction of steepest ascent)

We subtract the gradient because we want to minimize (go down), not maximize.

---

## 📁 Files Created

### 1. [gradient_descent.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/gradient_descent.ex)

**Core Implementation** with extensive inline documentation:

- `optimize/3` - Main optimization function
- `step/3` - Single gradient descent step
- `gradient_norm/1` - Compute gradient magnitude
- `minimize_quadratic/3` - Teaching example

**Documentation Highlights**:
- Module docstring explains the mountain analogy
- Each function has mathematical formulas
- Inline comments explain the "why" behind each step

### 2. [gradient_descent_test.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/test/gradient_descent_test.exs)

**12 Comprehensive Tests** (all passing ✓):

- Basic step mechanics
- Gradient norm calculation
- Quadratic function minimization
- History tracking
- Learning rate effects
- Convergence criteria

### 3. [gradient_descent_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/gradient_descent_demo.exs)

**4 Interactive Examples**:

1. **Simple Quadratic**: Minimize f(x) = (x - 3)²
2. **Learning Rate Comparison**: Show impact of α on convergence
3. **2D Optimization**: Multi-dimensional parameter space
4. **Convergence Criteria**: Tight vs loose tolerance

---

## ✅ Test Results

```
Running ExUnit with seed: 800088, max_cases: 8

............
Finished in 1.4 seconds (0.00s async, 1.4s sync)
12 tests, 0 failures
```

All tests pass! Key validations:
- ✓ Update rule works correctly
- ✓ Moves opposite to gradient direction
- ✓ Converges to minimum
- ✓ Learning rate affects convergence speed
- ✓ History tracking captures optimization journey

---

## 🎬 Demo Highlights

### Example 1: Minimizing f(x) = (x - 3)²

Starting from x = 10.0, the algorithm converged to x ≈ 3.0 in 50 iterations:

```
Iter |      x      | Gradient Norm | Distance to Min
----------------------------------------------------------------------
   1 |    3.000125 |      0.000249 |        0.000125
  10 |    3.000930 |      0.001861 |        0.000930
  50 |    3.000100 |      0.000200 |        0.000100
```

**Final error**: 9.97e-5 (very close to target!)

### Example 2: Learning Rate Impact

| Learning Rate | Iterations | Final Error |
|--------------|------------|-------------|
| 0.01 (small) | 100 | 1.326 (didn't converge) |
| 0.10 (good) | 77 | 0.000 (converged!) |
| 0.50 (large) | 2 | 0.000 (fast!) |
| 0.90 (large) | 77 | 0.000 (oscillated) |

**Insight**: Medium learning rates balance speed and stability.

### Example 3: 2D Optimization

Minimizing f(x, y) = (x - 2)² + (y + 1)²

- Started at: [5.0, 3.0]
- Converged to: [2.000, -1.000] in 50 iterations
- Both parameters optimized simultaneously!

### Example 4: Convergence Tolerance

| Tolerance | Iterations | Final Error |
|-----------|------------|-------------|
| 1e-8 (tight) | 97 | 4.97e-9 |
| 1.0 (loose) | 15 | 0.440 |

**Tradeoff**: Tighter tolerance = more accuracy but more computation.

---

## 🔑 Key Takeaways

### 1. The Update Rule

```elixir
defn step(params, gradient, learning_rate) do
  Nx.subtract(params, Nx.multiply(learning_rate, gradient))
end
```

This simple formula is the heart of ML optimization!

### 2. Learning Rate Matters

- **Too small**: Slow convergence, wasted computation
- **Too large**: Overshooting, divergence, instability
- **Just right**: Efficient path to minimum

### 3. Convergence Detection

Stop when:
- Gradient norm < tolerance (we're at a minimum)
- OR max iterations reached (computational budget)

### 4. Real-World Applications

Gradient descent is used in:
- Linear regression (minimize MSE)
- Logistic regression (minimize cross-entropy)
- Neural networks (backpropagation)
- Any differentiable loss function!

---

## 🚀 Git Commit

```bash
git log --oneline -1
# 3fc538b Commit 1: Gradient Descent - The Foundation of ML Optimization
```

Commit includes:
- 3 files changed
- 760 insertions
- Educational commit message explaining concepts

---

## 📚 What's Next?

**Commit 2: Loss Functions** will cover:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss (robust to outliers)
- Cross-Entropy (for classification)

You'll learn when to use each loss function and how they affect optimization!

---

## 🎯 Learning Objectives Achieved

✅ Understand gradient descent intuition (mountain analogy)  
✅ Know the mathematical update rule  
✅ Recognize learning rate impact  
✅ Implement convergence criteria  
✅ Optimize multi-dimensional functions  
✅ Read and write heavily documented ML code  

**You now understand the foundation of ALL machine learning optimization!** 🎉
