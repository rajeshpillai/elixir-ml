# Linear Regression - Your First ML Model

## 🎓 What is Linear Regression?

Linear regression is the simplest machine learning algorithm. It finds the best straight line (or hyperplane) that fits your data.

### The Model

```
y = Xw + b
```

Where:
- **y** = predictions (what we're trying to predict)
- **X** = input features (data we have)
- **w** = weights (slope of the line)
- **b** = bias (y-intercept)

### The Goal

Find the values of **w** and **b** that minimize the prediction error!

---

## 📁 Files

### [linreg.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/linreg.ex)

Core implementation with:
- `predict/3` - Make predictions: ŷ = Xw + b
- `mse/2` - Mean Squared Error loss
- `loss/4` - Compute loss for given parameters
- `step/5` - One gradient descent step
- `train/3` - Full training loop

### [linreg_test.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/test/linreg_test.exs)

Tests covering:
- Prediction with single and multiple features
- MSE calculation
- Training convergence
- Parameter learning

### [00_linreg_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/00_linreg_demo.exs)

Interactive examples:
1. Simple linear relationship (y = 3x + 2)
2. Multiple features (y = 2x₁ + 3x₂ + 1)
3. House price prediction

---

## 🔑 Key Concepts

### 1. The Prediction Function

```elixir
defn predict(x, w, b) do
  Nx.dot(x, w) + b
end
```

This is just matrix multiplication plus a bias term!

### 2. Mean Squared Error

```elixir
defn mse(y_hat, y) do
  err = y_hat - y
  Nx.mean(err * err)
end
```

Measures how far off our predictions are (on average).

### 3. Gradient Descent Step

```elixir
defn step(x, y, w, b, lr) do
  {gw, gb} = grad({w, b}, fn {w, b} ->
    loss(x, y, w, b)
  end)
  
  w = w - lr * gw
  b = b - lr * gb
  
  {w, b}
end
```

Updates weights and bias to reduce the loss.

### 4. Training Loop

```elixir
def train(x, y, opts \\ []) do
  iters = Keyword.get(opts, :iters, 500)
  lr = Keyword.get(opts, :lr, 0.05)
  
  # Initialize parameters
  {_, d} = Nx.shape(x)
  w = Nx.broadcast(0.0, {d})
  b = Nx.tensor(0.0)
  
  # Iterate to minimize loss
  Enum.reduce(1..iters, {w, b}, fn _, {w, b} ->
    step(x, y, w, b, lr)
  end)
end
```

Repeatedly applies gradient descent until convergence.

---

## 📊 Example: Learning y = 3x + 2

```elixir
# Training data
x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = Nx.tensor([5.0, 8.0, 11.0, 14.0, 17.0])

# Train the model
{w, b} = MLNx.LinReg.train(x, y, iters: 1000, lr: 0.01)

# Results
# w ≈ 3.0 (slope)
# b ≈ 2.0 (intercept)
```

The model successfully learned the relationship!

---

## 🎯 What You Learned

✅ Linear regression finds the best line through data  
✅ Model: y = Xw + b  
✅ Loss: Mean Squared Error  
✅ Optimization: Gradient Descent  
✅ Training: Iterate until loss is minimized  

---

## 🔗 Connection to Later Lessons

This simple linear regression contains ALL the core ML concepts:

1. **Model** (predict function) → Neural networks are just more complex models
2. **Loss Function** (MSE) → Commit 2 explores more loss functions
3. **Gradient Descent** (step function) → Commit 1 deep dives into this
4. **Training Loop** (train function) → Foundation for all ML training

Linear regression is the perfect starting point because it's simple but contains everything you need to understand more complex models!

---

## 🚀 Run the Code

```bash
# Run tests
mix test test/linreg_test.exs

# Run demo
mix run examples/00_linreg_demo.exs
```

---

## 📚 Mathematical Details

### Why MSE?

MSE = (1/n) Σ(ŷᵢ - yᵢ)²

- Differentiable (smooth gradients)
- Penalizes large errors heavily
- Convex (one global minimum)

### Gradient Computation

Using automatic differentiation (`grad`):

```
∂MSE/∂w = (2/n) Σ xᵢ(ŷᵢ - yᵢ)
∂MSE/∂b = (2/n) Σ(ŷᵢ - yᵢ)
```

Nx computes these automatically!

### Update Rule

```
w_new = w_old - α * ∂MSE/∂w
b_new = b_old - α * ∂MSE/∂b
```

Where α (alpha) is the learning rate.

---

## 🎓 Next Steps

Now that you understand linear regression, you're ready to learn:

- **Commit 1**: Deep dive into gradient descent
- **Commit 2**: Different loss functions (MAE, Huber, Cross-Entropy)
- **Commit 3**: Regularization (preventing overfitting)

Linear regression is your foundation - everything builds on these concepts! 🚀
