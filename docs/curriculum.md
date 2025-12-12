# ML Learning Curriculum

A 15-commit journey through machine learning fundamentals using Elixir and Nx.

Each commit includes:
- ✅ Documented implementation
- ✅ Comprehensive test suite
- ✅ Interactive demo
- ✅ Educational walkthrough

---

## Completed Commits

### ✅ Commit 0: Linear Regression
**Status**: Complete  
**Files**: [linear_regression.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/linear_regression.ex), [00_linreg_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/00_linreg_demo.exs)  
**Docs**: [00_linear_regression.md](file:///home/rajesh/lab/elixir/ml/ml_nx/docs/00_linear_regression.md)

**What You Learned**:
- Linear regression fundamentals
- Hypothesis function: ŷ = wx + b
- Making predictions
- Basic model structure

---

### ✅ Commit 1: Gradient Descent
**Status**: Complete  
**Files**: [gradient_descent.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/gradient_descent.ex), [01_gradient_descent_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/01_gradient_descent_demo.exs)  
**Docs**: [01_gradient_descent.md](file:///home/rajesh/lab/elixir/ml/ml_nx/docs/01_gradient_descent.md)

**What You Learned**:
- Gradient descent optimization
- Computing gradients
- Learning rate (α)
- Iterative weight updates
- Convergence

---

### ✅ Commit 2: Loss Functions
**Status**: Complete  
**Files**: [loss_functions.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/loss_functions.ex), [02_loss_functions_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/02_loss_functions_demo.exs)  
**Docs**: [02_loss_functions.md](file:///home/rajesh/lab/elixir/ml/ml_nx/docs/02_loss_functions.md)

**What You Learned**:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Binary Cross-Entropy
- Categorical Cross-Entropy
- When to use each loss function

---

### ✅ Commit 3: Regularization
**Status**: Complete  
**Files**: [regularization.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/regularization.ex), [03_regularization_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/03_regularization_demo.exs)  
**Docs**: [03_regularization.md](file:///home/rajesh/lab/elixir/ml/ml_nx/docs/03_regularization.md)

**What You Learned**:
- L1 regularization (Lasso) - feature selection
- L2 regularization (Ridge) - weight shrinkage
- Elastic Net - combining L1 and L2
- Preventing overfitting
- Choosing lambda (λ)

---

### ✅ Commit 4: Feature Normalization
**Status**: Complete  
**Files**: [normalization.ex](file:///home/rajesh/lab/elixir/ml/ml_nx/lib/ml_nx/normalization.ex), [04_normalization_demo.exs](file:///home/rajesh/lab/elixir/ml/ml_nx/examples/04_normalization_demo.exs)  
**Docs**: [04_normalization.md](file:///home/rajesh/lab/elixir/ml/ml_nx/docs/04_normalization.md)

**What You Learned**:
- Min-max scaling to [0, 1]
- Standardization (z-score)
- Why normalization improves gradient descent
- Inverse transformations
- When to use each technique

---

## Upcoming Commits

### 🔜 Commit 5: Batch Gradient Descent
**Topics**:
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Convergence comparison
- When to use each variant

**Why It Matters**: Learn how to efficiently train on large datasets by processing data in batches.

---

### 📋 Commit 6: Learning Rate Scheduling
**Topics**:
- Fixed learning rate
- Step decay
- Exponential decay
- Adaptive learning rates
- Learning rate warmup

**Why It Matters**: Optimize training speed and convergence by adjusting learning rate over time.

---

### 📋 Commit 7: Momentum & Optimization
**Topics**:
- Momentum
- Nesterov accelerated gradient
- RMSprop
- Adam optimizer
- Comparing optimizers

**Why It Matters**: Advanced optimization techniques that converge faster than vanilla gradient descent.

---

### 📋 Commit 8: Logistic Regression
**Topics**:
- Binary classification
- Sigmoid activation
- Decision boundary
- Probability predictions
- Classification metrics

**Why It Matters**: Foundation for neural networks and classification tasks.

---

### 📋 Commit 9: Multi-class Classification
**Topics**:
- Softmax activation
- One-hot encoding
- Multi-class cross-entropy
- Confusion matrix
- Precision, recall, F1-score

**Why It Matters**: Classify data into multiple categories (e.g., digit recognition).

---

### 📋 Commit 10: Neural Networks - Single Layer
**Topics**:
- Perceptron
- Activation functions (ReLU, tanh)
- Forward propagation
- Backpropagation
- Training a single-layer network

**Why It Matters**: Building blocks of deep learning.

---

### 📋 Commit 11: Neural Networks - Multi-layer
**Topics**:
- Hidden layers
- Deep networks
- Vanishing/exploding gradients
- Weight initialization
- Network architecture design

**Why It Matters**: Create powerful models that learn complex patterns.

---

### 📋 Commit 12: Dropout & Batch Normalization
**Topics**:
- Dropout regularization
- Batch normalization
- Layer normalization
- Preventing overfitting in deep networks
- Training vs inference mode

**Why It Matters**: Essential techniques for training deep networks effectively.

---

### 📋 Commit 13: Convolutional Neural Networks (CNNs)
**Topics**:
- Convolution operation
- Pooling layers
- CNN architecture
- Image classification
- Feature maps

**Why It Matters**: State-of-the-art for computer vision tasks.

---

### 📋 Commit 14: Model Evaluation & Validation
**Topics**:
- Train/validation/test splits
- Cross-validation
- Overfitting vs underfitting
- Learning curves
- Model selection

**Why It Matters**: Ensure your model generalizes well to new data.

---

### 📋 Commit 15: Complete ML Pipeline
**Topics**:
- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Model deployment
- End-to-end project

**Why It Matters**: Put it all together in a real-world ML project.

---

## Learning Path

```
Foundation (Commits 0-4):
  ├─ Linear Regression
  ├─ Gradient Descent
  ├─ Loss Functions
  ├─ Regularization
  └─ Feature Normalization

Optimization (Commits 5-7):
  ├─ Batch Gradient Descent
  ├─ Learning Rate Scheduling
  └─ Advanced Optimizers

Classification (Commits 8-9):
  ├─ Logistic Regression
  └─ Multi-class Classification

Neural Networks (Commits 10-13):
  ├─ Single Layer
  ├─ Multi-layer
  ├─ Regularization Techniques
  └─ CNNs

Production (Commits 14-15):
  ├─ Model Evaluation
  └─ Complete Pipeline
```

---

## Progress Tracker

- [x] Commit 0: Linear Regression
- [x] Commit 1: Gradient Descent
- [x] Commit 2: Loss Functions
- [x] Commit 3: Regularization
- [x] Commit 4: Feature Normalization
- [ ] Commit 5: Batch Gradient Descent
- [ ] Commit 6: Learning Rate Scheduling
- [ ] Commit 7: Momentum & Optimization
- [ ] Commit 8: Logistic Regression
- [ ] Commit 9: Multi-class Classification
- [ ] Commit 10: Neural Networks - Single Layer
- [ ] Commit 11: Neural Networks - Multi-layer
- [ ] Commit 12: Dropout & Batch Normalization
- [ ] Commit 13: Convolutional Neural Networks
- [ ] Commit 14: Model Evaluation & Validation
- [ ] Commit 15: Complete ML Pipeline

**Current Progress**: 5/15 commits complete (33%)

---

## How to Use This Curriculum

1. **Read the walkthrough** for each commit in the `docs/` folder
2. **Run the tests** to verify your understanding
3. **Explore the demo** to see concepts in action
4. **Study the code** to learn implementation details
5. **Experiment** by modifying examples

Each commit builds on previous knowledge, so follow the order!

---

## Resources

- **Nx Documentation**: https://hexdocs.pm/nx
- **EXLA Backend**: https://hexdocs.pm/exla
- **Machine Learning Fundamentals**: Study the math in commit messages

---

## Next Steps

Currently working on: **Commit 5: Batch Gradient Descent**

Ready to learn how to efficiently train on large datasets! 🚀
