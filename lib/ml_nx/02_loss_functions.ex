defmodule MLNx.LossFunctions do
  @moduledoc """
  Loss Functions - Measuring Prediction Error

  ## What is a Loss Function?

  A loss function (also called cost function or objective function) measures how
  wrong our model's predictions are. It takes the predicted values and true values,
  and returns a single number representing the error.

  The goal of machine learning is to MINIMIZE this loss function.

  ## Why Do We Need Different Loss Functions?

  Different problems and data characteristics require different loss functions:

  - **Regression** (predicting continuous values): MSE, MAE, Huber
  - **Classification** (predicting categories): Cross-Entropy
  - **Outliers present**: Huber Loss (robust)
  - **Interpretability**: MAE (average error in original units)

  ## The Big Picture

  ```
  Training Loop:
  1. Make predictions: ŷ = model(X)
  2. Compute loss: L = loss_function(ŷ, y)
  3. Compute gradients: ∇L
  4. Update parameters: θ = θ - α∇L
  5. Repeat until loss is small
  ```

  The loss function is what we're trying to minimize with gradient descent!

  ## Loss Functions Covered

  1. **Mean Squared Error (MSE)** - L2 loss, penalizes large errors heavily
  2. **Mean Absolute Error (MAE)** - L1 loss, robust to outliers
  3. **Huber Loss** - Combines MSE and MAE, best of both worlds
  4. **Binary Cross-Entropy** - For binary classification (0 or 1)
  5. **Categorical Cross-Entropy** - For multi-class classification
  """

  import Nx.Defn

  # ============================================================================
  # REGRESSION LOSSES
  # ============================================================================

  @doc """
  Mean Squared Error (MSE) - L2 Loss

  ## Formula

      MSE = (1/n) Σ(ŷᵢ - yᵢ)²

  Where:
  - ŷ (y-hat) = predicted values
  - y = true values
  - n = number of samples

  ## Characteristics

  - **Penalizes large errors heavily** (squared term)
  - **Sensitive to outliers** (one huge error dominates)
  - **Differentiable everywhere** (smooth gradients)
  - **Units**: squared units of y (e.g., if y is in dollars, MSE is in dollars²)

  ## When to Use

  - When large errors are particularly bad
  - When you have no outliers in your data
  - Most common choice for regression

  ## Gradient

      ∂MSE/∂ŷ = 2(ŷ - y) / n

  ## Example

      predictions = Nx.tensor([2.5, 3.0, 4.5])
      targets = Nx.tensor([3.0, 3.0, 4.0])
      
      loss = MLNx.LossFunctions.mse(predictions, targets)
      # MSE = ((2.5-3)² + (3-3)² + (4.5-4)²) / 3
      #     = (0.25 + 0 + 0.25) / 3
      #     = 0.167
  """
  defn mse(predictions, targets) do
    # Compute error for each sample
    # Error = predicted - actual
    errors = Nx.subtract(predictions, targets)

    # Square each error
    # This makes all errors positive and penalizes large errors more
    squared_errors = Nx.multiply(errors, errors)

    # Take the mean across all samples
    # This gives us a single number representing average squared error
    Nx.mean(squared_errors)
  end

  @doc """
  Root Mean Squared Error (RMSE)

  ## Formula

      RMSE = √(MSE) = √((1/n) Σ(ŷᵢ - yᵢ)²)

  ## Characteristics

  - Same as MSE but in original units (not squared)
  - **More interpretable** than MSE
  - Still sensitive to outliers

  ## When to Use

  - When you want error in the same units as your target
  - For reporting/interpretation purposes

  ## Example

      predictions = Nx.tensor([2.5, 3.0, 4.5])
      targets = Nx.tensor([3.0, 3.0, 4.0])
      
      loss = MLNx.LossFunctions.rmse(predictions, targets)
      # RMSE = √0.167 ≈ 0.408
  """
  defn rmse(predictions, targets) do
    # Compute MSE first, then take square root
    predictions
    |> mse(targets)
    |> Nx.sqrt()
  end

  @doc """
  Mean Absolute Error (MAE) - L1 Loss

  ## Formula

      MAE = (1/n) Σ|ŷᵢ - yᵢ|

  ## Characteristics

  - **Linear penalty** (not squared)
  - **Robust to outliers** (doesn't square large errors)
  - **Not differentiable at zero** (gradient discontinuity)
  - **Units**: same units as y (interpretable!)

  ## When to Use

  - When you have outliers in your data
  - When all errors should be weighted equally
  - When you want interpretable error (e.g., "average $5 off")

  ## Comparison with MSE

  For error = 2:
  - MSE contribution: 2² = 4
  - MAE contribution: |2| = 2

  For error = 10 (outlier):
  - MSE contribution: 10² = 100 (huge!)
  - MAE contribution: |10| = 10 (linear)

  ## Gradient

      ∂MAE/∂ŷ = sign(ŷ - y) / n

  ## Example

      predictions = Nx.tensor([2.5, 3.0, 4.5])
      targets = Nx.tensor([3.0, 3.0, 4.0])
      
      loss = MLNx.LossFunctions.mae(predictions, targets)
      # MAE = (|2.5-3| + |3-3| + |4.5-4|) / 3
      #     = (0.5 + 0 + 0.5) / 3
      #     = 0.333
  """
  defn mae(predictions, targets) do
    # Compute error for each sample
    errors = Nx.subtract(predictions, targets)

    # Take absolute value
    # This makes all errors positive without squaring
    absolute_errors = Nx.abs(errors)

    # Take the mean
    Nx.mean(absolute_errors)
  end

  @doc """
  Huber Loss - Robust Loss Function

  ## Formula

      Huber(e) = {
        (1/2)e²           if |e| ≤ δ
        δ(|e| - δ/2)      if |e| > δ
      }

  Where e = ŷ - y (error) and δ (delta) is a threshold parameter.

  ## Characteristics

  - **Quadratic for small errors** (like MSE, smooth gradients)
  - **Linear for large errors** (like MAE, robust to outliers)
  - **Best of both worlds!**
  - **Differentiable everywhere** (unlike MAE)

  ## When to Use

  - When you have some outliers but still want smooth gradients
  - When you want robustness without sacrificing differentiability
  - Common in robust regression and reinforcement learning

  ## How It Works

  ```
  Small error (|e| ≤ δ): Use MSE behavior
    - Smooth gradients for optimization
    - Precise near the minimum

  Large error (|e| > δ): Use MAE behavior
    - Don't let outliers dominate
    - Linear penalty instead of quadratic
  ```

  ## Choosing Delta (δ)

  - Smaller δ: More robust (more like MAE)
  - Larger δ: Less robust (more like MSE)
  - Typical: δ = 1.0 or δ = standard deviation of errors

  ## Example

      predictions = Nx.tensor([1.0, 5.0, 10.0])
      targets = Nx.tensor([1.5, 5.0, 2.0])  # Last one is outlier
      
      # With delta = 1.0
      loss = MLNx.LossFunctions.huber(predictions, targets, delta: 1.0)
      
      # Errors: [-0.5, 0, 8.0]
      # First two use quadratic (small), last uses linear (outlier)
  """
  def huber(predictions, targets, opts \\ []) do
    delta = Keyword.get(opts, :delta, 1.0)
    huber_impl(predictions, targets, delta)
  end

  defnp huber_impl(predictions, targets, delta) do
    # Compute errors
    errors = Nx.subtract(predictions, targets)
    abs_errors = Nx.abs(errors)

    # For small errors (|e| ≤ δ): use quadratic loss = 0.5 * e²
    quadratic = Nx.multiply(0.5, Nx.multiply(errors, errors))

    # For large errors (|e| > δ): use linear loss = δ(|e| - δ/2)
    linear = Nx.multiply(delta, Nx.subtract(abs_errors, Nx.multiply(0.5, delta)))

    # Choose quadratic or linear based on error magnitude
    # If |error| <= delta, use quadratic, else use linear
    loss_per_sample = Nx.select(
      Nx.less_equal(abs_errors, delta),
      quadratic,
      linear
    )

    # Return mean loss
    Nx.mean(loss_per_sample)
  end

  # ============================================================================
  # CLASSIFICATION LOSSES
  # ============================================================================

  @doc """
  Binary Cross-Entropy Loss (Log Loss)

  ## Formula

      BCE = -(1/n) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

  Where:
  - y ∈ {0, 1} = true labels (0 or 1)
  - ŷ ∈ (0, 1) = predicted probabilities

  ## Characteristics

  - **For binary classification** (two classes)
  - **Measures probability distribution difference**
  - **Heavily penalizes confident wrong predictions**
  - **Always positive** (minimum is 0 for perfect predictions)

  ## Intuition

  When true label is 1:
  - If ŷ = 0.9 (confident, correct): loss ≈ 0.1
  - If ŷ = 0.1 (confident, wrong): loss ≈ 2.3 (huge!)

  When true label is 0:
  - If ŷ = 0.1 (confident, correct): loss ≈ 0.1
  - If ŷ = 0.9 (confident, wrong): loss ≈ 2.3 (huge!)

  ## When to Use

  - Binary classification problems
  - When predictions are probabilities (0 to 1)
  - After sigmoid activation in neural networks

  ## Important Note

  Predictions must be in range (0, 1). Use sigmoid activation:
  
      ŷ = sigmoid(z) = 1 / (1 + e^(-z))

  ## Example

      # Predicting if email is spam (1) or not (0)
      predictions = Nx.tensor([0.9, 0.2, 0.8])  # Probabilities
      targets = Nx.tensor([1.0, 0.0, 1.0])      # True labels
      
      loss = MLNx.LossFunctions.binary_cross_entropy(predictions, targets)
      # Low loss because predictions match targets well
  """
  def binary_cross_entropy(predictions, targets, opts \\ []) do
    epsilon = Keyword.get(opts, :epsilon, 1.0e-7)
    binary_cross_entropy_impl(predictions, targets, epsilon)
  end

  defnp binary_cross_entropy_impl(predictions, targets, epsilon) do
    # Clip predictions to avoid numerical instability
    # log(0) = -∞, so we ensure predictions are in (ε, 1-ε)
    predictions = Nx.clip(predictions, epsilon, 1.0 - epsilon)

    # Compute loss for each sample
    # Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
    
    # First term: y * log(ŷ)
    # This is non-zero only when y = 1
    term1 = Nx.multiply(targets, Nx.log(predictions))

    # Second term: (1-y) * log(1-ŷ)
    # This is non-zero only when y = 0
    term2 = Nx.multiply(
      Nx.subtract(1.0, targets),
      Nx.log(Nx.subtract(1.0, predictions))
    )

    # Combine and negate
    loss_per_sample = Nx.negate(Nx.add(term1, term2))

    # Return mean loss
    Nx.mean(loss_per_sample)
  end

  @doc """
  Categorical Cross-Entropy Loss

  ## Formula

      CCE = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)

  Where:
  - y = one-hot encoded true labels (e.g., [0, 1, 0] for class 2)
  - ŷ = predicted probability distribution (from softmax)
  - i = sample index
  - j = class index

  ## Characteristics

  - **For multi-class classification** (more than 2 classes)
  - **Predictions must sum to 1** (use softmax activation)
  - **Measures KL divergence** between true and predicted distributions

  ## Intuition

  For 3 classes, if true class is 2:
  - True: [0, 1, 0]
  - Pred: [0.1, 0.8, 0.1] → loss ≈ 0.22 (good)
  - Pred: [0.5, 0.3, 0.2] → loss ≈ 1.20 (bad)

  ## When to Use

  - Multi-class classification (3+ classes)
  - After softmax activation
  - When classes are mutually exclusive

  ## Softmax Activation

  Converts logits to probabilities:

      softmax(zⱼ) = e^(zⱼ) / Σₖ e^(zₖ)

  ## Example

      # Classifying images into 3 categories: cat, dog, bird
      # True labels (one-hot): [cat, dog, cat]
      targets = Nx.tensor([
        [1.0, 0.0, 0.0],  # cat
        [0.0, 1.0, 0.0],  # dog
        [1.0, 0.0, 0.0]   # cat
      ])
      
      # Predicted probabilities (from softmax)
      predictions = Nx.tensor([
        [0.8, 0.1, 0.1],  # Confident cat (correct)
        [0.2, 0.7, 0.1],  # Confident dog (correct)
        [0.4, 0.4, 0.2]   # Uncertain (wrong)
      ])
      
      loss = MLNx.LossFunctions.categorical_cross_entropy(predictions, targets)
  """
  def categorical_cross_entropy(predictions, targets, opts \\ []) do
    epsilon = Keyword.get(opts, :epsilon, 1.0e-7)
    categorical_cross_entropy_impl(predictions, targets, epsilon)
  end

  defnp categorical_cross_entropy_impl(predictions, targets, epsilon) do
    # Clip predictions to avoid log(0)
    predictions = Nx.clip(predictions, epsilon, 1.0)

    # Compute -Σ(y * log(ŷ)) for each sample
    # Only the true class contributes (others have y=0)
    log_predictions = Nx.log(predictions)
    loss_per_sample = Nx.multiply(targets, log_predictions)
    
    # Sum across classes (axis 1), then negate
    loss_per_sample = Nx.negate(Nx.sum(loss_per_sample, axes: [1]))

    # Return mean across samples
    Nx.mean(loss_per_sample)
  end

  # ============================================================================
  # UTILITY FUNCTIONS
  # ============================================================================

  @doc """
  Computes multiple loss metrics at once for comparison.

  Useful for evaluating model performance with different metrics.

  ## Returns

  Map with keys: :mse, :rmse, :mae, :huber

  ## Example

      predictions = Nx.tensor([2.5, 3.0, 4.5])
      targets = Nx.tensor([3.0, 3.0, 4.0])
      
      metrics = MLNx.LossFunctions.regression_metrics(predictions, targets)
      # %{mse: 0.167, rmse: 0.408, mae: 0.333, huber: 0.146}
  """
  def regression_metrics(predictions, targets, opts \\ []) do
    %{
      mse: mse(predictions, targets) |> Nx.to_number(),
      rmse: rmse(predictions, targets) |> Nx.to_number(),
      mae: mae(predictions, targets) |> Nx.to_number(),
      huber: huber(predictions, targets, opts) |> Nx.to_number()
    }
  end
end
