defmodule MLNx.Regularization do
  @moduledoc """
  Regularization - Preventing Overfitting

  ## What is Regularization?

  Regularization adds a penalty term to the loss function to discourage the model
  from learning overly complex patterns. This prevents overfitting - when a model
  memorizes training data instead of learning generalizable patterns.

  ## The Problem: Overfitting

  Without regularization:
  - Model fits training data perfectly (low training error)
  - But performs poorly on new data (high test error)
  - Learns noise instead of signal

  ## The Solution: Penalty Terms

  Add a penalty for large weights to the loss function:

      Total Loss = Data Loss + λ * Regularization Term

  Where λ (lambda) controls the strength of regularization.

  ## Types of Regularization

  1. **L2 Regularization (Ridge)** - Penalizes sum of squared weights
     - Shrinks all weights toward zero
     - Keeps all features
     - Formula: λ * Σw²

  2. **L1 Regularization (Lasso)** - Penalizes sum of absolute weights
     - Forces some weights to exactly zero
     - Performs feature selection
     - Formula: λ * Σ|w|

  3. **Elastic Net** - Combines L1 and L2
     - Gets benefits of both
     - Formula: λ₁ * Σ|w| + λ₂ * Σw²

  ## How It Works

  ```
  Without regularization:
    Loss = MSE(predictions, targets)
    → Model can use any weight values

  With L2 regularization:
    Loss = MSE(predictions, targets) + λ * Σw²
    → Large weights increase loss
    → Model prefers smaller weights
    → Simpler, more generalizable model
  ```

  ## Choosing Lambda (λ)

  - λ = 0: No regularization (may overfit)
  - Small λ: Mild regularization
  - Large λ: Strong regularization (may underfit)
  - Tune λ using validation data!
  """

  import Nx.Defn

  @doc """
  L2 Regularization (Ridge) - Sum of Squared Weights

  ## Formula

      L2 = λ * Σwᵢ²

  ## Characteristics

  - **Shrinks all weights** toward zero (but not to exactly zero)
  - **Keeps all features** in the model
  - **Smooth penalty** (differentiable everywhere)
  - **Penalizes large weights heavily** (squared term)

  ## When to Use

  - When all features might be relevant
  - When you want to reduce weight magnitudes
  - Most common choice for regularization

  ## Effect on Weights

  Without L2: weights can be [10, -8, 15, -12]
  With L2:    weights become [2, -1.5, 3, -2.2]
  
  All weights shrink, but none become exactly zero.

  ## Mathematical Intuition

  The gradient of L2 penalty is:
  
      ∂(λΣw²)/∂w = 2λw

  This adds a term proportional to the weight itself, pulling it toward zero.

  ## Example

      weights = Nx.tensor([2.0, -3.0, 1.5])
      lambda = 0.1
      
      penalty = MLNx.Regularization.l2_penalty(weights, lambda)
      # L2 = 0.1 * (2² + 3² + 1.5²) = 0.1 * 15.25 = 1.525
  """
  defn l2_penalty(weights, lambda) do
    # Compute sum of squared weights: Σw²
    squared_weights = Nx.multiply(weights, weights)
    sum_squared = Nx.sum(squared_weights)
    
    # Multiply by lambda
    Nx.multiply(lambda, sum_squared)
  end

  @doc """
  L1 Regularization (Lasso) - Sum of Absolute Weights

  ## Formula

      L1 = λ * Σ|wᵢ|

  ## Characteristics

  - **Forces some weights to exactly zero** (feature selection!)
  - **Sparse solutions** (many weights = 0)
  - **Not differentiable at zero** (uses subgradient)
  - **Linear penalty** (doesn't square weights)

  ## When to Use

  - When you want automatic feature selection
  - When you believe many features are irrelevant
  - When you want interpretable models (fewer features)

  ## Effect on Weights

  Without L1: weights can be [10, -8, 0.5, -0.3, 15]
  With L1:    weights become [8, -6, 0, 0, 12]
  
  Small weights become exactly zero! This is feature selection.

  ## Why Does L1 Create Sparsity?

  The L1 penalty has "corners" at zero. During optimization, weights
  often get pushed to these corners and stick there.

  L2 penalty is smooth everywhere, so weights rarely hit exactly zero.

  ## Mathematical Intuition

  The subgradient of L1 penalty is:
  
      ∂(λΣ|w|)/∂w = λ * sign(w)

  This is constant (doesn't depend on weight magnitude), so small weights
  get the same push toward zero as large weights.

  ## Example

      weights = Nx.tensor([2.0, -3.0, 1.5])
      lambda = 0.1
      
      penalty = MLNx.Regularization.l1_penalty(weights, lambda)
      # L1 = 0.1 * (|2| + |-3| + |1.5|) = 0.1 * 6.5 = 0.65
  """
  defn l1_penalty(weights, lambda) do
    # Compute sum of absolute weights: Σ|w|
    abs_weights = Nx.abs(weights)
    sum_abs = Nx.sum(abs_weights)
    
    # Multiply by lambda
    Nx.multiply(lambda, sum_abs)
  end

  @doc """
  Elastic Net - Combination of L1 and L2

  ## Formula

      Elastic Net = λ₁ * Σ|wᵢ| + λ₂ * Σwᵢ²

  Or equivalently:

      Elastic Net = λ * [α * Σ|wᵢ| + (1-α) * Σwᵢ²]

  Where:
  - λ = overall regularization strength
  - α = mixing parameter (0 to 1)
    - α = 0: Pure L2 (Ridge)
    - α = 1: Pure L1 (Lasso)
    - α = 0.5: Equal mix

  ## Characteristics

  - **Combines benefits of L1 and L2**
  - **Some feature selection** (from L1)
  - **Grouped selection** (from L2)
  - **More stable than pure L1**

  ## When to Use

  - When you want some feature selection but not too aggressive
  - When features are correlated (L1 alone picks one randomly)
  - When you want the best of both worlds

  ## Effect on Weights

  Pure L2: [2.1, -1.8, 1.5, -1.2, 2.5]  (all features kept)
  Pure L1: [3, 0, 0, 0, 4]               (aggressive selection)
  Elastic: [2.5, -1.2, 0, 0, 3.1]        (balanced)

  ## Why Elastic Net?

  L1 (Lasso) problems:
  - Unstable when features are correlated
  - Randomly picks one from correlated group

  L2 (Ridge) problems:
  - Doesn't do feature selection
  - Keeps all features (even irrelevant ones)

  Elastic Net solves both!

  ## Example

      weights = Nx.tensor([2.0, -3.0, 1.5, 0.5])
      lambda = 0.1
      alpha = 0.5  # Equal mix of L1 and L2
      
      penalty = MLNx.Regularization.elastic_net_penalty(
        weights, lambda, alpha
      )
  """
  def elastic_net_penalty(weights, lambda, alpha) do
    elastic_net_impl(weights, lambda, alpha)
  end

  defnp elastic_net_impl(weights, lambda, alpha) do
    # L1 component: α * Σ|w|
    l1_component = Nx.multiply(alpha, Nx.sum(Nx.abs(weights)))
    
    # L2 component: (1-α) * Σw²
    one_minus_alpha = Nx.subtract(1.0, alpha)
    l2_component = Nx.multiply(
      one_minus_alpha,
      Nx.sum(Nx.multiply(weights, weights))
    )
    
    # Combine and multiply by lambda
    total = Nx.add(l1_component, l2_component)
    Nx.multiply(lambda, total)
  end

  @doc """
  Compute regularized loss for linear regression.

  Adds regularization penalty to MSE loss.

  ## Formula

      Total Loss = MSE + Regularization Penalty

  ## Parameters

  - `predictions` - Model predictions
  - `targets` - True values
  - `weights` - Model weights (to penalize)
  - `opts` - Options:
    - `:reg_type` - :l1, :l2, or :elastic_net (default: :l2)
    - `:lambda` - Regularization strength (default: 0.01)
    - `:alpha` - Elastic net mixing (default: 0.5, only for elastic_net)

  ## Example

      # L2 regularization
      loss = MLNx.Regularization.regularized_loss(
        predictions, targets, weights,
        reg_type: :l2, lambda: 0.1
      )

      # L1 regularization
      loss = MLNx.Regularization.regularized_loss(
        predictions, targets, weights,
        reg_type: :l1, lambda: 0.05
      )

      # Elastic Net
      loss = MLNx.Regularization.regularized_loss(
        predictions, targets, weights,
        reg_type: :elastic_net, lambda: 0.1, alpha: 0.5
      )
  """
  def regularized_loss(predictions, targets, weights, opts \\ []) do
    reg_type = Keyword.get(opts, :reg_type, :l2)
    lambda = Keyword.get(opts, :lambda, 0.01)
    alpha = Keyword.get(opts, :alpha, 0.5)

    # Compute base MSE loss
    mse = compute_mse(predictions, targets)

    # Add regularization penalty
    penalty = case reg_type do
      :l1 -> l1_penalty(weights, lambda)
      :l2 -> l2_penalty(weights, lambda)
      :elastic_net -> elastic_net_impl(weights, lambda, alpha)
    end

    Nx.add(mse, penalty)
  end

  defnp compute_mse(predictions, targets) do
    errors = Nx.subtract(predictions, targets)
    squared_errors = Nx.multiply(errors, errors)
    Nx.mean(squared_errors)
  end

  @doc """
  Compare different regularization strengths.

  Useful for understanding the effect of lambda on the penalty.

  ## Returns

  Map with lambda values as keys and penalty values as values.

  ## Example

      weights = Nx.tensor([2.0, -3.0, 1.5])
      
      penalties = MLNx.Regularization.compare_lambdas(
        weights,
        reg_type: :l2,
        lambdas: [0.0, 0.01, 0.1, 1.0]
      )
      
      # %{
      #   0.0 => 0.0,
      #   0.01 => 0.1525,
      #   0.1 => 1.525,
      #   1.0 => 15.25
      # }
  """
  def compare_lambdas(weights, opts \\ []) do
    reg_type = Keyword.get(opts, :reg_type, :l2)
    lambdas = Keyword.get(opts, :lambdas, [0.0, 0.01, 0.1, 1.0])
    alpha = Keyword.get(opts, :alpha, 0.5)

    Enum.map(lambdas, fn lambda ->
      penalty = case reg_type do
        :l1 -> l1_penalty(weights, lambda)
        :l2 -> l2_penalty(weights, lambda)
        :elastic_net -> elastic_net_impl(weights, lambda, alpha)
      end
      
      {lambda, Nx.to_number(penalty)}
    end)
    |> Map.new()
  end
end
