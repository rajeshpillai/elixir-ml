defmodule MLNx.Normalization do
  @moduledoc """
  Feature normalization techniques for machine learning.

  Normalization scales features to similar ranges, which improves:
  - Gradient descent convergence speed
  - Model training stability
  - Feature importance comparison

  ## Techniques

  ### Min-Max Scaling
  Scales features to a specific range (default [0, 1]):

      x_scaled = (x - min) / (max - min)

  Use when:
  - You need bounded output (e.g., [0, 1] or [-1, 1])
  - Features have known bounds
  - Preserving zero is important

  ### Standardization (Z-score)
  Centers features around mean=0 with std=1:

      x_standardized = (x - mean) / std

  Use when:
  - Features follow normal distribution
  - You want to preserve outliers
  - No specific range requirement

  ## Examples

      # Min-max scaling to [0, 1]
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      scaled = MLNx.Normalization.min_max_scale(data)
      # => [0.0, 0.25, 0.5, 0.75, 1.0]

      # Standardization (z-score)
      standardized = MLNx.Normalization.standardize(data)
      # => [-1.414, -0.707, 0.0, 0.707, 1.414]

      # Custom range scaling
      scaled = MLNx.Normalization.min_max_scale(data, -1, 1)
      # => [-1.0, -0.5, 0.0, 0.5, 1.0]
  """

  @doc """
  Scales features to [0, 1] range using min-max normalization.

  Formula: `(x - min) / (max - min)`

  Returns a tuple `{scaled_data, {min, max}}` where the second element
  contains the statistics needed for inverse transformation.

  ## Examples

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {scaled, {min, max}} = MLNx.Normalization.min_max_scale(data)
      iex> Nx.to_flat_list(scaled)
      [0.0, 0.25, 0.5, 0.75, 1.0]
      iex> {Nx.to_number(min), Nx.to_number(max)}
      {1.0, 5.0}

      iex> data = Nx.tensor([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
      iex> {scaled, _stats} = MLNx.Normalization.min_max_scale(data)
      iex> Nx.shape(scaled)
      {3, 2}
  """
  def min_max_scale(data) do
    min_max_scale(data, 0.0, 1.0)
  end

  @doc """
  Scales features to a custom [new_min, new_max] range.

  Formula: `new_min + (x - min) * (new_max - new_min) / (max - min)`

  Returns a tuple `{scaled_data, {min, max}}` where the second element
  contains the original statistics needed for inverse transformation.

  ## Examples

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {scaled, _} = MLNx.Normalization.min_max_scale(data, -1, 1)
      iex> Nx.to_flat_list(scaled)
      [-1.0, -0.5, 0.0, 0.5, 1.0]

      iex> data = Nx.tensor([0.0, 50.0, 100.0])
      iex> {scaled, _} = MLNx.Normalization.min_max_scale(data, 0, 10)
      iex> Nx.to_flat_list(scaled)
      [0.0, 5.0, 10.0]
  """
  def min_max_scale(data, new_min, new_max) do
    # Compute along axis 0 for 2D tensors (per-feature normalization)
    axis = if Nx.rank(data) == 2, do: [axes: [0]], else: []
    
    min_val = Nx.reduce_min(data, axis)
    max_val = Nx.reduce_max(data, axis)
    
    # Handle edge case: if min == max, return all zeros (or new_min)
    range = Nx.subtract(max_val, min_val)
    
    # Avoid division by zero
    safe_range = Nx.select(Nx.equal(range, 0), 1.0, range)
    
    # Scale to [0, 1] first
    normalized = Nx.divide(Nx.subtract(data, min_val), safe_range)
    
    # Then scale to [new_min, new_max]
    new_range = new_max - new_min
    scaled = Nx.add(Nx.multiply(normalized, new_range), new_min)
    
    {scaled, {min_val, max_val}}
  end

  @doc """
  Reverses min-max scaling to original scale.

  Given scaled data and the original statistics, reconstructs the original values.

  Formula: `x_original = x_scaled * (max - min) + min`

  ## Examples

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {scaled, stats} = MLNx.Normalization.min_max_scale(data)
      iex> original = MLNx.Normalization.inverse_min_max_scale(scaled, stats)
      iex> Nx.all_close(data, original) |> Nx.to_number()
      1

      iex> data = Nx.tensor([10.0, 20.0, 30.0])
      iex> {scaled, stats} = MLNx.Normalization.min_max_scale(data, -1, 1)
      iex> original = MLNx.Normalization.inverse_min_max_scale(scaled, stats, -1, 1)
      iex> Nx.all_close(data, original) |> Nx.to_number()
      1
  """
  def inverse_min_max_scale(scaled_data, {min_val, max_val}, new_min \\ 0.0, new_max \\ 1.0) do
    # Reverse the scaling from [new_min, new_max] to [0, 1]
    new_range = new_max - new_min
    normalized = Nx.divide(Nx.subtract(scaled_data, new_min), new_range)
    
    # Then scale from [0, 1] to original range
    range = Nx.subtract(max_val, min_val)
    Nx.add(Nx.multiply(normalized, range), min_val)
  end

  @doc """
  Standardizes features to have mean=0 and std=1 (z-score normalization).

  Formula: `(x - mean) / std`

  Returns a tuple `{standardized_data, {mean, std}}` where the second element
  contains the statistics needed for inverse transformation.

  ## Examples

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {standardized, {mean, std}} = MLNx.Normalization.standardize(data)
      iex> Nx.to_number(mean)
      3.0
      iex> Float.round(Nx.to_number(std), 2)
      1.41

      iex> data = Nx.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
      iex> {standardized, _} = MLNx.Normalization.standardize(data)
      iex> Nx.shape(standardized)
      {3, 2}
  """
  def standardize(data) do
    # Compute along axis 0 for 2D tensors (per-feature normalization)
    axis = if Nx.rank(data) == 2, do: [axes: [0]], else: []
    
    mean = Nx.mean(data, axis)
    std = Nx.standard_deviation(data, axis)
    
    # Avoid division by zero
    safe_std = Nx.select(Nx.equal(std, 0), 1.0, std)
    
    standardized = Nx.divide(Nx.subtract(data, mean), safe_std)
    
    {standardized, {mean, std}}
  end

  @doc """
  Reverses standardization to original scale.

  Given standardized data and the original statistics, reconstructs the original values.

  Formula: `x_original = x_standardized * std + mean`

  ## Examples

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {standardized, stats} = MLNx.Normalization.standardize(data)
      iex> original = MLNx.Normalization.inverse_standardize(standardized, stats)
      iex> Nx.all_close(data, original) |> Nx.to_number()
      1

      iex> data = Nx.tensor([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
      iex> {standardized, stats} = MLNx.Normalization.standardize(data)
      iex> original = MLNx.Normalization.inverse_standardize(standardized, stats)
      iex> Nx.all_close(data, original) |> Nx.to_number()
      1
  """
  def inverse_standardize(standardized_data, {mean, std}) do
    Nx.add(Nx.multiply(standardized_data, std), mean)
  end

  @doc """
  Computes mean and standard deviation for features.

  Useful for understanding data distribution before normalization.

  ## Examples

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {mean, std} = MLNx.Normalization.compute_stats(data)
      iex> Nx.to_number(mean)
      3.0
      iex> Float.round(Nx.to_number(std), 2)
      1.41

      iex> data = Nx.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
      iex> {mean, std} = MLNx.Normalization.compute_stats(data)
      iex> Nx.to_flat_list(mean)
      [2.0, 20.0]
  """
  def compute_stats(data) do
    axis = if Nx.rank(data) == 2, do: [axes: [0]], else: []
    
    mean = Nx.mean(data, axis)
    std = Nx.standard_deviation(data, axis)
    
    {mean, std}
  end

  @doc """
  Generic normalization function with options.

  Supports both min-max scaling and standardization.

  ## Options

  - `:method` - `:min_max` (default) or `:standardize`
  - `:range` - For min-max: `{min, max}` tuple (default: `{0, 1}`)

  ## Examples

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {scaled, _} = MLNx.Normalization.normalize(data, method: :min_max)
      iex> Nx.to_flat_list(scaled)
      [0.0, 0.25, 0.5, 0.75, 1.0]

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {scaled, _} = MLNx.Normalization.normalize(data, method: :min_max, range: {-1, 1})
      iex> Nx.to_flat_list(scaled)
      [-1.0, -0.5, 0.0, 0.5, 1.0]

      iex> data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {standardized, _} = MLNx.Normalization.normalize(data, method: :standardize)
      iex> Nx.shape(standardized)
      {5}
  """
  def normalize(data, opts \\ []) do
    method = Keyword.get(opts, :method, :min_max)
    
    case method do
      :min_max ->
        {min_val, max_val} = Keyword.get(opts, :range, {0.0, 1.0})
        min_max_scale(data, min_val, max_val)
      
      :standardize ->
        standardize(data)
    end
  end
end
