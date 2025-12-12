defmodule MLNx.BatchTraining do
  @moduledoc """
  Batch training strategies for gradient descent optimization.

  Provides three variants of gradient descent:
  - **Batch Gradient Descent**: Uses entire dataset per update
  - **Stochastic Gradient Descent (SGD)**: Uses one example per update
  - **Mini-Batch Gradient Descent**: Uses small batches per update

  ## Comparison

  | Method | Batch Size | Speed | Convergence | Memory |
  |--------|-----------|-------|-------------|--------|
  | Batch | Full dataset | Slow | Smooth | High |
  | Stochastic | 1 | Fast | Noisy | Low |
  | Mini-Batch | 32-256 | Medium | Balanced | Medium |

  ## When to Use

  - **Batch GD**: Small datasets (<10k examples), need smooth convergence
  - **Stochastic GD**: Very large datasets, online learning
  - **Mini-Batch GD**: Most cases (best balance), standard for deep learning

  ## Examples

      # Generate sample data
      x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      y = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
      
      # Batch gradient descent
      {weights, bias, history} = MLNx.BatchTraining.batch_gradient_descent(
        x, y, 
        learning_rate: 0.01,
        iterations: 100
      )
      
      # Stochastic gradient descent
      {weights, bias, history} = MLNx.BatchTraining.stochastic_gradient_descent(
        x, y,
        learning_rate: 0.01,
        epochs: 10
      )
      
      # Mini-batch gradient descent
      {weights, bias, history} = MLNx.BatchTraining.mini_batch_gradient_descent(
        x, y,
        batch_size: 2,
        learning_rate: 0.01,
        epochs: 10
      )
  """

  @doc """
  Batch Gradient Descent - uses entire dataset for each update.

  Computes gradients using all training examples, then updates weights once.
  Most accurate gradient estimate but slow for large datasets.

  ## Options

  - `:learning_rate` - Step size for updates (default: 0.01)
  - `:iterations` - Number of iterations (default: 100)
  - `:track_history` - Whether to track loss history (default: true)

  ## Returns

  `{weights, bias, history}` where history is a list of losses per iteration.

  ## Examples

      iex> x = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> y = Nx.tensor([[2.0], [4.0], [6.0]])
      iex> {w, b, history} = MLNx.BatchTraining.batch_gradient_descent(x, y, iterations: 50)
      iex> length(history)
      50
      iex> Nx.shape(w)
      {1, 1}
  """
  def batch_gradient_descent(features, targets, opts \\ []) do
    learning_rate = Keyword.get(opts, :learning_rate, 0.01)
    iterations = Keyword.get(opts, :iterations, 100)
    track_history = Keyword.get(opts, :track_history, true)
    
    # Initialize weights and bias
    {n_samples, n_features} = Nx.shape(features)
    {_, n_outputs} = Nx.shape(targets)
    
    weights = Nx.broadcast(0.0, {n_features, n_outputs})
    bias = Nx.broadcast(0.0, {1, n_outputs})
    
    # Training loop
    {final_weights, final_bias, history} = 
      Enum.reduce(1..iterations, {weights, bias, []}, fn _iter, {w, b, hist} ->
        # Forward pass: predictions = X * w + b
        predictions = Nx.add(Nx.dot(features, w), b)
        
        # Compute loss (MSE)
        loss = if track_history do
          MLNx.LossFunctions.mse(predictions, targets) |> Nx.to_number()
        else
          0.0
        end
        
        # Compute gradients using ALL examples
        error = Nx.subtract(predictions, targets)
        
        # Gradient for weights: (1/n) * X^T * error
        grad_w = Nx.divide(Nx.dot(Nx.transpose(features), error), n_samples)
        
        # Gradient for bias: (1/n) * sum(error)
        grad_b = Nx.divide(Nx.sum(error, axes: [0], keep_axes: true), n_samples)
        
        # Update weights and bias
        new_w = Nx.subtract(w, Nx.multiply(learning_rate, grad_w))
        new_b = Nx.subtract(b, Nx.multiply(learning_rate, grad_b))
        
        new_hist = if track_history, do: hist ++ [loss], else: hist
        
        {new_w, new_b, new_hist}
      end)
    
    {final_weights, final_bias, history}
  end

  @doc """
  Stochastic Gradient Descent (SGD) - uses one example per update.

  Randomly selects one training example, computes gradient, and updates weights.
  Fast updates but noisy convergence. Good for very large datasets.

  ## Options

  - `:learning_rate` - Step size for updates (default: 0.01)
  - `:epochs` - Number of passes through dataset (default: 10)
  - `:track_history` - Whether to track loss history (default: true)
  - `:shuffle` - Whether to shuffle data each epoch (default: true)

  ## Returns

  `{weights, bias, history}` where history tracks loss after each epoch.

  ## Examples

      iex> x = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> y = Nx.tensor([[2.0], [4.0], [6.0]])
      iex> {w, b, history} = MLNx.BatchTraining.stochastic_gradient_descent(x, y, epochs: 5)
      iex> length(history)
      5
  """
  def stochastic_gradient_descent(features, targets, opts \\ []) do
    learning_rate = Keyword.get(opts, :learning_rate, 0.01)
    epochs = Keyword.get(opts, :epochs, 10)
    track_history = Keyword.get(opts, :track_history, true)
    shuffle = Keyword.get(opts, :shuffle, true)
    
    # Initialize weights and bias
    {n_samples, n_features} = Nx.shape(features)
    {_, n_outputs} = Nx.shape(targets)
    
    weights = Nx.broadcast(0.0, {n_features, n_outputs})
    bias = Nx.broadcast(0.0, {1, n_outputs})
    
    # Training loop - multiple epochs
    {final_weights, final_bias, history} = 
      Enum.reduce(1..epochs, {weights, bias, []}, fn _epoch, {w, b, hist} ->
        # Shuffle data if requested
        {epoch_features, epoch_targets} = if shuffle do
          shuffle_data(features, targets)
        else
          {features, targets}
        end
        
        # Update weights for each example
        {epoch_w, epoch_b} = 
          Enum.reduce(0..(n_samples - 1), {w, b}, fn i, {curr_w, curr_b} ->
            # Get single example
            x_i = epoch_features[[i, 0..-1//1]] |> Nx.reshape({1, n_features})
            y_i = epoch_targets[[i, 0..-1//1]] |> Nx.reshape({1, n_outputs})
            
            # Forward pass
            pred = Nx.add(Nx.dot(x_i, curr_w), curr_b)
            
            # Compute error
            error = Nx.subtract(pred, y_i)
            
            # Compute gradients (no averaging since single example)
            grad_w = Nx.dot(Nx.transpose(x_i), error)
            grad_b = error
            
            # Update weights
            new_w = Nx.subtract(curr_w, Nx.multiply(learning_rate, grad_w))
            new_b = Nx.subtract(curr_b, Nx.multiply(learning_rate, grad_b))
            
            {new_w, new_b}
          end)
        
        # Compute loss for this epoch
        loss = if track_history do
          predictions = Nx.add(Nx.dot(epoch_features, epoch_w), epoch_b)
          MLNx.LossFunctions.mse(predictions, epoch_targets) |> Nx.to_number()
        else
          0.0
        end
        
        new_hist = if track_history, do: hist ++ [loss], else: hist
        
        {epoch_w, epoch_b, new_hist}
      end)
    
    {final_weights, final_bias, history}
  end

  @doc """
  Mini-Batch Gradient Descent - uses small batches per update.

  Divides dataset into small batches, computes gradients per batch, and updates.
  Best balance between accuracy and speed. Standard for deep learning.

  ## Options

  - `:batch_size` - Number of examples per batch (default: 32)
  - `:learning_rate` - Step size for updates (default: 0.01)
  - `:epochs` - Number of passes through dataset (default: 10)
  - `:track_history` - Whether to track loss history (default: true)
  - `:shuffle` - Whether to shuffle data each epoch (default: true)

  ## Returns

  `{weights, bias, history}` where history tracks loss after each epoch.

  ## Examples

      iex> x = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])
      iex> y = Nx.tensor([[2.0], [4.0], [6.0], [8.0]])
      iex> {w, b, history} = MLNx.BatchTraining.mini_batch_gradient_descent(x, y, batch_size: 2, epochs: 5)
      iex> length(history)
      5
  """
  def mini_batch_gradient_descent(features, targets, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    learning_rate = Keyword.get(opts, :learning_rate, 0.01)
    epochs = Keyword.get(opts, :epochs, 10)
    track_history = Keyword.get(opts, :track_history, true)
    shuffle = Keyword.get(opts, :shuffle, true)
    
    # Initialize weights and bias
    {n_samples, n_features} = Nx.shape(features)
    {_, n_outputs} = Nx.shape(targets)
    
    weights = Nx.broadcast(0.0, {n_features, n_outputs})
    bias = Nx.broadcast(0.0, {1, n_outputs})
    
    # Training loop - multiple epochs
    {final_weights, final_bias, history} = 
      Enum.reduce(1..epochs, {weights, bias, []}, fn _epoch, {w, b, hist} ->
        # Shuffle data if requested
        {epoch_features, epoch_targets} = if shuffle do
          shuffle_data(features, targets)
        else
          {features, targets}
        end
        
        # Create batches
        batches = create_batches(epoch_features, epoch_targets, batch_size)
        
        # Update weights for each batch
        {epoch_w, epoch_b} = 
          Enum.reduce(batches, {w, b}, fn {batch_x, batch_y}, {curr_w, curr_b} ->
            batch_n = Nx.axis_size(batch_x, 0)
            
            # Forward pass
            pred = Nx.add(Nx.dot(batch_x, curr_w), curr_b)
            
            # Compute error
            error = Nx.subtract(pred, batch_y)
            
            # Compute gradients (average over batch)
            grad_w = Nx.divide(Nx.dot(Nx.transpose(batch_x), error), batch_n)
            grad_b = Nx.divide(Nx.sum(error, axes: [0], keep_axes: true), batch_n)
            
            # Update weights
            new_w = Nx.subtract(curr_w, Nx.multiply(learning_rate, grad_w))
            new_b = Nx.subtract(curr_b, Nx.multiply(learning_rate, grad_b))
            
            {new_w, new_b}
          end)
        
        # Compute loss for this epoch
        loss = if track_history do
          predictions = Nx.add(Nx.dot(epoch_features, epoch_w), epoch_b)
          MLNx.LossFunctions.mse(predictions, epoch_targets) |> Nx.to_number()
        else
          0.0
        end
        
        new_hist = if track_history, do: hist ++ [loss], else: hist
        
        {epoch_w, epoch_b, new_hist}
      end)
    
    {final_weights, final_bias, history}
  end

  @doc """
  Shuffles features and targets together, preserving correspondence.

  ## Examples

      iex> x = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> y = Nx.tensor([[10.0], [20.0], [30.0]])
      iex> {shuffled_x, shuffled_y} = MLNx.BatchTraining.shuffle_data(x, y)
      iex> Nx.shape(shuffled_x)
      {3, 1}
  """
  def shuffle_data(features, targets) do
    n_samples = Nx.axis_size(features, 0)
    
    # Create random permutation
    indices = Enum.shuffle(0..(n_samples - 1))
    
    # Reorder both features and targets
    shuffled_features = Nx.stack(Enum.map(indices, fn i -> features[i] end))
    shuffled_targets = Nx.stack(Enum.map(indices, fn i -> targets[i] end))
    
    {shuffled_features, shuffled_targets}
  end

  @doc """
  Creates mini-batches from features and targets.

  Divides the dataset into batches of specified size. Last batch may be smaller.

  ## Examples

      iex> x = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      iex> y = Nx.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])
      iex> batches = MLNx.BatchTraining.create_batches(x, y, 2)
      iex> length(batches)
      3
  """
  def create_batches(features, targets, batch_size) do
    n_samples = Nx.axis_size(features, 0)
    
    # Calculate number of batches
    n_batches = ceil(n_samples / batch_size)
    
    # Create batches
    Enum.map(0..(n_batches - 1), fn i ->
      start_idx = i * batch_size
      end_idx = min(start_idx + batch_size - 1, n_samples - 1)
      
      batch_features = features[start_idx..end_idx]
      batch_targets = targets[start_idx..end_idx]
      
      {batch_features, batch_targets}
    end)
  end

  @doc """
  Generic training function with method selection.

  ## Options

  - `:method` - `:batch`, `:stochastic`, or `:mini_batch` (default: `:mini_batch`)
  - `:batch_size` - For mini-batch method (default: 32)
  - `:learning_rate` - Step size (default: 0.01)
  - `:epochs` or `:iterations` - Training duration
  - Other method-specific options

  ## Examples

      iex> x = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> y = Nx.tensor([[2.0], [4.0], [6.0]])
      iex> {w, b, _} = MLNx.BatchTraining.train(x, y, method: :batch, iterations: 10)
      iex> Nx.shape(w)
      {1, 1}
  """
  def train(features, targets, opts \\ []) do
    method = Keyword.get(opts, :method, :mini_batch)
    
    case method do
      :batch ->
        batch_gradient_descent(features, targets, opts)
      
      :stochastic ->
        stochastic_gradient_descent(features, targets, opts)
      
      :mini_batch ->
        mini_batch_gradient_descent(features, targets, opts)
    end
  end
end
