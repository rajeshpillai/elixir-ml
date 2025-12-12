defmodule MLNx.LRScheduler do
  @moduledoc """
  Learning rate scheduling strategies for optimization.

  Learning rate schedules adapt the learning rate during training:
  - Start with larger steps for fast initial progress
  - Reduce to smaller steps for fine-tuning
  - Improves convergence and final performance

  ## Strategies

  - **Fixed**: Constant learning rate (baseline)
  - **Step Decay**: Reduce by factor every N epochs
  - **Exponential Decay**: Smooth exponential reduction
  - **Cosine Annealing**: Cosine-based smooth reduction

  ## Examples

      # Fixed learning rate
      lr = MLNx.LRScheduler.get_lr(:fixed, epoch, initial_lr: 0.1)
      
      # Step decay: reduce by 0.5 every 10 epochs
      lr = MLNx.LRScheduler.get_lr(:step_decay, epoch, 
        initial_lr: 0.1, decay_rate: 0.5, decay_steps: 10)
      
      # Exponential decay
      lr = MLNx.LRScheduler.get_lr(:exponential, epoch,
        initial_lr: 0.1, decay_rate: 0.95)
      
      # Cosine annealing
      lr = MLNx.LRScheduler.get_lr(:cosine, epoch,
        initial_lr: 0.1, min_lr: 0.001, total_epochs: 100)
  """

  @doc """
  Get learning rate for current epoch based on schedule.

  ## Options

  - `:initial_lr` - Starting learning rate (required)
  - `:decay_rate` - Decay factor (for step/exponential)
  - `:decay_steps` - Steps between decays (for step decay)
  - `:min_lr` - Minimum learning rate (for cosine)
  - `:total_epochs` - Total training epochs (for cosine)

  ## Examples

      iex> MLNx.LRScheduler.get_lr(:fixed, 10, initial_lr: 0.1)
      0.1

      iex> lr = MLNx.LRScheduler.get_lr(:step_decay, 15, initial_lr: 0.1, decay_rate: 0.5, decay_steps: 10)
      iex> Float.round(lr, 3)
      0.05
  """
  def get_lr(schedule, epoch, opts \\ []) do
    initial_lr = Keyword.fetch!(opts, :initial_lr)
    
    case schedule do
      :fixed ->
        initial_lr
      
      :step_decay ->
        decay_rate = Keyword.get(opts, :decay_rate, 0.1)
        decay_steps = Keyword.get(opts, :decay_steps, 10)
        step_decay(initial_lr, epoch, decay_rate, decay_steps)
      
      :exponential ->
        decay_rate = Keyword.get(opts, :decay_rate, 0.95)
        exponential_decay(initial_lr, epoch, decay_rate)
      
      :cosine ->
        min_lr = Keyword.get(opts, :min_lr, 0.0)
        total_epochs = Keyword.get(opts, :total_epochs, 100)
        cosine_annealing(initial_lr, epoch, min_lr, total_epochs)
    end
  end

  @doc """
  Step decay: reduce learning rate by factor every N steps.

  Formula: `lr = initial_lr * (decay_rate ^ floor(epoch / decay_steps))`

  ## Examples

      iex> MLNx.LRScheduler.step_decay(0.1, 0, 0.5, 10)
      0.1
      
      iex> MLNx.LRScheduler.step_decay(0.1, 10, 0.5, 10)
      0.05
      
      iex> MLNx.LRScheduler.step_decay(0.1, 20, 0.5, 10)
      0.025
  """
  def step_decay(initial_lr, epoch, decay_rate, decay_steps) do
    num_decays = div(epoch, decay_steps)
    initial_lr * :math.pow(decay_rate, num_decays)
  end

  @doc """
  Exponential decay: smooth exponential reduction.

  Formula: `lr = initial_lr * (decay_rate ^ epoch)`

  ## Examples

      iex> MLNx.LRScheduler.exponential_decay(0.1, 0, 0.95)
      0.1
      
      iex> lr = MLNx.LRScheduler.exponential_decay(0.1, 10, 0.95)
      iex> Float.round(lr, 4)
      0.0599
  """
  def exponential_decay(initial_lr, epoch, decay_rate) do
    initial_lr * :math.pow(decay_rate, epoch)
  end

  @doc """
  Cosine annealing: smooth cosine-based reduction.

  Formula: `lr = min_lr + (initial_lr - min_lr) * (1 + cos(π * epoch / total_epochs)) / 2`

  Decreases from initial_lr to min_lr following a cosine curve.

  ## Examples

      iex> MLNx.LRScheduler.cosine_annealing(0.1, 0, 0.0, 100)
      0.1
      
      iex> lr = MLNx.LRScheduler.cosine_annealing(0.1, 100, 0.0, 100)
      iex> Float.round(lr, 6)
      0.0
  """
  def cosine_annealing(initial_lr, epoch, min_lr, total_epochs) do
    # Ensure we don't divide by zero
    if total_epochs == 0 do
      initial_lr
    else
      progress = min(epoch / total_epochs, 1.0)
      # Cosine decreases from 1 to -1, so (1 + cos) goes from 2 to 0
      cosine_factor = (1 + :math.cos(:math.pi() * progress)) / 2
      min_lr + (initial_lr - min_lr) * cosine_factor
    end
  end

  @doc """
  Generate learning rate schedule for all epochs.

  Returns a list of learning rates for each epoch.

  ## Examples

      iex> schedule = MLNx.LRScheduler.generate_schedule(:fixed, 5, initial_lr: 0.1)
      iex> Enum.all?(schedule, &(&1 == 0.1))
      true
      
      iex> schedule = MLNx.LRScheduler.generate_schedule(:step_decay, 3, initial_lr: 0.1, decay_rate: 0.5, decay_steps: 2)
      iex> Enum.map(schedule, &Float.round(&1, 2))
      [0.1, 0.1, 0.05]
  """
  def generate_schedule(schedule_type, num_epochs, opts) do
    Enum.map(0..(num_epochs - 1), fn epoch ->
      get_lr(schedule_type, epoch, opts)
    end)
  end
end
