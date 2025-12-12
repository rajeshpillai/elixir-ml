# Learning Rate Scheduling Demo
# Run with: mix run examples/06_lr_scheduler_demo.exs

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("LEARNING RATE SCHEDULING: Adaptive Optimization")
IO.puts(String.duplicate("=", 70) <> "\n")

IO.puts("""
CONCEPT: Learning rate schedules adapt the learning rate during training.

Why schedule learning rates?
  - Start with larger steps for fast initial progress
  - Reduce to smaller steps for fine-tuning
  - Improves convergence and final performance

Common strategies:
  1. Fixed: Constant (baseline)
  2. Step Decay: Reduce by factor every N epochs
  3. Exponential Decay: Smooth exponential reduction
  4. Cosine Annealing: Cosine-based smooth reduction
""")

# ============================================================================
# Example 1: Fixed Learning Rate (Baseline)
# ============================================================================

IO.puts(String.duplicate("-", 70))
IO.puts("Example 1: Fixed Learning Rate - The Baseline")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Fixed LR uses the same learning rate throughout training.

Simple but often suboptimal:
  - Too large: May overshoot minimum
  - Too small: Slow convergence
  - No adaptation to training progress
""")

schedule = MLNx.LRScheduler.generate_schedule(:fixed, 20, initial_lr: 0.1)

IO.puts("\nLearning rate over 20 epochs:")
IO.puts("Epoch | LR")
IO.puts(String.duplicate("-", 70))
Enum.with_index(schedule)
|> Enum.take(10)
|> Enum.each(fn {lr, epoch} ->
  IO.puts("#{String.pad_leading("#{epoch}", 5)} | #{Float.round(lr, 4)}")
end)
IO.puts("  ... | #{Float.round(List.last(schedule), 4)}")

# ============================================================================
# Example 2: Step Decay
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 2: Step Decay - Periodic Reduction")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Step decay reduces LR by a factor every N epochs.

Formula: lr = initial_lr * (decay_rate ^ floor(epoch / decay_steps))

Example: Reduce by 0.5 every 10 epochs
""")

schedule = MLNx.LRScheduler.generate_schedule(:step_decay, 30,
  initial_lr: 0.1, decay_rate: 0.5, decay_steps: 10)

IO.puts("\nLearning rate over 30 epochs (decay every 10):")
IO.puts("Epoch | LR      | Note")
IO.puts(String.duplicate("-", 70))
[0, 9, 10, 19, 20, 29]
|> Enum.each(fn epoch ->
  lr = Enum.at(schedule, epoch)
  note = cond do
    epoch == 0 -> "Initial"
    rem(epoch, 10) == 9 -> "Before decay"
    rem(epoch, 10) == 0 -> "After decay ↓"
    true -> ""
  end
  IO.puts("#{String.pad_leading("#{epoch}", 5)} | #{String.pad_leading("#{Float.round(lr, 4)}", 7)} | #{note}")
end)

IO.puts("""

OBSERVATION:
  - Epochs 0-9: LR = 0.1
  - Epochs 10-19: LR = 0.05 (reduced by 0.5)
  - Epochs 20-29: LR = 0.025 (reduced again)
  
  Sudden drops allow exploration then refinement!
""")

# ============================================================================
# Example 3: Exponential Decay
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 3: Exponential Decay - Smooth Reduction")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Exponential decay smoothly reduces LR over time.

Formula: lr = initial_lr * (decay_rate ^ epoch)

Smoother than step decay, continuous adaptation.
""")

schedule = MLNx.LRScheduler.generate_schedule(:exponential, 30,
  initial_lr: 0.1, decay_rate: 0.95)

IO.puts("\nLearning rate over 30 epochs (decay_rate=0.95):")
IO.puts("Epoch | LR")
IO.puts(String.duplicate("-", 70))
[0, 5, 10, 15, 20, 25, 29]
|> Enum.each(fn epoch ->
  lr = Enum.at(schedule, epoch)
  IO.puts("#{String.pad_leading("#{epoch}", 5)} | #{Float.round(lr, 4)}")
end)

IO.puts("""

OBSERVATION:
  - Smooth, continuous decay
  - No sudden jumps
  - Gradual refinement
""")

# ============================================================================
# Example 4: Cosine Annealing
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 4: Cosine Annealing - Smooth Curve")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Cosine annealing follows a cosine curve from initial to min LR.

Formula: lr = min_lr + (initial_lr - min_lr) * (1 + cos(π*t/T)) / 2

Smooth, predictable decay with nice mathematical properties.
""")

schedule = MLNx.LRScheduler.generate_schedule(:cosine, 30,
  initial_lr: 0.1, min_lr: 0.001, total_epochs: 30)

IO.puts("\nLearning rate over 30 epochs:")
IO.puts("Epoch | LR")
IO.puts(String.duplicate("-", 70))
[0, 5, 10, 15, 20, 25, 29]
|> Enum.each(fn epoch ->
  lr = Enum.at(schedule, epoch)
  IO.puts("#{String.pad_leading("#{epoch}", 5)} | #{Float.round(lr, 4)}")
end)

IO.puts("""

OBSERVATION:
  - Smooth cosine curve
  - Fast initial decay
  - Slower as it approaches min_lr
  - Popular in deep learning!
""")

# ============================================================================
# Example 5: Comparison of All Strategies
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 5: Comparing All Strategies")
IO.puts(String.duplicate("-", 70))

IO.puts("\nLearning rate comparison over 30 epochs:\n")

fixed = MLNx.LRScheduler.generate_schedule(:fixed, 30, initial_lr: 0.1)
step = MLNx.LRScheduler.generate_schedule(:step_decay, 30, 
  initial_lr: 0.1, decay_rate: 0.5, decay_steps: 10)
exp = MLNx.LRScheduler.generate_schedule(:exponential, 30,
  initial_lr: 0.1, decay_rate: 0.95)
cos = MLNx.LRScheduler.generate_schedule(:cosine, 30,
  initial_lr: 0.1, min_lr: 0.001, total_epochs: 30)

IO.puts("Epoch | Fixed  | Step   | Exp    | Cosine")
IO.puts(String.duplicate("-", 70))
[0, 10, 20, 29]
|> Enum.each(fn epoch ->
  IO.puts("#{String.pad_leading("#{epoch}", 5)} | #{String.pad_leading("#{Float.round(Enum.at(fixed, epoch), 4)}", 6)} | #{String.pad_leading("#{Float.round(Enum.at(step, epoch), 4)}", 6)} | #{String.pad_leading("#{Float.round(Enum.at(exp, epoch), 4)}", 6)} | #{String.pad_leading("#{Float.round(Enum.at(cos, epoch), 4)}", 6)}")
end)

IO.puts("""

KEY DIFFERENCES:
  - Fixed: Constant (no adaptation)
  - Step: Sudden drops every 10 epochs
  - Exponential: Smooth continuous decay
  - Cosine: Smooth with fast initial decay

WHEN TO USE:
  - Fixed: Simple problems, quick experiments
  - Step: When you know good decay points
  - Exponential: General purpose, smooth decay
  - Cosine: Deep learning, smooth convergence
""")

# ============================================================================
# Summary
# ============================================================================

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("KEY TAKEAWAYS")
IO.puts(String.duplicate("=", 70))

IO.puts("""
1. WHY SCHEDULE LEARNING RATES?
   - Start fast: Large LR for quick initial progress
   - End slow: Small LR for fine-tuning
   - Better final performance than fixed LR

2. FIXED LEARNING RATE
   - Simplest baseline
   - No adaptation
   - Often suboptimal

3. STEP DECAY
   - Reduce by factor every N epochs
   - Sudden drops
   - Good when you know decay schedule

4. EXPONENTIAL DECAY
   - Smooth continuous reduction
   - Formula: lr * (decay_rate ^ epoch)
   - General purpose

5. COSINE ANNEALING
   - Smooth cosine curve
   - Fast initial decay, slower later
   - Popular in deep learning

6. PRACTICAL TIPS
   - Start with cosine or exponential
   - Tune decay_rate (0.9-0.99 typical)
   - Monitor validation loss to adjust
   - Can combine with other techniques

NEXT STEPS:
  - Learn about adaptive optimizers (Adam, RMSprop)
  - Explore learning rate warmup
  - Understand momentum and acceleration
""")

IO.puts(String.duplicate("=", 70))
IO.puts("Demo complete! 🎓")
IO.puts(String.duplicate("=", 70) <> "\n")
