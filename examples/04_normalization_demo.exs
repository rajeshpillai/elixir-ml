# Feature Normalization Demo
# Run with: mix run examples/04_normalization_demo.exs

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("FEATURE NORMALIZATION: Scaling for Better Learning")
IO.puts(String.duplicate("=", 70) <> "\n")

IO.puts("""
CONCEPT: Feature normalization scales features to similar ranges.
This improves gradient descent convergence and model performance.

Why normalize?
  - Features on different scales cause uneven gradient updates
  - Large-scale features dominate the learning process
  - Normalization ensures all features contribute equally
""")

# ============================================================================
# Example 1: Understanding Min-Max Scaling
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 1: Min-Max Scaling - Scaling to [0, 1]")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Min-Max Scaling transforms features to a fixed range [0, 1]:

  x_scaled = (x - min) / (max - min)

Effect: Smallest value → 0, Largest value → 1
""")

data = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

IO.puts("\nOriginal data: #{inspect(Nx.to_list(data))}")

{scaled, {min_val, max_val}} = MLNx.Normalization.min_max_scale(data)

IO.puts("Min value: #{Nx.to_number(min_val)}")
IO.puts("Max value: #{Nx.to_number(max_val)}")
IO.puts("Scaled data: #{inspect(Nx.to_list(scaled))}")

IO.puts("""

OBSERVATION:
  - Original range: [10, 50]
  - Scaled range: [0, 1]
  - 10 (min) → 0.0
  - 50 (max) → 1.0
  - 30 (middle) → 0.5
  
FORMULA CHECK:
  For x = 30: (30 - 10) / (50 - 10) = 20/40 = 0.5 ✓
""")

# ============================================================================
# Example 2: Custom Range Scaling
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 2: Custom Range Scaling - Scaling to [-1, 1]")
IO.puts(String.duplicate("-", 70))

IO.puts("""
You can scale to any range, not just [0, 1].
Common choice: [-1, 1] for neural networks.

Formula: new_min + (x - min) * (new_max - new_min) / (max - min)
""")

data = Nx.tensor([0.0, 25.0, 50.0, 75.0, 100.0])

IO.puts("\nOriginal data: #{inspect(Nx.to_list(data))}")

{scaled_01, _} = MLNx.Normalization.min_max_scale(data, 0, 1)
{scaled_neg1_1, _} = MLNx.Normalization.min_max_scale(data, -1, 1)
{scaled_0_10, _} = MLNx.Normalization.min_max_scale(data, 0, 10)

IO.puts("\nScaling to different ranges:")
IO.puts(String.duplicate("-", 70))
IO.puts("Range [0, 1]:   #{inspect(Nx.to_list(scaled_01))}")
IO.puts("Range [-1, 1]:  #{inspect(Nx.to_list(scaled_neg1_1))}")
IO.puts("Range [0, 10]:  #{inspect(Nx.to_list(scaled_0_10))}")

IO.puts("""

OBSERVATION:
  - Same data, different ranges
  - Relative spacing preserved
  - Choose range based on your algorithm's needs
""")

# ============================================================================
# Example 3: Understanding Standardization (Z-score)
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 3: Standardization - Z-score Normalization")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Standardization transforms data to have mean=0 and std=1:

  x_standardized = (x - mean) / std

Effect: Centers data around 0 with unit variance
""")

data = Nx.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

IO.puts("\nOriginal data: #{inspect(Nx.to_list(data))}")

{standardized, {mean, std}} = MLNx.Normalization.standardize(data)

IO.puts("Mean: #{Float.round(Nx.to_number(mean), 2)}")
IO.puts("Std:  #{Float.round(Nx.to_number(std), 2)}")
IO.puts("Standardized: #{inspect(Enum.map(Nx.to_list(standardized), &Float.round(&1, 3)))}")

# Verify properties
new_mean = Nx.mean(standardized) |> Nx.to_number() |> Float.round(6)
new_std = Nx.standard_deviation(standardized) |> Nx.to_number() |> Float.round(6)

IO.puts("\nVerification:")
IO.puts("  New mean: #{new_mean} (should be ≈ 0)")
IO.puts("  New std:  #{new_std} (should be ≈ 1)")

IO.puts("""

OBSERVATION:
  - Original mean: 30.0 → New mean: 0.0
  - Original std: 14.14 → New std: 1.0
  - Values below mean are negative
  - Values above mean are positive
  - Tells you how many std deviations from mean!
""")

# ============================================================================
# Example 4: Min-Max vs Standardization
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 4: Comparing Min-Max vs Standardization")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Let's compare both techniques on the same data:
""")

data = Nx.tensor([5.0, 10.0, 15.0, 20.0, 100.0])  # Note the outlier!

IO.puts("\nOriginal data: #{inspect(Nx.to_list(data))} (note the outlier: 100)")

{min_max_scaled, _} = MLNx.Normalization.min_max_scale(data)
{standardized, _} = MLNx.Normalization.standardize(data)

IO.puts("\nValue | Min-Max [0,1] | Standardized")
IO.puts(String.duplicate("-", 70))

Enum.zip([Nx.to_list(data), Nx.to_list(min_max_scaled), Nx.to_list(standardized)])
|> Enum.each(fn {orig, mm, std} ->
  IO.puts("#{String.pad_leading("#{trunc(orig)}", 5)} | #{String.pad_leading("#{Float.round(mm, 3)}", 13)} | #{String.pad_leading("#{Float.round(std, 3)}", 12)}")
end)


IO.puts("""

OBSERVATION:
  - Min-Max: Outlier (100) becomes 1.0, squashes others near 0
  - Standardization: Outlier preserved as large value (2.4 std devs)
  - Min-Max: Sensitive to outliers
  - Standardization: More robust to outliers

WHEN TO USE:
  - Min-Max: Known bounds, no outliers, need specific range
  - Standardization: Unknown bounds, outliers present, normal distribution
""")

# ============================================================================
# Example 5: Why Normalization Matters for Gradient Descent
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 5: Impact on Gradient Descent")
IO.puts(String.duplicate("-", 70))

IO.puts("""
Without normalization, features on different scales cause problems:

Example: Predicting house price from [bedrooms, square_feet]
  - Bedrooms: 1-5 (small scale)
  - Square feet: 500-5000 (large scale)

Problem: Gradients for square_feet will be MUCH larger!
""")

# Simulate two features with different scales
bedrooms = Nx.tensor([2.0, 3.0, 4.0, 2.0, 5.0])
sqft = Nx.tensor([800.0, 1500.0, 2200.0, 900.0, 3000.0])

# Combine into feature matrix
features = Nx.stack([bedrooms, sqft], axis: 1)

IO.puts("\nOriginal features:")
IO.puts("Bedrooms | Sq Ft")
IO.puts(String.duplicate("-", 70))
Enum.zip([Nx.to_list(bedrooms), Nx.to_list(sqft)])
|> Enum.each(fn {bed, sq} ->
  IO.puts("#{String.pad_leading("#{trunc(bed)}", 8)} | #{String.pad_leading("#{trunc(sq)}", 6)}")
end)

# Normalize
{normalized, _} = MLNx.Normalization.min_max_scale(features)

IO.puts("\nNormalized features (each column scaled independently):")
IO.puts("Bedrooms | Sq Ft")
IO.puts(String.duplicate("-", 70))

for i <- 0..4 do
  bed = Nx.to_number(normalized[i][0])
  sq = Nx.to_number(normalized[i][1])
  IO.puts("#{String.pad_leading("#{Float.round(bed, 3)}", 8)} | #{String.pad_leading("#{Float.round(sq, 3)}", 5)}")
end

IO.puts("""

OBSERVATION:
  - Before: Bedrooms [2-5], Sq Ft [800-3000] - VERY different scales!
  - After: Both features in [0, 1] - SAME scale!
  
GRADIENT DESCENT BENEFIT:
  Without normalization:
    - Learning rate too small → bedrooms update slowly
    - Learning rate too large → square feet diverge
    - Zig-zag path, slow convergence
  
  With normalization:
    - Same learning rate works for all features
    - Direct path to minimum
    - Fast, stable convergence!
""")

# ============================================================================
# Example 6: Inverse Transformations
# ============================================================================

IO.puts("\n" <> String.duplicate("-", 70))
IO.puts("Example 6: Inverse Transformations - Getting Back Original Scale")
IO.puts(String.duplicate("-", 70))

IO.puts("""
After training on normalized data, you often need to convert
predictions back to the original scale.

This is where inverse transformations come in!
""")

original_prices = Nx.tensor([100_000.0, 250_000.0, 500_000.0, 750_000.0, 1_000_000.0])

IO.puts("\nOriginal house prices: #{inspect(Enum.map(Nx.to_list(original_prices), &trunc(&1)))}")

# Normalize for training
{normalized_prices, stats} = MLNx.Normalization.min_max_scale(original_prices)

IO.puts("Normalized (for training): #{inspect(Nx.to_list(normalized_prices))}")

# Simulate a prediction in normalized space
prediction_normalized = Nx.tensor([0.6])  # Model predicts 0.6

IO.puts("\nModel prediction (normalized): #{Nx.to_number(prediction_normalized[0])}")

# Convert back to original scale
prediction_original = MLNx.Normalization.inverse_min_max_scale(prediction_normalized, stats)

IO.puts("Prediction (original scale): $#{trunc(Nx.to_number(prediction_original[0]))}")

# Verify roundtrip
recovered = MLNx.Normalization.inverse_min_max_scale(normalized_prices, stats)
all_close = Nx.all_close(original_prices, recovered) |> Nx.to_number()

IO.puts("\nRoundtrip verification:")
IO.puts("  Original → Normalize → Inverse → Original")
IO.puts("  Success: #{if all_close == 1, do: "✓", else: "✗"}")

IO.puts("""

OBSERVATION:
  - Normalize before training
  - Train on normalized data
  - Inverse transform predictions back to original scale
  - Perfect roundtrip: no information lost!

PRACTICAL WORKFLOW:
  1. Normalize training data, save statistics
  2. Train model on normalized data
  3. Normalize new inputs using SAME statistics
  4. Get predictions (normalized)
  5. Inverse transform predictions to original scale
""")

# ============================================================================
# Summary
# ============================================================================

IO.puts("\n" <> String.duplicate("=", 70))
IO.puts("KEY TAKEAWAYS")
IO.puts(String.duplicate("=", 70))

IO.puts("""
1. WHY NORMALIZE?
   - Features on different scales cause gradient descent problems
   - Normalization ensures all features contribute equally
   - Faster, more stable training

2. MIN-MAX SCALING
   Formula: (x - min) / (max - min)
   - Scales to specific range (default [0, 1])
   - Preserves relationships between values
   - Sensitive to outliers
   - Use when: Known bounds, need specific range

3. STANDARDIZATION (Z-SCORE)
   Formula: (x - mean) / std
   - Centers data: mean = 0, std = 1
   - More robust to outliers
   - No fixed range (unbounded)
   - Use when: Normal distribution, outliers present

4. CHOOSING THE RIGHT METHOD
   - Min-Max: Bounded features, no outliers, neural networks
   - Standardization: Unbounded features, outliers, linear models
   - Both: Try both and compare results!

5. INVERSE TRANSFORMATIONS
   - Always save normalization statistics!
   - Use same statistics for training and prediction
   - Inverse transform to get interpretable results

6. PRACTICAL TIPS
   - Normalize AFTER splitting train/test (use train stats for both)
   - Apply same normalization to all data
   - For 2D data, normalize per-feature (column-wise)

NEXT STEPS:
  - Learn about batch normalization
  - Explore feature engineering
  - Build models with normalized features
""")

IO.puts(String.duplicate("=", 70))
IO.puts("Demo complete! 🎓")
IO.puts(String.duplicate("=", 70) <> "\n")
