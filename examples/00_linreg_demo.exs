# Linear Regression Demo
# Run this with: mix run examples/linreg_demo.exs

IO.puts("\n=== Linear Regression Demo ===\n")

# Example 1: Simple linear relationship (y = 3x + 2)
IO.puts("Example 1: Learning y = 3x + 2")
IO.puts("--------------------------------")

x1 = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
y1 = Nx.tensor([5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0])

IO.puts("Training data:")
IO.inspect(Nx.to_list(x1), label: "X")
IO.inspect(Nx.to_list(y1), label: "Y")

{w1, b1} = MLNx.LinReg.train(x1, y1, iters: 1000, lr: 0.01)

IO.puts("\nLearned parameters:")
IO.puts("  w = #{Nx.to_number(w1[0])} (expected: 3.0)")
IO.puts("  b = #{Nx.to_number(b1)} (expected: 2.0)")

# Test prediction
x_test = Nx.tensor([[10.0]])
y_pred = MLNx.LinReg.predict(x_test, w1, b1)
IO.puts("\nPrediction for x=10: #{Nx.to_number(y_pred[0])} (expected: 32.0)")

# Example 2: Multiple features (y = 2*x1 + 3*x2 + 1)
IO.puts("\n\nExample 2: Multiple features (y = 2*x1 + 3*x2 + 1)")
IO.puts("---------------------------------------------------")

x2 = Nx.tensor([
  [1.0, 1.0],
  [2.0, 1.0],
  [1.0, 2.0],
  [2.0, 2.0],
  [3.0, 1.0],
  [1.0, 3.0],
  [3.0, 3.0],
  [4.0, 2.0]
])

y2 = Nx.tensor([6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 16.0, 13.0])

IO.puts("Training with #{Nx.axis_size(x2, 0)} samples, #{Nx.axis_size(x2, 1)} features")

{w2, b2} = MLNx.LinReg.train(x2, y2, iters: 2000, lr: 0.01)

IO.puts("\nLearned parameters:")
IO.puts("  w = #{inspect(Nx.to_list(w2))} (expected: [2.0, 3.0])")
IO.puts("  b = #{Nx.to_number(b2)} (expected: 1.0)")

# Calculate final loss
y_pred2 = MLNx.LinReg.predict(x2, w2, b2)
final_loss = MLNx.LinReg.mse(y_pred2, y2)
IO.puts("\nFinal MSE loss: #{Nx.to_number(final_loss)}")

# Example 3: Real-world-like data with noise
IO.puts("\n\nExample 3: Noisy data (house prices)")
IO.puts("-------------------------------------")

# Simulate: price = 50 * size + 20 * bedrooms + 100 (in thousands)
# size in 100s of sq ft, bedrooms count
x3 = Nx.tensor([
  [10.0, 2.0],  # 1000 sqft, 2 bed
  [15.0, 3.0],  # 1500 sqft, 3 bed
  [20.0, 3.0],  # 2000 sqft, 3 bed
  [12.0, 2.0],  # 1200 sqft, 2 bed
  [18.0, 4.0],  # 1800 sqft, 4 bed
  [25.0, 4.0],  # 2500 sqft, 4 bed
  [8.0, 1.0],   # 800 sqft, 1 bed
  [22.0, 3.0]   # 2200 sqft, 3 bed
])

# Prices with some noise
y3 = Nx.tensor([640.0, 870.0, 1160.0, 740.0, 1080.0, 1430.0, 520.0, 1260.0])

IO.puts("Training house price model...")
{w3, b3} = MLNx.LinReg.train(x3, y3, iters: 3000, lr: 0.001)

IO.puts("\nLearned parameters:")
IO.puts("  Price per 100 sqft: $#{Nx.to_number(w3[0])}k")
IO.puts("  Price per bedroom: $#{Nx.to_number(w3[1])}k")
IO.puts("  Base price: $#{Nx.to_number(b3)}k")

# Predict for a new house
new_house = Nx.tensor([[16.0, 3.0]])  # 1600 sqft, 3 bedrooms
predicted_price = MLNx.LinReg.predict(new_house, w3, b3)
IO.puts("\nPredicted price for 1600 sqft, 3 bed house: $#{Nx.to_number(predicted_price[0])}k")

IO.puts("\n=== Demo Complete ===\n")
