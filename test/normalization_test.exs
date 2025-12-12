defmodule MLNx.NormalizationTest do
  use ExUnit.Case
  doctest MLNx.Normalization

  describe "min_max_scale/1" do
    test "scales to [0, 1] range" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {scaled, {min_val, max_val}} = MLNx.Normalization.min_max_scale(data)
      
      assert Nx.to_flat_list(scaled) == [0.0, 0.25, 0.5, 0.75, 1.0]
      assert Nx.to_number(min_val) == 1.0
      assert Nx.to_number(max_val) == 5.0
    end

    test "handles negative values" do
      data = Nx.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
      {scaled, _} = MLNx.Normalization.min_max_scale(data)
      
      assert Nx.to_flat_list(scaled) == [0.0, 0.25, 0.5, 0.75, 1.0]
    end

    test "handles single value" do
      data = Nx.tensor([5.0])
      {scaled, _} = MLNx.Normalization.min_max_scale(data)
      
      # When min == max, should return 0.0
      assert Nx.to_flat_list(scaled) == [0.0]
    end

    test "handles all same values" do
      data = Nx.tensor([3.0, 3.0, 3.0, 3.0])
      {scaled, _} = MLNx.Normalization.min_max_scale(data)
      
      # When all values are same, should return all 0.0
      assert Nx.to_flat_list(scaled) == [0.0, 0.0, 0.0, 0.0]
    end

    test "works with 2D tensors (per-feature normalization)" do
      # Two features: [1,2,3] and [10,20,30]
      data = Nx.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
      {scaled, {min_vals, max_vals}} = MLNx.Normalization.min_max_scale(data)
      
      # Each column should be scaled independently
      assert Nx.shape(scaled) == {3, 2}
      
      # First column: [1,2,3] -> [0, 0.5, 1]
      first_col = scaled[[0..-1//1, 0]] |> Nx.to_flat_list()
      assert first_col == [0.0, 0.5, 1.0]
      
      # Second column: [10,20,30] -> [0, 0.5, 1]
      second_col = scaled[[0..-1//1, 1]] |> Nx.to_flat_list()
      assert second_col == [0.0, 0.5, 1.0]
      
      # Check stats
      assert Nx.to_flat_list(min_vals) == [1.0, 10.0]
      assert Nx.to_flat_list(max_vals) == [3.0, 30.0]
    end
  end

  describe "min_max_scale/3" do
    test "scales to custom range [-1, 1]" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {scaled, _} = MLNx.Normalization.min_max_scale(data, -1, 1)
      
      assert Nx.to_flat_list(scaled) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    end

    test "scales to custom range [0, 10]" do
      data = Nx.tensor([0.0, 50.0, 100.0])
      {scaled, _} = MLNx.Normalization.min_max_scale(data, 0, 10)
      
      assert Nx.to_flat_list(scaled) == [0.0, 5.0, 10.0]
    end

    test "scales to custom range [10, 20]" do
      data = Nx.tensor([0.0, 5.0, 10.0])
      {scaled, _} = MLNx.Normalization.min_max_scale(data, 10, 20)
      
      assert Nx.to_flat_list(scaled) == [10.0, 15.0, 20.0]
    end
  end

  describe "inverse_min_max_scale/3" do
    test "reverses scaling to original values" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {scaled, stats} = MLNx.Normalization.min_max_scale(data)
      original = MLNx.Normalization.inverse_min_max_scale(scaled, stats)
      
      assert Nx.all_close(data, original) |> Nx.to_number() == 1
    end

    test "reverses custom range scaling" do
      data = Nx.tensor([10.0, 20.0, 30.0, 40.0])
      {scaled, stats} = MLNx.Normalization.min_max_scale(data, -1, 1)
      original = MLNx.Normalization.inverse_min_max_scale(scaled, stats, -1, 1)
      
      assert Nx.all_close(data, original) |> Nx.to_number() == 1
    end

    test "works with 2D tensors" do
      data = Nx.tensor([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
      {scaled, stats} = MLNx.Normalization.min_max_scale(data)
      original = MLNx.Normalization.inverse_min_max_scale(scaled, stats)
      
      assert Nx.all_close(data, original) |> Nx.to_number() == 1
    end
  end

  describe "standardize/1" do
    test "standardizes to mean=0, std=1" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {standardized, {mean, std}} = MLNx.Normalization.standardize(data)
      
      # Check mean is 3.0
      assert Nx.to_number(mean) == 3.0
      
      # Check std is approximately sqrt(2) ≈ 1.414
      assert_in_delta(Nx.to_number(std), 1.414, 0.01)
      
      # Check standardized values have mean ≈ 0
      standardized_mean = Nx.mean(standardized) |> Nx.to_number()
      assert_in_delta(standardized_mean, 0.0, 0.0001)
      
      # Check standardized values have std ≈ 1
      standardized_std = Nx.standard_deviation(standardized) |> Nx.to_number()
      assert_in_delta(standardized_std, 1.0, 0.0001)
    end

    test "handles negative values" do
      data = Nx.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
      {standardized, {mean, _std}} = MLNx.Normalization.standardize(data)
      
      # Mean should be 0
      assert Nx.to_number(mean) == 0.0
      
      # Standardized mean should still be ≈ 0
      standardized_mean = Nx.mean(standardized) |> Nx.to_number()
      assert_in_delta(standardized_mean, 0.0, 0.0001)
    end

    test "handles all same values" do
      data = Nx.tensor([5.0, 5.0, 5.0, 5.0])
      {standardized, {mean, std}} = MLNx.Normalization.standardize(data)
      
      # Mean should be 5.0
      assert Nx.to_number(mean) == 5.0
      
      # Std should be 0
      assert Nx.to_number(std) == 0.0
      
      # Standardized values should be 0 (to avoid division by zero)
      assert Nx.to_flat_list(standardized) == [0.0, 0.0, 0.0, 0.0]
    end

    test "works with 2D tensors (per-feature normalization)" do
      # Two features with different scales
      data = Nx.tensor([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
      {standardized, {means, stds}} = MLNx.Normalization.standardize(data)
      
      assert Nx.shape(standardized) == {3, 2}
      
      # Check means
      assert Nx.to_flat_list(means) == [2.0, 200.0]
      
      # Each column should have mean ≈ 0 and std ≈ 1
      first_col = standardized[[0..-1//1, 0]]
      first_mean = Nx.mean(first_col) |> Nx.to_number()
      first_std = Nx.standard_deviation(first_col) |> Nx.to_number()
      
      assert_in_delta(first_mean, 0.0, 0.0001)
      assert_in_delta(first_std, 1.0, 0.0001)
    end
  end

  describe "inverse_standardize/2" do
    test "reverses standardization to original values" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {standardized, stats} = MLNx.Normalization.standardize(data)
      original = MLNx.Normalization.inverse_standardize(standardized, stats)
      
      assert Nx.all_close(data, original) |> Nx.to_number() == 1
    end

    test "works with negative values" do
      data = Nx.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
      {standardized, stats} = MLNx.Normalization.standardize(data)
      original = MLNx.Normalization.inverse_standardize(standardized, stats)
      
      assert Nx.all_close(data, original) |> Nx.to_number() == 1
    end

    test "works with 2D tensors" do
      data = Nx.tensor([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
      {standardized, stats} = MLNx.Normalization.standardize(data)
      original = MLNx.Normalization.inverse_standardize(standardized, stats)
      
      assert Nx.all_close(data, original) |> Nx.to_number() == 1
    end
  end

  describe "compute_stats/1" do
    test "computes mean and std for 1D tensor" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {mean, std} = MLNx.Normalization.compute_stats(data)
      
      assert Nx.to_number(mean) == 3.0
      assert_in_delta(Nx.to_number(std), 1.414, 0.01)
    end

    test "computes per-feature stats for 2D tensor" do
      data = Nx.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
      {means, stds} = MLNx.Normalization.compute_stats(data)
      
      assert Nx.to_flat_list(means) == [2.0, 20.0]
      assert_in_delta(Nx.to_number(stds[0]), 0.816, 0.01)
      assert_in_delta(Nx.to_number(stds[1]), 8.165, 0.01)
    end
  end

  describe "normalize/2" do
    test "defaults to min-max scaling" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {scaled, _} = MLNx.Normalization.normalize(data)
      
      assert Nx.to_flat_list(scaled) == [0.0, 0.25, 0.5, 0.75, 1.0]
    end

    test "supports custom range for min-max" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {scaled, _} = MLNx.Normalization.normalize(data, method: :min_max, range: {-1, 1})
      
      assert Nx.to_flat_list(scaled) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    end

    test "supports standardization method" do
      data = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      {standardized, {mean, _std}} = MLNx.Normalization.normalize(data, method: :standardize)
      
      assert Nx.to_number(mean) == 3.0
      
      # Check mean is ≈ 0
      standardized_mean = Nx.mean(standardized) |> Nx.to_number()
      assert_in_delta(standardized_mean, 0.0, 0.0001)
    end
  end
end
