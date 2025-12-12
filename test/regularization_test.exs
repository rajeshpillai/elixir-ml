defmodule MLNx.RegularizationTest do
  use ExUnit.Case
  doctest MLNx.Regularization

  describe "l2_penalty/2 - L2 Regularization" do
    test "computes L2 penalty correctly" do
      # L2 = λ * Σw²
      # For weights [2, -3, 1], L2 = λ * (4 + 9 + 1) = λ * 14
      weights = Nx.tensor([2.0, -3.0, 1.0])
      lambda = 0.1

      penalty = MLNx.Regularization.l2_penalty(weights, lambda)

      expected = 0.1 * 14.0
      assert_in_delta(Nx.to_number(penalty), expected, 0.01)
    end

    test "returns zero for zero weights" do
      weights = Nx.tensor([0.0, 0.0, 0.0])
      lambda = 0.1

      penalty = MLNx.Regularization.l2_penalty(weights, lambda)

      assert_in_delta(Nx.to_number(penalty), 0.0, 1.0e-6)
    end

    test "penalty increases with lambda" do
      weights = Nx.tensor([2.0, -3.0])
      
      penalty_small = MLNx.Regularization.l2_penalty(weights, 0.01)
      penalty_large = MLNx.Regularization.l2_penalty(weights, 1.0)

      assert Nx.to_number(penalty_large) > Nx.to_number(penalty_small)
    end

    test "penalizes large weights more heavily" do
      # Due to squaring, large weights contribute more
      small_weights = Nx.tensor([1.0, 1.0])
      large_weights = Nx.tensor([2.0, 2.0])
      lambda = 0.1

      penalty_small = MLNx.Regularization.l2_penalty(small_weights, lambda)
      penalty_large = MLNx.Regularization.l2_penalty(large_weights, lambda)

      # Large weights: 2² + 2² = 8
      # Small weights: 1² + 1² = 2
      # Ratio should be 4:1
      ratio = Nx.to_number(penalty_large) / Nx.to_number(penalty_small)
      assert_in_delta(ratio, 4.0, 0.01)
    end
  end

  describe "l1_penalty/2 - L1 Regularization" do
    test "computes L1 penalty correctly" do
      # L1 = λ * Σ|w|
      # For weights [2, -3, 1], L1 = λ * (2 + 3 + 1) = λ * 6
      weights = Nx.tensor([2.0, -3.0, 1.0])
      lambda = 0.1

      penalty = MLNx.Regularization.l1_penalty(weights, lambda)

      expected = 0.1 * 6.0
      assert_in_delta(Nx.to_number(penalty), expected, 0.01)
    end

    test "treats positive and negative weights equally" do
      weights_pos = Nx.tensor([2.0, 3.0])
      weights_neg = Nx.tensor([-2.0, -3.0])
      lambda = 0.1

      penalty_pos = MLNx.Regularization.l1_penalty(weights_pos, lambda)
      penalty_neg = MLNx.Regularization.l1_penalty(weights_neg, lambda)

      assert_in_delta(
        Nx.to_number(penalty_pos),
        Nx.to_number(penalty_neg),
        1.0e-6
      )
    end

    test "linear penalty for weight magnitude" do
      # L1 penalty grows linearly with weight magnitude
      weights_1x = Nx.tensor([1.0, 1.0])
      weights_2x = Nx.tensor([2.0, 2.0])
      lambda = 0.1

      penalty_1x = MLNx.Regularization.l1_penalty(weights_1x, lambda)
      penalty_2x = MLNx.Regularization.l1_penalty(weights_2x, lambda)

      # Should be exactly 2x
      ratio = Nx.to_number(penalty_2x) / Nx.to_number(penalty_1x)
      assert_in_delta(ratio, 2.0, 0.01)
    end
  end

  describe "elastic_net_penalty/3" do
    test "reduces to L2 when alpha = 0" do
      weights = Nx.tensor([2.0, -3.0, 1.0])
      lambda = 0.1

      elastic = MLNx.Regularization.elastic_net_penalty(weights, lambda, 0.0)
      l2 = MLNx.Regularization.l2_penalty(weights, lambda)

      assert_in_delta(Nx.to_number(elastic), Nx.to_number(l2), 0.01)
    end

    test "reduces to L1 when alpha = 1" do
      weights = Nx.tensor([2.0, -3.0, 1.0])
      lambda = 0.1

      elastic = MLNx.Regularization.elastic_net_penalty(weights, lambda, 1.0)
      l1 = MLNx.Regularization.l1_penalty(weights, lambda)

      assert_in_delta(Nx.to_number(elastic), Nx.to_number(l1), 0.01)
    end

    test "combines L1 and L2 with alpha = 0.5" do
      weights = Nx.tensor([2.0, -3.0])
      lambda = 0.1
      alpha = 0.5

      elastic = MLNx.Regularization.elastic_net_penalty(weights, lambda, alpha)
      l1 = MLNx.Regularization.l1_penalty(weights, lambda)
      l2 = MLNx.Regularization.l2_penalty(weights, lambda)

      # Elastic = λ * [α*L1_term + (1-α)*L2_term]
      # Should be between L1 and L2
      elastic_val = Nx.to_number(elastic)
      l1_val = Nx.to_number(l1)
      l2_val = Nx.to_number(l2)

      # Elastic should be between min and max of L1 and L2
      min_val = min(l1_val, l2_val)
      max_val = max(l1_val, l2_val)
      
      assert elastic_val >= min_val - 0.01
      assert elastic_val <= max_val + 0.01
    end
  end

  describe "regularized_loss/4" do
    test "adds L2 penalty to MSE loss" do
      predictions = Nx.tensor([2.0, 3.0, 4.0])
      targets = Nx.tensor([2.5, 3.5, 4.5])
      weights = Nx.tensor([1.0, 2.0])

      # Compute separately
      mse = MLNx.LossFunctions.mse(predictions, targets)
      l2_pen = MLNx.Regularization.l2_penalty(weights, 0.1)
      expected = Nx.add(mse, l2_pen)

      # Compute with regularized_loss
      loss = MLNx.Regularization.regularized_loss(
        predictions, targets, weights,
        reg_type: :l2, lambda: 0.1
      )

      assert_in_delta(Nx.to_number(loss), Nx.to_number(expected), 0.01)
    end

    test "adds L1 penalty to MSE loss" do
      predictions = Nx.tensor([2.0, 3.0])
      targets = Nx.tensor([2.5, 3.5])
      weights = Nx.tensor([1.0, -2.0])

      loss = MLNx.Regularization.regularized_loss(
        predictions, targets, weights,
        reg_type: :l1, lambda: 0.1
      )

      # Should be MSE + L1 penalty
      assert is_struct(loss, Nx.Tensor)
      assert Nx.to_number(loss) > 0
    end

    test "uses default L2 regularization" do
      predictions = Nx.tensor([1.0, 2.0])
      targets = Nx.tensor([1.5, 2.5])
      weights = Nx.tensor([1.0])

      # Should work without specifying reg_type
      loss = MLNx.Regularization.regularized_loss(
        predictions, targets, weights
      )

      assert is_struct(loss, Nx.Tensor)
    end

    test "regularization increases total loss" do
      predictions = Nx.tensor([2.0, 3.0, 4.0])
      targets = Nx.tensor([2.5, 3.5, 4.5])
      weights = Nx.tensor([2.0, -3.0])

      # Loss without regularization
      mse = MLNx.LossFunctions.mse(predictions, targets)

      # Loss with regularization
      reg_loss = MLNx.Regularization.regularized_loss(
        predictions, targets, weights,
        reg_type: :l2, lambda: 0.1
      )

      assert Nx.to_number(reg_loss) > Nx.to_number(mse)
    end
  end

  describe "compare_lambdas/2" do
    test "shows penalty increases with lambda" do
      weights = Nx.tensor([2.0, -3.0])
      
      penalties = MLNx.Regularization.compare_lambdas(
        weights,
        reg_type: :l2,
        lambdas: [0.0, 0.1, 1.0]
      )

      assert penalties[0.0] == 0.0
      assert penalties[0.1] < penalties[1.0]
    end

    test "compares different regularization types" do
      weights = Nx.tensor([2.0, -3.0, 1.0])
      lambdas = [0.1]

      l1_penalties = MLNx.Regularization.compare_lambdas(
        weights,
        reg_type: :l1,
        lambdas: lambdas
      )

      l2_penalties = MLNx.Regularization.compare_lambdas(
        weights,
        reg_type: :l2,
        lambdas: lambdas
      )

      # L1 and L2 should give different penalties
      assert l1_penalties[0.1] != l2_penalties[0.1]
    end
  end

  describe "regularization effects" do
    test "L1 vs L2 penalty comparison" do
      # For same weights, L1 and L2 give different penalties
      weights = Nx.tensor([2.0, -3.0, 1.0])
      lambda = 0.1

      l1 = MLNx.Regularization.l1_penalty(weights, lambda) |> Nx.to_number()
      l2 = MLNx.Regularization.l2_penalty(weights, lambda) |> Nx.to_number()

      # L1 = 0.1 * (2 + 3 + 1) = 0.6
      # L2 = 0.1 * (4 + 9 + 1) = 1.4
      assert_in_delta(l1, 0.6, 0.01)
      assert_in_delta(l2, 1.4, 0.01)
    end

    test "L2 penalizes large weights more than L1" do
      # Large weight
      large_weight = Nx.tensor([10.0])
      lambda = 0.1

      l1 = MLNx.Regularization.l1_penalty(large_weight, lambda) |> Nx.to_number()
      l2 = MLNx.Regularization.l2_penalty(large_weight, lambda) |> Nx.to_number()

      # L1 = 0.1 * 10 = 1.0
      # L2 = 0.1 * 100 = 10.0
      # L2 should be much larger
      assert l2 > l1 * 5
    end
  end
end
