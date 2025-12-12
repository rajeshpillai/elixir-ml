defmodule MLNx.LRSchedulerTest do
  use ExUnit.Case
  doctest MLNx.LRScheduler

  describe "fixed learning rate" do
    test "returns constant learning rate" do
      lr = MLNx.LRScheduler.get_lr(:fixed, 0, initial_lr: 0.1)
      assert lr == 0.1
      
      lr = MLNx.LRScheduler.get_lr(:fixed, 100, initial_lr: 0.1)
      assert lr == 0.1
    end
  end

  describe "step_decay/4" do
    test "reduces learning rate at decay steps" do
      # Epoch 0-9: lr = 0.1
      assert MLNx.LRScheduler.step_decay(0.1, 0, 0.5, 10) == 0.1
      assert MLNx.LRScheduler.step_decay(0.1, 9, 0.5, 10) == 0.1
      
      # Epoch 10-19: lr = 0.05
      assert MLNx.LRScheduler.step_decay(0.1, 10, 0.5, 10) == 0.05
      assert MLNx.LRScheduler.step_decay(0.1, 19, 0.5, 10) == 0.05
      
      # Epoch 20-29: lr = 0.025
      assert MLNx.LRScheduler.step_decay(0.1, 20, 0.5, 10) == 0.025
    end

    test "works with different decay rates" do
      # Decay by 0.1 every 5 epochs
      assert MLNx.LRScheduler.step_decay(1.0, 0, 0.1, 5) == 1.0
      assert MLNx.LRScheduler.step_decay(1.0, 5, 0.1, 5) == 0.1
      assert_in_delta(MLNx.LRScheduler.step_decay(1.0, 10, 0.1, 5), 0.01, 0.0001)
    end
  end

  describe "exponential_decay/3" do
    test "decays exponentially" do
      # Epoch 0: lr = 0.1
      assert MLNx.LRScheduler.exponential_decay(0.1, 0, 0.95) == 0.1
      
      # Epoch 10: lr ≈ 0.0599
      lr = MLNx.LRScheduler.exponential_decay(0.1, 10, 0.95)
      assert_in_delta(lr, 0.0599, 0.001)
      
      # Epoch 20: lr ≈ 0.0358
      lr = MLNx.LRScheduler.exponential_decay(0.1, 20, 0.95)
      assert_in_delta(lr, 0.0358, 0.001)
    end

    test "approaches zero over time" do
      lr_0 = MLNx.LRScheduler.exponential_decay(0.1, 0, 0.9)
      lr_50 = MLNx.LRScheduler.exponential_decay(0.1, 50, 0.9)
      lr_100 = MLNx.LRScheduler.exponential_decay(0.1, 100, 0.9)
      
      assert lr_0 > lr_50
      assert lr_50 > lr_100
      assert lr_100 < 0.01
    end
  end

  describe "cosine_annealing/4" do
    test "starts at initial_lr" do
      lr = MLNx.LRScheduler.cosine_annealing(0.1, 0, 0.0, 100)
      assert lr == 0.1
    end

    test "reaches min_lr at end" do
      lr = MLNx.LRScheduler.cosine_annealing(0.1, 100, 0.0, 100)
      assert_in_delta(lr, 0.0, 0.0001)
    end

    test "halfway point is between initial and min" do
      lr = MLNx.LRScheduler.cosine_annealing(0.1, 50, 0.0, 100)
      assert lr > 0.0 and lr < 0.1
    end

    test "respects min_lr" do
      lr = MLNx.LRScheduler.cosine_annealing(0.1, 100, 0.01, 100)
      assert_in_delta(lr, 0.01, 0.0001)
    end
  end

  describe "get_lr/3" do
    test "supports all schedule types" do
      # Fixed
      lr = MLNx.LRScheduler.get_lr(:fixed, 10, initial_lr: 0.1)
      assert lr == 0.1
      
      # Step decay
      lr = MLNx.LRScheduler.get_lr(:step_decay, 15, initial_lr: 0.1, decay_rate: 0.5, decay_steps: 10)
      assert lr == 0.05
      
      # Exponential
      lr = MLNx.LRScheduler.get_lr(:exponential, 10, initial_lr: 0.1, decay_rate: 0.95)
      assert_in_delta(lr, 0.0599, 0.001)
      
      # Cosine
      lr = MLNx.LRScheduler.get_lr(:cosine, 100, initial_lr: 0.1, min_lr: 0.0, total_epochs: 100)
      assert_in_delta(lr, 0.0, 0.0001)
    end
  end

  describe "generate_schedule/3" do
    test "generates schedule for all epochs" do
      schedule = MLNx.LRScheduler.generate_schedule(:fixed, 10, initial_lr: 0.1)
      
      assert length(schedule) == 10
      assert Enum.all?(schedule, &(&1 == 0.1))
    end

    test "generates step decay schedule" do
      schedule = MLNx.LRScheduler.generate_schedule(:step_decay, 25, 
        initial_lr: 0.1, decay_rate: 0.5, decay_steps: 10)
      
      # First 10 epochs: 0.1
      assert Enum.at(schedule, 0) == 0.1
      assert Enum.at(schedule, 9) == 0.1
      
      # Next 10 epochs: 0.05
      assert Enum.at(schedule, 10) == 0.05
      assert Enum.at(schedule, 19) == 0.05
      
      # Last 5 epochs: 0.025
      assert Enum.at(schedule, 20) == 0.025
    end

    test "generates exponential decay schedule" do
      schedule = MLNx.LRScheduler.generate_schedule(:exponential, 20,
        initial_lr: 0.1, decay_rate: 0.9)
      
      # Should be monotonically decreasing
      Enum.chunk_every(schedule, 2, 1, :discard)
      |> Enum.each(fn [prev, curr] ->
        assert prev > curr
      end)
    end
  end
end
