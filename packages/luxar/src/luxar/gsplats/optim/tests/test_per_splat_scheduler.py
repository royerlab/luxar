"""
Comprehensive tests for per-splat learning rate schedulers.

Tests cover PerSplatReduceLROnPlateau and PerSplatExponentialLR schedulers
with various edge cases, error handling, and state management scenarios.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.optim.per_splat_adam import PerSplatAdam
from luxar.gsplats.optim.per_splat_scheduler import (
    PerSplatExponentialLR,
    PerSplatReduceLROnPlateau,
)


# Test fixtures
@pytest.fixture
def simple_model():
    """Create a simple test model."""
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

    shape = (32, 32)  # 2D image shape
    centers0 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
    L0 = torch.stack([torch.eye(2) * 0.5 for _ in range(3)], dim=0)
    amps0 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    return GaussianSplatModel(
        shape=shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5, 0.5],
        device=torch.device("cpu"),
    )


@pytest.fixture
def optimizer(simple_model):
    """Create test optimizer."""
    return PerSplatAdam(simple_model, lr=0.1)


@pytest.fixture
def plateau_scheduler(optimizer):
    """Create default plateau scheduler."""
    return PerSplatReduceLROnPlateau(optimizer, patience=2, factor=0.5)


class TestPerSplatReduceLROnPlateauInitialization:
    """Test scheduler initialization and configuration."""

    def test_default_initialization(self, optimizer) -> None:
        """Test initialization with default parameters."""
        scheduler = PerSplatReduceLROnPlateau(optimizer)

        assert scheduler.mode == "min"
        assert scheduler.factor == 0.5
        assert scheduler.patience == 10
        assert scheduler.threshold == 1e-4
        assert scheduler.cooldown == 0
        assert scheduler.min_lr == 1e-8
        assert scheduler.global_patience == 20
        assert len(scheduler.splat_scheduler_states) == 0
        assert scheduler.global_best is None
        assert scheduler.global_bad_epochs == 0
        assert scheduler.last_epoch == 0

    def test_custom_initialization(self, optimizer) -> None:
        """Test initialization with custom parameters."""
        scheduler = PerSplatReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.8,
            patience=5,
            threshold=0.01,
            cooldown=3,
            min_lr=1e-6,
            global_patience=15,
        )

        assert scheduler.mode == "max"
        assert scheduler.factor == 0.8
        assert scheduler.patience == 5
        assert scheduler.threshold == 0.01
        assert scheduler.cooldown == 3
        assert scheduler.min_lr == 1e-6
        assert scheduler.global_patience == 15

    def test_mode_max_initialization(self, optimizer) -> None:
        """Test initialization with mode='max'."""
        scheduler = PerSplatReduceLROnPlateau(optimizer, mode="max")

        # Initialize a splat state and check it uses -inf as best for max mode
        scheduler._init_splat_state(0)
        assert scheduler.splat_scheduler_states[0]["best"] == float("-inf")


class TestPerSplatReduceLROnPlateauStepGlobalMetrics:
    """Test step() method with global metrics."""

    def test_step_with_float_metric(self, plateau_scheduler) -> None:
        """Test step with float metric (global)."""
        plateau_scheduler.step(1.0)

        assert plateau_scheduler.last_epoch == 1
        assert plateau_scheduler.global_best == 1.0
        # Should initialize states for all 3 splats
        assert len(plateau_scheduler.splat_scheduler_states) == 3

    def test_step_with_int_metric(self, plateau_scheduler) -> None:
        """Test step with integer metric."""
        plateau_scheduler.step(5)

        assert plateau_scheduler.last_epoch == 1
        assert plateau_scheduler.global_best == 5.0

    def test_step_with_scalar_tensor(self, plateau_scheduler) -> None:
        """Test step with scalar tensor metric."""
        metric = torch.tensor(2.5)
        plateau_scheduler.step(metric)

        assert plateau_scheduler.last_epoch == 1
        assert plateau_scheduler.global_best == 2.5

    def test_step_with_none_raises_error(self, plateau_scheduler) -> None:
        """Test that None metric raises ValueError."""
        with pytest.raises(ValueError, match="Metrics cannot be None"):
            plateau_scheduler.step(None)

    def test_step_with_invalid_float_raises_error(self, plateau_scheduler) -> None:
        """Test that invalid float values raise errors."""
        with pytest.raises(RuntimeError, match="Error processing metrics"):
            plateau_scheduler.step(float("nan"))

        with pytest.raises(RuntimeError, match="Error processing metrics"):
            plateau_scheduler.step(float("inf"))


class TestPerSplatReduceLROnPlateauStepPerSplatMetrics:
    """Test step() method with per-splat metrics."""

    def test_step_with_1d_tensor(self, plateau_scheduler) -> None:
        """Test step with 1D tensor (per-splat metrics)."""
        metrics = torch.tensor([1.0, 2.0, 3.0])
        plateau_scheduler.step(metrics)

        assert plateau_scheduler.last_epoch == 1
        assert len(plateau_scheduler.splat_scheduler_states) == 3
        # Global metric should be mean
        assert plateau_scheduler.global_best == 2.0

    def test_step_with_dict_metrics(self, plateau_scheduler) -> None:
        """Test step with dict metrics."""
        metrics = {0: 1.0, 1: 2.0, 2: 3.0}
        plateau_scheduler.step(metrics)

        assert plateau_scheduler.last_epoch == 1
        assert len(plateau_scheduler.splat_scheduler_states) == 3
        # Global metric should be mean
        assert plateau_scheduler.global_best == 2.0

    def test_step_with_multidim_tensor_raises_error(self, plateau_scheduler) -> None:
        """Test that multi-dimensional tensors raise error."""
        metrics = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(RuntimeError, match="Error processing metrics"):
            plateau_scheduler.step(metrics)

    def test_step_with_invalid_tensor_values(self, plateau_scheduler) -> None:
        """Test that tensors with invalid values raise errors."""
        metrics = torch.tensor([1.0, float("nan"), 3.0])

        with pytest.raises(RuntimeError, match="Error processing metrics"):
            plateau_scheduler.step(metrics)

    def test_step_with_invalid_dict_keys(self, plateau_scheduler) -> None:
        """Test that invalid dict keys raise errors."""
        # Negative key
        with pytest.raises(RuntimeError, match="Error processing metrics"):
            plateau_scheduler.step({-1: 1.0, 0: 2.0})

        # Non-integer key
        with pytest.raises(RuntimeError, match="Error processing metrics"):
            plateau_scheduler.step({"0": 1.0, 1: 2.0})

    def test_step_with_invalid_dict_values(self, plateau_scheduler) -> None:
        """Test that invalid dict values raise errors."""
        with pytest.raises(RuntimeError, match="Error processing metrics"):
            plateau_scheduler.step({0: float("nan"), 1: 2.0})

    def test_step_with_unsupported_type_raises_error(self, plateau_scheduler) -> None:
        """Test that unsupported types raise TypeError."""
        with pytest.raises(RuntimeError, match="Error processing metrics"):
            plateau_scheduler.step([1.0, 2.0, 3.0])  # List not supported


class TestPerSplatReduceLROnPlateauLRReduction:
    """Test learning rate reduction logic."""

    def test_lr_reduction_on_plateau_min_mode(self, optimizer) -> None:
        """Test that LR is reduced when metric plateaus (min mode)."""
        scheduler = PerSplatReduceLROnPlateau(
            optimizer, patience=2, factor=0.5, mode="min"
        )

        initial_lrs = [optimizer.get_learning_rate(i) for i in range(3)]

        # No improvement for patience+1 steps
        for _ in range(3):
            scheduler.step(1.0)

        # LRs should be reduced
        final_lrs = [optimizer.get_learning_rate(i) for i in range(3)]
        for initial, final in zip(initial_lrs, final_lrs):
            assert final < initial
            assert final == pytest.approx(initial * 0.5)

    def test_lr_reduction_on_plateau_max_mode(self, optimizer) -> None:
        """Test LR reduction with mode='max'."""
        scheduler = PerSplatReduceLROnPlateau(
            optimizer, patience=2, factor=0.5, mode="max"
        )

        initial_lrs = [optimizer.get_learning_rate(i) for i in range(3)]

        # No improvement for patience+1 steps (metric not increasing)
        for _ in range(3):
            scheduler.step(1.0)

        # LRs should be reduced
        final_lrs = [optimizer.get_learning_rate(i) for i in range(3)]
        for initial, final in zip(initial_lrs, final_lrs):
            assert final < initial

    def test_no_reduction_with_improvement(self, plateau_scheduler) -> None:
        """Test that LR is not reduced when metric improves."""
        initial_lrs = [
            plateau_scheduler.optimizer.get_learning_rate(i) for i in range(3)
        ]

        # Improving metrics
        for i in range(5):
            plateau_scheduler.step(1.0 - i * 0.1)

        # LRs should not change
        final_lrs = [plateau_scheduler.optimizer.get_learning_rate(i) for i in range(3)]
        for initial, final in zip(initial_lrs, final_lrs):
            assert final == pytest.approx(initial)

    def test_min_lr_respected(self, optimizer) -> None:
        """Test that learning rate doesn't go below min_lr."""
        scheduler = PerSplatReduceLROnPlateau(
            optimizer, patience=1, factor=0.1, min_lr=0.01
        )

        # Trigger many reductions
        for _ in range(10):
            scheduler.step(1.0)

        # Check all LRs are at or above min_lr
        for i in range(3):
            lr = optimizer.get_learning_rate(i)
            assert lr >= 0.01


class TestPerSplatReduceLROnPlateauCooldown:
    """Test cooldown behavior."""

    def test_cooldown_prevents_immediate_reduction(self, optimizer) -> None:
        """Test that cooldown prevents immediate consecutive reductions."""
        scheduler = PerSplatReduceLROnPlateau(
            optimizer, patience=1, factor=0.5, cooldown=2
        )

        # Trigger first reduction
        scheduler.step(1.0)
        scheduler.step(1.0)

        lr_after_first_reduction = optimizer.get_learning_rate(0)

        # During cooldown, more bad epochs shouldn't trigger reduction
        scheduler.step(1.0)
        scheduler.step(1.0)

        # LR should not have changed during cooldown
        assert optimizer.get_learning_rate(0) == pytest.approx(lr_after_first_reduction)

    def test_global_cooldown(self, optimizer) -> None:
        """Test global cooldown behavior."""
        scheduler = PerSplatReduceLROnPlateau(
            optimizer, patience=1, global_patience=3, cooldown=2
        )

        # Trigger global reduction (after 4 steps, global_bad_epochs reaches 3)
        for _ in range(4):
            scheduler.step(1.0)

        # After cooldown is set, it decrements on next step (since we call step again)
        assert scheduler.global_cooldown_counter == 1  # Already decremented

        # Next step should decrement cooldown further
        scheduler.step(1.0)
        assert scheduler.global_cooldown_counter == 0


class TestPerSplatReduceLROnPlateauComparison:
    """Test _is_better() comparison logic."""

    def test_is_better_min_mode(self, optimizer) -> None:
        """Test comparison in min mode."""
        scheduler = PerSplatReduceLROnPlateau(optimizer, mode="min", threshold=0.01)

        # Significantly better
        assert scheduler._is_better(0.5, 1.0) is True
        # Not better enough (within threshold)
        assert scheduler._is_better(0.995, 1.0) is False
        # Worse
        assert scheduler._is_better(1.5, 1.0) is False

    def test_is_better_max_mode(self, optimizer) -> None:
        """Test comparison in max mode."""
        scheduler = PerSplatReduceLROnPlateau(optimizer, mode="max", threshold=0.01)

        # Significantly better
        assert scheduler._is_better(1.5, 1.0) is True
        # Not better enough (within threshold)
        assert scheduler._is_better(1.005, 1.0) is False
        # Worse
        assert scheduler._is_better(0.5, 1.0) is False


class TestPerSplatReduceLROnPlateauTopologyChanges:
    """Test add_splats and remove_splats methods."""

    def test_add_splats_positive(self, plateau_scheduler) -> None:
        """Test adding new splats."""
        # Initialize some states
        plateau_scheduler.step(1.0)
        assert len(plateau_scheduler.splat_scheduler_states) == 3

        # Add 2 new splats
        plateau_scheduler.add_splats(2)
        assert len(plateau_scheduler.splat_scheduler_states) == 5

        # Check new states are initialized
        for i in [3, 4]:
            assert i in plateau_scheduler.splat_scheduler_states
            state = plateau_scheduler.splat_scheduler_states[i]
            assert state["best"] == float("inf")  # min mode
            assert state["num_bad_epochs"] == 0

    def test_add_splats_zero(self, plateau_scheduler) -> None:
        """Test adding zero splats (no-op)."""
        plateau_scheduler.step(1.0)
        initial_count = len(plateau_scheduler.splat_scheduler_states)

        plateau_scheduler.add_splats(0)
        assert len(plateau_scheduler.splat_scheduler_states) == initial_count

    def test_add_splats_negative_raises_error(self, plateau_scheduler) -> None:
        """Test that negative n_new_splats raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            plateau_scheduler.add_splats(-1)

    def test_remove_splats_basic(self, plateau_scheduler) -> None:
        """Test removing splats."""
        # Initialize states
        plateau_scheduler.step(1.0)
        assert len(plateau_scheduler.splat_scheduler_states) == 3

        # Remove middle splat
        keep_mask = torch.tensor([True, False, True])
        plateau_scheduler.remove_splats(keep_mask)

        assert len(plateau_scheduler.splat_scheduler_states) == 2
        # Check indices are remapped
        assert 0 in plateau_scheduler.splat_scheduler_states
        assert 1 in plateau_scheduler.splat_scheduler_states
        assert 2 not in plateau_scheduler.splat_scheduler_states

    def test_remove_splats_invalid_type_raises_error(self, plateau_scheduler) -> None:
        """Test that non-tensor keep_mask raises error."""
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            plateau_scheduler.remove_splats([True, False, True])

    def test_remove_splats_invalid_dtype_raises_error(self, plateau_scheduler) -> None:
        """Test that non-boolean tensor raises error."""
        plateau_scheduler.step(1.0)
        keep_mask = torch.tensor([1, 0, 1])  # int tensor

        with pytest.raises(TypeError, match="must be boolean tensor"):
            plateau_scheduler.remove_splats(keep_mask)

    def test_remove_splats_invalid_shape_raises_error(self, plateau_scheduler) -> None:
        """Test that non-1D tensor raises error."""
        plateau_scheduler.step(1.0)
        keep_mask = torch.tensor([[True, False], [True, False]])

        with pytest.raises(ValueError, match="must be 1D tensor"):
            plateau_scheduler.remove_splats(keep_mask)

    def test_remove_splats_length_mismatch_raises_error(
        self, plateau_scheduler
    ) -> None:
        """Test that mismatched length raises error."""
        plateau_scheduler.step(1.0)
        keep_mask = torch.tensor([True, False])  # Only 2 elements, but have 3 splats

        with pytest.raises(ValueError, match="doesn't match"):
            plateau_scheduler.remove_splats(keep_mask)


class TestPerSplatReduceLROnPlateauMonitoring:
    """Test monitoring and introspection methods."""

    def test_get_lr_reduction_counts(self, plateau_scheduler) -> None:
        """Test getting LR reduction counts per splat."""
        # Trigger some reductions
        for _ in range(5):
            plateau_scheduler.step(1.0)

        counts = plateau_scheduler.get_lr_reduction_counts()
        assert counts.shape == (3,)
        # All splats should have had reductions
        assert torch.all(counts > 0)

    def test_get_lr_reduction_counts_before_init(self, optimizer) -> None:
        """Test counts before any steps (all zeros)."""
        scheduler = PerSplatReduceLROnPlateau(optimizer)
        counts = scheduler.get_lr_reduction_counts()

        assert counts.shape == (3,)
        assert torch.all(counts == 0)


class TestPerSplatReduceLROnPlateauStateSerialization:
    """Test state_dict and load_state_dict."""

    def test_state_dict(self, plateau_scheduler) -> None:
        """Test state_dict() returns correct structure."""
        # Run a few steps to populate state
        plateau_scheduler.step(1.0)
        plateau_scheduler.step(1.5)

        state = plateau_scheduler.state_dict()

        assert "splat_scheduler_states" in state
        assert "global_best" in state
        assert "global_bad_epochs" in state
        assert "global_cooldown_counter" in state
        assert "last_epoch" in state

        assert state["last_epoch"] == 2
        assert state["global_best"] == 1.0  # min mode, so 1.0 is best

    def test_load_state_dict(self, optimizer) -> None:
        """Test load_state_dict() restores state."""
        # Create and run scheduler
        scheduler1 = PerSplatReduceLROnPlateau(optimizer, patience=2)
        for i in range(5):
            scheduler1.step(1.0 + i * 0.1)

        # Save state
        state = scheduler1.state_dict()

        # Create new scheduler and load state
        scheduler2 = PerSplatReduceLROnPlateau(optimizer, patience=2)
        scheduler2.load_state_dict(state)

        # Check state matches
        assert scheduler2.last_epoch == scheduler1.last_epoch
        assert scheduler2.global_best == scheduler1.global_best
        assert scheduler2.global_bad_epochs == scheduler1.global_bad_epochs
        assert len(scheduler2.splat_scheduler_states) == len(
            scheduler1.splat_scheduler_states
        )


class TestPerSplatExponentialLRBasic:
    """Test PerSplatExponentialLR basic functionality."""

    def test_initialization(self, optimizer) -> None:
        """Test basic initialization."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.9)

        assert scheduler.gamma == 0.9
        assert scheduler.age_based_decay is True
        assert scheduler.current_epoch == 0
        assert len(scheduler.splat_ages) == 3

    def test_initialization_no_age_decay(self, optimizer) -> None:
        """Test initialization with age_based_decay=False."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.95, age_based_decay=False)

        assert scheduler.age_based_decay is False

    def test_basic_decay(self, optimizer) -> None:
        """Test basic exponential decay."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.9, age_based_decay=False)

        initial_lrs = [optimizer.get_learning_rate(i) for i in range(3)]

        scheduler.step()

        final_lrs = [optimizer.get_learning_rate(i) for i in range(3)]
        for initial, final in zip(initial_lrs, final_lrs):
            assert final == pytest.approx(initial * 0.9)

    def test_multiple_steps(self, optimizer) -> None:
        """Test multiple decay steps."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.9, age_based_decay=False)

        initial_lr = optimizer.get_learning_rate(0)

        for _ in range(5):
            scheduler.step()

        final_lr = optimizer.get_learning_rate(0)
        expected_lr = initial_lr * (0.9**5)
        assert final_lr == pytest.approx(expected_lr, rel=1e-5)

    def test_age_based_decay(self, optimizer) -> None:
        """Test age-based decay behavior."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.9, age_based_decay=True)

        initial_lrs = [optimizer.get_learning_rate(i) for i in range(3)]

        # First step - all splats are same age
        scheduler.step()

        first_step_lrs = [optimizer.get_learning_rate(i) for i in range(3)]
        # All should decay by same amount
        for i in range(3):
            assert first_step_lrs[i] < initial_lrs[i]


class TestPerSplatExponentialLRTopologyChanges:
    """Test add_splats and remove_splats for exponential scheduler."""

    def test_add_splats(self, optimizer) -> None:
        """Test adding new splats."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.9)

        # Run a few steps
        scheduler.step()
        scheduler.step()
        assert scheduler.current_epoch == 2

        # Add new splats
        scheduler.add_splats(2)

        assert len(scheduler.splat_ages) == 5
        # New splats should have current epoch as birth time
        assert scheduler.splat_ages[3] == 2
        assert scheduler.splat_ages[4] == 2

    def test_remove_splats(self, optimizer) -> None:
        """Test removing splats."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.9)

        # Run steps and add splats to have different ages
        scheduler.step()
        scheduler.add_splats(1)
        scheduler.step()

        # Remove middle two splats
        keep_mask = torch.tensor([True, False, False, True])
        scheduler.remove_splats(keep_mask)

        assert len(scheduler.splat_ages) == 2
        # Check indices remapped
        assert 0 in scheduler.splat_ages
        assert 1 in scheduler.splat_ages

    def test_newer_splats_decay_slower(self, optimizer) -> None:
        """Test that newer splats have different ages tracked."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.9, age_based_decay=True)

        # Initial splats all have age 0
        assert scheduler.splat_ages[0] == 0
        assert scheduler.splat_ages[1] == 0
        assert scheduler.splat_ages[2] == 0

        # Run some steps to age existing splats
        for _ in range(3):
            scheduler.step()

        # Add a new splat
        centers_new = torch.tensor([[3.0, 3.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2) * 0.5], dim=0)
        amps_new = torch.tensor([1.0], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0], dtype=torch.float32)

        optimizer.model.append_(centers_new, Ls_new, amps_new, sharpness_new)
        optimizer.add_splats(1)
        scheduler.add_splats(1)

        # New splat should have birth epoch = current epoch
        assert scheduler.splat_ages[3] == 3

        # After another step, verify ages are tracked correctly
        scheduler.step()

        # Old splats have age = current_epoch - 0 = 4
        # New splat has age = current_epoch - 3 = 1
        # So new splat is younger and should apply different decay rate


class TestPerSplatSchedulerIntegration:
    """Integration tests combining schedulers with optimization."""

    def test_scheduler_reduces_lr_during_training(self, optimizer) -> None:
        """Test that scheduler actually affects training loop."""
        scheduler = PerSplatReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        # Simulate training loop with plateau
        losses = []
        for epoch in range(10):
            # Simulate a loss that plateaus
            loss = 1.0 if epoch < 5 else 1.0 + np.random.normal(0, 0.001)
            losses.append(loss)

            scheduler.step(loss)

        # LRs should have been reduced
        final_lr = optimizer.get_learning_rate(0)
        assert final_lr < 0.1  # Started at 0.1

    def test_exponential_scheduler_continuous_decay(self, optimizer) -> None:
        """Test exponential scheduler provides continuous decay."""
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.95, age_based_decay=False)

        lrs_over_time = []
        for _ in range(20):
            lrs_over_time.append(optimizer.get_learning_rate(0))
            scheduler.step()

        # Should be monotonically decreasing
        for i in range(len(lrs_over_time) - 1):
            assert lrs_over_time[i] > lrs_over_time[i + 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
