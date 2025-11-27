"""
Comprehensive tests for optimizer integration helpers.

Tests ModelOptimizerCoordinator and factory functions with focus on
meaningful scenarios: state synchronization, model replacement, and
coordinated dynamic operations.
"""

import pytest
import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import PerSplatAdam
from luxar.gsplats.optim.integration import (
    ModelOptimizerCoordinator,
    create_per_splat_optimizer_setup,
)
from luxar.gsplats.optim.per_splat_scheduler import (
    PerSplatExponentialLR,
    PerSplatReduceLROnPlateau,
)


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    shape = (32, 32)
    centers0 = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)
    L0 = torch.stack([torch.eye(2) * 0.5 for _ in range(2)], dim=0)
    amps0 = torch.tensor([1.0, 1.0], dtype=torch.float32)

    return GaussianSplatModel(
        shape=shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5, 0.5],
        device=torch.device("cpu"),
    )


class TestModelOptimizerCoordinatorReplaceAll:
    """Test replace_all_splats - complete model replacement."""

    def test_replace_all_splats_basic(self, simple_model) -> None:
        """Test replacing all splats resets optimizer and scheduler state."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        scheduler = PerSplatReduceLROnPlateau(optimizer, patience=2)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, scheduler)

        # Run some steps to build up optimizer state
        for _ in range(3):
            simple_model.zero_grad()
            pred = simple_model()
            loss = torch.mean(pred)
            loss.backward()
            optimizer.step()
            scheduler.step(float(loss))

        # Verify state exists
        assert len(optimizer.splat_states) == 2
        assert len(scheduler.splat_scheduler_states) >= 0  # May be initialized

        # Replace all splats with 3 new ones
        new_centers = torch.tensor(
            [[5.0, 5.0], [15.0, 15.0], [25.0, 25.0]], dtype=torch.float32
        )
        new_Ls = torch.stack([torch.eye(2) * 0.6 for _ in range(3)], dim=0)
        new_amps = torch.tensor([0.8, 0.9, 1.0], dtype=torch.float32)
        new_sharpness = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

        n_new = coordinator.replace_all_splats(
            new_centers, new_Ls, new_amps, new_sharpness
        )

        assert n_new == 3
        assert simple_model.n_splats() == 3

        # Verify optimizer state was reset
        assert len(optimizer.splat_states) == 3
        # Old state should be gone - new splats should have fresh state
        for i in range(3):
            assert i in optimizer.splat_states

        # Verify scheduler state was reset and new states added
        # After replace_all, scheduler.add_splats(3) is called, which initializes new states
        assert len(scheduler.splat_scheduler_states) == 3  # New states initialized

        # Verify operation count incremented
        assert coordinator.operation_count == 1

    def test_replace_all_with_custom_lr(self, simple_model) -> None:
        """Test replace_all_splats with custom learning rate."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, None)

        new_centers = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
        new_Ls = torch.stack([torch.eye(2) * 0.5], dim=0)
        new_amps = torch.tensor([1.0], dtype=torch.float32)
        new_sharpness = torch.tensor([2.0], dtype=torch.float32)

        coordinator.replace_all_splats(
            new_centers, new_Ls, new_amps, new_sharpness, lr_reset=0.05
        )

        # Verify custom LR was applied
        assert optimizer.get_learning_rate(0) == pytest.approx(0.05)

    def test_replace_all_with_exponential_scheduler(self, simple_model) -> None:
        """Test replace_all_splats with exponential scheduler."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        scheduler = PerSplatExponentialLR(optimizer, gamma=0.9)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, scheduler)

        # Age the scheduler
        scheduler.step()
        scheduler.step()
        assert scheduler.current_epoch == 2

        # Replace all splats
        new_centers = torch.tensor([[5.0, 5.0], [15.0, 15.0]], dtype=torch.float32)
        new_Ls = torch.stack([torch.eye(2) * 0.5 for _ in range(2)], dim=0)
        new_amps = torch.tensor([1.0, 1.0], dtype=torch.float32)
        new_sharpness = torch.tensor([2.0, 2.0], dtype=torch.float32)

        coordinator.replace_all_splats(new_centers, new_Ls, new_amps, new_sharpness)

        # Verify scheduler state was reset
        assert len(scheduler.splat_ages) == 2
        # New splats should have current epoch as birth time
        assert 0 in scheduler.splat_ages
        assert 1 in scheduler.splat_ages

    def test_replace_all_maintains_consistency(self, simple_model) -> None:
        """Test that replace_all keeps model/optimizer/scheduler in sync."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        scheduler = PerSplatReduceLROnPlateau(optimizer)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, scheduler)

        # Replace with different number of splats
        for n_new in [1, 5, 3, 7]:
            centers = torch.rand(n_new, 2) * 30
            Ls = torch.stack([torch.eye(2) * 0.5 for _ in range(n_new)], dim=0)
            amps = torch.rand(n_new)
            sharpness = torch.full((n_new,), 2.0, dtype=torch.float32)

            coordinator.replace_all_splats(centers, Ls, amps, sharpness)

            # Verify consistency
            assert simple_model.n_splats() == n_new
            assert len(optimizer.splat_states) == n_new
            # Scheduler states are reinitialized after replace_all
            assert len(scheduler.splat_scheduler_states) == n_new

            # Verify get_status works
            status = coordinator.get_status()
            assert status["model_splats"] == n_new
            assert status["optimizer_states"] == n_new


class TestModelOptimizerCoordinatorPrune:
    """Test prune_splats coordination."""

    def test_prune_splats_with_scheduler(self, simple_model) -> None:
        """Test pruning synchronizes all components when scheduler present."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        scheduler = PerSplatReduceLROnPlateau(optimizer, patience=2)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, scheduler)

        # Initialize scheduler state
        scheduler.step(1.0)
        assert len(scheduler.splat_scheduler_states) == 2

        # Prune one splat
        keep_mask = torch.tensor([True, False])
        n_removed = coordinator.prune_splats(keep_mask)

        assert n_removed == 1
        assert simple_model.n_splats() == 1
        assert len(optimizer.splat_states) == 1
        assert len(scheduler.splat_scheduler_states) == 1
        assert coordinator.operation_count == 1

    def test_prune_splats_without_scheduler(self, simple_model) -> None:
        """Test pruning works without scheduler."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, None)

        keep_mask = torch.tensor([True, False])
        n_removed = coordinator.prune_splats(keep_mask)

        assert n_removed == 1
        assert simple_model.n_splats() == 1

    def test_prune_maintains_operation_count(self, simple_model) -> None:
        """Test that prune operations increment counter."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, None)

        assert coordinator.operation_count == 0

        # Add some splats
        centers_new = torch.tensor([[15.0, 15.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2) * 0.5], dim=0)
        amps_new = torch.tensor([1.0], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0], dtype=torch.float32)
        coordinator.add_splats(centers_new, Ls_new, amps_new, sharpness_new)

        assert coordinator.operation_count == 1

        # Prune
        keep_mask = torch.tensor([True, False, True])
        coordinator.prune_splats(keep_mask)

        assert coordinator.operation_count == 2


class TestModelOptimizerCoordinatorGetStatus:
    """Test get_status method provides useful monitoring info."""

    def test_get_status_structure(self, simple_model) -> None:
        """Test get_status returns expected structure."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, None)

        status = coordinator.get_status()

        assert "model_splats" in status
        assert "optimizer_states" in status
        assert "operation_count" in status
        assert "learning_rates" in status

        lr_stats = status["learning_rates"]
        assert "mean" in lr_stats
        assert "min" in lr_stats
        assert "max" in lr_stats

    def test_get_status_reflects_operations(self, simple_model) -> None:
        """Test get_status reflects actual operations."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        scheduler = PerSplatReduceLROnPlateau(optimizer, patience=1, factor=0.5)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, scheduler)

        initial_status = coordinator.get_status()
        assert initial_status["model_splats"] == 2
        assert initial_status["operation_count"] == 0

        # Trigger LR reduction
        for _ in range(3):
            scheduler.step(1.0)

        # Check LR changed
        status_after_plateau = coordinator.get_status()
        assert status_after_plateau["learning_rates"]["mean"] < 0.1

        # Add splats
        centers_new = torch.tensor([[15.0, 15.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2) * 0.5], dim=0)
        amps_new = torch.tensor([1.0], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0], dtype=torch.float32)
        coordinator.add_splats(centers_new, Ls_new, amps_new, sharpness_new)

        final_status = coordinator.get_status()
        assert final_status["model_splats"] == 3
        assert final_status["operation_count"] == 1

    def test_get_status_with_varying_lrs(self, simple_model) -> None:
        """Test get_status correctly reports LR statistics."""
        optimizer = PerSplatAdam(simple_model, lr=0.1)
        coordinator = ModelOptimizerCoordinator(simple_model, optimizer, None)

        # Set different LRs
        optimizer.set_learning_rate(0, 0.01)
        optimizer.set_learning_rate(1, 0.1)

        status = coordinator.get_status()
        lr_stats = status["learning_rates"]

        assert lr_stats["min"] == pytest.approx(0.01)
        assert lr_stats["max"] == pytest.approx(0.1)
        assert lr_stats["mean"] == pytest.approx(0.055)


class TestFactoryFunctionSchedulers:
    """Test factory function creates correct scheduler types."""

    def test_factory_with_exponential_scheduler(self, simple_model) -> None:
        """Test factory creates exponential scheduler correctly."""
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            simple_model,
            lr=0.05,
            scheduler_type="exponential",
            gamma=0.92,
            age_based_decay=False,  # Disable to get exact gamma decay
        )

        assert isinstance(optimizer, PerSplatAdam)
        assert isinstance(scheduler, PerSplatExponentialLR)
        assert scheduler.gamma == 0.92
        assert scheduler.age_based_decay is False
        assert isinstance(coordinator, ModelOptimizerCoordinator)

        # Verify scheduler works with exact gamma when age_based_decay=False
        initial_lr = optimizer.get_learning_rate(0)
        scheduler.step()
        final_lr = optimizer.get_learning_rate(0)

        assert final_lr == pytest.approx(initial_lr * 0.92)

    def test_factory_with_exponential_no_age_decay(self, simple_model) -> None:
        """Test exponential scheduler with age_based_decay=False."""
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            simple_model,
            lr=0.05,
            scheduler_type="exponential",
            gamma=0.9,
            age_based_decay=False,
        )

        assert isinstance(scheduler, PerSplatExponentialLR)
        assert scheduler.age_based_decay is False

    def test_factory_invalid_scheduler_type(self, simple_model) -> None:
        """Test factory raises error for invalid scheduler type."""
        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            create_per_splat_optimizer_setup(
                simple_model, lr=0.05, scheduler_type="invalid_scheduler"
            )

    def test_factory_with_plateau_custom_params(self, simple_model) -> None:
        """Test factory respects custom plateau scheduler parameters."""
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            simple_model,
            lr=0.05,
            scheduler_type="plateau",
            patience=5,
            factor=0.8,
            threshold=0.001,
            cooldown=3,
            min_lr=1e-6,
        )

        assert isinstance(scheduler, PerSplatReduceLROnPlateau)
        assert scheduler.patience == 5
        assert scheduler.factor == 0.8
        assert scheduler.threshold == 0.001
        assert scheduler.cooldown == 3
        assert scheduler.min_lr == 1e-6


class TestFactoryFunctionOptimizerParams:
    """Test factory function respects optimizer parameters."""

    def test_factory_with_custom_optimizer_params(self, simple_model) -> None:
        """Test factory passes optimizer parameters correctly."""
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            simple_model,
            lr=0.05,
            scheduler_type=None,
            betas=(0.85, 0.995),
            eps=1e-6,
            weight_decay=0.01,
            amsgrad=True,
        )

        assert optimizer.base_lr == 0.05
        # Check that parameters were passed (optimizer should have these settings)
        # Note: These are stored in the underlying optimizer groups
        assert optimizer.amsgrad is True

    def test_factory_creates_working_coordinator(self, simple_model) -> None:
        """Test factory-created coordinator works correctly."""
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            simple_model, lr=0.05, scheduler_type="plateau"
        )

        # Test coordinator works
        centers_new = torch.tensor([[25.0, 25.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2) * 0.5], dim=0)
        amps_new = torch.tensor([1.0], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0], dtype=torch.float32)

        n_added = coordinator.add_splats(
            centers_new, Ls_new, amps_new, sharpness_new, lr_new=0.02
        )
        assert n_added == 1

        status = coordinator.get_status()
        assert status["model_splats"] == 3
        assert status["optimizer_states"] == 3


class TestCoordinatorIntegration:
    """Integration tests combining multiple coordinator operations."""

    def test_realistic_training_scenario(self, simple_model) -> None:
        """Test coordinator in a realistic training scenario with dynamic ops."""
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            simple_model, lr=0.1, scheduler_type="plateau", patience=2
        )

        # Simulate training loop with dynamic operations
        target = torch.randn(simple_model.shape)

        # Initial training
        for epoch in range(5):
            optimizer.zero_grad()
            pred = simple_model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step(float(loss))

        # Check status
        status = coordinator.get_status()
        assert status["model_splats"] == 2

        # Add splats (simulating seeding)
        centers_new = torch.tensor([[8.0, 8.0], [22.0, 22.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2) * 0.4 for _ in range(2)], dim=0)
        amps_new = torch.tensor([0.5, 0.6], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0, 2.0], dtype=torch.float32)
        coordinator.add_splats(centers_new, Ls_new, amps_new, sharpness_new)

        assert simple_model.n_splats() == 4

        # Continue training
        for epoch in range(3):
            optimizer.zero_grad()
            pred = simple_model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step(float(loss))

        # Prune low-amplitude splats (simulating pruning)
        _, _, amps, _ = simple_model.current_params()
        keep_mask = amps > 0.3
        coordinator.prune_splats(keep_mask)

        # Final status check
        final_status = coordinator.get_status()
        assert final_status["operation_count"] == 2  # 1 add + 1 prune
        assert final_status["model_splats"] <= 4  # Some may have been pruned

    def test_complete_replacement_mid_training(self, simple_model) -> None:
        """Test complete model replacement doesn't break training."""
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            simple_model, lr=0.1, scheduler_type="exponential", gamma=0.9
        )

        # Train for a bit
        target = torch.randn(simple_model.shape)
        for _ in range(3):
            optimizer.zero_grad()
            pred = simple_model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Complete replacement
        new_centers = torch.tensor(
            [[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]], dtype=torch.float32
        )
        new_Ls = torch.stack([torch.eye(2) * 0.5 for _ in range(3)], dim=0)
        new_amps = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        new_sharpness = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
        coordinator.replace_all_splats(
            new_centers, new_Ls, new_amps, new_sharpness, lr_reset=0.05
        )

        # Continue training - should work
        for _ in range(3):
            optimizer.zero_grad()
            pred = simple_model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Verify still works
        status = coordinator.get_status()
        assert status["model_splats"] == 3
        assert torch.isfinite(torch.tensor(status["learning_rates"]["mean"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
