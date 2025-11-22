"""
Comprehensive tests for per-splat optimizer integration.

This test file focuses on testing the per-splat optimizer in a proper testing framework,
including edge cases and integration with dynamic operations.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import (
    PerSplatAdam,
    PerSplatReduceLROnPlateau,
    create_per_splat_optimizer_setup,
)


class TestPerSplatAdam:
    """Test per-splat Adam optimizer functionality."""

    @staticmethod
    def create_test_model(n_splats=3, device="cpu"):
        """Create a simple test model."""
        shape = (16, 16)
        if n_splats == 0:
            # Handle empty model case
            centers0 = np.zeros((0, 2), dtype=np.float32)
            L0 = np.zeros((0, 2, 2), dtype=np.float32)
            amps0 = np.zeros((0,), dtype=np.float32)
        else:
            centers0 = np.random.uniform(2, 14, (n_splats, 2)).astype(np.float32)
            L0 = np.stack([np.eye(2) * 1.0] * n_splats).astype(np.float32)
            amps0 = np.random.uniform(0.1, 1.0, n_splats).astype(np.float32)

        return GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
            device=torch.device(device),
        )

    def test_initialization(self) -> None:
        """Test optimizer initialization."""
        model = self.create_test_model(5)
        optimizer = PerSplatAdam(model, lr=0.01)

        assert len(optimizer.splat_states) == 5
        assert optimizer.base_lr == 0.01

        # Check that all splats have initialized states
        for i in range(5):
            assert i in optimizer.splat_states
            state = optimizer.splat_states[i]
            assert "step" in state
            assert "exp_avg_mu" in state
            assert "exp_avg_sq_mu" in state

    def test_optimization_step(self) -> None:
        """Test basic optimization step."""
        model = self.create_test_model(3)
        optimizer = PerSplatAdam(model, lr=0.05)

        # Create target
        target = torch.randn(model.shape)

        # Record initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Optimization step
        optimizer.zero_grad()
        pred = model()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        # Parameters should have changed
        current_params = list(model.parameters())
        for initial, current in zip(initial_params, current_params):
            assert not torch.allclose(initial, current, atol=1e-6)

        # Loss should be finite
        assert torch.isfinite(loss).all()

    def test_add_splats(self) -> None:
        """Test adding new splats to the optimizer."""
        model = self.create_test_model(2)
        optimizer = PerSplatAdam(model, lr=0.02)

        # Add splats to model
        centers_new = torch.tensor([[8.0, 8.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2)], dim=0)
        amps_new = torch.tensor([0.5], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0], dtype=torch.float32)

        model.append_(centers_new, Ls_new, amps_new, sharpness_new)
        optimizer.add_splats(1, lr_new=0.1)

        assert len(optimizer.splat_states) == 3
        assert model.n_splats() == 3

        # Check that new splat has different learning rate
        lrs = optimizer.get_effective_learning_rates()
        assert len(lrs) == 3
        assert lrs[2] == 0.1  # New splat with different LR

    def test_remove_splats(self) -> None:
        """Test removing splats from the optimizer."""
        model = self.create_test_model(4)
        optimizer = PerSplatAdam(model, lr=0.02)

        # Prune middle splats (keep 0, 3)
        keep_mask = torch.tensor([True, False, False, True])
        model.prune_(keep_mask)
        optimizer.remove_splats(keep_mask)

        assert len(optimizer.splat_states) == 2
        assert model.n_splats() == 2

        # Check that remaining splats have correct indices
        assert 0 in optimizer.splat_states
        assert 1 in optimizer.splat_states
        assert 2 not in optimizer.splat_states
        assert 3 not in optimizer.splat_states

    def test_momentum_preservation(self) -> None:
        """Test that momentum is preserved when topology changes."""
        model = self.create_test_model(3)
        optimizer = PerSplatAdam(model, lr=0.05)

        # Build momentum with several steps
        target = torch.randn(model.shape)
        for _ in range(5):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

        # Check that splats have momentum
        initial_momentum = []
        for i in range(3):
            state = optimizer.splat_states[i]
            initial_momentum.append(state["exp_avg_mu"].clone())
            assert torch.any(state["exp_avg_mu"].abs() > 1e-8)  # Non-zero momentum

        # Add a new splat
        centers_new = torch.tensor([[12.0, 12.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2)], dim=0)
        amps_new = torch.tensor([0.6], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0], dtype=torch.float32)

        model.append_(centers_new, Ls_new, amps_new, sharpness_new)
        optimizer.add_splats(1, lr_new=0.08)

        # Check momentum is preserved for original splats
        for i in range(3):
            state = optimizer.splat_states[i]
            assert torch.allclose(state["exp_avg_mu"], initial_momentum[i])

        # Check new splat starts with zero momentum
        new_state = optimizer.splat_states[3]
        assert torch.allclose(
            new_state["exp_avg_mu"], torch.zeros_like(new_state["exp_avg_mu"])
        )

    def test_individual_learning_rates(self) -> None:
        """Test setting individual learning rates."""
        model = self.create_test_model(3)
        optimizer = PerSplatAdam(model, lr=0.02)

        # Set different learning rates
        optimizer.set_learning_rate(0, 0.01)
        optimizer.set_learning_rate(1, 0.05)
        optimizer.set_learning_rate(2, 0.001)

        lrs = optimizer.get_effective_learning_rates()
        assert lrs[0] == 0.01
        assert lrs[1] == 0.05
        assert lrs[2] == 0.001

    def test_amsgrad_variant(self) -> None:
        """Test AMSGrad variant."""
        model = self.create_test_model(2)
        optimizer = PerSplatAdam(model, lr=0.02, amsgrad=True)

        # Run a few optimization steps
        target = torch.randn(model.shape)
        for _ in range(3):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

        # Check that AMSGrad states exist
        for i in range(2):
            state = optimizer.splat_states[i]
            assert "max_exp_avg_sq_mu" in state
            assert (
                "max_exp_avg_sq_L_diag" in state
            )  # Separate tracking for diag and off-diagonal
            assert "max_exp_avg_sq_L_off" in state
            assert "max_exp_avg_sq_a" in state


class TestPerSplatScheduler:
    """Test per-splat learning rate scheduler."""

    def test_plateau_scheduler(self) -> None:
        """Test ReduceLROnPlateau scheduler."""
        model = TestPerSplatAdam.create_test_model(2)
        optimizer = PerSplatAdam(model, lr=0.1)
        scheduler = PerSplatReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        initial_lrs = optimizer.get_effective_learning_rates()

        # Simulate plateau (no improvement)
        for _ in range(5):
            scheduler.step(1.0)  # Same loss

        # Learning rates should have been reduced
        final_lrs = optimizer.get_effective_learning_rates()
        assert all(
            final_lr < initial_lr
            for final_lr, initial_lr in zip(final_lrs, initial_lrs)
        )

    def test_scheduler_with_topology_changes(self) -> None:
        """Test scheduler behavior with topology changes."""
        model = TestPerSplatAdam.create_test_model(2)
        optimizer = PerSplatAdam(model, lr=0.1)
        scheduler = PerSplatReduceLROnPlateau(optimizer)

        # Initialize scheduler states for existing splats
        scheduler.step(0.5)  # This will initialize states for existing splats

        # Add splats
        centers_new = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2)], dim=0)
        amps_new = torch.tensor([0.5], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0], dtype=torch.float32)

        model.append_(centers_new, Ls_new, amps_new, sharpness_new)
        optimizer.add_splats(1)
        scheduler.add_splats(1)

        assert len(scheduler.splat_scheduler_states) == 3

        # Remove splats
        keep_mask = torch.tensor([True, False, True])
        model.prune_(keep_mask)
        optimizer.remove_splats(keep_mask)
        scheduler.remove_splats(keep_mask)

        assert len(scheduler.splat_scheduler_states) == 2


class TestFactoryFunction:
    """Test the factory function for creating optimizer setups."""

    def test_create_optimizer_setup(self) -> None:
        """Test factory function creates all components."""
        model = TestPerSplatAdam.create_test_model(3)

        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            model, lr=0.03, scheduler_type="plateau"
        )

        assert isinstance(optimizer, PerSplatAdam)
        assert isinstance(scheduler, PerSplatReduceLROnPlateau)
        assert optimizer.base_lr == 0.03

        # Test coordinator
        status = coordinator.get_status()
        assert status["model_splats"] == 3

    def test_factory_with_different_scheduler(self) -> None:
        """Test factory with different scheduler type."""
        model = TestPerSplatAdam.create_test_model(2)

        # Test with None scheduler
        optimizer, scheduler, coordinator = create_per_splat_optimizer_setup(
            model, lr=0.02, scheduler_type=None
        )

        assert scheduler is None
        assert isinstance(optimizer, PerSplatAdam)

    def test_coordinated_operations(self) -> None:
        """Test coordinated operations through the coordinator."""
        model = TestPerSplatAdam.create_test_model(2)

        _, _, coordinator = create_per_splat_optimizer_setup(model, lr=0.02)

        # Add splats through coordinator
        centers_new = torch.tensor([[6.0, 6.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2) * 0.8], dim=0)
        amps_new = torch.tensor([0.4], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0], dtype=torch.float32)

        n_added = coordinator.add_splats(centers_new, Ls_new, amps_new, sharpness_new, lr_new=0.05)
        assert n_added == 1

        status = coordinator.get_status()
        assert status["model_splats"] == 3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_model(self) -> None:
        """Test optimizer with empty model."""
        model = TestPerSplatAdam.create_test_model(0)
        optimizer = PerSplatAdam(model, lr=0.01)

        assert len(optimizer.splat_states) == 0

        # Should handle zero_grad and step gracefully
        optimizer.zero_grad()
        optimizer.step()

    def test_large_model_performance(self) -> None:
        """Test with larger model to check performance."""
        model = TestPerSplatAdam.create_test_model(50)
        optimizer = PerSplatAdam(model, lr=0.01)

        # Should handle large model without issues
        target = torch.randn(model.shape)

        optimizer.zero_grad()
        pred = model()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_invalid_learning_rates(self) -> None:
        """Test handling of invalid learning rates."""
        model = TestPerSplatAdam.create_test_model(2)
        optimizer = PerSplatAdam(model, lr=0.01)

        # Test negative learning rate
        with pytest.raises(ValueError):
            optimizer.set_learning_rate(0, -0.01)

        # Test very small learning rate (should be valid)
        optimizer.set_learning_rate(0, 1e-8)
        lrs = optimizer.get_effective_learning_rates()
        assert lrs[0] == 1e-8  # Should match what we set

    def test_gradient_dilution_compensation(self):
        """Test gradient dilution compensation for different dimensions."""
        # 2D model
        model_2d = TestPerSplatAdam.create_test_model(3)
        optimizer_2d = PerSplatAdam(model_2d, lr=0.01)

        # 2D: params = 5, factor = 5/5 = 1.0
        expected_2d = 0.01 * 1.0
        assert abs(optimizer_2d.effective_lr - expected_2d) < 1e-6

        # 3D model
        shape_3d = (16, 16, 16)
        centers_3d = np.random.uniform(2, 14, (3, 3)).astype(np.float32)
        L0_3d = np.stack([np.eye(3)] * 3).astype(np.float32)
        amps_3d = np.ones(3, dtype=np.float32)

        model_3d = GaussianSplatModel(
            shape=shape_3d,
            centers0=centers_3d,
            L0=L0_3d,
            amps0=amps_3d,
            sigma_min_diag=[0.5, 0.5, 0.5],
        )
        optimizer_3d = PerSplatAdam(model_3d, lr=0.01)

        # 3D: params = 9, factor = 9/5 = 1.8
        expected_3d = 0.01 * 1.8
        assert abs(optimizer_3d.effective_lr - expected_3d) < 1e-6

    def test_amsgrad_max_tracking(self):
        """Test that AMSGrad properly tracks max_exp_avg_sq."""
        model = TestPerSplatAdam.create_test_model(2)
        optimizer = PerSplatAdam(model, lr=0.02, amsgrad=True)

        target = torch.randn(model.shape)

        # Run first step
        optimizer.zero_grad()
        pred = model()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        # Store first max values
        first_max = optimizer.splat_states[0]["max_exp_avg_sq_mu"].clone()

        # Run more steps
        for _ in range(3):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

        # Verify max values are monotonically non-decreasing
        final_max = optimizer.splat_states[0]["max_exp_avg_sq_mu"]
        assert torch.all(final_max >= first_max), (
            "AMSGrad max_exp_avg_sq should be monotonically non-decreasing"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
