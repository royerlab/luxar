"""
Tests for optimizer integration helpers.

Tests create_optimizer_and_scheduler factory function.
"""

import pytest
import torch

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim.integration import create_optimizer_and_scheduler


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

class TestOptimizerCreation:
    """Test optimizer and scheduler creation."""

    def test_creates_adam_optimizer(self, simple_model) -> None:
        """Test factory creates PyTorch Adam optimizer."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model, lr=0.1, scheduler_type="plateau"
        )

        assert isinstance(optimizer, torch.optim.Adam)
        # Verify LR was set (with gradient dilution adjustment)
        assert optimizer.param_groups[0]["lr"] > 0

    def test_creates_plateau_scheduler(self, simple_model) -> None:
        """Test factory creates ReduceLROnPlateau scheduler."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model, lr=0.1, scheduler_type="plateau"
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_creates_exponential_scheduler(self, simple_model) -> None:
        """Test factory creates ExponentialLR scheduler."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model, lr=0.1, scheduler_type="exponential", gamma=0.95
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_no_scheduler(self, simple_model) -> None:
        """Test factory can skip scheduler creation."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model, lr=0.1, scheduler_type=None
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert scheduler is None

    def test_invalid_scheduler_type(self, simple_model) -> None:
        """Test factory raises error for invalid scheduler type."""
        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            create_optimizer_and_scheduler(
                simple_model, lr=0.1, scheduler_type="invalid_scheduler"
            )

class TestSchedulerParameters:
    """Test scheduler parameter passing."""

    def test_plateau_scheduler_params(self, simple_model) -> None:
        """Test factory respects plateau scheduler parameters."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model,
            lr=0.1,
            scheduler_type="plateau",
            patience=5,
            factor=0.8,
            threshold=0.001,
            cooldown=3,
            min_lr=1e-6,
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.patience == 5
        assert scheduler.factor == 0.8
        assert scheduler.threshold == 0.001
        assert scheduler.cooldown == 3
        assert scheduler.min_lrs[0] == 1e-6

    def test_exponential_scheduler_params(self, simple_model) -> None:
        """Test factory respects exponential scheduler parameters."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model,
            lr=0.1,
            scheduler_type="exponential",
            gamma=0.92,
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == 0.92

class TestOptimizerParameters:
    """Test optimizer parameter passing."""

    def test_custom_optimizer_params(self, simple_model) -> None:
        """Test factory passes optimizer parameters correctly."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model,
            lr=0.05,
            scheduler_type=None,
            betas=(0.85, 0.995),
            eps=1e-6,
            weight_decay=0.01,
            amsgrad=True,
        )

        param_group = optimizer.param_groups[0]
        assert param_group["betas"] == (0.85, 0.995)
        assert param_group["eps"] == 1e-6
        assert param_group["weight_decay"] == 0.01
        assert param_group["amsgrad"] is True

class TestGradientDilution:
    """Test gradient dilution compensation."""

    def test_2d_gradient_dilution(self, simple_model) -> None:
        """Test 2D model gets appropriate LR scaling."""
        optimizer, _ = create_optimizer_and_scheduler(
            simple_model, lr=0.1, scheduler_type=None
        )

        # For 2D, gradient dilution factor is 1.0 (no scaling needed)
        actual_lr = optimizer.param_groups[0]["lr"]
        assert actual_lr == 0.1  # Should be unchanged for 2D

    def test_3d_gradient_dilution(self) -> None:
        """Test 3D model gets appropriate LR scaling."""
        # Create 3D model
        shape = (16, 16, 16)
        centers0 = torch.tensor(
            [[8.0, 8.0, 8.0], [10.0, 10.0, 10.0]], dtype=torch.float32
        )
        L0 = torch.stack([torch.eye(3) * 0.5 for _ in range(2)], dim=0)
        amps0 = torch.tensor([1.0, 1.0], dtype=torch.float32)

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5, 0.5],
            device=torch.device("cpu"),
        )

        optimizer, _ = create_optimizer_and_scheduler(
            model, lr=0.1, scheduler_type=None
        )

        # For 3D, should have larger scaling factor than 2D
        actual_lr = optimizer.param_groups[0]["lr"]
        assert actual_lr > 0.1  # Should be scaled up

class TestTrainingIntegration:
    """Test optimizer works in training loop."""

    def test_training_updates_parameters(self, simple_model) -> None:
        """Test optimizer can perform training iterations."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model, lr=0.01, scheduler_type="plateau"
        )

        target = torch.randn(simple_model.shape)
        initial_loss = None

        for _ in range(10):
            optimizer.zero_grad()
            pred = simple_model()
            loss = torch.nn.functional.mse_loss(pred, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        # Verify optimizer updated parameters
        final_pred = simple_model()
        final_loss = torch.nn.functional.mse_loss(final_pred, target).item()
        assert final_loss != initial_loss

    def test_exponential_scheduler_decays_lr(self, simple_model) -> None:
        """Test exponential scheduler decays learning rate."""
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model, lr=0.1, scheduler_type="exponential", gamma=0.9
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler
        scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr == pytest.approx(initial_lr * 0.9)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
