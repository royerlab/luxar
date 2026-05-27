"""
Unit tests for optimizer compatibility with Metal backend.

Ensures GaussianSplatModelMetal works with Luxar's optimizers.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

# Skip entire module on non-macOS platforms
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not torch.backends.mps.is_available(),
    reason="Metal backend only available on macOS with MPS",
)

# Import Metal-specific modules only on macOS
if sys.platform == "darwin":
    from luxar.gsplats.models.gsplats.metal import (
        GaussianSplatModelMetal,
        is_metal_available,
    )
else:
    # Provide dummies for type checking
    GaussianSplatModelMetal = None  # type: ignore[misc, assignment]
    is_metal_available = lambda: False  # noqa: E731


class TestOptimizerCompatibility:
    """Test that Metal model works with Luxar optimizers."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple Metal model for testing."""
        np.random.seed(42)
        return GaussianSplatModelMetal(
            shape=(16, 16, 16),
            centers0=np.random.rand(5, 3) * 8 + 4,
            L0=np.tile(np.eye(3) * 1.5, (5, 1, 1)).astype(np.float32),
            amps0=np.ones(5, dtype=np.float32),
            sigma_min_diag=[0.5, 0.5, 0.5],
            device="mps",
        )

    def test_model_has_shape_property(self, simple_model):
        """Test that model.shape is accessible."""
        assert hasattr(simple_model, "shape")
        assert simple_model.shape == (16, 16, 16)

    def test_model_has_dim_property(self, simple_model):
        """Test that model.dim is accessible."""
        assert hasattr(simple_model, "dim")
        assert simple_model.dim == 3

    def test_model_has_truncate_property(self, simple_model):
        """Test that model.truncate is accessible."""
        assert hasattr(simple_model, "truncate")
        assert simple_model.truncate == 3.0

    def test_model_has_internal_parameters(self, simple_model):
        """Test that internal parameters are accessible (required by optimizer)."""
        assert hasattr(simple_model, "raw_mu")
        assert hasattr(simple_model, "raw_L_diag")
        assert hasattr(simple_model, "L_off")
        assert hasattr(simple_model, "raw_a")

        # Verify shapes
        assert simple_model.raw_mu.shape == (5, 3)
        assert simple_model.L_off.shape == (5, 3)

    def test_works_with_torch_adam(self, simple_model):
        """Test that model works with standard torch.optim.Adam.

        Audit W4 fix: previous `assert True` only verified no-crash.
        Now assert that the optimizer step actually advanced at least
        one parameter (the strongest behavioural assertion a no-crash
        smoke test can make).
        """
        # Snapshot the parameters BEFORE the optimizer step.
        params_before = {
            name: p.detach().clone() for name, p in simple_model.named_parameters()
        }

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.05)

        # Training iteration
        output = simple_model()
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # At least one parameter must have changed — a no-op optimizer
        # would leave every param identical to its pre-step value.
        moved = [
            name
            for name, p in simple_model.named_parameters()
            if not torch.equal(p.detach(), params_before[name])
        ]
        assert moved, "No parameter advanced after optimizer.step()"

    def test_works_with_luxar_optimizer(self, simple_model):
        """Test that model works with Luxar's create_optimizer_and_scheduler.

        Audit W4 fix: replace `assert True` with same parameter-moved
        behavioural assertion as the torch.optim.Adam test above.
        """
        from luxar.gsplats.optim import create_optimizer_and_scheduler

        params_before = {
            name: p.detach().clone() for name, p in simple_model.named_parameters()
        }

        # Create optimizer using Luxar's factory function
        optimizer, scheduler = create_optimizer_and_scheduler(
            simple_model, lr=0.05, scheduler_type=None
        )

        # Training iteration
        output = simple_model()
        loss = output.sum()
        loss.backward()
        optimizer.step()
        simple_model.zero_grad()

        moved = [
            name
            for name, p in simple_model.named_parameters()
            if not torch.equal(p.detach(), params_before[name])
        ]
        assert moved, "No parameter advanced via Luxar optimizer step"

    def test_multiple_training_steps(self, simple_model):
        """Test multiple training iterations (ensures state is maintained)."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.05)

        losses = []
        for _ in range(5):
            output = simple_model()
            loss = output.sum()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            simple_model.zero_grad()

        # Losses should be finite
        assert all(np.isfinite(loss_val) for loss_val in losses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
