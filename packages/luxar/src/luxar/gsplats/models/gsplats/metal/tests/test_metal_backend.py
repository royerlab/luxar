"""
Unit tests for Metal backend.

Tests the Metal-accelerated Gaussian splatting implementation.
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
    # Provide dummy for type checking
    GaussianSplatModelMetal = None  # type: ignore[misc, assignment]
    is_metal_available = lambda: False  # noqa: E731


class TestMetalBackendAvailability:
    """Test Metal backend availability checks."""

    def test_metal_available(self):
        """Test that Metal backend is available."""
        assert is_metal_available(), "Metal backend should be available"

    def test_mps_available(self):
        """Test that MPS device is available."""
        assert torch.backends.mps.is_available(), "MPS should be available"

    def test_can_import_model(self):
        """Test that GaussianSplatModelMetal can be imported."""
        assert GaussianSplatModelMetal is not None


class TestMetalForwardPass:
    """Test Metal forward pass."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        shape = (32, 32, 32)
        n_splats = 10

        # Random centers in the middle of the volume
        centers = np.random.rand(n_splats, 3) * 16 + 8
        L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
        amps = np.ones(n_splats, dtype=np.float32)

        model = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            intensity_floor=1e-5,
            device="mps",
        )

        return model, shape

    def test_forward_executes(self, simple_model):
        """Test that forward pass produces a usable tensor.

        Audit W2 fix: previous `output is not None` would pass even if
        the forward returned a degenerate 0-d sentinel. Pin shape +
        dtype + device + finite-ness so a real regression surfaces.
        (test_forward_output_shape / _device follow with their own
        narrower assertions, this one is the smoke check.)
        """
        model, shape = simple_model
        output = model()
        assert isinstance(output, torch.Tensor)
        assert output.shape == shape
        assert output.dtype in (torch.float32, torch.float16, torch.bfloat16)
        assert torch.isfinite(output).all(), (
            "forward() produced NaN/Inf in output tensor"
        )

    def test_forward_output_shape(self, simple_model):
        """Test that output has correct shape."""
        model, shape = simple_model
        output = model()
        assert output.shape == shape

    def test_forward_output_device(self, simple_model):
        """Test that output is on MPS device."""
        model, shape = simple_model
        output = model()
        assert output.device.type == "mps"

    def test_forward_output_dtype(self, simple_model):
        """Test that output has correct dtype."""
        model, shape = simple_model
        output = model()
        assert output.dtype == torch.float32

    def test_forward_output_values(self, simple_model):
        """Test that output values are reasonable."""
        model, shape = simple_model
        output = model()

        # Should be non-negative
        assert output.min() >= 0, "Output should be non-negative"

        # Should have some positive values
        assert output.max() > 0, "Output should have some positive values"

        # Should have some non-zero pixels
        assert (output > 1e-6).sum() > 0, "Should have some non-zero pixels"


class TestMetalBackwardPass:
    """Test Metal backward pass."""

    @pytest.fixture
    def centered_model(self):
        """Create model with centered splats for testing gradients."""
        shape = (16, 16, 16)
        n_splats = 5

        # Place splats in a cross pattern
        centers = np.array(
            [[8, 8, 8], [6, 8, 8], [10, 8, 8], [8, 6, 8], [8, 10, 8]],
            dtype=np.float32,
        )

        L = np.tile(np.eye(3) * 1.5, (n_splats, 1, 1)).astype(np.float32)
        amps = np.ones(n_splats, dtype=np.float32)

        model = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            intensity_floor=1e-5,
            device="mps",
        )

        return model, shape

    def test_backward_executes(self, centered_model):
        """Test that backward pass executes and at least one grad exists.

        Audit W1 fix: previous `assert True` only verified no-crash. Now
        assert at least one parameter actually received a (non-None)
        gradient — the strongest assertion a smoke test can make
        without duplicating test_backward_computes_gradients below.
        """
        model, _shape = centered_model

        output = model()
        loss = output.sum()
        loss.backward()

        # At least one parameter must have a non-None grad attached
        # after backward(). A mutant that silently swallows the
        # backward (e.g. detaches the graph) would fail this check.
        assert any(p.grad is not None for p in model.parameters()), (
            "backward() left every parameter with grad=None"
        )

    def test_backward_computes_gradients(self, centered_model):
        """Test that gradients are computed."""
        model, shape = centered_model

        output = model()
        loss = output.sum()
        loss.backward()

        # Check that at least one parameter has gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.norm() > 0:
                has_grad = True
                break

        assert has_grad, "At least one parameter should have gradients"

    def test_backward_gradient_values(self, centered_model):
        """Test that gradient values are reasonable."""
        model, shape = centered_model

        output = model()
        loss = output.sum()
        loss.backward()

        # Check gradient norms are finite and non-zero
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert np.isfinite(grad_norm), f"{name}: gradient should be finite"
                # Note: Some gradients might be zero if splat doesn't contribute


class TestMetalEdgeCases:
    """Test edge cases and error handling."""

    def test_single_splat(self):
        """Test with a single splat."""
        shape = (16, 16, 16)
        centers = np.array([[8, 8, 8]], dtype=np.float32)
        L = np.array([np.eye(3) * 2.0], dtype=np.float32)
        amps = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="mps",
        )

        output = model()
        assert output.shape == shape
        assert output.max() > 0

    def test_small_volume(self):
        """Test with a small volume."""
        shape = (8, 8, 8)
        centers = np.array([[4, 4, 4]], dtype=np.float32)
        L = np.array([np.eye(3) * 1.0], dtype=np.float32)
        amps = np.array([1.0], dtype=np.float32)

        model = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="mps",
        )

        output = model()
        assert output.shape == shape

    def test_multiple_forward_calls(self):
        """Test that multiple forward calls work."""
        shape = (16, 16, 16)
        centers = np.random.rand(5, 3) * 8 + 4
        L = np.tile(np.eye(3) * 1.5, (5, 1, 1)).astype(np.float32)
        amps = np.ones(5, dtype=np.float32)

        model = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="mps",
        )

        output1 = model()
        output2 = model()

        # Outputs should be identical (no randomness in forward)
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
