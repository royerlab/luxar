"""
Tests for stable inverse softplus function.
"""

import numpy as np
import pytest

from luxar.gsplats.models.utils.inverse_softplus import (
    stable_inverse_softplus,
    stable_inverse_softplus_torch,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestInverseSoftplus:
    """Test stable_inverse_softplus function."""

    def test_inverse_softplus_basic(self) -> None:
        """Test basic inverse softplus functionality."""
        y = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
        x = stable_inverse_softplus(y)

        # Check that softplus(x) ≈ y
        softplus_x = np.log1p(np.exp(x))  # softplus implementation
        np.testing.assert_array_almost_equal(softplus_x, y, decimal=5)

    def test_inverse_softplus_with_beta(self) -> None:
        """Test inverse softplus with different beta values."""
        y = np.array([1.0, 2.0], dtype=np.float32)

        for beta in [0.5, 1.0, 2.0, 5.0]:
            x = stable_inverse_softplus(y, beta=beta)

            # Verify: softplus(x, beta) = y
            softplus_x = np.log1p(np.exp(beta * x)) / beta
            np.testing.assert_array_almost_equal(
                softplus_x, y, decimal=5, err_msg=f"Failed for beta={beta}"
            )

    def test_inverse_softplus_small_values(self) -> None:
        """Test numerical stability for small y values."""
        y = np.array([1e-6, 1e-4, 1e-2, 0.1], dtype=np.float32)
        x = stable_inverse_softplus(y)

        # Should not produce NaN or inf
        assert np.all(np.isfinite(x))

        # Verify inverse relationship
        softplus_x = np.log1p(np.exp(x))
        np.testing.assert_array_almost_equal(softplus_x, y, decimal=5)

    def test_inverse_softplus_large_values(self) -> None:
        """Test for large y values."""
        y = np.array([10.0, 50.0, 100.0], dtype=np.float32)
        x = stable_inverse_softplus(y)

        # Should not produce NaN or inf
        assert np.all(np.isfinite(x))

        # For large y, inverse should be approximately y (since softplus(y) ≈ y for large y)
        np.testing.assert_array_almost_equal(x, y, decimal=1)

    def test_inverse_softplus_single_value(self) -> None:
        """Test with single scalar value."""
        y = 2.5
        x = stable_inverse_softplus(y)

        assert np.isscalar(x) or x.shape == ()

        # Verify inverse
        softplus_x = float(np.log1p(np.exp(x)))
        np.testing.assert_almost_equal(softplus_x, y, decimal=5)

    def test_inverse_softplus_array_shapes(self) -> None:
        """Test with various array shapes."""
        shapes_to_test = [
            (5,),  # 1D
            (3, 4),  # 2D
            (2, 3, 4),  # 3D
            (1, 1, 1, 5),  # 4D
        ]

        for shape in shapes_to_test:
            y = (
                np.random.exponential(1.0, shape).astype(np.float32) + 0.1
            )  # Ensure positive
            x = stable_inverse_softplus(y)

            assert x.shape == shape, f"Shape mismatch for {shape}"

            # Verify inverse relationship
            softplus_x = np.log1p(np.exp(x))
            np.testing.assert_array_almost_equal(softplus_x, y, decimal=4)

    def test_inverse_softplus_dtype_preservation(self) -> None:
        """Test that output dtype matches input dtype."""
        y_float32 = np.array([1.0, 2.0], dtype=np.float32)
        y_float64 = np.array([1.0, 2.0], dtype=np.float64)

        x_32 = stable_inverse_softplus(y_float32)
        x_64 = stable_inverse_softplus(y_float64)

        assert x_32.dtype == np.float32
        assert x_64.dtype == np.float64

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_inverse_softplus_vs_pytorch(self) -> None:
        """Test consistency with PyTorch's inverse softplus when available."""
        y = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        x_ours = stable_inverse_softplus(y)

        # Compare with manual PyTorch implementation
        y_torch = torch.from_numpy(y)
        x_torch = torch.log(torch.expm1(y_torch))  # PyTorch equivalent

        np.testing.assert_array_almost_equal(x_ours, x_torch.numpy(), decimal=5)

    def test_monotonicity(self) -> None:
        """Test that inverse softplus is monotonically increasing."""
        y = np.linspace(0.1, 10.0, 100)
        x = stable_inverse_softplus(y)

        # Check monotonicity: x[i] < x[i+1] for increasing y
        assert np.all(np.diff(x) > 0), (
            "Inverse softplus should be monotonically increasing"
        )

    def test_asymptotic_behavior(self) -> None:
        """Test asymptotic behavior for extreme values."""
        # For very small y, inverse should be approximately log(y)
        y_small = np.array([1e-8, 1e-6, 1e-4])
        x_small = stable_inverse_softplus(y_small)
        log_y_small = np.log(y_small)

        # Should be approximately equal for very small y
        np.testing.assert_array_almost_equal(x_small, log_y_small, decimal=3)

        # For large y, inverse should be approximately y
        y_large = np.array([20.0, 50.0, 100.0])
        x_large = stable_inverse_softplus(y_large)

        np.testing.assert_array_almost_equal(x_large, y_large, decimal=1)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_negative_values_error(self) -> None:
        """Test behavior with negative input values (should still work but may not be meaningful)."""
        # Inverse softplus is only mathematically defined for positive y
        # But numerically, let's see what happens
        y_negative = np.array([-1.0, -0.5])

        # This should produce a warning about non-positive values
        # Our implementation should warn but still compute results (NaN)
        with pytest.warns(RuntimeWarning, match="non-positive values"):
            x = stable_inverse_softplus(y_negative)
            # Should produce NaN for negative inputs
            assert np.any(np.isnan(x))

    def test_zero_input(self) -> None:
        """Test edge case of zero input."""
        y = np.array([0.0])
        x = stable_inverse_softplus(y)

        # Should be -inf since softplus(x) = 0 implies x = -inf
        assert np.isinf(x) and x < 0

    def test_very_small_beta(self) -> None:
        """Test edge case with very small beta."""
        y = np.array([1.0, 2.0])
        x = stable_inverse_softplus(y, beta=1e-10)

        # Should still be finite
        assert np.all(np.isfinite(x))

    def test_large_beta(self) -> None:
        """Test with large beta values."""
        # Use smaller input values to avoid overflow in verification
        y = np.array([0.1, 0.2])  # Smaller values to prevent overflow
        beta = 10.0  # Smaller but still "large" beta

        x = stable_inverse_softplus(y, beta=beta)

        # Should still be finite
        assert np.all(np.isfinite(x))

        # Verify inverse relationship with safe computation
        # For large beta*x, use asymptotic approximation for verification too
        softplus_x = []
        for x_val in x:
            if beta * x_val > 50:
                # Use asymptotic approximation: log1p(exp(z)) ≈ z for large z
                softplus_x.append(x_val)
            else:
                # Use standard formula
                softplus_x.append(np.log1p(np.exp(beta * x_val)) / beta)

        softplus_x = np.array(softplus_x)
        np.testing.assert_array_almost_equal(softplus_x, y, decimal=3)


class TestNumericalStability:
    """Test numerical stability compared to naive implementation."""

    def naive_inverse_softplus(self, y, beta=1.0):  # type: ignore[no-untyped-def]
        """Naive implementation that may be numerically unstable."""
        return np.log(np.exp(beta * y) - 1) / beta

    def test_stability_comparison(self) -> None:
        """Compare stable implementation with naive implementation."""
        # For most values, both should agree
        for y in [0.1, 1.0, 10.0]:
            x_naive = self.naive_inverse_softplus(np.array([y]))
            x_stable_single = stable_inverse_softplus(np.array([y]))

            np.testing.assert_array_almost_equal(x_naive, x_stable_single, decimal=6)

        # For very small values, naive might fail but stable should work
        y_tiny = np.array([1e-10])
        x_stable_tiny = stable_inverse_softplus(y_tiny)
        assert np.isfinite(x_stable_tiny).all()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestTorchNumpyEquivalence:
    """Test that PyTorch and NumPy implementations produce identical results."""

    def test_basic_values_match(self) -> None:
        """Test that basic values produce identical results."""
        y_np = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
        y_torch = torch.from_numpy(y_np)

        x_np = stable_inverse_softplus(y_np)
        x_torch = stable_inverse_softplus_torch(y_torch)

        np.testing.assert_array_almost_equal(
            x_np,
            x_torch.numpy(),
            decimal=5,
            err_msg="NumPy and PyTorch implementations should match for basic values",
        )

    def test_small_values_match(self) -> None:
        """Test numerical stability for small values matches."""
        y_np = np.array([1e-6, 1e-4, 1e-2, 0.1], dtype=np.float32)
        y_torch = torch.from_numpy(y_np)

        x_np = stable_inverse_softplus(y_np)
        x_torch = stable_inverse_softplus_torch(y_torch)

        np.testing.assert_array_almost_equal(
            x_np,
            x_torch.numpy(),
            decimal=5,
            err_msg="NumPy and PyTorch should match for small values",
        )

    def test_large_values_match(self) -> None:
        """Test asymptotic behavior for large values matches."""
        y_np = np.array([50.0, 100.0, 200.0], dtype=np.float32)
        y_torch = torch.from_numpy(y_np)

        x_np = stable_inverse_softplus(y_np)
        x_torch = stable_inverse_softplus_torch(y_torch)

        np.testing.assert_array_almost_equal(
            x_np,
            x_torch.numpy(),
            decimal=5,
            err_msg="NumPy and PyTorch should match for large values (asymptotic)",
        )

    def test_multidimensional_arrays_match(self) -> None:
        """Test that multi-dimensional arrays produce matching results."""
        shapes = [(10,), (5, 5), (3, 4, 5), (2, 3, 4, 5)]

        for shape in shapes:
            y_np = np.random.rand(*shape).astype(np.float32) * 10 + 0.1
            y_torch = torch.from_numpy(y_np)

            x_np = stable_inverse_softplus(y_np)
            x_torch = stable_inverse_softplus_torch(y_torch)

            np.testing.assert_array_almost_equal(
                x_np,
                x_torch.numpy(),
                decimal=5,
                err_msg=f"NumPy and PyTorch should match for shape {shape}",
            )

    def test_different_beta_values_match(self) -> None:
        """Test that different beta values produce matching results."""
        y_np = np.array([1.0, 2.0, 5.0], dtype=np.float32)

        for beta in [0.5, 1.0, 2.0, 5.0]:
            y_torch = torch.from_numpy(y_np)

            x_np = stable_inverse_softplus(y_np, beta=beta)
            x_torch = stable_inverse_softplus_torch(y_torch, beta=beta)

            np.testing.assert_array_almost_equal(
                x_np,
                x_torch.numpy(),
                decimal=5,
                err_msg=f"NumPy and PyTorch should match for beta={beta}",
            )

    def test_gpu_produces_same_as_cpu(self) -> None:
        """Test that GPU computation produces same results as CPU/NumPy."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        y_np = np.random.rand(100).astype(np.float32) * 10 + 0.1
        y_cpu = torch.from_numpy(y_np)
        y_gpu = y_cpu.cuda()

        x_np = stable_inverse_softplus(y_np)
        x_cpu = stable_inverse_softplus_torch(y_cpu)
        x_gpu = stable_inverse_softplus_torch(y_gpu)

        np.testing.assert_array_almost_equal(
            x_np, x_cpu.numpy(), decimal=5, err_msg="NumPy and PyTorch CPU should match"
        )
        np.testing.assert_array_almost_equal(
            x_np,
            x_gpu.cpu().numpy(),
            decimal=5,
            err_msg="NumPy and PyTorch GPU should match",
        )

    def test_inverse_property_torch(self) -> None:
        """Test that softplus(inverse_softplus(y)) = y for PyTorch version."""
        y_np = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
        y_torch = torch.from_numpy(y_np)

        x_torch = stable_inverse_softplus_torch(y_torch)
        y_reconstructed = torch.nn.functional.softplus(x_torch)

        np.testing.assert_array_almost_equal(
            y_np,
            y_reconstructed.numpy(),
            decimal=5,
            err_msg="softplus(inverse_softplus(y)) should equal y",
        )

    def test_device_preservation(self) -> None:
        """Test that output stays on same device as input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        y_cpu = torch.tensor([1.0, 2.0, 3.0])
        y_gpu = y_cpu.cuda()

        x_cpu = stable_inverse_softplus_torch(y_cpu)
        x_gpu = stable_inverse_softplus_torch(y_gpu)

        assert x_cpu.device.type == "cpu", "CPU input should produce CPU output"
        assert x_gpu.device.type == "cuda", "GPU input should produce GPU output"
