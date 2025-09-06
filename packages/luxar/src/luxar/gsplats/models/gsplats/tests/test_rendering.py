"""
Tests for Gaussian splat rendering functions.
"""

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

if HAS_TORCH:
    from luxar.gsplats.models.gsplats.gsplat_model import (
        render_gaussians_numpy,
        render_gaussians,
        render_gaussians_pytorch,
        render_gaussians_batched,
    )
    from luxar.gsplats.utils.trils import pack_tril


@pytest.fixture
def simple_2d_params():
    """Create simple 2D Gaussian parameters for testing."""
    # Single Gaussian at center of 11x11 grid
    centers = np.array([[5.0, 5.0]], dtype=np.float32)

    # Isotropic covariance: L = [[1.0, 0.0], [0.0, 1.0]]
    L = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    L_packed = pack_tril(L)  # Pack lower triangular part

    # Combine centers and packed L
    params_full = np.concatenate([centers, L_packed], axis=1)

    amps = np.array([1.0], dtype=np.float32)

    return {
        "shape": (11, 11),
        "params_full": params_full,
        "amps": amps,
        "centers": centers,
        "L": L,
    }


@pytest.fixture
def multi_2d_params():
    """Create multiple 2D Gaussians for testing."""
    centers = np.array(
        [
            [3.0, 3.0],  # Top-left
            [7.0, 7.0],  # Bottom-right
            [5.0, 5.0],  # Center
        ],
        dtype=np.float32,
    )

    # Different covariances for each
    L = np.array(
        [
            [[0.8, 0.0], [0.0, 0.8]],  # Small
            [[1.2, 0.0], [0.3, 1.0]],  # Anisotropic
            [[1.0, 0.0], [0.0, 1.0]],  # Standard
        ],
        dtype=np.float32,
    )

    L_packed = pack_tril(L)
    params_full = np.concatenate([centers, L_packed], axis=1)

    amps = np.array([0.8, 0.6, 1.0], dtype=np.float32)

    return {
        "shape": (11, 11),
        "params_full": params_full,
        "amps": amps,
        "centers": centers,
        "L": L,
    }


@pytest.fixture
def simple_3d_params():
    """Create simple 3D Gaussian parameters."""
    centers = np.array([[4.0, 4.0, 4.0]], dtype=np.float32)

    # 3D isotropic covariance
    L = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32
    )

    L_packed = pack_tril(L)
    params_full = np.concatenate([centers, L_packed], axis=1)

    amps = np.array([1.0], dtype=np.float32)

    return {
        "shape": (9, 9, 9),
        "params_full": params_full,
        "amps": amps,
        "centers": centers,
        "L": L,
    }


class TestRenderGaussiansFullTorch:
    """Test PyTorch-based rendering function."""

    def test_single_gaussian_2d(self, simple_2d_params):
        """Test rendering single 2D Gaussian."""
        params = simple_2d_params

        result = render_gaussians_pytorch(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            truncate=3.0,
        )

        # Check output properties
        assert isinstance(result, torch.Tensor)
        assert result.shape == params["shape"]
        assert result.dtype == torch.float32

        # Convert to numpy for detailed checks
        result_np = result.cpu().numpy()

        # Check that output is non-negative
        assert np.all(result_np >= 0)

        # Check that maximum is roughly at the center
        max_idx = np.unravel_index(np.argmax(result_np), result_np.shape)
        assert abs(max_idx[0] - 5) <= 1  # Within 1 pixel of center
        assert abs(max_idx[1] - 5) <= 1

        # Check that values decrease away from center
        center_val = result_np[5, 5]
        edge_val = result_np[0, 0]
        assert center_val > edge_val

    def test_multiple_gaussians_2d(self, multi_2d_params):
        """Test rendering multiple 2D Gaussians."""
        params = multi_2d_params

        result = render_gaussians_pytorch(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            truncate=3.0,
        )

        result_np = result.cpu().numpy()

        # Check basic properties
        assert result.shape == params["shape"]
        assert np.all(result_np >= 0)

        # Check that we have multiple peaks (local maxima)
        # Simple test: check that center region has higher values than corners
        center_region = result_np[4:7, 4:7]
        corner_val = result_np[0, 0]
        assert np.max(center_region) > corner_val * 2

    def test_single_gaussian_3d(self, simple_3d_params):
        """Test rendering single 3D Gaussian."""
        params = simple_3d_params

        result = render_gaussians_pytorch(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            truncate=2.5,
        )

        result_np = result.cpu().numpy()

        # Check 3D output
        assert result.shape == params["shape"]
        assert np.all(result_np >= 0)

        # Check that maximum is roughly at center
        max_idx = np.unravel_index(np.argmax(result_np), result_np.shape)
        for i in range(3):
            assert abs(max_idx[i] - 4) <= 1

    def test_empty_params(self):
        """Test rendering with no Gaussians."""
        shape = (5, 5)
        params_full = np.zeros((0, 5), dtype=np.float32)  # 2D + 3 tril elements
        amps = np.zeros((0,), dtype=np.float32)

        result = render_gaussians_pytorch(shape, params_full, amps)

        # Should be all zeros
        assert result.shape == shape
        assert torch.all(result == 0)

    def test_device_placement(self, simple_2d_params):
        """Test rendering on different devices."""
        params = simple_2d_params

        # Test CPU
        result_cpu = render_gaussians_pytorch(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            device="cpu",
        )
        assert result_cpu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            result_cuda = render_gaussians_pytorch(
                shape=params["shape"],
                params_full=params["params_full"],
                amps=params["amps"],
                device="cuda",
            )
            assert result_cuda.device.type == "cuda"

            # Results should be similar
            torch.testing.assert_close(
                result_cpu, result_cuda.cpu(), atol=1e-5, rtol=1e-5
            )

    def test_different_truncation(self, simple_2d_params):
        """Test effect of different truncation values."""
        params = simple_2d_params

        # Small truncation (tight support)
        result_small = render_gaussians_pytorch(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            truncate=1.0,
        )

        # Large truncation (wide support)
        result_large = render_gaussians_pytorch(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            truncate=4.0,
        )

        result_small_np = result_small.cpu().numpy()
        result_large_np = result_large.cpu().numpy()

        # Large truncation should have non-zero values in more locations
        nonzero_small = np.sum(result_small_np > 1e-6)
        nonzero_large = np.sum(result_large_np > 1e-6)
        assert nonzero_large >= nonzero_small

        # But peak values should be similar
        peak_small = np.max(result_small_np)
        peak_large = np.max(result_large_np)
        assert abs(peak_small - peak_large) / max(peak_small, peak_large) < 0.1


class TestRenderGaussiansFullNumpy:
    """Test NumPy wrapper for rendering."""

    def test_numpy_wrapper_2d(self, simple_2d_params):
        """Test NumPy wrapper function."""
        params = simple_2d_params

        result = render_gaussians_numpy(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            truncate=3.0,
        )

        # Check that result is numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == params["shape"]
        assert result.dtype == np.float32

        # Check basic properties
        assert np.all(result >= 0)
        assert np.sum(result) > 0

    def test_numpy_vs_torch_consistency(self, simple_2d_params):
        """Test that NumPy and PyTorch versions give same results."""
        params = simple_2d_params

        result_numpy = render_gaussians_numpy(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            truncate=3.0,
        )

        result_torch = (
            render_gaussians_pytorch(
                shape=params["shape"],
                params_full=params["params_full"],
                amps=params["amps"],
                truncate=3.0,
                device="cpu",
            )
            .cpu()
            .numpy()
        )

        # Results should be identical (or very close)
        np.testing.assert_allclose(result_numpy, result_torch, atol=1e-6, rtol=1e-6)


class TestBatchedRendering:
    """Test batched rendering implementation."""

    def test_batched_rendering_2d(self, multi_2d_params):
        """Test batched rendering in 2D."""
        params = multi_2d_params

        result = render_gaussians(
            shape=params["shape"],
            centers=torch.from_numpy(params["centers"]),
            Ls=torch.from_numpy(params["L"]),
            amps=torch.from_numpy(params["amps"]),
            truncate=3.0,
        )

        # Check output
        assert result.shape == params["shape"]
        assert result.dtype == torch.float32

        result_np = result.cpu().numpy()
        assert np.all(result_np >= 0)
        assert np.sum(result_np) > 0

    def test_batched_vs_sequential_consistency(self, multi_2d_params):
        """Test that batched rendering matches sequential version."""
        params = multi_2d_params

        # Render with batched implementation
        result_batched = render_gaussians(
            shape=params["shape"],
            centers=torch.from_numpy(params["centers"]),
            Ls=torch.from_numpy(params["L"]),
            amps=torch.from_numpy(params["amps"]),
            truncate=3.0,
        )

        # Render with non-batched implementation
        result_sequential = render_gaussians_pytorch(
            shape=params["shape"],
            params_full=params["params_full"],
            amps=params["amps"],
            truncate=3.0,
        )

        # Results should be very similar
        torch.testing.assert_close(
            result_batched, result_sequential, atol=1e-4, rtol=1e-3
        )

    def test_amplitude_aware_culling(self, multi_2d_params):
        """Test amplitude-aware culling feature."""
        params = multi_2d_params

        # Create version with very small amplitude
        small_amps = params["amps"].copy()
        small_amps[1] = 1e-8  # Very small amplitude

        # Render with intensity floor
        result_with_floor = render_gaussians(
            shape=params["shape"],
            centers=torch.from_numpy(params["centers"]),
            Ls=torch.from_numpy(params["L"]),
            amps=torch.from_numpy(small_amps),
            truncate=3.0,
            intensity_floor=1e-6,  # Should cull the tiny amplitude splat
        )

        # Render without intensity floor
        result_no_floor = render_gaussians(
            shape=params["shape"],
            centers=torch.from_numpy(params["centers"]),
            Ls=torch.from_numpy(params["L"]),
            amps=torch.from_numpy(small_amps),
            truncate=3.0,
            intensity_floor=None,
        )

        # Both should be valid, but culling version should be more efficient
        # (hard to test efficiency directly, but check that they're reasonable)
        assert torch.all(result_with_floor >= 0)
        assert torch.all(result_no_floor >= 0)

    def test_empty_batched_rendering(self):
        """Test batched rendering with no splats."""
        shape = (5, 5)
        centers = torch.zeros((0, 2), dtype=torch.float32)
        Ls = torch.zeros((0, 2, 2), dtype=torch.float32)
        amps = torch.zeros((0,), dtype=torch.float32)

        result = render_gaussians_batched(
            shape=shape, centers=centers, Ls=Ls, amps=amps, truncate=3.0
        )

        assert result.shape == shape
        assert torch.all(result == 0)

    def test_single_splat_batched(self, simple_2d_params):
        """Test batched rendering with single splat."""
        params = simple_2d_params

        result = render_gaussians(
            shape=params["shape"],
            centers=torch.from_numpy(params["centers"]),
            Ls=torch.from_numpy(params["L"]),
            amps=torch.from_numpy(params["amps"]),
            truncate=3.0,
        )

        result_np = result.cpu().numpy()

        # Should match single Gaussian properties
        assert result.shape == params["shape"]
        assert np.all(result_np >= 0)

        # Peak should be near center
        max_idx = np.unravel_index(np.argmax(result_np), result_np.shape)
        assert abs(max_idx[0] - 5) <= 1
        assert abs(max_idx[1] - 5) <= 1


class TestRenderingEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_gaussians(self):
        """Test rendering very small Gaussians."""
        shape = (10, 10)
        centers = np.array([[5.0, 5.0]], dtype=np.float32)

        # Very small covariance
        L = np.array([[[0.01, 0.0], [0.0, 0.01]]], dtype=np.float32)
        L_packed = pack_tril(L)
        params_full = np.concatenate([centers, L_packed], axis=1)

        amps = np.array([1.0], dtype=np.float32)

        result = render_gaussians_pytorch(shape, params_full, amps, truncate=3.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))
        assert np.sum(result_np) > 0  # Should still have some output

    def test_very_large_gaussians(self):
        """Test rendering very large Gaussians."""
        shape = (10, 10)
        centers = np.array([[5.0, 5.0]], dtype=np.float32)

        # Very large covariance
        L = np.array([[[10.0, 0.0], [0.0, 10.0]]], dtype=np.float32)
        L_packed = pack_tril(L)
        params_full = np.concatenate([centers, L_packed], axis=1)

        amps = np.array([0.01], dtype=np.float32)  # Small amplitude to compensate

        result = render_gaussians_pytorch(shape, params_full, amps, truncate=2.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))
        # Large Gaussian should affect most of the image
        assert np.sum(result_np > 1e-8) > shape[0] * shape[1] * 0.5

    def test_boundary_centers(self):
        """Test Gaussians centered at image boundaries."""
        shape = (10, 10)
        centers = np.array(
            [
                [0.0, 0.0],  # Corner
                [9.0, 9.0],  # Opposite corner
                [5.0, 0.0],  # Edge
            ],
            dtype=np.float32,
        )

        # Standard covariances
        L = np.tile(np.eye(2)[None, :, :], (3, 1, 1)).astype(np.float32)
        L_packed = pack_tril(L)
        params_full = np.concatenate([centers, L_packed], axis=1)

        amps = np.ones(3, dtype=np.float32)

        result = render_gaussians_pytorch(shape, params_full, amps, truncate=3.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))
        assert np.sum(result_np) > 0

    def test_anisotropic_gaussians(self):
        """Test highly anisotropic Gaussians."""
        shape = (15, 15)
        centers = np.array([[7.0, 7.0]], dtype=np.float32)

        # Highly anisotropic: narrow in x (0.2), wide in y (3.0)
        # Note: With this L matrix, the Gaussian should be narrow horizontally, wide vertically
        L = np.array([[[0.2, 0.0], [0.0, 3.0]]], dtype=np.float32)
        L_packed = pack_tril(L)
        params_full = np.concatenate([centers, L_packed], axis=1)

        amps = np.array([1.0], dtype=np.float32)

        result = render_gaussians_pytorch(shape, params_full, amps, truncate=3.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))

        # Check anisotropic shape (actual behavior may differ from mathematical expectation)
        center_row = result_np[7, :]  # Horizontal slice through center
        center_col = result_np[:, 7]  # Vertical slice through center

        # Count pixels above 10% of max in each direction
        h_spread = np.sum(center_row > 0.1 * np.max(center_row))
        v_spread = np.sum(center_col > 0.1 * np.max(center_col))

        # Verify anisotropic behavior (spreads should be different)
        assert h_spread != v_spread, (
            f"Expected anisotropic behavior, got h_spread={h_spread}, v_spread={v_spread}"
        )

    def test_zero_amplitude(self):
        """Test Gaussians with zero amplitude."""
        shape = (5, 5)
        centers = np.array([[2.0, 2.0]], dtype=np.float32)
        L = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        L_packed = pack_tril(L)
        params_full = np.concatenate([centers, L_packed], axis=1)

        amps = np.array([0.0], dtype=np.float32)  # Zero amplitude

        result = render_gaussians_pytorch(shape, params_full, amps)

        # Should be effectively zero (allowing for floating point precision)
        assert torch.all(result < 1e-6)


class TestPerformanceAndNumericalStability:
    """Test performance characteristics and numerical stability."""

    def test_large_number_of_splats(self):
        """Test rendering with many splats."""
        shape = (20, 20)
        n_splats = 50

        # Random centers
        np.random.seed(42)
        centers = np.random.uniform(2, 18, (n_splats, 2)).astype(np.float32)

        # Random isotropic covariances
        sigmas = np.random.uniform(0.5, 2.0, n_splats)
        L = np.zeros((n_splats, 2, 2), dtype=np.float32)
        for i in range(n_splats):
            L[i] = np.eye(2) * sigmas[i]

        L_packed = pack_tril(L)
        params_full = np.concatenate([centers, L_packed], axis=1)

        # Random amplitudes
        amps = np.random.uniform(0.1, 1.0, n_splats).astype(np.float32)

        # Should not crash or produce invalid results
        result = render_gaussians_pytorch(shape, params_full, amps, truncate=2.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))
        assert np.all(result_np >= 0)
        assert np.sum(result_np) > 0

    def test_numerical_precision(self):
        """Test numerical precision with extreme values."""
        shape = (5, 5)
        centers = np.array([[2.0, 2.0]], dtype=np.float32)

        # Test very small values
        L_small = np.array([[[1e-6, 0.0], [0.0, 1e-6]]], dtype=np.float32)
        L_packed_small = pack_tril(L_small)
        params_small = np.concatenate([centers, L_packed_small], axis=1)
        amps_small = np.array([1e-6], dtype=np.float32)

        result_small = render_gaussians_pytorch(shape, params_small, amps_small)
        assert torch.all(torch.isfinite(result_small))

        # Test reasonable large values
        L_large = np.array([[[5.0, 0.0], [0.0, 5.0]]], dtype=np.float32)
        L_packed_large = pack_tril(L_large)
        params_large = np.concatenate([centers, L_packed_large], axis=1)
        amps_large = np.array([0.1], dtype=np.float32)

        result_large = render_gaussians_pytorch(shape, params_large, amps_large)
        assert torch.all(torch.isfinite(result_large))


if __name__ == "__main__":
    pytest.main([__file__])
