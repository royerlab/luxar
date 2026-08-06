"""
Tests for Gaussian splat rendering functions.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock
from unittest.mock import PropertyMock

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
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.models.gsplats import (
        render_gaussians,
        render_gaussians_numpy,
        render_gaussians_pytorch,
    )
    from luxar.gsplats.utils.trils import pack_tril

    @contextlib.contextmanager
    def _force_mps_device():
        """Make every tensor report device.type == 'mps' (keeping real CPU
        storage) and turn a `.to(<that fake device>)` into a no-op, so the
        MPS-only CPU fallback branch in group_by_box / group_by_box_gpu
        executes on a CPU host. A fallback body that does nothing (e.g.
        `pass`) leaves uniq/inv unbound and raises here — which is exactly
        the mutation these tests must catch."""
        fake_device = SimpleNamespace(type="mps")
        orig_to = torch.Tensor.to

        def fake_to(self, *args, **kwargs):
            # Swallow the move back to the fake mps device in either the
            # positional (`.to(device)`) or keyword (`.to(device=...)`) form.
            if (args and args[0] is fake_device) or kwargs.get("device") is fake_device:
                return self
            return orig_to(self, *args, **kwargs)

        with (
            mock.patch.object(torch.Tensor, "device", new_callable=PropertyMock) as dev,
            mock.patch.object(torch.Tensor, "to", fake_to),
        ):
            dev.return_value = fake_device
            yield


@pytest.fixture(autouse=True)
def clear_rendering_grid_cache(monkeypatch):
    """Keep rendering grid-cache tests isolated from process/global state."""
    from luxar.gsplats.models.gsplats import rendering_core

    for name in (
        "LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES",
        "LUXAR_GSPLAT_GRID_CACHE_MAX_GB",
        "LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_BYTES",
        "LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_GB",
    ):
        monkeypatch.delenv(name, raising=False)
    rendering_core.clear_grid_cache()
    yield
    rendering_core.clear_grid_cache()


@pytest.fixture
def simple_2d_params():
    """Create simple 2D Gaussian parameters for testing."""
    # Single Gaussian at center of 11x11 grid
    centers = np.array([[5.0, 5.0]], dtype=np.float32)

    # Isotropic covariance: L = [[1.0, 0.0], [0.0, 1.0]]
    L = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    L_packed = pack_tril(L)  # Pack lower triangular part

    # Combine centers and packed L (for backward compat tests)
    params_full = np.concatenate([centers, L_packed], axis=1)

    amps = np.array([1.0], dtype=np.float32)

    # Create result object
    result = GSplatData(
        centers=centers,
        amplitudes=amps,
        cholesky_factors=L_packed,
        stats={},
    )

    return {
        "shape": (11, 11),
        "params_full": params_full,
        "amps": amps,
        "centers": centers,
        "L": L,
        "result": result,
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

    # Create result object
    result = GSplatData(
        centers=centers,
        amplitudes=amps,
        cholesky_factors=L_packed,
        stats={},
    )

    return {
        "shape": (11, 11),
        "params_full": params_full,
        "amps": amps,
        "centers": centers,
        "L": L,
        "result": result,
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

    # Create result object
    result = GSplatData(
        centers=centers,
        amplitudes=amps,
        cholesky_factors=L_packed,
        stats={},
    )

    return {
        "shape": (9, 9, 9),
        "params_full": params_full,
        "amps": amps,
        "centers": centers,
        "L": L,
        "result": result,
    }


class TestGridCacheMemoryPolicy:
    """Test adaptive byte-budgeted grid-cache behavior."""

    def test_grid_cache_can_be_disabled_with_zero_budget(self, monkeypatch) -> None:
        from luxar.gsplats.models.gsplats import rendering_core

        monkeypatch.setenv("LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES", "0")
        strides = torch.tensor([4, 1], dtype=torch.long)

        base, lin_offsets = rendering_core.cached_base_and_offsets(
            (2, 2), strides, torch.device("cpu")
        )

        assert base.shape == (2, 4)
        assert lin_offsets.shape == (4,)
        assert rendering_core.get_grid_cache_stats()["devices"] == {}

    def test_oversized_entry_is_returned_but_not_cached(self, monkeypatch) -> None:
        from luxar.gsplats.models.gsplats import rendering_core

        # A 2D (2, 2) float32 grid has base=32 bytes and int64 offsets=32 bytes.
        monkeypatch.setenv("LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES", "1024")
        monkeypatch.setenv("LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_BYTES", "63")
        strides = torch.tensor([4, 1], dtype=torch.long)

        base, lin_offsets = rendering_core.cached_base_and_offsets(
            (2, 2), strides, torch.device("cpu")
        )

        assert rendering_core._grid_entry_nbytes(base, lin_offsets) == 64
        assert rendering_core.get_grid_cache_stats()["devices"] == {}

    def test_grid_cache_evicts_lru_entries_by_device_byte_budget(
        self, monkeypatch
    ) -> None:
        from luxar.gsplats.models.gsplats import rendering_core

        monkeypatch.setenv("LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES", "160")
        monkeypatch.setenv("LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_BYTES", "160")
        device = torch.device("cpu")
        strides = torch.tensor([10, 1], dtype=torch.long)

        rendering_core.cached_base_and_offsets((2, 2), strides, device)  # 64 bytes
        rendering_core.cached_base_and_offsets((3, 2), strides, device)  # 96 bytes
        rendering_core.cached_base_and_offsets((2, 2), strides, device)  # Refresh LRU
        rendering_core.cached_base_and_offsets((2, 3), strides, device)  # Evicts (3, 2)

        stats = rendering_core.get_grid_cache_stats()["devices"]["cpu"]
        assert stats["entries"] == 2
        assert stats["bytes"] == 160

        cache = rendering_core._GRID_CACHE[("cpu", None)]
        cached_shapes = {key[2] for key in cache}
        assert cached_shapes == {(2, 2), (2, 3)}

    def test_clear_grid_cache_resets_entries_and_byte_accounting(
        self, monkeypatch
    ) -> None:
        from luxar.gsplats.models.gsplats import rendering_core

        monkeypatch.setenv("LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES", "1024")
        monkeypatch.setenv("LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_BYTES", "1024")
        strides = torch.tensor([4, 1], dtype=torch.long)

        rendering_core.cached_base_and_offsets((2, 2), strides, torch.device("cpu"))
        assert rendering_core.get_grid_cache_stats()["devices"]["cpu"]["bytes"] == 64

        rendering_core.clear_grid_cache()

        assert rendering_core.get_grid_cache_stats()["devices"] == {}
        assert rendering_core._GRID_CACHE == {}
        assert rendering_core._GRID_CACHE_BYTES == {}


class TestRenderGaussiansFullTorch:
    """Test PyTorch-based rendering function."""

    def test_single_gaussian_2d(self, simple_2d_params) -> None:
        """Test rendering single 2D Gaussian."""
        params = simple_2d_params

        result = render_gaussians_pytorch(
            shape=params["shape"],
            result=params["result"],
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

    def test_multiple_gaussians_2d(self, multi_2d_params) -> None:
        """Test rendering multiple 2D Gaussians."""
        params = multi_2d_params

        result = render_gaussians_pytorch(
            shape=params["shape"],
            result=params["result"],
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

    def test_single_gaussian_3d(self, simple_3d_params) -> None:
        """Test rendering single 3D Gaussian."""
        params = simple_3d_params

        result = render_gaussians_pytorch(
            shape=params["shape"],
            result=params["result"],
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

    def test_empty_params(self) -> None:
        """Test rendering with no Gaussians."""
        shape = (5, 5)

        # Create empty result object
        empty_result = GSplatData(
            centers=np.zeros((0, 2), dtype=np.float32),
            amplitudes=np.zeros((0,), dtype=np.float32),
            cholesky_factors=np.zeros(
                (0, 3), dtype=np.float32
            ),  # 2D has 3 tril elements
            stats={},
        )

        result = render_gaussians_pytorch(shape, empty_result)

        # Should be all zeros
        assert result.shape == shape
        assert torch.all(result == 0)

    def test_device_placement(self, simple_2d_params) -> None:
        """Test rendering on different devices."""
        params = simple_2d_params

        # Test CPU
        result_cpu = render_gaussians_pytorch(
            shape=params["shape"],
            result=params["result"],
            device="cpu",
        )
        assert result_cpu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            result_cuda = render_gaussians_pytorch(
                shape=params["shape"],
                result=params["result"],
                device="cuda",
            )
            assert result_cuda.device.type == "cuda"

            # Results should be similar
            torch.testing.assert_close(
                result_cpu, result_cuda.cpu(), atol=1e-5, rtol=1e-5
            )

    def test_different_truncation(self, simple_2d_params) -> None:
        """Test effect of different truncation values."""
        params = simple_2d_params

        # Small truncation (tight support)
        result_small = render_gaussians_pytorch(
            shape=params["shape"],
            result=params["result"],
            truncate=1.0,
        )

        # Large truncation (wide support)
        result_large = render_gaussians_pytorch(
            shape=params["shape"],
            result=params["result"],
            truncate=4.0,
        )

        result_small_np = result_small.cpu().numpy()
        result_large_np = result_large.cpu().numpy()

        # Large truncation should have non-zero values in more locations
        nonzero_small: int = int(np.sum(result_small_np > 1e-6))
        nonzero_large: int = int(np.sum(result_large_np > 1e-6))
        assert nonzero_large >= nonzero_small

        # But peak values should be similar
        peak_small: float = float(np.max(result_small_np))
        peak_large: float = float(np.max(result_large_np))
        assert abs(peak_small - peak_large) / max(peak_small, peak_large) < 0.1


class TestRenderGaussiansFullNumpy:
    """Test NumPy wrapper for rendering."""

    def test_numpy_wrapper_2d(self, simple_2d_params) -> None:
        """Test NumPy wrapper function."""
        params = simple_2d_params

        result = render_gaussians_numpy(
            shape=params["shape"],
            result=params["result"],
            truncate=3.0,
        )

        # Check that result is numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == params["shape"]
        assert result.dtype == np.float32

        # Check basic properties
        assert np.all(result >= 0)
        assert np.sum(result) > 0

    def test_numpy_vs_torch_consistency(self, simple_2d_params) -> None:
        """Test that NumPy and PyTorch versions give same results."""
        params = simple_2d_params

        result_numpy = render_gaussians_numpy(
            shape=params["shape"],
            result=params["result"],
            truncate=3.0,
        )

        result_torch = (
            render_gaussians_pytorch(
                shape=params["shape"],
                result=params["result"],
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

    def test_batched_rendering_2d(self, multi_2d_params) -> None:
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

    def test_batched_vs_sequential_consistency(self, multi_2d_params) -> None:
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

        # Render with wrapper (uses GSplatData)
        result_sequential = render_gaussians_pytorch(
            shape=params["shape"],
            result=params["result"],
            truncate=3.0,
        )

        # Results should be very similar
        torch.testing.assert_close(
            result_batched, result_sequential, atol=1e-4, rtol=1e-3
        )

    def test_amplitude_aware_culling(self, multi_2d_params) -> None:
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

    def test_empty_input_renders_zeros(self) -> None:
        """render_gaussians with zero splats returns an all-zero tensor."""
        shape = (5, 5)
        centers = torch.zeros((0, 2), dtype=torch.float32)
        Ls = torch.zeros((0, 2, 2), dtype=torch.float32)
        amps = torch.zeros((0,), dtype=torch.float32)

        result = render_gaussians(
            shape=shape,
            centers=centers,
            Ls=Ls,
            amps=amps,
            truncate=3.0,
        )

        assert result.shape == shape
        assert torch.all(result == 0)

    def test_single_splat_batched(self, simple_2d_params) -> None:
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

    def test_very_small_gaussians(self) -> None:
        """Test rendering very small Gaussians."""
        shape = (10, 10)
        centers = np.array([[5.0, 5.0]], dtype=np.float32)

        # Very small covariance
        L = np.array([[[0.01, 0.0], [0.0, 0.01]]], dtype=np.float32)
        L_packed = pack_tril(L)

        amps = np.array([1.0], dtype=np.float32)

        test_result = GSplatData(
            centers=centers,
            amplitudes=amps,
            cholesky_factors=L_packed,
            stats={},
        )

        result = render_gaussians_pytorch(shape, test_result, truncate=3.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))
        assert np.sum(result_np) > 0  # Should still have some output

    def test_very_large_gaussians(self) -> None:
        """Test rendering very large Gaussians."""
        shape = (10, 10)
        centers = np.array([[5.0, 5.0]], dtype=np.float32)

        # Very large covariance
        L = np.array([[[10.0, 0.0], [0.0, 10.0]]], dtype=np.float32)
        L_packed = pack_tril(L)

        amps = np.array([0.01], dtype=np.float32)  # Small amplitude to compensate

        test_result = GSplatData(
            centers=centers,
            amplitudes=amps,
            cholesky_factors=L_packed,
            stats={},
        )

        result = render_gaussians_pytorch(shape, test_result, truncate=2.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))
        # Large Gaussian should affect most of the image
        assert np.sum(result_np > 1e-8) > shape[0] * shape[1] * 0.5

    def test_boundary_centers(self) -> None:
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

        amps = np.ones(3, dtype=np.float32)

        test_result = GSplatData(
            centers=centers,
            amplitudes=amps,
            cholesky_factors=L_packed,
            stats={},
        )

        result = render_gaussians_pytorch(shape, test_result, truncate=3.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))
        assert np.sum(result_np) > 0

    def test_anisotropic_gaussians(self) -> None:
        """Test highly anisotropic Gaussians."""
        shape = (15, 15)
        centers = np.array([[7.0, 7.0]], dtype=np.float32)

        # Highly anisotropic: narrow in x (0.2), wide in y (3.0)
        # Note: With this L matrix, the Gaussian should be narrow horizontally, wide vertically
        L = np.array([[[0.2, 0.0], [0.0, 3.0]]], dtype=np.float32)
        L_packed = pack_tril(L)

        amps = np.array([1.0], dtype=np.float32)

        test_result = GSplatData(
            centers=centers,
            amplitudes=amps,
            cholesky_factors=L_packed,
            stats={},
        )

        result = render_gaussians_pytorch(shape, test_result, truncate=3.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))

        # Check anisotropic shape (actual behavior may differ from mathematical expectation)
        center_row = result_np[7, :]  # Horizontal slice through center
        center_col = result_np[:, 7]  # Vertical slice through center

        # Count pixels above 10% of max in each direction
        h_spread: int = int(np.sum(center_row > 0.1 * np.max(center_row)))
        v_spread: int = int(np.sum(center_col > 0.1 * np.max(center_col)))

        # Verify anisotropic behavior (spreads should be different)
        assert h_spread != v_spread, (
            f"Expected anisotropic behavior, got h_spread={h_spread}, v_spread={v_spread}"
        )

    def test_zero_amplitude(self) -> None:
        """Test Gaussians with zero amplitude."""
        shape = (5, 5)
        centers = np.array([[2.0, 2.0]], dtype=np.float32)
        L = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        L_packed = pack_tril(L)

        amps = np.array([0.0], dtype=np.float32)  # Zero amplitude

        test_result = GSplatData(
            centers=centers,
            amplitudes=amps,
            cholesky_factors=L_packed,
            stats={},
        )

        result = render_gaussians_pytorch(shape, test_result)

        # Should be effectively zero (allowing for floating point precision)
        assert torch.all(result < 1e-6)


class TestPerformanceAndNumericalStability:
    """Test performance characteristics and numerical stability."""

    def test_large_number_of_splats(self) -> None:
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

        # Random amplitudes
        amps = np.random.uniform(0.1, 1.0, n_splats).astype(np.float32)

        test_result = GSplatData(
            centers=centers,
            amplitudes=amps,
            cholesky_factors=L_packed,
            stats={},
        )

        # Should not crash or produce invalid results
        result = render_gaussians_pytorch(shape, test_result, truncate=2.0)

        result_np = result.cpu().numpy()
        assert np.all(np.isfinite(result_np))
        assert np.all(result_np >= 0)
        assert np.sum(result_np) > 0

    def test_numerical_precision(self) -> None:
        """Test numerical precision with extreme values."""
        shape = (5, 5)
        centers = np.array([[2.0, 2.0]], dtype=np.float32)

        # Test very small values
        L_small = np.array([[[1e-6, 0.0], [0.0, 1e-6]]], dtype=np.float32)
        L_packed_small = pack_tril(L_small)
        amps_small = np.array([1e-6], dtype=np.float32)

        test_result_small = GSplatData(
            centers=centers,
            amplitudes=amps_small,
            cholesky_factors=L_packed_small,
            stats={},
        )

        result_small = render_gaussians_pytorch(shape, test_result_small)
        assert torch.all(torch.isfinite(result_small))

        # Test reasonable large values
        L_large = np.array([[[5.0, 0.0], [0.0, 5.0]]], dtype=np.float32)
        L_packed_large = pack_tril(L_large)
        amps_large = np.array([0.1], dtype=np.float32)

        test_result_large = GSplatData(
            centers=centers,
            amplitudes=amps_large,
            cholesky_factors=L_packed_large,
            stats={},
        )

        result_large = render_gaussians_pytorch(shape, test_result_large)
        assert torch.all(torch.isfinite(result_large))


class TestRenderingWrappersEdgeCases:
    """Test edge cases in rendering wrapper functions."""

    def test_numpy_wrapper_empty_params(self) -> None:
        """Test numpy wrapper with empty params."""
        shape = (10, 10)

        # Create empty result object
        empty_result = GSplatData(
            centers=np.zeros((0, 2), dtype=np.float32),
            amplitudes=np.zeros((0,), dtype=np.float32),
            cholesky_factors=np.zeros(
                (0, 3), dtype=np.float32
            ),  # 2D has 3 tril elements
            stats={},
        )

        result = render_gaussians_numpy(shape, empty_result)

        assert result.shape == shape
        assert np.all(result == 0)

    def test_numpy_wrapper_with_standard_result(self) -> None:
        """Test numpy wrapper with standard GSplatData."""
        shape = (10, 10)
        centers = np.array([[5.0, 5.0]], dtype=np.float32)
        L = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        L_packed = pack_tril(L)

        amps = np.ones(1, dtype=np.float32)

        test_result = GSplatData(
            centers=centers,
            amplitudes=amps,
            cholesky_factors=L_packed,
            stats={},
        )

        result = render_gaussians_numpy(shape, test_result)

        # Should render correctly
        assert result.shape == shape
        assert np.all(np.isfinite(result))
        assert np.sum(result) > 0

    def test_pytorch_wrapper_with_standard_result(self) -> None:
        """Test pytorch wrapper with standard GSplatData."""
        shape = (10, 10)
        centers = np.array([[5.0, 5.0]], dtype=np.float32)
        L = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        L_packed = pack_tril(L)

        amps = np.ones(1, dtype=np.float32)

        test_result = GSplatData(
            centers=centers,
            amplitudes=amps,
            cholesky_factors=L_packed,
            stats={},
        )

        result = render_gaussians_pytorch(shape, test_result)

        # Should render correctly
        assert result.shape == shape
        assert torch.all(torch.isfinite(result))
        assert torch.sum(result) > 0


class TestMPSFallbackHandling:
    """Test MPS fallback handling for torch.unique operations."""

    def test_group_by_box_cpu(self) -> None:
        """Test group_by_box works on CPU."""
        from luxar.gsplats.models.gsplats.rendering_core import group_by_box

        # Create test data on CPU with different box shapes
        # Box 0: (0,0) to (5,5) -> shape (5,5)
        # Box 1: (10,10) to (17,13) -> shape (7,3)
        # Box 2: (0,0) to (5,5) -> shape (5,5) - same as Box 0
        lo = torch.tensor([[0, 0], [10, 10], [0, 0]], dtype=torch.long)
        hi = torch.tensor([[5, 5], [17, 13], [5, 5]], dtype=torch.long)

        groups = group_by_box(lo, hi)

        # Should group identical box shapes together
        assert len(groups) == 2  # Two unique shapes: (5,5) and (7,3)
        assert (5, 5) in groups
        assert (7, 3) in groups
        # Indices 0 and 2 have same box shape (5,5)
        assert len(groups[(5, 5)]) == 2

    def test_group_by_box_gpu_cpu(self) -> None:
        """Test group_by_box_gpu works on CPU."""
        from luxar.gsplats.models.gsplats.rendering_core import group_by_box_gpu

        lo = torch.tensor([[0, 0], [10, 10], [0, 0]], dtype=torch.long)
        hi = torch.tensor([[5, 5], [17, 13], [5, 5]], dtype=torch.long)

        uniq, inv = group_by_box_gpu(lo, hi)

        # Should find 2 unique sizes: (5,5) and (7,3)
        assert uniq.shape[0] == 2
        # inv should map back to unique sizes
        assert inv.shape[0] == 3

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_group_by_box_mps(self) -> None:
        """Test group_by_box MPS fallback path."""
        from luxar.gsplats.models.gsplats.rendering_core import group_by_box

        # Create test data on MPS with different box shapes
        lo = torch.tensor([[0, 0], [10, 10], [0, 0]], dtype=torch.long, device="mps")
        hi = torch.tensor([[5, 5], [17, 13], [5, 5]], dtype=torch.long, device="mps")

        groups = group_by_box(lo, hi)

        # Should still work correctly on MPS with CPU fallback
        assert len(groups) == 2
        assert (5, 5) in groups
        assert (7, 3) in groups

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_group_by_box_gpu_mps(self) -> None:
        """Test group_by_box_gpu MPS fallback path."""
        from luxar.gsplats.models.gsplats.rendering_core import group_by_box_gpu

        lo = torch.tensor([[0, 0], [10, 10], [0, 0]], dtype=torch.long, device="mps")
        hi = torch.tensor([[5, 5], [17, 13], [5, 5]], dtype=torch.long, device="mps")

        uniq, inv = group_by_box_gpu(lo, hi)

        # Results should be on MPS device
        assert uniq.device.type == "mps"
        assert inv.device.type == "mps"
        assert uniq.shape[0] == 2

    def test_group_by_box_mps_fallback_cpu(self) -> None:
        """Drive the MPS CPU-fallback branch of group_by_box on a CPU host.

        `_force_mps_device()` makes every tensor report device.type == 'mps'
        while keeping real CPU storage, so the `if sizes.device.type == 'mps'`
        branch runs and produces the same groupings as the plain CPU path. A
        no-op fallback body (e.g. `pass`) would leave uniq/inv unbound and
        raise UnboundLocalError, so this test genuinely exercises the branch.
        """
        from luxar.gsplats.models.gsplats.rendering_core import group_by_box

        lo = torch.tensor([[0, 0], [10, 10], [0, 0]], dtype=torch.long)
        hi = torch.tensor([[5, 5], [17, 13], [5, 5]], dtype=torch.long)

        with _force_mps_device():
            groups = group_by_box(lo, hi)

        # Exact groupings, member indices included — keying by box shape makes
        # this independent of the order torch.unique returns the sizes in.
        assert {key: idx.tolist() for key, idx in groups.items()} == {
            (5, 5): [0, 2],
            (7, 3): [1],
        }

    def test_group_by_box_gpu_mps_fallback_cpu(self) -> None:
        """Drive the MPS CPU-fallback branch of group_by_box_gpu on a CPU host.

        Same mechanism as ``test_group_by_box_mps_fallback_cpu``: the fake mps
        device forces the fallback branch, and a no-op body would raise
        UnboundLocalError instead of returning the unique sizes / inverse map.
        """
        from luxar.gsplats.models.gsplats.rendering_core import group_by_box_gpu

        lo = torch.tensor([[0, 0], [10, 10], [0, 0]], dtype=torch.long)
        hi = torch.tensor([[5, 5], [17, 13], [5, 5]], dtype=torch.long)

        with _force_mps_device():
            uniq, inv = group_by_box_gpu(lo, hi)

        # Exact unique sizes, and `inv` must index them back to each input's
        # own box shape (order-independent, so no reliance on unique's sort).
        sizes = uniq.tolist()
        assert sorted(sizes) == [[5, 5], [7, 3]]
        assert [sizes[g] for g in inv.tolist()] == [[5, 5], [7, 3], [5, 5]]


class TestMPSPeakFindingFallback:
    """Test MPS fallback handling for max_pool3d in peak finding."""

    def test_mps_fallback_code_path_exists_global(self) -> None:
        """Verify MPS fallback exists in _find_residual_peaks_global."""
        import inspect

        from luxar.gsplats.fitting.dynamic_ops.peak_finding import (
            _find_residual_peaks_global,
        )

        source = inspect.getsource(_find_residual_peaks_global)
        assert 'device.type == "mps"' in source, (
            "_find_residual_peaks_global should have MPS fallback for max_pool3d"
        )

    def test_mps_fallback_code_path_exists_tiled(self) -> None:
        """Verify MPS fallback exists in _find_peaks_in_tile."""
        import inspect

        from luxar.gsplats.fitting.dynamic_ops.peak_finding import _find_peaks_in_tile

        source = inspect.getsource(_find_peaks_in_tile)
        assert 'device.type == "mps"' in source, (
            "_find_peaks_in_tile should have MPS fallback for max_pool3d"
        )

    def test_global_peak_finding_cpu_3d(self) -> None:
        """Test global peak finding works on CPU for 3D."""
        from luxar.gsplats.fitting.dynamic_ops.peak_finding import (
            _find_residual_peaks_global,
        )

        # Create 3D residual with clear peaks
        residual = torch.zeros(16, 16, 16, dtype=torch.float32)
        residual[5, 5, 5] = 1.0
        residual[10, 10, 10] = 0.8

        peaks = _find_residual_peaks_global(
            residual, k_max_residuals=2, nms_radius_vox=2.0
        )

        assert len(peaks) == 2
        assert (5, 5, 5) in {tuple(p.tolist()) for p in peaks}

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_global_peak_finding_mps_3d(self) -> None:
        """Test global peak finding works on MPS for 3D with CPU fallback."""
        from luxar.gsplats.fitting.dynamic_ops.peak_finding import (
            _find_residual_peaks_global,
        )

        # Create 3D residual on MPS
        residual = torch.zeros(16, 16, 16, dtype=torch.float32, device="mps")
        residual[5, 5, 5] = 1.0
        residual[10, 10, 10] = 0.8

        # Should work without error due to CPU fallback
        peaks = _find_residual_peaks_global(
            residual, k_max_residuals=2, nms_radius_vox=2.0
        )

        assert len(peaks) == 2
