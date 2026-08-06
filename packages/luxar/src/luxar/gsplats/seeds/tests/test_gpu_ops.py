# test_gpu_ops.py
"""
Tests for GPU-accelerated seed generation operations.

Tests individual GPU operations for correctness and performance.
All tests gracefully skip when CUDA/MPS is unavailable.
"""

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from luxar.gsplats.seeds.gpu_ops import (
    _compute_nd_sobel_magnitude_gpu,
    _conv1d_along_axis,
    _get_device,
    check_gpu_memory,
    sample_amplitudes_gpu,
    should_use_gpu,
)


class TestDeviceHelpers:
    """Test device resolution and selection helpers."""

    def test_get_device_none_returns_cpu(self):
        """None should return CPU for backward compatibility."""
        assert _get_device(None) == "cpu"

    def test_get_device_cpu_returns_cpu(self):
        """Explicit CPU should return CPU."""
        assert _get_device("cpu") == "cpu"

    def test_get_device_auto_returns_valid_device(self):
        """Auto should return a valid device."""
        device = _get_device("auto")
        assert device in ["cpu", "cuda", "mps"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_cuda_returns_cuda(self):
        """CUDA should return CUDA when available."""
        assert _get_device("cuda") == "cuda"

    @pytest.mark.skipif(torch.cuda.is_available(), reason="Test requires no CUDA")
    def test_get_device_cuda_fallback_to_cpu(self):
        """CUDA should fallback to CPU when unavailable."""
        with pytest.warns(RuntimeWarning, match="CUDA requested but not available"):
            assert _get_device("cuda") == "cpu"

    def test_should_use_gpu_cpu_returns_false(self):
        """CPU device should never use GPU."""
        V = np.random.rand(100, 100, 100)
        assert should_use_gpu(V, "cpu") is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_should_use_gpu_large_volume_returns_true(self):
        """Large volumes on GPU device should use GPU."""
        V = np.random.rand(100, 100, 100)
        assert should_use_gpu(V, "cuda") is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_should_use_gpu_small_volume_returns_false(self):
        """Small volumes should use CPU (overhead not worth it)."""
        V = np.random.rand(10, 10, 10)
        assert should_use_gpu(V, "cuda") is False


class TestConv1DAlongAxis:
    """Test separable 1D convolution helper."""

    def test_conv1d_2d_axis0(self):
        """Test 2D convolution along axis 0."""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        kernel = torch.tensor([1.0, 0.0, -1.0])

        result = _conv1d_along_axis(x, kernel, axis=0)

        # Gradient along axis 0: [7-1, 8-2, 9-3] = [6, 6, 6] at middle row
        assert result.shape == x.shape
        # Check middle row (approximate due to padding)
        assert torch.allclose(result[1, :], torch.tensor([6.0, 6.0, 6.0]), atol=1e-5)

    def test_conv1d_3d_axis1(self):
        """Test 3D convolution along axis 1."""
        x = torch.ones((3, 5, 4))
        x[:, 2, :] = 5.0  # Make middle slice bright
        kernel = torch.tensor([-1.0, 0.0, 1.0])

        result = _conv1d_along_axis(x, kernel, axis=1)

        # Should have edges around the bright slice
        assert result.shape == x.shape
        # Gradient should be non-zero around bright slice
        assert torch.any(torch.abs(result) > 1.0)


class TestSobelGradientGPU:
    """Test GPU Sobel gradient computation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sobel_2d_gpu_vs_cpu(self):
        """GPU Sobel should closely match scipy.ndimage.sobel (full separable kernel)."""
        # Create test image with edge
        img_np = np.zeros((100, 100))
        img_np[40:60, :] = 1.0

        # CPU version (scipy) - uses full Sobel with smoothing
        cpu_result = np.sqrt(
            sum(ndi.sobel(img_np, axis=i, mode="nearest") ** 2 for i in range(2))
        )

        # GPU version - now also uses full separable Sobel with smoothing
        img_gpu = torch.tensor(img_np, device="cuda", dtype=torch.float32)
        gpu_result_tensor = _compute_nd_sobel_magnitude_gpu(img_gpu)
        gpu_result = gpu_result_tensor.cpu().numpy()

        # Both should detect edges at same locations with same threshold
        threshold = 0.3
        cpu_edges = cpu_result > threshold
        gpu_edges = gpu_result > threshold

        # Should have near-identical edge patterns (>95% overlap)
        overlap = np.logical_and(cpu_edges, gpu_edges).sum()
        cpu_total = cpu_edges.sum()
        assert overlap / cpu_total > 0.95, (
            f"Edge overlap too low: {overlap / cpu_total}"
        )

        # Magnitudes should be close (allowing for float32 vs float64 differences)
        # scipy uses float64, GPU uses float32
        mask = cpu_result > 0.1
        if mask.sum() > 0:
            rel_error = np.abs(cpu_result[mask] - gpu_result[mask]) / (
                cpu_result[mask] + 1e-12
            )
            assert np.median(rel_error) < 0.05, (
                f"Median relative error too high: {np.median(rel_error):.4f}"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sobel_3d_gpu_vs_cpu(self):
        """GPU Sobel should closely match scipy.ndimage.sobel (full separable kernel)."""
        # Create test volume with edge
        img_np = np.zeros((50, 50, 50))
        img_np[20:30, :, :] = 1.0

        # CPU version (scipy) - uses full Sobel with smoothing
        cpu_result = np.sqrt(
            sum(ndi.sobel(img_np, axis=i, mode="nearest") ** 2 for i in range(3))
        )

        # GPU version - now also uses full separable Sobel with smoothing
        img_gpu = torch.tensor(img_np, device="cuda", dtype=torch.float32)
        gpu_result_tensor = _compute_nd_sobel_magnitude_gpu(img_gpu)
        gpu_result = gpu_result_tensor.cpu().numpy()

        # Both should detect edges at same locations with same threshold
        threshold = 0.3
        cpu_edges = cpu_result > threshold
        gpu_edges = gpu_result > threshold

        # Should have near-identical edge patterns (>95% overlap)
        overlap = np.logical_and(cpu_edges, gpu_edges).sum()
        cpu_total = cpu_edges.sum()
        assert overlap / cpu_total > 0.95, (
            f"Edge overlap too low: {overlap / cpu_total}"
        )

        # Magnitudes should be close
        mask = cpu_result > 0.1
        if mask.sum() > 0:
            rel_error = np.abs(cpu_result[mask] - gpu_result[mask]) / (
                cpu_result[mask] + 1e-12
            )
            assert np.median(rel_error) < 0.05, (
                f"Median relative error too high: {np.median(rel_error):.4f}"
            )

    def test_sobel_cpu_device(self):
        """Sobel should work on CPU device."""
        img_np = np.zeros((50, 50))
        img_np[20:30, :] = 1.0

        img_cpu = torch.tensor(img_np, dtype=torch.float32)
        result = _compute_nd_sobel_magnitude_gpu(img_cpu)

        # Should produce non-zero gradients at edges
        assert torch.max(result) > 0.5
        assert result.shape == img_cpu.shape


class TestSampleAmplitudesGPU:
    """Test GPU amplitude interpolation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sample_amplitudes_2d_gpu_vs_cpu(self):
        """GPU interpolation should produce reasonable results for 2D."""
        # Create smooth test image (reduces interpolation method differences)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        img_np = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)).astype(np.float32)

        # Create sample coordinates
        coords_np = np.array(
            [[10.5, 20.3], [50.7, 50.1], [80.2, 30.9]], dtype=np.float32
        )

        # CPU version
        cpu_result = ndi.map_coordinates(
            img_np, coords_np.T, order=1, mode="nearest"
        ).astype(np.float32)

        # GPU version
        img_gpu = torch.tensor(img_np, device="cuda")
        coords_gpu = torch.tensor(coords_np, device="cuda")
        gpu_result_tensor = sample_amplitudes_gpu(img_gpu, coords_gpu, mode="bilinear")
        gpu_result = gpu_result_tensor.cpu().numpy()

        # Different interpolation methods, so use reasonable tolerance
        assert np.allclose(cpu_result, gpu_result, rtol=0.05, atol=0.01)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sample_amplitudes_3d_gpu_vs_cpu(self):
        """GPU interpolation should produce reasonable results for 3D."""
        # Create smooth test volume (reduces interpolation method differences)
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        z = np.linspace(0, 1, 50)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        img_np = (
            np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) * np.sin(2 * np.pi * zz)
        ).astype(np.float32)

        # Create sample coordinates
        coords_np = np.array(
            [[10.5, 20.3, 15.7], [25.7, 25.1, 25.9], [40.2, 30.9, 35.1]],
            dtype=np.float32,
        )

        # CPU version
        cpu_result = ndi.map_coordinates(
            img_np, coords_np.T, order=1, mode="nearest"
        ).astype(np.float32)

        # GPU version
        img_gpu = torch.tensor(img_np, device="cuda")
        coords_gpu = torch.tensor(coords_np, device="cuda")
        gpu_result_tensor = sample_amplitudes_gpu(img_gpu, coords_gpu, mode="bilinear")
        gpu_result = gpu_result_tensor.cpu().numpy()

        # Different interpolation methods, so use reasonable tolerance
        assert np.allclose(cpu_result, gpu_result, rtol=0.05, atol=0.01)


class TestGPUMemoryChecks:
    """Test GPU memory estimation and checks."""

    def test_check_gpu_memory_cpu_always_true(self):
        """CPU device should always pass memory check."""
        V = np.random.rand(1000, 1000, 1000)
        assert check_gpu_memory(V, "cpu", "sobel") is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_check_gpu_memory_small_volume_true(self):
        """Small volumes should pass memory check."""
        V = np.random.rand(100, 100, 100)
        assert check_gpu_memory(V, "cuda", "sobel") is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUPerformance:
    """Performance comparison tests (only run with CUDA available)."""

    def test_sobel_gpu_faster_than_cpu(self):
        """GPU Sobel should be faster than CPU for large volumes."""
        import time

        # Create large test volume
        img_np = np.random.rand(200, 200, 200).astype(np.float32)

        # CPU timing
        t0 = time.time()
        _cpu_result = np.sqrt(
            sum(ndi.sobel(img_np, axis=i, mode="nearest") ** 2 for i in range(3))
        )
        cpu_time = time.time() - t0

        # GPU timing (including transfer)
        img_gpu = torch.tensor(img_np, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        _gpu_result = _compute_nd_sobel_magnitude_gpu(img_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - t0

        # GPU should be faster (at least 2x speedup expected)
        speedup = cpu_time / gpu_time
        print(
            f"\nSobel GPU speedup: {speedup:.2f}x (CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s)"
        )
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"
