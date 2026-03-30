"""Tests for the PyTorch NLM backend.

Validates correctness against the skimage reference, plus shape/dtype/device
preservation and GPU acceleration.
"""

from __future__ import annotations

import pytest
import torch

from luxar.gsplats.preprocessing import denoise_nlm

from .conftest import Tolerances, psnr

# Some cross-validation tests need skimage as reference
_HAS_SKIMAGE = True
try:
    import skimage  # noqa: F401
except ImportError:
    _HAS_SKIMAGE = False


class TestPytorchBasic:
    """Shape / dtype / device preservation."""

    def test_2d_shape_preserved(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy, h=0.05, backend="pytorch")
        assert result.shape == noisy.shape

    def test_3d_shape_preserved(self, noisy_3d):
        _, noisy = noisy_3d
        result = denoise_nlm(noisy, h=0.05, backend="pytorch")
        assert result.shape == noisy.shape

    def test_dtype_preserved_float32(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.float(), h=0.05, backend="pytorch")
        assert result.dtype == torch.float32

    def test_dtype_preserved_float64(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.double(), h=0.05, backend="pytorch")
        assert result.dtype == torch.float64

    def test_cpu_device_preserved(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.cpu(), h=0.05, backend="pytorch")
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_preserved(self, noisy_2d):
        _, noisy = noisy_2d
        noisy_cuda = noisy.cuda()
        result = denoise_nlm(noisy_cuda, h=0.05, backend="pytorch")
        assert result.device.type == "cuda"


class TestPytorchEffectiveness:
    """Denoising should reduce noise."""

    def test_2d_psnr_improvement(self, noisy_2d):
        clean, noisy = noisy_2d
        denoised = denoise_nlm(noisy, h=0.05, backend="pytorch")
        psnr_noisy = psnr(clean, noisy)
        psnr_denoised = psnr(clean, denoised)
        assert psnr_denoised > psnr_noisy + Tolerances.MIN_PSNR_IMPROVEMENT_DB

    def test_3d_psnr_improvement(self, noisy_3d):
        clean, noisy = noisy_3d
        denoised = denoise_nlm(noisy, h=0.05, backend="pytorch")
        psnr_noisy = psnr(clean, noisy)
        psnr_denoised = psnr(clean, denoised)
        assert psnr_denoised > psnr_noisy + Tolerances.MIN_PSNR_IMPROVEMENT_DB

    def test_uniform_input_unchanged(self):
        uniform = torch.full((64, 64), 0.5)
        result = denoise_nlm(uniform, h=0.05, backend="pytorch")
        assert torch.allclose(result, uniform, atol=1e-5)


class TestPytorchVsSkimage:
    """Cross-validate PyTorch backend against skimage reference."""

    @pytest.mark.skipif(not _HAS_SKIMAGE, reason="scikit-image not installed")
    def test_2d_close_to_skimage(self, noisy_2d):
        _, noisy = noisy_2d
        # Use small search_distance for faster test
        kwargs = dict(h=0.05, patch_size=3, search_distance=3)
        ref = denoise_nlm(noisy, **kwargs, backend="skimage")
        pt = denoise_nlm(noisy, **kwargs, backend="pytorch")
        assert torch.allclose(
            pt,
            ref,
            atol=Tolerances.PYTORCH_VS_SKIMAGE_ATOL,
            rtol=Tolerances.PYTORCH_VS_SKIMAGE_RTOL,
        ), (
            f"Max diff: {(pt - ref).abs().max().item():.6f}, "
            f"Mean diff: {(pt - ref).abs().mean().item():.6f}"
        )

    @pytest.mark.skipif(not _HAS_SKIMAGE, reason="scikit-image not installed")
    def test_3d_close_to_skimage(self, noisy_3d):
        _, noisy = noisy_3d
        kwargs = dict(h=0.05, patch_size=3, search_distance=2)
        ref = denoise_nlm(noisy, **kwargs, backend="skimage")
        pt = denoise_nlm(noisy, **kwargs, backend="pytorch")
        assert torch.allclose(
            pt,
            ref,
            atol=Tolerances.PYTORCH_VS_SKIMAGE_ATOL,
            rtol=Tolerances.PYTORCH_VS_SKIMAGE_RTOL,
        ), (
            f"Max diff: {(pt - ref).abs().max().item():.6f}, "
            f"Mean diff: {(pt - ref).abs().mean().item():.6f}"
        )


class TestPytorchChunked:
    """Chunked processing for large 3D volumes."""

    def test_chunked_matches_full(self, noisy_3d):
        """Chunked result should closely match non-chunked result."""
        _, noisy = noisy_3d
        kwargs = dict(h=0.05, patch_size=3, search_distance=2)
        full = denoise_nlm(noisy, **kwargs, backend="pytorch")
        chunked = denoise_nlm(noisy, **kwargs, backend="pytorch", chunk_size=8)
        assert torch.allclose(full, chunked, atol=1e-5), (
            f"Max diff: {(full - chunked).abs().max().item():.6f}"
        )

    def test_chunked_shape_preserved(self, noisy_3d):
        _, noisy = noisy_3d
        result = denoise_nlm(noisy, h=0.05, backend="pytorch", chunk_size=8)
        assert result.shape == noisy.shape
