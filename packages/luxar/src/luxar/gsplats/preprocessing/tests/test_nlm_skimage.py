"""Tests for the skimage (reference) NLM backend."""

from __future__ import annotations

import pytest
import torch

from luxar.gsplats.preprocessing import denoise_nlm

from .conftest import Tolerances, psnr

# Skip entire file if scikit-image is not installed
pytest.importorskip("skimage", reason="scikit-image not installed")


class TestSkimageBasic:
    """Basic shape / dtype / device preservation."""

    def test_2d_shape_preserved(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy, h=0.05, backend="skimage")
        assert result.shape == noisy.shape

    def test_3d_shape_preserved(self, noisy_3d):
        _, noisy = noisy_3d
        result = denoise_nlm(noisy, h=0.05, backend="skimage")
        assert result.shape == noisy.shape

    def test_dtype_preserved_float32(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.float(), h=0.05, backend="skimage")
        assert result.dtype == torch.float32

    def test_dtype_preserved_float64(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.double(), h=0.05, backend="skimage")
        assert result.dtype == torch.float64

    def test_cpu_device_preserved(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.cpu(), h=0.05, backend="skimage")
        assert result.device.type == "cpu"


class TestSkimageEffectiveness:
    """Denoising should actually reduce noise."""

    def test_2d_psnr_improvement(self, noisy_2d):
        clean, noisy = noisy_2d
        denoised = denoise_nlm(noisy, h=0.05, backend="skimage")
        psnr_noisy = psnr(clean, noisy)
        psnr_denoised = psnr(clean, denoised)
        assert psnr_denoised > psnr_noisy + Tolerances.MIN_PSNR_IMPROVEMENT_DB

    def test_3d_psnr_improvement(self, noisy_3d):
        clean, noisy = noisy_3d
        denoised = denoise_nlm(noisy, h=0.05, backend="skimage")
        psnr_noisy = psnr(clean, noisy)
        psnr_denoised = psnr(clean, denoised)
        assert psnr_denoised > psnr_noisy + Tolerances.MIN_PSNR_IMPROVEMENT_DB

    def test_uniform_input_unchanged(self):
        """A uniform image should not change (no noise to remove)."""
        uniform = torch.full((64, 64), 0.5)
        result = denoise_nlm(uniform, h=0.05, backend="skimage")
        assert torch.allclose(result, uniform, atol=1e-5)


class TestSkimageValidation:
    """Parameter validation."""

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D and 3D"):
            denoise_nlm(torch.randn(100), h=0.05, backend="skimage")

    def test_rejects_4d_input(self):
        with pytest.raises(ValueError, match="2D and 3D"):
            denoise_nlm(torch.randn(2, 8, 8, 8), h=0.05, backend="skimage")

    def test_rejects_even_patch_size(self, noisy_2d):
        _, noisy = noisy_2d
        with pytest.raises(ValueError, match="odd"):
            denoise_nlm(noisy, h=0.05, patch_size=4, backend="skimage")

    def test_rejects_negative_h(self, noisy_2d):
        _, noisy = noisy_2d
        with pytest.raises(ValueError, match="positive"):
            denoise_nlm(noisy, h=-0.05, backend="skimage")

    def test_rejects_zero_search_distance(self, noisy_2d):
        _, noisy = noisy_2d
        with pytest.raises(ValueError, match="search_distance"):
            denoise_nlm(noisy, h=0.05, search_distance=0, backend="skimage")
