"""Tests for the CUDA NLM backend.

Validates correctness against the PyTorch backend (tight tolerance since
both implement the same direct-comparison NLM algorithm).
"""

from __future__ import annotations

import pytest
import torch

from luxar.gsplats.preprocessing import denoise_nlm
from luxar.gsplats.preprocessing.cuda import NLM_CUDA_AVAILABLE

from .conftest import Tolerances, psnr

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not NLM_CUDA_AVAILABLE, reason="NLM CUDA backend not compiled"),
]


class TestCudaBasic:
    """Shape / dtype / device preservation."""

    def test_2d_shape_preserved(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.cuda(), h=0.05, backend="cuda")
        assert result.shape == noisy.shape

    def test_3d_shape_preserved(self, noisy_3d):
        _, noisy = noisy_3d
        result = denoise_nlm(noisy.cuda(), h=0.05, backend="cuda")
        assert result.shape == noisy.shape

    def test_dtype_preserved(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.float().cuda(), h=0.05, backend="cuda")
        assert result.dtype == torch.float32

    def test_cuda_device_preserved(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.cuda(), h=0.05, backend="cuda")
        assert result.device.type == "cuda"

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need multiple GPUs")
    @pytest.mark.parametrize("shape", [(16, 16), (16, 16, 16)])
    def test_non_default_cuda_device(self, shape):
        """NLM should launch on the input device and restore the caller's device."""
        input_tensor = torch.full(shape, 0.5, device="cuda:1")

        with torch.cuda.device(0):
            result = denoise_nlm(input_tensor, h=0.05, backend="cuda")
            assert torch.cuda.current_device() == 0

        assert result.device == input_tensor.device
        torch.testing.assert_close(result, input_tensor, rtol=0, atol=1e-5)


class TestCudaEffectiveness:
    """Denoising should reduce noise."""

    def test_2d_psnr_improvement(self, noisy_2d):
        clean, noisy = noisy_2d
        clean_cuda, noisy_cuda = clean.cuda(), noisy.cuda()
        denoised = denoise_nlm(noisy_cuda, h=0.05, backend="cuda")
        psnr_noisy = psnr(clean_cuda.cpu(), noisy_cuda.cpu())
        psnr_denoised = psnr(clean_cuda.cpu(), denoised.cpu())
        assert psnr_denoised > psnr_noisy + Tolerances.MIN_PSNR_IMPROVEMENT_DB

    def test_3d_psnr_improvement(self, noisy_3d):
        clean, noisy = noisy_3d
        clean_cuda, noisy_cuda = clean.cuda(), noisy.cuda()
        denoised = denoise_nlm(noisy_cuda, h=0.05, backend="cuda")
        psnr_noisy = psnr(clean_cuda.cpu(), noisy_cuda.cpu())
        psnr_denoised = psnr(clean_cuda.cpu(), denoised.cpu())
        assert psnr_denoised > psnr_noisy + Tolerances.MIN_PSNR_IMPROVEMENT_DB

    def test_uniform_input_unchanged(self):
        uniform = torch.full((64, 64), 0.5, device="cuda")
        result = denoise_nlm(uniform, h=0.05, backend="cuda")
        assert torch.allclose(result, uniform, atol=1e-5)


class TestCudaVsPytorch:
    """Cross-validate CUDA backend against PyTorch (same algorithm)."""

    @pytest.mark.parametrize("patch_size", [3, 5])
    @pytest.mark.parametrize("search_distance", [5, 7])
    def test_2d_matches_pytorch(self, noisy_2d, patch_size, search_distance):
        _, noisy = noisy_2d
        noisy_cuda = noisy.cuda()
        kwargs = dict(h=0.05, patch_size=patch_size, search_distance=search_distance)

        ref = denoise_nlm(noisy_cuda, **kwargs, backend="pytorch")
        cuda_result = denoise_nlm(noisy_cuda, **kwargs, backend="cuda")

        assert torch.allclose(
            cuda_result,
            ref,
            atol=Tolerances.CUDA_VS_PYTORCH_ATOL,
        ), (
            f"patch_size={patch_size}, search_distance={search_distance}: "
            f"Max diff: {(cuda_result - ref).abs().max().item():.6f}, "
            f"Mean diff: {(cuda_result - ref).abs().mean().item():.6f}"
        )

    @pytest.mark.parametrize("patch_size", [3, 5])
    @pytest.mark.parametrize("search_distance", [5, 7])
    def test_3d_matches_pytorch(self, noisy_3d, patch_size, search_distance):
        _, noisy = noisy_3d
        noisy_cuda = noisy.cuda()
        kwargs = dict(h=0.05, patch_size=patch_size, search_distance=search_distance)

        ref = denoise_nlm(noisy_cuda, **kwargs, backend="pytorch")
        cuda_result = denoise_nlm(noisy_cuda, **kwargs, backend="cuda")

        assert torch.allclose(
            cuda_result,
            ref,
            atol=Tolerances.CUDA_VS_PYTORCH_ATOL,
        ), (
            f"patch_size={patch_size}, search_distance={search_distance}: "
            f"Max diff: {(cuda_result - ref).abs().max().item():.6f}, "
            f"Mean diff: {(cuda_result - ref).abs().mean().item():.6f}"
        )


class TestCudaParameterValidation:
    """Unsupported parameter combos should warn and fall back to PyTorch."""

    def test_unsupported_patch_size_warns_and_falls_back(self, noisy_2d):
        _, noisy = noisy_2d
        with pytest.warns(UserWarning, match="CUDA NLM"):
            result = denoise_nlm(noisy.cuda(), h=0.05, patch_size=7, backend="cuda")
        # Should still return a valid result via PyTorch fallback
        assert result.shape == noisy.shape

    def test_unsupported_search_distance_warns_and_falls_back(self, noisy_2d):
        _, noisy = noisy_2d
        with pytest.warns(UserWarning, match="CUDA NLM"):
            result = denoise_nlm(
                noisy.cuda(), h=0.05, search_distance=3, backend="cuda"
            )
        # Should still return a valid result via PyTorch fallback
        assert result.shape == noisy.shape
