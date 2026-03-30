"""Tests for backend auto-detection and dispatch logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from luxar.gsplats.preprocessing.nlm_core import _resolve_backend, denoise_nlm


class TestResolveBackend:
    """Unit tests for _resolve_backend."""

    def test_explicit_backend_returned_as_is(self):
        device = torch.device("cpu")
        assert _resolve_backend("skimage", device) == "skimage"
        assert _resolve_backend("pytorch", device) == "pytorch"
        assert _resolve_backend("cuda", device) == "cuda"

    def test_auto_cpu_selects_skimage(self):
        assert _resolve_backend("auto", torch.device("cpu")) == "skimage"

    def test_auto_cuda_selects_pytorch_when_no_cuda_ext(self):
        """Without compiled CUDA extension, auto on CUDA falls back to pytorch."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # Mock the CUDA extension as not available
        with patch.dict("sys.modules", {"luxar.gsplats.preprocessing.cuda": None}):
            result = _resolve_backend("auto", torch.device("cuda"))
            assert result == "pytorch"


class TestDispatchIntegration:
    """Integration tests: verify backend kwarg routes correctly."""

    def test_skimage_backend_on_cpu(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.cpu(), h=0.05, backend="skimage")
        assert result.device.type == "cpu"
        assert result.shape == noisy.shape

    def test_pytorch_backend_on_cpu(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.cpu(), h=0.05, backend="pytorch")
        assert result.device.type == "cpu"
        assert result.shape == noisy.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pytorch_backend_on_cuda(self, noisy_2d):
        _, noisy = noisy_2d
        result = denoise_nlm(noisy.cuda(), h=0.05, backend="pytorch")
        assert result.device.type == "cuda"

    def test_unknown_backend_raises(self, noisy_2d):
        _, noisy = noisy_2d
        with pytest.raises(ValueError, match="Unknown backend"):
            denoise_nlm(noisy, h=0.05, backend="unknown_backend")

    def test_device_kwarg_moves_tensor(self, noisy_2d):
        """The device= kwarg should move the tensor before processing."""
        _, noisy = noisy_2d
        # Process on CPU explicitly
        result = denoise_nlm(noisy, h=0.05, backend="pytorch", device="cpu")
        assert result.device.type == "cpu"

    def test_rejects_non_tensor_input(self):
        import numpy as np

        with pytest.raises(TypeError, match="torch.Tensor"):
            denoise_nlm(np.zeros((10, 10)), h=0.05)
