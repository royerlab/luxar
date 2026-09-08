"""Tests for rendering Gaussian splats back to dense volumes."""

from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import luxar.gsplats.rendering.volume_rendering as volume_rendering
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor


def _cuda_render_inputs(ndim: int = 3) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((1, ndim), dtype=torch.float32),
        torch.eye(ndim, dtype=torch.float32).unsqueeze(0),
        torch.ones(1, dtype=torch.float32),
    )


def test_render_to_volume_tensor_accepts_auto_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )
    data = GSplatData(
        centers=np.array([[2.0, 2.0]], dtype=np.float32),
        amplitudes=np.array([1.0], dtype=np.float32),
        cholesky_factors=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
    )

    rendered = render_to_volume_tensor(data, shape=(5, 5), device="auto")

    assert rendered.device.type == "cpu"
    assert rendered.shape == (5, 5)
    assert torch.isfinite(rendered).all()
    assert rendered.max() > 0


def test_cuda_render_failure_warns_once_and_falls_back(monkeypatch) -> None:
    class RaisingCUDASplatFunction:
        @staticmethod
        def apply(*_args):
            raise RuntimeError("kernel launch failed")

    module_name = "luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            CUDA_BACKEND_AVAILABLE=True,
            CUDASplatFunction=RaisingCUDASplatFunction,
        ),
    )
    monkeypatch.setattr(volume_rendering, "_cuda_render_warning_emitted", False)
    centers, cholesky, amplitudes = _cuda_render_inputs()

    with pytest.warns(
        RuntimeWarning,
        match="CUDA volume renderer failed with RuntimeError: kernel launch failed",
    ) as caught:
        first = volume_rendering._try_cuda_render(
            centers, cholesky, amplitudes, (4, 4, 4), 3.0, 1e-5
        )
        second = volume_rendering._try_cuda_render(
            centers, cholesky, amplitudes, (4, 4, 4), 3.0, 1e-5
        )

    assert first is None
    assert second is None
    assert len(caught) == 1


def test_cuda_render_import_failure_falls_back_quietly(monkeypatch) -> None:
    module_name = "luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda"
    monkeypatch.setitem(sys.modules, module_name, None)
    centers, cholesky, amplitudes = _cuda_render_inputs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = volume_rendering._try_cuda_render(
            centers, cholesky, amplitudes, (4, 4, 4), 3.0, 1e-5
        )

    assert rendered is None
    assert not caught


def test_cuda_render_skips_unsupported_dimensions_without_warning(monkeypatch) -> None:
    class UnexpectedCUDASplatFunction:
        @staticmethod
        def apply(*_args):
            raise AssertionError("unsupported dimensions must not reach CUDA")

    module_name = "luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            CUDA_BACKEND_AVAILABLE=True,
            CUDASplatFunction=UnexpectedCUDASplatFunction,
        ),
    )
    centers, cholesky, amplitudes = _cuda_render_inputs(ndim=9)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = volume_rendering._try_cuda_render(
            centers, cholesky, amplitudes, (2,) * 9, 3.0, 1e-5
        )

    assert rendered is None
    assert not caught
