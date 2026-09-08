"""Tests for rendering Gaussian splats back to dense volumes."""

from __future__ import annotations

import inspect
import re
import sys
import warnings
from pathlib import Path
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


def test_cuda_render_warning_points_to_external_caller(monkeypatch) -> None:
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

    def render_from_internal_caller():
        return volume_rendering._try_cuda_render(
            centers, cholesky, amplitudes, (4, 4, 4), 3.0, 1e-5
        )

    with pytest.warns(RuntimeWarning) as caught:
        expected_line = inspect.currentframe().f_lineno + 1
        rendered = render_from_internal_caller()

    assert rendered is None
    assert Path(caught[0].filename) == Path(__file__)
    assert caught[0].lineno == expected_line


def test_cuda_render_import_failure_falls_back_quietly(monkeypatch) -> None:
    module_name = "luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda"
    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.setattr(volume_rendering, "_cuda_render_warning_emitted", False)
    centers, cholesky, amplitudes = _cuda_render_inputs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = volume_rendering._try_cuda_render(
            centers, cholesky, amplitudes, (4, 4, 4), 3.0, 1e-5
        )

    assert rendered is None
    assert not caught


def test_cuda_render_skips_unsupported_dimensions_without_warning(monkeypatch) -> None:
    cuda_calls = 0

    class UnexpectedCUDASplatFunction:
        @staticmethod
        def apply(*_args):
            nonlocal cuda_calls
            cuda_calls += 1
            return torch.empty((2,) * 9)

    module_name = "luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            CUDA_BACKEND_AVAILABLE=True,
            CUDASplatFunction=UnexpectedCUDASplatFunction,
        ),
    )
    monkeypatch.setattr(volume_rendering, "_cuda_render_warning_emitted", False)
    centers, cholesky, amplitudes = _cuda_render_inputs(ndim=9)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rendered = volume_rendering._try_cuda_render(
            centers, cholesky, amplitudes, (2,) * 9, 3.0, 1e-5
        )

    assert rendered is None
    assert cuda_calls == 0
    assert not caught


def test_cuda_dimension_bounds_match_extension_header() -> None:
    header = (
        Path(volume_rendering.__file__).parents[1]
        / "models/gsplats/cuda/src/cuda_splatting.h"
    ).read_text()

    min_match = re.search(r"MIN_DIM = (\d+)", header)
    max_match = re.search(r"MAX_SUPPORTED_DIM = (\d+)", header)

    assert min_match is not None
    assert max_match is not None
    assert int(min_match.group(1)) == volume_rendering._CUDA_MIN_DIM
    assert int(max_match.group(1)) == volume_rendering._CUDA_MAX_SUPPORTED_DIM
