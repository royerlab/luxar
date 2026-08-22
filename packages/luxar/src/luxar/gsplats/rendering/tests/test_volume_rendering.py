"""Tests for rendering Gaussian splats back to dense volumes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor


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
