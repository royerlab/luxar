"""Regression: progressive fit must honor an explicit float ``seeds`` ratio.

The progressive branch of ``add_gsplats_from_volume`` previously did
``max_splats = seeds if isinstance(seeds, int) else 50000`` — silently
discarding a documented float compression-ratio. These tests pin the
budget-resolution contract (int = exact count, float = ratio, None = default)
by mocking the fitter (to capture ``max_splats``) and the writer (to skip the
scene write) — only the resolution logic in the impl is under test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from luxar.core.group.gsplats_pipeline import from_io
from luxar.gsplats.fitting.preprocessing import _compression_ratio_to_target_count


def _capture_max_splats(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the fitter + writer; return a dict that records max_splats."""
    captured: dict[str, Any] = {}

    def fake_progressive(volume: np.ndarray, *, max_splats: int, **kwargs: Any) -> Any:
        captured["max_splats"] = max_splats
        return object()  # opaque "result"; the writer is mocked too

    import luxar.gsplats as gsplats_pkg

    monkeypatch.setattr(
        gsplats_pkg, "fit_progressive_gaussian_splats", fake_progressive
    )
    monkeypatch.setattr(
        from_io,
        "add_gsplats_from_data_impl",
        lambda *a, **k: "written",  # short-circuit the scene write
    )
    return captured


def test_progressive_float_seeds_resolved_as_compression_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_max_splats(monkeypatch)
    vol = np.zeros((32, 32, 32), dtype=np.float32)

    from_io.add_gsplats_from_volume_impl(
        group=object(), name="g", volume=vol, progressive=True, seeds=0.3
    )

    expected = _compression_ratio_to_target_count(0.3, vol.shape)
    assert captured["max_splats"] == expected
    assert captured["max_splats"] != 50000  # NOT the old silent fallback


def test_progressive_int_seeds_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_max_splats(monkeypatch)
    vol = np.zeros((16, 16, 16), dtype=np.float32)

    from_io.add_gsplats_from_volume_impl(
        group=object(), name="g", volume=vol, progressive=True, seeds=1234
    )

    assert captured["max_splats"] == 1234


def test_progressive_none_seeds_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_max_splats(monkeypatch)
    vol = np.zeros((16, 16, 16), dtype=np.float32)

    from_io.add_gsplats_from_volume_impl(
        group=object(), name="g", volume=vol, progressive=True, seeds=None
    )

    assert captured["max_splats"] == 50000
