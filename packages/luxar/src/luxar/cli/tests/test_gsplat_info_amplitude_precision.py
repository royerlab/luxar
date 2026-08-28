"""Precision regression tests for ``luxar gsplat info`` amplitude analysis."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.cli.gsplat_ops import inspect_commands
from luxar.cli.gsplat_ops.inspect_commands import (
    _info_report,
    _normalized_amplitude_cdf,
)
from luxar.gsplats.gsplat_data import GSplatData


def test_amplitude_cdf_uses_float64_at_production_scale() -> None:
    n_splats = 2_000_000
    rng = np.random.default_rng(0)
    amplitudes = (
        (rng.random(n_splats).astype(np.float32) ** 3) * 1500.0 + 0.002
    ).astype(np.float32)

    actual = _normalized_amplitude_cdf(amplitudes)
    ordered = np.sort(amplitudes)[::-1]
    expected = np.cumsum(ordered, dtype=np.float64)
    expected /= expected[-1]

    assert actual is not None
    actual_crossing = int(np.searchsorted(actual, 0.95)) + 1
    expected_crossing = int(np.searchsorted(expected, 0.95)) + 1
    assert actual_crossing == expected_crossing


def _rgba_data(n_splats: int, *, zero_amplitudes: bool = False) -> GSplatData:
    amplitudes = np.zeros(n_splats, dtype=np.float32)
    if not zero_amplitudes:
        amplitudes.fill(1.0)
    colors = np.ones((n_splats, 4), dtype=np.float32)
    colors[:, 3] = np.linspace(0.0, 1.0, n_splats, dtype=np.float32)
    centers = np.linspace(0.0, 1.0, n_splats * 3, dtype=np.float32).reshape(-1, 3)
    cholesky = np.zeros((n_splats, 6), dtype=np.float32)
    cholesky[:, [0, 2, 5]] = 1.0
    return GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky,
        colors=colors,
    )


def test_info_contributor_count_matches_cumulative_cull_for_rgba(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _rgba_data(10_000)
    path = tmp_path / "rgba.gsplats.zarr"
    data.save(path, ordering="none")
    stored = GSplatData.load(path)
    expected = stored.cull(method="cumulative", retention=0.95).n_splats
    lines: list[str] = []
    monkeypatch.setattr(
        inspect_commands, "aprint", lambda value: lines.append(str(value))
    )

    _info_report(path, show_histograms=False, bins=40, full_provenance=False)

    output = "\n".join(lines)
    assert f"Top {expected:,} splats" in output
    assert "contribute 95% of total rendered amplitude (A·α)" in output


def test_info_skips_contributor_claims_for_zero_total(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _rgba_data(1_000, zero_amplitudes=True)
    path = tmp_path / "zero.gsplats.zarr"
    data.save(path, ordering="none")
    lines: list[str] = []
    monkeypatch.setattr(
        inspect_commands, "aprint", lambda value: lines.append(str(value))
    )

    _info_report(path, show_histograms=False, bins=40, full_provenance=False)

    output = "\n".join(lines)
    assert "contribute 95% of total rendered amplitude (A·α)" not in output
    assert "Culling Suggestion" not in output
