"""Precision regression tests for ``luxar gsplat info`` amplitude analysis."""

from __future__ import annotations

import numpy as np

from luxar.cli.gsplat_ops.inspect_commands import _normalized_amplitude_cdf


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

    actual_crossing = int(np.searchsorted(actual, 0.95)) + 1
    expected_crossing = int(np.searchsorted(expected, 0.95)) + 1
    assert actual_crossing == expected_crossing
