"""Smoke tests for the pure volume-preprocessing helper in
demo_gsplats_3d_milky_way_dust.

Only ``normalize_dust_volume`` is exercised (no network, no h5py, no GPU fit).
The demo is loaded by file path (see test_demo_ppi_flow_field for the rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_milky_way_dust.py"


def _load_demo_module():
    name = "_luxar_demo_milkyway_dust_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
normalize_dust_volume = _demo.normalize_dust_volume


class TestNormalizeDustVolume:
    def test_downsamples_to_target_and_normalizes(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.0, 5.0, size=(16, 16, 12)).astype(np.float32)
        out = normalize_dust_volume(raw, target_size=8)
        assert out.shape == (8, 8, 8)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_log_density_input_exponentiated(self) -> None:
        # Negative values ⇒ treated as log-density ⇒ exp() ⇒ all finite, non-negative.
        raw = np.linspace(-5.0, 2.0, 8 * 8 * 8).reshape(8, 8, 8).astype(np.float32)
        out = normalize_dust_volume(raw, target_size=8)
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_no_resize_when_target_matches(self) -> None:
        raw = np.abs(np.random.default_rng(1).normal(size=(8, 8, 8))).astype(np.float32)
        out = normalize_dust_volume(raw, target_size=8)
        assert out.shape == (8, 8, 8)

    def test_deterministic(self) -> None:
        raw = np.abs(np.random.default_rng(2).normal(size=(10, 10, 10))).astype(
            np.float32
        )
        a = normalize_dust_volume(raw, target_size=6)
        b = normalize_dust_volume(raw, target_size=6)
        np.testing.assert_array_equal(a, b)
