"""Smoke tests for the pure helpers in demo_desi_galaxies.

Exercises the deterministic array helpers only (no network, no astropy read, no
scene build). The demo is loaded by file path (see test_demo_ppi_flow_field).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_desi_galaxies.py"


def _load_demo_module():
    import pytest

    name = "_luxar_demo_desi_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
radec_z_to_xyz = _demo.radec_z_to_xyz
tracer_colors = _demo.tracer_colors
redshift_colors = _demo.redshift_colors
quantize_positions = _demo.quantize_positions
dequantize_positions = _demo.dequantize_positions
save_derived = _demo.save_derived
load_derived = _demo.load_derived


class TestRadecToXyz:
    def test_origin_axis_directions(self) -> None:
        # (RA=0, Dec=0) at distance d → +x axis.
        p = radec_z_to_xyz(np.array([0.0]), np.array([0.0]), np.array([100.0]))
        np.testing.assert_allclose(p[0], [100.0, 0.0, 0.0], atol=1e-3)
        # (RA=90, Dec=0) → +y.
        p = radec_z_to_xyz(np.array([90.0]), np.array([0.0]), np.array([100.0]))
        np.testing.assert_allclose(p[0], [0.0, 100.0, 0.0], atol=1e-3)
        # (Dec=90) → +z (north pole), independent of RA.
        p = radec_z_to_xyz(np.array([37.0]), np.array([90.0]), np.array([100.0]))
        np.testing.assert_allclose(p[0], [0.0, 0.0, 100.0], atol=1e-3)

    def test_radius_preserved(self) -> None:
        rng = np.random.default_rng(0)
        ra = rng.uniform(0, 360, 500)
        dec = rng.uniform(-90, 90, 500)
        d = rng.uniform(10, 3000, 500)
        p = radec_z_to_xyz(ra, dec, d)
        np.testing.assert_allclose(np.linalg.norm(p, axis=1), d, rtol=1e-4)
        assert p.dtype == np.float32


class TestTracerColors:
    def test_maps_ids_to_palette(self) -> None:
        cols = tracer_colors(np.array([0, 1, 2, 3], dtype=np.uint8))
        assert cols.shape == (4, 3)
        assert cols.dtype == np.float32
        # Each row is a distinct, in-gamut color.
        assert cols.min() >= 0.0 and cols.max() <= 1.0
        assert len({tuple(row) for row in cols}) == 4


class TestRedshiftColors:
    def test_shape_dtype_gamut(self) -> None:
        z = np.linspace(0.01, 3.5, 100).astype(np.float32)
        cols = redshift_colors(z)
        assert cols.shape == (100, 3)
        assert cols.dtype == np.float32
        assert cols.min() >= 0.0 and cols.max() <= 1.0

    def test_low_vs_high_z_distinct(self) -> None:
        # turbo maps low→cold, high→hot; nearby and distant must differ.
        cols = redshift_colors(np.array([0.02, 0.05, 0.5, 1.0, 3.0], dtype=np.float32))
        assert not np.allclose(cols[0], cols[-1])
        assert len({tuple(np.round(c, 3)) for c in cols}) >= 4

    def test_degenerate_and_empty(self) -> None:
        # all-equal redshift → valid (no div-by-zero), single color.
        same = redshift_colors(np.full(5, 0.3, dtype=np.float32))
        assert same.shape == (5, 3) and np.isfinite(same).all()
        empty = redshift_colors(np.array([], dtype=np.float32))
        assert empty.shape == (0, 3)


class TestQuantizeRoundtrip:
    def test_positions_roundtrip_sub_mpc(self) -> None:
        rng = np.random.default_rng(1)
        pos = rng.uniform(-3000, 3000, size=(2000, 3)).astype(np.float32)
        q, offset, scale = quantize_positions(pos)
        assert q.dtype == np.int16
        back = dequantize_positions(q, offset, scale)
        # 6000 Mpc span / 65534 levels ≈ 0.09 Mpc/step → within ~0.1 Mpc.
        assert np.max(np.abs(back - pos)) < 0.15

    def test_derived_npz_roundtrip(self, tmp_path) -> None:
        rng = np.random.default_rng(2)
        pos = rng.uniform(-2000, 2000, size=(1000, 3)).astype(np.float32)
        z = rng.uniform(0.01, 3.9, size=1000).astype(np.float32)
        tid = rng.integers(0, 4, size=1000).astype(np.uint8)
        p = tmp_path / "d.npz"
        save_derived(p, pos, z, tid)
        assert p.stat().st_size > 0
        pos2, z2, tid2 = load_derived(p)
        assert pos2.dtype == np.float32 and z2.dtype == np.float32
        np.testing.assert_array_equal(tid2, tid)
        assert np.max(np.abs(pos2 - pos)) < 0.15
        # float16 redshift → ~3 significant digits.
        np.testing.assert_allclose(z2, z, atol=2e-3)
