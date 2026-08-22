"""Smoke tests for the pure helpers in demo_gsplats_3d_tng_cosmic_web.

Only ``deposit_cic`` and ``finalize_density`` are exercised (no network, no
HDF5 IO, no GPU). The demo is loaded by file path (see test_demo_ppi_flow_field
for the rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_tng_cosmic_web.py"
)


def _load_demo_module():
    name = "_luxar_demo_tng_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
deposit_cic = _demo.deposit_cic
finalize_density = _demo.finalize_density


class TestDepositCIC:
    def test_shape_and_dtype(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        grid = deposit_cic(coords, box=10.0, grid=4)
        assert grid.shape == (4, 4, 4)
        assert grid.dtype == np.float32

    def test_mass_is_conserved(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.uniform(0.0, 10.0, size=(1000, 3))
        grid = deposit_cic(coords, box=10.0, grid=8)
        # CIC spreads each particle over 8 voxels but total weight per particle is 1.
        assert np.isclose(grid.sum(), len(coords), rtol=1e-4)

    def test_particle_on_node_is_localized(self) -> None:
        # A particle sitting exactly on a grid node deposits all its mass there.
        coords = np.array([[2.5, 2.5, 2.5]])  # box=10, grid=4 -> node (1,1,1)
        grid = deposit_cic(coords, box=10.0, grid=4)
        assert np.isclose(grid[1, 1, 1], 1.0, rtol=1e-5)
        assert np.isclose(grid.sum(), 1.0, rtol=1e-5)

    def test_periodic_wrap(self) -> None:
        # A particle just inside the far boundary wraps mass onto voxel 0.
        coords = np.array([[9.9, 0.0, 0.0]])  # box=10, grid=10 -> between 9 and 0
        grid = deposit_cic(coords, box=10.0, grid=10)
        assert grid[0, 0, 0] > 0.0  # wrapped contribution
        assert np.isclose(grid.sum(), 1.0, rtol=1e-5)

    def test_out_of_range_coords_wrap(self) -> None:
        # Coords at/beyond the box and negatives wrap periodically; mass kept.
        for c in ([[10.0, 10.0, 10.0]], [[12.5, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]):
            grid = deposit_cic(np.array(c), box=10.0, grid=10)
            assert np.isclose(grid.sum(), 1.0, rtol=1e-5)


class TestFinalizeDensity:
    def test_range_and_dtype(self) -> None:
        rng = np.random.default_rng(1)
        raw = np.abs(rng.normal(size=(16, 16, 16))).astype(np.float32)
        out = finalize_density(raw, sigma=0.7)
        assert out.shape == (16, 16, 16)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert np.isclose(out.max(), 1.0, rtol=1e-5)

    def test_sigma_zero_is_passthrough_shape(self) -> None:
        raw = np.abs(np.random.default_rng(2).normal(size=(8, 8, 8))).astype(
            np.float32
        )
        out = finalize_density(raw, sigma=0.0)
        assert out.shape == (8, 8, 8)
        assert out.max() <= 1.0

    def test_deterministic(self) -> None:
        raw = np.abs(np.random.default_rng(3).normal(size=(10, 10, 10))).astype(
            np.float32
        )
        np.testing.assert_array_equal(
            finalize_density(raw, 0.7), finalize_density(raw, 0.7)
        )

    def test_all_zero_input_no_division(self) -> None:
        # An all-zero field must not divide by zero (guarded): all zeros, no NaN.
        out = finalize_density(np.zeros((6, 6, 6), dtype=np.float32), sigma=0.25)
        assert out.shape == (6, 6, 6)
        assert float(out.max()) == 0.0
        assert not np.isnan(out).any()
