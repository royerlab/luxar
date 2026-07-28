"""Tests for the deterministic fractal generators in demo_4d_fractals.

The demo's core contract: every slider-exposed w value shows structure for
every fractal. The viewer snaps discrete-dim navigation to exact multiples
of ``step`` and fetches chunks only within 0.25×step of the snapped
position, so the tests verify (a) data planes sit exactly on that snap
grid, (b) the declared range is covered end to end, and (c) no fractal has
an empty w-plane.

The demo is loaded by file path (see test_demo_ppi_flow_field for the
rationale: ``luxar.demos`` is aliased to ``luxar.utils.demos``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_4d_fractals.py"


def _load_demo_module():
    name = "_luxar_demo_4d_fractals_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
axis_world_values = _demo.axis_world_values
generate_4d_fractal = _demo.generate_4d_fractal
checkerboard_4d = _demo.checkerboard_4d

GRID = 24  # small but structurally representative (fast: 24^4 = 331K samples)
N_FRACTALS = 6


class TestAxisWorldValues:
    def test_values_are_exact_step_multiples(self):
        """The viewer snaps to k×step anchored at 0 — data must sit there.

        The load-bearing invariant is the snap round-trip the viewer
        computes (``Math.round(v/step)*step``): it must reproduce every
        axis value BIT-EXACTLY, so a snapped slider stop equals the data
        plane it targets.
        """
        for grid in (24, 50, 51):
            axis = axis_world_values(grid)
            step = 2.0 / grid
            k = axis / step
            assert np.abs(k - np.round(k)).max() < 1e-12, (
                f"grid={grid}: axis values not step multiples"
            )
            snapped = np.round(k) * step
            assert np.array_equal(snapped, axis), (
                f"grid={grid}: snap round-trip not bit-exact"
            )

    def test_includes_zero_and_is_monotonic(self):
        for grid in (24, 50, 51):
            axis = axis_world_values(grid)
            assert 0.0 in axis
            assert (np.diff(axis) > 0).all()
            assert axis[0] >= -1.0 and axis[-1] < 1.0


class TestEveryWPlanePopulated:
    """Every slider stop must show points, for every fractal type."""

    @pytest.mark.parametrize("fractal_type", range(N_FRACTALS))
    def test_no_empty_w_plane(self, fractal_type):
        positions, values = generate_4d_fractal(fractal_type, grid_size=GRID)
        assert len(positions) > 0
        assert len(values) == len(positions)

        axis = axis_world_values(GRID).astype(np.float32)
        w = positions[:, 0]
        # Every one of the GRID planes (= slider stops) has points
        for plane_w in axis:
            n = int((w == plane_w).sum())
            assert n > 0, f"fractal {fractal_type}: empty w-plane at {plane_w:.4f}"

    @pytest.mark.parametrize("fractal_type", range(N_FRACTALS))
    def test_positions_on_snap_grid(self, fractal_type):
        """All 4 coordinates take only the declared axis values."""
        positions, _ = generate_4d_fractal(fractal_type, grid_size=GRID)
        axis = set(axis_world_values(GRID).astype(np.float32).tolist())
        for col in range(4):
            unique = set(np.unique(positions[:, col]).tolist())
            assert unique <= axis

    @pytest.mark.parametrize("fractal_type", range(N_FRACTALS))
    def test_within_budget(self, fractal_type):
        positions, _ = generate_4d_fractal(fractal_type, grid_size=GRID)
        assert len(positions) <= _demo.TARGET_MAX_POINTS

    @pytest.mark.parametrize("grid_size", [0, 1, 2])
    def test_too_small_grid_rejected(self, grid_size):
        """Below grid 3 some rules cannot populate every w-plane; the
        generator must fail with a clear ValueError up front."""
        with pytest.raises(ValueError, match="grid_size must be >= 3"):
            generate_4d_fractal(0, grid_size=grid_size)

    def test_minimum_grid_works_for_all_fractals(self):
        for fractal_type in range(N_FRACTALS):
            positions, _ = generate_4d_fractal(fractal_type, grid_size=3)
            assert len(positions) > 0


class TestCheckerboardIsDeterministicStructure:
    """Guards the historical failure: the parity rule degenerated to zero
    points on integer grids and was silently replaced by random noise."""

    def test_half_density_parity(self):
        n = 20
        idx = np.arange(n, dtype=np.int32)
        IW, IX, IY, IZ = np.meshgrid(idx, idx, idx, idx, indexing="ij")
        keep, diag = checkerboard_4d(IW, IX, IY, IZ, n, cells=5)
        frac = keep.mean()
        assert 0.4 < frac < 0.6, f"checkerboard density {frac} not ~50%"
        # Deterministic parity: kept exactly where the cell diagonal is odd
        assert bool(((diag % 2 == 1) == keep).all())

    def test_pattern_inverts_across_cell_boundary(self):
        n = 20
        idx = np.arange(n, dtype=np.int32)
        IW, IX, IY, IZ = np.meshgrid(idx, idx, idx, idx, indexing="ij")
        keep, _ = checkerboard_4d(IW, IX, IY, IZ, n, cells=5)
        # w-planes 0..3 are in cell 0, planes 4..7 in cell 1: inverted
        assert bool((keep[0] == keep[3]).all())
        assert bool((keep[0] == ~keep[4]).all())


class TestGeneratorDeterminism:
    def test_same_seed_same_output(self):
        p1, v1 = generate_4d_fractal(3, grid_size=GRID)  # Cantor uses the rng
        p2, v2 = generate_4d_fractal(3, grid_size=GRID)
        assert np.array_equal(p1, p2)
        assert np.array_equal(v1, v2)


class TestWrittenDatasetContract:
    """The zarr the demo actually ships must satisfy the slider contract
    after the encode→decode round-trip (quantized position storage)."""

    def test_dimension_metadata_and_decoded_planes(self, tmp_path):
        from luxar.io.reader import LuxarScene

        grid = 8
        out = tmp_path / "fractals_4d_test.luxar.zarr"
        _demo.generate_4d_fractal_dataset(out, grid_size=grid)

        scene = LuxarScene.load(out)
        dims = scene._root.attrs["scene_dimensions"]["dimensions"]
        wdim = dims[1]
        step = 2.0 / grid
        axis = axis_world_values(grid)
        assert wdim["name"] == "w"
        assert wdim["discrete"] is True
        assert wdim["step"] == step
        assert wdim["range"] == [float(axis[0]), float(axis[-1])]

        pos = scene.get_points("Fractals4D")["positions"]
        w = pos[:, 1]
        fractal_ids = pos[:, 0]
        # Every slider stop (k*step within range) must have decoded points
        # within the viewer's 0.25*step fetch reach, for EVERY fractal.
        ks = np.arange(round(axis[0] / step), round(axis[-1] / step) + 1)
        for fid in range(6):
            wf = w[fractal_ids == fid]
            for k in ks:
                stop = k * step
                assert (np.abs(wf - stop) <= 0.25 * step).any(), (
                    f"fractal {fid}: no decoded points at stop {stop}"
                )
