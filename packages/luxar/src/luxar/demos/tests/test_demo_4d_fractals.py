"""Tests for the deterministic fractal generators in demo_4d_fractals.

The demo's core contract: every slider-exposed w value shows structure for
every fractal. The viewer snaps discrete-dim navigation to exact multiples
of ``step`` and fetches chunks only within 0.25×step of the snapped
position, so the tests verify (a) data planes sit exactly on that snap
grid, (b) the declared range is covered end to end, and (c) no fractal has
an empty w-plane.

The demo is loaded by file path, matching the sibling demo tests: demo
modules are standalone scripts in the demos directory, not part of the
package's importable API.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_4d_fractals.py"
_CHECKER_PATH = Path(__file__).resolve().parents[6] / "scripts/check_demo_ladders.py"


def _load_demo_module():
    name = "_luxar_demo_4d_fractals_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_checker_module():
    name = "_luxar_check_demo_ladders_for_fractal_tests"
    spec = importlib.util.spec_from_file_location(name, _CHECKER_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate ladder auditor at {_CHECKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
_checker = _load_checker_module()
axis_world_values = _demo.axis_world_values
generate_4d_fractal = _demo.generate_4d_fractal
materialised_w_planes = _demo.materialised_w_planes
checkerboard_4d = _demo.checkerboard_4d
surface_normals = _demo.surface_normals
surface_of = _demo.surface_of
apply_fractal_ambient_occlusion = _demo.apply_fractal_ambient_occlusion

GRID = 24  # small but structurally representative (fast: 24^4 = 331K samples)
N_FRACTALS = 6
# Precision assertions below use the shipped slider's fetch reach so the
# coarser test grid cannot mask a production regression.
PRODUCTION_FETCH_REACH = 0.25 * (_demo.W_STRIDE * 2.0 / _demo.GRID_SIZE_DEFAULT)


def test_per_plane_budget_stays_below_ladder_audit_threshold() -> None:
    assert _demo.TARGET_MAX_POINTS_PER_PLANE < _checker.DEFAULT_MIN_ELEMENTS


class TestAxisWorldValues:
    def test_values_are_exact_step_multiples(self) -> None:
        """The declared step must reproduce every materialised plane.

        The load-bearing invariant is that ``axis[0] + k*step`` reproduces
        every axis value BIT-EXACTLY, so each slider stop equals the data plane
        it targets.
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

    def test_includes_zero_and_is_monotonic(self) -> None:
        for grid in (24, 50, 51):
            axis = axis_world_values(grid)
            assert 0.0 in axis
            assert (np.diff(axis) > 0).all()
            assert axis[0] >= -1.0 and axis[-1] < 1.0

    def test_materialised_planes_are_on_the_declared_stride_grid(self) -> None:
        for grid in (12, 16, 50, 100, 150, 200):
            planes = materialised_w_planes(grid, _demo.W_STRIDE)
            values = axis_world_values(grid)[planes]
            step = _demo.W_STRIDE * 2.0 / grid
            assert np.allclose(values / step, np.round(values / step), atol=1e-12), (
                f"grid={grid}: materialised planes miss the viewer snap grid"
            )

        assert np.array_equal(materialised_w_planes(16, 4), [0, 4, 8, 12])
        assert np.array_equal(materialised_w_planes(12, 4), [2, 6, 10])


class TestEveryWPlanePopulated:
    """Every slider stop must show points, for every fractal type."""

    @pytest.mark.parametrize("fractal_type", range(N_FRACTALS))
    def test_no_empty_w_plane(self, fractal_type: int) -> None:
        """Every MATERIALISED plane (= every slider stop) has points.

        Only every ``W_STRIDE``-th plane of the lattice is written, so the
        contract is about the stops the slider actually offers, not about the
        full lattice.
        """
        positions, values, normals = generate_4d_fractal(fractal_type, grid_size=GRID)
        assert len(positions) > 0
        assert len(values) == len(positions)
        assert normals.shape == (len(positions), 3)

        axis = axis_world_values(GRID).astype(np.float32)
        planes = materialised_w_planes(GRID, _demo.W_STRIDE)
        w = positions[:, 0]
        for plane_w in axis[planes]:
            n = int((w == plane_w).sum())
            assert n > 0, f"fractal {fractal_type}: empty w-plane at {plane_w:.4f}"

    @pytest.mark.parametrize("fractal_type", range(N_FRACTALS))
    def test_no_empty_w_plane_at_full_stride(self, fractal_type: int) -> None:
        """With ``w_stride=1`` EVERY lattice plane must be populated.

        The stronger of the two: it is the property the strided default relies
        on, so a rule that empties an interior plane must fail here even though
        the shipped stride would skip that plane.
        """
        positions, _, _ = generate_4d_fractal(fractal_type, grid_size=GRID, w_stride=1)
        axis = axis_world_values(GRID).astype(np.float32)
        w = positions[:, 0]
        for plane_w in axis:
            assert (w == plane_w).any(), (
                f"fractal {fractal_type}: empty w-plane at {plane_w:.4f}"
            )

    @pytest.mark.parametrize("fractal_type", range(N_FRACTALS))
    def test_positions_on_snap_grid(self, fractal_type: int) -> None:
        """All 4 coordinates take only the declared axis values."""
        positions, _, _ = generate_4d_fractal(fractal_type, grid_size=GRID)
        full = axis_world_values(GRID).astype(np.float32)
        axis = set(full.tolist())
        planes = materialised_w_planes(GRID, _demo.W_STRIDE)
        # w is restricted to the MATERIALISED stops; x/y/z use the full lattice.
        assert set(np.unique(positions[:, 0]).tolist()) <= set(full[planes].tolist())
        for col in range(1, 4):
            unique = set(np.unique(positions[:, col]).tolist())
            assert unique <= axis

    @pytest.mark.parametrize("fractal_type", range(N_FRACTALS))
    def test_within_budget(self, fractal_type: int) -> None:
        """The budget is PER PLANE, so the total is bounded by planes x budget.

        Checked per plane rather than only in total: a global-only check passes
        even when the cap has silently become a whole-dataset budget again,
        which is the regression that made a finer grid render SPARSER.
        """
        positions, _, _ = generate_4d_fractal(fractal_type, grid_size=GRID)
        planes = materialised_w_planes(GRID, _demo.W_STRIDE)
        budget = _demo.TARGET_MAX_POINTS_PER_PLANE
        assert len(positions) <= len(planes) * budget
        counts = np.unique(positions[:, 0], return_counts=True)[1]
        assert counts.max() <= budget

    @pytest.mark.parametrize("grid_size", [0, 1, 2])
    def test_too_small_grid_rejected(self, grid_size: int) -> None:
        """Below grid 3 some rules cannot populate every w-plane; the
        generator must fail with a clear ValueError up front."""
        with pytest.raises(ValueError, match="grid_size must be >= 3"):
            generate_4d_fractal(0, grid_size=grid_size)

    def test_minimum_grid_works_for_all_fractals(self) -> None:
        for fractal_type in range(N_FRACTALS):
            positions, _, _ = generate_4d_fractal(fractal_type, grid_size=3)
            assert len(positions) > 0

    def test_empty_plane_error_reports_lattice_plane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty_plane = int(materialised_w_planes(GRID, _demo.W_STRIDE)[1])

        def rule(iw, ix, iy, iz, grid_size):
            del ix, iy, iz, grid_size
            keep = iw != empty_plane
            return keep, np.ones(iw.shape, dtype=np.float32)

        monkeypatch.setattr(
            _demo, "_fractal_rule", lambda fractal_type, rng: ("test", rule)
        )

        with pytest.raises(RuntimeError, match=rf"empty w-planes \[{empty_plane}\]"):
            generate_4d_fractal(0, grid_size=GRID, surface_only=False)

    def test_subsample_branch_preserves_every_plane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At the production grid the dense fractals exceed the budget and
        take the subsample branch; at the test grid they don't. Shrink the
        budget so that branch (and the post-subsample plane check) runs."""
        planes = materialised_w_planes(GRID, _demo.W_STRIDE)
        positions, _, _ = generate_4d_fractal(
            0, grid_size=GRID, max_points_per_plane=200
        )  # XOR is dense
        assert len(positions) == 200 * len(planes)
        axis = axis_world_values(GRID).astype(np.float32)
        w = positions[:, 0]
        for plane_w in axis[planes]:
            assert (w == plane_w).any(), f"subsample emptied plane {plane_w}"


class TestCheckerboardIsDeterministicStructure:
    """Guards the historical failure: the parity rule degenerated to zero
    points on integer grids and was silently replaced by random noise."""

    def test_half_density_parity(self) -> None:
        n = 20
        idx = np.arange(n, dtype=np.int32)
        IW, IX, IY, IZ = np.meshgrid(idx, idx, idx, idx, indexing="ij")
        keep, diag = checkerboard_4d(IW, IX, IY, IZ, n, cells=5)
        frac = keep.mean()
        assert 0.4 < frac < 0.6, f"checkerboard density {frac} not ~50%"
        # Deterministic parity: kept exactly where the cell diagonal is odd
        assert bool(((diag % 2 == 1) == keep).all())
        # Independent of the implementation's own derivation: a known cell
        # must be lit — index 4 maps to cell 1 (odd diagonal 1,0,0,0).
        assert bool(keep[4, 0, 0, 0])
        assert not bool(keep[0, 0, 0, 0])

    def test_pattern_inverts_across_cell_boundary(self) -> None:
        n = 20
        idx = np.arange(n, dtype=np.int32)
        IW, IX, IY, IZ = np.meshgrid(idx, idx, idx, idx, indexing="ij")
        keep, _ = checkerboard_4d(IW, IX, IY, IZ, n, cells=5)
        # w-planes 0..3 are in cell 0, planes 4..7 in cell 1: inverted
        assert bool((keep[0] == keep[3]).all())
        assert bool((keep[0] == ~keep[4]).all())


class TestGeneratorDeterminism:
    @pytest.mark.parametrize("fractal_type", range(N_FRACTALS))
    def test_same_seed_same_output(self, fractal_type: int) -> None:
        p1, v1, n1 = generate_4d_fractal(fractal_type, grid_size=GRID)
        p2, v2, n2 = generate_4d_fractal(fractal_type, grid_size=GRID)
        assert np.array_equal(p1, p2)
        assert np.array_equal(v1, v2)
        assert np.array_equal(n1, n2)


class TestAmbientOcclusionInputs:
    def test_generator_derives_normals_before_surface_extraction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def solid_rule(iw, ix, iy, iz, grid_size):
            del iw
            keep = (
                (ix >= 1)
                & (ix < grid_size - 1)
                & (iy >= 1)
                & (iy < grid_size - 1)
                & (iz >= 1)
                & (iz < grid_size - 1)
            )
            return keep, np.ones(keep.shape, dtype=np.float32)

        monkeypatch.setattr(
            _demo, "_fractal_rule", lambda fractal_type, rng: ("solid", solid_rule)
        )

        _, _, normals = generate_4d_fractal(0, grid_size=5)

        assert len(normals) == 26
        assert np.all(np.linalg.norm(normals, axis=1) > 0.0)

    def test_surface_normals_point_outward_from_the_solid(self) -> None:
        solid = np.zeros((5, 5, 5), dtype=bool)
        solid[1:4, 1:4, 1:4] = True
        normals = surface_normals(solid)
        surface = surface_of(solid)

        np.testing.assert_allclose(normals[1, 2, 2], [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(normals[3, 2, 2], [1.0, 0.0, 0.0])
        assert np.all(np.linalg.norm(normals[surface], axis=1) > 0.0)

        isolated = np.zeros((5, 5, 5), dtype=bool)
        isolated[2, 2, 2] = True
        np.testing.assert_array_equal(surface_normals(isolated)[2, 2, 2], 0.0)

    def test_occlusion_isolated_by_fractal_and_w_slice(self) -> None:
        solid = np.zeros((17, 17, 17), dtype=bool)
        solid[1:16, 1:16, 1:16] = True
        surface = surface_of(solid)
        spatial = np.column_stack(np.nonzero(surface)).astype(np.float32) / 8.0 - 1.0
        normals = surface_normals(solid)[surface]

        def block(fractal: float, w: float, x_offset: float) -> np.ndarray:
            shifted = spatial.copy()
            shifted[:, 0] += x_offset
            return np.column_stack(
                [
                    np.full(len(spatial), fractal, dtype=np.float32),
                    np.full(len(spatial), w, dtype=np.float32),
                    shifted,
                ]
            )

        positions = np.vstack(
            [
                block(0.0, -0.5, 0.0),
                block(0.0, 0.5, 0.5),
                block(1.0, -0.5, 1.0),
            ]
        )
        repeated_normals = np.vstack([normals, normals, normals])
        base_colors = np.ones((len(positions), 3), dtype=np.float32)

        colors = apply_fractal_ambient_occlusion(
            positions, repeated_normals, base_colors
        )
        scaled_positions = positions.copy()
        scaled_positions[:, :2] *= 10.0
        scaled_colors = apply_fractal_ambient_occlusion(
            scaled_positions, repeated_normals, base_colors
        )
        expected_shade = _demo.bake_ambient_occlusion(
            positions,
            normals=repeated_normals,
            occluder="opaque",
            n_directions=_demo.AO_N_DIRECTIONS,
            spatial_dims=(2, 3, 4),
            group_by=np.repeat([0, 1, 2], len(spatial)),
        )
        merged_shade = _demo.bake_ambient_occlusion(
            positions,
            normals=repeated_normals,
            occluder="opaque",
            n_directions=_demo.AO_N_DIRECTIONS,
            spatial_dims=(2, 3, 4),
        )
        no_normal_shade = _demo.bake_ambient_occlusion(
            positions,
            normals=None,
            occluder="opaque",
            n_directions=_demo.AO_N_DIRECTIONS,
            spatial_dims=(2, 3, 4),
            group_by=np.repeat([0, 1, 2], len(spatial)),
        )
        rolled_normal_colors = apply_fractal_ambient_occlusion(
            positions,
            np.roll(repeated_normals, 1, axis=1),
            base_colors,
        )

        np.testing.assert_array_equal(colors, scaled_colors)
        np.testing.assert_allclose(colors[:, 0], expected_shade, atol=1e-6)
        blocks = colors.reshape(3, len(spatial), 3)
        np.testing.assert_allclose(blocks[0], blocks[1], atol=1e-6)
        np.testing.assert_allclose(blocks[0], blocks[2], atol=1e-6)
        assert float(np.max(np.abs(colors[:, 0] - merged_shade))) > 0.02
        assert float(np.max(np.abs(colors[:, 0] - no_normal_shade))) > 0.1
        assert float(np.max(np.abs(colors - rolled_normal_colors))) > 0.05
        assert float(colors.mean()) < 0.98
        assert float(np.ptp(colors[:, 0])) > 0.05
        assert np.all(colors <= base_colors)


class TestWrittenDatasetContract:
    """The zarr the demo actually ships must satisfy the slider contract
    after the encode→decode round-trip (quantized position storage)."""

    @pytest.mark.parametrize("grid", [12, 16])
    def test_dimension_metadata_and_decoded_planes(
        self, tmp_path: Path, grid: int
    ) -> None:
        import zarr

        from luxar.io.reader import LuxarScene

        # 12 exercises phase alignment when grid//2 is not divisible by the
        # stride; 16 proves the already-aligned case does not move.
        out = tmp_path / f"fractals_4d_test_{grid}.luxar.zarr"
        _demo.generate_4d_fractal_dataset(out, grid_size=grid)

        scene = LuxarScene.load(out)
        dims = scene.root_attrs["scene_dimensions"]["dimensions"]
        wdim = dims[1]
        # The slider's snap grid is the MATERIALISED spacing — stride lattice
        # steps — not the lattice step. Declaring the lattice step here would
        # put three out of every four slider stops on empty space, which is
        # exactly the class of bug this test exists to catch.
        step = _demo.W_STRIDE * 2.0 / grid
        planes = materialised_w_planes(grid, _demo.W_STRIDE)
        axis = axis_world_values(grid)[planes]
        assert wdim["name"] == "w"
        assert wdim["discrete"] is True
        assert wdim["step"] == step
        assert wdim["range"] == [float(axis[0]), float(axis[-1])]
        assert len(planes) > 2, "test grid too small to exercise the slider"

        root = zarr.open_group(out, mode="r")
        leaf = root["Fractals4D"]
        attrs = leaf.attrs
        assert attrs["type"] == "points"
        assert attrs["blending_mode"] == "volumetric"
        assert attrs["opacity"] == pytest.approx(0.43)
        assert attrs["absorption"] == pytest.approx(1.23)
        assert attrs["intensity"] == pytest.approx(1.0 / _demo.DISPLAY_MAX)

        assert not list(leaf.group_keys())
        pos = scene.get_points("Fractals4D")["positions"]
        w = pos[:, 1]
        fractal_ids = pos[:, 0]
        # Quantization precision is judged against the PRODUCTION fetch
        # reach (grid=50), not this test grid's much looser 0.25*step —
        # a decode error that would break the shipped slider must fail here.
        ideal = axis.astype(np.float64)
        decode_err = np.abs(w[:, None] - ideal[None, :]).min(axis=1).max()
        assert decode_err < PRODUCTION_FETCH_REACH / 10, (
            f"decode error {decode_err} too close to production fetch reach"
        )
        # Every slider stop (range_min + k*step) must have decoded points
        # within the viewer's 0.25*step fetch reach, for EVERY fractal.
        stops = axis[0] + np.arange(len(axis)) * step
        for fid in range(6):
            wf = w[fractal_ids == fid]
            for stop in stops:
                assert (np.abs(wf - stop) <= 0.25 * step).any(), (
                    f"fractal {fid}: no decoded points at stop {stop}"
                )
