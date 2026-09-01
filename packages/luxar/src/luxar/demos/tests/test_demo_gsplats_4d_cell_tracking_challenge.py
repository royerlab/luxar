"""Tests for the pure helpers in demo_gsplats_4d_cell_tracking_challenge.

No network, no Kaggle credentials, no GPU — the grid layout, the lineage palette,
the display-range window, the file manifests and the cache bookkeeping. The demo
is loaded by file path (see test_demo_gsplats_3d_tng_cosmic_web for the
rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_4d_cell_tracking_challenge.py"
)


def _load_demo_module(name: str = "_luxar_demo_cell_tracking_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def _translation(xform) -> tuple[float, float, float]:
    """The (x, y, z) offset of a ``luxar.transforms`` 4x4.

    ``luxar.transforms`` works in NumPy (row-major) convention, so translation
    lives in the last COLUMN — ``[0, 3], [1, 3], [2, 3]``. The compiler transposes
    on write, which is what the viewer's column-major loader expects; a demo that
    pre-transposed here would place every tile wrongly.
    """
    m = np.asarray(xform, dtype=float)
    assert m.shape == (4, 4)
    return float(m[0, 3]), float(m[1, 3]), float(m[2, 3])


class TestGridLayout:
    """The matrix placement: N crops onto the most compact grid that holds them."""

    @pytest.mark.parametrize(
        "n,cols,rows",
        [(1, 1, 1), (2, 2, 1), (4, 2, 2), (5, 3, 2), (6, 3, 2), (7, 3, 3), (9, 3, 3)],
    )
    def test_grid_shape_is_the_shortest_rectangle(
        self, n: int, cols: int, rows: int
    ) -> None:
        assert _demo.grid_shape(n) == (cols, rows)
        assert cols * rows >= n, "the grid must hold every crop"

    def test_six_crops_fill_a_rectangle_with_no_holes(self) -> None:
        """The whole point of 6: a complete 3x2, not a 3x3 with three gaps."""
        cols, rows = _demo.grid_shape(6)
        assert (cols, rows) == (3, 2)
        assert cols * rows == 6, "no empty slots"

    @pytest.mark.parametrize("n,cols", [(1, 1), (2, 2), (4, 2), (5, 3), (6, 3), (9, 3)])
    def test_distinct_columns_match_the_grid_shape(self, n: int, cols: int) -> None:
        xforms = _demo.grid_transforms(n, 100.0)
        assert len(xforms) == n
        xs = {round(_translation(x)[0], 4) for x in xforms}
        assert len(xs) == min(n, cols)

    def test_a_partial_row_is_still_centred(self) -> None:
        """Centring follows the OCCUPIED rows, not the enclosing square.

        Two crops occupy one row, so both sit at y=0. Centring on a 2x2 square
        instead would push the pair half a pitch off-centre and the default
        camera would frame empty space below them.
        """
        offsets = np.array([_translation(x) for x in _demo.grid_transforms(2, 100.0)])
        np.testing.assert_allclose(offsets[:, 1], 0.0)
        assert offsets[:, 0].mean() == pytest.approx(0.0)

    def test_grid_is_centred_on_the_origin(self) -> None:
        """So the default camera frames the whole matrix, not a corner of it."""
        pitch = 120.0
        offsets = np.array([_translation(x) for x in _demo.grid_transforms(9, pitch)])
        assert offsets[:, 0].mean() == pytest.approx(0.0)
        assert offsets[:, 1].mean() == pytest.approx(0.0)
        np.testing.assert_allclose(offsets[:, 2], 0.0)
        assert sorted(set(np.round(offsets[:, 0], 6))) == [-pitch, 0.0, pitch]

    def test_single_crop_sits_at_the_origin(self) -> None:
        (xform,) = _demo.grid_transforms(1, 120.0)
        assert _translation(xform) == (0.0, 0.0, 0.0)

    def test_pitch_scales_the_spacing(self) -> None:
        near = _translation(_demo.grid_transforms(4, 10.0)[0])
        far = _translation(_demo.grid_transforms(4, 20.0)[0])
        assert far[0] == pytest.approx(2.0 * near[0])
        assert far[1] == pytest.approx(2.0 * near[1])

    def test_translation_is_in_numpy_convention(self) -> None:
        """Guards the row-major/column-major trap in both directions.

        The demo must hand the compiler a NumPy-convention matrix (translation in
        the last column) and let the writer transpose. If it were pre-transposed,
        [3]/[7]/[11] would be zero and the offsets would silently vanish.
        """
        (xform,) = _demo.grid_transforms(2, 50.0)[:1]
        m = np.asarray(xform, dtype=float)
        assert m[0, 3] != 0.0, "translation should be in the last column"
        np.testing.assert_allclose(m[3, :3], 0.0)
        assert m[3, 3] == 1.0


class TestLineageColors:
    def test_one_distinct_colour_per_lineage(self) -> None:
        cols = _demo.lineage_colors(64)
        assert cols.shape == (64, 3)
        assert cols.dtype == np.float32
        # Golden-ratio hue stepping: neighbours must not be near-identical, which
        # is the whole reason for not using a linear ramp.
        deltas = np.linalg.norm(np.diff(cols, axis=0), axis=1)
        assert deltas.min() > 0.15

    def test_colours_are_in_range(self) -> None:
        cols = _demo.lineage_colors(200)
        assert cols.min() >= 0.0
        assert cols.max() <= 1.0

    def test_single_lineage_is_handled(self) -> None:
        assert _demo.lineage_colors(1).shape == (1, 3)


class TestDisplayWindow:
    """The (intensity, offset) pair the viewer recovers a display range from."""

    def test_window_spans_min_to_p99(self) -> None:
        rng = np.random.default_rng(0)
        amps = rng.uniform(0.05, 0.8, size=20000).astype(np.float32)
        intensity, offset = _demo.display_window(amps)
        # Invert exactly as ui/layers/layer-state.ts::computeDisplayRange does.
        lo = -offset / intensity
        hi = (1.0 - offset) / intensity
        assert lo == pytest.approx(float(amps.min()), rel=1e-5)
        assert hi == pytest.approx(float(np.percentile(amps, 99.0)), rel=1e-5)

    def test_window_follows_the_data_downward(self) -> None:
        """A dimmer distribution must yield a tighter window, not a frozen one.

        This is the regression that matters: raising the splat budget divides the
        same signal across more splats, so a hard-coded window would sit above the
        data and render everything dark.
        """
        bright = np.linspace(0.05, 1.0, 10000, dtype=np.float32)
        dim = bright * 0.5
        i_bright, o_bright = _demo.display_window(bright)
        i_dim, o_dim = _demo.display_window(dim)
        top_bright = (1.0 - o_bright) / i_bright
        top_dim = (1.0 - o_dim) / i_dim
        assert top_dim == pytest.approx(0.5 * top_bright, rel=1e-4)

    def test_uniform_amplitudes_give_the_identity_window(self) -> None:
        """A degenerate span must not divide by zero or invert the colormap."""
        intensity, offset = _demo.display_window(np.full(100, 0.3, dtype=np.float32))
        assert (intensity, offset) == (1.0, 0.0)

    def test_p99_clips_the_bright_tail(self) -> None:
        """The top of the window tracks p99, not the maximum.

        That is the point of the percentile: a heavy-tailed amplitude
        distribution would otherwise spend most of the colormap on the brightest
        couple of percent and leave the nuclei dim.
        """
        # The tail must be thinner than 1% to fall ABOVE p99 — a 5% tail would
        # contain p99 and legitimately raise the window.
        base = np.linspace(0.1, 0.4, 9950, dtype=np.float32)
        tail = np.linspace(1.0, 5.0, 50, dtype=np.float32)  # 0.5% bright tail
        amps = np.concatenate([base, tail])
        intensity, offset = _demo.display_window(amps)
        top = (1.0 - offset) / intensity
        assert top < 0.5, f"the 0.5% tail must not set the window top (got {top:.3f})"
        assert top == pytest.approx(float(np.percentile(amps, 99.0)), rel=1e-5)


class TestVoxelSize:
    """Reading the crop's voxel size out of its OME-Zarr metadata.

    Both nestings have to work. OME-Zarr **0.5** puts everything under an ``ome``
    key; 0.4 puts ``multiscales`` at the top level — and a zarr v3 store written
    by a 0.4-era tool has v3 chunks with 0.4 attributes, so the store version
    does not settle which spelling is on disk.
    """

    @staticmethod
    def _multiscales() -> list[dict]:
        return [
            {
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.625, 0.40625, 0.40625]}
                        ],
                    }
                ]
            }
        ]

    def _store(self, tmp_path, attrs: dict):
        import zarr

        path = tmp_path / "crop.zarr"
        group = zarr.create_group(store=str(path), overwrite=True, zarr_format=3)
        for key, value in attrs.items():
            group.attrs[key] = value
        return path

    def test_ome_zarr_0_5_nesting_is_read(self, tmp_path) -> None:
        path = self._store(
            tmp_path, {"ome": {"version": "0.5", "multiscales": self._multiscales()}}
        )
        assert _demo.voxel_size_of(path) == pytest.approx((1.625, 0.40625, 0.40625))

    def test_ome_zarr_0_4_layout_is_still_read(self, tmp_path) -> None:
        path = self._store(tmp_path, {"multiscales": self._multiscales()})
        assert _demo.voxel_size_of(path) == pytest.approx((1.625, 0.40625, 0.40625))

    def test_the_time_axis_scale_is_dropped(self, tmp_path) -> None:
        """The stores are TZYX; only the three spatial scales are voxel size."""
        path = self._store(tmp_path, {"multiscales": self._multiscales()})
        assert len(_demo.voxel_size_of(path)) == 3

    def test_metadata_without_multiscales_fails_clearly(self, tmp_path) -> None:
        """Not a KeyError from the middle of a subscript chain."""
        path = self._store(tmp_path, {"ome": {"version": "0.5"}})
        with pytest.raises(ValueError, match="has no OME-Zarr `multiscales`"):
            _demo.voxel_size_of(path)

    def test_multiscales_without_a_scale_transform_says_which_half_is_missing(
        self, tmp_path
    ) -> None:
        """The store IS an OME-Zarr crop; it just states no spacing.

        The two failures need different messages: blaming absent `multiscales`
        for a store that declares them sends the reader looking for the wrong
        thing (and lists attributes that plainly include `ome`).
        """
        no_scale = [
            {
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "translation", "translation": [0, 0, 0, 0]}
                        ],
                    }
                ]
            }
        ]
        path = self._store(
            tmp_path, {"ome": {"version": "0.5", "multiscales": no_scale}}
        )

        with pytest.raises(ValueError, match="states no `scale`") as excinfo:
            _demo.voxel_size_of(path)

        assert "has no OME-Zarr `multiscales`" not in str(excinfo.value)


class TestTrackWindow:
    """`--timepoints` has to cut the tracks, on either path."""

    @staticmethod
    def _tracks(n_timepoints: int = 6) -> dict:
        """One cell tracked straight through, one vertex per timepoint."""
        verts = np.zeros((n_timepoints, 4), np.float32)
        verts[:, 3] = np.arange(n_timepoints)
        edges = np.asarray(
            [(t, t + 1) for t in range(n_timepoints - 1)], dtype=np.uint32
        )
        return {
            "point_positions": verts,
            "point_colors": np.zeros((n_timepoints, 3), np.float32),
            "point_radii": np.full(n_timepoints, 1.3, np.float32),
            "line_vertices": verts,
            "line_colors": np.zeros((n_timepoints, 3), np.float32),
            "line_widths": np.full(n_timepoints, 0.35, np.float32),
            "line_indices": edges,
            "n_lineages": 1,
            "n_cells": n_timepoints,
            "n_divisions": 0,
        }

    def test_markers_and_edges_are_cut_to_the_window(self) -> None:
        got = _demo.window_tracks(self._tracks(6), 3)
        assert got is not None
        assert len(got["point_positions"]) == 3
        assert got["n_cells"] == 3
        assert np.all(got["point_positions"][:, 3] < 3)
        # Only edges wholly inside the window survive: (0,1) and (1,2).
        assert len(got["line_indices"]) == 2
        assert got["line_indices"].max() < 3

    def test_vertices_are_kept_so_edge_indices_stay_valid(self) -> None:
        """Dropping vertices would renumber every edge — the classic corruption."""
        original = self._tracks(6)
        got = _demo.window_tracks(original, 3)
        assert len(got["line_vertices"]) == len(original["line_vertices"])
        assert len(got["line_widths"]) == len(got["line_vertices"])
        assert len(got["point_radii"]) == len(got["point_positions"])

    def test_a_full_window_changes_nothing(self) -> None:
        original = self._tracks(6)
        got = _demo.window_tracks(original, 6)
        np.testing.assert_array_equal(got["line_indices"], original["line_indices"])
        assert got["n_cells"] == original["n_cells"]

    def test_divisions_are_recounted_inside_the_window(self) -> None:
        """A fork past the window must not still be reported as a division."""
        tracks = self._tracks(6)
        # Vertex 4 gains a second parent-child link FROM vertex 3, i.e. 3 divides
        # at t=3 — outside a 3-frame window, inside a 6-frame one.
        tracks["line_indices"] = np.vstack(
            [tracks["line_indices"], np.asarray([(3, 4)], np.uint32)]
        ).astype(np.uint32)
        assert _demo.window_tracks(tracks, 6)["n_divisions"] == 1
        assert _demo.window_tracks(tracks, 3)["n_divisions"] == 0

    def test_no_surviving_edge_means_volume_only(self) -> None:
        """A tile with nothing left to draw returns None, as track_geometry does."""
        assert _demo.window_tracks(self._tracks(6), 1) is None
        assert _demo.window_tracks(None, 5) is None


class TestTrackGeometry:
    """GEFF graph -> the arrays the scene is authored from.

    Uses the GEFF reader's own fixture writer, so what is under test is the
    demo's column order and recentring rather than a hand-built store.
    """

    @staticmethod
    def _store(tmp_path):
        from luxar.gsplats.interop.tests.test_geff_interop import write_geff

        # One founder dividing at t=2, exactly as the reader's fixture, at a
        # voxel size that makes the um conversion visible.
        return write_geff(
            tmp_path / "c.geff",
            node_ids=[10, 11, 12, 13, 14],
            t=[0, 1, 2, 3, 3],
            z=[1, 1, 1, 1, 2],
            y=[0, 1, 2, 3, 2],
            x=[0, 0, 0, 0, 1],
            edges=[(10, 11), (11, 12), (12, 13), (12, 14)],
            scale=(2.0, 0.5, 0.5),
        )

    def test_columns_are_zyx_then_time_and_recentred(self, tmp_path) -> None:
        """The arrays are authored with `dim_order=["z","y","x","time"]`.

        A swapped column order would put every marker somewhere else in the
        embryo, silently — this pins it against the GEFF store's own values.
        """
        centre = np.array([2.0, 1.0, 0.5])  # um, ZYX
        tracks = _demo.track_geometry(self._store(tmp_path), centre, 4)
        assert tracks is not None
        # node 14: voxel (2, 2, 1) x scale (2, .5, .5) = (4, 1, .5) um, at t=3
        np.testing.assert_allclose(
            tracks["line_vertices"][4], [4.0 - 2.0, 1.0 - 1.0, 0.5 - 0.5, 3.0]
        )
        assert tracks["line_vertices"].shape == (5, 4)
        assert tracks["n_divisions"] == 1, "the fork at t=2 is a division"
        assert tracks["n_cells"] == 5
        assert tracks["n_lineages"] == 1

    def test_the_time_window_is_honoured(self, tmp_path) -> None:
        tracks = _demo.track_geometry(self._store(tmp_path), np.zeros(3), 3)
        assert len(tracks["point_positions"]) == 3  # t=0,1,2
        assert tracks["n_divisions"] == 0  # both daughters are at t=3
        assert len(tracks["line_indices"]) == 2

    def test_marker_and_track_sizes_match_the_constants(self, tmp_path) -> None:
        tracks = _demo.track_geometry(self._store(tmp_path), np.zeros(3), 4)
        assert np.all(tracks["point_radii"] == _demo.CELL_MARKER_RADIUS_UM)
        assert np.all(tracks["line_widths"] == _demo.TRACK_WIDTH_UM)
        assert len(tracks["line_widths"]) == len(tracks["line_vertices"])


class TestFileManifests:
    """What the demo asks Kaggle for — split so a warm cache skips the bulk."""

    def test_metadata_manifest_has_the_graph_and_image_metadata(self) -> None:
        paths = _demo._crop_metadata_files("crop_x")
        assert "train/crop_x.geff/zarr.json" in paths
        assert "train/crop_x.zarr/zarr.json" in paths
        assert "train/crop_x.zarr/0/zarr.json" in paths
        # every spatial axis plus time, values array and its single chunk
        for axis in ("t", "z", "y", "x"):
            assert f"train/crop_x.geff/nodes/props/{axis}/values/c/0" in paths
        assert not any("/0/c/" in p for p in paths), "must contain NO image chunks"

    def test_chunk_manifest_is_one_file_per_timepoint(self) -> None:
        paths = _demo._crop_chunk_files("crop_x", 100)
        assert len(paths) == 100
        assert paths[0] == "train/crop_x.zarr/0/c/0/0/0/0"
        assert paths[-1] == "train/crop_x.zarr/0/c/99/0/0/0"
        assert len(set(paths)) == len(paths)

    def test_manifests_are_disjoint(self) -> None:
        meta = set(_demo._crop_metadata_files("c"))
        chunks = set(_demo._crop_chunk_files("c", 10))
        assert not (meta & chunks)

    def test_timepoint_count_is_honoured(self) -> None:
        assert len(_demo._crop_chunk_files("c", 7)) == 7


class TestCacheBookkeeping:
    def test_everything_cached_lives_under_the_declared_cache(self) -> None:
        """`luxar demo cache list/clear` only sees the declared directory."""
        from luxar.demos.registry import DEMO_CACHE_ROOT

        claimed = _demo.DEMO_META["caches"]
        assert claimed == ["gsplats_cell_tracking"]
        root = DEMO_CACHE_ROOT / claimed[0]
        for path in (_demo.CACHE_DIR, _demo.DATA_DIR, _demo.FITS_DIR):
            assert path == root or root in path.parents, f"{path} escapes {root}"

    def test_fit_cache_path_encodes_the_splat_budget(self) -> None:
        """Two budgets must not collide in the cache, or a re-run loads the wrong fit."""
        path = _demo._fit_cache_path("crop_x", 7)
        assert path.name.startswith("t0007_k")
        assert str(_demo.SEEDS) in path.name
        assert path.parent.name == "crop_x"

    def test_is_fit_cached_rejects_an_interrupted_save(
        self, tmp_path, monkeypatch
    ) -> None:
        """A leftover .tmp marker means the store may be truncated — refit instead."""
        monkeypatch.setattr(_demo, "FITS_DIR", tmp_path)
        target = _demo._fit_cache_path("crop_x", 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not-really-a-store")
        assert _demo.is_fit_cached("crop_x", 0) is True

        marker = target.with_suffix(target.suffix + ".tmp")
        marker.touch()
        assert _demo.is_fit_cached("crop_x", 0) is False

    def test_needs_fitting_is_false_only_when_every_timepoint_is_cached(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(_demo, "FITS_DIR", tmp_path)
        monkeypatch.setitem(_demo.FLAGS, "recompute", False)
        assert _demo.needs_fitting("crop_x", 3) is True

        for t in range(3):
            p = _demo._fit_cache_path("crop_x", t)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        assert _demo.needs_fitting("crop_x", 3) is False

        # --recompute must force a refit even off a complete cache.
        monkeypatch.setitem(_demo.FLAGS, "recompute", True)
        assert _demo.needs_fitting("crop_x", 3) is True


class TestCuratedCrops:
    def test_nine_distinct_crops_covering_both_embryos(self) -> None:
        crops = _demo.DATASETS
        assert len(crops) == 9
        assert len(set(crops)) == 9
        prefixes = {c.split("_")[0] for c in crops}
        assert prefixes == {"6bba", "44b6"}, "the matrix should show both embryos"

    def test_the_drawn_level_lands_in_the_measured_quality_band(self) -> None:
        """The matrix draws — and fetches — the COARSEST level; pin its SIZE.

        The screen-area selector anchors the finest level at half the screen and
        halves per step, so a matrix tile lands below the lowest threshold and
        the coarsest level is what the demo opens on. Its absolute count per
        timepoint sets both the first paint and the cost of scrubbing time.

        MEASURED, not derived. The viewer reports `activeLevel: 0` (coarsest) for
        every group at the shipped 3x2, drawing 29,774 splats per frame across
        six crops. A naive `1/(cols*rows*GRID_GAP_FACTOR^2)` puts a 3x2 tile at
        0.128 — just OVER the 0.125 threshold — and would predict one level
        finer, so that formula is not the ground truth and is deliberately not
        asserted here: the camera frames the matrix with padding and the viewport
        aspect does not match the grid's. Re-read
        `__luxarDebug.getState().lodGroups[].activeLevel` after any grid change.

        The band comes from rendering merged levels back to the volume and
        looking: below ~3k/timepoint the nuclei smear together (the original
        blur), above ~12k the extra splats are barely distinguishable while the
        per-frame cost keeps doubling — 20k/tp made scrubbing choppy.

        Deliberately an ABSOLUTE count, not a fraction of the fit: a fraction
        says nothing without the fit size, which is exactly how the earlier
        version of this test passed while the demo was drawing 20k/tp.
        """
        cols, rows = _demo.grid_shape(_demo.DEFAULT_N_DATASETS)
        assert cols * rows == _demo.DEFAULT_N_DATASETS, (
            f"the shipped {cols}x{rows} grid has holes; the default crop count "
            "should fill its rectangle exactly"
        )

        # The fit keeps ~70% of the requested seeds, minus the p95 size filter.
        fitted_per_tp = _demo.DEFAULT_SEEDS * 0.70 * (_demo.SCALE_MAX_PERCENTILE / 100)
        drawn_per_tp = fitted_per_tp / (_demo.LOD_COMPRESSION_FACTOR**_demo.LOD_LEVELS)
        assert 3_000 <= drawn_per_tp <= 12_000, (
            f"a tile would draw ~{drawn_per_tp:,.0f} splats/timepoint, outside the "
            "measured 3k-12k band: below it the nuclei smear, above it the frame "
            "cost doubles for no visible gain. Adjust LOD_LEVELS."
        )
        per_frame = drawn_per_tp * _demo.DEFAULT_N_DATASETS
        assert per_frame <= 60_000, (
            f"the whole matrix would draw ~{per_frame:,.0f} splats per frame; "
            "120k was visibly choppy when scrubbing time"
        )


class TestPrecomputedRoundTrip:
    """The hosted derived product: what `save_` writes, `load_` must rebuild.

    Both halves of the format live in the demo module precisely so they cannot
    drift; these tests are what proves it. No network — the manifest and cache
    root are injected, which is what `ensure_dataset` exposes them for.
    """

    @staticmethod
    def _tiny_gsplats(n: int = 32, n_timepoints: int = 4):
        """A minimal 4D (ZYX + time) GSplatData, cheap enough for a unit test."""
        from luxar.gsplats import GSplatData

        rng = np.random.default_rng(0)
        centers = np.zeros((n, 4), dtype=np.float32)
        centers[:, :3] = rng.uniform(-10.0, 10.0, size=(n, 3))
        centers[:, 3] = rng.integers(0, n_timepoints, size=n)
        # Packed lower-triangular Cholesky for 4 dims = 10 entries; a diagonal
        # factor keeps it positive-definite. The time column gets a tiny sigma:
        # the splats are instantaneous.
        chol = np.zeros((n, 10), dtype=np.float32)
        for col, idx in enumerate((0, 2, 5, 9)):
            chol[:, idx] = 1.0 if col < 3 else 1e-3
        return GSplatData(
            centers=centers,
            amplitudes=rng.uniform(0.05, 1.0, size=n).astype(np.float32),
            cholesky_factors=chol,
        )

    def _crop(self, name: str = "crop_x", n_timepoints: int = 4) -> dict:
        n_pts, n_verts, n_edges = 6, 10, 7
        rng = np.random.default_rng(1)
        return {
            "name": name,
            "lod": self._tiny_gsplats(n_timepoints=n_timepoints),
            "n_splats": 32,
            "intensity": 1.5,
            "offset": -0.25,
            "tracks": {
                "point_positions": rng.uniform(size=(n_pts, 4)).astype(np.float32),
                "point_colors": rng.uniform(size=(n_pts, 3)).astype(np.float32),
                # Deliberately NOT the shipped constants: the loader re-derives
                # both from CELL_MARKER_RADIUS_UM / TRACK_WIDTH_UM, and a fixture
                # that already matched them could not tell that apart from a
                # faithful round-trip. See test_sizes_are_rederived_not_stored.
                "point_radii": np.full(n_pts, 9.5, np.float32),
                "line_vertices": rng.uniform(size=(n_verts, 4)).astype(np.float32),
                "line_colors": rng.uniform(size=(n_verts, 3)).astype(np.float32),
                "line_widths": np.full(n_verts, 9.5, np.float32),
                "line_indices": rng.integers(
                    0, n_verts, size=(n_edges, 2), dtype=np.uint32
                ),
                "n_lineages": 3,
                "n_cells": n_pts,
                "n_divisions": 1,
            },
            "extent_um": 104.0,
        }

    def _manifest(self, files: list) -> dict:
        return {
            "records": {"cc-by": {}},
            "datasets": {
                _demo.PRECOMPUTED_DATASET: {
                    "bucket": "zenodo",
                    "record": "cc-by",
                    "dir": "",
                    "files": [
                        {"name": p.name, "bytes": p.stat().st_size} for p in files
                    ],
                }
            },
        }

    def _stage(self, crop: dict, tmp_path, n_timepoints: int = 4):
        """Write a crop straight into the cache location ensure_dataset resolves.

        With no sha256 in the synthetic manifest and the bytes already cached,
        nothing is downloaded or verified — the test exercises the FORMAT.
        """
        cache_root = tmp_path / "cache"
        staged = cache_root / _demo.PRECOMPUTED_DATASET
        written = _demo.save_precomputed_crop(crop, staged, n_timepoints)
        return cache_root, written

    def test_saved_crop_is_reloaded_faithfully(self, tmp_path) -> None:
        original = self._crop(n_timepoints=4)
        cache_root, written = self._stage(original, tmp_path)
        assert [p.name for p in written] == list(_demo.precomputed_file_names("crop_x"))

        crops = _demo.load_precomputed_crops(
            ["crop_x"], manifest=self._manifest(written), cache_root=cache_root
        )
        assert crops is not None and len(crops) == 1
        got = crops[0]

        assert got["name"] == "crop_x"
        assert got["extent_um"] == pytest.approx(104.0)
        assert got["n_timepoints"] == 4
        assert got["lod"].n_splats == original["lod"].n_splats
        # `point_radii` / `line_widths` are appearance re-derived from constants,
        # not data — they are covered by test_sizes_are_rederived_not_stored.
        rederived = {"point_radii", "line_widths"}
        for key, want in original["tracks"].items():
            if key in rederived:
                continue
            if isinstance(want, int):
                assert got["tracks"][key] == want, key
            else:
                np.testing.assert_allclose(got["tracks"][key], want, err_msg=key)

    def test_sizes_are_rederived_not_stored(self, tmp_path) -> None:
        """Retuning marker/track size must not require re-uploading the bundle.

        Both fields are uniform, so they carry no measured information; storing
        them would pin appearance to a 0.68 GB hosted artifact. The staged crop
        carries 9.5 for both — the loader must return the constants instead, at
        the payload's own lengths.
        """
        original = self._crop()
        cache_root, written = self._stage(original, tmp_path)
        (got,) = _demo.load_precomputed_crops(
            ["crop_x"], manifest=self._manifest(written), cache_root=cache_root
        )
        tracks = got["tracks"]

        assert np.all(tracks["point_radii"] == _demo.CELL_MARKER_RADIUS_UM)
        assert np.all(tracks["line_widths"] == _demo.TRACK_WIDTH_UM)
        # The stored values really were different, so this is not vacuous.
        assert not np.allclose(tracks["line_widths"], original["tracks"]["line_widths"])
        # One entry per element, matching the geometry that WAS loaded.
        assert len(tracks["point_radii"]) == len(tracks["point_positions"])
        assert len(tracks["line_widths"]) == len(tracks["line_vertices"])

    def test_window_is_rederived_not_stored(self, tmp_path) -> None:
        """The display window follows the hosted amplitudes, per display_window."""
        cache_root, written = self._stage(self._crop(), tmp_path)
        (got,) = _demo.load_precomputed_crops(
            ["crop_x"], manifest=self._manifest(written), cache_root=cache_root
        )
        expected = _demo.display_window(got["lod"].amplitudes)
        assert got["intensity"] == pytest.approx(expected[0])
        assert got["offset"] == pytest.approx(expected[1])

    def test_recompute_forces_the_local_path(self, tmp_path, monkeypatch) -> None:
        cache_root, written = self._stage(self._crop(), tmp_path)
        monkeypatch.setitem(_demo.FLAGS, "recompute", True)
        assert (
            _demo.load_precomputed_crops(
                ["crop_x"], manifest=self._manifest(written), cache_root=cache_root
            )
            is None
        )

    def test_an_unpublished_record_falls_back_instead_of_raising(
        self, tmp_path
    ) -> None:
        """Pinned draft files have no public URL until the record is published."""
        files = [
            {"name": name, "bytes": 1}
            for name in _demo.precomputed_file_names("crop_x")
        ]
        manifest = {
            "records": {
                "cc-by": {
                    "zenodo_record": 21912280,
                    "published": False,
                }
            },
            "datasets": {
                _demo.PRECOMPUTED_DATASET: {
                    "bucket": "zenodo",
                    "record": "cc-by",
                    "dir": "",
                    "files": files,
                }
            },
        }
        assert (
            _demo.load_precomputed_crops(
                ["crop_x"], manifest=manifest, cache_root=tmp_path
            )
            is None
        )

    def test_an_uncovered_crop_bails_out_before_fetching_anything(
        self, tmp_path, monkeypatch
    ) -> None:
        """`--datasets 9` must not download the seven hosted crops to discard them.

        Coverage is decided from the manifest, which is a pure read. Asking
        ``ensure_dataset`` first and checking coverage afterwards downloaded the
        whole 632 MB set, rehydrated every crop that WAS present, and threw the
        lot away before starting the 3.2 GB Kaggle download — so this asserts the
        fetch never happens, not merely that the answer is ``None``.
        """
        covered = _demo.precomputed_file_names("crop_x")
        manifest = {
            "records": {"cc-by": {"id": 1, "published": True}},
            "datasets": {
                _demo.PRECOMPUTED_DATASET: {
                    "bucket": "zenodo",
                    "record": "cc-by",
                    "dir": "",
                    "files": [
                        {"name": n, "sha256": "0" * 64, "bytes": 1} for n in covered
                    ],
                }
            },
        }

        calls: list[str] = []

        def _must_not_run(*args, **kwargs):
            calls.append("ensure_dataset")
            raise AssertionError("ensure_dataset was called for an uncovered crop")

        import luxar.demos as _demos_pkg

        monkeypatch.setattr(_demos_pkg, "ensure_dataset", _must_not_run)

        # crop_y is not in the manifest's file list; crop_x is.
        assert (
            _demo.load_precomputed_crops(
                ["crop_x", "crop_y"], manifest=manifest, cache_root=tmp_path
            )
            is None
        )
        assert calls == [], "the uncovered crop still triggered a fetch"

    def test_a_fault_propagates_instead_of_triggering_kaggle_and_a_gpu_fit(
        self, tmp_path
    ) -> None:
        """Only a routable absence may be answered with the fallback (#1618).

        The fallback here is not cheap: an authenticated Kaggle download plus a
        multi-crop GPU fit. A dataset key the manifest does not carry is a typo
        or a rename — a fault — and used to come back as ``None`` while the
        expensive fallback was announced over it.
        """
        from luxar.demos._support.datasets.data_fetch import DatasetNotFound

        manifest = {"records": {"cc-by": {}}, "datasets": {}}
        with pytest.raises(DatasetNotFound):
            _demo.load_precomputed_crops(
                ["crop_x"], manifest=manifest, cache_root=tmp_path
            )

    def test_a_crop_without_annotation_round_trips_as_none(self, tmp_path) -> None:
        """`tracks: None` is what the scene builder reads as "volume only".

        Saving stores empty arrays for it, so rehydrating them as a dict of empty
        geometry would send zero-length vertices into `add_lines` instead.
        """
        crop = self._crop()
        crop["tracks"] = None
        cache_root, written = self._stage(crop, tmp_path)
        (got,) = _demo.load_precomputed_crops(
            ["crop_x"], manifest=self._manifest(written), cache_root=cache_root
        )
        assert got["tracks"] is None

    def test_a_missing_crop_falls_back_rather_than_half_building(
        self, tmp_path
    ) -> None:
        cache_root = tmp_path / "cache"
        staged = cache_root / _demo.PRECOMPUTED_DATASET
        staged.mkdir(parents=True)
        written = []
        for name in _demo.precomputed_file_names("crop_a"):
            path = staged / name
            path.write_bytes(b"must not be opened")
            written.append(path)

        got = _demo.load_precomputed_crops(
            ["crop_a", "crop_b"],
            manifest=self._manifest(written),
            cache_root=cache_root,
        )
        assert got is None, "a partially hosted set must not build a partial matrix"


class TestManifestRegistration:
    """The demo's hosted dataset must be registered the way the standard expects."""

    def test_dataset_is_registered_as_a_zenodo_bucket(self) -> None:
        from luxar.demos import dataset_spec

        spec = dataset_spec(_demo.PRECOMPUTED_DATASET)
        assert spec["bucket"] == "zenodo"
        assert spec["record"] in {"cc-by", "cc-by-sa", "h2afva"}
        assert spec["license"], "a redistributed derived product needs a license"
        assert spec["attribution"]

    def test_manifest_key_matches_the_declared_cache(self) -> None:
        """Otherwise the fetched bytes land in a directory no demo claims."""
        assert _demo.PRECOMPUTED_DATASET in _demo.DEMO_META["caches"]

    def test_one_file_set_at_full_timepoints_no_variants(self) -> None:
        """One hosted set, deliberately: a decimated variant would undercut the
        demo's whole subject (the 100-timepoint timelapse), and ~700 MB is modest
        for this catalogue."""
        from luxar.demos import dataset_spec

        spec = dataset_spec(_demo.PRECOMPUTED_DATASET)
        assert "variants" not in spec, "hosting one full set — no size variants"
        assert isinstance(spec["files"], list)

    def test_hosted_pair_names_are_stable_per_crop(self) -> None:
        """The manifest addresses these names, so they are part of the contract."""
        volume, tracks = _demo.precomputed_file_names("crop_x")
        assert volume == "crop_x.gsplats.zarr.zip"
        assert tracks == "crop_x_tracks.npz"
