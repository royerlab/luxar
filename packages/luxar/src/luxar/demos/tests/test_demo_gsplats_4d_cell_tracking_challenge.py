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
    """The matrix placement: N crops onto the smallest square grid that fits."""

    @pytest.mark.parametrize("n,side", [(1, 1), (2, 2), (4, 2), (5, 3), (7, 3), (9, 3)])
    def test_grid_side_is_ceil_sqrt(self, n: int, side: int) -> None:
        xforms = _demo.grid_transforms(n, 100.0)
        assert len(xforms) == n
        # Distinct column offsets reveal the side length actually used.
        xs = {round(_translation(x)[0], 4) for x in xforms}
        assert len(xs) == min(n, side)

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

    def test_lod_ladder_is_short_enough_to_survive_the_grid_framing(self) -> None:
        """A 3x3 tile draws the COARSEST level, so that level must keep enough.

        The screen-area selector anchors its finest level at half the screen and
        halves per step, and a tile of a 3x3 grid covers only
        1/(3*GRID_GAP_FACTOR)^2 of the screen area — below any ladder's lowest
        threshold. So what a tile actually draws is
        finest / compression^levels, and this pins that fraction at >= 1/4.
        """
        tile_area = 1.0 / (3.0 * _demo.GRID_GAP_FACTOR) ** 2
        assert tile_area < 0.125, "premise: a 3x3 tile is under the lowest threshold"
        coarsest_fraction = 1.0 / (_demo.LOD_COMPRESSION_FACTOR**_demo.LOD_LEVELS)
        assert coarsest_fraction >= 0.25, (
            f"a tile would draw only {coarsest_fraction:.3f} of the fitted splats; "
            "shorten the ladder or lower the compression factor"
        )
