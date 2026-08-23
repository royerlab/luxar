"""Tests for demo_gsplats_4d_zebrafish_timelapse.

No network, no LSM, no GPU. Two things are worth pinning here, and they are the
two the rewrite introduced:

* the **acquisition cage** — a wireframe the user reads distances off, so its
  geometry has to be right rather than merely present, and
* the **default path** — archive on disk, read back, scene built from it. The
  refit path is the one an author exercises; the read-back path is the one every
  user takes, and a topology that does not survive the round trip breaks only
  the second (see ``_lod_policy.TREE_RECIPES`` for the sibling of this trap).

The demo is loaded by file path, like its cell-tracking sibling.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_4d_zebrafish_timelapse.py"
)


def _load_demo_module(name: str = "_luxar_demo_zebrafish_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


class TestTimepointSelection:
    """``--max-timepoints`` must stay ON one step grid, or slider stops go empty."""

    def test_no_limit_takes_every_frame(self) -> None:
        assert _demo.select_timepoints(151, None) == list(range(151))

    def test_a_limit_at_or_above_the_run_takes_every_frame(self) -> None:
        assert _demo.select_timepoints(10, 10) == list(range(10))
        assert _demo.select_timepoints(10, 99) == list(range(10))

    @pytest.mark.parametrize("n_total,limit", [(151, 4), (151, 8), (151, 64), (100, 7)])
    def test_a_subsample_is_uniformly_spaced(self, n_total: int, limit: int) -> None:
        frames = _demo.select_timepoints(n_total, limit)
        gaps = np.diff(frames)
        assert len(frames) <= limit
        assert len(set(gaps.tolist())) == 1, (
            f"{frames} is not uniformly spaced; time is a DISCRETE viewer "
            "dimension whose navigation snaps to multiples of one step, so "
            "uneven frames land on stops that render nothing"
        )

    def test_frames_stay_inside_the_recording(self) -> None:
        assert max(_demo.select_timepoints(151, 64)) < 151

    def test_a_single_timepoint_is_refused(self) -> None:
        # One timepoint has no step to derive, and the scene it would build is
        # not a timelapse. Say so instead of writing a degenerate axis.
        with pytest.raises(ValueError, match="at least 2"):
            _demo.select_timepoints(151, 1)


class TestAcquisitionBox:
    """The cage is the imaged block, centred — that is what makes it a ruler."""

    def test_the_box_is_the_grid_times_the_voxel_size(self) -> None:
        bmin, bmax = _demo.acquisition_box_um()
        extent = bmax - bmin
        expected = np.asarray(_demo.VOXEL_SIZE_ZYX_UM) * np.asarray(
            _demo.ACQUISITION_SHAPE_ZYX
        )
        assert np.allclose(extent, expected)

    def test_the_box_is_centred_on_the_origin(self) -> None:
        bmin, bmax = _demo.acquisition_box_um()
        assert np.allclose(bmin, -bmax)


class TestCageLines:
    """``line_type="segments"`` means consecutive PAIRS are independent edges."""

    @pytest.fixture
    def cage(self):
        bmin = np.array([-10.0, -20.0, -20.0])
        bmax = np.array([10.0, 20.0, 20.0])
        return bmin, bmax, _demo.cage_lines(bmin, bmax, 10.0)

    def test_vertices_come_in_pairs_with_matching_colors_and_widths(self, cage) -> None:
        _, _, (verts, colors, widths) = cage
        assert verts.shape[1] == 3
        assert len(verts) % 2 == 0, "an odd vertex count leaves a dangling segment"
        assert len(colors) == len(verts) == len(widths)

    def test_every_vertex_lies_on_the_box(self, cage) -> None:
        bmin, bmax, (verts, _, _) = cage
        assert np.all(verts >= bmin - 1e-4) and np.all(verts <= bmax + 1e-4)
        # A cage vertex is on the SURFACE: at least one axis sits on a face.
        on_face = np.isclose(verts, bmin, atol=1e-4) | np.isclose(
            verts, bmax, atol=1e-4
        )
        assert np.all(on_face.any(axis=1))

    def test_the_twelve_box_edges_are_drawn_and_are_the_thick_ones(self, cage) -> None:
        _, _, (verts, _, widths) = cage
        thick = widths == _demo.BOX_EDGE_WIDTH_UM
        assert int(thick.sum()) == 24, "12 edges x 2 vertices"
        # ...and they are the first 24, which is what pairs them correctly.
        assert np.all(thick[:24])

    def test_grid_lines_land_on_multiples_of_the_step(self, cage) -> None:
        _, _, (verts, _, widths) = cage
        grid = verts[widths == _demo.GRID_LINE_WIDTH_UM]
        # Each grid segment is constant on two axes; both must be on the grid
        # (a face position or a ruling), which is what makes it readable.
        assert len(grid) > 0
        assert np.all(np.isclose(np.remainder(grid + 5.0, 10.0), 5.0, atol=1e-4))

    def test_a_step_larger_than_the_box_still_draws_the_box(self) -> None:
        """A step that does not fit must degrade to the box, not to nothing."""
        bmin, bmax = np.array([-1.0] * 3), np.array([1.0] * 3)
        verts, colors, widths = _demo.cage_lines(bmin, bmax, 1000.0)
        assert int((widths == _demo.BOX_EDGE_WIDTH_UM).sum()) == 24
        assert len(colors) == len(widths) == len(verts)
        assert np.all(verts >= bmin - 1e-4) and np.all(verts <= bmax + 1e-4)


class TestTheStackedArchiveSurvivesItsRoundTrip:
    """Fit -> stack -> ladder -> save -> READ BACK -> scene, on a toy timelapse.

    Small enough to run on a laptop CPU in seconds, and it covers the join that
    no static check can: the archive is written in one step of the demo and read
    in another, and only the read-back path is what a user runs.
    """

    @pytest.fixture(scope="class")
    def toy(self, tmp_path_factory):
        torch = pytest.importorskip("torch", reason="fitting needs luxar[gsplats]")
        del torch
        from scipy.ndimage import gaussian_filter

        from luxar.gsplats import fit_gaussian_splats

        shape = (6, 32, 32)
        rng = np.random.default_rng(0)
        fits = []
        for _ in range(3):
            volume = np.zeros(shape, dtype=np.float32)
            for _blob in range(5):
                idx = tuple(int(rng.integers(1, s - 1)) for s in shape)
                volume[idx] = 1.0
            volume = gaussian_filter(volume, 1.2)
            volume /= volume.max()
            fits.append(
                fit_gaussian_splats(
                    volume,
                    seeds=80,
                    n_iters=60,
                    cull_retention=_demo.CULL_RETENTION,
                    floor=_demo.FLOOR,
                    output_space="voxel",
                    device="cpu",
                    verbose=False,
                    source_shape=shape,
                    source_dtype="uint8",
                )
            )
        return shape, fits, tmp_path_factory.mktemp("zebrafish")

    def test_the_ladder_and_the_time_axis_survive_the_archive(self, toy, monkeypatch):
        from luxar.demos import load_local_fit_gsplats_at
        from luxar.encoding import EncodingMode

        shape, fits, tmp = toy
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")

        times = [t * _demo.FRAME_INTERVAL_S / 60.0 for t in range(len(fits))]
        laddered = _demo.build_lod(_demo.combine_to_4d(fits, times))
        assert laddered.n_substitutive > 1, "no coarse levels were built"

        archive = tmp / "zebrafish_4d.gsplats.zarr.zip"
        laddered.save(
            archive,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )
        reloaded = load_local_fit_gsplats_at([archive], label="test")
        assert reloaded is not None, "the demo's reader refused the archive it wrote"
        assert reloaded[0].ndim == 4
        assert reloaded[0].n_substitutive == laddered.n_substitutive

    def test_time_is_a_hard_barrier_at_every_coarse_level(self, toy, monkeypatch):
        """A coarse level that merged across time would smear the recording.

        Checked on the LADDER rather than on the archive, because this is a
        property of the reduction: with ``coarsen_dims=(0, 1, 2)`` every level
        must still carry every timepoint, however few splats it has left.
        """
        shape, fits, _ = toy
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")

        times = [t * _demo.FRAME_INTERVAL_S / 60.0 for t in range(len(fits))]
        laddered = _demo.build_lod(_demo.combine_to_4d(fits, times))
        for level in laddered.substitutive_levels:
            centers = np.concatenate([sub.centers for sub in level.additive_sublods])
            present = np.unique(np.round(centers[:, 3], 4))
            assert len(present) == len(times), (
                f"level {level.level_index} kept {len(present)} of {len(times)} "
                "timepoints; coarsening crossed the time barrier"
            )

    def test_the_scene_declares_the_cage_box_and_a_gridded_time_axis(
        self, toy, monkeypatch
    ):
        import zarr

        shape, fits, tmp = toy
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        monkeypatch.setattr(_demo, "GRID_STEP_UM", 10.0)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")

        times = [t * _demo.FRAME_INTERVAL_S / 60.0 for t in range(len(fits))]
        laddered = _demo.build_lod(_demo.combine_to_4d(fits, times))
        out = _demo.create_luxar_scene(laddered, tmp / "scene.luxar.zarr")

        root = zarr.open_group(str(out), mode="r")
        assert {"endoderm", "acquisition cage"} <= set(root.group_keys())

        dims = {
            d["name"]: d for d in dict(root.attrs)["scene_dimensions"]["dimensions"]
        }
        assert [*dims] == ["Z", "Y", "X", "Time"], "column order must match the stack"

        bmin, bmax = _demo.acquisition_box_um()
        for axis, lo, hi in zip("ZYX", bmin, bmax):
            assert dims[axis]["range"] == pytest.approx([lo, hi]), (
                f"{axis} is declared over the splats, not the imaged block"
            )

        time = dims["Time"]
        assert time["unit"] == "min" and time["discrete"]
        assert time["step"] == pytest.approx(_demo.FRAME_INTERVAL_S / 60.0)
        assert time["range"] == pytest.approx([times[0], times[-1]])

        cage = dict(root["acquisition cage"].attrs)
        assert cage["extend_to_all"] == ["Time"], (
            "the cage is the instrument, not the specimen: it must survive "
            "every scrub rather than appear at one timepoint"
        )
