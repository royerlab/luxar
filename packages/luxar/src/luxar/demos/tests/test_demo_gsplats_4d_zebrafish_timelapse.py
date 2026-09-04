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
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_4d_zebrafish_timelapse.py"
)
_MANIFEST_PATH = _DEMO_PATH.parent / "data_manifest.json"
# The component-filtered build, as pinned in the manifest.
_ARCHIVE_SHA256 = "b4c0cf690f6906414c449fb8c713b78ed7d5f5f88c270ab58cf741f019456f93"
_ARCHIVE_BYTES = 19_229_817


def _load_demo_module(name: str = "_luxar_demo_zebrafish_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def test_the_manifest_pins_the_component_filtered_build() -> None:
    """The pin is what a user downloads, and it can rot without anyone noticing.

    The read-back below only runs where a fetched copy is cached. This half runs
    everywhere: reverting the pin to the deposition's pre-component-filter
    upload would hand the NLM build to every hosted-path user and fail nothing
    else.
    """
    entry = json.loads(_MANIFEST_PATH.read_text())["datasets"][_demo.DEMO_NAME]
    pins = {f["name"]: (f["sha256"], f["bytes"]) for f in entry["files"]}
    assert pins.get(_demo.GSPLATS_FILE) == (_ARCHIVE_SHA256, _ARCHIVE_BYTES), (
        f"{_demo.GSPLATS_FILE} is pinned as {pins.get(_demo.GSPLATS_FILE)}, not the "
        f"component-filtered build ({_ARCHIVE_SHA256[:8]}…/{_ARCHIVE_BYTES:,} "
        "bytes); the fetch verifies downloads against this pin, so a stale one "
        "serves the old archive. A deliberate refit updates the manifest and both "
        "constants here together, plus the demo's `download_mb`."
    )


def test_the_record_archive_is_the_component_filtered_build() -> None:
    """Catch a stale precomputed archive whose preprocessing disagrees with code."""
    assert (
        _demo.SEEDS,
        _demo.FLOOR,
        _demo.MIN_COMPONENT_VOXELS,
        _demo.COMPONENT_CONNECTIVITY,
    ) == (32_000, "none", 4, 1), (
        "the record archive was fitted at these values, and the README, changelog "
        "and docstring tables quote them; changing one means refitting and reshipping"
    )
    # Match the fetch-cache copy on the pinned SIZE — a wrong copy staged by hand
    # into the record slot would otherwise be read back and fail as a splat-count
    # mismatch, which reads like a bad fit rather than the wrong file. (The demo's
    # own refits live in the sibling `local/` namespace and are never seen here.)
    archive_path = _demo.CACHE_DIR / _demo.GSPLATS_FILE
    if not archive_path.exists() or archive_path.stat().st_size != _ARCHIVE_BYTES:
        pytest.skip("the pinned zebrafish archive is not present in the fetch cache")

    from luxar.gsplats.gsplat_data import GSplatData

    archive = GSplatData.load(archive_path, include_stats=False)
    finest = archive.substitutive_levels[0]
    frame_zero_splats = sum(
        int(np.count_nonzero(np.isclose(sublod.centers[:, 3], 0.0)))
        for sublod in finest.additive_sublods
    )
    assert frame_zero_splats == 1_369, (
        f"record frame 0 has {frame_zero_splats:,} splats, not the measured "
        "component-filtered 1,369; the archive may still contain the NLM build"
    )


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

    @pytest.mark.parametrize("limit", [2, 3, 4, 8, 16, 32, 64, 100, 150])
    def test_a_subsample_reaches_the_end_of_the_recording(self, limit: int) -> None:
        """Span over count: a subsample must not stop two thirds of the way in.

        Rounding the stride DOWN maximises the frame count and can leave a third
        of the recording out — ``--max-timepoints=100`` of 151 took frames 0-99,
        the very defect this function replaced.
        """
        frames = _demo.select_timepoints(151, limit)
        # 0.953 is the true worst over every limit at n=151, so this is a bound
        # rather than a comfortable margin.
        assert frames[-1] / 150 >= 0.95, (
            f"limit {limit} stops at frame {frames[-1]} of 150, spanning only "
            f"{frames[-1] / 150:.0%} of the recording"
        )

    @pytest.mark.parametrize(
        "n_total,limit,expected",
        [
            # Rounding (n-1)/(limit-1) DOWN and truncating (the original) takes
            # frames 0-99 here and stops at 66% of the run.
            (151, 100, [0, 2, 150]),
            # Rounding it UP fixes that and breaks this one: 2 frames, 56%.
            (10, 3, [0, 4, 8]),
            (151, 8, [0, 21, 147]),
        ],
    )
    def test_the_stride_beats_both_closed_forms(
        self, n_total: int, limit: int, expected: list[int]
    ) -> None:
        """Pinned by example, because neither rounding rule is good enough.

        Each row is a case where one of the two obvious closed forms loses. The
        expected values are (first, second, last) of the chosen sampling.
        """
        frames = _demo.select_timepoints(n_total, limit)
        assert [frames[0], frames[1], frames[-1]] == expected, frames
        assert len(frames) <= limit

    def test_no_stride_would_fit_more_frames_in_the_budget(self) -> None:
        """The search must actually find the optimum, not merely something legal."""
        for n_total in (10, 37, 100, 151):
            for limit in range(2, min(n_total, 40)):
                frames = _demo.select_timepoints(n_total, limit)
                best = max(
                    (len(f), f[-1])
                    for f in (list(range(0, n_total, s)) for s in range(1, n_total))
                    if len(f) <= limit
                )
                assert (len(frames), frames[-1]) == best, (
                    f"n={n_total} limit={limit}: chose {len(frames)} frames "
                    f"ending at {frames[-1]}, but {best} was available"
                )

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


class TestTheManifestCarriesTheArchiveTheDemoAsksFor:
    """The default path resolves a NAME through the manifest, so they must agree.

    Nothing else in the suite pairs the two: ``test_demo_fetch_path`` checks the
    SHAPE of a demo's fetch call, not that the name it passes exists. A demo can
    therefore ask for an archive the manifest does not list, stay green in CI,
    and raise ``FileNotFoundError`` for every user on its default path while the
    ``--recompute`` path an author exercises works perfectly.
    """

    def test_the_requested_archive_is_a_manifest_file(self) -> None:
        import json

        manifest = json.loads(
            (Path(_DEMO_PATH).resolve().parents[0] / "data_manifest.json").read_text()
        )
        entry = manifest["datasets"][_demo.DEMO_NAME]
        names = [f["name"] for f in entry.get("files", [])]
        assert _demo.GSPLATS_FILE in names, (
            f"{_demo.DEMO_NAME} does not list {_demo.GSPLATS_FILE!r} (has "
            f"{names}); the demo's default path would raise FileNotFoundError "
            "for every user. Land the archive and regenerate the manifest."
        )

    def test_the_dataset_directory_is_the_demo_cache_name(self) -> None:
        import json

        manifest = json.loads(
            (Path(_DEMO_PATH).resolve().parents[0] / "data_manifest.json").read_text()
        )
        assert manifest["datasets"][_demo.DEMO_NAME]["dir"] == _demo.DEMO_NAME


class TestVoxelToMicrons:
    """A fitted centre is a voxel INDEX; the cage is the imaged BLOCK."""

    def test_each_axis_is_scaled_by_its_own_voxel_pitch(self, monkeypatch) -> None:
        """A geometric-mean check would pass a permuted or partial scaling."""
        from luxar.gsplats.gsplat_data import GSplatData

        shape = (8, 16, 16)
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        # One isotropic unit-sigma splat: Cholesky diag of 1 per axis (Σ = L·Lᵀ).
        fit = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
        )
        scaled = _demo.to_microns(fit)
        ratio = scaled.marginal_sigmas()[0] / fit.marginal_sigmas()[0]
        assert np.allclose(ratio, _demo.VOXEL_SIZE_ZYX_UM, rtol=1e-4), (
            f"axes scaled by {ratio}, not by the voxel size {_demo.VOXEL_SIZE_ZYX_UM}"
        )

    def test_the_grid_maps_symmetrically_inside_the_box(self, monkeypatch) -> None:
        """Half a voxel of slack at BOTH faces, not zero at one and a whole at the other.

        A centre is a voxel index and voxel i spans [i, i+1) of the block, so
        index 0 belongs half a voxel inside bmin. Getting this wrong puts any
        splat at a slightly negative index outside the scene's declared range.
        """
        from luxar.gsplats.gsplat_data import GSplatData

        shape = (8, 16, 16)
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        voxel = np.asarray(_demo.VOXEL_SIZE_ZYX_UM)
        corners = np.array(
            [[0.0, 0.0, 0.0], [s - 1.0 for s in shape], [-0.5, -0.5, -0.5]],
            dtype=np.float32,
        )
        mapped = _demo.to_microns(
            GSplatData(
                centers=corners,
                amplitudes=np.ones(3, dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
                ),
            )
        ).centers
        bmin, bmax = _demo.acquisition_box_um()

        assert np.allclose(mapped[0] - bmin, 0.5 * voxel, rtol=1e-4)
        assert np.allclose(bmax - mapped[1], 0.5 * voxel, rtol=1e-4)
        # The half-index below the first voxel is exactly the block's face.
        assert np.allclose(mapped[2], bmin, atol=1e-3)
        # And the mapping is centred: first and last voxel are mirror images.
        assert np.allclose(mapped[0], -mapped[1], atol=1e-3)


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
        bmin, bmax, (verts, _, widths) = cage
        thick = widths == _demo.BOX_EDGE_WIDTH_UM
        assert int(thick.sum()) == 24, "12 edges x 2 vertices"
        # ...and they are the first 24, which is what pairs them correctly.
        assert np.all(thick[:24])

        # Counting 24 thick vertices would also pass for twelve DIAGONALS, so
        # identify each edge by the pair of corners it joins and demand exactly
        # the twelve real ones. In `_corners` axis a is bit (2 - a), so two
        # corners share an edge iff their ids differ in exactly one bit.
        segs = verts[:24].reshape(-1, 2, 3)
        corner_id = lambda p: tuple(  # noqa: E731
            int(np.isclose(p[a], bmax[a], atol=1e-4)) for a in range(3)
        )
        drawn = {frozenset((corner_id(s[0]), corner_id(s[1]))) for s in segs}
        expected = {
            frozenset((c, tuple(v ^ (i == a) for i, v in enumerate(c))))
            for c in itertools.product((0, 1), repeat=3)
            for a in range(3)
        }
        assert drawn == expected, "the twelve edges are not the box's twelve edges"

    def test_every_ruling_lies_in_a_face_and_spans_it(self, cage) -> None:
        """A ruling that floats inside the box, or stops short, is not a ruler."""
        bmin, bmax, (verts, _, widths) = cage
        rulings = verts.reshape(-1, 2, 3)[
            widths.reshape(-1, 2)[:, 0] == _demo.GRID_LINE_WIDTH_UM
        ]
        assert len(rulings) > 0
        for seg in rulings:
            varying = np.flatnonzero(np.abs(seg[0] - seg[1]) > 1e-4)
            assert len(varying) == 1, "a ruling must be axis-aligned"
            axis = int(varying[0])
            pinned = [a for a in range(3) if a != axis]
            assert any(
                np.isclose(seg[0][a], bmin[a], atol=1e-4)
                or np.isclose(seg[0][a], bmax[a], atol=1e-4)
                for a in pinned
            ), "a ruling must lie in one of the six faces"
            lo, hi = sorted((seg[0][axis], seg[1][axis]))
            assert np.isclose(lo, bmin[axis], atol=1e-4)
            assert np.isclose(hi, bmax[axis], atol=1e-4)

    def test_the_grid_is_anchored_at_the_centre_not_at_a_corner(self, cage) -> None:
        """The docstring's claim: a reading off the grid is a signed distance."""
        _, _, (verts, _, widths) = cage
        rulings = verts.reshape(-1, 2, 3)[
            widths.reshape(-1, 2)[:, 0] == _demo.GRID_LINE_WIDTH_UM
        ]
        # Some ruling must sit at 0 on its own axis, which only holds for a
        # grid anchored at the origin.
        at_origin = [
            s
            for s in rulings
            if np.any(
                np.isclose(s[0], 0.0, atol=1e-4) & np.isclose(s[1], 0.0, atol=1e-4)
            )
        ]
        assert at_origin, "no ruling passes through the centre"

    def test_colours_are_linearized_and_the_edges_read_brighter(self, cage) -> None:
        """Demo colours are authored sRGB and consumed as LINEAR light."""
        _, _, (verts, colors, widths) = cage
        edge = colors[widths == _demo.BOX_EDGE_WIDTH_UM][0]
        grid = colors[widths == _demo.GRID_LINE_WIDTH_UM][0]
        assert np.allclose(edge, np.float32(_demo.BOX_EDGE_COLOR) ** 2.2, atol=1e-6)
        assert np.allclose(grid, np.float32(_demo.GRID_LINE_COLOR) ** 2.2, atol=1e-6)
        assert edge.mean() > grid.mean(), "the box must read brighter than the grid"

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

        times = [t * _demo.AXIS_STEP_MIN for t in range(len(fits))]
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
        # The fragile half. MEMORY re-quantizes centres to per-axis uint16, and
        # the time column survives exactly only because the encoder detects a
        # gridded axis and snaps to it. If that detection regresses, every
        # slider stop goes empty — and everything above here stays green.
        assert np.allclose(np.unique(reloaded[0].centers[:, 3]), times), (
            "the time coordinate did not survive the archive's quantization"
        )

    def test_time_is_a_hard_barrier_at_every_coarse_level(self, toy, monkeypatch):
        """A coarse level that merged across time would smear the recording.

        Checked on the LADDER rather than on the archive, because this is a
        property of the reduction: with ``coarsen_dims=(0, 1, 2)`` every level
        must still carry every timepoint, however few splats it has left.

        The assertion compares the VALUES, not how many there are. Counting is
        the trap: with the barrier off, a coarse level of three timepoints comes
        out at ``[0, 1.84, 4]`` — two frames averaged into one — which is still
        three distinct values and passes a count test. Measured, with both
        ``coarsen_dims=(0, 1, 2, 3)`` and ``coarsen_dims=None``.
        """
        shape, fits, _ = toy
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")

        times = [t * _demo.AXIS_STEP_MIN for t in range(len(fits))]
        laddered = _demo.build_lod(_demo.combine_to_4d(fits, times))
        for level in laddered.substitutive_levels:
            centers = np.concatenate([sub.centers for sub in level.additive_sublods])
            present = np.unique(np.round(centers[:, 3], 4))
            assert np.allclose(present, times), (
                f"level {level.level_index} carries times {present}, not "
                f"{times}; coarsening crossed the time barrier and averaged "
                "frames together"
            )

    def test_a_single_timepoint_archive_is_refused_by_name(self, toy, monkeypatch):
        """Not a timelapse — say so, rather than divide by a zero step.

        Without the guard this reaches `Dimension(step=0.0)`, which raises
        "Step size must be positive" several frames later, after a numpy
        divide-by-zero warning and with nothing in the message about time.
        """
        shape, fits, tmp = toy
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")

        one = _demo.combine_to_4d(fits[:1], [0.0])
        with pytest.raises(ValueError, match="at least 2 timepoints"):
            _demo.create_luxar_scene(one, tmp / "one.luxar.zarr")

    def test_a_uniform_time_grid_may_start_off_the_zero_anchor(self, tmp_path) -> None:
        """The declared minimum, not zero, anchors discrete time navigation."""
        import zarr

        from luxar.gsplats.gsplat_data import GSplatData

        centers = np.zeros((64, 4), dtype=np.float32)
        centers[:, 0] = np.arange(64) % 4
        centers[:, 1] = (np.arange(64) // 4) % 4
        centers[:, 2] = (np.arange(64) // 16) % 4
        centers[:, 3] = np.repeat([5.0, 15.0, 25.0, 35.0], 16)
        cholesky = np.zeros((64, 10), dtype=np.float32)
        cholesky[:, [0, 2, 5, 9]] = 1.0
        stacked = _demo.build_lod(
            GSplatData(
                centers=centers,
                amplitudes=np.ones(64, dtype=np.float32),
                cholesky_factors=cholesky,
            )
        )

        out = _demo.create_luxar_scene(stacked, tmp_path / "offset-time.luxar.zarr")
        root = zarr.open_group(str(out), mode="r")
        time = dict(root.attrs)["scene_dimensions"]["dimensions"][3]
        assert time["range"] == pytest.approx([5.0, 35.0])
        assert time["step"] == pytest.approx(10.0)

    def test_the_scene_declares_the_cage_box_and_a_gridded_time_axis(
        self, toy, monkeypatch
    ):
        import zarr

        shape, fits, tmp = toy
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        monkeypatch.setattr(_demo, "GRID_STEP_UM", 10.0)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 0)

        times = [t * _demo.AXIS_STEP_MIN for t in range(len(fits))]
        laddered = _demo.build_lod(_demo.combine_to_4d(fits, times))
        out = _demo.create_luxar_scene(laddered, tmp / "scene.luxar.zarr")

        root = zarr.open_group(str(out), mode="r")
        assert {"endoderm", "acquisition cage"} <= set(root.group_keys())
        assert "connected components" not in root.attrs["description"]

        dims = {
            d["name"]: d for d in dict(root.attrs)["scene_dimensions"]["dimensions"]
        }
        # Lateral first: this list's ORDER is what the viewer maps to screen
        # x/y/z, and Z first would show a 859x859x316 um slab edge-on as a tall
        # narrow column. It deliberately does NOT match the centre-column order,
        # which `dim_order` maps by name.
        assert [*dims] == ["X", "Y", "Z", "Time"], "the screen axes are misordered"

        bmin, bmax = _demo.acquisition_box_um()
        for axis, lo, hi in zip("ZYX", bmin, bmax):
            assert dims[axis]["range"] == pytest.approx([lo, hi]), (
                f"{axis} is declared over the splats, not the imaged block"
            )

        time = dims["Time"]
        assert time["unit"] == "min" and time["discrete"]
        assert time["step"] == pytest.approx(_demo.AXIS_STEP_MIN)
        # Exactly representable, and so is every stop up to the last: a range
        # input snaps onto min + k*step, and with the raw recorded interval the
        # browser clamps the final timepoint out of reach.
        assert all(
            (i * _demo.AXIS_STEP_MIN) == pytest.approx(t, abs=0.0)
            for i, t in enumerate(times)
        )
        assert time["range"] == pytest.approx([times[0], times[-1]])

        cage = dict(root["acquisition cage"].attrs)
        assert cage["extend_to_all"] == ["Time"], (
            "the cage is the instrument, not the specimen: it must survive "
            "every scrub rather than appear at one timepoint"
        )

    def test_the_scene_records_the_filter_when_enabled(self, toy, monkeypatch):
        import zarr

        shape, fits, tmp = toy
        monkeypatch.setattr(_demo, "ACQUISITION_SHAPE_ZYX", shape)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 4)

        times = [t * _demo.AXIS_STEP_MIN for t in range(len(fits))]
        laddered = _demo.build_lod(_demo.combine_to_4d(fits, times))
        out = _demo.create_luxar_scene(laddered, tmp / "denoised.luxar.zarr")

        root = zarr.open_group(str(out), mode="r")
        assert (
            "connected components smaller than 4 voxels removed"
            in root.attrs["description"]
        )


class TestTheFitCacheKeyMovesWithEveryKnob:
    """A cached fit may only be reused by a run that would have produced it.

    An earlier version of this demo keyed its per-timepoint cache on the frame
    index alone. That is silent and expensive in exactly the wrong way: every
    sweep in the module docstring — the seed budget, the floor, the denoising
    strength — would have been served the first arm's fits and reported it as
    the answer for all of them.

    So this enumerates the knobs rather than spot-checking one, because the
    failure is introduced by ADDING a knob and forgetting the key, and a
    spot-check on the knobs that already exist cannot see that.
    """

    #: Every module constant the fit result depends on. A new one belongs here
    #: at the same time it starts being passed to ``fit_gaussian_splats``.
    KNOBS = {
        "SEEDS": 12_345,
        "N_ITERS": 999,
        "EARLY_STOP_PATIENCE": 7,
        "CULL_RETENTION": 0.5,
        "FLOOR": "p90",
        "MIN_COMPONENT_VOXELS": 7,
        "COMPONENT_CONNECTIVITY": 3,
    }

    #: Constants that reach the fit but deliberately stay OUT of the key, each
    #: for a reason that has to be written down. ``device`` is a *how*, not a
    #: *what*: the same config on CPU and on CUDA is meant to describe the same
    #: fit, and if it does not, the answer is to fix the fitter rather than to
    #: refit an entire timelapse per machine.
    EXEMPT = {
        # `device` is a *how*, not a *what*: the same config on CPU and on CUDA
        # is meant to describe the same fit, and if it does not, the answer is
        # to fix the fitter rather than refit a timelapse per machine.
        "DEVICE",
        # REFIT_ALL decides whether the cache is READ at all. Keying on it would
        # mean a forced refit writes to a different path than the run it is
        # meant to replace, so the stale entry would survive forever.
        "REFIT_ALL",
    }

    def test_the_frame_index_is_in_the_key(self) -> None:
        assert _demo._fit_cache_path(3) != _demo._fit_cache_path(4)

    @pytest.mark.parametrize("knob", sorted(KNOBS))
    def test_changing_a_knob_changes_the_path(self, knob: str, monkeypatch) -> None:
        before = _demo._fit_cache_path(0)
        monkeypatch.setattr(_demo, knob, self.KNOBS[knob])
        after = _demo._fit_cache_path(0)
        assert before != after, (
            f"{knob} does not appear in the fit cache key, so a run that "
            f"changes it silently reuses fits made with the old value"
        )

    def test_every_fitting_knob_is_covered_by_this_test(self) -> None:
        """The list above must not drift behind the call it mirrors.

        Reads the actual fitting and preprocessing calls and requires that each
        module constant they pass is one this test varies. Without this the
        parametrization silently stops covering new knobs.
        """
        import ast
        import inspect

        # Scan the whole fit path for module-level constants rather than only
        # the arguments of a named call. The previous version keyed on
        # `denoise_volume_array`, which vanished when NLM was replaced by a
        # component filter -- and a guard that silently stops finding its own
        # anchor guards nothing. A constant referenced ANYWHERE on this path can
        # change the result, so that is what has to be enumerated.
        module_consts = {
            name for name in vars(_demo) if name.isupper() and not name.startswith("_")
        }
        passed = {
            node.id
            for function in (
                _demo.fit_all_timepoints,
                _demo.fit_timepoint,
                _demo.denoise,
            )
            for node in ast.walk(ast.parse(inspect.getsource(function).lstrip()))
            if isinstance(node, ast.Name) and node.id in module_consts
        }
        assert passed, "found no constants on the fit path; this test has gone stale"
        assert set(self.KNOBS) <= passed, (
            f"{sorted(set(self.KNOBS) - passed)} are in KNOBS but no longer reach "
            "the fit, so the cache key advertises a distinction the fit ignores"
        )
        assert passed <= set(self.KNOBS) | self.EXEMPT, (
            f"{sorted(passed - set(self.KNOBS) - self.EXEMPT)} reach the fit "
            "but are not in KNOBS, so nothing checks they are in the cache "
            "key. Add them there, or to EXEMPT with a reason."
        )
        assert not (set(self.KNOBS) & self.EXEMPT), "a knob cannot also be exempt"


class TestTheComponentFilter:
    """Delete small objects, leave every larger one BIT-IDENTICAL.

    That second half is the whole reason this replaced NLM, so it is the part
    worth pinning: a smoothing filter cannot promise it, and the promise is what
    lets the demo claim zero cell cost without re-measuring per dataset.
    """

    @staticmethod
    def _volume() -> np.ndarray:
        # Sizes 1 and 3 (below the threshold of 4), size 4 sitting EXACTLY on
        # it, and an 8-voxel block well above it. The 4 is the one that matters:
        # it is what separates `< threshold` from `<= threshold`, and without it
        # an off-by-one changes nothing and no test notices.
        v = np.zeros((4, 10, 10), dtype=np.float32)
        v[0, 0, 0] = 0.9  # 1 voxel -> deleted
        v[0, 4, 0:3] = 0.7  # 3 voxels -> deleted
        v[0, 8, 0:4] = 0.6  # 4 voxels -> KEPT (boundary)
        v[2, 2:4, 2:4] = 0.5  # 8 voxels -> kept
        v[3, 2:4, 2:4] = 0.5
        return v

    def test_small_components_go_and_the_large_one_is_untouched(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 4)
        v = self._volume()
        out = _demo.denoise(v)
        assert out[0, 0, 0] == 0.0, "the 1-voxel speck survived"
        assert not out[0, 4, 0:3].any(), "the 3-voxel speck survived"
        assert np.array_equal(out[0, 8, 0:4], v[0, 8, 0:4]), (
            "the 4-voxel object was deleted; the threshold is exclusive, so an "
            "object OF exactly that size must be kept (`<`, not `<=`)"
        )
        block = (slice(2, 4), slice(2, 4), slice(2, 4))
        assert np.array_equal(out[block], v[block]), (
            "the 8-voxel object changed; a size filter must never alter an "
            "object it keeps -- that is the property NLM could not offer"
        )

    def test_the_peak_of_a_kept_object_is_exactly_preserved(self, monkeypatch) -> None:
        # The measured claim in the docstring is `cell_pk == 1.0000`, exactly --
        # so check every KEPT object's own peak, not just the global max, which
        # a single surviving bright object would satisfy on its own.
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 4)
        v = self._volume()
        out = _demo.denoise(v)
        for name, sel in (
            ("4-voxel", (0, 8, slice(0, 4))),
            ("8-voxel", (slice(2, 4), slice(2, 4), slice(2, 4))),
        ):
            assert out[sel].max() == v[sel].max(), f"{name} object lost its peak"

    def test_connectivity_is_explicit_and_changes_diagonal_membership(
        self, monkeypatch
    ) -> None:
        volume = np.zeros((5, 5, 5), dtype=np.float32)
        diagonal = np.arange(5)
        volume[diagonal, diagonal, diagonal] = 1.0
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 4)

        monkeypatch.setattr(_demo, "COMPONENT_CONNECTIVITY", 1)
        assert not _demo.denoise(volume).any(), (
            "6-connectivity must treat a corner-touching diagonal as five specks"
        )

        monkeypatch.setattr(_demo, "COMPONENT_CONNECTIVITY", 3)
        assert np.array_equal(_demo.denoise(volume), volume), (
            "26-connectivity must treat the same diagonal as one 5-voxel object"
        )

    def test_the_input_is_not_mutated(self, monkeypatch) -> None:
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 4)
        v = self._volume()
        before = v.copy()
        _demo.denoise(v)
        assert np.array_equal(v, before), "denoise() edited its caller's array"

    @pytest.mark.parametrize("off", [0, 1])
    def test_a_threshold_below_two_is_a_no_op(self, off: int, monkeypatch) -> None:
        # Below 2 there is no component small enough to delete, so the filter
        # must hand the frame straight back rather than pay for a copy.
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", off)
        v = self._volume()
        assert _demo.denoise(v) is v

    def test_the_off_state_is_still_distinguishable_in_the_cache_key(
        self, monkeypatch
    ) -> None:
        # Otherwise a filtered and an unfiltered run share a cache, which is the
        # comparison the docstring's threshold table depends on being able to make.
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 0)
        off = _demo._fit_cache_path(0)
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 4)
        assert off != _demo._fit_cache_path(0)


class TestRecomputeResumes:
    """``--recompute`` rebuilds the archive but must not refit cached frames.

    The distinction is worth a test because it is invisible until a long run is
    interrupted, and then it costs the whole run. It is only safe because the
    cache key covers every knob that changes a fit — see
    ``TestTheFitCacheKeyMovesWithEveryKnob``, which is what this leans on.
    """

    def test_a_cached_frame_is_reused_under_recompute(self, monkeypatch) -> None:
        monkeypatch.setattr(_demo, "RECOMPUTE", True)
        monkeypatch.setattr(_demo, "REFIT_ALL", False)
        sentinel = object()
        monkeypatch.setattr(
            _demo.GSplatData, "load", staticmethod(lambda *a, **k: sentinel)
        )
        monkeypatch.setattr(_demo, "_fit_cache_path", lambda frame: _Exists())
        assert _demo.fit_timepoint(None, 0, (None, None)) is sentinel

    def test_a_cached_frame_is_not_denoised(self, monkeypatch) -> None:
        monkeypatch.setattr(_demo, "REFIT_ALL", False)
        sentinel = _CachedFit()
        monkeypatch.setattr(
            _demo.GSplatData, "load", staticmethod(lambda *a, **k: sentinel)
        )
        monkeypatch.setattr(_demo, "_fit_cache_path", lambda frame: _Exists())
        monkeypatch.setattr(
            _demo,
            "denoise",
            lambda volume: pytest.fail("a cached frame was denoised"),
        )
        monkeypatch.setattr(_demo, "report_fit_quality", lambda fits: None)
        array = np.zeros((1, 2, 3, 4), dtype=np.uint8)
        assert _demo.fit_all_timepoints(array, [0]) == [sentinel]

    def test_a_cold_frame_is_filtered_before_it_reaches_the_fitter(
        self, tmp_path, monkeypatch
    ) -> None:
        """A cache MISS must filter; the sibling test covers the hit skipping it.

        Asserted on the array the fitter actually receives, not on a call
        record, so it survives the filter being reimplemented.
        """
        import luxar.gsplats

        class FitReached(Exception):
            pass

        # One 1-voxel speck (must be gone) and one 8-voxel block (must remain).
        volume = np.zeros((2, 6, 6), dtype=np.float32)
        volume[0, 0, 0] = 0.9
        volume[0, 2:4, 2:4] = 0.5
        volume[1, 2:4, 2:4] = 0.5
        seen: list[np.ndarray] = []

        monkeypatch.setattr(_demo, "REFIT_ALL", False)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")
        monkeypatch.setattr(_demo, "MIN_COMPONENT_VOXELS", 4)
        monkeypatch.setattr(
            _demo, "_fit_cache_path", lambda frame: tmp_path / "missing.gsplats.zarr"
        )

        def stop_at_fit(input_volume, **kwargs):
            seen.append(np.asarray(input_volume).copy())
            raise FitReached

        monkeypatch.setattr(luxar.gsplats, "fit_gaussian_splats", stop_at_fit)
        with pytest.raises(FitReached):
            _demo.fit_timepoint(volume, 0, (volume.shape, str(volume.dtype)))

        assert len(seen) == 1
        got = seen[0]
        assert got[0, 0, 0] == 0.0, "the speck reached the fitter unfiltered"
        assert got[0, 2:4, 2:4].sum() == volume[0, 2:4, 2:4].sum(), (
            "the kept block was altered on its way to the fitter"
        )

    def test_refit_all_ignores_the_cache(self, monkeypatch) -> None:
        import luxar.gsplats

        class FitReached(Exception):
            pass

        read: list[str] = []
        fitted: list[str] = []
        monkeypatch.setattr(_demo, "REFIT_ALL", True)
        monkeypatch.setattr(_demo, "DEVICE", "cpu")
        monkeypatch.setattr(
            _demo.GSplatData,
            "load",
            staticmethod(lambda *a, **k: read.append("hit")),
        )
        monkeypatch.setattr(_demo, "_fit_cache_path", lambda frame: _Exists())
        monkeypatch.setattr(_demo, "denoise", lambda volume: volume)

        def stop_at_fit(*args, **kwargs):
            fitted.append("fit")
            raise FitReached

        monkeypatch.setattr(
            luxar.gsplats,
            "fit_gaussian_splats",
            stop_at_fit,
        )
        with pytest.raises(FitReached):
            _demo.fit_timepoint(None, 0, (None, None))
        assert read == [], "the cache was read despite --refit-all"
        assert fitted == ["fit"]


class _Exists:
    """A Path stand-in that is always present and never actually touched."""

    def exists(self) -> bool:
        return True

    def unlink(self, missing_ok: bool = False) -> None:
        pass


class _CachedFit:
    n_splats = 1
