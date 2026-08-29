"""Tests for the neuromast demo's per-channel recompute path.

Three things here are worth pinning, each with a quiet failure mode:

1. **The background subtraction.** A single measured floor per channel,
   subtracted with a clip at 0. Forgetting the clip leaves negative intensities,
   which the fitter reads as signal; re-measuring the floor instead of using the
   recorded one drifts the whole fit.
2. **Two sources, not one.** The channels were acquired and assembled
   separately, so the recompute needs a path per channel and the two floors are
   different. A single ``--source`` would fit one channel twice.
3. **The recipe constants.** ``--jobs-per-gpu`` must not be ``auto``: on the
   acquisition box ``auto`` sized 100 concurrent workers for 100 tasks and every
   one was OOM-killed before a tile landed.
"""

import numpy as np
import pytest

from luxar._zarr_compat import open_group
from luxar.demos import demo_gsplats_4d_neuromast_2ch as demo

SMALL = (4, 3, 5, 6)

#: Captured at IMPORT, before the autouse fixture below shrinks the module
#: constant. The constants tests must compare against the real recorded extent;
#: reading ``demo.SOURCE_SHAPE`` inside a test would read the 360-voxel stand-in
#: and assert nothing.
REAL_SOURCE_SHAPE = demo.SOURCE_SHAPE


@pytest.fixture(autouse=True)
def _small_shape(monkeypatch):
    """Run the subtraction against a 360-voxel array, not an 11 GB one."""
    monkeypatch.setattr(demo, "SOURCE_SHAPE", SMALL)


def _source(tmp_path, name="src", *, values=None, shape=SMALL):
    """A plain zarr array store whose ROOT is the array, as the sources are."""
    import zarr

    from luxar._zarr_compat import create_array

    group = open_group(tmp_path / f"{name}_store", mode="w")
    array = create_array(group, "root", shape=shape, dtype="float32", compressor=None)
    array[:] = (
        values
        if values is not None
        else np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    )
    # Re-open as an array-rooted store by pointing at the array itself, which is
    # what `zarr.open` on the real sources returns.
    del zarr
    return tmp_path / f"{name}_store" / "root"


class TestTheBackgroundSubtractionIsClippedAtZero:
    def test_values_below_the_floor_become_zero_not_negative(self, tmp_path):
        values = np.full(SMALL, 5.0, dtype=np.float32)
        values[0, 0, 0, 0] = 100.0
        src = _source(tmp_path, values=values)
        demo._subtract_background(src, tmp_path / "out.zarr", floor=10.0)
        out = open_group(tmp_path / "out.zarr", mode="r")[demo.BGSUB_ARRAY_KEY]
        data = np.asarray(out[:])
        assert data.min() == 0.0, "a negative intensity reads to the fitter as signal"
        assert data[0, 0, 0, 0] == pytest.approx(90.0)

    def test_the_floor_is_subtracted_not_divided_out(self, tmp_path):
        values = np.full(SMALL, 30.0, dtype=np.float32)
        src = _source(tmp_path, values=values)
        demo._subtract_background(src, tmp_path / "out.zarr", floor=10.0)
        out = open_group(tmp_path / "out.zarr", mode="r")[demo.BGSUB_ARRAY_KEY]
        assert np.asarray(out[:]).max() == pytest.approx(20.0)

    def test_the_floor_actually_used_is_recorded_on_the_output(self, tmp_path):
        src = _source(tmp_path)
        demo._subtract_background(src, tmp_path / "out.zarr", floor=7.25)
        out = open_group(tmp_path / "out.zarr", mode="r")[demo.BGSUB_ARRAY_KEY]
        assert dict(out.attrs)["background_floor"] == pytest.approx(7.25)
        assert dict(out.attrs)["axes"] == demo.SOURCE_AXES

    def test_every_timepoint_is_written_not_just_the_first(self, tmp_path):
        """The loop writes per timepoint; an off-by-one leaves the tail at zero,
        which looks like a dark stretch of the movie rather than an error."""
        values = np.full(SMALL, 20.0, dtype=np.float32)
        src = _source(tmp_path, values=values)
        demo._subtract_background(src, tmp_path / "out.zarr", floor=5.0)
        out = open_group(tmp_path / "out.zarr", mode="r")[demo.BGSUB_ARRAY_KEY]
        data = np.asarray(out[:])
        for t in range(SMALL[0]):
            assert data[t].min() == pytest.approx(15.0), f"timepoint {t} not written"


class TestAWrongShapedSourceIsRefused:
    def test_a_mismatched_extent_is_refused_before_the_gpu(self, tmp_path):
        src = _source(tmp_path, shape=(4, 3, 5, 7))
        with pytest.raises(SystemExit, match="want"):
            demo._subtract_background(src, tmp_path / "out.zarr", floor=1.0)

    def test_a_store_with_no_array_at_its_root_is_refused(self, tmp_path):
        open_group(tmp_path / "empty.zarr", mode="w")
        with pytest.raises(SystemExit, match="no array at its root"):
            demo._subtract_background(tmp_path / "empty.zarr", tmp_path / "o.zarr", 1.0)


class TestBothChannelsNeedTheirOwnSource:
    def test_a_missing_source_flag_names_every_channel_it_needs(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setitem(demo.SOURCE_ARGS, "membranes", None)
        monkeypatch.setitem(demo.SOURCE_ARGS, "nuclei", None)
        with pytest.raises(SystemExit) as exc:
            demo.recompute_channel_paths(tmp_path / "work")
        message = str(exc.value)
        assert "--source-membranes" in message
        assert "--source-nuclei" in message

    def test_one_source_alone_is_not_enough(self, monkeypatch, tmp_path):
        monkeypatch.setitem(demo.SOURCE_ARGS, "membranes", tmp_path / "m.zarr")
        monkeypatch.setitem(demo.SOURCE_ARGS, "nuclei", None)
        with pytest.raises(SystemExit) as exc:
            demo.recompute_channel_paths(tmp_path / "work")
        assert "--source-nuclei" in str(exc.value)
        assert "--source-membranes" not in str(exc.value)

    def test_the_two_channels_carry_different_floors(self):
        """They were assembled separately from different acquisitions, so a
        shared floor would be wrong for at least one of them."""
        floors = {ch["name"]: ch["background_floor"] for ch in demo.CHANNELS}
        assert len(set(floors.values())) == 2

    def test_the_two_channels_carry_different_expected_counts(self):
        counts = {ch["name"]: ch["expected_splats"] for ch in demo.CHANNELS}
        assert len(set(counts.values())) == 2

    def test_a_nonexistent_source_path_is_reported_with_its_flag(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setitem(demo.SOURCE_ARGS, "membranes", tmp_path / "nope.zarr")
        monkeypatch.setitem(demo.SOURCE_ARGS, "nuclei", tmp_path / "also-nope.zarr")
        with pytest.raises(FileNotFoundError, match="source-membranes"):
            demo.recompute_channel_paths(tmp_path / "work")


class TestTheRecipeConstantsMatchTheRecordedRun:
    def test_jobs_per_gpu_is_pinned_not_auto(self):
        """`auto` OOM-killed all 100 workers on the acquisition box."""
        assert demo.JOBS_PER_GPU == 12
        assert not isinstance(demo.JOBS_PER_GPU, str)

    def test_the_tile_size_keeps_the_real_volume_whole(self):
        """One tile per timepoint means no Hann halo and no seams to apodize.

        Written against the literal recorded extent, not ``demo.SOURCE_SHAPE``:
        this module's autouse fixture shrinks that constant, so reading it here
        would compare 640 against 6 and pass no matter what.
        """
        assert demo.TILE_SIZE >= max((84, 580, 576))
        assert REAL_SOURCE_SHAPE == (100, 84, 580, 576)

    def test_the_recorded_source_shape_is_the_full_recording(self):
        """The fixture must be shrinking the real constant, not a stale one."""
        assert REAL_SOURCE_SHAPE[0] == 100, "100 timepoints per channel"

    def test_the_stacked_axis_is_last_in_the_fitted_output(self):
        """The source is time-FIRST; the fit emits spatial-first, stacked-last.
        The Z scale therefore applies to index 0, not index 1."""
        assert demo.SOURCE_AXES.startswith("time")
        assert demo.VOXEL_SCALE[0] == 2.5
        assert demo.VOXEL_SCALE[1:] == (1.0, 1.0, 1.0)

    def test_the_cull_threshold_is_the_measured_one(self):
        """Chosen from a 0.02/0.05/0.10/0.20/0.35 sweep as the most aggressive
        setting still SSIM-flat."""
        assert demo.REDUNDANCY_THRESHOLD == 0.20

    def test_the_seed_budget_is_the_calibrated_k_star(self):
        assert demo.SEEDS == 64_000
        assert demo.PRESET == "n2s"
