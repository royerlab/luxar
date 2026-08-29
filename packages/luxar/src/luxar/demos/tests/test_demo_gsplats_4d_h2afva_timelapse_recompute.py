"""Tests for the h2afva timelapse demo's recompute path.

The path itself is a walk over 602M splats, so what is testable here is the
gate in front of it: ``--parent`` must be the 253-timepoint partition archive,
and it costs the better part of an hour to discover otherwise from the splat
count alone. A real sibling exists to be confused with it -- the
single-timepoint tp234 fit is also a ``kind=partition`` of the same recording
with the same appearance attrs -- and these tests pin that it is rejected.

The constants are also asserted against the shipped archive's own recorded
figures, so a drift in either direction is caught at test time rather than after
a rebuild.
"""

import pytest

from luxar.demos import demo_gsplats_4d_h2afva_timelapse as demo


def _attrs(monkeypatch, attrs):
    monkeypatch.setattr(demo, "_read_source_attrs", lambda _p: attrs)


def _parent_attrs(frames=253, kind="partition"):
    bounds = {"min": [0, 0, 0, 0], "max": [1624, 2038, 2046, float(frames - 1)]}
    return {"kind": kind, "position_bounds": bounds}


class TestTheRecordedParentIsAccepted:
    def test_the_253_timepoint_partition_passes(self, monkeypatch, tmp_path):
        _attrs(monkeypatch, _parent_attrs())
        demo._validate_parent(tmp_path / "h2afva_253tp.gsplats.zarr")  # must not raise

    def test_a_float_stacked_bound_is_rounded_not_truncated(
        self, monkeypatch, tmp_path
    ):
        """Stored bounds are float32, so 252 can read back as 251.99998.

        Truncating instead of rounding would read 251 timepoints as 252 -- which
        happens to still yield 51 here, so this only bites at a boundary. Pinned
        anyway because the failure would be a silently wrong frame count.
        """
        attrs = _parent_attrs()
        attrs["position_bounds"]["max"][demo.TIME_COL] = 251.99998
        _attrs(monkeypatch, attrs)
        demo._validate_parent(tmp_path / "p.gsplats.zarr")

    def test_a_251_frame_parent_also_passes(self, monkeypatch, tmp_path):
        """251 and 253 both yield 51 frames at stride 5, so the check must be on
        the YIELD, not on a pinned parent length."""
        _attrs(monkeypatch, _parent_attrs(frames=251))
        demo._validate_parent(tmp_path / "p.gsplats.zarr")

    def test_one_frame_too_many_is_refused(self, monkeypatch, tmp_path):
        """256 timepoints yields 52 -- the boundary the yield check exists for."""
        _attrs(monkeypatch, _parent_attrs(frames=256))
        with pytest.raises(SystemExit, match="yields 52 frames"):
            demo._validate_parent(tmp_path / "p.gsplats.zarr")


class TestTheWrongParentIsRefused:
    def test_the_single_timepoint_sibling_fit_is_refused(self, monkeypatch, tmp_path):
        """tp234 is also a partition of this recording with the same appearance
        attrs; the timepoint count is what distinguishes them."""
        _attrs(monkeypatch, _parent_attrs(frames=1))
        with pytest.raises(SystemExit) as exc:
            demo._validate_parent(tmp_path / "h2afva_stack.gsplats.zarr")
        assert "1 timepoints" in str(exc.value)
        assert "yields 1 frames" in str(exc.value)

    def test_the_51_frame_variant_itself_is_refused(self, monkeypatch, tmp_path):
        """Re-striding the output would give 11 frames, not 51 -- silently."""
        _attrs(monkeypatch, _parent_attrs(frames=51))
        with pytest.raises(SystemExit, match="51 timepoints"):
            demo._validate_parent(tmp_path / "h2afva_51tp.gsplats.zarr")

    def test_a_flattened_parent_is_refused_before_the_walk(self, monkeypatch, tmp_path):
        _attrs(monkeypatch, _parent_attrs(kind="leaf"))
        with pytest.raises(SystemExit, match="must be the 253tp partition"):
            demo._validate_parent(tmp_path / "flat.gsplats.zarr")

    def test_a_parent_without_position_bounds_is_refused_not_assumed(
        self, monkeypatch, tmp_path
    ):
        """Absent is not 'probably fine': without bounds there is no check, and
        the walk is an hour long."""
        _attrs(monkeypatch, {"kind": "partition"})
        with pytest.raises(SystemExit, match="no position_bounds"):
            demo._validate_parent(tmp_path / "p.gsplats.zarr")


class TestTheRecipeConstantsAgreeWithEachOther:
    def test_the_stride_yields_the_expected_frames_from_the_real_parent(self):
        """The shipped parent is 253 timepoints; stride 5 must yield exactly 51."""
        assert -(-253 // demo.SOURCE_STRIDE) == demo.EXPECTED_FRAMES

    def test_the_stacked_column_is_last_not_first(self):
        """The fit puts spatial dims first and the stacked axis last. A time-first
        index (0) would filter a spatial axis and keep a slab of the embryo."""
        assert demo.TIME_COL == 3

    def test_the_expected_total_is_about_a_fifth_of_the_parents_finest(self):
        """Sanity, not exactness: timepoints differ slightly in splat count, so
        the ratio lands near but not on 51/253."""
        ratio = demo.EXPECTED_SPLATS / demo.PARENT_FINEST_SPLATS
        assert 0.95 < ratio / (demo.EXPECTED_FRAMES / 253) < 1.05

    def test_the_chunk_profile_is_the_whole_read_one(self):
        """`hosting` (256 KB) costs 4.5x the bytes per partial hit; this node is
        read a whole timepoint at a time."""
        assert demo.CHUNK_PROFILE == "archive"
