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

from types import SimpleNamespace

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
    def test_an_unscaled_parent_is_refused_before_the_walk(self, monkeypatch, tmp_path):
        attrs = _parent_attrs()
        attrs["position_bounds"]["max"][0] = 406
        _attrs(monkeypatch, attrs)
        with pytest.raises(SystemExit, match="Z anisotropy"):
            demo._validate_parent(tmp_path / "unscaled.gsplats.zarr")

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

    def test_a_three_dimensional_parent_names_the_missing_time_bound(
        self, monkeypatch, tmp_path
    ):
        _attrs(
            monkeypatch,
            {"kind": "partition", "position_bounds": {"max": [10, 20, 30]}},
        )
        with pytest.raises(SystemExit, match="no position_bounds"):
            demo._validate_parent(tmp_path / "3d.gsplats.zarr")


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

    def test_the_progressive_ladder_has_the_recorded_twelve_rungs(self):
        assert demo.EXPECTED_RUNGS == 12


class TestTheRecomputeCommandsAreRealCliPaths:
    def test_the_recipe_invokes_top_level_optimise(self, monkeypatch, tmp_path):
        parent = tmp_path / "parent.gsplats.zarr"
        parent.mkdir()
        calls = []
        monkeypatch.setattr(demo, "PARENT_ARG", parent)
        monkeypatch.setattr(demo, "_validate_parent", lambda _path: None)
        monkeypatch.setattr(
            demo,
            "restride_stacked_axis",
            lambda *_args, **_kwargs: {"frames": demo.EXPECTED_FRAMES},
        )
        monkeypatch.setattr(demo, "run_luxar_cli", lambda *args: calls.append(args))
        monkeypatch.setattr(demo, "load_gsplat_node", lambda _path: (object(), {}))
        monkeypatch.setattr(
            demo,
            "iter_leaves",
            lambda _node: [
                SimpleNamespace(
                    n_splats=demo.EXPECTED_SPLATS,
                    n_additive_sublods=demo.EXPECTED_RUNGS,
                )
            ],
        )

        result = demo.recompute_archive(tmp_path / "work")

        assert result.name == "h2afva_51tp.gsplats.zarr"
        assert [call[:2] for call in calls[:2]] == [
            ("gsplat", "flatten"),
            ("gsplat", "lod"),
        ]
        assert calls[2][0] == "optimise"


class TestTheRebuiltArchiveShapeIsPinned:
    def test_the_recorded_leaf_and_rungs_are_accepted(self, monkeypatch):
        leaf = SimpleNamespace(
            n_splats=17, n_additive_sublods=demo.EXPECTED_RUNGS
        )
        monkeypatch.setattr(demo, "iter_leaves", lambda _node: [leaf])

        assert demo._validate_rebuilt_archive(object()) == 17

    def test_multiple_leaves_are_rejected_even_when_the_count_matches(
        self, monkeypatch
    ):
        leaves = [
            SimpleNamespace(
                n_splats=demo.EXPECTED_SPLATS // 2,
                n_additive_sublods=demo.EXPECTED_RUNGS,
            ),
            SimpleNamespace(
                n_splats=demo.EXPECTED_SPLATS - demo.EXPECTED_SPLATS // 2,
                n_additive_sublods=demo.EXPECTED_RUNGS,
            ),
        ]
        monkeypatch.setattr(demo, "iter_leaves", lambda _node: leaves)

        with pytest.raises(RuntimeError, match="2 leaves, expected one"):
            demo._validate_rebuilt_archive(object())

    def test_a_changed_progressive_ladder_is_rejected(self, monkeypatch):
        leaf = SimpleNamespace(
            n_splats=demo.EXPECTED_SPLATS,
            n_additive_sublods=demo.EXPECTED_RUNGS - 1,
        )
        monkeypatch.setattr(demo, "iter_leaves", lambda _node: [leaf])

        with pytest.raises(RuntimeError, match="progressive rungs"):
            demo._validate_rebuilt_archive(object())
