"""Tests for the strided stacked-axis re-authoring pass.

Four behaviours are load-bearing and each has a silent failure mode:

1. **Both part shapes are walked.** A part may be a ``kind=lod`` group of levels
   or a bare leaf. Reading only ``child_*`` yields an EMPTY result on the second
   shape with no error, and the second shape is exactly what a store gets after
   its substitutive levels are dropped.
2. **The stacked axis is renumbered onto a dense grid.** Keeping the original
   labels leaves ``stride - 1`` empty positions between frames, so most steps of
   the viewer's discrete navigation land on nothing.
3. **`root_attrs` is the only attribute channel that reaches disk.** The writer
   stamps every node from its own key set and drops arbitrary node ``meta``
   silently, so appearance routed through ``meta`` -- the obvious-looking choice
   -- writes a store with those attrs simply missing.
4. **A non-integral stacked axis is refused rather than rounded.** Rounding one
   would merge or mislabel frames with no error.
"""

import numpy as np
import pytest

from luxar._zarr_compat import open_group, read_node_attrs
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.restride import _MAX_INTEGRALITY_OFFSET, restride_stacked_axis
from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

TIME_COL = 3


def _sublod(frames, *, n_per_frame=4, seed=0, jitter=0.0):
    """One sub-LOD holding ``n_per_frame`` splats at each frame in ``frames``."""
    rng = np.random.default_rng(seed)
    times = np.repeat(np.asarray(frames, dtype=np.float32), n_per_frame)
    if jitter:
        times = times + rng.uniform(-jitter, jitter, times.shape).astype(np.float32)
    n = times.size
    centers = np.zeros((n, 4), dtype=np.float32)
    centers[:, :3] = rng.uniform(0.0, 10.0, (n, 3)).astype(np.float32)
    centers[:, TIME_COL] = times
    return AdditiveSubLOD(
        centers=centers,
        amplitudes=rng.uniform(0.1, 1.0, n).astype(np.float32),
        # Isotropic sigma = 1 in the packed 4D lower-triangular layout (10 terms).
        cholesky_factors=np.tile(
            np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32), (n, 1)
        ),
    )


def _write(path, node, **kwargs):
    write_gsplats_tree(
        path,
        node,
        encoding_mode=EncodingMode.PRECISION,
        barrier_dims=[TIME_COL],
        **kwargs,
    )
    return path


def _partition_of_lod_groups(path, frames, *, n_parts=2, levels=3):
    """The shape a tiled timelapse fit produces: parts -> levels -> ladder."""
    parts = [
        GSplatLodGroup(
            children=[
                GSplatLeaf(
                    additive_sublods=[_sublod(frames, seed=p * 10 + level)],
                    meta={"level_index": level},
                )
                for level in range(levels)
            ],
            meta={"kind": "lod"},
        )
        for p in range(n_parts)
    ]
    return _write(path, GSplatPartition(children=parts, meta={"kind": "partition"}))


def _partition_of_leaves(path, frames, *, n_parts=2, rungs=2):
    """The same store after its substitutive levels were dropped."""
    parts = [
        GSplatLeaf(
            additive_sublods=[_sublod(frames, seed=p * 10 + r) for r in range(rungs)],
            meta={},
        )
        for p in range(n_parts)
    ]
    return _write(path, GSplatPartition(children=parts, meta={"kind": "partition"}))


def _tally_frames(path):
    """Distinct stacked values and their counts, read back off disk."""
    import zarr

    tally: dict[float, int] = {}
    arrays = sorted(
        {
            p.parent
            for p in path.rglob("centers/*")
            if p.name in ("zarr.json", ".zarray")
        }
    )
    for array_path in arrays:
        column = np.asarray(zarr.open(store=str(array_path), mode="r")[:, TIME_COL])
        values, counts = np.unique(
            np.round(column.astype(np.float64), 6), return_counts=True
        )
        for v, c in zip(values.tolist(), counts.tolist()):
            tally[v] = tally.get(v, 0) + int(c)
    return tally


class TestBothPartShapesAreWalked:
    """A part is either an lod group of levels or a leaf; both must survive.

    The leaf shape is the one a store has AFTER its substitutive levels are
    dropped, and a walk that only looks for `child_*` returns an empty result on
    it without raising -- so this is the test that distinguishes "handled" from
    "silently produced nothing".
    """

    def test_a_partition_of_lod_groups_keeps_every_level(self, tmp_path):
        src = _partition_of_lod_groups(tmp_path / "in.gsplats.zarr", range(10))
        summary = restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=5, time_col=TIME_COL
        )
        assert summary["parts_out"] == 2
        # 2 parts x 3 levels x 2 kept frames x 4 splats
        assert summary["n_splats_out"] == 2 * 3 * 2 * 4
        out = read_node_attrs(tmp_path / "out.gsplats.zarr") or {}
        assert out["kind"] == "partition"
        group = open_group(tmp_path / "out.gsplats.zarr", mode="r")
        assert (read_node_attrs(tmp_path / "out.gsplats.zarr" / "part_0") or {})[
            "kind"
        ] == "lod"
        assert len(list(group["part_0"].group_keys())) == 3

    def test_a_partition_of_leaves_keeps_its_splats(self, tmp_path):
        src = _partition_of_leaves(tmp_path / "in.gsplats.zarr", range(10))
        summary = restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=5, time_col=TIME_COL
        )
        assert summary["parts_out"] == 2
        # 2 parts x 2 rungs x 2 kept frames x 4 splats -- NOT zero, which is what
        # a child_*-only walk would report.
        assert summary["n_splats_out"] == 2 * 2 * 2 * 4

    def test_a_leaf_part_is_not_wrapped_in_a_one_child_lod_group(self, tmp_path):
        """Wrapping would invent a level ladder the input never had."""
        src = _partition_of_leaves(tmp_path / "in.gsplats.zarr", range(10))
        restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=5, time_col=TIME_COL
        )
        attrs = read_node_attrs(tmp_path / "out.gsplats.zarr" / "part_0") or {}
        assert attrs.get("kind") != "lod"


class TestTheStackedAxisIsRenumberedDense:
    def test_kept_frames_are_relabelled_zero_to_n_minus_one(self, tmp_path):
        src = _partition_of_lod_groups(tmp_path / "in.gsplats.zarr", range(20))
        summary = restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=5, time_col=TIME_COL
        )
        # Original labels are recorded...
        assert summary["source_timepoints"] == [0, 5, 10, 15]
        assert summary["frames"] == 4
        # ...but the stored column is dense, so every navigation step lands on data.
        assert sorted(_tally_frames(tmp_path / "out.gsplats.zarr")) == [
            0.0,
            1.0,
            2.0,
            3.0,
        ]

    def test_the_recorded_stride_and_source_frames_reach_the_root(self, tmp_path):
        src = _partition_of_lod_groups(tmp_path / "in.gsplats.zarr", range(20))
        restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=5, time_col=TIME_COL
        )
        attrs = read_node_attrs(tmp_path / "out.gsplats.zarr") or {}
        assert attrs["source_stride"] == 5
        assert attrs["source_timepoints"] == [0, 5, 10, 15]
        assert attrs["stacked_axis_renumbered"] is True
        assert attrs["source_archive"] == "in.gsplats.zarr"

    def test_stride_one_keeps_every_frame_and_is_not_renumbered(self, tmp_path):
        src = _partition_of_lod_groups(tmp_path / "in.gsplats.zarr", range(4))
        summary = restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=1, time_col=TIME_COL
        )
        assert summary["frames"] == 4
        assert summary["n_splats_out"] == summary["n_splats_in"]
        attrs = read_node_attrs(tmp_path / "out.gsplats.zarr") or {}
        assert attrs["stacked_axis_renumbered"] is False
        assert attrs["stacked_axis_rounded_to_grid"] is True

    def test_stride_one_still_rounds_a_smeared_axis_onto_its_lattice(self, tmp_path):
        """The narrow but real use of stride=1: making the gridded snap exact."""
        leaf = GSplatLeaf(
            additive_sublods=[_sublod(range(4), jitter=0.002, seed=3)], meta={}
        )
        src = _write(
            tmp_path / "in.gsplats.zarr",
            GSplatPartition(children=[leaf], meta={"kind": "partition"}),
        )
        restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=1, time_col=TIME_COL
        )
        assert sorted(_tally_frames(tmp_path / "out.gsplats.zarr")) == [
            0.0,
            1.0,
            2.0,
            3.0,
        ]


class TestANonIntegralAxisIsRefused:
    """Rounding a non-integer axis would merge or mislabel frames silently."""

    def test_an_axis_off_its_lattice_raises_rather_than_guessing(self, tmp_path):
        leaf = GSplatLeaf(
            additive_sublods=[_sublod(range(4), jitter=0.4, seed=7)], meta={}
        )
        src = _write(
            tmp_path / "in.gsplats.zarr",
            GSplatPartition(children=[leaf], meta={"kind": "partition"}),
        )
        with pytest.raises(ValueError, match="not integral"):
            restride_stacked_axis(
                src, tmp_path / "out.gsplats.zarr", stride=2, time_col=TIME_COL
            )

    def test_quantization_scale_smear_is_tolerated(self, tmp_path):
        """~2e-3 is normal on a gridded uint16 axis and must NOT trip the guard."""
        assert _MAX_INTEGRALITY_OFFSET > 0.01
        leaf = GSplatLeaf(
            additive_sublods=[_sublod(range(4), jitter=0.002, seed=11)], meta={}
        )
        src = _write(
            tmp_path / "in.gsplats.zarr",
            GSplatPartition(children=[leaf], meta={"kind": "partition"}),
        )
        restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=2, time_col=TIME_COL
        )  # must not raise


class TestAttributesReachTheRootAndNowhereElse:
    """`root_attrs` is the ONLY channel that reaches disk.

    `write_gsplats_tree` stamps every node from its own key set and drops
    arbitrary node `meta` without a word, at leaves and at lod groups alike.
    That is why this pass copies no attributes forward: an input's per-leaf
    `absorption` of 9.0 cannot survive to shadow a root of 1.34, and equally a
    caller cannot stamp a leaf. The first test pins the channel that works; the
    second pins the one that does not, because routing appearance through `meta`
    is the obvious-looking mistake and it fails silently.
    """

    def test_caller_root_attrs_reach_the_root_document(self, tmp_path):
        src = _partition_of_lod_groups(tmp_path / "in.gsplats.zarr", range(4))
        restride_stacked_axis(
            src,
            tmp_path / "out.gsplats.zarr",
            stride=2,
            time_col=TIME_COL,
            root_attrs={"blending_mode": "volumetric", "absorption": 1.34},
        )
        root = read_node_attrs(tmp_path / "out.gsplats.zarr") or {}
        assert root.get("blending_mode") == "volumetric"
        assert root.get("absorption") == 1.34
        # ...and they must not have displaced the stamps this pass owns.
        assert root["source_stride"] == 2

    def test_node_meta_does_not_reach_disk(self, tmp_path):
        """The premise the no-copying decision rests on. If this ever starts
        passing meta through, the input's compositing values WOULD survive and
        this module would need the filter back."""
        node = GSplatPartition(
            children=[
                GSplatLodGroup(
                    children=[
                        GSplatLeaf(
                            additive_sublods=[_sublod(range(2))],
                            meta={"probe": "leaf", "colormap": "plasma"},
                        )
                    ],
                    meta={"kind": "lod", "probe": "group"},
                )
            ],
            meta={"kind": "partition", "probe": "root"},
        )
        _write(tmp_path / "p.gsplats.zarr", node)
        for rel in ("", "part_0", "part_0/child_0"):
            attrs = read_node_attrs(tmp_path / "p.gsplats.zarr" / rel) or {}
            assert "probe" not in attrs, f"{rel or 'root'} kept arbitrary meta"
        leaf = read_node_attrs(tmp_path / "p.gsplats.zarr" / "part_0" / "child_0") or {}
        assert leaf["colormap"] == "gray"  # the writer's default, not "plasma"

    def test_an_input_leafs_compositing_values_do_not_reach_the_output(self, tmp_path):
        """End to end: an input `absorption` of 9.0 must not shadow a root 1.34."""
        parts = [
            GSplatLodGroup(
                children=[
                    GSplatLeaf(
                        additive_sublods=[_sublod(range(4), seed=1)],
                        meta={"blending_mode": "additive", "absorption": 9.0},
                    )
                ],
                meta={"kind": "lod"},
            )
        ]
        src = _write(
            tmp_path / "in.gsplats.zarr",
            GSplatPartition(children=parts, meta={"kind": "partition"}),
        )
        restride_stacked_axis(
            src,
            tmp_path / "out.gsplats.zarr",
            stride=2,
            time_col=TIME_COL,
            root_attrs={"blending_mode": "volumetric", "absorption": 1.34},
        )
        root = read_node_attrs(tmp_path / "out.gsplats.zarr") or {}
        assert root["blending_mode"] == "volumetric"
        assert root["absorption"] == 1.34
        leaf = (
            read_node_attrs(tmp_path / "out.gsplats.zarr" / "part_0" / "child_0") or {}
        )
        assert "blending_mode" not in leaf
        assert leaf["absorption"] != 9.0

    def test_the_partitions_bsp_tree_and_element_cap_are_carried(self, tmp_path):
        """These are constructor arguments, not meta, so they DO survive -- and
        they must: the parts keep their spatial extents, only their contents
        thin out."""
        src = _partition_of_lod_groups(tmp_path / "in.gsplats.zarr", range(4))
        before = read_node_attrs(tmp_path / "in.gsplats.zarr") or {}
        restride_stacked_axis(
            src, tmp_path / "out.gsplats.zarr", stride=2, time_col=TIME_COL
        )
        after = read_node_attrs(tmp_path / "out.gsplats.zarr") or {}
        assert after["max_elements"] == before["max_elements"]
        assert after.get("bsp_tree") == before.get("bsp_tree")


class TestInputsThatCannotBeRestridedAreRejected:
    def test_a_zero_or_negative_stride_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="stride must be"):
            restride_stacked_axis(tmp_path / "x", tmp_path / "y", stride=0)

    def test_a_store_with_no_parts_is_refused_with_its_kind_named(self, tmp_path):
        src = _write(
            tmp_path / "flat.gsplats.zarr",
            GSplatLeaf(additive_sublods=[_sublod(range(4))], meta={}),
        )
        with pytest.raises(ValueError, match="no part_\\* groups"):
            restride_stacked_axis(
                src, tmp_path / "out.gsplats.zarr", stride=2, time_col=TIME_COL
            )

    def test_a_stride_that_keeps_nothing_is_refused(self, tmp_path):
        """Every frame filtered out is a wrong time_col, not an empty result."""
        leaf = GSplatLeaf(additive_sublods=[_sublod([1, 3, 7], seed=2)], meta={})
        src = _write(
            tmp_path / "in.gsplats.zarr",
            GSplatPartition(children=[leaf], meta={"kind": "partition"}),
        )
        with pytest.raises(ValueError, match="nothing survived"):
            restride_stacked_axis(
                src, tmp_path / "out.gsplats.zarr", stride=10, time_col=TIME_COL
            )


class TestTheProvenanceRecordsWhatWasDone:
    def test_the_method_line_distinguishes_slicing_from_rounding(self, tmp_path):
        src = _partition_of_lod_groups(tmp_path / "in.gsplats.zarr", range(10))
        restride_stacked_axis(
            src, tmp_path / "sliced.gsplats.zarr", stride=5, time_col=TIME_COL
        )
        restride_stacked_axis(
            src, tmp_path / "rounded.gsplats.zarr", stride=1, time_col=TIME_COL
        )
        sliced = read_node_attrs(tmp_path / "sliced.gsplats.zarr" / "provenance") or {}
        rounded = (
            read_node_attrs(tmp_path / "rounded.gsplats.zarr" / "provenance") or {}
        )
        assert "every 5th frame" in sliced["method"]
        assert "every frame kept" in rounded["method"]

    def test_derived_from_can_name_a_store_other_than_the_input_path(self, tmp_path):
        src = _partition_of_lod_groups(tmp_path / "in.gsplats.zarr", range(10))
        restride_stacked_axis(
            src,
            tmp_path / "out.gsplats.zarr",
            stride=5,
            time_col=TIME_COL,
            derived_from="h2afva_253tp.gsplats.zarr",
        )
        prov = read_node_attrs(tmp_path / "out.gsplats.zarr" / "provenance") or {}
        assert prov["derived_from"] == "h2afva_253tp.gsplats.zarr"
        # `source_archive` still names the path actually read -- the two are
        # different claims and must not collapse into one.
        root = read_node_attrs(tmp_path / "out.gsplats.zarr") or {}
        assert root["source_archive"] == "in.gsplats.zarr"
