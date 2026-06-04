"""Tests for ``add_lines(partition=...)`` — polyline-centroid BSP.

Lines partitions are polyline-atomic: every vertex of a polyline stays in
exactly one output part, and the BSP runs over per-polyline centroids
(mean of constituent vertex positions). Covers:

- :func:`luxar.core.group.partition.midpoint_bsp_polylines` directly.
- The ``partition=`` kwarg on ``add_lines`` for line_type='segments' and
  'indexed'.
- The new ``rule='sah'`` opt-in.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.group.lod.lines import identify_polylines
from luxar.core.group.partition import midpoint_bsp_polylines
from luxar.io.compiler import LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# midpoint_bsp_polylines — pure algorithm
# ────────────────────────────────────────────────────────────────────────


class TestMidpointBspPolylines:
    @staticmethod
    def _make_polylines(n_polylines: int, vertices_per: int, seed: int = 0):
        """Build (vertices, polyline_indices) for ``n_polylines`` short polylines."""
        rng = np.random.RandomState(seed)
        verts = []
        plys = []
        cursor = 0
        for _ in range(n_polylines):
            center = rng.uniform(-10, 10, 3)
            offsets = rng.uniform(-0.1, 0.1, (vertices_per, 3))
            poly = (center + offsets).astype(np.float32)
            verts.append(poly)
            plys.append(np.arange(cursor, cursor + vertices_per, dtype=np.intp))
            cursor += vertices_per
        vertices = np.concatenate(verts, axis=0)
        return vertices, plys

    def test_single_part_under_cap(self):
        v, ps = self._make_polylines(5, vertices_per=3)
        parts = midpoint_bsp_polylines(v, ps, max_elements=50)
        assert len(parts) == 1
        assert sorted(parts[0]) == list(range(5))

    def test_polylines_stay_atomic(self):
        """A polyline's vertices must land in exactly one part."""
        v, ps = self._make_polylines(40, vertices_per=4, seed=1)
        parts = midpoint_bsp_polylines(v, ps, max_elements=30)
        seen = set()
        for part in parts:
            for p in part:
                assert p not in seen, f"polyline {p} duplicated across parts"
                seen.add(p)
        assert seen == set(range(40))

    def test_total_vertices_preserved(self):
        v, ps = self._make_polylines(60, vertices_per=5, seed=2)
        parts = midpoint_bsp_polylines(v, ps, max_elements=80)
        total = 0
        for part in parts:
            for poly_id in part:
                total += int(ps[poly_id].size)
        assert total == v.shape[0]

    def test_part_caps_respected_for_short_polylines(self):
        """When polylines are small compared to cap, parts stay under it."""
        v, ps = self._make_polylines(20, vertices_per=2, seed=3)
        parts = midpoint_bsp_polylines(v, ps, max_elements=15)
        for part in parts:
            vertex_count = sum(int(ps[p].size) for p in part)
            assert vertex_count <= 15, f"part of size {vertex_count} exceeds cap 15"

    def test_single_oversized_polyline_kept_atomic(self):
        """A polyline larger than the cap stays in its own part."""
        # 50 vertices in one polyline, cap=20.
        verts = np.random.RandomState(4).uniform(-1, 1, (50, 3)).astype(np.float32)
        ps = [np.arange(50, dtype=np.intp)]
        parts = midpoint_bsp_polylines(verts, ps, max_elements=20)
        assert len(parts) == 1
        assert parts[0] == [0]

    def test_empty_input(self):
        verts = np.empty((0, 3), dtype=np.float32)
        parts = midpoint_bsp_polylines(verts, [], max_elements=10)
        assert parts == []


# ────────────────────────────────────────────────────────────────────────
# add_lines(partition=...) — end-to-end
# ────────────────────────────────────────────────────────────────────────


class TestAddLinesPartition:
    @staticmethod
    def _segments_data(n_segments: int, seed: int = 0):
        """Build (vertices, widths) for ``n_segments`` random line segments."""
        rng = np.random.RandomState(seed)
        vertices = rng.uniform(-10, 10, (2 * n_segments, 3)).astype(np.float32)
        widths = np.full(2 * n_segments, 0.05, dtype=np.float32)
        return vertices, widths

    def test_over_cap_creates_partition_wrapper(self, tmp_path):
        v, w = self._segments_data(200, seed=0)  # 400 vertices total
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="segments",
                partition=dict(max_elements=120),
            )
            assert isinstance(node, Group)
            assert node.attrs.get("kind") == "partition"
            assert node.attrs.get("display_type") == "lines"

    def test_polylines_atomic_at_write_time(self, tmp_path):
        """All segments are preserved; total vertex count matches input."""
        v, w = self._segments_data(150, seed=1)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="segments",
                partition=dict(max_elements=100),
            )
        store = zarr.open(str(tmp_path / "t.zarr"), mode="r")
        grp = store["lines"]

        def collect(g):
            counts = []
            for k in g.keys():
                child = g[k]
                if child.attrs.get("type") == "lines":
                    counts.append(int(child["vertices"].shape[0]))
                elif child.attrs.get("type") == "group":
                    counts.extend(collect(child))
            return counts

        counts = collect(grp)
        assert sum(counts) == v.shape[0], (
            f"total vertices changed: {sum(counts)} vs {v.shape[0]}"
        )
        assert len(counts) >= 2

    def test_partition_under_cap_falls_through(self, tmp_path):
        """partition= with input under the cap → no wrapper."""
        v, w = self._segments_data(5, seed=2)  # 10 vertices
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="segments",
                partition=dict(max_elements=100),
            )
        assert node.attrs.get("kind") != "partition"

    def test_partition_with_polyline_type_is_no_op(self, tmp_path):
        """A single polyline = one polyline → BSP can't partition → single leaf."""
        # 200 vertices in one polyline; cap=50. With polyline-atomic
        # invariant, the single polyline can't be broken up so we expect
        # to fall through to a single-leaf write.
        v = np.random.RandomState(3).uniform(-10, 10, (200, 3)).astype(np.float32)
        w = np.full(200, 0.05, dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="polyline",
                partition=dict(max_elements=50),
            )
        assert node.attrs.get("kind") != "partition"

    def test_invalid_partition_rule_raises(self, tmp_path):
        v, w = self._segments_data(50, seed=4)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="partition rule"):
                scene.add_lines(
                    "lines",
                    vertices=v,
                    widths=w,
                    line_type="segments",
                    partition=dict(max_elements=20, rule="invalid"),
                )

    def test_sah_rule_produces_partition(self, tmp_path):
        v, w = self._segments_data(200, seed=5)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="segments",
                partition=dict(max_elements=120, rule="sah"),
            )
        assert isinstance(node, Group)
        assert node.attrs.get("kind") == "partition"

    def test_image_labels_alongside_partition_raises(self, tmp_path):
        v, w = self._segments_data(100, seed=6)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="image_labels.*partition"):
                scene.add_lines(
                    "lines",
                    vertices=v,
                    widths=w,
                    line_type="segments",
                    partition=dict(max_elements=50),
                    image_labels=["foo"],
                )

    def test_partition_forwards_additive_lod(self, tmp_path):
        """M2: partition + additive_lod composes (parity with add_points) —
        each spatial part builds its own additive LOD ladder instead of the
        ladder being silently dropped."""
        v, w = self._segments_data(300, seed=7)  # 600 vertices
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="segments",
                partition=dict(max_elements=120),
                additive_lod=dict(method="random", n_lods=3),
            )
            assert node.attrs.get("kind") == "partition"

        store = zarr.open(str(tmp_path / "t.zarr"), mode="r")
        # Lines additive-LOD is a single node carrying n_additive_sublods > 1
        # (with additive_<i>/ subgroups), not a kind=lod group.
        found_ladder: list = []

        def walk(g):
            for k in g.keys():
                ch = g[k]
                if int(ch.attrs.get("n_additive_sublods", 1)) > 1:
                    found_ladder.append(ch.path)
                if hasattr(ch, "keys"):
                    walk(ch)

        walk(store["lines"])
        assert found_ladder, "no per-part additive ladder — additive_lod was dropped"


# ────────────────────────────────────────────────────────────────────────
# identify_polylines smoke check (the BSP feeds on this helper)
# ────────────────────────────────────────────────────────────────────────


class TestIdentifyPolylinesForPartition:
    def test_segments_pairs_per_polyline(self):
        polys = identify_polylines(10, "segments")
        assert len(polys) == 5
        for p in polys:
            assert p.size == 2


# ────────────────────────────────────────────────────────────────────────
# indexed line_type topology preservation across partition (H2)
# ────────────────────────────────────────────────────────────────────────


class TestIndexedPartitionTopology:
    """H2: partitioning an indexed graph must preserve the exact edge set
    (no dropped/fabricated edges) and must not crash on odd-sized
    components."""

    # Three spatially separated connected components with DISTINCT local
    # topologies: a 4-cycle, a 4-path, and a triangle (odd-sized — the case
    # that used to crash). Distinct shapes keep each part's segment array
    # byte-unique, isolating this test to the partition-remap fix.
    _VERTS = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # comp A (4-cycle)
            [100, 100, 100], [101, 100, 100], [101, 101, 100], [100, 101, 100],  # B
            [200, 0, 0], [201, 0, 0], [200, 1, 0],  # comp C (triangle)
        ],
        dtype=np.float32,
    )
    _EDGES = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],  # A: 4-cycle (4 edges)
            [4, 5], [5, 6], [6, 7],  # B: 4-path (3 edges)
            [8, 9], [9, 10], [10, 8],  # C: triangle (3 edges, odd component)
        ],
        dtype=np.intp,
    )

    def _collect_leaf_segment_arrays(self, store_path) -> list:
        """Return each written lines-leaf's (n_vertices, segments-array)."""
        store = zarr.open(str(store_path), mode="r")
        leaves: list = []

        def collect(g):
            for k in g.keys():
                child = g[k]
                t = child.attrs.get("type")
                if t == "lines":
                    seg = np.asarray(child["segments"]).reshape(-1, 2)
                    leaves.append((int(child["vertices"].shape[0]), seg))
                elif t == "group":
                    collect(child)

        collect(store["graph"])
        return leaves

    def test_indexed_partition_preserves_all_edges(self, tmp_path):
        widths = np.full(self._VERTS.shape[0], 0.05, dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "graph",
                vertices=self._VERTS,
                widths=widths,
                indices=self._EDGES.reshape(-1),  # add_lines wants flat (2E,)
                line_type="indexed",
                partition=dict(max_elements=4),
            )
            assert node.attrs.get("kind") == "partition"  # actually split

        leaves = self._collect_leaf_segment_arrays(tmp_path / "t.zarr")
        assert len(leaves) >= 2  # genuinely partitioned
        # No edge dropped or fabricated: total segment count is preserved.
        total_segments = sum(seg.shape[0] for _, seg in leaves)
        assert total_segments == self._EDGES.shape[0]  # 11
        # Every part's edges reference only that part's own vertices (a
        # fabricated cross-part edge would index out of range).
        for n_v, seg in leaves:
            if seg.size:
                assert int(seg.max()) < n_v

    def test_indexed_partition_odd_component_does_not_crash(self, tmp_path):
        # The triangle (component C) alone is an odd-sized component; force it
        # into its own part. This used to raise "Indexed requires at least 2
        # indices" mid-write.
        verts = self._VERTS[8:].copy()  # the 3 triangle vertices
        edges = np.array([0, 1, 1, 2, 2, 0], dtype=np.intp)  # flat (2E,)
        widths = np.full(3, 0.05, dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # Single component → can't split below cap → single leaf, no crash.
            scene.add_lines(
                "graph",
                vertices=verts,
                widths=widths,
                indices=edges,
                line_type="indexed",
                partition=dict(max_elements=2),
            )
