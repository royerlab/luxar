"""Tests for ``add_lines(split=...)`` — polyline-centroid BSP.

Lines splits are polyline-atomic: every vertex of a polyline stays in
exactly one output part, and the BSP runs over per-polyline centroids
(mean of constituent vertex positions). Covers:

- :func:`luxar.core.split.midpoint_bsp_polylines` directly.
- The ``split=`` kwarg on ``add_lines`` for line_type='segments' and
  'indexed'.
- The new ``rule='sah'`` opt-in.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.lod_lines import identify_polylines
from luxar.core.split import midpoint_bsp_polylines
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
            assert vertex_count <= 15, (
                f"part of size {vertex_count} exceeds cap 15"
            )

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
# add_lines(split=...) — end-to-end
# ────────────────────────────────────────────────────────────────────────


class TestAddLinesSplit:
    @staticmethod
    def _segments_data(n_segments: int, seed: int = 0):
        """Build (vertices, widths) for ``n_segments`` random line segments."""
        rng = np.random.RandomState(seed)
        vertices = rng.uniform(-10, 10, (2 * n_segments, 3)).astype(np.float32)
        widths = np.full(2 * n_segments, 0.05, dtype=np.float32)
        return vertices, widths

    def test_over_cap_creates_split_wrapper(self, tmp_path):
        v, w = self._segments_data(200, seed=0)  # 400 vertices total
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="segments",
                split=dict(max_elements=120),
            )
            assert isinstance(node, Group)
            assert node.attrs.get("kind") == "split"
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
                split=dict(max_elements=100),
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

    def test_split_under_cap_falls_through(self, tmp_path):
        """split= with input under the cap → no wrapper."""
        v, w = self._segments_data(5, seed=2)  # 10 vertices
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="segments",
                split=dict(max_elements=100),
            )
        assert node.attrs.get("kind") != "split"

    def test_split_with_polyline_type_is_no_op(self, tmp_path):
        """A single polyline = one polyline → BSP can't split → single leaf."""
        # 200 vertices in one polyline; cap=50. With polyline-atomic
        # invariant, the single polyline can't be broken up so we expect
        # to fall through to a single-leaf write.
        v = np.random.RandomState(3).uniform(-10, 10, (200, 3)).astype(
            np.float32
        )
        w = np.full(200, 0.05, dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="polyline",
                split=dict(max_elements=50),
            )
        assert node.attrs.get("kind") != "split"

    def test_invalid_split_rule_raises(self, tmp_path):
        v, w = self._segments_data(50, seed=4)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="split rule"):
                scene.add_lines(
                    "lines",
                    vertices=v,
                    widths=w,
                    line_type="segments",
                    split=dict(max_elements=20, rule="invalid"),
                )

    def test_sah_rule_produces_split(self, tmp_path):
        v, w = self._segments_data(200, seed=5)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "lines",
                vertices=v,
                widths=w,
                line_type="segments",
                split=dict(max_elements=120, rule="sah"),
            )
        assert isinstance(node, Group)
        assert node.attrs.get("kind") == "split"

    def test_image_labels_alongside_split_raises(self, tmp_path):
        v, w = self._segments_data(100, seed=6)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="image_labels.*split"):
                scene.add_lines(
                    "lines",
                    vertices=v,
                    widths=w,
                    line_type="segments",
                    split=dict(max_elements=50),
                    image_labels=["foo"],
                )


# ────────────────────────────────────────────────────────────────────────
# identify_polylines smoke check (the BSP feeds on this helper)
# ────────────────────────────────────────────────────────────────────────


class TestIdentifyPolylinesForSplit:
    def test_segments_pairs_per_polyline(self):
        polys = identify_polylines(10, "segments")
        assert len(polys) == 5
        for p in polys:
            assert p.size == 2
