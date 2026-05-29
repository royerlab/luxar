"""Tests for additive-LOD support on Lines (polyline-level granularity).

Covers:

- ``identify_polylines`` for all four ``line_type`` variants.
- ``compute_additive_order_lines`` and ``make_additive_lod_lines``.
- ``resolve_additive_axis_lines``.
- End-to-end ``add_lines(..., additive_lod=...)`` round-trip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.lines import (
    compute_additive_order_lines,
    identify_polylines,
    make_additive_lod_lines,
    resolve_additive_axis_lines,
)
from luxar.core.lines import Lines
from luxar.io.compiler import LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# identify_polylines
# ────────────────────────────────────────────────────────────────────────


class TestIdentifyPolylines:
    def test_segments_pairs(self) -> None:
        polys = identify_polylines(20, "segments")
        assert len(polys) == 10
        for p in polys:
            assert p.size == 2

    def test_segments_odd_n_raises(self) -> None:
        with pytest.raises(ValueError, match="even n_vertices"):
            identify_polylines(7, "segments")

    def test_polyline_single(self) -> None:
        polys = identify_polylines(10, "polyline")
        assert len(polys) == 1
        assert polys[0].size == 10

    def test_loop_single(self) -> None:
        polys = identify_polylines(8, "loop")
        assert len(polys) == 1
        assert polys[0].size == 8

    def test_indexed_components(self) -> None:
        # Two chains: 0-1-2 and 3-4
        indices = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.uint32)
        polys = identify_polylines(5, "indexed", indices)
        assert len(polys) == 2
        sizes = sorted(p.size for p in polys)
        assert sizes == [2, 3]

    def test_indexed_with_isolated_vertex(self) -> None:
        # Vertex 4 has no segment — gets its own 1-element "polyline".
        indices = np.array([[0, 1], [2, 3]], dtype=np.uint32)
        polys = identify_polylines(5, "indexed", indices)
        assert len(polys) == 3  # {0,1}, {2,3}, {4}

    def test_indexed_requires_indices(self) -> None:
        with pytest.raises(ValueError, match="indices array"):
            identify_polylines(4, "indexed")

    def test_empty(self) -> None:
        assert identify_polylines(0, "segments") == []

    def test_invalid_line_type_raises(self) -> None:
        with pytest.raises(ValueError, match="line_type must be"):
            identify_polylines(4, "bogus")


# ────────────────────────────────────────────────────────────────────────
# compute_additive_order_lines
# ────────────────────────────────────────────────────────────────────────


class TestComputeAdditiveOrderLines:
    def _setup(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        verts = rng.rand(20, 3).astype(np.float32)
        widths = rng.rand(20).astype(np.float32)
        polys = identify_polylines(20, "segments")  # 10 polylines
        return verts, widths, polys

    def test_random(self) -> None:
        verts, widths, polys = self._setup()
        perm, counts = compute_additive_order_lines(
            verts, polys, widths=widths, method="random", seed=42
        )
        assert perm.shape == (10,)
        assert sorted(perm.tolist()) == list(range(10))

    def test_salience_sorts_by_length_times_width(self) -> None:
        # Build polylines where length × width is deterministic.
        verts = np.array(
            [
                [0, 0, 0], [10, 0, 0],     # length 10
                [0, 0, 0], [1, 0, 0],      # length 1
                [0, 0, 0], [5, 0, 0],      # length 5
            ],
            dtype=np.float32,
        )
        widths = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        polys = identify_polylines(6, "segments")
        perm, _ = compute_additive_order_lines(
            verts, polys, widths=widths, method="salience"
        )
        # Largest length (polyline 0) first; shortest (polyline 1) last.
        assert perm[0] == 0
        assert perm[-1] == 1

    def test_salience_requires_widths(self) -> None:
        verts, _, polys = self._setup()
        with pytest.raises(ValueError, match="requires per-vertex widths"):
            compute_additive_order_lines(verts, polys, method="salience")

    def test_spatial_uniform_returns_counts(self) -> None:
        verts, widths, polys = self._setup()
        perm, counts = compute_additive_order_lines(
            verts, polys, method="spatial-uniform", n_lods=4
        )
        assert sum(counts) == 10
        assert len(counts) == 4


# ────────────────────────────────────────────────────────────────────────
# make_additive_lod_lines
# ────────────────────────────────────────────────────────────────────────


class TestMakeAdditiveLodLines:
    def test_segments_random(self) -> None:
        verts = np.random.RandomState(0).rand(20, 3).astype(np.float32)
        widths = np.random.RandomState(1).rand(20).astype(np.float32)
        levels = make_additive_lod_lines(
            verts, line_type="segments", widths=widths,
            method="random", n_lods=4,
        )
        total_polys = sum(len(level) for level in levels)
        assert total_polys == 10  # 20 vertices ÷ 2 per segment

    def test_polyline_single_warns_and_emits_one_level(self) -> None:
        verts = np.random.RandomState(0).rand(10, 3).astype(np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            levels = make_additive_lod_lines(
                verts, line_type="polyline", n_lods=4,
            )
            assert len(w) == 1
            assert "single polyline" in str(w[0].message)
        assert len(levels) == 1
        assert len(levels[0]) == 1  # one polyline in that level

    def test_loop_single_no_op(self) -> None:
        verts = np.random.RandomState(0).rand(8, 3).astype(np.float32)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            levels = make_additive_lod_lines(
                verts, line_type="loop", n_lods=4,
            )
        assert len(levels) == 1

    def test_indexed_multi_component(self) -> None:
        indices = np.array(
            [[0, 1], [1, 2], [3, 4], [5, 6], [6, 7]], dtype=np.uint32
        )
        verts = np.random.RandomState(0).rand(8, 3).astype(np.float32)
        widths = np.ones(8, dtype=np.float32)
        levels = make_additive_lod_lines(
            verts, line_type="indexed", indices=indices,
            widths=widths, method="random", n_lods=4,
        )
        total_polys = sum(len(level) for level in levels)
        assert total_polys == 3  # three connected components

    def test_empty(self) -> None:
        verts = np.zeros((0, 3), dtype=np.float32)
        levels = make_additive_lod_lines(verts, line_type="segments")
        assert levels == []


# ────────────────────────────────────────────────────────────────────────
# Resolver
# ────────────────────────────────────────────────────────────────────────


class TestResolveAdditiveAxisLines:
    def test_none_is_noop(self) -> None:
        assert resolve_additive_axis_lines(None) is None

    def test_false_is_noop(self) -> None:
        assert resolve_additive_axis_lines(False) is None

    def test_true_returns_defaults(self) -> None:
        spec = resolve_additive_axis_lines(True)
        assert spec is not None
        assert spec["method"] == "random"


# ────────────────────────────────────────────────────────────────────────
# End-to-end via ``add_lines(additive_lod=...)``
# ────────────────────────────────────────────────────────────────────────


class TestAddLinesAdditiveLod:
    def test_segments_round_trip(self, tmp_path) -> None:
        output = tmp_path / "t.zarr"
        rng = np.random.RandomState(0)
        vertices = rng.rand(40, 3).astype(np.float32)
        widths = np.ones(40, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln", vertices, widths=widths, line_type="segments",
                additive_lod=dict(n_lods=4, method="random"),
            )

        store = zarr.open(str(output), mode="r")
        grp = store["ln"]
        assert grp.attrs["type"] == "lines"
        assert grp.attrs["n_vertices"] == 40
        assert grp.attrs["n_segments"] == 20  # 40 vertices / 2 per segment
        assert grp.attrs["n_polylines"] == 20  # 40 / 2
        assert grp.attrs["n_additive_sublods"] == 4
        subgroups = sorted(k for k in grp.keys() if k.startswith("additive_"))
        assert len(subgroups) == 4

    def test_polyline_single_falls_through_to_single_shot(
        self, tmp_path
    ) -> None:
        """Single-polyline + additive_lod=True → warning + single-shot write."""
        output = tmp_path / "t.zarr"
        rng = np.random.RandomState(1)
        vertices = rng.rand(20, 3).astype(np.float32)
        widths = np.ones(20, dtype=np.float32) * 0.1

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with LuxarZarrCompiler(output) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                node = scene.add_lines(
                    "ln", vertices, widths=widths, line_type="polyline",
                    additive_lod=True,
                )
                assert isinstance(node, Lines)

        store = zarr.open(str(output), mode="r")
        grp = store["ln"]
        # No multi-LOD subgroups; falls through to single-shot write.
        assert "n_additive_sublods" not in grp.attrs

    def test_single_shot_path_still_works(self, tmp_path) -> None:
        """No additive_lod → existing single-LOD layout."""
        output = tmp_path / "t.zarr"
        rng = np.random.RandomState(2)
        vertices = rng.rand(10, 3).astype(np.float32)
        widths = np.ones(10, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines("ln", vertices, widths=widths)
            assert isinstance(node, Lines)

        store = zarr.open(str(output), mode="r")
        grp = store["ln"]
        assert "n_additive_sublods" not in grp.attrs
