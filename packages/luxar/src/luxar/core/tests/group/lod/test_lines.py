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
                [0, 0, 0],
                [10, 0, 0],  # length 10
                [0, 0, 0],
                [1, 0, 0],  # length 1
                [0, 0, 0],
                [5, 0, 0],  # length 5
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
            verts,
            line_type="segments",
            widths=widths,
            method="random",
            n_lods=4,
        )
        total_polys = sum(len(level) for level in levels)
        assert total_polys == 10  # 20 vertices ÷ 2 per segment

    def test_polyline_single_warns_and_emits_one_level(self) -> None:
        verts = np.random.RandomState(0).rand(10, 3).astype(np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            levels = make_additive_lod_lines(
                verts,
                line_type="polyline",
                n_lods=4,
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
                verts,
                line_type="loop",
                n_lods=4,
            )
        assert len(levels) == 1

    def test_indexed_multi_component(self) -> None:
        indices = np.array([[0, 1], [1, 2], [3, 4], [5, 6], [6, 7]], dtype=np.uint32)
        verts = np.random.RandomState(0).rand(8, 3).astype(np.float32)
        widths = np.ones(8, dtype=np.float32)
        levels = make_additive_lod_lines(
            verts,
            line_type="indexed",
            indices=indices,
            widths=widths,
            method="random",
            n_lods=4,
        )
        total_polys = sum(len(level) for level in levels)
        assert total_polys == 3  # three connected components

    def test_empty(self) -> None:
        verts = np.zeros((0, 3), dtype=np.float32)
        levels = make_additive_lod_lines(verts, line_type="segments")
        assert levels == []

    def test_fewer_polylines_than_n_lods(self) -> None:
        """B8-G1/[P8]: symmetric with ``test_points.py::
        test_fewer_elements_than_n_lods`` — when there are fewer polylines
        than ``n_lods``, emit only the non-empty levels (no zero-length
        levels) while still covering every polyline."""
        verts = np.random.RandomState(0).rand(4, 3).astype(np.float32)  # 2 segments
        widths = np.ones(4, dtype=np.float32)
        levels = make_additive_lod_lines(
            verts, line_type="segments", widths=widths, method="random", n_lods=4
        )
        assert sum(len(L) for L in levels) == 2  # both polylines covered
        assert all(len(L) > 0 for L in levels)  # empty levels dropped


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
        assert spec["n_lods"] == 4

    def test_dict_overrides(self) -> None:
        spec = resolve_additive_axis_lines(
            {"method": "salience", "n_lods": 3, "seed": 42}
        )
        assert spec is not None
        assert spec["method"] == "salience"
        assert spec["n_lods"] == 3
        assert spec["seed"] == 42

    def test_recompute_tolerated_for_symmetry(self) -> None:
        # gsplats has recompute=True semantics; Lines tolerates without action.
        spec = resolve_additive_axis_lines({"recompute": True})
        assert spec is not None  # doesn't raise, returns defaults

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized keys"):
            resolve_additive_axis_lines({"bogus": 42})

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            resolve_additive_axis_lines({"method": "bogus"})

    def test_invalid_spec_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be None, bool, or dict"):
            resolve_additive_axis_lines("auto")  # type: ignore[arg-type]

    def test_breakpoints_pass_through(self) -> None:
        spec = resolve_additive_axis_lines({"breakpoints": [2, 4]})
        assert spec is not None
        assert spec["counts"] == [2, 4]

    def test_energy_salience_kind(self) -> None:
        spec = resolve_additive_axis_lines({"salience_kind": "energy"})
        assert spec is not None
        assert spec["salience_kind"] == "energy"

    def test_counts_and_breakpoints_conflict_raises(self) -> None:
        with pytest.raises(ValueError, match="either 'counts' OR 'breakpoints'"):
            resolve_additive_axis_lines({"counts": [1, 2], "breakpoints": [3, 4]})


# ────────────────────────────────────────────────────────────────────────
# End-to-end via ``add_lines(additive_lod=...)``
# ────────────────────────────────────────────────────────────────────────


class TestAddLinesAdditiveLod:
    def test_colors_and_colormap_rejected_on_additive_path(self, tmp_path) -> None:
        # Regression: the additive multi-LOD branch used to return before the
        # colors/colormap mutual-exclusivity validation, silently accepting
        # invalid combinations that the flat path rejects.
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        vertices = rng.rand(40, 3).astype(np.float32)
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="colors.*colormap"):
                scene.add_lines(
                    "ln",
                    vertices,
                    0.1,
                    colors=rng.rand(40, 3).astype(np.float32),
                    colormap="viridis",
                    line_type="segments",
                    additive_lod=dict(n_lods=3, method="random"),
                )
            with pytest.raises(ValueError, match="scalars.*colormap"):
                scene.add_lines(
                    "ln2",
                    vertices,
                    0.1,
                    scalars=rng.rand(40).astype(np.float32),
                    line_type="segments",
                    additive_lod=dict(n_lods=3, method="random"),
                )

    def test_segments_round_trip(self, tmp_path) -> None:
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        vertices = rng.rand(40, 3).astype(np.float32)
        widths = np.ones(40, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                vertices,
                widths=widths,
                line_type="segments",
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

    def test_image_labels_suppress_ladder_and_are_kept(self, tmp_path) -> None:
        # Regression: the plain additive multi-LOD writer has no image_labels
        # channel, so an explicit ladder used to SILENTLY DROP the labels. It
        # must instead refuse the ladder (write a single leaf) and keep the
        # labels — mirroring the substitutive path's suppress_reason guard.
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        vertices = rng.rand(40, 3).astype(np.float32)
        widths = np.ones(40, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                vertices,
                widths=widths,
                line_type="segments",
                image_labels=[b"x"] * 40,
                additive_lod=dict(n_lods=4, method="random"),
            )

        grp = zarr.open(str(output), mode="r")["ln"]
        assert grp.attrs["type"] == "lines"
        assert grp.attrs["n_vertices"] == 40
        # Ladder suppressed → single leaf, no additive_<i> subgroups.
        assert "n_additive_sublods" not in grp.attrs
        assert not [k for k in grp.keys() if k.startswith("additive_")]
        # Labels survived to the leaf.
        assert grp.attrs.get("has_image_labels") is True

    def test_invalid_additive_spec_raises_even_with_image_labels(
        self, tmp_path
    ) -> None:
        # The image_labels guard refuses the ladder but must still validate
        # the spec — a malformed additive_lod= fails fast on every path.
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        vertices = rng.rand(40, 3).astype(np.float32)
        widths = np.ones(40, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="method must be"):
                scene.add_lines(
                    "ln",
                    vertices,
                    widths=widths,
                    line_type="segments",
                    image_labels=[b"x"] * 40,
                    additive_lod=dict(method="bogus"),
                )

    def test_explicit_n_lods_is_honored(self, tmp_path) -> None:
        """B8-G2/[P8]: symmetric with ``test_points.py::
        test_dict_with_explicit_n_lods`` — an explicit ``n_lods`` (≠ the
        default 4) must flow end-to-end to the on-disk sub-LOD count, proving
        it is honored rather than hardcoded."""
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(3)
        vertices = rng.rand(40, 3).astype(np.float32)
        widths = np.ones(40, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                vertices,
                widths=widths,
                line_type="segments",
                additive_lod=dict(n_lods=3, method="random"),
            )

        store = zarr.open(str(output), mode="r")
        grp = store["ln"]
        assert grp.attrs["n_additive_sublods"] == 3
        subgroups = sorted(k for k in grp.keys() if k.startswith("additive_"))
        assert len(subgroups) == 3

    def test_polyline_single_falls_through_to_single_shot(self, tmp_path) -> None:
        """Single-polyline + additive_lod=True → warning + single-shot write."""
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(1)
        vertices = rng.rand(20, 3).astype(np.float32)
        widths = np.ones(20, dtype=np.float32) * 0.1

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with LuxarZarrCompiler(output) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                node = scene.add_lines(
                    "ln",
                    vertices,
                    widths=widths,
                    line_type="polyline",
                    additive_lod=True,
                )
                assert isinstance(node, Lines)

        store = zarr.open(str(output), mode="r")
        grp = store["ln"]
        # No multi-LOD subgroups; falls through to single-shot write.
        assert "n_additive_sublods" not in grp.attrs

    def test_single_shot_path_still_works(self, tmp_path) -> None:
        """No additive_lod → existing single-LOD layout."""
        output = tmp_path / "t.luxar.zarr"
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


class TestLinesStreamBreakpoints:
    """``stream:<c>`` on Lines — sized in vertices, cut on polylines."""

    @staticmethod
    def _segments(n_seg: int, seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.rand(n_seg * 2, 3).astype(np.float32)

    def test_chunk_is_converted_from_vertices_to_polylines(self) -> None:
        # 500 two-vertex polylines: mean length 2, so stream:200 vertices
        # becomes 100 polylines in the first level.
        verts = self._segments(500)

        levels = make_additive_lod_lines(
            verts, line_type="segments", counts="stream:200"
        )

        assert len(levels[0]) == 100

    def test_every_level_holds_whole_polylines(self) -> None:
        verts = self._segments(500, seed=1)

        levels = make_additive_lod_lines(
            verts, line_type="segments", counts="stream:200"
        )

        for level in levels:
            for member in level:
                # `segments` polylines are exactly the two endpoints of one
                # segment; a split polyline would break segment topology.
                assert member.size == 2

    def test_levels_partition_the_polylines_exactly(self) -> None:
        verts = self._segments(400, seed=2)

        levels = make_additive_lod_lines(
            verts, line_type="segments", counts="stream:100"
        )

        joined = np.concatenate([m for level in levels for m in level])
        assert joined.size == 800
        assert np.array_equal(np.unique(joined), np.arange(800))

    def test_smaller_than_one_chunk_collapses_to_a_single_level(self) -> None:
        verts = self._segments(20, seed=3)

        levels = make_additive_lod_lines(
            verts, line_type="segments", counts="stream:40000"
        )

        assert len(levels) == 1
        assert len(levels[0]) == 20

    def test_unrecognized_string_names_both_vocabularies(self) -> None:
        verts = self._segments(20, seed=4)

        with pytest.raises(ValueError, match="energy:.*stream:|stream:.*energy:"):
            make_additive_lod_lines(verts, line_type="segments", counts="bogus:1")
