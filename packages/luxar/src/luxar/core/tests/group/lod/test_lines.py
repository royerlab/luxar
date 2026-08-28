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

from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.group.lod.lines import (
    _indexed_connected_components,
    _indexed_ladder_preserves_edges,
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

    def test_indexed_components_are_ordered_by_smallest_vertex(self) -> None:
        # The edge direction/order used to leave roots at 5 and 2, so sorting
        # by those incidental union-find roots put {1,2} before {0,5}. Bulk
        # root hooking defines the deterministic order by component minimum.
        segments = np.array([[0, 5], [1, 2]], dtype=np.intp)
        polys = _indexed_connected_components(6, segments)
        assert [poly.tolist() for poly in polys] == [[0, 5], [1, 2], [3], [4]]

    @pytest.mark.parametrize(
        ("n_vertices", "segments", "expected"),
        [
            (
                4,
                np.array([[0, 0], [1, 2], [2, 3]], dtype=np.intp),
                [[0], [1, 2, 3]],
            ),
            (
                3,
                np.array([[0, 1], [0, 1], [1, 0], [1, 2]], dtype=np.intp),
                [[0, 1, 2]],
            ),
            (5, np.empty((0, 2), dtype=np.intp), [[0], [1], [2], [3], [4]]),
        ],
        ids=["self-loop", "duplicate-and-reversed", "all-isolated"],
    )
    def test_indexed_pathological_edges(
        self,
        n_vertices: int,
        segments: np.ndarray,
        expected: list[list[int]],
    ) -> None:
        polys = _indexed_connected_components(n_vertices, segments)
        assert [poly.tolist() for poly in polys] == expected

    @pytest.mark.parametrize(
        ("n_vertices", "indices", "expected"),
        [
            (3, [[0, 1], [1, 2]], True),
            (3, [[2, 1], [1, 0]], True),
            (4, [[0, 1], [1, 2]], True),
            (3, [], True),
            (2, [[0, 1], [0, 1]], False),
            (1, [[0, 0]], False),
            (3, [[0, 1], [1, 2], [2, 0]], False),
            (3, [[0, 2], [2, 1]], False),
            (4, [[0, 1], [0, 2], [0, 3]], False),
        ],
        ids=[
            "ascending-chain",
            "reversed-reordered-chain",
            "chain-with-isolated-vertex",
            "edge-less",
            "duplicate-edge",
            "self-loop",
            "cycle",
            "non-ascending-path",
            "branching-star",
        ],
    )
    def test_indexed_ladder_preserves_exact_edge_multiset(
        self,
        n_vertices: int,
        indices: list[list[int]],
        expected: bool,
    ) -> None:
        edge_array = np.asarray(indices, dtype=np.intp).reshape(-1, 2)
        polylines = identify_polylines(n_vertices, "indexed", edge_array)
        assert _indexed_ladder_preserves_edges(edge_array, polylines) is expected

    def test_indexed_many_small_components_cover_vertices_once(self) -> None:
        # Representative ribbon-heavy shape: many short disjoint chains. The
        # component builder must group in one root-label sort, not rescan all
        # vertices once per chain.
        n_components = 50_000
        vertices_per_component = 3
        starts = np.arange(n_components, dtype=np.intp) * vertices_per_component
        segments = np.column_stack(
            (
                np.repeat(starts, 2) + np.tile([0, 1], n_components),
                np.repeat(starts, 2) + np.tile([1, 2], n_components),
            )
        )
        polys = _indexed_connected_components(
            n_components * vertices_per_component, segments
        )
        assert len(polys) == n_components
        # Every component is a non-overlapping view into one sorted vertex
        # permutation. Reintroducing one allocation/full scan per component
        # breaks this invariant even if small functional fixtures still pass.
        shared_order = polys[0].base
        assert shared_order is not None
        assert all(poly.base is shared_order for poly in polys)
        assert all(not poly.flags.owndata for poly in polys)
        np.testing.assert_array_equal(polys[0], [0, 1, 2])
        np.testing.assert_array_equal(
            polys[-1],
            [
                n_components * vertices_per_component - 3,
                n_components * vertices_per_component - 2,
                n_components * vertices_per_component - 1,
            ],
        )
        assert sum(int(poly.size) for poly in polys) == (
            n_components * vertices_per_component
        )

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


class TestRadialOrderLines:
    """``radial`` — the concentric-shell reveal, ordering WHOLE polylines.

    The per-polyline granularity is the point: a prefix of a vertex-ordered
    reveal would cut polylines in half and leave dangling segment topology,
    which is the invariant the whole Lines ladder exists to protect.
    """

    @staticmethod
    def _fan():
        """Five 2-vertex segments at x = 1..5, each its own polyline.

        Vertex order is shuffled relative to distance so a pass cannot be
        explained by the input arriving pre-ordered.
        """
        xs = [3.0, 4.0, 2.0, 5.0, 1.0]
        verts = np.array(
            [[x, 0.0, 0.0] for x in xs for _ in range(2)], dtype=np.float32
        )
        return verts, identify_polylines(len(verts), "segments")

    def test_orders_innermost_polyline_first(self) -> None:
        verts, polys = self._fan()
        perm, counts = compute_additive_order_lines(
            verts, polys, method="radial", reveal_centre=[0.0]
        )

        # Each polyline's representative is its own bbox centre; here that is
        # its x. Innermost (x=1, input position 4) must come first.
        order = [float(verts[polys[i][0], 0]) for i in perm]
        assert order == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert counts == []

    def test_permutation_indexes_polylines_not_vertices(self) -> None:
        # The property that keeps every prefix topologically valid: 10 vertices,
        # 5 polylines, so a polyline-granular permutation has length 5.
        verts, polys = self._fan()
        perm, _ = compute_additive_order_lines(verts, polys, method="radial")

        assert perm.shape == (len(polys),) == (5,)
        assert sorted(perm.tolist()) == list(range(5))

    def test_non_finite_vertices_are_refused_naming_the_vertices(self) -> None:
        """The error must blame the VERTICES, not ``reveal_centre``.

        Lines derives its default origin from the vertices, so before the
        data-side guard a NaN vertex produced a NaN origin that then tripped the
        scorer's ``reveal_centre must be finite`` check — an error naming a knob
        the caller never passed. Points and GSplats meanwhile returned input order
        silently. One shared validator makes all three agree AND report the input
        the caller actually supplied.
        """
        verts, polys = self._fan()
        verts = verts.copy()
        verts[3, 0] = float("nan")

        with pytest.raises(ValueError, match="vertices must be finite"):
            compute_additive_order_lines(verts, polys, method="radial")

    def test_is_translation_invariant(self) -> None:
        verts, polys = self._fan()
        far = verts + np.float32(1000.0)
        near, _ = compute_additive_order_lines(verts, polys, method="radial")
        moved, _ = compute_additive_order_lines(far, polys, method="radial")
        np.testing.assert_array_equal(near, moved)

    def test_zero_extent_column_is_not_a_shell_dimension(self) -> None:
        # A 4-D input whose time column is constant must order identically to
        # its 3-D equivalent. The Lines path is the one that needed care here:
        # the per-polyline centre helper is shared with the samplers, which want
        # only the first 3 columns, while radial needs to SEE the time column in
        # order to exclude it by extent.
        verts, polys = self._fan()
        verts4 = np.hstack([verts, np.full((verts.shape[0], 1), 7.0, dtype=np.float32)])
        p3, _ = compute_additive_order_lines(verts, polys, method="radial")
        p4, _ = compute_additive_order_lines(verts4, polys, method="radial")
        np.testing.assert_array_equal(p3, p4)

    def test_default_centre_is_the_bbox_centre_not_the_origin(self) -> None:
        verts, polys = self._fan()
        perm, _ = compute_additive_order_lines(verts, polys, method="radial")

        # bbox centre over x is 3.0, so the x=3 polyline is innermost — not x=1,
        # which is what an origin-centred implementation would pick.
        assert float(verts[polys[perm[0]][0], 0]) == 3.0

    def test_default_centre_is_the_vertex_bbox_not_the_representatives(self) -> None:
        # Unequal polyline lengths separate the two candidate origins, which
        # equal-length fixtures cannot: one long polyline spanning x=0..100 and
        # two short ones near x=0. The node's bbox centre is x=50; the bbox of
        # the per-polyline representatives (50, 0, 4) centres at 25.
        verts = np.array(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]  # long: centre 50
            + [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # short: centre 0
            + [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]],  # short: centre 4
            dtype=np.float32,
        )
        polys = identify_polylines(len(verts), "segments")
        perm, _ = compute_additive_order_lines(verts, polys, method="radial")

        # Centred on 50 the long polyline is innermost (|50-50| = 0); centred on
        # the representatives' own bbox centre 25 it would be second, behind the
        # x=4 one (|4-25| = 21 < |50-25| = 25).
        assert perm[0] == 0


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

    @staticmethod
    def _ladder_fan(n=40, span=39.0):
        xs = np.linspace(0.0, span, n, dtype=np.float32)
        verts = np.repeat(xs, 2)[:, None] * np.array(
            [[1.0, 0.0, 0.0]], dtype=np.float32
        )
        return verts.astype(np.float32)

    def test_radial_levels_grow_outward(self) -> None:
        verts = self._ladder_fan()
        levels = make_additive_lod_lines(
            verts, line_type="segments", method="radial", n_lods=4
        )

        assert sum(len(L) for L in levels) == 40
        # Max distance from the bbox centre must be monotone across levels iff
        # the ordering really is by distance.
        centre = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
        max_r = [
            float(max(abs(verts[p, 0].mean() - centre) for p in level))
            for level in levels
        ]
        assert max_r == sorted(max_r), max_r

    def test_radial_honours_the_stream_vocabulary(self) -> None:
        # Guards the same trap as the Points twin: `radial` must NOT be treated
        # like the samplers, whose natural partition bypasses `counts:`.
        verts = self._ladder_fan()
        levels = make_additive_lod_lines(
            verts, line_type="segments", method="radial", counts=[4, 12, 28]
        )

        assert [len(L) for L in levels] == [4, 8, 16, 12]

    def test_radial_kwargs_reach_the_scorer_through_the_builder(self) -> None:
        # Threading test: the centre override must survive the builder.
        verts = self._ladder_fan()
        pinned = make_additive_lod_lines(
            verts,
            line_type="segments",
            method="radial",
            n_lods=4,
            reveal_centre=[0.0],
        )
        first_xs = [float(verts[p, 0].mean()) for p in pinned[0]]
        assert max(first_xs) < 10.0

        default = make_additive_lod_lines(
            verts, line_type="segments", method="radial", n_lods=4
        )
        default_xs = [float(verts[p, 0].mean()) for p in default[0]]
        assert min(default_xs) > 10.0 and max(default_xs) < 30.0

    def test_radial_spatial_dims_reach_the_scorer_through_the_builder(self) -> None:
        verts = self._ladder_fan()
        # Add an opposing, much larger y spread that would dominate the distance
        # unless `spatial_dims` restricts it away.
        ys = np.linspace(500.0, 0.0, verts.shape[0], dtype=np.float32)
        verts = verts.copy()
        verts[:, 1] = ys

        restricted = make_additive_lod_lines(
            verts,
            line_type="segments",
            method="radial",
            n_lods=4,
            reveal_centre=[0.0],
            spatial_dims=[0],
        )
        first_xs = [float(verts[p, 0].mean()) for p in restricted[0]]
        assert max(first_xs) < 10.0


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

    def test_valid_stream_counts_resolves(self) -> None:
        spec = resolve_additive_axis_lines({"counts": "stream:1000"})
        assert spec is not None
        assert spec["counts"] == "stream:1000"

    def test_malformed_stream_counts_raise_at_resolve(self) -> None:
        # A ``stream:<c>`` with c < 1 must fail at resolve time, before any
        # kind=lod wrapper group is written under a substitutive ladder.
        with pytest.raises(ValueError, match="stream first-chunk size must be >= 1"):
            resolve_additive_axis_lines({"counts": "stream:0"})
        with pytest.raises(ValueError, match="stream"):
            resolve_additive_axis_lines({"counts": "stream:-5"})

    def test_valid_energy_counts_resolves(self) -> None:
        spec = resolve_additive_axis_lines({"counts": "energy:0.5,0.9,1.0"})
        assert spec is not None
        assert spec["counts"] == "energy:0.5,0.9,1.0"

    def test_other_doomed_counts_raise_at_resolve(self) -> None:
        # Same partial-group trap as stream:0 — any counts value the write
        # path is guaranteed to reject must fail at resolve time too.
        with pytest.raises(ValueError, match="unrecognized breakpoints string"):
            resolve_additive_axis_lines({"counts": "equal-count"})
        with pytest.raises(ValueError, match="energy: fractions must be numbers"):
            resolve_additive_axis_lines({"counts": "energy:abc"})
        with pytest.raises(ValueError, match="energy: fractions must be non-empty"):
            resolve_additive_axis_lines({"counts": "energy:"})
        with pytest.raises(ValueError, match="non-empty"):
            resolve_additive_axis_lines({"counts": []})

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
        assert grp.attrs["n_additive_sublods"] == 4
        subgroups = sorted(k for k in grp.keys() if k.startswith("additive_"))
        assert len(subgroups) == 4

    def test_indexed_chains_round_trip_exact_edges(self, tmp_path) -> None:
        from luxar.encoding import ArrayDecoder

        output = tmp_path / "t.luxar.zarr"
        vertices = np.array(
            [[component, offset, 0.0] for component in range(6) for offset in range(4)],
            dtype=np.float32,
        )
        indices = np.array(
            [
                (start + offset, start + offset + 1)
                for start in range(0, len(vertices), 4)
                for offset in range(3)
            ],
            dtype=np.uint32,
        )

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "chains",
                vertices,
                widths=np.ones(len(vertices), dtype=np.float32),
                indices=indices[::-1, ::-1],
                line_type="indexed",
                additive_lod={"n_lods": 3, "method": "random", "seed": 0},
            )

        node = zarr.open(str(output), mode="r")["chains"]
        assert node.attrs["n_additive_sublods"] == 3
        decoder = ArrayDecoder()
        stored_edges = []
        for key in sorted(k for k in node.keys() if k.startswith("additive_")):
            level = node[key]
            level_vertices = np.asarray(decoder.decode(level["vertices"]))
            level_segments = np.asarray(level["segments"])
            for endpoint_a, endpoint_b in level_segments:
                stored_edges.append(
                    tuple(
                        sorted(
                            (
                                tuple(level_vertices[endpoint_a]),
                                tuple(level_vertices[endpoint_b]),
                            )
                        )
                    )
                )

        expected_edges = [
            tuple(sorted((tuple(vertices[endpoint_a]), tuple(vertices[endpoint_b]))))
            for endpoint_a, endpoint_b in indices
        ]
        assert sorted(stored_edges) == sorted(expected_edges)

    def test_indexed_branching_graph_raises_before_writing(self, tmp_path) -> None:
        output = tmp_path / "t.luxar.zarr"
        vertices = np.array(
            [[component, offset, 0.0] for component in range(6) for offset in range(4)],
            dtype=np.float32,
        )
        indices = np.array(
            [
                (start, start + spoke)
                for start in range(0, len(vertices), 4)
                for spoke in range(1, 4)
            ],
            dtype=np.uint32,
        )

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(
                ValueError, match="cannot preserve the explicit edge list"
            ):
                scene.add_lines(
                    "stars",
                    vertices,
                    widths=np.ones(len(vertices), dtype=np.float32),
                    indices=indices,
                    line_type="indexed",
                    additive_lod={"n_lods": 3},
                )

        assert "stars" not in zarr.open(str(output), mode="r")

    @pytest.mark.parametrize(
        ("component_count", "n_lods"),
        [(1, 3), (6, 1)],
        ids=["single-component", "single-requested-level"],
    )
    def test_indexed_branching_graph_writes_flat_when_no_ladder_is_emitted(
        self,
        tmp_path,
        component_count: int,
        n_lods: int,
    ) -> None:
        from luxar.encoding import ArrayDecoder

        output = tmp_path / "t.luxar.zarr"
        vertices = np.array(
            [
                [component, offset, 0.0]
                for component in range(component_count)
                for offset in range(4)
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                (start, start + spoke)
                for start in range(0, len(vertices), 4)
                for spoke in range(1, 4)
            ],
            dtype=np.uint32,
        )

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "stars",
                vertices,
                widths=np.ones(len(vertices), dtype=np.float32),
                indices=indices,
                line_type="indexed",
                additive_lod={"n_lods": n_lods},
            )

        node = zarr.open(str(output), mode="r")["stars"]
        assert "n_additive_sublods" not in node.attrs
        stored_vertices = np.asarray(ArrayDecoder().decode(node["vertices"]))
        stored_edges = [
            tuple(
                sorted(
                    (
                        tuple(stored_vertices[endpoint_a]),
                        tuple(stored_vertices[endpoint_b]),
                    )
                )
            )
            for endpoint_a, endpoint_b in np.asarray(node["segments"])
        ]
        expected_edges = [
            tuple(sorted((tuple(vertices[endpoint_a]), tuple(vertices[endpoint_b]))))
            for endpoint_a, endpoint_b in indices
        ]
        assert sorted(stored_edges) == sorted(expected_edges)

    def test_image_labels_suppress_ladder_and_are_kept(self, tmp_path) -> None:
        # Regression: the plain additive multi-LOD writer has no image_labels
        # channel, so an explicit ladder used to SILENTLY DROP the labels. It
        # must instead refuse the ladder (write a single leaf) and keep the
        # labels — mirroring the substitutive path's suppress_reason guard —
        # and warn, since the explicit request cannot be honoured.
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        vertices = rng.rand(40, 3).astype(np.float32)
        widths = np.ones(40, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.warns(UserWarning, match="cannot be honoured"):
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

    def test_uniform_lengths_size_first_level_by_vertex_budget(self) -> None:
        # 500 uniform two-vertex polylines: cuts are sized by CUMULATIVE
        # vertices, so a stream:200 vertex budget is exactly the first 100 whole
        # polylines (100 × 2 = 200 vertices).
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

    @staticmethod
    def _skewed_indexed(
        n_big: int = 50, big_len: int = 100, n_small: int = 500
    ) -> tuple:
        # Skewed `indexed` set: `n_big` "big" polylines (`big_len` vertices each,
        # spanning a long physical distance) + `n_small` tiny 2-vertex polylines.
        # `salience` puts the big polylines FIRST in the additive order, so the
        # early cuts are dominated by the large polylines — the exact regime
        # where sizing the `stream:` chunk from the MEAN polyline length misses
        # the budget. Returns (verts, indices, widths, n, p, big_len).
        n = n_big * big_len + 2 * n_small
        p = n_big + n_small
        verts = np.zeros((n, 3), dtype=np.float32)

        edges = []
        for b in range(n_big):
            base = b * big_len
            # A chain of `big_len` vertices with unit spacing → physical
            # length big_len - 1 (large salience).
            verts[base : base + big_len, 0] = np.arange(big_len, dtype=np.float32)
            verts[base : base + big_len, 1] = float(b * 10)
            a = np.arange(base, base + big_len - 1, dtype=np.int64)
            edges.append(np.stack([a, a + 1], axis=1))

        sbase = n_big * big_len
        # Tiny two-vertex polylines with a minute physical length (small
        # salience → they sort AFTER every big polyline).
        verts[sbase : sbase + 2 * n_small : 2, 0] = 0.0
        verts[sbase + 1 : sbase + 2 * n_small : 2, 0] = 0.01
        sa = np.arange(sbase, sbase + 2 * n_small, 2, dtype=np.int64)
        edges.append(np.stack([sa, sa + 1], axis=1))

        indices = np.concatenate(edges, axis=0)
        widths = np.ones(n, dtype=np.float32)
        return verts, indices, widths, n, p, big_len

    def test_skewed_lengths_track_the_vertex_budget_not_polyline_count(
        self,
    ) -> None:
        # n = 6000 vertices, p = 550 polylines, mean length = 6000/550 ≈ 10.9.
        # OLD (mean-based) code: c_polys = round(500 / 10.9) = 46, so the FIRST
        # level is the first 46 polylines = 46 big polylines = 4600 vertices —
        # 9.2x the first geometric vertex target of 500. NEW code sizes cuts
        # against the ACTUAL cumulative vertex count, so the first level reaches
        # exactly 500 vertices (5 big polylines) and every level tracks the
        # geometric [500, 1000, 2000, 4000] vertex schedule. chunk=500 is an exact
        # multiple of big_len, so every target lands on a whole-polyline boundary
        # (cuts == targets) — this pins the undershoot off-by-one cleanly; the
        # sibling test below exercises the mid-polyline overshoot regime.
        from luxar.utils.lod_breakpoints import stream_cuts

        verts, indices, widths, n, _p, big_len = self._skewed_indexed()

        chunk = 500
        levels = make_additive_lod_lines(
            verts,
            line_type="indexed",
            indices=indices,
            widths=widths,
            method="salience",
            counts=f"stream:{chunk}",
        )

        # Each cut is the FIRST whole-polyline boundary whose cumulative vertices
        # REACH its geometric target 2^k · C, so it lands in
        # [target, target + big_len] — it reaches the target and overshoots by at
        # most the one crossing polyline. The OLD mean-based sizing produces a
        # first cut of 4600 vertices (9.2x its 500-vertex target) and fails this.
        level_vertex_counts = [sum(int(m.size) for m in level) for level in levels]
        cum_cuts = np.cumsum(level_vertex_counts)[:-1]
        targets = stream_cuts(n, chunk)[:-1]
        assert len(cum_cuts) == len(targets)
        for cut, target in zip(cum_cuts, targets):
            assert target <= cut <= target + big_len, (cut, target)

        # Whole-polyline (segment-topology) invariant: every returned member is a
        # COMPLETE polyline — a tiny 2-vertex one or a whole 100-vertex big one.
        # This is what fails if an implementation sliced perm-ordered VERTICES at
        # the exact targets (which would still partition perfectly below).
        assert all(int(m.size) in (2, big_len) for level in levels for m in level)

        # Whole-polyline integrity: the concatenation of every level's polylines
        # is a partition of all n vertices — no split, loss, or duplication.
        joined = np.concatenate([m for level in levels for m in level])
        assert joined.size == n
        assert np.array_equal(np.unique(joined), np.arange(n))

    def test_skewed_lengths_overshoot_lands_within_one_polyline(self) -> None:
        # Same skewed fixture, but chunk=530 is NOT a multiple of big_len=100, so
        # no geometric target lands on a whole-polyline boundary. Each cut must
        # therefore overshoot its target by the one crossing big polyline:
        # targets = stream_cuts(6000, 530)[:-1] = [530, 1060, 2120, 4240] and the
        # cuts land at [600, 1100, 2200, 4300] — each strictly past its target but
        # within one big_len of it. This exercises the mid-polyline overshoot
        # regime the exact-alignment (chunk=500) case can never reach.
        from luxar.utils.lod_breakpoints import stream_cuts

        verts, indices, widths, n, _p, big_len = self._skewed_indexed()

        chunk = 530
        levels = make_additive_lod_lines(
            verts,
            line_type="indexed",
            indices=indices,
            widths=widths,
            method="salience",
            counts=f"stream:{chunk}",
        )

        level_vertex_counts = [sum(int(m.size) for m in level) for level in levels]
        cum_cuts = np.cumsum(level_vertex_counts)[:-1]
        targets = stream_cuts(n, chunk)[:-1]
        assert len(cum_cuts) == len(targets)
        for cut, target in zip(cum_cuts, targets):
            # Reaches the target and overshoots by at most the crossing polyline.
            assert target <= cut <= target + big_len, (cut, target)
            # Strictly past the target: the mid-polyline overshoot the
            # exact-alignment case cannot reach (there the cut equals the target).
            assert cut > target, (cut, target)

        # Same whole-polyline (segment-topology) invariant and full partition.
        assert all(int(m.size) in (2, big_len) for level in levels for m in level)
        joined = np.concatenate([m for level in levels for m in level])
        assert joined.size == n
        assert np.array_equal(np.unique(joined), np.arange(n))


class TestRevealSpatialDimsFromSceneLines:
    """``add_lines`` fills ``spatial_dims`` from the scene, like ``add_points``.

    The extent rule alone cannot drop a STACKED time column (it varies across
    polylines exactly like a spatial axis does), so the adder passes the scene's
    displayed dims. Six widely-spaced timepoints x two spatial shells at
    |x| = 1 and 40, symmetric about x = 0 so the shells sit at genuinely
    different radii from the bbox centre; one 2-vertex segment per (t, x).
    """

    _TIMES = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)

    @classmethod
    def _verts(cls) -> np.ndarray:
        rows = []
        for t in cls._TIMES:
            for x in (1.0, -1.0, 40.0, -40.0):
                rows.append([t, x, 0.0, 0.0])
                rows.append([t, x, 1.0, 0.0])
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _dims_4d() -> Dimensions:
        return Dimensions(
            [
                Dimension("time", range=(0.0, 100.0), discrete=True, display=False),
                Dimension("x", display=True),
                Dimension("y", display=True),
                Dimension("z", display=True),
            ]
        )

    def test_stacked_time_does_not_delay_an_off_centre_timepoint(
        self, tmp_path
    ) -> None:
        verts = self._verts()
        output = tmp_path / "reveal_lines_4d.luxar.zarr"
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=self._dims_4d())
            scene.add_lines(
                "ln",
                verts,
                widths=np.full(len(verts), 0.5, np.float32),
                line_type="segments",
                additive_lod=dict(method="radial", n_lods=2),
            )

        from luxar.encoding import ArrayDecoder

        grp = zarr.open(str(output), mode="r")["ln"]
        assert int(grp.attrs["n_additive_sublods"]) == 2
        first = np.asarray(
            ArrayDecoder().decode(grp["additive_0"]["vertices"]), dtype=np.float64
        )
        # Level 0 = the inner shell at EVERY timepoint, whole polylines only.
        assert first.shape[0] == 24
        np.testing.assert_allclose(np.abs(first[:, 1]), 1.0, atol=1e-2)
        assert sorted(set(np.round(first[:, 0]).tolist())) == list(self._TIMES)

    def test_a_same_column_count_dim_order_permutation_stays_aligned(
        self, tmp_path
    ) -> None:
        """A permuting ``dim_order`` does not misalign the scene-derived dims.

        ``default_reveal_spatial_dims`` treats a scene-dimension index as a
        position-column index once the column COUNTS match, which looks unsafe
        under a ``dim_order`` permutation that preserves the count. It is safe,
        and this pins why: both adders call ``apply_dim_order_positions`` BEFORE
        reading ``ndim`` or resolving the reveal dims, and ``apply_dim_order``
        builds ``np.zeros((N, scene_ndim))`` filled by iterating the SCENE's names
        — so the array reaching the resolver is already in scene order at scene
        dimensionality. Authoring the identical geometry column-permuted must
        therefore give the identical ladder.
        """
        canonical = self._verts()  # columns [time, x, y, z]
        permuted = canonical[:, [1, 2, 3, 0]]  # authored as [x, y, z, time]

        first_of = {}
        for tag, verts, dim_order in (
            ("canonical", canonical, None),
            ("permuted", permuted, ["x", "y", "z", "time"]),
        ):
            output = tmp_path / f"reveal_dimorder_{tag}.luxar.zarr"
            with LuxarZarrCompiler(output) as compiler:
                scene = compiler.create_scene(dimensions=self._dims_4d())
                scene.add_lines(
                    "ln",
                    verts,
                    widths=np.full(len(verts), 0.5, np.float32),
                    line_type="segments",
                    dim_order=dim_order,
                    additive_lod=dict(method="radial", n_lods=2),
                )

            from luxar.encoding import ArrayDecoder

            grp = zarr.open(str(output), mode="r")["ln"]
            assert int(grp.attrs["n_additive_sublods"]) == 2
            first_of[tag] = np.asarray(
                ArrayDecoder().decode(grp["additive_0"]["vertices"]), dtype=np.float64
            )

        # Same inner shell at every timepoint, in the same scene column order —
        # i.e. the permutation was normalized away before the reveal was scored.
        for tag, first in first_of.items():
            assert first.shape[0] == 24, tag
            np.testing.assert_allclose(np.abs(first[:, 1]), 1.0, atol=1e-2, err_msg=tag)
        np.testing.assert_allclose(
            np.sort(first_of["canonical"], axis=0),
            np.sort(first_of["permuted"], axis=0),
            atol=1e-2,
        )

    def test_derived_shell_axes_write_no_partial_group(self, tmp_path) -> None:
        """A mismatched ``reveal_centre`` must not strand a partial LOD group.

        The resolver can only cross-check the centre against ``spatial_dims``
        when the caller names both; here the axes are DERIVED. A planar cloud
        (constant z) resolves to shell axes ``[0, 1]``, so the natural
        3-coordinate centre used to raise inside the scorer — which for a
        substitutive ladder runs while writing the FINEST child, after the
        wrapper group and every coarse gsplat child are on disk.

        For Lines the derivation runs over the per-polyline bbox CENTRES (the
        scorer ranks whole polylines), which is what the wrapper now checks.
        """
        rng = np.random.RandomState(0)
        verts = np.zeros((200, 3), dtype=np.float32)
        verts[:, 0] = rng.uniform(-50, 50, 200)
        verts[:, 1] = rng.uniform(-50, 50, 200)
        verts[:, 2] = 7.0  # constant column -> zero extent -> shell axes [0, 1]
        widths = np.full(len(verts), 0.5, np.float32)

        output = tmp_path / "reveal_lines_partial.luxar.zarr"
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="shell axes resolve to"):
                scene.add_lines(
                    "ln",
                    verts,
                    widths=widths,
                    line_type="segments",
                    substitutive_lod={"levels": 2, "compression_factor": 4},
                    additive_lod={
                        "method": "radial",
                        "reveal_centre": [0.0, 0.0, 7.0],
                    },
                )
        assert not (output / "ln").exists()

        # SENSITIVITY CONTROL: a centre of the matching length still builds.
        ok = tmp_path / "reveal_lines_ok.luxar.zarr"
        with LuxarZarrCompiler(ok) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                verts,
                widths=widths,
                line_type="segments",
                substitutive_lod={"levels": 2, "compression_factor": 4},
                additive_lod={"method": "radial", "reveal_centre": [0.0, 0.0]},
            )
        assert (ok / "ln" / "child_2").exists()

    def test_preflight_does_no_work_without_an_explicit_centre(
        self, tmp_path, monkeypatch
    ) -> None:
        """The wrapper must not derive polyline representatives it will not use.

        The preflight only has something to cross-check when the caller named a
        ``reveal_centre`` under a reveal ordering. Getting its ``coords`` argument
        is the expensive part on Lines — ``identify_polylines`` plus
        ``polyline_bbox_centres`` loop in Python over every polyline (~2.5 s for a
        400k-vertex ``segments`` node) — and the composed ladder is ON by default,
        so an unguarded call paid that on every ``add_lines(substitutive_lod=…)``.

        Booby-trap the derivation: the default ladder must never reach it.
        """
        from luxar.core.group.lod import lines as lines_lod

        def _boom(*_args, **_kwargs):  # pragma: no cover - must not be called
            raise AssertionError("polyline representatives derived for a non-reveal")

        monkeypatch.setattr(lines_lod, "polyline_bbox_centres", _boom)

        rng = np.random.RandomState(0)
        verts = rng.uniform(-50, 50, (200, 3)).astype(np.float32)
        widths = np.full(len(verts), 0.5, np.float32)

        output = tmp_path / "no_preflight.luxar.zarr"
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                verts,
                widths=widths,
                line_type="segments",
                substitutive_lod={"levels": 2, "compression_factor": 4},
            )
        assert (output / "ln" / "child_2").exists()

        # SENSITIVITY CONTROL: the trap DOES fire once a centre is named, so the
        # test above proves the guard rather than a broken monkeypatch target.
        trapped = tmp_path / "preflight_runs.luxar.zarr"
        with LuxarZarrCompiler(trapped) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(AssertionError, match="derived for a non-reveal"):
                scene.add_lines(
                    "ln",
                    verts,
                    widths=widths,
                    line_type="segments",
                    substitutive_lod={"levels": 2, "compression_factor": 4},
                    additive_lod={
                        "method": "radial",
                        "reveal_centre": [0.0, 0.0, 0.0],
                    },
                )

    def test_control_the_extent_rule_alone_mixes_the_shells(self) -> None:
        """Sensitivity control on identical data, through the bare-array API."""
        verts = self._verts()
        levels = make_additive_lod_lines(
            verts,
            line_type="segments",
            widths=np.full(len(verts), 0.5, np.float32),
            method="radial",
            n_lods=2,
        )
        first = np.concatenate([verts[m] for m in levels[0]])
        assert np.abs(first[:, 1]).max() > 1.0
        assert sorted(set(np.round(first[:, 0]).tolist())) != list(self._TIMES)


def test_no_sub_LOD_carries_the_private_skip_scene_bounds_flag(tmp_path) -> None:
    """`_skip_scene_bounds` is plumbing between writers, not part of the format.

    The Points-side test of the same name states the full reasoning; this is the
    Lines half of the same three-writer fix. The flag was popped BELOW
    `group.attrs.update(attrs)`, so every sub-LOD carried it on disk.
    """
    output = tmp_path / "t.luxar.zarr"
    verts = np.random.RandomState(0).rand(200, 3).astype(np.float32)
    widths = np.full(len(verts), 0.5, np.float32)
    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_lines(
            "ln",
            verts,
            widths=widths,
            line_type="segments",
            additive_lod=dict(n_lods=3, method="random"),
        )

    parent = zarr.open_group(str(output), mode="r")["ln"]
    assert "_skip_scene_bounds" not in parent.attrs
    for i in range(int(parent.attrs["n_additive_sublods"])):
        assert "_skip_scene_bounds" not in parent[f"additive_{i}"].attrs, (
            f"additive_{i} carries the private flag; it is popped after "
            "`group.attrs.update(attrs)` again"
        )
    assert "position_bounds" in parent.attrs
