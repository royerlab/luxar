"""Tests for additive-LOD support on Points.

Covers:

- The pure ordering helpers (``compute_additive_order_points``).
- The ladder constructor (``make_additive_lod_points``).
- The resolver (``resolve_additive_axis_points``).
- End-to-end ``add_points(..., additive_lod=...)`` round-tripping through
  the writer; verify the on-disk layout matches the contract.
- Edge cases: empty, single-element, fewer-elements-than-n_lods.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.points import (
    DEFAULT_METHOD,
    DEFAULT_N_LODS,
    compute_additive_order_points,
    make_additive_lod_points,
    resolve_additive_axis_points,
)
from luxar.core.points import Points
from luxar.io.compiler import LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# Ordering helpers
# ────────────────────────────────────────────────────────────────────────


class TestComputeAdditiveOrderPoints:
    def test_random_returns_permutation(self) -> None:
        pos = np.random.RandomState(0).rand(50, 3).astype(np.float32)
        perm, counts = compute_additive_order_points(pos, method="random", seed=42)
        assert perm.shape == (50,)
        assert sorted(perm.tolist()) == list(range(50))
        assert counts == []

    def test_random_is_reproducible_with_seed(self) -> None:
        pos = np.random.RandomState(0).rand(20, 3).astype(np.float32)
        p1, _ = compute_additive_order_points(pos, method="random", seed=7)
        p2, _ = compute_additive_order_points(pos, method="random", seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_salience_sorts_by_radii_descending(self) -> None:
        pos = np.random.RandomState(0).rand(10, 3).astype(np.float32)
        radii = np.linspace(0.1, 1.0, 10, dtype=np.float32)
        perm, _ = compute_additive_order_points(pos, radii=radii, method="salience")
        # Largest radius (index 9) should be first.
        assert perm[0] == 9
        assert perm[-1] == 0

    def test_salience_requires_radii(self) -> None:
        pos = np.zeros((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="requires per-element radii"):
            compute_additive_order_points(pos, method="salience")

    def test_spatial_uniform_returns_counts(self) -> None:
        pos = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        perm, counts = compute_additive_order_points(
            pos, method="spatial-uniform", n_lods=4
        )
        assert perm.shape == (100,)
        assert sum(counts) == 100
        assert len(counts) == 4
        # Coarsest LOD should have ≤ 8 elements (2x2x2 = 8 buckets).
        assert counts[0] <= 8

    def test_spatial_uniform_deep_levels_no_overflow(self) -> None:
        """L1: many levels over coincident data must not overflow int64
        (res*res exceeded C long at level >= 31)."""
        from luxar.core.group.lod.spatial_uniform import stratified_grid_order

        pos = np.zeros((40, 3), dtype=np.float32)  # all coincident
        perm, counts = stratified_grid_order(pos, n_lods=64)
        assert sorted(perm.tolist()) == list(range(40))  # valid permutation
        assert sum(counts) == 40
        # And through the public additive-LOD entry point.
        perm2, counts2 = compute_additive_order_points(
            pos, method="spatial-uniform", n_lods=64
        )
        assert sorted(perm2.tolist()) == list(range(40))

    def test_unknown_method_raises(self) -> None:
        pos = np.zeros((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="method must be"):
            compute_additive_order_points(pos, method="bogus")  # type: ignore[arg-type]

    def test_empty_input(self) -> None:
        pos = np.zeros((0, 3), dtype=np.float32)
        perm, counts = compute_additive_order_points(pos)
        assert perm.shape == (0,)
        assert counts == []


class TestRadialOrderPoints:
    """``radial`` — the concentric-shell reveal ordering.

    It is the only method that sorts ASCENDING (its score is a distance, not a
    contribution to maximise), so the mutation that must fail these is flipping
    the sort direction.
    """

    #: Distances 1, 2, 3, 4 from the ORIGIN, one per axis so all three columns
    #: have extent (hence `spatial_dims` defaults to all of them), and in an
    #: order that is not already sorted — so a pass cannot be explained by the
    #: input arriving pre-ordered.
    _PTS = np.array(
        [
            [0.0, 2.0, 0.0],  # d = 2
            [3.0, 0.0, 0.0],  # d = 3
            [0.0, 0.0, 1.0],  # d = 1
            [0.0, 0.0, 4.0],  # d = 4
        ],
        dtype=np.float32,
    )

    def test_orders_innermost_first(self) -> None:
        perm, counts = compute_additive_order_points(
            self._PTS, method="radial", reveal_centre=[0.0, 0.0, 0.0]
        )
        assert perm.tolist() == [2, 0, 1, 3]
        assert counts == [], "radial must leave slicing to the breakpoint vocabularies"

    def test_default_centre_is_the_bbox_centre_not_the_origin(self) -> None:
        # Spread on x only, so `spatial_dims` derives to [0] and the centre takes
        # ONE coordinate. bbox centre is x=15, so x=12 is innermost; an
        # origin-centred (or scene-centred) implementation would start at x=10.
        pos = np.array(
            [[10.0, 0.0, 0.0], [12.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32
        )
        perm, _ = compute_additive_order_points(pos, method="radial")
        assert pos[perm[0], 0] == 12.0

        pinned, _ = compute_additive_order_points(
            pos, method="radial", reveal_centre=[0.0]
        )
        assert pos[pinned[0], 0] == 10.0

    def test_is_translation_invariant(self) -> None:
        # The bbox-centre default is what makes this hold: a dataset 1000 units
        # from the origin still reveals from its own middle, identically.
        far = self._PTS + np.float32(1000.0)
        near, _ = compute_additive_order_points(self._PTS, method="radial")
        moved, _ = compute_additive_order_points(far, method="radial")
        np.testing.assert_array_equal(near, moved)

    def test_zero_extent_column_is_not_a_shell_dimension(self) -> None:
        # A stacked time/channel column is a real coordinate here (unlike the
        # gsplat path's degenerate covariance axes), so it must be excluded by
        # extent or the shells would expand through time as well as space —
        # every timepoint of the inner shell before any of the next.
        pos4 = np.hstack(
            [self._PTS, np.full((self._PTS.shape[0], 1), 7.0, dtype=np.float32)]
        )
        p3, _ = compute_additive_order_points(self._PTS, method="radial")
        p4, _ = compute_additive_order_points(pos4, method="radial")
        np.testing.assert_array_equal(p3, p4)

    def test_explicit_spatial_dims_restrict_the_distance(self) -> None:
        # Measuring over x alone ignores a y spread that would otherwise dominate.
        pos = np.array(
            [[0.0, 50.0, 0.0], [9.0, 0.0, 0.0], [1.0, 40.0, 0.0]], dtype=np.float32
        )
        perm, _ = compute_additive_order_points(
            pos, method="radial", reveal_centre=[0.0], spatial_dims=[0]
        )
        assert pos[perm, 0].tolist() == [0.0, 1.0, 9.0]

    def test_out_of_range_spatial_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            compute_additive_order_points(self._PTS, method="radial", spatial_dims=[9])

    @pytest.mark.parametrize(
        ("dims", "match"),
        [
            ([], "must not be empty"),
            ([-1], "non-negative"),
            ([0, 0], "must not repeat"),
        ],
        ids=["empty", "negative", "duplicate"],
    )
    def test_malformed_spatial_dims_raise_rather_than_misorder(
        self, dims: list[int], match: str
    ) -> None:
        """Each of these silently produced a WRONG ordering before being caught.

        `resolve_additive_axis` rejects all three, but this entry point is public
        and bypasses the resolver, so the innermost scorer has to reject them too:
        a negative index ALIASES to another column via numpy indexing, a repeat
        DOUBLE-COUNTS that axis in the distance, and an empty list scores every
        element 0.0 — degrading the ordering to input order with no indication.
        """
        with pytest.raises(ValueError, match=match):
            compute_additive_order_points(self._PTS, method="radial", spatial_dims=dims)

    def test_scorer_on_empty_input_returns_empty(self) -> None:
        """Direct scorer call at N=0 — the bbox reductions have no identity there.

        Both element callers return early at n == 0, so this guards only a direct
        caller of the public helper; the bare numpy error named neither the
        argument nor the function.
        """
        from luxar.core.group.lod.group import radial_element_score

        out = radial_element_score(np.zeros((0, 3), dtype=np.float32))
        assert out.shape == (0,)

    def test_scorer_rejects_zero_column_coords(self) -> None:
        """No columns means no distance; all-zero scores would silently no-op.

        This guard had no test at first, and a later refactor of the surrounding
        block dropped it without anything going red — which is exactly why it has
        one now.
        """
        from luxar.core.group.lod.group import radial_element_score

        with pytest.raises(ValueError, match="at least one column"):
            radial_element_score(np.zeros((4, 0), dtype=np.float32))

    def test_wrong_length_centre_raises(self) -> None:
        with pytest.raises(ValueError, match="one coordinate per spatial axis"):
            compute_additive_order_points(
                self._PTS, method="radial", reveal_centre=[0.0, 0.0]
            )

    def test_works_in_2d_unlike_the_samplers(self) -> None:
        # `spatial-uniform` / `poisson-disk` demand d >= 3; a distance does not,
        # and 2D scenes are a first-class authoring path.
        pos = np.array([[3.0, 1.0], [1.0, 2.0], [5.0, 3.0]], dtype=np.float32)
        perm, _ = compute_additive_order_points(
            pos, method="radial", reveal_centre=[0.0, 0.0]
        )
        assert pos[perm, 0].tolist() == [1.0, 3.0, 5.0]

    def test_single_position_does_not_collapse_to_input_order(self) -> None:
        # No axis has extent, so the non-zero-extent default would select NO
        # columns and score everything 0.0. The fallback uses every axis instead.
        pos = np.zeros((3, 3), dtype=np.float32)
        perm, _ = compute_additive_order_points(pos, method="radial")
        assert sorted(perm.tolist()) == [0, 1, 2]


# ────────────────────────────────────────────────────────────────────────
# Ladder constructor
# ────────────────────────────────────────────────────────────────────────


class TestMakeAdditiveLodPoints:
    def test_equal_count_split_via_n_lods(self) -> None:
        pos = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        levels = make_additive_lod_points(pos, method="random", n_lods=4)
        assert sum(len(L) for L in levels) == 100
        assert len(levels) == 4
        # Each level ~25 elements.
        for L in levels:
            assert 20 <= len(L) <= 30

    def test_counts_breakpoints(self) -> None:
        pos = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        levels = make_additive_lod_points(pos, method="random", counts=[10, 30, 70])
        sizes = [len(L) for L in levels]
        assert sizes == [10, 20, 40, 30]  # cumulative → per-level deltas

    def test_spatial_uniform_uses_natural_partition(self) -> None:
        pos = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        levels = make_additive_lod_points(pos, method="spatial-uniform", n_lods=4)
        assert sum(len(L) for L in levels) == 100

    def test_fewer_elements_than_n_lods(self) -> None:
        """Plan: silently emit fewer levels when n_elements < n_lods."""
        pos = np.random.RandomState(0).rand(2, 3).astype(np.float32)
        levels = make_additive_lod_points(pos, method="random", n_lods=4)
        assert sum(len(L) for L in levels) == 2
        # Levels with 0 elements are dropped.
        assert all(len(L) > 0 for L in levels)

    def test_empty_input(self) -> None:
        pos = np.zeros((0, 3), dtype=np.float32)
        levels = make_additive_lod_points(pos, method="random", n_lods=4)
        assert levels == []

    def test_radial_levels_grow_outward(self) -> None:
        # 100 points on a line; a 4-level equal-count ladder must hand out the
        # innermost quarter first. Checked on max radius per level, which is
        # monotone iff the ordering really is by distance.
        pos = np.zeros((100, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(-50.0, 50.0, 100)
        levels = make_additive_lod_points(pos, method="radial", n_lods=4)

        assert sum(len(L) for L in levels) == 100
        assert len(levels) == 4
        max_r = [float(np.abs(pos[L, 0]).max()) for L in levels]
        assert max_r == sorted(max_r), max_r

    def test_radial_honours_the_stream_vocabulary(self) -> None:
        # The trap this guards: `spatial-uniform` / `poisson-disk` return a
        # natural partition and the builder then BYPASSES the breakpoint
        # vocabularies entirely. `radial` must not join that tuple, or a
        # `stream:`/`counts:` spec would be silently ignored on a reveal.
        pos = np.zeros((100, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(-50.0, 50.0, 100)
        levels = make_additive_lod_points(pos, method="radial", counts=[10, 30, 70])

        assert [len(L) for L in levels] == [10, 20, 40, 30]

    def test_radial_kwargs_reach_the_scorer_through_the_builder(self) -> None:
        # Threading test: the centre override must survive the builder, not just
        # `compute_additive_order_points`. Pinned at one end, the first level is
        # that end; by default (bbox centre) it would be the middle.
        pos = np.zeros((40, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(0.0, 39.0, 40)

        pinned = make_additive_lod_points(
            pos, method="radial", n_lods=4, reveal_centre=[0.0]
        )
        assert float(pos[pinned[0], 0].max()) < 10.0

        default = make_additive_lod_points(pos, method="radial", n_lods=4)
        first = pos[default[0], 0]
        assert float(first.min()) > 10.0 and float(first.max()) < 30.0

    def test_radial_spatial_dims_reach_the_scorer_through_the_builder(self) -> None:
        # The other kwarg, threaded the same way: restricting to x makes the
        # ladder ignore a y spread that would otherwise set the order.
        pos = np.zeros((40, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(0.0, 39.0, 40)
        pos[:, 1] = np.linspace(500.0, 0.0, 40)  # opposing, much larger spread

        restricted = make_additive_lod_points(
            pos, method="radial", n_lods=4, reveal_centre=[0.0], spatial_dims=[0]
        )
        assert float(pos[restricted[0], 0].max()) < 10.0


# ────────────────────────────────────────────────────────────────────────
# Resolver
# ────────────────────────────────────────────────────────────────────────


class TestResolveAdditiveAxisPoints:
    def test_none_is_noop(self) -> None:
        assert resolve_additive_axis_points(None) is None

    def test_false_is_noop(self) -> None:
        """Plan: False has no stored-state meaning for Points; treat as None."""
        assert resolve_additive_axis_points(False) is None

    def test_true_returns_defaults(self) -> None:
        spec = resolve_additive_axis_points(True)
        assert spec is not None
        assert spec["method"] == DEFAULT_METHOD
        assert spec["n_lods"] == DEFAULT_N_LODS

    def test_dict_overrides(self) -> None:
        spec = resolve_additive_axis_points(
            {"method": "salience", "n_lods": 3, "seed": 42}
        )
        assert spec is not None
        assert spec["method"] == "salience"
        assert spec["n_lods"] == 3
        assert spec["seed"] == 42

    def test_recompute_tolerated_for_symmetry(self) -> None:
        # gsplats has recompute=True semantics; Points tolerates without action.
        spec = resolve_additive_axis_points({"recompute": True})
        assert spec is not None  # doesn't raise, returns defaults

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized keys"):
            resolve_additive_axis_points({"bogus": 42})

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            resolve_additive_axis_points({"method": "bogus"})

    def test_invalid_spec_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be None, bool, or dict"):
            resolve_additive_axis_points("auto")  # type: ignore[arg-type]

    def test_valid_stream_counts_resolves(self) -> None:
        spec = resolve_additive_axis_points({"counts": "stream:1000"})
        assert spec is not None
        assert spec["counts"] == "stream:1000"

    def test_malformed_stream_counts_raise_at_resolve(self) -> None:
        # A ``stream:<c>`` with c < 1 must fail at resolve time, before any
        # kind=lod wrapper group is written under a substitutive ladder.
        with pytest.raises(ValueError, match="stream first-chunk size must be >= 1"):
            resolve_additive_axis_points({"counts": "stream:0"})
        with pytest.raises(ValueError, match="stream"):
            resolve_additive_axis_points({"counts": "stream:-5"})

    def test_valid_energy_counts_resolves(self) -> None:
        spec = resolve_additive_axis_points({"counts": "energy:0.5,0.9,1.0"})
        assert spec is not None
        assert spec["counts"] == "energy:0.5,0.9,1.0"

    def test_other_doomed_counts_raise_at_resolve(self) -> None:
        # Same partial-group trap as stream:0 — any counts value the write
        # path is guaranteed to reject must fail at resolve time too.
        with pytest.raises(ValueError, match="unrecognized breakpoints string"):
            resolve_additive_axis_points({"counts": "equal-count"})
        with pytest.raises(ValueError, match="energy: fractions must be numbers"):
            resolve_additive_axis_points({"counts": "energy:abc"})
        with pytest.raises(ValueError, match="energy: fractions must be non-empty"):
            resolve_additive_axis_points({"counts": "energy:"})
        with pytest.raises(ValueError, match="non-empty"):
            resolve_additive_axis_points({"counts": []})


# ────────────────────────────────────────────────────────────────────────
# End-to-end: ``add_points(additive_lod=...)``
# ────────────────────────────────────────────────────────────────────────


class TestAddPointsAdditiveLod:
    def test_colors_and_colormap_rejected_on_additive_path(self, tmp_path) -> None:
        # Regression: the additive multi-LOD branch used to return before the
        # colors/colormap mutual-exclusivity validation, silently accepting
        # invalid combinations that the flat path rejects.
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        positions = rng.rand(200, 3).astype(np.float32)
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="colors.*colormap"):
                scene.add_points(
                    "pts",
                    positions,
                    colors=rng.rand(200, 3).astype(np.float32),
                    colormap="viridis",
                    additive_lod=dict(n_lods=3, method="random"),
                )
            with pytest.raises(ValueError, match="scalars.*colormap"):
                scene.add_points(
                    "pts2",
                    positions,
                    scalars=rng.rand(200).astype(np.float32),
                    additive_lod=dict(n_lods=3, method="random"),
                )

    def test_image_labels_suppress_ladder_and_are_kept(self, tmp_path) -> None:
        # Regression: the plain additive multi-LOD writer has no image_labels
        # channel, so an explicit ladder used to SILENTLY DROP the labels. It
        # must instead refuse the ladder (write a single leaf) and keep the
        # labels — mirroring the substitutive path's suppress_reason guard —
        # and warn, since the explicit request cannot be honoured.
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        positions = rng.rand(200, 3).astype(np.float32)
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.warns(UserWarning, match="cannot be honoured"):
                scene.add_points(
                    "pts",
                    positions,
                    image_labels=[b"x"] * 200,
                    additive_lod=dict(n_lods=3, method="random"),
                )
        grp = zarr.open(str(output), mode="r")["pts"]
        assert grp.attrs["type"] == "points"
        assert grp.attrs["n_points"] == 200
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
        positions = np.random.RandomState(0).rand(20, 3).astype(np.float32)
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="method must be"):
                scene.add_points(
                    "pts",
                    positions,
                    image_labels=[b"x"] * 20,
                    additive_lod=dict(method="bogus"),
                )

    def test_default_true_writes_4_levels(self, tmp_path) -> None:
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        positions = rng.rand(200, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", positions, additive_lod=True)

        store = zarr.open(str(output), mode="r")
        grp = store["pts"]
        assert grp.attrs["type"] == "points"
        assert grp.attrs["n_points"] == 200
        assert grp.attrs["n_additive_sublods"] == 4
        subgroups = sorted(k for k in grp.keys() if k.startswith("additive_"))
        assert subgroups == ["additive_0", "additive_1", "additive_2", "additive_3"]
        # Total across subgroups equals total.
        total = sum(int(grp[s].attrs["n_points"]) for s in subgroups)
        assert total == 200

    def test_dict_with_explicit_n_lods(self, tmp_path) -> None:
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(1)
        positions = rng.rand(60, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts", positions, additive_lod=dict(n_lods=3, method="random", seed=42)
            )

        store = zarr.open(str(output), mode="r")
        grp = store["pts"]
        assert grp.attrs["n_additive_sublods"] == 3

    def test_counts_breakpoints_via_kwarg(self, tmp_path) -> None:
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(2)
        positions = rng.rand(100, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                positions,
                additive_lod=dict(method="random", counts=[10, 30, 70]),
            )

        store = zarr.open(str(output), mode="r")
        grp = store["pts"]
        # 4 levels emerge (3 explicit breakpoints + a tail).
        assert grp.attrs["n_additive_sublods"] == 4
        sizes = [int(grp[f"additive_{i}"].attrs["n_points"]) for i in range(4)]
        assert sizes == [10, 20, 40, 30]

    def test_single_shot_path_still_works(self, tmp_path) -> None:
        """No ``additive_lod=`` → existing single-LOD layout."""
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(4)
        positions = rng.rand(50, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", positions)
            assert isinstance(node, Points)

        store = zarr.open(str(output), mode="r")
        grp = store["pts"]
        assert grp.attrs["type"] == "points"
        assert "n_additive_sublods" not in grp.attrs
        # The original data arrays live directly under the path.
        assert "positions" in grp

    def test_layer_flag_lands_on_parent(self, tmp_path) -> None:
        """``layer=True`` rides onto the parent multi-LOD node."""
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(5)
        positions = rng.rand(100, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", positions, additive_lod=True, layer=True)

        store = zarr.open(str(output), mode="r")
        grp = store["pts"]
        assert grp.attrs.get("layer") is True


class TestStreamBreakpoints:
    """``stream:<c>`` — the geometric ladder shared with the GSplats path.

    This is the shape that makes a large leaf paint progressively; an
    equal-count split still ends in an N/n_lods-sized commit.
    """

    def test_geometric_level_sizes(self) -> None:
        rng = np.random.RandomState(0)
        positions = rng.rand(10_000, 3).astype(np.float32)

        levels = make_additive_lod_points(positions, counts="stream:1000")

        # Cumulative cuts [1000, 2000, 4000, 8000, 10000] -> increments below.
        assert [lvl.size for lvl in levels] == [1000, 1000, 2000, 4000, 2000]

    def test_levels_are_a_permutation_of_every_index(self) -> None:
        rng = np.random.RandomState(1)
        positions = rng.rand(5_000, 3).astype(np.float32)

        levels = make_additive_lod_points(positions, counts="stream:400")

        joined = np.concatenate(levels)
        assert joined.size == 5_000
        assert np.array_equal(np.unique(joined), np.arange(5_000))

    def test_first_level_is_the_chunk_size(self) -> None:
        rng = np.random.RandomState(2)
        positions = rng.rand(1_000, 3).astype(np.float32)

        levels = make_additive_lod_points(positions, counts="stream:64")

        assert levels[0].size == 64

    def test_smaller_than_one_chunk_collapses_to_a_single_level(self) -> None:
        # This is what lets a coarse level of a composed ladder stay a flat
        # leaf without any special-casing at the call site.
        rng = np.random.RandomState(3)
        positions = rng.rand(500, 3).astype(np.float32)

        levels = make_additive_lod_points(positions, counts="stream:40000")

        assert len(levels) == 1
        assert levels[0].size == 500

    def test_unrecognized_string_names_both_vocabularies(self) -> None:
        positions = np.random.RandomState(4).rand(100, 3).astype(np.float32)

        with pytest.raises(ValueError, match="energy:.*stream:|stream:.*energy:"):
            make_additive_lod_points(positions, counts="bogus:1,2")

    def test_end_to_end_writes_a_geometric_ladder(self, tmp_path) -> None:
        output = tmp_path / "t.luxar.zarr"
        positions = np.random.RandomState(6).rand(4_000, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", positions, additive_lod=dict(counts="stream:500"))

        grp = zarr.open(str(output), mode="r")["pts"]
        n_sub = int(grp.attrs["n_additive_sublods"])
        sizes = [int(grp[f"additive_{i}"].attrs["n_points"]) for i in range(n_sub)]
        assert sizes == [500, 500, 1000, 2000]
        assert sum(sizes) == 4_000
