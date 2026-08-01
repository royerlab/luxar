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

    def test_default_method_lands_on_parent_attrs(self, tmp_path) -> None:
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(3)
        positions = rng.rand(40, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", positions, additive_lod=True)

        store = zarr.open(str(output), mode="r")
        grp = store["pts"]
        assert grp.attrs["additive_lod_method"] == "random"

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
