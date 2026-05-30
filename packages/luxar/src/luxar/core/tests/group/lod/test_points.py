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


# ────────────────────────────────────────────────────────────────────────
# End-to-end: ``add_points(additive_lod=...)``
# ────────────────────────────────────────────────────────────────────────


class TestAddPointsAdditiveLod:
    def test_default_true_writes_4_levels(self, tmp_path) -> None:
        output = tmp_path / "t.zarr"
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
        output = tmp_path / "t.zarr"
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
        output = tmp_path / "t.zarr"
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
        output = tmp_path / "t.zarr"
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
        output = tmp_path / "t.zarr"
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
        output = tmp_path / "t.zarr"
        rng = np.random.RandomState(5)
        positions = rng.rand(100, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", positions, additive_lod=True, layer=True)

        store = zarr.open(str(output), mode="r")
        grp = store["pts"]
        assert grp.attrs.get("layer") is True
