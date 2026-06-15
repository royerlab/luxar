"""Unit tests for the representation-recipe builders (``gsplats.lod.recipes``).

Pure builder tests — no CLI, no zarr IO, ``device="cpu"`` throughout. They assert
the *tree shape* each recipe produces, splat-count conservation, and that the
three absorbed primitives (``additive`` / ``substitutive`` / ``pyramid``) are
structurally identical to calling the underlying ``make_*_lod`` builders directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.additive import make_additive_lod
from luxar.gsplats.lod.pyramid import make_lod_pyramid
from luxar.gsplats.lod.recipes import (
    RECIPE_NAMES,
    RecipeParams,
    build_recipe,
)
from luxar.gsplats.lod.substitutive import make_substitutive_lod
from luxar.gsplats.tree import (
    GSplatLeaf,
    GSplatLodGroup,
    GSplatPartition,
    iter_leaves,
    total_splats,
)


def _make_random_gsplat(n: int = 400, ndim: int = 3, seed: int = 0) -> GSplatData:
    """Random anisotropic 3D splats spread over a [0, 100] box (so BSP splits)."""
    rng = np.random.default_rng(seed)
    k = ndim * (ndim + 1) // 2
    chol = np.zeros((n, k), dtype=np.float32)
    diag = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag] = rng.uniform(0.5, 2.0, size=(n, ndim))
    return GSplatData(
        centers=rng.uniform(0, 100, size=(n, ndim)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        cholesky_factors=chol,
    )


def _params(**kw) -> RecipeParams:
    base = dict(
        n_lods=3, max_elements=120, compression_factor=4, levels=2, device="cpu"
    )
    base.update(kw)
    return RecipeParams(**base)


# ── shape per recipe ────────────────────────────────────────────────────


def test_all_recipe_names_build():
    data = _make_random_gsplat()
    for recipe in RECIPE_NAMES:
        result = build_recipe(data, recipe, _params())
        # Strong: the build must carry the full splat set, not merely be non-None.
        # Matrix recipes return GSplatData (count == N); composed recipes return a
        # node tree whose leaves total >= N (multiscale adds a coarse cap).
        if isinstance(result, GSplatData):
            assert result.flattened().n_splats == 400, recipe
        else:
            assert total_splats(result) >= 400, recipe


@pytest.mark.parametrize("ndim", [2, 4])
def test_matrix_recipes_are_dimension_agnostic(ndim: int):
    """flat/additive/substitutive/pyramid work at any dimensionality (boundary:
    ndim != 3) and conserve the splat count."""
    data = _make_random_gsplat(n=120, ndim=ndim)
    for recipe in ("flat", "additive", "substitutive", "pyramid"):
        res = build_recipe(data, recipe, _params())
        assert isinstance(res, GSplatData), recipe
        assert res.ndim == ndim, recipe
        assert res.flattened().n_splats == 120, recipe


def test_partition_recipes_support_4d():
    """partitioned/multiscale require >=3 dims (BSP); 4D must build and conserve."""
    data = _make_random_gsplat(n=400, ndim=4)
    part = build_recipe(data, "partitioned", _params(max_elements=120))
    assert isinstance(part, GSplatPartition)
    assert total_splats(part) == 400
    assert all(leaf.ndim == 4 for leaf in iter_leaves(part))
    ms = build_recipe(data, "multiscale", _params(max_elements=120))
    assert isinstance(ms, GSplatLodGroup)
    assert all(leaf.ndim == 4 for leaf in iter_leaves(ms))


def test_flat_is_single_leaf():
    data = _make_random_gsplat(n=200)
    res = build_recipe(data, "flat", _params())
    assert isinstance(res, GSplatData)
    assert res.n_substitutive == 1
    assert res.n_additive_sublods == 1
    assert res.n_splats == 200


def test_additive_is_single_leaf_with_ladder():
    data = _make_random_gsplat(n=200)
    res = build_recipe(
        data, "additive", _params(n_lods=4, additive_method="self_energy")
    )
    assert isinstance(res, GSplatData)
    assert res.n_substitutive == 1
    assert res.n_additive_sublods == 4
    # prefix counts grow monotonically to the full set
    counts = [res.additive_prefix(k).n_splats for k in range(res.n_additive_sublods)]
    assert counts == sorted(counts)
    assert counts[-1] == 200


def test_substitutive_is_lod_group_matrix():
    data = _make_random_gsplat(n=256)
    res = build_recipe(data, "substitutive", _params(compression_factor=4, levels=2))
    assert isinstance(res, GSplatData)
    assert res.n_substitutive == 3  # levels + 1
    assert res.tree.__class__ is GSplatLodGroup


def test_pyramid_is_substitutive_times_additive():
    data = _make_random_gsplat(n=256)
    res = build_recipe(
        data, "pyramid", _params(compression_factor=4, levels=2, n_lods=2)
    )
    assert isinstance(res, GSplatData)
    assert res.n_substitutive == 3
    for lev in res.substitutive_levels:
        assert 1 <= lev.n_additive_lods <= 2


def test_partitioned_is_partition_of_laddered_leaves():
    data = _make_random_gsplat(n=400)
    res = build_recipe(data, "partitioned", _params(max_elements=120, n_lods=3))
    assert isinstance(res, GSplatPartition)
    assert res.n_children >= 2
    # every part is a leaf, each carrying an additive ladder
    for leaf in iter_leaves(res):
        assert isinstance(leaf, GSplatLeaf)
        assert leaf.n_additive_sublods >= 1
    # the partition cap is respected and all splats are conserved
    assert all(leaf.n_splats <= 120 for leaf in iter_leaves(res))
    assert total_splats(res) == 400


def test_partitioned_clamps_ladder_on_small_parts():
    # tiny parts must not produce empty equal-count LOD bins
    data = _make_random_gsplat(n=20)
    res = build_recipe(data, "partitioned", _params(max_elements=4, n_lods=8))
    for leaf in iter_leaves(res):
        assert 1 <= leaf.n_additive_sublods <= leaf.n_splats
    assert total_splats(res) == 20


def test_multiscale_base_pixel_size_stamps_child_thresholds():
    """``base_pixel_size`` pre-stamps the coarse↔fine selector thresholds on the
    children's meta (ascending [0.0, fine]); without it, no threshold is authored
    and the serializer falls back to its count-derived default. The anchor lets a
    caller push the fine branch's switch to a farther zoom (the substitutive cap's
    bigger splats otherwise make the count proxy switch too early)."""
    import math

    data = _make_random_gsplat(n=400)

    # Default (no anchor): children carry no authored min_pixel_size.
    default = build_recipe(data, "multiscale", _params(max_elements=120))
    fine_d, coarse_d = default.children
    assert "min_pixel_size" not in fine_d.meta
    assert "min_pixel_size" not in coarse_d.meta

    # With an anchor: thresholds are stamped, ascending, scaled by the anchor.
    bps = 200.0
    res = build_recipe(data, "multiscale", _params(max_elements=120, base_pixel_size=bps))
    fine, coarse = res.children  # finest→coarsest in memory
    assert coarse.meta["min_pixel_size"] == 0.0  # coarsest is always eligible
    expected = bps * math.sqrt(total_splats(fine) / total_splats(coarse))
    assert fine.meta["min_pixel_size"] == pytest.approx(expected)
    assert fine.meta["min_pixel_size"] > coarse.meta["min_pixel_size"]  # ascending
    # A larger anchor pushes the fine-branch switch even farther out.
    bigger = build_recipe(
        data, "multiscale", _params(max_elements=120, base_pixel_size=2 * bps)
    )
    assert bigger.children[0].meta["min_pixel_size"] == pytest.approx(
        2 * fine.meta["min_pixel_size"]
    )


def test_multiscale_is_unbalanced_lod_over_partition():
    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data, "multiscale", _params(max_elements=120, n_lods=3, compression_factor=4)
    )
    assert isinstance(res, GSplatLodGroup)
    assert res.n_children == 2
    # children are finest -> coarsest in memory: [fine partition, coarse leaf]
    fine, coarse = res.children
    assert isinstance(fine, GSplatPartition)
    assert isinstance(coarse, GSplatLeaf)
    # the coarse cap is strictly smaller than the fine branch (it is a reduction)
    assert coarse.n_splats < total_splats(fine)
    # the fine branch carries the full resolution
    assert total_splats(fine) == 400


# ── absorption regression: recipes == the builders they wrap ──────────────


def test_additive_recipe_matches_make_additive_lod():
    data = _make_random_gsplat(n=128)
    p = _params(n_lods=4, additive_method="self_energy")
    via_recipe = build_recipe(data, "additive", p)
    direct = make_additive_lod(
        data.flattened(), n_lods=4, method="self_energy", seed=None
    )
    assert via_recipe.n_additive_sublods == direct.n_additive_sublods
    # Compare the actual ORDERED content of each sub-LOD, not just counts: with
    # equal-count breakpoints the per-level counts are method-independent, so a
    # counts-only check would pass even if the recipe dropped the `method` /
    # `seed` forwarding. Centers reflect the ordering -> this catches that.
    for i in range(via_recipe.n_additive_sublods):
        np.testing.assert_array_equal(
            via_recipe.additive_sublod(i).centers,
            direct.additive_sublod(i).centers,
        )

    # Guard the guard: a different method must actually change the ordering,
    # otherwise the comparison above is not method-sensitive.
    other = make_additive_lod(data.flattened(), n_lods=4, method="greedy", seed=None)
    assert not np.array_equal(
        direct.additive_sublod(0).centers, other.additive_sublod(0).centers
    )


def test_substitutive_recipe_matches_make_substitutive_lod():
    data = _make_random_gsplat(n=256)
    p = _params(compression_factor=4, levels=2, substitutive_method="kmeans_lloyd")
    via_recipe = build_recipe(data, "substitutive", p)
    direct = make_substitutive_lod(
        data.flattened(),
        compression_factor=4,
        levels=2,
        method="kmeans_lloyd",
        device="cpu",
        seed=None,
    )
    assert via_recipe.n_substitutive == direct.n_substitutive
    got = [lev.n_splats_total for lev in via_recipe.substitutive_levels]
    exp = [lev.n_splats_total for lev in direct.substitutive_levels]
    assert got == exp


def test_pyramid_recipe_matches_make_lod_pyramid():
    data = _make_random_gsplat(n=256)
    p = _params(
        compression_factor=4,
        levels=2,
        n_lods=2,
        additive_method="self_energy",
        substitutive_method="kmeans_lloyd",
    )
    via_recipe = build_recipe(data, "pyramid", p)
    direct = make_lod_pyramid(
        data.flattened(),
        compression_factor=4,
        levels=2,
        substitutive_method="kmeans_lloyd",
        n_additive_lods=2,
        additive_method="self_energy",
        device="cpu",
        seed=None,
    )
    assert via_recipe.n_substitutive == direct.n_substitutive
    got = [lev.n_additive_lods for lev in via_recipe.substitutive_levels]
    exp = [lev.n_additive_lods for lev in direct.substitutive_levels]
    assert got == exp
