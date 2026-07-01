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
    uniform_per_part_lod_warning,
)
from luxar.gsplats.lod.substitutive import make_substitutive_lod
from luxar.gsplats.tree import (
    GSplatLeaf,
    GSplatLodGroup,
    GSplatPartition,
    iter_leaves,
    total_splats,
)


@pytest.mark.parametrize(
    "tiling_mode,recipe,expect_warning",
    [
        # BOTH per-part recipes break partition-of-unity at coarse levels on
        # apodized uniform tiles (additive drops tapered halo splats → dims;
        # substitutive merges them per-part → smears).
        ("uniform", "substitutive", True),
        ("uniform", "additive", True),
        # Content (disjoint core-keep parts) carries no shared halos → exact.
        ("content", "substitutive", False),
        ("content", "additive", False),
        ("none", "substitutive", False),  # whole-volume, no overlap
        (None, "substitutive", False),
        ("uniform", None, False),  # no recipe at all
        ("uniform", "partitioned", False),  # not a per-part recipe
    ],
)
def test_uniform_per_part_lod_warning(tiling_mode, recipe, expect_warning):
    msg = uniform_per_part_lod_warning(tiling_mode, recipe)
    if expect_warning:
        assert msg is not None
        assert recipe in msg and "uniform" in msg
    else:
        assert msg is None


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


def test_recipe_params_additive_method_defaults_to_auto():
    """The additive ordering defaults to the size-adaptive ``auto`` so large
    per-part ladders avoid greedy's O(N·nnz·logN) hang (the friction fix)."""
    assert RecipeParams().additive_method == "auto"


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
    mo = build_recipe(data, "mosaic", _params(max_elements=120))
    assert isinstance(mo, GSplatPartition)
    assert all(leaf.ndim == 4 for leaf in iter_leaves(mo))


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


def test_extent_thresholds_are_scale_invariant():
    """The ``extent`` method's defining property: threshold = T·W/r is dimensionless
    in scale, so uniformly scaling a scene (centers AND covariance × k) leaves the
    switch thresholds unchanged — no per-dataset tuning (unlike the √N ``count``
    method, which is also scale-free, but here we lock the self-calibration that
    motivated the change). A mutation to a non-W/r formula would break this."""
    data = _make_random_gsplat(n=400)
    k = 7.0
    scaled = GSplatData(
        centers=data.centers * k,
        amplitudes=data.amplitudes.copy(),
        cholesky_factors=data.cholesky_factors * k,  # L×k → Σ×k² → semi-axis × k
    )
    p = _params(max_elements=120, compression_factor=4, n_lods=3)
    # children are coarsest→finest; the fine branch (last) carries the >0 threshold.
    base_mps = build_recipe(data, "multiscale", p).children[-1].meta["min_pixel_size"]
    scaled_mps = (
        build_recipe(scaled, "multiscale", p).children[-1].meta["min_pixel_size"]
    )
    assert base_mps > 0.0
    assert scaled_mps == pytest.approx(base_mps, rel=1e-4)


def test_multiscale_stamps_extent_thresholds_by_default():
    """multiscale always pre-stamps the coarse↔fine selector thresholds on the
    children's meta (ascending [0.0, fine]). The default ``extent`` method anchors
    the switch in physical element size (T·W/r), so the anchor ``base_pixel_size``
    (target px) scales it linearly; ``lod_method="count"`` falls back to √N."""
    import math

    data = _make_random_gsplat(n=400)

    # Default = extent method: thresholds stamped, coarsest = 0.0, ascending.
    res = build_recipe(data, "multiscale", _params(max_elements=120))
    coarse, fine = res.children  # coarsest→finest in memory
    assert coarse.meta["min_pixel_size"] == 0.0  # coarsest = always-eligible floor
    fine_mps = fine.meta["min_pixel_size"]
    assert fine_mps > 0.0  # ascending → coarse cap reachable at far zoom

    # base_pixel_size is the target-px anchor T in extent mode: 2× → 2× threshold.
    bigger = build_recipe(
        data, "multiscale", _params(max_elements=120, base_pixel_size=2 * 1.5)
    )
    # default T is DEFAULT_TARGET_PIXEL_SIZE (1.5); 3.0 is 2×.
    assert bigger.children[-1].meta["min_pixel_size"] == pytest.approx(2 * fine_mps)

    # lod_method="count" reproduces the legacy √N proxy exactly.
    cnt = build_recipe(
        data, "multiscale", _params(max_elements=120, lod_method="count")
    )
    ccoarse, cfine = cnt.children
    assert ccoarse.meta["min_pixel_size"] == 0.0
    expected_count = 10.0 * math.sqrt(total_splats(cfine) / total_splats(ccoarse))
    assert cfine.meta["min_pixel_size"] == pytest.approx(expected_count)
    # extent and count generally land the switch at different thresholds.
    assert cfine.meta["min_pixel_size"] != pytest.approx(fine_mps)

    # extent knobs change the radius → change the threshold.
    p50 = build_recipe(
        data, "multiscale", _params(max_elements=120, extent_percentile=50.0)
    )
    assert p50.children[-1].meta["min_pixel_size"] != pytest.approx(fine_mps)


def test_multiscale_is_unbalanced_lod_over_partition():
    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data, "multiscale", _params(max_elements=120, n_lods=3, compression_factor=4)
    )
    assert isinstance(res, GSplatLodGroup)
    assert res.n_children == 2
    # children are coarsest -> finest in memory: [coarse leaf, fine partition]
    coarse, fine = res.children
    assert isinstance(fine, GSplatPartition)
    assert isinstance(coarse, GSplatLeaf)
    # the coarse cap is strictly smaller than the fine branch (it is a reduction)
    assert coarse.n_splats < total_splats(fine)
    # the fine branch carries the full resolution
    assert total_splats(fine) == 400


def test_mosaic_is_partition_of_substitutive_lod_groups():
    """`mosaic` = a kind=partition whose EVERY part is its own substitutive lod
    group (per-part coarse↔fine swap) — the per-part substitutive sibling of
    `partitioned` (additive parts) and `multiscale` (one global cap)."""
    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data, "mosaic", _params(max_elements=120, compression_factor=4, levels=2)
    )
    assert isinstance(res, GSplatPartition)
    assert res.n_children >= 2
    # every part is a substitutive lod group with >= 2 levels (not a bare leaf).
    for part in res.children:
        assert isinstance(part, GSplatLodGroup)
        assert part.n_children >= 2  # >= coarse + fine
        # in-memory children are coarsest→finest; counts ascend that way.
        levels_coarse_to_fine = [total_splats(c) for c in part.children]
        assert levels_coarse_to_fine == sorted(levels_coarse_to_fine)
    # Conservation is at the FINEST level (the parts tile the original N); the
    # synthesized coarser substitutive levels are extra stored representatives.
    finest_total = sum(total_splats(part.children[-1]) for part in res.children)
    assert finest_total == 400
    assert total_splats(res) > 400  # synthesized coarse levels add storage
    # leaves are all real gsplat leaves at every level
    assert all(isinstance(leaf, GSplatLeaf) for leaf in iter_leaves(res))


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


# ── coarsen_dims threads through RecipeParams ─────────────────────────────


def _stacked_4d_gsplat(n_per=600, n_groups=3, seed=0) -> GSplatData:
    """4D splats: dim 0 categorical (0..G-1), dims 1-3 xyz shared across groups."""
    from luxar.gsplats.lift import lift_points_to_gsplats

    rng = np.random.default_rng(seed)
    xyz = rng.normal(0, 5, (n_per, 3)).astype(np.float32)
    parts = [
        np.column_stack([np.full(n_per, g, np.float32), xyz]) for g in range(n_groups)
    ]
    pos = np.vstack(parts).astype(np.float32)
    return lift_points_to_gsplats(pos, np.full(len(pos), 0.5, np.float32))


class TestRecipeParamsCoarsenDims:
    def test_default_none(self) -> None:
        assert RecipeParams().coarsen_dims is None

    def test_build_substitutive_respects_barrier(self) -> None:
        from luxar.gsplats.lod.recipes import build_substitutive

        data = _stacked_4d_gsplat()
        params = RecipeParams(
            compression_factor=4, levels=3, device="cpu", coarsen_dims=(1, 2, 3)
        )
        out = build_substitutive(data, params)
        for s in range(out.n_substitutive):
            c0 = np.asarray(out.at_substitutive(s).flattened().centers)[:, 0]
            assert np.abs(c0 - np.round(c0)).max() < 1e-4
