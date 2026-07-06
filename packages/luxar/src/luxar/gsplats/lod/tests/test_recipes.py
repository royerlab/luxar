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
        ("uniform", "levels", True),
        ("uniform", "stream", True),
        # Content (disjoint core-keep parts) carries no shared halos → exact.
        ("content", "levels", False),
        ("content", "stream", False),
        ("none", "levels", False),  # whole-volume, no overlap
        (None, "levels", False),
        ("uniform", None, False),  # no recipe at all
        ("uniform", "tiles", False),  # not a per-part recipe
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


def test_partitioned_counts_breakpoints_clamp_to_small_parts():
    """REGRESSION (pre-existing footgun): explicit `counts:` breakpoints larger
    than a small BSP part used to abort the WHOLE partitioned build with
    'largest breakpoint exceeds N'. Per-part ladders now clamp the counts to
    each part's own size instead."""
    data = _make_random_gsplat()  # 400 splats
    # max_elements=120 → parts of ≤120 splats; a 300-cut exceeds every part.
    params = _params(max_elements=120, breakpoints=[50, 300])
    tree = build_recipe(data, "tiles", params)  # must NOT raise
    assert total_splats(tree) == 400


def test_pyramid_counts_breakpoints_clamp_to_coarse_levels():
    """Same footgun on the pyramid: coarser substitutive levels are smaller by
    K^s, so counts sized for the finest level used to abort the build."""
    data = _make_random_gsplat()  # 400 splats; K=4,L=2 → levels 400/100/25
    params = _params(breakpoints=[50, 300])
    res = build_recipe(data, "levels", params)  # must NOT raise
    assert isinstance(res, GSplatData)
    assert res.flattened().n_splats == 400


def test_pyramid_counts_breakpoints_typo_scale_raises():
    """Counts exceeding the FULL dataset (the finest pyramid level) are a typo
    and must still abort loudly — the per-level clamp applies only to the
    coarser (smaller-by-K^s) levels, never to the whole-dataset check."""
    data = _make_random_gsplat()  # 400 splats
    params = _params(breakpoints=[1_000_000])
    with pytest.raises(ValueError, match="exceeds N=400"):
        build_recipe(data, "levels", params)


def test_partitioned_counts_breakpoints_typo_scale_raises():
    """The same union-level typo guard on the partitioned recipe: counts must
    fit the WHOLE dataset even though every individual part clamps."""
    data = _make_random_gsplat()  # 400 splats
    params = _params(max_elements=120, breakpoints=[1_000_000])
    with pytest.raises(ValueError, match="exceeds N=400"):
        build_recipe(data, "tiles", params)


def test_stream_breakpoints_flow_through_partitioned():
    """A `stream:<c>` spec sizes each part's ladder against ITS OWN N."""
    data = _make_random_gsplat()  # 400 splats
    params = _params(max_elements=120, breakpoints="stream:30")
    tree = build_recipe(data, "tiles", params)
    assert total_splats(tree) == 400
    from luxar.gsplats.tree import iter_leaves

    for leaf in iter_leaves(tree):
        incs = [s.n_splats for s in leaf.additive_sublods]
        assert incs[0] <= 30 or len(incs) == 1  # first chunk ≤ c (or single)
        assert sum(incs) == leaf.n_splats
        assert leaf.additive_sublods[0].stats["lod_breakpoints_kind"] == "stream"


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
    """flat/stream/levels work at any dimensionality (boundary: ndim != 3)
    and conserve the splat count."""
    data = _make_random_gsplat(n=120, ndim=ndim)
    for recipe in ("flat", "stream", "levels"):
        res = build_recipe(data, recipe, _params())
        assert isinstance(res, GSplatData), recipe
        assert res.ndim == ndim, recipe
        assert res.flattened().n_splats == 120, recipe


def test_partition_recipes_support_4d():
    """partitioned/multiscale require >=3 dims (BSP); 4D must build and conserve."""
    data = _make_random_gsplat(n=400, ndim=4)
    part = build_recipe(data, "tiles", _params(max_elements=120))
    assert isinstance(part, GSplatPartition)
    assert total_splats(part) == 400
    assert all(leaf.ndim == 4 for leaf in iter_leaves(part))
    ms = build_recipe(data, "overview", _params(max_elements=120))
    assert isinstance(ms, GSplatLodGroup)
    assert all(leaf.ndim == 4 for leaf in iter_leaves(ms))
    mo = build_recipe(data, "adaptive", _params(max_elements=120))
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
    res = build_recipe(data, "stream", _params(n_lods=4, additive_method="self_energy"))
    assert isinstance(res, GSplatData)
    assert res.n_substitutive == 1
    assert res.n_additive_sublods == 4
    # prefix counts grow monotonically to the full set
    counts = [res.additive_prefix(k).n_splats for k in range(res.n_additive_sublods)]
    assert counts == sorted(counts)
    assert counts[-1] == 200


def test_substitutive_is_lod_group_matrix():
    data = _make_random_gsplat(n=256)
    res = build_recipe(data, "levels", _params(compression_factor=4, levels=2))
    assert isinstance(res, GSplatData)
    assert res.n_substitutive == 3  # levels + 1
    assert res.tree.__class__ is GSplatLodGroup


def test_pyramid_is_substitutive_times_additive():
    data = _make_random_gsplat(n=256)
    res = build_recipe(
        data, "levels", _params(compression_factor=4, levels=2, n_lods=2)
    )
    assert isinstance(res, GSplatData)
    assert res.n_substitutive == 3
    for lev in res.substitutive_levels:
        assert 1 <= lev.n_additive_lods <= 2


def test_partitioned_is_partition_of_laddered_leaves():
    data = _make_random_gsplat(n=400)
    res = build_recipe(data, "tiles", _params(max_elements=120, n_lods=3))
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
    res = build_recipe(data, "tiles", _params(max_elements=4, n_lods=8))
    for leaf in iter_leaves(res):
        assert 1 <= leaf.n_additive_sublods <= leaf.n_splats
    assert total_splats(res) == 20


def test_multiscale_stamps_coverage_fractions_by_default():
    """multiscale always pre-stamps the coarse↔fine selector thresholds on the
    children's meta as viewport-relative ``coverage_fraction`` values
    (``sqrt(N_i/N_finest)``): the coarse cap gets 0.0 (always-eligible floor) and
    the fine partition gets 1.0 (fills-screen)."""
    data = _make_random_gsplat(n=400)

    res = build_recipe(data, "overview", _params(max_elements=120))
    coarse, fine = res.children  # coarsest→finest in memory
    assert coarse.meta["coverage_fraction"] == 0.0  # coarsest = always-eligible floor
    assert fine.meta["coverage_fraction"] == pytest.approx(1.0)  # fills-screen
    assert fine.meta["coverage_fraction"] > coarse.meta["coverage_fraction"]


def test_multiscale_is_unbalanced_lod_over_partition():
    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data, "overview", _params(max_elements=120, n_lods=3, compression_factor=4)
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
        data, "adaptive", _params(max_elements=120, compression_factor=4, levels=2)
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
    via_recipe = build_recipe(data, "stream", p)
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
    via_recipe = build_recipe(data, "levels", p)
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


def test_substitutive_recipe_threads_refine():
    """RecipeParams.refine/refine_iters reach make_substitutive_lod (stats
    carry the refine block). refine_iters defaults to None (the sentinel that
    resolves to each engine's own default — 120 for l2)."""
    assert RecipeParams().refine == "none"
    assert RecipeParams().refine_iters is None
    data = _make_random_gsplat(n=256)
    # Omitted refine_iters resolves to the l2 engine default of 120.
    default_out = build_recipe(
        data,
        "levels",
        _params(
            compression_factor=8, levels=1, substitutive_method="kmeans", refine="l2"
        ),
    )
    assert default_out.stats["refine_iters"] == 120
    p = _params(
        compression_factor=8,
        levels=1,
        substitutive_method="kmeans",
        refine="l2",
        refine_iters=6,
    )
    out = build_recipe(data, "levels", p)
    assert out.stats["refine"] == "l2"
    assert out.stats["refine_iters"] == 6
    assert out.substitutive_levels[1].stats.get("refine") == "l2"


def test_substitutive_recipe_threads_volume_refine():
    """RecipeParams.volume reaches make_substitutive_lod for refine="volume"
    (level stats carry the volume-refit block); the field defaults to None and
    is excluded from equality (an ndarray payload must not break eq)."""
    import numpy as np

    assert RecipeParams().volume is None
    assert RecipeParams() == RecipeParams(volume=np.zeros((2, 2, 2), np.float32))

    rng = np.random.default_rng(0)
    grid = np.mgrid[0:16, 0:16, 0:16].astype(np.float32)
    vol = np.zeros((16,) * 3, dtype=np.float32)
    for _ in range(3):
        c = rng.uniform(4, 12, 3)
        r2 = sum((grid[d] - c[d]) ** 2 for d in range(3))
        vol += np.exp(-r2 / (2 * 2.0**2)).astype(np.float32)
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    fine = fit_gaussian_splats(vol, seeds=60, n_iters=150, device="cpu", verbose=False)
    p = _params(
        compression_factor=4,
        levels=1,
        refine="volume",
        refine_iters=20,
        volume=vol,
    )
    out = build_recipe(fine, "levels", p)
    assert out.stats["refine"] == "volume"
    lev = out.substitutive_levels[1]
    assert lev.stats.get("refine") == "volume"
    assert "mse_seed" in lev.stats["refine_stats"]


def test_adaptive_recipe_rejects_volume_refine():
    """Per-part levels must not re-fit against the FULL volume (splats would
    leave their tile); the adaptive recipe rejects refine="volume" loudly."""
    import numpy as np

    data = _make_random_gsplat(n=256)
    p = _params(refine="volume", volume=np.zeros((8, 8, 8), np.float32))
    with pytest.raises(ValueError, match="per-part levels"):
        build_recipe(data, "adaptive", p)


def test_additive_ladders_default_on_everywhere():
    """Project convention: additive LODs by default. The substitutive recipe
    ladders every level; mosaic ladders every per-part level; the multiscale
    coarse cap is laddered; --no-additive (additive_ladders=False) restores
    bare leaves."""
    data = _make_random_gsplat(n=400)
    # substitutive: default == pyramid behavior (multi-sublod levels)
    on = build_recipe(data, "levels", _params(levels=1, n_lods=3))
    assert all(lev.n_additive_lods == 3 for lev in on.substitutive_levels)
    off = build_recipe(
        data, "levels", _params(levels=1, n_lods=3, additive_ladders=False)
    )
    assert all(lev.n_additive_lods == 1 for lev in off.substitutive_levels)
    # mosaic: per-part lod-group children are multi-sublod leaves
    mos = build_recipe(data, "adaptive", _params(levels=1, n_lods=2, max_elements=150))
    laddered_leaves = [
        leaf
        for part in mos.children
        for leaf in iter_leaves(part)
        if len(leaf.additive_sublods) > 1
    ]
    assert laddered_leaves, "mosaic parts carry no additive ladders"
    # multiscale: the coarse cap (first child) is a laddered leaf
    ms = build_recipe(data, "overview", _params(levels=1, n_lods=2, max_elements=150))
    cap = ms.children[0]
    assert isinstance(cap, GSplatLeaf) and len(cap.additive_sublods) > 1


def test_pyramid_recipe_matches_make_lod_pyramid():
    data = _make_random_gsplat(n=256)
    p = _params(
        compression_factor=4,
        levels=2,
        n_lods=2,
        additive_method="self_energy",
        substitutive_method="kmeans_lloyd",
    )
    via_recipe = build_recipe(data, "levels", p)
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
        from luxar.gsplats.lod.recipes import build_levels

        data = _stacked_4d_gsplat()
        params = RecipeParams(
            compression_factor=4, levels=3, device="cpu", coarsen_dims=(1, 2, 3)
        )
        out = build_levels(data, params)
        for s in range(out.n_substitutive):
            c0 = np.asarray(out.at_substitutive(s).flattened().centers)[:, 0]
            assert np.abs(c0 - np.round(c0)).max() < 1e-4


# ── sibling-aware stream ladders across the recipes ─────────────────────────


def test_tiles_stream_parts_keep_user_base():
    """Plain tiles have no coarser lod-group sibling — parts keep the user's
    stream base (no sibling-aware raise)."""
    data = _make_random_gsplat(n=400)
    tree = build_recipe(
        data, "tiles", _params(max_elements=120, breakpoints="stream:30")
    )
    for leaf in iter_leaves(tree):
        assert leaf.additive_sublods[0].stats["lod_stream_chunk_splats"] == 30


def test_overview_fine_parts_are_sibling_aware():
    """overview: the fine partition sits UNDER the coarse cap, so each part's
    stream base is raised to ceil(part_n / 2K); the cap itself is the group's
    coarsest and keeps the user base."""
    import math

    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data,
        "overview",
        _params(max_elements=120, compression_factor=4, breakpoints="stream:4"),
    )
    coarse, fine = res.children  # coarsest→finest in memory
    assert coarse.additive_sublods[0].stats["lod_stream_chunk_splats"] == 4
    for leaf in iter_leaves(fine):
        expected = max(4, math.ceil(leaf.n_splats / 8.0))
        assert leaf.additive_sublods[0].stats["lod_stream_chunk_splats"] == expected


def test_adaptive_per_part_levels_are_sibling_aware():
    """adaptive: within each part's lod group, every level finer than the
    part's coarsest gets a raised stream base; the coarsest keeps the user
    base."""
    import math

    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data,
        "adaptive",
        _params(
            max_elements=120,
            compression_factor=4,
            levels=2,
            breakpoints="stream:4",
        ),
    )
    for part in res.children:
        for i, child in enumerate(part.children):  # coarsest→finest in memory
            chunk = child.additive_sublods[0].stats["lod_stream_chunk_splats"]
            if i == 0:  # the part's coarsest level
                assert chunk == 4
            else:
                assert chunk == max(4, math.ceil(child.n_splats / 8.0))
