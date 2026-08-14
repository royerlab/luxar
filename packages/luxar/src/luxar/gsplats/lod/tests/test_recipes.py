"""Unit tests for the representation-recipe builders (``gsplats.lod.recipes``).

Pure builder tests — no CLI, no zarr IO, ``device="cpu"`` throughout. They assert
the *tree shape* each recipe produces, splat-count conservation, and that the
three absorbed primitives (``additive`` / ``substitutive`` / ``pyramid``) are
structurally identical to calling the underlying ``make_*_lod`` builders directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core.group.lod.group import (
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
)
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


def test_tiles_counts_breakpoints_clamp_to_small_parts():
    """REGRESSION (pre-existing footgun): explicit `counts:` breakpoints larger
    than a small BSP part used to abort the WHOLE tiles build with
    'largest breakpoint exceeds N'. Per-part ladders now clamp the counts to
    each part's own size instead."""
    data = _make_random_gsplat()  # 400 splats
    # max_elements=120 → parts of ≤120 splats; a 300-cut exceeds every part.
    params = _params(max_elements=120, breakpoints=[50, 300])
    tree = build_recipe(data, "tiles", params)  # must NOT raise
    assert total_splats(tree) == 400


def test_levels_counts_breakpoints_clamp_to_coarse_levels():
    """Same footgun on levels: coarser substitutive levels are smaller by
    K^s, so counts sized for the finest level used to abort the build."""
    data = _make_random_gsplat()  # 400 splats; K=4,L=2 → levels 400/100/25
    params = _params(breakpoints=[50, 300])
    res = build_recipe(data, "levels", params)  # must NOT raise
    assert isinstance(res, GSplatData)
    assert res.flattened().n_splats == 400


def test_levels_counts_breakpoints_typo_scale_raises():
    """Counts exceeding the FULL dataset (the finest level) are a typo
    and must still abort loudly — the per-level clamp applies only to the
    coarser (smaller-by-K^s) levels, never to the whole-dataset check."""
    data = _make_random_gsplat()  # 400 splats
    params = _params(breakpoints=[1_000_000])
    with pytest.raises(ValueError, match="exceeds N=400"):
        build_recipe(data, "levels", params)


def test_tiles_counts_breakpoints_typo_scale_raises():
    """The same union-level typo guard on the tiles recipe: counts must
    fit the WHOLE dataset even though every individual part clamps."""
    data = _make_random_gsplat()  # 400 splats
    params = _params(max_elements=120, breakpoints=[1_000_000])
    with pytest.raises(ValueError, match="exceeds N=400"):
        build_recipe(data, "tiles", params)


def test_stream_breakpoints_flow_through_tiles():
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
        # node tree whose leaves total >= N (overview adds a coarse cap).
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
    """tiles/overview/adaptive require >=2 dims (BSP); 4D must build and conserve."""
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


def test_levels_is_substitutive_times_additive():
    data = _make_random_gsplat(n=256)
    res = build_recipe(
        data, "levels", _params(compression_factor=4, levels=2, n_lods=2)
    )
    assert isinstance(res, GSplatData)
    assert res.n_substitutive == 3
    for lev in res.substitutive_levels:
        assert 1 <= lev.n_additive_lods <= 2


def test_tiles_is_partition_of_laddered_leaves():
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


def test_tiles_clamps_ladder_on_small_parts():
    # tiny parts must not produce empty equal-count LOD bins
    data = _make_random_gsplat(n=20)
    res = build_recipe(data, "tiles", _params(max_elements=4, n_lods=8))
    for leaf in iter_leaves(res):
        assert 1 <= leaf.n_additive_sublods <= leaf.n_splats
    assert total_splats(res) == 20


def test_overview_stamps_coverage_fractions_by_default():
    """overview always pre-stamps the coarse↔fine selector thresholds on the
    children's meta as screen-area ``coverage_fraction`` values: the coarse cap
    gets 0.0 (always-eligible floor) and the fine partition gets the finest rung.

    The finest rung is ``PARTITION_FINEST_AREA`` (1.0 — the node alone fills
    the screen), not the whole-object 0.5: this is a PARTITION-bound ladder
    (the fine child is the whole dataset as a kind=partition, reached by
    zooming in), so it keeps the fills-screen anchor rather than the
    whole-object half-screen-area one — see
    ``partitioned_coverage_fractions``."""
    data = _make_random_gsplat(n=400)

    res = build_recipe(data, "overview", _params(max_elements=120))
    coarse, fine = res.children  # coarsest→finest in memory
    assert coarse.meta["coverage_fraction"] == 0.0  # coarsest = always-eligible floor
    assert fine.meta["coverage_fraction"] == pytest.approx(PARTITION_FINEST_AREA)
    assert fine.meta["coverage_fraction"] > coarse.meta["coverage_fraction"]


def test_overview_fine_branch_keeps_the_fills_screen_anchor():
    """Regression for the anchor move: the overview recipe's contract is
    "instant coarse overview level + fine tiles on zoom".

    Under ``selector="screen-area"`` a threshold is a literal screen-area
    fraction, and the fine partition is pinned at ``PARTITION_FINEST_AREA``
    (1.0 — the node alone occupying the FULL screen). An object never covers
    the full screen at the fit (opening) framing — measured area occupancy is
    ~0.5–0.86 at most — so the coarse cap shows on frame 1 and the fine branch
    (the entire dataset, eagerly loaded) engages only on zoom-in. Anchored at
    the whole-object 0.5 instead, the fine branch WOULD be selected at the
    opening framing, inverting the recipe."""
    data = _make_random_gsplat(n=400)
    res = build_recipe(data, "overview", _params(max_elements=120))
    coarse, fine = res.children

    # Pin the stamped thresholds themselves (a 2-child group: floor + anchor).
    assert fine.meta["coverage_fraction"] == pytest.approx(PARTITION_FINEST_AREA)
    assert coarse.meta["coverage_fraction"] == 0.0
    assert coarse.meta["coverage_fraction"] < fine.meta["coverage_fraction"]

    # A generous upper bound on any normal opening framing's area occupancy —
    # strictly below the fills-screen anchor, so the coarse cap shows first.
    opening_area = 0.86
    assert opening_area < fine.meta["coverage_fraction"], (
        "the fine partition must NOT be selected at the opening framing"
    )

    # Zoomed until the node alone occupies the whole screen (area 1.0) → the
    # fine branch engages.
    assert 1.0 >= fine.meta["coverage_fraction"]


def test_overview_is_unbalanced_lod_over_partition():
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


def test_adaptive_is_partition_of_substitutive_lod_groups():
    """`adaptive` = a kind=partition whose EVERY part is its own substitutive lod
    group (per-part coarse↔fine swap) — the per-part substitutive sibling of
    `tiles` (additive parts) and `overview` (one global cap)."""
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


def test_adaptive_per_part_ladders_keep_the_fills_screen_anchor():
    """Each adaptive part is its OWN lod group whose bbox is one BSP tile.

    A tile's projected footprint is intrinsically a fraction of the whole
    object's, so the whole-object half-screen-area anchor would put every tile
    on its FINEST level while the object is merely full-frame (#1361
    follow-up). Per-part ladders therefore keep the fills-screen anchor:
    coarsest 0.0, finest PARTITION_FINEST_AREA (1.0 — the tile alone fills the
    screen), strictly ascending."""
    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data, "adaptive", _params(max_elements=120, compression_factor=4, levels=2)
    )
    # An 8-part split puts a tile's projected bbox rect at roughly 0.09 of the
    # screen AREA at whole-object framing (~0.30 of the viewport diagonal),
    # which must stay below the finest rung.
    tile_area_at_whole_object_framing = 0.09
    for part in res.children:
        assert isinstance(part, GSplatLodGroup)
        covs = [c.meta["coverage_fraction"] for c in part.children]
        assert covs[0] == 0.0  # coarsest = always-eligible floor
        assert covs[-1] == pytest.approx(PARTITION_FINEST_AREA)
        assert covs == sorted(covs) and len(set(covs)) == len(covs)  # strict ascent
        assert all(0.0 <= c <= PARTITION_FINEST_AREA for c in covs)
        assert tile_area_at_whole_object_framing < covs[-1], (
            "a tile must not sit on its finest level at whole-object framing"
        )


def test_adaptive_single_part_keeps_the_whole_object_anchor():
    """REGRESSION: a ONE-part "partition" is not a tiling, so it must not take
    the fills-screen anchor.

    ``to_spatial_partition`` wraps even a single BSP leaf in a
    ``GSplatPartition``, and the BSP stops as soon as the whole dataset fits
    ``max_elements`` — whose default is 1,000,000, so ``gsplat lod --recipe
    adaptive`` lands here for any ordinary dataset. That lone part's bbox IS the
    whole object's, so anchoring its ladder at PARTITION_FINEST_AREA would hold
    the finest level back until the object alone filled the screen: exactly the
    #1361 blur, reintroduced by the fix for it."""
    data = _make_random_gsplat(n=120)
    res = build_recipe(
        data, "adaptive", _params(max_elements=None, compression_factor=4, levels=2)
    )
    assert isinstance(res, GSplatPartition)
    assert res.n_children == 1, "expected the whole dataset to fit one part"
    (part,) = res.children
    assert isinstance(part, GSplatLodGroup)
    covs = [c.meta["coverage_fraction"] for c in part.children]
    assert covs == pytest.approx([0.0, 0.25, 0.5])
    assert covs[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR), (
        f"a one-part partition must use the whole-object anchor; got {covs}"
    )
    # #1361 opening-framing contract, in AREA terms: the occupancy-halving
    # rule's deliberate design premise is that a normal opening (fit) framing
    # occupies well over half the screen area (measured ~0.5–0.8 for cube-ish
    # objects), so a finest threshold at half-screen occupancy is already
    # reached on frame 1.
    assert covs[-1] <= 0.5


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


def test_adaptive_recipe_supports_volume_refine_per_tile():
    """Per-part levels re-fit against each tile's own CROP of the volume.

    This replaces an older test that asserted the combination was rejected. The
    rejection existed because a part re-fitted against the FULL volume gets
    pulled out of its tile to explain a neighbour's signal; cropping to the tile
    removes the cause, and a re-fit that escapes anyway is discarded in favour of
    the merge.
    """
    import numpy as np

    from luxar.core.group.partition import serialized_bsp_leaf_cells
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.tree import center_bounds, iter_leaves

    size = 24
    rng = np.random.default_rng(0)
    grid = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    vol = np.zeros((size,) * 3, np.float32)
    for _ in range(8):
        c = rng.uniform(4, size - 4, 3)
        s = rng.uniform(1.5, 2.2)
        r2 = sum((grid[d] - c[d]) ** 2 for d in range(3))
        vol += rng.uniform(0.4, 1.0) * np.exp(-r2 / (2 * s * s))
    fine = fit_gaussian_splats(vol, seeds=200, n_iters=200, device="cpu", verbose=False)

    p = _params(
        max_elements=60,
        compression_factor=2,
        levels=1,
        refine="volume",
        refine_iters=25,
        volume=vol,
        device="cpu",
        quality_stamps=False,
    )
    node = build_recipe(fine, "adaptive", p)
    assert len(node.children) > 1, "need a real tiling for this to mean anything"

    stats = [
        st
        for child in node.children
        for leaf in iter_leaves(child)
        if (st := (leaf.meta.get("stats") or {}).get("refine_stats"))
    ]
    assert stats, "no part recorded a volume re-fit — the recipe did not run one"

    # The invariant the old rejection protected: a centre must not MIGRATE out of
    # its own tile, or the viewer's per-part frustum culling would stop drawing it
    # from most viewpoints. The allowance is the level's own median splat sigma,
    # matching `_relocated`'s reasoning that correcting within a splat's own
    # footprint is optimization rather than migration — a hard bound instead
    # discarded half the tiles' re-fits over sub-voxel drift.
    from luxar.gsplats.lod.volume_refit import _median_splat_sigma

    cells = serialized_bsp_leaf_cells(node.bsp_tree, 3)
    for i, child in enumerate(node.children):
        for leaf in iter_leaves(child):
            bounds = center_bounds(leaf)
            if bounds is None:
                continue
            slack = max(1e-3, _median_splat_sigma(GSplatData.from_tree(leaf)))
            lo, hi = bounds
            for d in range(3):
                assert lo[d] >= cells[i][d][0] - slack, (
                    f"part {i} dim {d}: a centre at {lo[d]} is more than one "
                    f"sigma ({slack:.3f}) below its cell {cells[i][d]} — the "
                    "re-fit migrated out of the tile"
                )
                assert hi[d] <= cells[i][d][1] + slack, (
                    f"part {i} dim {d}: a centre at {hi[d]} is more than one "
                    f"sigma ({slack:.3f}) above its cell {cells[i][d]} — the "
                    "re-fit migrated out of the tile"
                )


def test_adaptive_volume_refine_needs_a_cell():
    """Called without a tile to crop to, the per-part re-fit refuses rather than
    silently targeting the whole volume — the unsound case the old guard covered
    and the one thing that must still raise."""
    import numpy as np

    from luxar.gsplats.lod.recipes import _substitutive_for_part

    data = _make_random_gsplat(n=256)
    partition = data.flattened().to_spatial_partition(max_elements=64)
    p = _params(refine="volume", volume=np.zeros((8, 8, 8), np.float32))
    with pytest.raises(ValueError, match="needs the part's own cell"):
        _substitutive_for_part(partition.children[0], p, cell=None)


def test_additive_ladders_default_on_everywhere():
    """Project convention: additive LODs by default. The substitutive recipe
    ladders every level; adaptive ladders every per-part level; the overview
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
    # adaptive: per-part lod-group children are multi-sublod leaves
    mos = build_recipe(data, "adaptive", _params(levels=1, n_lods=2, max_elements=150))
    laddered_leaves = [
        leaf
        for part in mos.children
        for leaf in iter_leaves(part)
        if len(leaf.additive_sublods) > 1
    ]
    assert laddered_leaves, "adaptive parts carry no additive ladders"
    # overview: the coarse cap (first child) is a laddered leaf
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


# ── Q·e quality stamps through the pipeline layer ───────────────────────────


def test_quality_stamps_default_on_and_flow_through_levels():
    """RecipeParams defaults quality stamps ON (the primitive defaults OFF);
    the levels recipe stamps Q + the group-consistent reference energy w on
    every level's leaf meta."""
    assert RecipeParams().quality_stamps is True
    assert RecipeParams().quality_max_pair_splats == 2_000_000

    data = _make_random_gsplat(n=400)
    res = build_recipe(data, "levels", _params(levels=1, compression_factor=8))
    children = res.tree.children  # coarsest→finest in memory
    stats = [child.meta["stats"] for child in children]
    assert all("quality" in s for s in stats)
    assert stats[-1]["quality"] == 1.0  # the finest IS the reference
    assert 0.0 <= stats[0]["quality"] <= 1.0
    # w is the FINEST content's total self-energy, constant across the group
    # (self-energy is quadratic in amplitude — per-level totals differ, so a
    # per-level w would skew weighted aggregation).
    ws = [s["reference_energy"] for s in stats]
    assert ws[0] == pytest.approx(ws[-1])
    assert ws[0] > 0

    # The off switch removes the measurement entirely.
    res_off = build_recipe(
        data, "levels", _params(levels=1, compression_factor=8, quality_stamps=False)
    )
    for child in res_off.tree.children:
        assert "quality" not in child.meta["stats"]


def test_quality_stamps_overview_cap_and_fine_parts():
    """overview: the coarse cap carries its measured Q and the finest
    content's energy as w (surviving the `.flattened()` leaf rebuild); the
    fine parts are the finest content — quality 1.0, w = each part's own
    total (disjoint parts sum to the cap's w)."""
    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data, "overview", _params(max_elements=120, compression_factor=4)
    )
    coarse, fine = res.children
    cap_stats = coarse.meta["stats"]
    assert 0.0 <= cap_stats["quality"] <= 1.0
    assert cap_stats["reference_energy"] > 0
    part_ws = []
    for leaf in iter_leaves(fine):
        assert leaf.meta["stats"]["quality"] == 1.0
        part_ws.append(leaf.meta["stats"]["reference_energy"])
    assert all(w > 0 for w in part_ws)
    # Disjoint parts ⇒ self-energy is additive: Σ w_p == the cap's reference w.
    assert cap_stats["reference_energy"] == pytest.approx(sum(part_ws), rel=1e-3)


def test_quality_stamps_adaptive_parts():
    """adaptive: every part's lod group is quality-stamped per level, with a
    group-consistent w inside each part."""
    data = _make_random_gsplat(n=400)
    res = build_recipe(
        data,
        "adaptive",
        _params(max_elements=120, levels=1, compression_factor=8),
    )
    for part in res.children:
        stats = [child.meta["stats"] for child in part.children]  # coarse→fine
        assert all("quality" in s and "reference_energy" in s for s in stats)
        assert stats[-1]["quality"] == 1.0
        assert stats[0]["reference_energy"] == pytest.approx(
            stats[-1]["reference_energy"]
        )


def test_overview_cap_keeps_no_energy_weight_under_a_reveal_ordering():
    """The `overview` cap must not get its `reference_energy` re-attached.

    `_ladder_for_part` builds the cap's ladder and (for a reveal) leaves it with
    no per-rung `energy_fraction_cum` and no leaf weight. The quality block then
    re-attaches the capped reduction's `quality` AND `reference_energy` — which
    put the half-written pair straight back. `quality` is a standalone readout and
    still belongs there; the weight does not.
    """
    data = _make_random_gsplat(n=400)

    revealed = build_recipe(
        data, "overview", _params(max_elements=120, additive_method="radial")
    )
    coarse, _ = revealed.children
    cap_stats = coarse.meta.get("stats", {})
    assert "quality" in cap_stats, "the cap must keep its measured Q"
    assert "reference_energy" not in cap_stats

    # Control: the same build under an energy ordering keeps both, so this is
    # scoped to reveals rather than a blanket removal.
    ordered = build_recipe(
        data, "overview", _params(max_elements=120, additive_method="self_energy")
    )
    ordered_cap_stats = ordered.children[0].meta.get("stats", {})
    assert "quality" in ordered_cap_stats
    assert "reference_energy" in ordered_cap_stats


def test_every_volume_forwarding_site_also_forwards_the_axis_map():
    """Structural guard against a fix-N-minus-1.

    ``RecipeParams`` carries the source volume AND the map saying which volume
    axis holds which center dim. Four different recipe paths reach
    ``make_substitutive_lod``, and one of them originally forwarded ``volume``
    without ``volume_axes``: a stacked target then fell back to the identity map,
    every re-fit tripped the frame guard, and `--recipe levels --refine volume`
    reported success while refining nothing at all.

    Counting the two forwardings in the source is crude but catches exactly the
    regression that happened — a new path that copies the ``volume=`` line and
    forgets its partner.
    """
    from pathlib import Path

    import luxar.gsplats.lod.recipes as recipes_module

    source = Path(recipes_module.__file__).read_text()
    volume_sites = source.count("volume=params.volume,")
    axes_sites = source.count("volume_axes=params.volume_axes,")
    assert volume_sites > 0, "sanity: the forwarding pattern moved"
    assert axes_sites == volume_sites, (
        f"{volume_sites} sites forward `volume` but {axes_sites} forward "
        "`volume_axes`; a stacked target would silently frame-mismatch on the "
        "path that drops it"
    )


def test_levels_recipe_refines_a_stacked_target_through_the_ladder():
    """End-to-end at the recipe layer, on the path the CLI actually takes.

    `levels` builds through ``make_lod_pyramid`` (ladders are on by default), a
    different route to the reduction than the per-part ``adaptive`` path, so it
    needs its own coverage: the unit tests on ``make_substitutive_lod`` passed
    throughout while this route was broken.
    """
    import numpy as np

    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.utils.trils import embed_cholesky_packed

    size, n_t = 20, 3
    rng = np.random.default_rng(0)
    grid = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    field = np.zeros((size,) * 3, np.float32)
    for _ in range(5):
        c = rng.uniform(5, 15, 3)
        s = rng.uniform(1.6, 2.2)
        field += rng.uniform(0.4, 1.0) * np.exp(
            -sum((grid[d] - c[d]) ** 2 for d in range(3)) / (2 * s * s)
        )
    # Source array is (t, z, y, x) — time FIRST, as microscopy data comes.
    vol = np.stack([(1.0 + 0.3 * t) * field for t in range(n_t)]).astype(np.float32)

    fit = fit_gaussian_splats(
        vol[0], seeds=80, n_iters=150, device="cpu", verbose=False
    )
    n = fit.n_splats
    packed = embed_cholesky_packed(
        np.asarray(fit.cholesky_factors), 3, 4, [0, 1, 2], fill_sigma={3: 1e-4}
    )
    data = GSplatData(
        centers=np.concatenate(
            [
                np.column_stack([np.asarray(fit.centers), np.full(n, float(t))])
                for t in range(n_t)
            ]
        ).astype(np.float32),
        amplitudes=np.tile(np.asarray(fit.amplitudes), n_t).astype(np.float32),
        cholesky_factors=np.tile(packed, (n_t, 1)).astype(np.float32),
    )

    p = _params(
        compression_factor=4,
        levels=1,
        refine="volume",
        refine_iters=25,
        volume=vol,
        volume_axes=(1, 2, 3, 0),  # splats are (z,y,x,t); volume is (t,z,y,x)
        coarsen_dims=(0, 1, 2),
        device="cpu",
    )
    out = build_recipe(data, "levels", p)
    stats = out.substitutive_levels[1].stats["refine_stats"]
    assert stats["frame_mismatch_frac"] == 0.0, (
        "every group tripped the frame guard — the axis map did not reach the "
        f"reduction through this path (stats: {stats})"
    )
    assert stats["improved_frac"] > 0.0, "no group's re-fit was kept"
    assert stats["mse_stored"] <= stats["mse_seed"] + 1e-12


def test_tile_box_follows_the_subvolume_dim_order_not_the_spelling() -> None:
    """A tile's box is indexed by the RETAINED dims, which come back ascending.

    ``select_sub_volume`` drops the barrier dims and hands back the survivors in
    ascending order, and ``make_substitutive_lod`` normalises ``coarsen_dims`` the
    same way. So the box has to be built in that order too — built in the order
    the caller happened to spell them (``2,1,0``, or with a repeat), each axis
    would be cropped to another axis's bounds. Fails pre-fix on the reversed
    spelling.
    """
    from luxar.gsplats.lod.recipes import _cell_for_coarsened_dims

    cell = [(0.0, 1.0), (10.0, 11.0), (20.0, 21.0), (float("-inf"), float("inf"))]
    ascending = [(0.0, 1.0), (10.0, 11.0), (20.0, 21.0)]

    for spelling in ((0, 1, 2), (2, 1, 0), (1, 0, 2), (2, 0, 1, 0)):
        p = _params(refine="volume", coarsen_dims=spelling)
        assert _cell_for_coarsened_dims(cell, p, 4) == ascending, spelling

    # No barrier dims → every dim retained, still ascending.
    p = _params(refine="volume", coarsen_dims=None)
    assert _cell_for_coarsened_dims(cell, p, 4) == cell
    # Not a volume re-fit → the box would be dead weight.
    assert _cell_for_coarsened_dims(cell, _params(refine="l2"), 4) is None
    assert _cell_for_coarsened_dims(None, p, 4) is None
