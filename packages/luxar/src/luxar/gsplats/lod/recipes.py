"""Representation **recipes** — assemble a fitted gsplat set into a topology.

Intent-first vocabulary (every recipe carries streaming additive ladders by
default — see ``RecipeParams.additive_ladders``), ordered by dataset scale:

* ``flat`` — a single bare leaf. No LOD, no tiles. Tiny data / debugging.
* ``stream`` — one leaf whose splats are ordered into a progressive
  (prefix-sum, "additive") ladder: any prefix is the best preview, so the
  viewer paints fast and refines. Small-to-medium single-load data.
* ``levels`` — classic coarse→fine level-of-detail: each coarser level has
  ~``N/K^ℓ`` merged ("substitutive") representative splats that REPLACE the
  finer level, each level itself stream-laddered. Zooming across scales.
* ``tiles`` — a spatial BSP ``kind=partition``: off-screen tiles are culled,
  each visible tile streams its own ladder. Large scenes, single scale.
* ``overview`` — one cheap coarse level for the instant far view + a
  ``tiles`` fine branch for close-up (unbalanced by design: detail only
  where you look). Huge scenes with a "see everything first" need.
* ``adaptive`` — ``tiles`` where EVERY tile carries its own ``levels``
  group: each tile culls AND picks its own detail level by its on-screen
  size. The most locally adaptive; the largest scenes.

These functions are **pure** (``GSplatData`` in, a result out) with no Typer/IO —
the CLI wrapper lives in :mod:`luxar.cli.lod`. They only *compose* the existing
math builders (:func:`make_additive_lod`, :func:`make_substitutive_lod`,
:func:`make_lod_pyramid`, :meth:`GSplatData.to_spatial_partition`); the
mechanism vocabulary ("additive" prefix ladders, "substitutive" merged levels)
lives at that layer, while recipes name user intent.

Renamed (old → new): ``additive``→``stream``, ``substitutive`` and
``pyramid``→``levels``, ``partitioned``→``tiles``, ``multiscale``→``overview``,
``mosaic``→``adaptive``. :data:`LEGACY_RECIPE_NAMES` maps old spellings; the
CLI rejects them with a pointer, stored batch manifests translate silently.

Two return shapes (see :data:`RecipeResult`):

* the **matrix** recipes (``flat``/``stream``/``levels``) return a
  :class:`~luxar.gsplats.gsplat_data.GSplatData`, written via
  :meth:`GSplatData.save`.
* the **composed** recipes (``tiles``/``overview``/``adaptive``) return a
  :class:`~luxar.gsplats.tree.GSplatNode` tree, written via
  :func:`~luxar.gsplats.io.save_gsplats.write_gsplats_tree`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Union, get_args

from luxar.core.group.partition import DEFAULT_MAX_ELEMENTS
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.additive import (
    AutoOrMethod,
    BreakpointSpec,
    clamp_counts_breakpoints,
    make_additive_lod,
    validate_counts_breakpoints,
)
from luxar.gsplats.lod.pyramid import make_lod_pyramid
from luxar.gsplats.lod.substitutive import make_substitutive_lod
from luxar.gsplats.tree import (
    GSplatLodGroup,
    GSplatNode,
    GSplatPartition,
)

#: The recipe vocabulary, ordered by dataset scale.
RecipeName = Literal[
    "flat",
    "stream",
    "levels",
    "tiles",
    "overview",
    "adaptive",
]

#: Tuple form of :data:`RecipeName` for CLI choices / validation.
RECIPE_NAMES: tuple[str, ...] = get_args(RecipeName)

#: Old → new recipe spellings (renamed 2026-07-03). The CLI rejects old names
#: with a did-you-mean pointer; stored batch manifests translate silently via
#: :func:`canonical_recipe_name`.
LEGACY_RECIPE_NAMES: dict[str, str] = {
    "additive": "stream",
    "substitutive": "levels",
    "pyramid": "levels",
    "partitioned": "tiles",
    "multiscale": "overview",
    "mosaic": "adaptive",
}


def canonical_recipe_name(name: str) -> str:
    """Translate a legacy recipe spelling to the current one (identity for
    current names; unknown names pass through for the caller to reject)."""
    return LEGACY_RECIPE_NAMES.get(name, name)


#: Recipes whose result is a flat :class:`GSplatData` (written via ``.save``).
MATRIX_RECIPES: frozenset[str] = frozenset({"flat", "stream", "levels"})
#: Recipes whose result is a non-matrix :class:`GSplatNode` tree.
COMPOSED_RECIPES: frozenset[str] = frozenset({"tiles", "overview", "adaptive"})

PartitionRule = Literal["median", "midpoint", "sah"]

#: A recipe builds either a flat dataset or a node-tree (see module docstring).
RecipeResult = Union[GSplatData, GSplatNode]


@dataclass(frozen=True)
class RecipeParams:
    """Parameters for :func:`build_recipe`, with library-faithful defaults.

    The defaults here mirror the historical subcommands. The CLI applies its own
    *scale-derived* defaults (e.g. ``--parts`` → ``max_elements``) before calling
    :func:`build_recipe`; passing ``max_elements=None`` falls back to
    :data:`DEFAULT_MAX_ELEMENTS` so the builders are usable standalone too.
    """

    # stream (additive prefix) ladder — every recipe ladders by default
    n_lods: int = 4
    additive_method: AutoOrMethod = "auto"
    breakpoints: BreakpointSpec = "equal-count"
    truncation_sigmas: float = 3.0
    max_n_dense: int = 2000
    # spatial partition (tiles / overview)
    max_elements: Optional[int] = None
    partition_rule: PartitionRule = "median"
    # substitutive reduction (levels / overview coarse cap / adaptive parts)
    compression_factor: int = 4
    levels: int = 3
    substitutive_method: str = "auto"
    lloyd_iterations: int = 5
    candidate_bins_k: int = 12
    # Anti-grid inter-spread widening (see make_substitutive_lod); 1.0 disables.
    coverage_inflation: float = 3.0
    # Per-level mass conservation over the coarsened dims (kills the additive
    # brightness pop at LOD switches; see make_substitutive_lod).
    conserve_mass: bool = True
    # Additive ladders per substitutive leaf (streaming/progressive first paint).
    # True (default) laddders every substitutive level of the substitutive /
    # adaptive recipes and the overview coarse cap; False emits bare leaves.
    additive_ladders: bool = True
    # Post-merge L2 refit of each coarse level against its fine input
    # ("none" | "l2"; see make_substitutive_lod).
    refine: str = "none"
    refine_iters: int = 120
    # Center-column indices substitutive coarsening may merge over; the
    # complement become hard grouping barriers. None == coarsen all dims.
    coarsen_dims: Optional[tuple] = None
    # LOD switch thresholds are auto-derived as viewport-relative
    # ``coverage_fraction`` (``sqrt(N_i/N_finest)``) — see
    # ``core.group.lod.group.coverage_fractions``. No method selector or
    # per-dataset anchor knob: the fraction is a count ratio (immune to
    # non-displayed-dimension multiplicity) and the viewer anchors the finest at
    # fills-screen via the live viewport diagonal.
    # shared
    device: str = "auto"
    seed: Optional[int] = None

    @property
    def effective_max_elements(self) -> int:
        """``max_elements`` with the :data:`DEFAULT_MAX_ELEMENTS` fallback."""
        return DEFAULT_MAX_ELEMENTS if self.max_elements is None else self.max_elements


# ────────────────────────────────────────────────────────────────────────
# Individual recipe builders
# ────────────────────────────────────────────────────────────────────────


def build_flat(data: GSplatData, params: RecipeParams) -> GSplatData:
    """Collapse to a single leaf (no LOD, no partition)."""
    return data.flattened()


def build_stream(data: GSplatData, params: RecipeParams) -> GSplatData:
    """Reorder into a single additive (prefix-sum) ladder."""
    return make_additive_lod(
        data.flattened(),
        n_lods=params.n_lods,
        method=params.additive_method,
        breakpoints=params.breakpoints,
        truncation_sigmas=params.truncation_sigmas,
        max_n_dense=params.max_n_dense,
        seed=params.seed,
    )


def build_levels(data: GSplatData, params: RecipeParams) -> GSplatData:
    """Coarse→fine replacement levels (the substitutive reduction).

    By default every substitutive level also carries an additive ladder
    (``params.additive_ladders``; streaming-friendly first paint per level) —
    the project convention is additive LODs everywhere unless explicitly
    disabled (``--no-additive``), which emits bare per-level leaves.
    """
    if params.additive_ladders:
        return build_levels_matrix(data, params)
    return make_substitutive_lod(
        data.flattened(),
        compression_factor=params.compression_factor,
        levels=params.levels,
        method=params.substitutive_method,  # type: ignore[arg-type]
        lloyd_iterations=params.lloyd_iterations,
        candidate_bins_k=params.candidate_bins_k,
        coverage_inflation=params.coverage_inflation,
        conserve_mass=params.conserve_mass,
        refine=params.refine,  # type: ignore[arg-type]
        refine_iters=params.refine_iters,
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
    )


def build_levels_matrix(data: GSplatData, params: RecipeParams) -> GSplatData:
    """Build the balanced substitutive × additive matrix."""
    return make_lod_pyramid(
        data.flattened(),
        compression_factor=params.compression_factor,
        levels=params.levels,
        substitutive_method=params.substitutive_method,  # type: ignore[arg-type]
        lloyd_iterations=params.lloyd_iterations,
        candidate_bins_k=params.candidate_bins_k,
        coverage_inflation=params.coverage_inflation,
        conserve_mass=params.conserve_mass,
        refine=params.refine,
        refine_iters=params.refine_iters,
        device=params.device,
        coarsen_dims=params.coarsen_dims,
        n_additive_lods=params.n_lods,
        additive_method=params.additive_method,
        breakpoints=params.breakpoints,
        truncation_sigmas=params.truncation_sigmas,
        max_n_dense=params.max_n_dense,
        seed=params.seed,
    )


def build_tiles(data: GSplatData, params: RecipeParams) -> GSplatPartition:
    """Spatially partition, then build an additive ladder **within each part**.

    ``to_spatial_partition`` yields a flat :class:`GSplatPartition` whose children
    are single-level leaves; this replaces each part with its own additive ladder
    (clamped to the part's splat count so no empty LOD bins are produced).
    """
    base = data.flattened()
    # Typo guard: explicit `counts:` breakpoints must fit the WHOLE dataset
    # (strict, aborts loudly); the individual — smaller — parts then clamp in
    # _ladder_for_part.
    validate_counts_breakpoints(params.breakpoints, base.n_splats)
    partition = base.to_spatial_partition(
        max_elements=params.effective_max_elements,
        rule=params.partition_rule,
    )
    children: List[GSplatNode] = [
        _ladder_for_part(part, params) for part in partition.children
    ]
    return GSplatPartition(
        children=children,
        max_elements=partition.max_elements,
        meta=dict(partition.meta),
    )


def _substitutive_for_part(part: GSplatNode, params: RecipeParams) -> GSplatNode:
    """Rebuild one partition child as its own substitutive lod group (coarse↔fine
    swap), the per-part analogue of :func:`_ladder_for_part`."""
    import math

    part_data = GSplatData.from_tree(part)
    # Clamp the substitutive depth so a small part doesn't synthesise degenerate
    # (sub-1-splat) coarse levels: ``levels`` coarser levels need the coarsest to
    # hold >= 1 splat, i.e. n / K**levels >= 1  ->  levels <= log_K(n).
    n = part_data.n_splats
    k = max(2, params.compression_factor)
    max_levels = int(math.log(n) / math.log(k)) if n > 1 else 0
    eff_levels = max(1, min(params.levels, max_levels))
    sub = make_substitutive_lod(
        part_data,
        compression_factor=params.compression_factor,
        levels=eff_levels,
        method=params.substitutive_method,  # type: ignore[arg-type]
        lloyd_iterations=params.lloyd_iterations,
        candidate_bins_k=params.candidate_bins_k,
        coverage_inflation=params.coverage_inflation,
        conserve_mass=params.conserve_mass,
        refine=params.refine,  # type: ignore[arg-type]
        refine_iters=params.refine_iters,
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
    )
    if params.additive_ladders:
        # Additive ladder inside every per-part substitutive level (clamped to
        # each level's own count; `counts:` specs clamp per level).
        for s in range(sub.n_substitutive):
            level_n = sub.at_substitutive(s).n_splats
            sub = make_additive_lod(
                sub,
                n_lods=max(1, min(params.n_lods, level_n)),
                method=params.additive_method,
                breakpoints=clamp_counts_breakpoints(params.breakpoints, level_n),
                truncation_sigmas=params.truncation_sigmas,
                max_n_dense=params.max_n_dense,
                seed=None if params.seed is None else params.seed + s,
                substitutive_level=s,
            )
    # Build each part's lod group with viewport-relative coverage_fraction
    # thresholds, so adaptive's per-part coarse↔fine switch anchors at fills-screen.
    from luxar.gsplats.tree import tree_from_substitutive_levels

    return tree_from_substitutive_levels(sub.substitutive_levels)


def build_adaptive(data: GSplatData, params: RecipeParams) -> GSplatPartition:
    """Spatially partition, then give **each part its own substitutive lod group**.

    The result is a ``kind=partition`` whose every child is a ``kind=lod`` group
    (coarse↔fine *replacement* per part), so each spatial cell frustum-culls AND
    picks its own LOD level by its own on-screen size — locally adaptive detail.

    Contrast the siblings: ``tiles`` gives each part an *additive* (prefix-
    sum, accumulating) ladder; ``overview`` puts a single *global* substitutive
    cap above one partition. ``adaptive`` is the per-part substitutive form — the
    most adaptive of the three, for the largest scenes.
    """
    base = data.flattened()
    partition = base.to_spatial_partition(
        max_elements=params.effective_max_elements,
        rule=params.partition_rule,
    )
    children: List[GSplatNode] = [
        _substitutive_for_part(part, params) for part in partition.children
    ]
    return GSplatPartition(
        children=children,
        max_elements=partition.max_elements,
        meta=dict(partition.meta),
    )


def build_overview(data: GSplatData, params: RecipeParams) -> GSplatLodGroup:
    """Coarse substitutive cap (far view) + a ``tiles`` fine branch.

    The result is a ``kind=lod`` group with children **coarsest→finest in memory**
    (``[coarse_leaf, fine_partition]``, matching the on-disk order). Each child's
    ``coverage_fraction`` selector threshold is stamped onto its ``meta`` (honored
    by both the standalone writer and the scene graft) via ``coverage_fractions``
    (``sqrt(N_i/N_finest)``): the coarse cap gets a fraction below the fine branch's
    1.0, so the fine branch shows at fills-screen and the coarse cap takes over as
    the node shrinks. The viewer anchors the finest at fills-screen via the live
    viewport diagonal (no per-dataset tuning; see ``RecipeParams``).
    """
    from luxar.core.group.lod.group import coverage_fractions
    from luxar.gsplats.tree import total_splats

    base = data.flattened()
    fine_partition = build_tiles(base, params)

    # A single coarse substitutive level (levels=1 → n_substitutive == 2, with
    # index 1 the coarsest in the finest-first matrix view). Flatten its additive
    # ladder to one representative leaf.
    capped = make_substitutive_lod(
        base,
        compression_factor=params.compression_factor,
        levels=1,
        method=params.substitutive_method,  # type: ignore[arg-type]
        lloyd_iterations=params.lloyd_iterations,
        candidate_bins_k=params.candidate_bins_k,
        coverage_inflation=params.coverage_inflation,
        conserve_mass=params.conserve_mass,
        refine=params.refine,  # type: ignore[arg-type]
        refine_iters=params.refine_iters,
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
    )
    coarse_leaf = capped.at_substitutive(capped.n_substitutive - 1).flattened().tree
    if params.additive_ladders:
        # Additive ladder inside the coarse cap too (project convention:
        # additive LODs everywhere by default) — the fine branch's parts get
        # theirs from build_tiles.
        coarse_leaf = _ladder_for_part(coarse_leaf, params)

    # Children are coarsest→finest: [coarse cap, fine partition]. Derive per-child
    # coverage fractions from the whole-subtree splat counts (fine partition's count
    # is the whole partition via ``total_splats``) and stamp into the children's
    # (mutable) meta in place — same pattern as tree_from_substitutive_levels.
    group = GSplatLodGroup(children=[coarse_leaf, fine_partition])
    coarse_cov, fine_cov = coverage_fractions(
        [total_splats(coarse_leaf), total_splats(fine_partition)]
    )
    coarse_leaf.meta["coverage_fraction"] = coarse_cov
    fine_partition.meta["coverage_fraction"] = fine_cov
    return group


def _ladder_for_part(part: GSplatNode, params: RecipeParams) -> GSplatNode:
    """Rebuild one partition child as a leaf carrying an additive ladder."""
    part_data = GSplatData.from_tree(part)
    # Clamp ladder depth so a small part never yields empty equal-count bins,
    # and clamp explicit `counts:` breakpoints to THIS part's size — parts have
    # differing N, so a fixed counts list whose largest cut exceeds a small
    # part would otherwise abort the whole build ("largest breakpoint exceeds
    # N"). String/energy specs pass through (already size-adaptive).
    eff_n_lods = max(1, min(params.n_lods, part_data.n_splats))
    eff_breakpoints = clamp_counts_breakpoints(params.breakpoints, part_data.n_splats)
    laddered = make_additive_lod(
        part_data,
        n_lods=eff_n_lods,
        method=params.additive_method,
        breakpoints=eff_breakpoints,
        truncation_sigmas=params.truncation_sigmas,
        max_n_dense=params.max_n_dense,
        seed=params.seed,
    )
    return laddered.tree


#: Per-part recipes — the recipes that have a single-part form (the building
#: block of ``tiles``/``adaptive``), usable for streaming per-part assembly
#: such as the tiled-batch merge. ``additive`` → a prefix-sum ladder
#: (tiles), ``levels`` → a coarse↔fine lod group (adaptive).
PER_PART_RECIPES: tuple[str, ...] = ("stream", "levels")


def uniform_per_part_lod_warning(
    tiling_mode: Optional[str], recipe: Optional[str]
) -> Optional[str]:
    """Warn when ANY per-part LOD recipe is applied to uniform (apodized) tiles.

    Uniform (``--tiling uniform``) tiles overlap with Hann-apodized halos that
    form a *partition of unity*: a boundary feature is split into two tapered
    splats in adjacent parts whose amplitudes sum to 1.0. That identity holds
    **only at the finest level** — per-part LOD coarsens each part independently,
    so it breaks at coarse levels for BOTH recipes (the viewer hard-switches
    levels with no cross-level blending, so the artifact is visible):

    * ``stream`` (a prefix ladder) orders by mass and keeps a prefix, so the
      low-amplitude halo splats are dropped first at coarse levels — the
      overlap loses signal and **dims to a seam** (often the worse of the two).
    * ``levels`` merges each part's halo splats into representatives
      independently, so the complementary halves no longer align — the overlap
      **smears** at coarse levels.

    Content tiling (``--tiling content``, disjoint core-keep parts) carries no
    shared halos, so per-part LOD is exact there for either recipe.

    Returns the warning text (caller emits it) when ``tiling_mode`` is uniform
    and ``recipe`` is a per-part recipe, else ``None``.
    """
    recipe = canonical_recipe_name(recipe) if recipe is not None else None
    if tiling_mode != "uniform" or recipe not in PER_PART_RECIPES:
        return None
    consequence = (
        "the stream ladder drops the low-amplitude halo splats at coarse "
        "levels (the overlap dims to a seam)"
        if recipe == "stream"
        else "the level merge combines each part's halo splats independently "
        "(the overlap smears at coarse levels)"
    )
    return (
        f"--recipe {recipe} on uniform (Hann-apodized) tiles: the halo "
        f"partition-of-unity holds only at the finest level — {consequence}. For "
        "exact coarse LODs use --tiling content (disjoint core-keep parts)."
    )


def build_part_lod(part: GSplatNode, recipe: str, params: RecipeParams) -> GSplatNode:
    """Give ONE partition child its own per-part LOD, with depth clamped to the
    part's splat count (so a small part never synthesises degenerate levels).

    This is the exact building block :func:`build_tiles` (``stream``) and
    :func:`build_adaptive` (``levels``) apply to every part — exposed so a
    streaming assembler (e.g. the tiled-batch merge) can LOD one part at a time
    without materialising the whole partition. Returns the per-part node:
    a leaf-with-ladder (``stream``) or a substitutive ``GSplatLodGroup``
    (``levels``). Legacy spellings translate via :func:`canonical_recipe_name`.
    """
    recipe = canonical_recipe_name(recipe)
    if recipe == "stream":
        return _ladder_for_part(part, params)
    if recipe == "levels":
        return _substitutive_for_part(part, params)
    raise ValueError(
        f"build_part_lod: {recipe!r} has no per-part form; "
        f"choose from {', '.join(PER_PART_RECIPES)}"
    )


# ────────────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────────────

_BUILDERS: dict[str, Callable[[GSplatData, RecipeParams], RecipeResult]] = {
    "flat": build_flat,
    "stream": build_stream,
    "levels": build_levels,
    "tiles": build_tiles,
    "overview": build_overview,
    "adaptive": build_adaptive,
}


def build_recipe(
    data: GSplatData, recipe: RecipeName, params: RecipeParams
) -> RecipeResult:
    """Build ``recipe`` from ``data``.

    Returns a :class:`GSplatData` for the matrix recipes
    (``flat``/``stream``/``levels``) and a
    :class:`~luxar.gsplats.tree.GSplatNode` for the composed recipes
    (``tiles``/``overview``/``adaptive``) — see :data:`RecipeResult`.
    """
    try:
        builder = _BUILDERS[recipe]
    except KeyError:
        raise ValueError(
            f"unknown recipe {recipe!r}; choose from {', '.join(RECIPE_NAMES)}"
        ) from None
    return builder(data, params)
