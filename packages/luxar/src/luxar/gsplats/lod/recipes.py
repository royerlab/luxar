"""Representation **recipes** — assemble a fitted gsplat set into a topology.

A fitted ``GSplatData`` can be laid out as one of several renderable tree
topologies, ordered roughly by dataset scale:

* ``flat`` — a single leaf (no LOD, no partition).
* ``additive`` — one leaf carrying an additive (prefix-sum) ladder.
* ``partitioned`` — a spatial BSP ``kind=partition`` whose **every part carries
  its own additive ladder** (frustum-cull off-screen parts; stream detail within
  the visible ones).
* ``multiscale`` — an *unbalanced-by-design* ``kind=lod``: a single cheap coarse
  substitutive cap for the far view, plus a ``partitioned`` fine branch for
  close-up. Detail structure exists only where you look closely.
* ``mosaic`` — a spatial BSP ``kind=partition`` whose **every part is its own
  substitutive lod group** (per-part coarse↔fine *replacement*). Each cell
  frustum-culls AND picks its own level by its own on-screen size — locally
  adaptive detail, the per-part substitutive sibling of ``partitioned`` (additive
  parts) and ``multiscale`` (one global cap).

plus the two lower-level primitives:

* ``substitutive`` — a pure substitutive pyramid (synthesised coarser levels).
* ``pyramid`` — the balanced substitutive × additive matrix.

These functions are **pure** (``GSplatData`` in, a result out) with no Typer/IO —
the CLI wrapper lives in :mod:`luxar.cli.lod`. They only *compose* the existing
builders (:func:`make_additive_lod`, :func:`make_substitutive_lod`,
:func:`make_lod_pyramid`, :meth:`GSplatData.to_spatial_partition`); no new
algorithms are introduced.

Two return shapes (see :data:`RecipeResult`):

* the **matrix** recipes (``flat``/``additive``/``substitutive``/``pyramid``)
  return a :class:`~luxar.gsplats.gsplat_data.GSplatData`, so the CLI writes them
  via :meth:`GSplatData.save` — byte-identical to the historical
  ``lod additive``/``lod substitutive``/``lod pyramid`` subcommands they absorb.
* the **composed** recipes (``partitioned``/``multiscale``/``mosaic``) return a
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

#: The recipe vocabulary, ordered by dataset scale (the five-rung ladder first
#: — flat/additive/partitioned/multiscale/mosaic — then the two absorbed
#: primitives substitutive/pyramid).
RecipeName = Literal[
    "flat",
    "additive",
    "partitioned",
    "multiscale",
    "mosaic",
    "substitutive",
    "pyramid",
]

#: Tuple form of :data:`RecipeName` for CLI choices / validation.
RECIPE_NAMES: tuple[str, ...] = get_args(RecipeName)

#: Recipes whose result is a flat :class:`GSplatData` (written via ``.save``).
MATRIX_RECIPES: frozenset[str] = frozenset(
    {"flat", "additive", "substitutive", "pyramid"}
)
#: Recipes whose result is a non-matrix :class:`GSplatNode` tree.
COMPOSED_RECIPES: frozenset[str] = frozenset({"partitioned", "multiscale", "mosaic"})

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

    # additive ladder (additive / partitioned / multiscale fine parts / pyramid)
    n_lods: int = 4
    additive_method: AutoOrMethod = "auto"
    breakpoints: BreakpointSpec = "equal-count"
    truncation_sigmas: float = 3.0
    max_n_dense: int = 2000
    # spatial partition (partitioned / multiscale)
    max_elements: Optional[int] = None
    partition_rule: PartitionRule = "median"
    # substitutive reduction (substitutive / pyramid / multiscale coarse cap)
    compression_factor: int = 4
    levels: int = 3
    substitutive_method: str = "auto"
    lloyd_iterations: int = 5
    candidate_bins_k: int = 12
    # Anti-grid inter-spread widening (see make_substitutive_lod); 1.0 disables.
    coverage_inflation: float = 3.0
    # Center-column indices substitutive coarsening may merge over; the
    # complement become hard grouping barriers. None == coarsen all dims.
    coarsen_dims: Optional[tuple] = None
    # LOD switching-threshold derivation (multiscale coarse↔fine switch).
    #   ``lod_method="extent"`` (default): physically-anchored W / r — the switch
    #     is anchored in the element's on-screen pixel size, so it self-calibrates
    #     and the substitutive cap's fewer-but-LARGER splats raise its threshold
    #     correctly (no per-dataset tuning).
    #   ``lod_method="count"``: legacy scene-relative √(N/N₀) proxy.
    # ``extent_percentile`` / ``extent_anisotropy`` tune the per-level element
    # radius r (default p90, largest-semi-axis). ``base_pixel_size`` is the anchor:
    # the target element pixel size T (~1.5 px) in extent mode, or the √N anchor
    # (~10 px) in count mode; ``None`` → the method's default. See
    # ``core.group.lod.group.lod_thresholds`` / ``extent_min_pixel_sizes``.
    lod_method: str = "extent"
    extent_percentile: float = 90.0
    extent_anisotropy: bool = True
    base_pixel_size: Optional[float] = None
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


def build_additive(data: GSplatData, params: RecipeParams) -> GSplatData:
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


def build_substitutive(data: GSplatData, params: RecipeParams) -> GSplatData:
    """Synthesise a pure substitutive pyramid (coarser representative levels)."""
    return make_substitutive_lod(
        data.flattened(),
        compression_factor=params.compression_factor,
        levels=params.levels,
        method=params.substitutive_method,  # type: ignore[arg-type]
        lloyd_iterations=params.lloyd_iterations,
        candidate_bins_k=params.candidate_bins_k,
        coverage_inflation=params.coverage_inflation,
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
    )


def build_pyramid(data: GSplatData, params: RecipeParams) -> GSplatData:
    """Build the balanced substitutive × additive matrix."""
    return make_lod_pyramid(
        data.flattened(),
        compression_factor=params.compression_factor,
        levels=params.levels,
        substitutive_method=params.substitutive_method,  # type: ignore[arg-type]
        lloyd_iterations=params.lloyd_iterations,
        candidate_bins_k=params.candidate_bins_k,
        coverage_inflation=params.coverage_inflation,
        device=params.device,
        coarsen_dims=params.coarsen_dims,
        n_additive_lods=params.n_lods,
        additive_method=params.additive_method,
        breakpoints=params.breakpoints,
        truncation_sigmas=params.truncation_sigmas,
        max_n_dense=params.max_n_dense,
        seed=params.seed,
    )


def build_partitioned(data: GSplatData, params: RecipeParams) -> GSplatPartition:
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
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
    )
    # Build each part's lod group with the chosen threshold knobs (default
    # extent T·W/r), so mosaic's per-part coarse↔fine switch is tunable too.
    from luxar.gsplats.tree import tree_from_substitutive_levels

    return tree_from_substitutive_levels(
        sub.substitutive_levels,
        lod_method=params.lod_method,
        extent_percentile=params.extent_percentile,
        extent_anisotropy=params.extent_anisotropy,
        base_pixel_size=params.base_pixel_size,
    )


def build_mosaic(data: GSplatData, params: RecipeParams) -> GSplatPartition:
    """Spatially partition, then give **each part its own substitutive lod group**.

    The result is a ``kind=partition`` whose every child is a ``kind=lod`` group
    (coarse↔fine *replacement* per part), so each spatial cell frustum-culls AND
    picks its own LOD level by its own on-screen size — locally adaptive detail.

    Contrast the siblings: ``partitioned`` gives each part an *additive* (prefix-
    sum, accumulating) ladder; ``multiscale`` puts a single *global* substitutive
    cap above one partition. ``mosaic`` is the per-part substitutive form — the
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


def build_multiscale(data: GSplatData, params: RecipeParams) -> GSplatLodGroup:
    """Coarse substitutive cap (far view) + a ``partitioned`` fine branch.

    The result is a ``kind=lod`` group with children **coarsest→finest in memory**
    (``[coarse_leaf, fine_partition]``, matching the on-disk order). Each child's
    ``min_pixel_size`` selector threshold is stamped onto its ``meta`` (honored by
    both the standalone writer and the scene graft) via ``lod_thresholds``: by
    default the physically-anchored ``extent`` method (``T·W/r``, anisotropy-aware
    p90 radius), so the coarse cap — whose splats are fewer but LARGER — gets a
    correctly higher fine-branch threshold and shows when the node is far/small on
    screen, with the fine branch taking over up close.
    ``params.lod_method``/``extent_percentile``/``extent_anisotropy``/
    ``base_pixel_size`` tune this (see ``RecipeParams``).
    """
    from luxar.core.group.lod.group import lod_thresholds
    from luxar.gsplats.tree import (
        node_extent_diagonal,
        node_percentile_radius,
        total_splats,
    )

    base = data.flattened()
    fine_partition = build_partitioned(base, params)

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
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
    )
    coarse_leaf = capped.at_substitutive(capped.n_substitutive - 1).flattened().tree

    # Children are coarsest→finest: [coarse cap, fine partition]. Build the group
    # ONCE, derive per-child thresholds from it, and stamp min_pixel_size into the
    # children's (mutable) meta in place — same pattern as
    # tree_from_substitutive_levels (no throwaway group / replace()).
    group = GSplatLodGroup(children=[coarse_leaf, fine_partition])
    pct, aniso = params.extent_percentile, params.extent_anisotropy
    coarse_mps, fine_mps = lod_thresholds(
        params.lod_method,  # type: ignore[arg-type]
        element_counts=[total_splats(coarse_leaf), total_splats(fine_partition)],
        element_extents=[
            node_percentile_radius(coarse_leaf, pct, aniso),
            node_percentile_radius(fine_partition, pct, aniso),
        ],
        node_extent=node_extent_diagonal(group),
        base_pixel_size=params.base_pixel_size,
    )
    coarse_leaf.meta["min_pixel_size"] = coarse_mps
    fine_partition.meta["min_pixel_size"] = fine_mps
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
#: block of ``partitioned``/``mosaic``), usable for streaming per-part assembly
#: such as the tiled-batch merge. ``additive`` → a prefix-sum ladder
#: (partitioned), ``substitutive`` → a coarse↔fine lod group (mosaic).
PER_PART_RECIPES: tuple[str, ...] = ("additive", "substitutive")


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

    * ``additive`` orders by mass and keeps a prefix, so the low-amplitude halo
      splats are dropped first at coarse levels — the overlap loses signal and
      **dims to a seam** (often the worse of the two).
    * ``substitutive`` merges each part's halo splats into representatives
      independently, so the complementary halves no longer align — the overlap
      **smears** at coarse levels.

    Content tiling (``--tiling content``, disjoint core-keep parts) carries no
    shared halos, so per-part LOD is exact there for either recipe.

    Returns the warning text (caller emits it) when ``tiling_mode`` is uniform
    and ``recipe`` is a per-part recipe, else ``None``.
    """
    if tiling_mode != "uniform" or recipe not in PER_PART_RECIPES:
        return None
    consequence = (
        "additive drops the low-amplitude halo splats at coarse levels (the "
        "overlap dims to a seam)"
        if recipe == "additive"
        else "substitutive merges each part's halo splats independently (the "
        "overlap smears at coarse levels)"
    )
    return (
        f"--recipe {recipe} on uniform (Hann-apodized) tiles: the halo "
        f"partition-of-unity holds only at the finest level — {consequence}. For "
        "exact coarse LODs use --tiling content (disjoint core-keep parts)."
    )


def build_part_lod(part: GSplatNode, recipe: str, params: RecipeParams) -> GSplatNode:
    """Give ONE partition child its own per-part LOD, with depth clamped to the
    part's splat count (so a small part never synthesises degenerate levels).

    This is the exact building block :func:`build_partitioned` (``additive``) and
    :func:`build_mosaic` (``substitutive``) apply to every part — exposed so a
    streaming assembler (e.g. the tiled-batch merge) can LOD one part at a time
    without materialising the whole partition. Returns the per-part node:
    a leaf-with-ladder (``additive``) or a substitutive ``GSplatLodGroup``.
    """
    if recipe == "additive":
        return _ladder_for_part(part, params)
    if recipe == "substitutive":
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
    "additive": build_additive,
    "partitioned": build_partitioned,
    "multiscale": build_multiscale,
    "mosaic": build_mosaic,
    "substitutive": build_substitutive,
    "pyramid": build_pyramid,
}


def build_recipe(
    data: GSplatData, recipe: RecipeName, params: RecipeParams
) -> RecipeResult:
    """Build ``recipe`` from ``data``.

    Returns a :class:`GSplatData` for the matrix recipes
    (``flat``/``additive``/``substitutive``/``pyramid``) and a
    :class:`~luxar.gsplats.tree.GSplatNode` for the composed recipes
    (``partitioned``/``multiscale``/``mosaic``) — see :data:`RecipeResult`.
    """
    try:
        builder = _BUILDERS[recipe]
    except KeyError:
        raise ValueError(
            f"unknown recipe {recipe!r}; choose from {', '.join(RECIPE_NAMES)}"
        ) from None
    return builder(data, params)
