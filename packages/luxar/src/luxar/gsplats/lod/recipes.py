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

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Callable, List, Literal, Optional, Tuple, Union, get_args

import numpy as np

from luxar.core.group.lod.group import (
    coverage_fractions,
    partitioned_coverage_fractions,
)
from luxar.core.group.partition import DEFAULT_MAX_ELEMENTS
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.additive import (
    AutoOrMethod,
    BreakpointSpec,
    clamp_counts_breakpoints,
    make_additive_lod,
    sibling_aware_stream_breakpoints,
    validate_counts_breakpoints,
)
from luxar.gsplats.lod.pyramid import make_lod_pyramid
from luxar.gsplats.lod.substitutive import make_substitutive_lod
from luxar.gsplats.tree import (
    GSplatLodGroup,
    GSplatNode,
    GSplatPartition,
    iter_leaves,
)
from luxar.utils.lod_methods import is_reveal_method

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
    # `additive_method="radial"` only — the concentric-shell reveal. Defaults:
    # the bbox centre, over the non-degenerate axes.
    reveal_center: Optional[Sequence[float]] = None
    spatial_dims: Optional[Sequence[int]] = None
    breakpoints: BreakpointSpec = "equal-count"
    # None = the dataset's own `truncation_radius` (the support it was fitted and
    # is rendered at). Threaded through as-is; the additive builders resolve it.
    truncation_sigmas: Optional[float] = None
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
    # Post-merge refit of each coarse level: "l2" against its fine input,
    # "volume" against the source volume ("none" | "l2" | "volume"; see
    # make_substitutive_lod).
    refine: str = "none"
    # None resolves to each refine engine's own default (l2: 120, volume: 300).
    refine_iters: Optional[int] = None
    # Source volume for refine="volume" (full-res, splat coordinate frame).
    # Excluded from eq/repr: a large ndarray is payload, not identity.
    volume: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    # Normalization level removed from ``volume`` by the input fit. Per-part
    # paths cannot recover top-level stats from a bare tree node, so the owning
    # caller carries it alongside the volume payload.
    image_min: Optional[float] = None
    # ``volume_axes[i]`` is the volume axis holding center dim ``i``; None means
    # the identity. A stacked timelapse needs it: Luxar puts spatial dims first
    # and the stacked axis LAST, while the source array is usually time-FIRST.
    volume_axes: Optional[tuple] = None
    # Center-column indices substitutive coarsening may merge over; the
    # complement become hard grouping barriers. None == coarsen all dims.
    coarsen_dims: Optional[tuple] = None
    # LOD switch thresholds are auto-derived as SCREEN-AREA fractions
    # (``selector="screen-area"``, occupancy halving: full detail while the
    # node occupies at least half the screen, one level coarser per halving of
    # occupied area) — see ``core.group.lod.group.coverage_fractions``. No
    # per-dataset anchor knob.
    # Q·e quality stamps: measure each coarse substitutive level's mixture-L²
    # quality Q vs its group's finest content and stamp it (with the
    # reference_energy weight w) into level_stats — the build-time half of the
    # viewer's recursive Q·e quality algebra (see gsplats.lod.quality).
    # Constant-cost sampled estimator; ON by default at this pipeline layer
    # (the primitive make_substitutive_lod defaults to False).
    quality_stamps: bool = True
    quality_max_pair_splats: int = 2_000_000
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
        reveal_center=params.reveal_center,
        spatial_dims=params.spatial_dims,
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
        volume=params.volume,
        volume_axes=params.volume_axes,
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
        quality_stamps=params.quality_stamps,
        quality_max_pair_splats=params.quality_max_pair_splats,
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
        volume=params.volume,
        volume_axes=params.volume_axes,
        device=params.device,
        coarsen_dims=params.coarsen_dims,
        n_additive_lods=params.n_lods,
        additive_method=params.additive_method,
        additive_reveal_center=params.reveal_center,
        additive_spatial_dims=params.spatial_dims,
        breakpoints=params.breakpoints,
        truncation_sigmas=params.truncation_sigmas,
        max_n_dense=params.max_n_dense,
        seed=params.seed,
        quality_stamps=params.quality_stamps,
        quality_max_pair_splats=params.quality_max_pair_splats,
    )


def build_tiles(
    data: GSplatData,
    params: RecipeParams,
    *,
    sibling_compression: Optional[int] = None,
) -> GSplatPartition:
    """Spatially partition, then build an additive ladder **within each part**.

    ``to_spatial_partition`` yields a flat :class:`GSplatPartition` whose children
    are single-level leaves; this replaces each part with its own additive ladder
    (clamped to the part's splat count so no empty LOD bins are produced).
    ``sibling_compression`` forwards to :func:`_ladder_for_part` when this
    partition is the fine branch under a coarser lod-group sibling (overview).
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
        _ladder_for_part(part, params, sibling_compression=sibling_compression)
        for part in partition.children
    ]
    return GSplatPartition(
        children=children,
        max_elements=partition.max_elements,
        meta=dict(partition.meta),
        # Per-part ladders preserve child order + count, so the BSP tree's
        # leaf→part-index mapping still holds — carry it for viewer ordering.
        bsp_tree=partition.bsp_tree,
    )


def _part_cells(
    partition: GSplatPartition, ndim: int
) -> Optional[List[List[Tuple[float, float]]]]:
    """Each part's tile, per center dim, indexed by child position.

    Read from the partition's own ``bsp_tree`` so the bounds are the real split
    planes rather than the hull of the splats a part happens to hold — a hull is
    strictly tighter than the cell and would crop away signal the tile owns.
    Falls back to each part's center bounds when no tree was stored (an older
    artifact, or a producer that never built one), and to ``None`` when neither
    is available, which turns the per-part volume re-fit off rather than
    silently cropping to the wrong box.
    """
    from luxar.core.group.partition import serialized_bsp_leaf_cells
    from luxar.gsplats.tree import center_bounds

    n = len(partition.children)
    tree = getattr(partition, "bsp_tree", None)
    if tree:
        try:
            by_label = serialized_bsp_leaf_cells(tree, ndim)
        except (KeyError, TypeError, ValueError):
            by_label = {}
        if len(by_label) >= n:
            return (
                [by_label[i] for i in range(n)]
                if all(i in by_label for i in range(n))
                else None
            )
    cells: List[List[Tuple[float, float]]] = []
    for child in partition.children:
        bounds = center_bounds(child)
        if bounds is None:
            return None
        lo, hi = bounds
        cells.append([(float(lo[d]), float(hi[d])) for d in range(ndim)])
    return cells


def _cell_for_coarsened_dims(
    cell: Optional[List[Tuple[float, float]]],
    params: RecipeParams,
    ndim: int,
) -> Optional[List[Tuple[float, float]]]:
    """Reduce a per-center-dim tile cell to the dims the re-fit actually spans.

    The sub-volume drops the barrier dims, so its box is indexed by the
    COARSENED dims only. Returns ``None`` when there is nothing to crop to, or
    when the re-fit is not volume-based (the box would be dead weight).

    Sorted and de-duplicated the same way ``make_substitutive_lod`` normalises
    ``coarsen_dims``: the sub-volume's retained dims come back in ASCENDING order,
    so a box built in the order the caller happened to spell them (``2,1,0``)
    would crop each axis to another axis's bounds.
    """
    if cell is None or params.refine != "volume":
        return None
    free = (
        tuple(range(ndim))
        if params.coarsen_dims is None
        else tuple(sorted({int(d) for d in params.coarsen_dims}))
    )
    return [cell[d] for d in free]


def _substitutive_for_part(
    part: GSplatNode,
    params: RecipeParams,
    *,
    coverage: Callable[[List[int]], List[float]] = partitioned_coverage_fractions,
    cell: Optional[List[Tuple[float, float]]] = None,
) -> GSplatNode:
    """Rebuild one partition child as its own substitutive lod group (coarse↔fine
    swap), the per-part analogue of :func:`_ladder_for_part`.

    ``coverage`` is the selector-threshold derivation, defaulting to the
    partition-bound (fills-screen) anchor — see the note below the ladder build
    and :func:`~luxar.core.group.lod.group.partitioned_coverage_fractions`.

    ``cell`` is this part's own tile, per center dim, and is what makes
    ``refine="volume"`` sound here: the re-fit sees only that tile's CROP of the
    volume, so it is not tempted to pull splats out of the tile to explain signal
    that belongs to a neighbour. A re-fit that moves a centre out of the cell
    anyway is discarded in favour of the merge. Without a ``cell`` the re-fit
    would target the full volume, which is exactly the unsound case.
    """
    import math

    if params.refine == "volume" and cell is None:
        raise ValueError(
            "refine='volume' on per-part levels needs the part's own cell to "
            "crop the volume to; a fit against the FULL volume would pull "
            "splats out of their tile (breaking partition/frustum-culling "
            "semantics, which the never-worse MSE guard cannot see)."
        )
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
        volume=params.volume,
        volume_axes=params.volume_axes,
        image_min=params.image_min,
        volume_box=_cell_for_coarsened_dims(cell, params, part_data.ndim),
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
        quality_stamps=params.quality_stamps,
        quality_max_pair_splats=params.quality_max_pair_splats,
    )
    if params.additive_ladders:
        # Additive ladder inside every per-part substitutive level (clamped to
        # each level's own count; `counts:` specs clamp per level). Levels
        # finer than the part's coarsest get sibling-aware stream bases so an
        # in-tile upgrade catches its coarser sibling at chunk 1-2 (see
        # sibling_aware_stream_breakpoints); the coarsest keeps the user base.
        coarsest = sub.n_substitutive - 1
        for s in range(sub.n_substitutive):
            level_n = sub.at_substitutive(s).n_splats
            level_breakpoints = clamp_counts_breakpoints(params.breakpoints, level_n)
            if s < coarsest:
                level_breakpoints = sibling_aware_stream_breakpoints(
                    level_breakpoints, level_n, params.compression_factor
                )
            sub = make_additive_lod(
                sub,
                n_lods=max(1, min(params.n_lods, level_n)),
                method=params.additive_method,
                breakpoints=level_breakpoints,
                truncation_sigmas=params.truncation_sigmas,
                max_n_dense=params.max_n_dense,
                seed=None if params.seed is None else params.seed + s,
                substitutive_level=s,
                reveal_center=params.reveal_center,
                spatial_dims=params.spatial_dims,
            )
    # Build each part's lod group with screen-area coverage_fraction
    # thresholds. These are PARTITIONED ladders, and here the reason is GEOMETRIC:
    # the switching group's bbox is one BSP tile, intrinsically a fraction of the
    # whole object's, so its metric reads systematically low. They therefore keep
    # the fills-screen anchor (finest = PARTITION_FINEST_AREA = 1.0) instead of
    # the whole-object half-screen one — see partitioned_coverage_fractions.
    # Without this every tile would sit on its FINEST level while the object is
    # merely full-frame (~16x the resident geometry for a K=4/L=2 ladder). The
    # caller overrides it when the "partition" turns out to hold a single part.
    from luxar.gsplats.tree import tree_from_substitutive_levels

    # Explicit selector: both callables this recipe threads through `coverage`
    # (coverage_fractions / partitioned_coverage_fractions) produce SCREEN-AREA
    # fractions — without this, a custom-callable default of "coverage" would
    # mislabel them (that default exists for EXTERNAL callables written in the
    # legacy diagonal units).
    return tree_from_substitutive_levels(
        sub.substitutive_levels, coverage=coverage, selector="screen-area"
    )


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
    from luxar.gsplats.fit_basis import fit_image_min

    if params.image_min is None:
        params = replace(params, image_min=fit_image_min(data.stats))
    base = data.flattened()
    partition = base.to_spatial_partition(
        max_elements=params.effective_max_elements,
        rule=params.partition_rule,
    )
    # A one-part "partition" is not a tiling. The BSP stops as soon as the whole
    # dataset fits ``max_elements`` (default 1,000,000 — so this is the COMMON
    # case, not an edge one), and ``to_spatial_partition`` still wraps that single
    # leaf in a ``GSplatPartition``. That part's bbox IS the whole object's, so
    # the geometric reason for the fills-screen anchor is absent and applying it
    # anyway would hold the finest level back until the object overfills the
    # screen — precisely the #1361 blur this recipe's sibling anchors exist to
    # avoid. Fall back to the whole-object anchor for that shape.
    coverage = (
        partitioned_coverage_fractions
        if len(partition.children) > 1
        else coverage_fractions
    )
    cells = _part_cells(partition, base.ndim)
    children: List[GSplatNode] = [
        _substitutive_for_part(
            part, params, coverage=coverage, cell=cells[i] if cells else None
        )
        for i, part in enumerate(partition.children)
    ]
    return GSplatPartition(
        children=children,
        max_elements=partition.max_elements,
        meta=dict(partition.meta),
        # Per-part lod groups preserve child order + count — carry the BSP
        # tree's leaf→part-index mapping for viewer back-to-front ordering.
        bsp_tree=partition.bsp_tree,
    )


def build_overview(data: GSplatData, params: RecipeParams) -> GSplatLodGroup:
    """Coarse substitutive cap (far view) + a ``tiles`` fine branch.

    The result is a ``kind=lod`` group with children **coarsest→finest in memory**
    (``[coarse_leaf, fine_partition]``, matching the on-disk order). Each child's
    ``coverage_fraction`` selector threshold is stamped onto its ``meta`` (honored
    by both the standalone writer and the scene graft) via
    ``partitioned_coverage_fractions`` (occupancy halving re-anchored at
    fills-screen): the coarse cap gets a fraction below the fine branch's
    ``PARTITION_FINEST_AREA`` (screen-area 1.0), so the coarse overview shows at
    the opening framing and the fine partition takes over once you zoom the node
    up to filling the viewport. The fills-screen anchor is deliberate here — see
    ``partitioned_coverage_fractions`` for why a partition-bound ladder does not
    take the whole-object half-screen anchor (no per-dataset tuning; see
    ``RecipeParams``).
    """
    from luxar.gsplats.tree import total_splats

    base = data.flattened()
    # The fine branch sits under the coarse cap: sibling-aware part ladders so
    # the cap→partition upgrade catches up at chunk 1-2 per part.
    fine_partition = build_tiles(
        base, params, sibling_compression=params.compression_factor
    )

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
        volume=params.volume,
        volume_axes=params.volume_axes,
        device=params.device,
        seed=params.seed,
        coarsen_dims=params.coarsen_dims,
        quality_stamps=params.quality_stamps,
        quality_max_pair_splats=params.quality_max_pair_splats,
    )
    coarse_leaf = capped.at_substitutive(capped.n_substitutive - 1).flattened().tree
    if params.additive_ladders:
        # Additive ladder inside the coarse cap too (project convention:
        # additive LODs everywhere by default) — the fine branch's parts get
        # theirs from build_tiles.
        coarse_leaf = _ladder_for_part(coarse_leaf, params)
    if params.quality_stamps:
        # `.flattened()` above dropped the capped reduction's level_stats —
        # re-attach the measured Q and the group-consistent reference energy w
        # (the FINEST content's total; see make_substitutive_lod). The additive
        # ladder's fallback w (the cap's OWN energy) must not win here: self-
        # energy is quadratic in amplitude, so it would skew the group's
        # weighted-quality aggregation.
        cap_stats = capped.substitutive_levels[-1].stats
        leaf_stats = coarse_leaf.meta.setdefault("stats", {})
        # Not `reference_energy` on a REVEAL ladder: `_ladder_for_part` above
        # deliberately left the cap with no per-rung `energy_fraction_cum`, and w
        # is the weight for exactly those fractions — re-attaching it here would
        # put back the half-written pair `make_additive_lod` just removed. Q is a
        # standalone readout and still goes on.
        keys = (
            ("quality",)
            if _is_reveal_ladder(coarse_leaf)
            else ("quality", "reference_energy")
        )
        for key in keys:
            if key in cap_stats:
                leaf_stats[key] = cap_stats[key]
        # The fine parts ARE the group's finest content: quality 1.0 by
        # definition (each part's w is its ladder's own total, already stamped
        # by make_additive_lod).
        for leaf in iter_leaves(fine_partition):
            leaf.meta.setdefault("stats", {})["quality"] = 1.0

    # Children are coarsest→finest: [coarse cap, fine partition]. Derive per-child
    # coverage fractions from the whole-subtree splat counts (fine partition's count
    # is the whole partition via ``total_splats``) and stamp into the children's
    # (mutable) meta in place — same pattern as tree_from_substitutive_levels.
    group = GSplatLodGroup(children=[coarse_leaf, fine_partition])
    # PARTITIONED anchor — but by CONTRACT, not geometry (unlike adaptive's
    # per-tile groups). Both children cover the whole dataset, so this group's bbox
    # IS the whole object and its metric reads exactly like a `levels` group's.
    # It is pinned at fills-screen because the fine child is the zoom-in branch:
    # under the whole-object anchor it would be selected at the opening framing,
    # loading the entire dataset on frame 1 and inverting the recipe, whose whole
    # purpose is an instant coarse overview with fine tiles on zoom. So #1361's
    # blur is deliberately RETAINED here — use `levels` for detail immediately.
    coarse_cov, fine_cov = partitioned_coverage_fractions(
        [total_splats(coarse_leaf), total_splats(fine_partition)]
    )
    coarse_leaf.meta["coverage_fraction"] = coarse_cov
    fine_partition.meta["coverage_fraction"] = fine_cov
    # Derived thresholds are screen-area fractions — stamp the units with them.
    group.meta["selector"] = "screen-area"
    return group


def _is_reveal_ladder(node: GSplatNode) -> bool:
    """Was ``node``'s additive ladder built by a REVEAL ordering?

    Read off the ladder's own ``lod_method`` provenance (which a reveal DOES
    carry — only the energy keys are omitted) rather than off
    ``params.additive_method``, which may still be the unresolved ``auto``.
    """
    sublods = getattr(node, "additive_sublods", None) or []
    return any(
        is_reveal_method(str((getattr(sub, "stats", None) or {}).get("lod_method")))
        for sub in sublods
    )


def _ladder_for_part(
    part: GSplatNode,
    params: RecipeParams,
    *,
    sibling_compression: Optional[int] = None,
) -> GSplatNode:
    """Rebuild one partition child as a leaf carrying an additive ladder.

    ``sibling_compression`` is set when the partition sits UNDER a coarser
    sibling in a lod group (the ``overview`` recipe's fine branch beneath its
    coarse cap): each part's stream base is then raised sibling-aware so the
    cap→partition upgrade catches up at chunk 1-2 per part (see
    ``sibling_aware_stream_breakpoints``). Plain ``tiles`` partitions have no
    coarser sibling and pass ``None``.
    """
    part_data = GSplatData.from_tree(part)
    # Clamp ladder depth so a small part never yields empty equal-count bins,
    # and clamp explicit `counts:` breakpoints to THIS part's size — parts have
    # differing N, so a fixed counts list whose largest cut exceeds a small
    # part would otherwise abort the whole build ("largest breakpoint exceeds
    # N"). String/energy specs pass through (already size-adaptive).
    eff_n_lods = max(1, min(params.n_lods, part_data.n_splats))
    eff_breakpoints = clamp_counts_breakpoints(params.breakpoints, part_data.n_splats)
    if sibling_compression is not None:
        eff_breakpoints = sibling_aware_stream_breakpoints(
            eff_breakpoints, part_data.n_splats, sibling_compression
        )
    laddered = make_additive_lod(
        part_data,
        n_lods=eff_n_lods,
        method=params.additive_method,
        breakpoints=eff_breakpoints,
        truncation_sigmas=params.truncation_sigmas,
        max_n_dense=params.max_n_dense,
        seed=params.seed,
        reveal_center=params.reveal_center,
        spatial_dims=params.spatial_dims,
    )
    return laddered.tree


#: Per-part recipes — the recipes that have a single-part form (the building
#: block of ``tiles``/``adaptive``), usable for streaming per-part assembly
#: such as the tiled-batch merge. ``stream`` → a prefix-sum ladder
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


def build_part_lod(
    part: GSplatNode,
    recipe: str,
    params: RecipeParams,
    *,
    cell: Optional[List[Tuple[float, float]]] = None,
) -> GSplatNode:
    """Give ONE partition child its own per-part LOD, with depth clamped to the
    part's splat count (so a small part never synthesises degenerate levels).

    This is the exact building block :func:`build_tiles` (``stream``) and
    :func:`build_adaptive` (``levels``) apply to every part — exposed so a
    streaming assembler (e.g. the tiled-batch merge) can LOD one part at a time
    without materialising the whole partition. Returns the per-part node:
    a leaf-with-ladder (``stream``) or a substitutive ``GSplatLodGroup``
    (``levels``). Legacy spellings translate via :func:`canonical_recipe_name`.

    ``cell`` is this part's own tile, per center dim, and is required by
    ``refine="volume"``: the re-fit crops the volume to the tile so it is not
    tempted to pull splats out of it to explain a neighbour's signal. Callers
    that know the decomposition (the fit-time assembler, the batch merge) should
    pass it; without it a volume re-fit refuses rather than targeting the whole
    volume.
    """
    recipe = canonical_recipe_name(recipe)
    if recipe == "stream":
        return _ladder_for_part(part, params)
    if recipe == "levels":
        return _substitutive_for_part(part, params, cell=cell)
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
