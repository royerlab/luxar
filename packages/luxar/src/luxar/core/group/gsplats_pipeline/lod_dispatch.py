"""LOD-axis dispatch for `add_gsplats_from_data`.

Two free functions:

* :func:`add_gsplats_as_lod_group_impl` — build a kind=lod Group with
  one gsplats child per substitutive level.
* :func:`add_gsplats_multi_lod_impl` — write a multi-additive-LOD
  GSplats node (per-LOD subgroups under one parent gsplats node).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import numpy as np
from arbol import aprint

from ....typing_utils.json_safe import json_safe_value
from ...gsplats import GSplats
from ..compositing import (
    COMPOSITING_ATTRS,
    mirror_written_colormap,
    sync_custom_colormap_attr,
)
from ..dim_order import apply_dim_order_cholesky, apply_dim_order_positions

if TYPE_CHECKING:
    from ....gsplats.gsplat_data import GSplatData
    from ...node import Node
    from ..group import Group


def _remap_footprint_stats(
    stats: Dict[str, Any], dim_order: Optional[List[str]], scene_names: List[str]
) -> Dict[str, Any]:
    """Map footprint columns from input data order to stored scene order."""
    remapped = dict(stats)
    footprint_dims = remapped.get("footprint_dims")
    if dim_order is None or not isinstance(footprint_dims, list):
        return remapped
    try:
        remapped["footprint_dims"] = [
            scene_names.index(dim_order[column]) for column in footprint_dims
        ]
    except (IndexError, TypeError, ValueError):
        remapped.pop("median_footprint", None)
        remapped.pop("footprint_dims", None)
    return remapped


def add_gsplats_as_lod_group_impl(
    group: "Group",
    *,
    name: str,
    result: "GSplatData",
    explicit_coverage_fractions: Optional[List[float]],
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    **attrs: Any,
) -> "Group":
    """Build a kind=lod ``Group`` with one gsplats child per substitutive level.

    Children are written in coarsest→finest order and named
    ``child_<i>``. Compositing attrs (opacity, gamma, intensity,
    offset, blending_mode, transform, layer, visible, nd_transform)
    land on the kind=lod ``Group`` itself; per-leaf gsplats attrs
    (truncation_radius, extend_to_all, colormap) ride into each child.
    """
    from ..lod.group import resolve_lod_ladder

    # Substitutive convention: index 0 = finest, n-1 = coarsest. The
    # LOD group needs coarsest first.
    n_sub = result.n_substitutive
    order = list(range(n_sub - 1, -1, -1))
    parent_node = parent or group

    # Per-level total splat count (sum across each level's additive
    # ladder) — used both for logging and for auto-deriving
    # coverage_fractions when the user didn't supply them.
    splat_counts: List[int] = [
        sum(sub.n_splats for sub in result.substitutive_levels[s].additive_sublods)
        for s in order
    ]

    # Thresholds AND the selector naming their units, from the one shared rule
    # (``lod.group.resolve_lod_ladder``): an explicit ``coverage_fractions=[...]``
    # is used verbatim under the legacy units it was authored in, otherwise the
    # screen-area halving ladder is derived (coarsest-first) — re-anchored at
    # fills-screen when the insertion point sits under a hand-built
    # ``kind=partition``, since such a ladder switches on ONE TILE.
    coverage_vals, lod_selector = resolve_lod_ladder(
        explicit_coverage_fractions,
        splat_counts,
        parent_node,
        name=name,
        length_error=lambda n_explicit, n_levels: (
            f"coverage_fractions has {n_explicit} "
            f"entries but the lod_group has {n_levels} substitutive levels"
        ),
    )

    # Separate compositing attrs (go on the kind=lod Group) from
    # per-leaf gsplats attrs (go on each child). Anything not in the
    # compositing set falls through to the child level.
    #
    # ``colormap`` is not in COMPOSITING_ATTRS, so it falls through to each
    # child. Since #1600 that is a routing preference rather than a
    # correctness requirement (the writer no longer manufactures a shadowing
    # "gray" and the viewer composes the attr root→leaf) — but a palette on
    # every child is what the Layers panel's `deriveColormapFromDescendants`
    # reads back, so the shape stays.
    lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    # Defense in depth: ``add_gsplats_from_data`` rejects coverage_fraction
    # at entry, but if this method is invoked through a different path
    # (e.g. internal recursion) we still need to strip it — the loop
    # below passes a derived value as an explicit kwarg, and a duplicate
    # in ``**child_attrs`` would raise TypeError.
    child_attrs.pop("coverage_fraction", None)

    # All children in this branch are gsplats leaves, so the user-facing
    # ``display_type`` is unambiguously "gsplats". Set it here so the
    # on-disk attrs are self-describing.
    lod_attrs.setdefault("display_type", "gsplats")
    # default_level (the viewer's INITIAL level, before the coverage selector
    # runs) is left at add_lod_group's default of 0 = the COARSEST child
    # (child_0). This is a progressive-load hint: the scene appears instantly at
    # low detail, then refines. It is intentionally decoupled from the data-model
    # default_substitutive (the finest level the accessors return) — defaulting
    # the viewer to the finest would eager-load every lod group at full
    # resolution and render "backwards" (matches gsplat_tree.write_gsplat_node).
    lod_group_node = parent_node.add_lod_group(name, selector=lod_selector, **lod_attrs)

    aprint(
        f"Adding multi-resolution gsplats node '{name}' as kind=lod "
        f"Group with {n_sub} substitutive levels: "
        f"{[f'{n:,}' for n in splat_counts]} splats"
    )

    # Add one gsplats child per substitutive level, coarsest first.
    for child_idx, s in enumerate(order):
        child_name = f"child_{child_idx}"
        level_view = result.at_substitutive(s)
        level_attrs = dict(child_attrs)
        level_stats = _remap_footprint_stats(
            result.substitutive_levels[s].stats,
            dim_order,
            group._find_scene()._dimensions.names,
        )
        _, safe_level_stats = json_safe_value(level_stats)
        _, safe_caller_stats = json_safe_value(level_attrs.get("level_stats") or {})
        merged_level_stats = {
            **(safe_level_stats or {}),
            **(safe_caller_stats or {}),
        }
        if merged_level_stats:
            level_attrs["level_stats"] = merged_level_stats
        # Recursive dispatch — but explicitly None on both LOD axes so
        # the resolvers no-op and we never re-enter the kind=lod branch.
        lod_group_node.add_gsplats_from_data(
            name=child_name,
            result=level_view,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            lod_group=None,
            additive_lod=None,
            coverage_fraction=coverage_vals[child_idx],
            # The parent already normalised the WHOLE pyramid with one factor.
            # Letting each level normalise itself here is not merely redundant:
            # a coarser level's amplitudes are larger (merged representatives
            # carry combined mass), so its own p99.9 is higher and it would be
            # scaled DOWN relative to its siblings. Measured before this was
            # pinned: factors 0.1266 / 0.2549 / 0.0594 across three levels of
            # one node, i.e. the levels rescaled against each other and the
            # brightness pops at every LOD switch.
            normalize_amplitudes=False,
            **level_attrs,
        )

    return lod_group_node


def add_gsplats_multi_lod_impl(
    group: "Group",
    *,
    name: str,
    result: "GSplatData",
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    **attrs: Any,
) -> GSplats:
    """Write multi-additive-LOD GSplatData as a single leaf via the shared walker.

    Builds a :class:`~luxar.gsplats.tree.GSplatLeaf` (additive ladder) — applying
    ``dim_order`` per sub-LOD as a tree transform — and hands it to
    :meth:`LuxarZarrCompiler.write_gsplat_leaf_subtree`, the single authoring path
    shared with the standalone ``.gsplats.zarr`` writer. There is no parallel
    additive-ladder writer.
    """
    try:
        from ....gsplats.gsplat_data import AdditiveSubLOD
        from ....gsplats.tree import GSplatLeaf

        scene = group._find_scene()
        d_data = result.ndim

        # Build per-sub-LOD AdditiveSubLODs, applying dim_order as a tree
        # transform on each (centers + Cholesky reshaped/padded into scene dims).
        sublods: List[AdditiveSubLOD] = []
        for lod in result.additive_sublods:
            ctr_arr = lod.centers.copy()
            chol_arr = lod.cholesky_factors.copy()

            if dim_order is not None:
                ctr_arr, extend_to_all = apply_dim_order_positions(
                    ctr_arr, scene, dim_order, fill, extend_to_all
                )
                chol_arr = apply_dim_order_cholesky(
                    chol_arr, d_data, scene, dim_order, fill_sigma
                )

            scene._validate_data_dimensions(ctr_arr, name, data_type="centers")

            sublods.append(
                AdditiveSubLOD(
                    centers=ctr_arr.astype(np.float32),
                    amplitudes=lod.amplitudes,
                    cholesky_factors=chol_arr,
                    colors=lod.colors,
                    label_ids=lod.label_ids,
                    label_vocabulary=lod.label_vocabulary,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )

        leaf = GSplatLeaf(additive_sublods=sublods)

        # Pairing: the additive sub-LODs already carry per-level
        # ``energy_fraction_cum`` stamps (from ``make_additive_lod``), so the
        # leaf MUST ride the paired absolute ``reference_energy`` the resolver
        # stamped onto the substitutive level — the viewer's display-gate needs
        # both halves or it falls back to counting elements. Propagate ONLY what
        # the resolver already computed (do not recompute it) via ``level_stats``,
        # the allowed leaf attr the writer persists onto the parent gsplats group.
        leaf_stats: Dict[str, Any] = {}
        if result.n_substitutive >= 1:
            # Route each copied value through the SAME JSON-safety guard the
            # mirror writer ``_meta_to_node_attrs`` uses: a non-finite float
            # (NaN/±Inf) in ``level_stats`` would land in .zattrs as a bare
            # NaN/Infinity token the viewer's strict JSON.parse rejects. Skip
            # any key whose value is not strictly JSON-safe (do not fabricate).
            src_stats = result.substitutive_levels[0].stats
            for key in ("reference_energy", "quality", "energy_kind"):
                if key in src_stats:
                    ok, safe = json_safe_value(src_stats[key])
                    if ok:
                        leaf_stats[key] = safe
        if leaf_stats:
            caller_level_stats = attrs.get("level_stats")
            if caller_level_stats is None:
                attrs["level_stats"] = leaf_stats
            else:
                # Caller keys win; fill only the missing half of the pairing.
                # The caller dict rides through the same JSON-safety guard as
                # the copied values: a non-finite caller float (NaN/±Inf) is
                # dropped — the resolver's finite value then shows through —
                # instead of reaching .zattrs as a bare NaN token.
                _, safe_caller = json_safe_value(caller_level_stats)
                attrs["level_stats"] = {**leaf_stats, **(safe_caller or {})}

        n_splats = result.n_splats
        ndim = sublods[0].centers.shape[1]
        aprint(
            f"Adding multi-LOD gsplats node '{name}' with "
            f"{n_splats:,} splats in {ndim}D ({result.n_additive_sublods} LODs)."
        )

        final_extend_dims = scene._resolve_extend_to_all(
            extend_to_all, sublods[0].centers, "splats"
        )
        if final_extend_dims:
            attrs["extend_to_all"] = final_extend_dims
            aprint(f"  📡 Extending visibility across: {final_extend_dims}")

        colormap = attrs.get("colormap")
        if any(s.colors is not None for s in sublods) and colormap is not None:
            raise ValueError(
                "Cannot specify both 'colors' and 'colormap'. Use one or the other."
            )

        parent_node = parent or group
        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name

        metadata = writer.write_gsplat_leaf_subtree(  # type: ignore[attr-defined]
            path,
            leaf,
            **attrs,
        )

        sync_custom_colormap_attr(attrs)

        # Mirror the writer's own colormap decision (see
        # mirror_written_colormap): no manufactured gray under an
        # ancestor-authored palette.
        mirror_written_colormap(attrs, writer, path)

        return GSplats(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )
    except (ValueError, TypeError) as e:
        aprint(f"Failed to add multi-LOD gsplats node '{name}': {e}")
        raise ValueError(f"Could not add gsplats '{name}': {e}") from e
