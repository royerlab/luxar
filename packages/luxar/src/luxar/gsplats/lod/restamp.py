"""Refresh artifact-local LOD stamps after a content-changing rewrite."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from luxar.utils.lod_methods import is_reveal_method

if TYPE_CHECKING:
    from luxar.gsplats._data.base import _GSplatDataOps
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
    from luxar.gsplats.tree import GSplatNode


def _has_ladder_stamps(level: "SubstitutiveLevel") -> bool:
    if any(
        key in level.stats
        for key in ("quality", "reference_energy", "n_splats_total", "refine_stats")
    ):
        return True
    return any(
        any(
            key in lod.stats
            for key in ("energy_fraction_cum", "lod_n_splats", "lod_cumulative_n")
        )
        for lod in level.additive_sublods
    )


def _is_reveal(level: "SubstitutiveLevel") -> bool:
    return any(
        is_reveal_method(str(lod.stats.get("lod_method")))
        for lod in level.additive_sublods
    )


def _lod_energy(lod: "AdditiveSubLOD") -> float:
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.lod.energy import total_self_energy

    return total_self_energy(GSplatData.from_additive_sublods([lod]))


def _restore_node_meta(fresh: "GSplatNode", inherited: "GSplatNode") -> "GSplatNode":
    """Overlay refreshed metadata without dropping unrelated node attributes."""
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    if isinstance(fresh, GSplatLeaf) and isinstance(inherited, GSplatLeaf):
        return replace(fresh, meta={**inherited.meta, **fresh.meta})
    if isinstance(fresh, GSplatLodGroup) and isinstance(inherited, GSplatLodGroup):
        children = [
            _restore_node_meta(fresh_child, inherited_child)
            for fresh_child, inherited_child in zip(fresh.children, inherited.children)
        ]
        return replace(fresh, children=children, meta={**inherited.meta, **fresh.meta})
    return fresh


def _authored_level_for(
    source_levels: "list[SubstitutiveLevel]",
    result_count: int,
    index: int,
) -> "SubstitutiveLevel":
    if result_count == len(source_levels):
        return source_levels[index]
    if result_count == 1:
        return source_levels[0]
    return source_levels[min(index, len(source_levels) - 1)]


def _refreshed_level_stats(
    level: "SubstitutiveLevel", authored: "SubstitutiveLevel"
) -> dict:
    """Refresh authored structural level stamps before energy restamping."""
    stats = dict(level.stats)
    stats.pop("quality", None)
    stats.pop("refine_stats", None)
    if "lod_n_lods" in authored.stats:
        stats["lod_n_lods"] = len(level.additive_sublods)
    if "lod_cutpoints" in authored.stats:
        stats["lod_cutpoints"] = [
            int(value)
            for value in np.cumsum([lod.n_splats for lod in level.additive_sublods])
        ]
    else:
        stats.pop("lod_cutpoints", None)
    return stats


def _refresh_level(
    level: "SubstitutiveLevel",
    authored: "SubstitutiveLevel",
    *,
    reference_energy: float,
    reference_authored: bool,
    total_count_authored: bool,
) -> "SubstitutiveLevel":
    from luxar.gsplats.gsplat_data import SubstitutiveLevel

    energy_authored = any(
        "energy_fraction_cum" in lod.stats for lod in authored.additive_sublods
    )
    rung_counts_authored = any(
        "lod_n_splats" in lod.stats or "lod_cumulative_n" in lod.stats
        for lod in authored.additive_sublods
    )
    ladder_authored = energy_authored or reference_authored
    level_stats = _refreshed_level_stats(level, authored)
    if total_count_authored or ladder_authored:
        level_stats["n_splats_total"] = int(level.n_splats_total)

    energies = [_lod_energy(lod) for lod in level.additive_sublods]
    total_energy = float(sum(energies))
    cumulative_n = 0
    cumulative_energy = 0.0
    reveal = _is_reveal(authored)
    new_lods: list[AdditiveSubLOD] = []
    for lod, energy in zip(level.additive_sublods, energies):
        lod_stats = dict(lod.stats)
        cumulative_n += int(lod.n_splats)
        cumulative_energy += energy
        if rung_counts_authored or ladder_authored:
            lod_stats["lod_n_splats"] = int(lod.n_splats)
            lod_stats["lod_cumulative_n"] = cumulative_n
        if energy_authored and not reveal and total_energy > 0.0:
            fraction = cumulative_energy / total_energy
            if np.isfinite(fraction):
                lod_stats["energy_fraction_cum"] = min(1.0, max(0.0, float(fraction)))
            else:
                lod_stats.pop("energy_fraction_cum", None)
        elif energy_authored and total_energy == 0.0 and level.n_splats_total == 0:
            lod_stats["energy_fraction_cum"] = 1.0
        else:
            lod_stats.pop("energy_fraction_cum", None)
        new_lods.append(replace(lod, stats=lod_stats))

    if reference_authored:
        if reveal:
            level_stats.pop("reference_energy", None)
        else:
            level_stats["reference_energy"] = reference_energy

    return SubstitutiveLevel(
        additive_sublods=new_lods,
        compression_factor=level.compression_factor,
        parent_method=level.parent_method,
        level_index=level.level_index,
        stats=level_stats,
    )


def refresh_reduction_lod_stats(
    result: "GSplatData", source: "_GSplatDataOps"
) -> "GSplatData":
    """Return ``result`` with inherited artifact-local LOD stamps refreshed.

    Only stamp families authored on ``source`` are recomputed; an unannotated
    dataset stays unannotated. The rewritten finest level supplies the
    group-consistent ``reference_energy`` for every substitutive level, and an
    authored quality is dropped until ``annotate-quality --with-quality``
    remeasures it. Source-volume ``refine_stats`` cannot be remeasured without
    the volume, so it is removed rather than published for a different splat set.
    """
    source_levels = source.substitutive_levels
    result_levels = result.substitutive_levels
    if (
        not source_levels
        or not result_levels
        or not any(_has_ladder_stamps(level) for level in source_levels)
    ):
        return result

    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.lod.energy import total_self_energy

    # GSplatData orders substitutive levels finest-first.
    finest = result.at_substitutive(0).flattened()
    reference_energy = total_self_energy(finest)
    if not np.isfinite(reference_energy):
        reference_energy = 0.0
    reference_authored = any(
        "reference_energy" in level.stats for level in source_levels
    )
    total_count_authored = any(
        "n_splats_total" in level.stats for level in source_levels
    )
    new_levels = [
        _refresh_level(
            level,
            _authored_level_for(source_levels, len(result_levels), index),
            reference_energy=reference_energy,
            reference_authored=reference_authored,
            total_count_authored=total_count_authored,
        )
        for index, level in enumerate(result_levels)
    ]

    rebuilt = GSplatData.from_substitutive_levels(new_levels, stats=dict(result.stats))
    return GSplatData.from_tree(
        _restore_node_meta(rebuilt.tree, result.tree), stats=dict(result.stats)
    )


def refresh_reduction_lod_tree(
    result: "GSplatNode", source: "GSplatNode"
) -> "GSplatNode":
    """Tree counterpart of :func:`refresh_reduction_lod_stats`.

    Tree-aware CLI rewrites map partition leaves independently. A nested LOD
    group still needs its leaf children weighted against one shared finest
    child, so the group pass happens after the leaf pass and overwrites each
    leaf-local weight with the group-consistent value.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    if isinstance(result, GSplatLeaf) and isinstance(source, GSplatLeaf):
        refreshed = refresh_reduction_lod_stats(
            GSplatData.from_tree(result), GSplatData.from_tree(source)
        )
        return refreshed.tree

    if isinstance(result, GSplatPartition) and isinstance(source, GSplatPartition):
        children = [
            refresh_reduction_lod_tree(result_child, source_child)
            for result_child, source_child in zip(result.children, source.children)
        ]
        return replace(result, children=children)

    if isinstance(result, GSplatLodGroup) and isinstance(source, GSplatLodGroup):
        return _refresh_lod_group(result, source)

    return result


def _refresh_lod_group(result: "GSplatNode", source: "GSplatNode") -> "GSplatNode":
    from luxar.gsplats.lod.annotate import _node_content
    from luxar.gsplats.lod.energy import total_self_energy
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    assert isinstance(result, GSplatLodGroup)
    assert isinstance(source, GSplatLodGroup)
    children = [
        refresh_reduction_lod_tree(result_child, source_child)
        for result_child, source_child in zip(result.children, source.children)
    ]
    if not children:
        return replace(result, children=children)
    source_stats = [
        child.meta.get("stats", {})
        if isinstance(child.meta.get("stats", {}), dict)
        else {}
        for child in source.children
    ]
    reference_authored = any("reference_energy" in stats for stats in source_stats)
    total_count_authored = any("n_splats_total" in stats for stats in source_stats)
    reference_energy = 0.0
    if reference_authored:
        # Tree LOD groups order children coarse-first and finest-last.
        reference_energy = total_self_energy(_node_content(children[-1]))
        if not np.isfinite(reference_energy):
            reference_energy = 0.0
    stamped_children: list[GSplatNode] = []
    for child in children:
        if not isinstance(child, GSplatLeaf):
            stamped_children.append(child)
            continue
        stats = dict(child.meta.get("stats") or {})
        stats.pop("quality", None)
        stats.pop("refine_stats", None)
        if total_count_authored or reference_authored:
            stats["n_splats_total"] = sum(
                lod.n_splats for lod in child.additive_sublods
            )
        if reference_authored:
            stats["reference_energy"] = reference_energy
        stamped_children.append(replace(child, meta={**child.meta, "stats": stats}))
    return replace(result, children=stamped_children)
