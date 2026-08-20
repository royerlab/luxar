"""Refresh artifact-local LOD stamps after a content-changing rewrite."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from luxar.utils.lod_methods import is_reveal_method

if TYPE_CHECKING:
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
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    if isinstance(fresh, GSplatLeaf) and isinstance(inherited, GSplatLeaf):
        return replace(fresh, meta={**inherited.meta, **fresh.meta})
    if isinstance(fresh, GSplatLodGroup) and isinstance(inherited, GSplatLodGroup):
        children = [
            _restore_node_meta(fresh_child, inherited_child)
            for fresh_child, inherited_child in zip(fresh.children, inherited.children)
        ]
        return replace(fresh, children=children, meta={**inherited.meta, **fresh.meta})
    if isinstance(fresh, GSplatPartition) and isinstance(inherited, GSplatPartition):
        children = [
            _restore_node_meta(fresh_child, inherited_child)
            for fresh_child, inherited_child in zip(fresh.children, inherited.children)
        ]
        return replace(fresh, children=children, meta={**inherited.meta, **fresh.meta})
    return fresh


def refresh_reduction_lod_stats(
    result: "GSplatData", source: "GSplatData"
) -> "GSplatData":
    """Return ``result`` with inherited artifact-local LOD stamps refreshed.

    Only stamp families authored on ``source`` are recomputed; an unannotated
    dataset stays unannotated. The rewritten finest level supplies the
    group-consistent ``reference_energy`` for every substitutive level, and an
    authored quality is remeasured against that same finest content. Source-
    volume ``refine_stats`` cannot be remeasured without the volume, so it is
    removed rather than published for a different splat set.
    """
    source_levels = source.substitutive_levels
    result_levels = result.substitutive_levels
    if (
        not source_levels
        or not result_levels
        or not any(_has_ladder_stamps(level) for level in source_levels)
    ):
        return result

    if len(result_levels) == len(source_levels):
        authored_levels = source_levels
    elif len(result_levels) == 1:
        authored_levels = [source_levels[0]]
    else:
        authored_levels = [
            source_levels[min(index, len(source_levels) - 1)]
            for index in range(len(result_levels))
        ]

    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
    from luxar.gsplats.lod.energy import total_self_energy

    finest = result.at_substitutive(0).flattened()
    reference_energy = total_self_energy(finest)
    quality_authored = any("quality" in level.stats for level in source_levels)
    reference_authored = any(
        "reference_energy" in level.stats for level in source_levels
    )
    total_count_authored = any(
        "n_splats_total" in level.stats for level in source_levels
    )
    new_levels: list[SubstitutiveLevel] = []

    for index, (level, authored) in enumerate(zip(result_levels, authored_levels)):
        level_stats = dict(level.stats)
        level_stats.pop("refine_stats", None)

        energy_authored = any(
            "energy_fraction_cum" in lod.stats for lod in authored.additive_sublods
        )
        rung_counts_authored = any(
            "lod_n_splats" in lod.stats or "lod_cumulative_n" in lod.stats
            for lod in authored.additive_sublods
        )
        ladder_authored = energy_authored or reference_authored

        if total_count_authored or ladder_authored or quality_authored:
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
                    lod_stats["energy_fraction_cum"] = min(
                        1.0, max(0.0, float(fraction))
                    )
                else:
                    lod_stats.pop("energy_fraction_cum", None)
            elif energy_authored and total_energy == 0.0 and level.n_splats_total == 0:
                lod_stats["energy_fraction_cum"] = 1.0
            else:
                lod_stats.pop("energy_fraction_cum", None)
            new_lods.append(
                AdditiveSubLOD(
                    centers=lod.centers,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod.cholesky_factors,
                    colors=lod.colors,
                    stats=lod_stats,
                    truncation_radius=lod.truncation_radius,
                )
            )

        if reference_authored:
            if reveal:
                level_stats.pop("reference_energy", None)
            else:
                level_stats["reference_energy"] = float(reference_energy)

        if quality_authored:
            if index == 0:
                level_stats["quality"] = 1.0
            else:
                try:
                    from luxar.gsplats.lod.quality import mixture_quality

                    level_stats["quality"] = mixture_quality(
                        result.at_substitutive(index).flattened(), finest
                    ).quality
                except Exception as exc:
                    level_stats.pop("quality", None)
                    warnings.warn(
                        f"could not recompute LOD quality after rewrite: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )

        new_levels.append(
            SubstitutiveLevel(
                additive_sublods=new_lods,
                compression_factor=level.compression_factor,
                parent_method=level.parent_method,
                level_index=level.level_index,
                stats=level_stats,
            )
        )

    rebuilt = GSplatData.from_substitutive_levels(new_levels, stats=result.stats)
    return GSplatData.from_tree(
        _restore_node_meta(rebuilt.tree, result.tree), stats=result.stats
    )


def refresh_reduction_lod_tree(
    result: "GSplatNode", source: "GSplatNode"
) -> "GSplatNode":
    """Tree counterpart of :func:`refresh_reduction_lod_stats`.

    Tree-aware CLI rewrites map partition leaves independently. A nested LOD
    group still needs its children measured against one shared finest child, so
    the group pass happens after the leaf pass and overwrites the temporary
    leaf-local quality/weight with the group-consistent values.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.lod.annotate import _node_content
    from luxar.gsplats.lod.energy import total_self_energy
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    if isinstance(result, GSplatLeaf) and isinstance(source, GSplatLeaf):
        refreshed = refresh_reduction_lod_stats(
            GSplatData.from_tree(result), GSplatData.from_tree(source)
        )
        return _restore_node_meta(refreshed.tree, result)

    if isinstance(result, GSplatPartition) and isinstance(source, GSplatPartition):
        children = [
            refresh_reduction_lod_tree(result_child, source_child)
            for result_child, source_child in zip(result.children, source.children)
        ]
        return replace(result, children=children)

    if isinstance(result, GSplatLodGroup) and isinstance(source, GSplatLodGroup):
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
        quality_authored = any("quality" in stats for stats in source_stats)
        reference_authored = any("reference_energy" in stats for stats in source_stats)
        total_count_authored = any("n_splats_total" in stats for stats in source_stats)
        finest = _node_content(children[-1])
        reference_energy = total_self_energy(finest)
        stamped_children: list[GSplatNode] = []
        for index, child in enumerate(children):
            stats = dict(child.meta.get("stats") or {})
            stats.pop("refine_stats", None)
            content = _node_content(child)
            if total_count_authored or quality_authored or reference_authored:
                stats["n_splats_total"] = int(content.n_splats)
            if reference_authored:
                stats["reference_energy"] = float(reference_energy)
            if quality_authored:
                if index == len(children) - 1:
                    stats["quality"] = 1.0
                else:
                    try:
                        from luxar.gsplats.lod.quality import mixture_quality

                        stats["quality"] = mixture_quality(content, finest).quality
                    except Exception as exc:
                        stats.pop("quality", None)
                        warnings.warn(
                            f"could not recompute LOD quality after rewrite: {exc}",
                            UserWarning,
                            stacklevel=2,
                        )
            stamped_children.append(replace(child, meta={**child.meta, "stats": stats}))
        return replace(result, children=stamped_children)

    return result
