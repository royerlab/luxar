"""Refresh artifact-local LOD stamps after a rewrite.

Two scopes, both #1600: the per-level / per-rung stamps a content-changing
rewrite invalidates (:func:`refresh_reduction_lod_stats` and its tree
counterpart), and the ROOT ladder summary a structure-PRESERVING re-ladder
leaves describing the ladder it just replaced
(:func:`refresh_root_ladder_summary`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import numpy as np

from luxar.gsplats._data.filtering import (
    _REDUCTION_LOD_LEVEL_STATS_KEYS,
    _REDUCTION_LOD_RUNG_STATS_KEYS,
)
from luxar.utils.lod_methods import is_reveal_method

if TYPE_CHECKING:
    from luxar.gsplats._data.base import _GSplatDataOps
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
    from luxar.gsplats.tree import GSplatLeaf, GSplatNode


#: The root ``pipeline/`` block that summarises ONE ladder, exactly as
#: :func:`~luxar.gsplats.lod.additive.make_additive_lod` stamps it: four keys
#: describing a ladder, plus ``lod_substitutive_level`` naming WHICH substitutive
#: level they describe.
_ROOT_LADDER_SUMMARY_KEYS = (
    "lod_method",
    "lod_n_lods",
    "lod_breakpoints_kind",
    "lod_cutpoints",
    "lod_substitutive_level",
)

#: The SAME ladder summary, one spelling over: the un-prefixed knobs
#: :func:`~luxar.gsplats.batch.merge_orchestrator._recipe_pipeline_info` stamps
#: for a ``batch-fit merge --recipe stream``. ``gsplat additive`` over a
#: batch-fit partition is an advertised use case, so these rode through
#: describing the ladder the re-ladder had just replaced, exactly as the
#: ``lod_*`` five did (#1600) — and ``_STRUCTURE_SCOPED_STATS_KEYS`` already
#: classifies both groups as one family.
#:
#: Only meaningful when ``lod_kind == "additive"``: the ``levels`` branch of the
#: same producer stamps a bare ``method`` too, but it is the SUBSTITUTIVE merge
#: method, which a re-ladder does not touch. See :func:`_recipe_ladder_keys`.
#:
#: ``per_part`` is deliberately absent: ``additive`` ladders every leaf
#: independently, so a per-part ladder is still exactly what the store has.
_RECIPE_LADDER_SUMMARY_KEYS = ("n_lods", "method", "breakpoints")


def _root_summary_leaf(
    node: "GSplatNode", stats: "Mapping[str, Any]"
) -> "Optional[GSplatLeaf]":
    """The single leaf the root ladder summary describes, or ``None``.

    The rule is the producers': a bare leaf summarises itself, and a
    substitutive ``kind=lod`` group summarises the level named by
    ``lod_substitutive_level`` — the same summary-level choice
    :meth:`~luxar.gsplats._data.transforms.TransformsMixin._map_substitutive`
    makes (clamped into range for a store whose index no longer fits). That
    index is a MATRIX index (``GSplatData`` orders levels finest-first) while a
    tree lod group stores its children coarsest-first, hence the mirror.

    ``None`` for every other shape, and it means "no single root ladder exists":
    a ``kind=partition``'s parts hold different splat counts and therefore
    different rung counts, so any number published at the root would be true of
    at most one part. Same for a lod group whose summary child is itself a
    subtree (the ``overview`` cap over a partition).
    """
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    if isinstance(node, GSplatLeaf):
        return node
    if isinstance(node, GSplatLodGroup) and node.children:
        index = int(stats.get("lod_substitutive_level") or 0)
        index = min(max(index, 0), len(node.children) - 1)
        child = node.children[len(node.children) - 1 - index]
        return child if isinstance(child, GSplatLeaf) else None
    return None


def _recipe_ladder_keys(stats: "Mapping[str, Any]") -> "list[str]":
    """The present :data:`_RECIPE_LADDER_SUMMARY_KEYS`, or ``[]``.

    Gated on ``lod_kind``, which is what disambiguates the shared ``method``
    key: ``_recipe_pipeline_info`` writes ``lod_kind="additive"`` exactly for its
    ``stream`` branch (where ``method`` is the ADDITIVE ordering, replaced by a
    re-ladder) and ``"substitutive"`` for ``levels`` (where it is the merge
    method, which a re-ladder leaves true). ``n_lods`` / ``breakpoints`` are
    stamped by the ``stream`` branch alone, so the gate only ever costs a
    hypothetical mislabelled store its refresh — never a false one.
    """
    if stats.get("lod_kind") != "additive":
        return []
    return [key for key in _RECIPE_LADDER_SUMMARY_KEYS if key in stats]


def refresh_root_ladder_summary(
    stats: "Mapping[str, Any]", node: "GSplatNode"
) -> "Dict[str, Any]":
    """A copy of ``stats`` whose root ladder summary describes ``node``.

    For a rewrite that REBUILDS every leaf's additive ladder while preserving the
    structure kind — ``gsplat additive``. The leaves get fresh stats from
    :func:`~luxar.gsplats.lod.additive.make_additive_lod`, but the root block is
    threaded through from the input, so a store re-laddered from four rungs to
    six went on advertising the four it no longer had (#1600).

    PRESENT KEYS ONLY, matching
    :func:`~luxar.gsplats._data.transforms._refresh_ladder_summary` (whose
    count/cutpoint half this reuses): a store that never published a ladder
    summary does not acquire one here. ``lod_method`` and
    ``lod_breakpoints_kind`` are read back off the leaf the rebuild wrote rather
    than from the request, so ``auto`` publishes the method it resolved to — and
    when the rebuilt leaf does not publish it, the root key is DELETED rather than
    left describing the ladder that is gone, exactly as
    :func:`_refresh_recipe_ladder` treats its ``method`` twin. Absence is the
    format's "this artifact does not know"; keeping the inherited value is the
    one thing this function exists to prevent.
    ``lod_substitutive_level`` is left alone — a re-ladder moves no level.

    When no single leaf is the summary (see :func:`_root_summary_leaf`) the whole
    block is DROPPED rather than refreshed from an arbitrary part: an absent
    summary is what the ``tiles`` / ``overview`` / ``adaptive`` builders publish
    for exactly that reason, and a wrong number is worse than no number.

    Both spellings of the summary are covered — the five ``lod_*`` keys
    ``make_additive_lod`` stamps, and :data:`_RECIPE_LADDER_SUMMARY_KEYS`, the
    un-prefixed trio ``batch-fit merge --recipe stream`` stamps for the same
    ladder. Otherwise a re-laddered batch-fit partition dropped ``lod_n_lods``
    (because parts hold different rung counts) while leaving ``n_lods: 6``
    asserting exactly that number two keys away.
    """
    refreshed = dict(stats)
    present = [key for key in _ROOT_LADDER_SUMMARY_KEYS if key in refreshed]
    recipe_present = _recipe_ladder_keys(refreshed)
    if not present and not recipe_present:
        return refreshed

    leaf = _root_summary_leaf(node, refreshed)
    if leaf is None:
        for key in (*present, *recipe_present):
            del refreshed[key]
        return refreshed

    from luxar.gsplats._data.transforms import _refresh_ladder_summary

    cutpoints = [
        int(value)
        for value in np.cumsum([lod.n_splats for lod in leaf.additive_sublods])
    ]
    refreshed = _refresh_ladder_summary(refreshed, cutpoints)
    leaf_stats = leaf.meta.get("stats")
    leaf_stats = leaf_stats if isinstance(leaf_stats, dict) else {}
    for key in ("lod_method", "lod_breakpoints_kind"):
        if key not in refreshed:
            continue
        if key in leaf_stats:
            refreshed[key] = leaf_stats[key]
        else:
            del refreshed[key]
    if recipe_present:
        _refresh_recipe_ladder(refreshed, leaf_stats, len(cutpoints))
    return refreshed


def _refresh_recipe_ladder(
    refreshed: "Dict[str, Any]", leaf_stats: "Mapping[str, Any]", n_rungs: int
) -> None:
    """Refresh the un-prefixed ladder trio in place (present keys only).

    ``n_lods`` and ``method`` are recoverable from the tree that was written —
    the rung count, and the resolved ordering the rebuilt leaf published (so an
    ``auto`` request records what it became, as the ``lod_*`` half already does).
    The count is restated here rather than left to
    :func:`~luxar.gsplats._data.transforms._refresh_ladder_summary` (which the
    caller runs first, and which now refreshes it too, for the prune-family
    rewrites that never reach this function) so the trio's three rules read in
    one place; the two agree by construction — both are the rung count.

    ``breakpoints`` is DROPPED rather than refreshed: it holds the build SPEC
    (``"stream:14000"``, ``"counts:5,15,40"``, ``"equal-count"``) in a different
    vocabulary from the leaf's resolved ``lod_breakpoints_kind``
    (``"stream"`` / ``"explicit-counts"`` / ``"equal-count"``), and the spec that
    produced a ladder cannot be read back off it. An absent key is the format's
    "this artifact does not know"; writing the kind into a spec field would be a
    new wrong claim rather than a scrubbed one.
    """
    if "n_lods" in refreshed:
        refreshed["n_lods"] = int(n_rungs)
    if "method" in refreshed:
        if "lod_method" in leaf_stats:
            refreshed["method"] = leaf_stats["lod_method"]
        else:
            del refreshed["method"]
    refreshed.pop("breakpoints", None)


def _has_ladder_stamps(level: "SubstitutiveLevel") -> bool:
    if any(key in level.stats for key in _REDUCTION_LOD_LEVEL_STATS_KEYS):
        return True
    return any(
        any(key in lod.stats for key in _REDUCTION_LOD_RUNG_STATS_KEYS)
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
    stats.pop("median_footprint", None)
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
        stats.pop("median_footprint", None)
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
