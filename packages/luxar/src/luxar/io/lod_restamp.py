"""Re-derive a stored LOD ladder's thresholds under the current selector.

Every ``kind=lod`` group on disk writes two things that are really one decision:
the per-child ``coverage_fraction`` thresholds, and the group-level ``selector``
naming the UNITS they are in. The whole shipped demo corpus predates the
``"screen-area"`` selector, so its ladders carry ``selector="coverage"`` — the
legacy diagonal metric — over thresholds that were *derived* rather than
authored (issue #1727). Re-deriving them under the current metric is a pure
ATTRIBUTE rewrite: the pass itself moves no chunk and opens no array, so a store
already on disk can be upgraded in place instead of regenerated.

**One part of a real run is not free, and it is not the attrs pass.** When
something actually changed, the store's ``content_hash`` is restamped so a warm
viewer cache invalidates — and for a compiled SCENE that digest is over array
VALUES, so the restamp streams every array in the store exactly once (linear in
total store size: a 100 GB scene reads 100 GB to change two attrs). A standalone
``.gsplats.zarr`` takes the other branch, a metadata-only root stamp, and stays
cheap. A ``--dry-run``, and a run that finds nothing to change, hash nothing and
so read nothing.

**Why this is not part of ``luxar optimize``.** That pass documents "every
attribute is preserved" and refuses same-path work outright; this one changes
attributes and nothing else, in place. They are siblings, not one command with a
flag.

**Why it is never automatic.** An authored ``coverage_fractions=[...]`` list and
a derived one are INDISTINGUISHABLE on disk, including a hand-authored ladder
already stamped ``screen-area`` — the point
``io/_compiler/finalize/lod_backfill.py::warn_one_part_partition_anchors``
makes normatively, which is why that check only ever warns. Running this command
IS the opt-in: nothing else may trigger it, and it prints the old→new ladder for
every group it rewrites so the audit trail exists even when the rewrite is
overriding a deliberate choice.

**Which anchor a ladder gets is decided from the STORE**, mirroring the tree
writers' rule (``io/_compiler/gsplat_tree.py::write_gsplat_node`` and the
identical ``core/group/gsplats_pipeline/from_io.py::graft_gsplat_node``) rather
than trusting the stamped values. That rule has TWO clauses and a ladder is
tile-anchored when EITHER holds:

* **ancestry** — some enclosing ``kind=partition`` is a real tiling (>1 part).
  The ``tiles`` and ``adaptive`` per-tile ladders.
* **its own children** — one of THIS lod group's ladder children is itself a
  ``kind=partition``. That is the ``overview`` recipe's
  ``[coarse_leaf, fine_partition]`` cap, which
  :func:`~luxar.core.group.lod.group.partitioned_coverage_fractions` documents as
  a deliberate product contract rather than geometry: the coarse cap is what the
  opening framing shows and the fine branch is the zoom-in branch. Deriving such
  a cap at the whole-object anchor instead would select the fine partition at
  half-screen occupancy and load the WHOLE dataset on frame one — precisely the
  cost the recipe exists to avoid, on the largest stores there are.

The binding a ``kind=lod`` group resolves is threaded down to its own
descendants, exactly as the writers pass ``under_partition=partition_bound`` into
that group's children.

One consequence is deliberate — a store whose one-part partition holds a
tile-anchored ladder, exactly what ``warn_one_part_partition_anchors`` reports
and cannot repair, comes out of this pass re-anchored at whole-object. That is a
rewrite, which is why it happens only here and only when asked for. (The one-part
exclusion belongs to the ANCESTRY clause only, again mirroring the writers: their
own-children test is a bare ``any(isinstance(c, GSplatPartition))``.)

**A failed run leaves the store as it found it.** The pass plans read-only, then
applies; any exception during the apply restores every attr it had already
written — including removing a ``coverage_fraction`` that was absent before, and
putting back each ``content_hash`` the store arrived with rather than recomputing
a digest for it — and re-raises with a note saying so. It re-consolidates only
when it rewrote the ROOT document, since that is the write which destroys a
format-3 index; a rollback that never touched the root leaves the existing index
alone rather than risking a second failure on it. A torn ladder (a screen-area
threshold under ``selector="coverage"``) is the silent-and-unrecoverable failure
:func:`~luxar.core.group.lod.group.resolve_lod_ladder` warns about, and a store
mid-way through this pass would carry one.

**An index is rebuilt, never introduced.** A store that arrives without
consolidated metadata leaves without it, on the success path as on the failure
one: :func:`~luxar._zarr_compat.is_consolidated` is ``batch-fit``'s "this tile
finished" sentinel, so consolidating an interrupted tile here would mark it
complete.

The public entry point is :func:`restamp_lod_store`; the CLI wrapper is
``luxar.cli.restamp_lod_command``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import zarr
from arbol import aprint, asection

from .._zarr_compat import (
    close,
    consolidate,
    group_keys,
    is_consolidated,
    open_group,
    read_consolidated_attrs,
)
from ..core.group.lod.group import (
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
    coverage_fractions,
    partitioned_coverage_fractions,
)
from ..typing_utils.constants import (
    DERIVED_LOD_SELECTOR,
    ENVIRONMENT_GROUP,
    LOD_SELECTORS,
)

# Private imports, deliberately. Both express rules this pass must MIRROR
# EXACTLY rather than re-state: `_lod_children` is the three-tier coarsest→finest
# ordering (child_index, then a `child_<i>` numeric suffix, then sorted name) and
# `_child_nodes` is the "which subgroups are scene nodes" filter that keeps the
# reserved fitting/provenance/pipeline buckets out. A local copy of either would
# be one more place for the ordering convention to drift. (Considered: those
# three names are documented as reserved on a standalone `.gsplats.zarr` ROOT,
# while this pass applies the filter at EVERY depth — a partition part literally
# named `pipeline` would go uncounted. Negligible, and not worth forking the
# helper; a dropped child that carries a `coverage_fraction` is refused below.)
from ._compiler.finalize.amplitude_window import _child_nodes, _lod_children
from .optimize import _is_luxar_store, _restamp_content_hash

__all__ = [
    "HASH_RESTAMPED",
    "HASH_UNCHANGED",
    "HASH_UNSTAMPABLE",
    "RestampReport",
    "RestampedGroup",
    "SkippedGroup",
    "restamp_lod_store",
]


#: The attr naming a leaf's PRIMARY element count, per geometry ``type``. The
#: same "``n=`` slot" convention ``cli/utils.py`` renders a node summary with:
#: lines and mesh are sized by their vertices (their second count — segments,
#: faces — is a different unit, not a second opinion on the same one).
_ELEMENT_COUNT_ATTR: Dict[str, str] = {
    "points": "n_points",
    "lines": "n_vertices",
    "mesh": "n_vertices",
    "gsplats": "n_splats",
}

#: ``RestampReport.content_hash_status`` — what became of the store's
#: ``content_hash``. ``None`` alone cannot say: a store that carries neither the
#: scene ``type`` nor a ``.gsplats.zarr`` ``content_hash`` marker has no digest
#: to move (``optimize._restamp_content_hash`` returns ``None`` for it), and that
#: is a very different report from "nothing changed, so nothing was restamped" —
#: the first means a warm viewer cache will NOT invalidate.
HASH_UNCHANGED = "unchanged"
HASH_RESTAMPED = "restamped"
HASH_UNSTAMPABLE = "unstampable"


def _carries_restampable_digest(root: zarr.Group) -> bool:
    """Mirror :func:`~luxar.io.optimize._restamp_content_hash`'s marker gate."""
    attrs = dict(root.attrs)
    return attrs.get("type") == "scene" or "content_hash" in attrs


@dataclass(frozen=True)
class RestampedGroup:
    """One ``kind=lod`` group whose ladder was (or would be) re-derived."""

    path: str
    partition_bound: bool
    anchor: float
    old_selector: Optional[str]
    old_thresholds: List[Optional[float]]
    new_thresholds: List[float]
    element_counts: List[Optional[int]]

    @property
    def anchor_name(self) -> str:
        """Human name of the anchor binding the re-derivation used."""
        return "fills-screen (tile)" if self.partition_bound else "whole-object"


@dataclass(frozen=True)
class SkippedGroup:
    """One ``kind=lod`` group the pass deliberately left alone."""

    path: str
    reason: str
    detail: str


@dataclass
class RestampReport:
    """What one :func:`restamp_lod_store` pass found and did."""

    path: str
    dry_run: bool
    restamped: List[RestampedGroup] = field(default_factory=list)
    already_current: List[SkippedGroup] = field(default_factory=list)
    unsupported: List[SkippedGroup] = field(default_factory=list)
    unresolved: List[SkippedGroup] = field(default_factory=list)
    content_hash: Optional[str] = None
    content_hash_status: str = HASH_UNCHANGED
    residual: List[str] = field(default_factory=list)
    #: Did the store carry a consolidated index when the run started — and
    #: therefore when it ended? The pass rebuilds an index it found and never
    #: introduces one it did not (``is_consolidated`` is ``batch-fit``'s
    #: finished-tile sentinel), so this says why a store may still have none.
    was_consolidated: bool = True

    @property
    def clean(self) -> bool:
        """Was every ``kind=lod`` group either restamped or already current?

        False when a group was skipped for a reason the caller must act on — an
        out-of-vocabulary selector, an unresolvable finest element count, or a
        re-verification residual. Nothing is ever silently ignored, so the CLI
        keys its exit code on this.

        Also False on :data:`HASH_UNSTAMPABLE`, which only ever happens when
        ladders WERE rewritten: a store carrying neither the scene ``type`` nor a
        ``.gsplats.zarr`` ``content_hash`` (a ``kind=partition`` root, say) has no
        digest to move, and at zarr format 2 — precisely the legacy corpus this
        pass targets — the root ``.zattrs`` bytes the viewer's ``zattrs-hash``
        fallback digests instead do not move either. A warm cache would then serve
        the OLD ladder indefinitely, which is the failure the pass exists to
        prevent, so the run must not report success: the operator has to
        republish under a new URL prefix.
        """
        return not (
            self.unsupported
            or self.unresolved
            or self.residual
            or self.content_hash_status == HASH_UNSTAMPABLE
        )


def _count_of(group: "zarr.Group", attrs: Dict[str, Any]) -> Optional[int]:
    """This node's element count, or ``None`` when the store does not record one.

    A LEAF is read from the one attr its geometry stamps
    (:data:`_ELEMENT_COUNT_ATTR`). A WRAPPER child carries no single count, and
    which number describes it depends on what the wrapper MEANS:

    * a ``kind=partition`` (or a plain group) holds PARTS that all render
      together, so its count is the sum over its descendant leaves;
    * a nested ``kind=lod`` holds ALTERNATIVES — exactly one of its levels is
      ever drawn — so summing them would report a level count nobody can see.
      Its own finest level is the honest answer, and the one the enclosing
      ladder's operator is comparing against its siblings.

    ``None`` (rather than ``0``) when NOTHING resolved, because the two must stay
    distinguishable: :func:`coverage_fractions` raises on a finest count of ``0``
    precisely so a ladder whose finest level was culled fails loudly, and
    fabricating a count would defeat that guard on exactly the store that needs
    it. A wrapper that resolves SOME of its leaves is summed over those: the
    result is still a genuine positive measured from real leaves, never invented.
    """
    key = _ELEMENT_COUNT_ATTR.get(str(attrs.get("type")))
    if key is not None:
        raw = attrs.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return int(raw)
        return None
    if attrs.get("kind") == "lod":
        nested = _lod_children(group)
        if not nested:
            return None
        _, finest, finest_attrs = nested[-1]
        return _count_of(finest, finest_attrs)
    total = 0
    resolved = False
    for _, child, child_attrs in _child_nodes(group):
        sub = _count_of(child, child_attrs)
        if sub is not None:
            total += sub
            resolved = True
    return total if resolved else None


def _threshold_of(attrs: Dict[str, Any]) -> Optional[float]:
    """A child's stored ``coverage_fraction`` as a float, or ``None``."""
    raw = attrs.get("coverage_fraction")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return None


def _format_ladder(values: Sequence[Optional[float]]) -> str:
    """A ladder as compact text; a missing threshold renders as ``?``."""
    return "[" + ", ".join("?" if v is None else f"{v:g}" for v in values) + "]"


def _format_counts(values: Sequence[Optional[int]]) -> str:
    """Element counts as compact text; an unresolved one renders as ``?``."""
    return ", ".join("?" if v is None else f"{v:,}" for v in values)


@dataclass(frozen=True)
class _PlannedRestamp:
    """One group's re-derivation, decided but not yet written.

    Planning is a strictly READ-ONLY pass over the whole store, so a group that
    the pass refuses (an unsupported selector, an unresolvable count, a
    descending ladder) is discovered before ANY attr has been written — and the
    apply phase below is then the only place a write can fail.

    PATHS, not the ``zarr.Group`` handles the planning walk held: a zarr attr
    write serialises the handle's whole cached attrs dict, so two handles on the
    same node silently undo each other's writes. A nested ``kind=lod`` inside a
    ``kind=lod`` is exactly that shape — the inner group's ``selector`` write
    would restore the outer group's pre-run ``coverage_fraction`` on the node
    they share. :func:`_handle` hands out one handle per path instead.
    """

    path: str
    child_paths: List[str]
    entry: RestampedGroup


#: What ``_lod_children`` / ``_child_nodes`` yield: ``(name, group, attrs)``.
_ChildNode = Tuple[str, "zarr.Group", Dict[str, Any]]


def _is_partition_bound(under_partition: bool, children: Sequence[_ChildNode]) -> bool:
    """The writers' full two-clause tile-binding rule for a ``kind=lod`` group.

    ``under_partition`` is the ancestry clause (a real >1-part ``kind=partition``
    somewhere above, threaded down by :func:`_walk`); the second clause is a
    ladder child that IS a ``kind=partition`` — the ``overview`` cap. Both tree
    writers spell it ``under_partition or any(isinstance(c, GSplatPartition) for
    c in on_disk)``; this is the same test read off the store's ``kind`` attrs.

    Args:
        under_partition: Is some enclosing partition a real tiling?
        children: This group's ladder children as ``(name, group, attrs)``, in
            any order — the test is an ``any``.

    Returns:
        True when the ladder must be derived at the fills-screen (tile) anchor.
    """
    return under_partition or any(
        child_attrs.get("kind") == "partition" for _, _, child_attrs in children
    )


def _empty_ladder_refusal(
    group: "zarr.Group", children: Sequence[_ChildNode]
) -> Optional[SkippedGroup]:
    """Classify a ``kind=lod`` group with no resolved ladder children."""
    if children:
        return None
    path = group.path or "/"
    subgroups = sorted(group_keys(group))
    if subgroups:
        return SkippedGroup(
            path,
            "unclassifiable-children",
            f"kind=lod group has {len(subgroups)} child group(s) "
            f"({', '.join(subgroups)}) but none of them carries a "
            "scene-node 'type' attr, so the ladder cannot be ordered",
        )
    return SkippedGroup(
        path, "no-children", "kind=lod group has no child groups at all"
    )


def _orphan_ladder_child_refusal(
    group: "zarr.Group", children: Sequence[_ChildNode]
) -> Optional[SkippedGroup]:
    """Refuse a group holding a ladder rung the node filter cannot see, or ``None``.

    A child group the node filter DROPPED but which carries a
    ``coverage_fraction`` is a ladder rung this pass cannot see. Deriving over the
    survivors alone writes a PARTIAL ladder: the dropped rung keeps its legacy
    threshold under the new ``screen-area`` selector, so the result is
    non-monotonic AND — a legacy value being on the 0..4 diagonal scale — may sit
    above the screen-area ceiling of 1.0, which no clipped area metric can ever
    satisfy. The viewer re-sorts such a ladder with a warning
    (``load-lod-group-node.ts``), swapping levels and stranding the real finest
    one. Refuse the group and name the child instead.

    Shared with :mod:`luxar.io.lod_screening`, which must refuse exactly the
    groups this pass refuses or its report predicts a rewrite that never happens.

    Args:
        group: The ``kind=lod`` group.
        children: Its resolved ladder children, as ``_lod_children`` yields them.

    Returns:
        The refusal, or ``None`` when every ``coverage_fraction`` child resolved.
    """
    resolved = {name for name, _, _ in children}
    orphans = sorted(
        str(name)
        for name in group_keys(group)
        if str(name) not in resolved
        and _threshold_of(dict(group[str(name)].attrs)) is not None
    )
    if not orphans:
        return None
    return SkippedGroup(
        group.path or "/",
        "unclassifiable-ladder-child",
        f"{len(orphans)} child group(s) ({', '.join(orphans)}) carry a "
        "'coverage_fraction' but do not resolve as ladder levels (no "
        "scene-node 'type' attr, or a reserved bucket name), so "
        f"re-deriving over the {len(children)} that do would leave a "
        "PARTIAL, non-monotonic ladder with those rungs stranded on "
        "their legacy thresholds. Fix the children's 'type' stamps first",
    )


def _descending_ladder_refusal(
    path: str, old: Sequence[Optional[float]]
) -> Optional[SkippedGroup]:
    """Refuse a stored ladder that DESCENDS in child order, or ``None``.

    A ladder is coarsest→finest, so its thresholds must ASCEND. When they
    descend, the resolved child order and the stored thresholds disagree about
    which level is finest, and re-deriving would write an ascending ladder onto
    a descending order — silently INVERTING it (the 100-element level shown at
    half-screen, the 10,000-element one only when tiny). Only checked when the
    whole ladder is present: a partially-stamped one carries no such claim.

    Shared with :mod:`luxar.io.lod_screening` for the same reason
    :func:`_orphan_ladder_child_refusal` is.

    Args:
        path: The group's store path, for the message.
        old: Its stored thresholds in resolved coarsest→finest child order.

    Returns:
        The refusal, or ``None`` when the ladder is not a complete descending one.
    """
    if not all(value is not None for value in old):
        return None
    if not any(
        b < a  # type: ignore[operator]
        for a, b in zip(old, old[1:])
    ):
        return None
    return SkippedGroup(
        path,
        "descending-ladder",
        f"the stored ladder {_format_ladder(old)} DESCENDS in the "
        "resolved coarsest→finest child order, so the two disagree "
        "about which level is finest; re-deriving would invert it. Fix "
        "the children's 'child_index' stamps first",
    )


def _derive_thresholds(
    counts: List[Optional[int]],
    *,
    partition_bound: bool,
    finest_anchor: Optional[float],
    path: str,
) -> List[float]:
    """Derive one ladder, applying an explicit whole-object anchor if requested."""
    # Only the LENGTH and the finest entry are consumed (see the two derivation
    # docstrings). The caller checks the finest, so a coarser level whose count
    # the store never recorded is passed as 0 rather than blocking a
    # re-derivation it cannot affect. The report keeps the honest `None`.
    resolved_counts = [0 if count is None else count for count in counts]
    if partition_bound:
        return [
            float(value) for value in partitioned_coverage_fractions(resolved_counts)
        ]

    thresholds = coverage_fractions(resolved_counts)
    if finest_anchor is None:
        return [float(value) for value in thresholds]

    scale = finest_anchor / WHOLE_OBJECT_FINEST_ANCHOR
    reanchored = [float(value * scale) for value in thresholds]
    if any(
        right <= left for left, right in zip(reanchored, reanchored[1:], strict=False)
    ):
        raise ValueError(
            f"finest_anchor={finest_anchor!r} is too small to represent a "
            f"strictly increasing {len(reanchored)}-level ladder at {path}"
        )
    return reanchored


def _plan_lod(
    group: "zarr.Group",
    attrs: Dict[str, Any],
    *,
    partition_bound: bool,
    finest_anchor: Optional[float],
    report: RestampReport,
) -> Optional[_PlannedRestamp]:
    """Classify one ``kind=lod`` group and, when it is legacy, plan its rewrite.

    Args:
        group: The ``kind=lod`` group.
        attrs: Its attrs, already read.
        partition_bound: The resolved anchor binding (see
            :func:`_is_partition_bound`).
        finest_anchor: An explicit whole-object anchor, or ``None`` for the
            default migration behavior.
        report: Collects the classification — restamped, or one of the three
            skip buckets.

    Returns:
        The planned rewrite, or ``None`` when the group is skipped.
    """
    path = group.path or "/"
    selector = attrs.get("selector")

    if selector == DERIVED_LOD_SELECTOR and (finest_anchor is None or partition_bound):
        report.already_current.append(
            SkippedGroup(path, "already-current", f"selector={DERIVED_LOD_SELECTOR!r}")
        )
        return None

    # An ABSENT selector is legacy — `add_lod_group`'s historical default — but a
    # PRESENT one outside the vocabulary (`pixel_size`, the pre-v3.2 gsplats
    # spelling) describes thresholds in units this pass cannot convert. Reported,
    # never touched: re-deriving would silently relabel a scale nobody checked.
    if selector is not None and selector not in LOD_SELECTORS:
        report.unsupported.append(
            SkippedGroup(
                path,
                "unsupported-selector",
                f"selector={selector!r} is outside {sorted(LOD_SELECTORS)}; "
                "migrate the store first (`luxar gsplat migrate-format`)",
            )
        )
        return None

    children = _lod_children(group)
    empty_refusal = _empty_ladder_refusal(group, children)
    if empty_refusal is not None:
        report.unresolved.append(empty_refusal)
        return None

    orphan_refusal = _orphan_ladder_child_refusal(group, children)
    if orphan_refusal is not None:
        report.unresolved.append(orphan_refusal)
        return None

    old = [_threshold_of(child_attrs) for _, _, child_attrs in children]

    descending_refusal = _descending_ladder_refusal(path, old)
    if descending_refusal is not None:
        report.unresolved.append(descending_refusal)
        return None

    counts = [_count_of(child, child_attrs) for _, child, child_attrs in children]

    finest = counts[-1]
    if finest is None:
        report.unresolved.append(
            SkippedGroup(
                path,
                "unresolved-finest-count",
                "the finest child records no element count, so the "
                "'finest LOD level is empty' guard cannot be honoured",
            )
        )
        return None
    if finest <= 0:
        report.unresolved.append(
            SkippedGroup(
                path,
                "empty-finest-level",
                f"the finest child holds {finest} elements — a broken ladder, "
                "not one to re-anchor",
            )
        )
        return None

    new_thresholds = _derive_thresholds(
        counts,
        partition_bound=partition_bound,
        finest_anchor=finest_anchor,
        path=path,
    )
    resolved_anchor = (
        PARTITION_FINEST_AREA if partition_bound else WHOLE_OBJECT_FINEST_ANCHOR
    )
    if finest_anchor is not None and not partition_bound:
        resolved_anchor = finest_anchor

    if selector == DERIVED_LOD_SELECTOR and old == new_thresholds:
        report.already_current.append(
            SkippedGroup(
                path,
                "already-current",
                f"stored ladder already matches anchor {resolved_anchor:g}",
            )
        )
        return None

    entry = RestampedGroup(
        path=path,
        partition_bound=partition_bound,
        anchor=resolved_anchor,
        old_selector=None if selector is None else str(selector),
        old_thresholds=old,
        new_thresholds=new_thresholds,
        element_counts=counts,
    )
    report.restamped.append(entry)

    aprint(
        f"  🪜 {path}: anchor {entry.anchor_name} {entry.anchor:g}, "
        f"selector {entry.old_selector or '<absent>'} → {DERIVED_LOD_SELECTOR}"
    )
    aprint(
        f"       {_format_ladder(old)} → {_format_ladder(entry.new_thresholds)}"
        f"  (elements {_format_counts(counts)})"
    )
    return _PlannedRestamp(
        path=path,
        child_paths=[child.path for _, child, _ in children],
        entry=entry,
    )


def _lod_paths(group: "zarr.Group") -> List[str]:
    """Every ``kind=lod`` group path in the store, in walk order.

    A read-only pre-pass so ``--group`` can be validated before anything is
    written.
    """
    out: List[str] = []
    if dict(group.attrs).get("kind") == "lod":
        out.append(group.path or "/")
    for name in group_keys(group):
        out.extend(_lod_paths(group[str(name)]))
    return out


def _walk(
    group: "zarr.Group",
    *,
    under_partition: bool,
    selected: Optional[Set[str]],
    finest_anchor: Optional[float],
    report: RestampReport,
    plans: List[_PlannedRestamp],
) -> None:
    """Recurse the store read-only, planning every ``kind=lod`` ``selected`` allows.

    The tile binding is threaded with the WRITERS' rule
    (``io/_compiler/gsplat_tree.py::write_gsplat_node``), which has two halves
    and needs both:

    * a ``kind=partition`` binds its children only when it holds more than one
      part — a one-part partition is not a tiling, its single part's bbox IS the
      whole object — and the flag ORs in going down, never clears, so a lone
      wrapper nested inside a real tiling is still inside one tile;
    * a ``kind=lod`` group with a ``kind=partition`` among its OWN ladder
      children (the ``overview`` cap) is bound, and passes that binding — not the
      ancestral flag — down to its descendants, exactly as the writers do with
      ``under_partition=partition_bound``.

    A group excluded by ``selected`` still resolves and threads its binding: what
    ``--group`` restricts is which ladders get REWRITTEN, not what the topology
    is.

    Child GROUPS only (``group_keys()``, never ``keys()``): a group's arrays are
    listed by the latter and recursing into a ``zarr.Array`` dies on
    ``Array.keys()``.
    """
    attrs = dict(group.attrs)
    kind = attrs.get("kind")
    child_under = under_partition
    if kind == "lod":
        child_under = _is_partition_bound(under_partition, _lod_children(group))
        if selected is None or (group.path or "/") in selected:
            planned = _plan_lod(
                group,
                attrs,
                partition_bound=child_under,
                finest_anchor=finest_anchor,
                report=report,
            )
            if planned is not None:
                plans.append(planned)
    elif kind == "partition":
        child_under = under_partition or len(_child_nodes(group)) > 1
    for name in group_keys(group):
        _walk(
            group[str(name)],
            under_partition=child_under,
            selected=selected,
            finest_anchor=finest_anchor,
            report=report,
            plans=plans,
        )


def _normalise(path: str) -> str:
    """A user-supplied group path in the store's own spelling (root is ``"/"``)."""
    return path.strip().strip("/") or "/"


@dataclass(frozen=True)
class _AttrWrite:
    """One attr write, with whatever was in its place before it.

    ``written`` distinguishes an attr this run actually WROTE from one merely
    RECORDED before a later step might write it — the ``content_hash`` snapshot
    :func:`_snapshot_content_hashes` takes before the hash pass. Both are undone
    identically; only the first proves a document was rewritten, which is what
    :func:`_roll_back` keys its re-consolidation on.
    """

    path: str
    key: str
    existed: bool
    previous: Any
    written: bool = True


def _handle(
    root: "zarr.Group", path: str, cache: Dict[str, "zarr.Group"]
) -> "zarr.Group":
    """The ONE ``zarr.Group`` handle this run uses for ``path``.

    zarr updates a handle's cached attrs in place and rewrites the node's whole
    document on every attr write, so two handles on one node overwrite each
    other with their own stale views. Memoising by path makes that impossible
    without paying a metadata re-read per write.
    """
    node = cache.get(path)
    if node is None:
        node = root if path == "/" else root[path]
        cache[path] = node
    return node


def _write_attr(
    node: "zarr.Group", path: str, key: str, value: Any, undo: List[_AttrWrite]
) -> None:
    """Set ``node.attrs[key]``, recording the undo entry FIRST.

    Before, never after: a write that raises part-way must still be covered by
    the rollback, and an entry that restores an already-correct value is free.
    """
    attrs = dict(node.attrs)
    undo.append(_AttrWrite(path, key, key in attrs, attrs.get(key)))
    node.attrs[key] = value


def _baked_environment_group(
    root: zarr.Group, cache: Dict[str, zarr.Group]
) -> zarr.Group | None:
    """Return the baked sidecar, never ordinary data that only shares its name."""
    if ENVIRONMENT_GROUP not in root:
        return None
    candidate = _handle(root, ENVIRONMENT_GROUP, cache)
    faces = dict(candidate.attrs).get("faces")
    if not isinstance(faces, str) or faces not in candidate.array_keys():
        return None
    return candidate


def _undo_attr(
    root: "zarr.Group", entry: _AttrWrite, cache: Dict[str, "zarr.Group"]
) -> bool:
    """Put one attr back exactly as it was — absent included.

    Returns whether the node's document was rewritten, so :func:`_roll_back` knows
    whether the ROOT one moved: at zarr format 3 writing the root destroys the
    consolidated index, and re-consolidating a store whose index is still intact
    is the very risk this rollback exists to avoid. An attr already holding the
    value it would be restored to is left alone for the same reason.

    WHICH HANDLE answers that comparison is the subtle part. An attr this run
    WROTE is read back through the run's own cached handle, the only writer of it.
    A SNAPSHOT entry — a ``content_hash`` the hash pass may since have overwritten
    through handles of its own — is read and written through a FRESH one, because
    a cached view of such a node predates the hash pass and would report the digest
    unchanged when disk says otherwise. The two orders compose: snapshots are
    undone first (they were recorded last), and the ladder restore that follows on
    the same node rewrites its whole document from the run's pre-hash-pass view,
    which carries the same restored digest.
    """
    node = (
        _handle(root, entry.path, cache)
        if entry.written
        else (root if entry.path == "/" else root[entry.path])
    )
    attrs = dict(node.attrs)
    present = entry.key in attrs
    if not entry.existed:
        if not present:
            return False
        del node.attrs[entry.key]
        return True
    if present and attrs[entry.key] == entry.previous:
        return False
    node.attrs[entry.key] = entry.previous
    return True


def _snapshot_content_hashes(
    group: "zarr.Group", undo: List[_AttrWrite], *, deep: bool
) -> None:
    """Record the ``content_hash`` digests the store carries, before restamping.

    The rollback RESTORES those digests rather than recomputing them. A recompute
    only lands back on the stored value when the stored value was already this
    walk's answer, which a legacy store hashed by an older walk, a hand-edited
    store, or a scene whose inner groups carry no per-group hash is not — so a
    FAILED run would rewrite digests while reporting that it changed nothing.
    Restoring is also the cheap direction: one attr read per group here, against a
    full value walk over every array in the store on the failure path.

    ``deep`` mirrors :func:`~luxar.io.optimize._restamp_content_hash`: a scene's
    value walk stamps every group, a ``.gsplats.zarr`` root stamp only the root.
    A store with neither marker is stamped nowhere, and the single recorded root
    entry then undoes to nothing.
    """
    attrs = dict(group.attrs)
    undo.append(
        _AttrWrite(
            path=group.path or "/",
            key="content_hash",
            existed="content_hash" in attrs,
            previous=attrs.get("content_hash"),
            written=False,
        )
    )
    if not deep:
        return
    for name in group_keys(group):
        _snapshot_content_hashes(group[str(name)], undo, deep=True)


def _apply_one(
    root: "zarr.Group",
    plan: _PlannedRestamp,
    undo: List[_AttrWrite],
    cache: Dict[str, "zarr.Group"],
) -> None:
    """Write one planned ladder: the child thresholds, then the group selector.

    Thresholds first so that the window in which the two disagree is as short as
    possible, and the selector — the attr that DECLARES the units — is the very
    last thing to move.
    """
    for child_path, value in zip(plan.child_paths, plan.entry.new_thresholds):
        _write_attr(
            _handle(root, child_path, cache),
            child_path,
            "coverage_fraction",
            float(value),
            undo,
        )
    _write_attr(
        _handle(root, plan.path, cache),
        plan.path,
        "selector",
        DERIVED_LOD_SELECTOR,
        undo,
    )


def _roll_back(
    root: "zarr.Group",
    undo: List[_AttrWrite],
    error: BaseException,
    *,
    cache: Dict[str, "zarr.Group"],
    was_consolidated: bool,
) -> None:
    """Undo every attr this run wrote, then leave exactly one valid index.

    The failure this exists for is not hypothetical: a mid-walk ``PermissionError``
    leaves one ladder restamped, the failing one TORN (a screen-area threshold
    under ``selector="coverage"`` — thresholds and selector disagreeing about
    their units, which nothing downstream can detect), and the consolidated index
    describing neither. ``optimize`` is all-or-nothing for exactly this reason;
    this sibling writes in place and so has to unwind rather than stage.

    The ``content_hash`` is RESTORED, not recomputed: the pre-run digests are in
    the same ledger (:func:`_snapshot_content_hashes` records them before the hash
    pass runs), so a store whose stored digest was never this walk's answer — a
    legacy one, a hand-edited one, a scene whose inner groups carry none — comes
    out carrying exactly the digests it came in with. That is what makes the
    rollback byte-for-byte rather than merely semantic, and it is what lets the
    failure path stay metadata-only instead of re-reading every array in the store.

    The index is then rebuilt ONLY when this run rewrote the ROOT document, which
    at zarr format 3 is what destroys a consolidated index (the viewer builds its
    whole scene graph from that index with no directory-walk fallback, so a store
    left without one loads as an empty scene). A rollback that never touched the
    root leaves the index alone: it already describes the restored attrs, and
    re-consolidating it is a write that can itself fail — turning a recoverable
    failure at attr write #1 into a published store with no index at all.

    Args:
        root: The open store root.
        undo: Every attr write made and every digest recorded, in order.
        error: The exception being unwound; notes are attached to it.
        cache: The apply phase's per-path handle cache, reused so a restore sees
            the writes it is undoing.
        was_consolidated: Did the store carry a consolidated index before the run?
            A store that had none is not given one here.
    """
    failures: List[str] = []
    restored = 0
    root_written = any(entry.path == "/" and entry.written for entry in undo)
    for entry in reversed(undo):
        try:
            if _undo_attr(root, entry, cache):
                restored += 1
                root_written = root_written or entry.path == "/"
        except BaseException as undo_error:  # pragma: no cover - defensive
            failures.append(f"{entry.path}.{entry.key}: {undo_error}")
            # A restore that raised may have written the document part-way, so
            # the index is presumed lost rather than presumed intact.
            root_written = root_written or entry.path == "/"

    if failures:
        headline = (
            f"restamp-lod could NOT fully roll back: {len(failures)} of "
            f"{len(undo)} attr restores failed ({'; '.join(failures)}). The "
            "store may carry a torn ladder (a screen-area threshold under "
            "selector='coverage') and must be re-run or regenerated."
        )
    elif restored:
        headline = (
            f"restamp-lod rolled back {restored} attr write(s): every LOD "
            "ladder is exactly as it was before this run."
        )
    else:
        headline = (
            "restamp-lod failed before writing anything, so nothing needed "
            "rolling back: the store is exactly as it was."
        )
    aprint(f"  ❌ {headline}")
    error.add_note(headline)

    if was_consolidated and root_written:
        try:
            consolidate(root)
        except BaseException as index_error:
            error.add_note(
                "restamp-lod could not re-consolidate the store; it may now "
                f"carry no consolidated index (the viewer needs one): {index_error}"
            )


def _apply(
    root: "zarr.Group",
    plans: List[_PlannedRestamp],
    report: RestampReport,
    *,
    store_path: Path,
    was_consolidated: bool,
) -> None:
    """Write every planned ladder, then the hash and the index — or nothing.

    The writers' finalize order: hash BEFORE consolidating, so the new hash lands
    inside the index too and the viewer's persistent cache invalidates on an
    attrs-only change it would never see otherwise. Any exception anywhere in
    here — including inside the hash walk, which is the only step that reads
    arrays — unwinds the whole run through :func:`_roll_back`.

    An index is only ever REBUILT, never introduced: a store that arrived without
    one leaves without one. :func:`~luxar._zarr_compat.is_consolidated` is
    ``batch-fit``'s "this tile finished" sentinel, so consolidating an
    interrupted tile here would mark it complete.
    """
    undo: List[_AttrWrite] = []
    cache: Dict[str, "zarr.Group"] = {"/": root}
    try:
        for plan in plans:
            _apply_one(root, plan, undo, cache)

        is_scene = dict(root.attrs).get("type") == "scene"
        if is_scene:
            aprint(
                "  ℹ️  Restamping the scene content_hash: that digest covers "
                "array VALUES, so this reads every array in the store once "
                "(the ladder rewrite above opened none). Expect it to take as "
                "long as reading the whole store."
            )
        else:
            aprint(
                "  ℹ️  Restamping the root content_hash (metadata only — this "
                "store is not a scene, so no array is read)."
            )
        # One attr read per group, before the stamp overwrites them, so a
        # failure restores the store's OWN digests instead of recomputing them.
        _snapshot_content_hashes(root, undo, deep=is_scene)
        report.content_hash = _restamp_content_hash(root)
        environment = _baked_environment_group(root, cache)
        if report.content_hash is not None and environment is not None:
            _write_attr(
                environment,
                ENVIRONMENT_GROUP,
                "scene_content_hash",
                report.content_hash,
                undo,
            )
        report.content_hash_status = (
            HASH_RESTAMPED if report.content_hash is not None else HASH_UNSTAMPABLE
        )
        if was_consolidated:
            consolidate(root)
        else:
            aprint(
                "  ℹ️  This store carries no consolidated index and was not "
                "given one: `is_consolidated` is how batch-fit tells a finished "
                "tile from an interrupted one. The ladders and the content_hash "
                "were written to the per-node documents."
            )
    except BaseException as error:
        _roll_back(root, undo, error, cache=cache, was_consolidated=was_consolidated)
        raise
    report.residual = _verify(store_path, report, expect_index=was_consolidated)


def _verify(
    store_path: Path, report: RestampReport, *, expect_index: bool = True
) -> List[str]:
    """Re-read the written store and confirm every restamp is really there.

    Both readers are consulted, because they can disagree and the disagreement is
    the failure this exists to catch: :func:`~luxar._zarr_compat.open_group`
    (``use_consolidated=False``) reports the per-node documents on disk, while
    :func:`~luxar._zarr_compat.read_consolidated_attrs` reports the ROOT index —
    which is the only thing the viewer ever fetches. A consolidation mistake
    leaves the first correct and the second stale, and nothing raises.

    ``expect_index`` is False for a store that carried no consolidated index to
    begin with: the pass does not create one (see :func:`_apply`), so a missing
    index is then the expected state rather than a consolidation mistake, and the
    per-node documents are the whole contract.

    Returns one message per group that does not read back as expected.
    """
    residual: List[str] = []
    expected = {entry.path: entry for entry in report.restamped}
    if not expected:
        return residual

    def check(
        path: str,
        source: str,
        group_attrs: Dict[str, Any],
        ladder: List[Optional[float]],
    ) -> None:
        entry = expected[path]
        if group_attrs.get("selector") != DERIVED_LOD_SELECTOR:
            residual.append(
                f"{path}: {source} still reports "
                f"selector={group_attrs.get('selector')!r}"
            )
        if ladder != [float(v) for v in entry.new_thresholds]:
            residual.append(
                f"{path}: {source} reports thresholds {_format_ladder(ladder)}, "
                f"expected {_format_ladder(entry.new_thresholds)}"
            )

    consolidated = read_consolidated_attrs(store_path)
    root = open_group(store_path, mode="r")
    try:
        for path in expected:
            group = root if path == "/" else root[path]
            children = _lod_children(group)
            check(
                path,
                "the store",
                dict(group.attrs),
                [_threshold_of(a) for _, _, a in children],
            )
            if not expect_index:
                continue
            if not consolidated:
                residual.append(f"{path}: the store carries no consolidated index")
                continue
            index_attrs = consolidated.get(path)
            if index_attrs is None:
                residual.append(f"{path}: missing from the consolidated index")
                continue
            prefix = "" if path == "/" else f"{path}/"
            check(
                path,
                "the consolidated index",
                index_attrs,
                [
                    _threshold_of(consolidated.get(f"{prefix}{name}", {}))
                    for name, _, _ in children
                ],
            )
    finally:
        close(root)
    return residual


def _summarise(report: RestampReport) -> None:
    """Print the tail of the run: what was skipped, and why.

    ``❌`` rather than ``⚠️`` for the two skip buckets, per
    ``docs/guides/developer/CONSOLE_OUTPUT_STYLE.md``: both make
    :attr:`RestampReport.clean` False and so exit the CLI non-zero, which is an
    error and not a warning. The unstampable-digest line keeps its ``⚠️`` even
    though it too makes ``clean`` False: the ladders it describes were written
    correctly and the store is not damaged — what is missing is the cache
    invalidation, which the operator fixes by republishing, not by re-running.
    """
    for entry in report.unsupported:
        aprint(f"  ❌ {entry.path}: {entry.detail}")
    for entry in report.unresolved:
        aprint(f"  ❌ {entry.path}: {entry.detail}")
    if report.content_hash_status == HASH_UNSTAMPABLE:
        aprint(
            "  ⚠️  This store carries neither a scene 'type' nor a "
            "'.gsplats.zarr' content_hash, so there was no digest to restamp: "
            "at zarr format 2, a warm viewer cache will NOT see the new ladder. "
            "Republish under a new URL prefix."
        )
    aprint(
        f"  {len(report.restamped)} restamped, "
        f"{len(report.already_current)} already {DERIVED_LOD_SELECTOR}, "
        f"{len(report.unsupported)} unsupported, "
        f"{len(report.unresolved)} unresolved"
    )
    for message in report.residual:
        aprint(f"  ❌ verify: {message}")


def _validate_finest_anchor(finest_anchor: Optional[float]) -> None:
    """Reject values that cannot be a finite screen-area fraction."""
    if finest_anchor is None:
        return
    if math.isfinite(finest_anchor) and 0.0 < finest_anchor <= 1.0:
        return
    raise ValueError(
        "--anchor must be finite and in the screen-area interval (0, 1]; "
        f"got {finest_anchor!r}"
    )


def restamp_lod_store(
    path: "str | Path",
    *,
    dry_run: bool = False,
    groups: Optional[Sequence[str]] = None,
    finest_anchor: Optional[float] = None,
) -> RestampReport:
    """Re-derive ``kind=lod`` ladders in a store, in place.

    An attrs-only pass: for each ``kind=lod`` group still on the legacy
    ``"coverage"`` selector (or carrying none, which means the same thing), the
    per-child ``coverage_fraction`` thresholds are re-derived by screen-occupancy
    halving — :func:`~luxar.core.group.lod.group.partitioned_coverage_fractions`
    when the group is tile-bound (a real >1-part partition above it, or a
    ``kind=partition`` among its own ladder children — the ``overview`` cap),
    :func:`~luxar.core.group.lod.group.coverage_fractions` otherwise — and the
    group is stamped :data:`~luxar.typing_utils.constants.DERIVED_LOD_SELECTOR`.
    A group already on that selector is skipped unless ``finest_anchor`` is
    supplied. That explicit override sets the requested screen-area fraction for
    every whole-object ladder processed, both legacy and already current;
    partition-bound ladders remain anchored at fills-screen ``1.0``. A requested
    ladder that already matches is still a no-op down to the ``content_hash``.

    The ladder rewrite moves no chunk and opens no array. When anything changed,
    the store's ``content_hash`` is restamped and the metadata re-consolidated,
    in that order — an attrs-only change must still invalidate a warm viewer
    cache — and the result is then read back and verified. A store that carried
    NO consolidated index is not given one (``is_consolidated`` is ``batch-fit``'s
    finished-tile sentinel); the report says so in
    :attr:`~RestampReport.was_consolidated`. That hash restamp is
    the one expensive step: for a compiled SCENE the digest covers array VALUES,
    so it streams the whole store once; a standalone ``.gsplats.zarr`` gets a
    metadata-only stamp instead. A store with NEITHER marker carries no digest to
    move: the ladders are still written, but the report comes back
    :data:`HASH_UNSTAMPABLE` and therefore NOT :attr:`~RestampReport.clean`, since
    a warm viewer cache would keep serving the old ladder until the store is
    republished under a new URL prefix.

    All-or-nothing on the write side. Every group is classified in a read-only
    planning walk first; if any write then fails, every attr already written is
    restored (an absent ``coverage_fraction`` back to absent, a ``content_hash``
    back to the digest the store arrived with), the index is re-consolidated if
    and only if the root document was rewritten, and the original error is
    re-raised carrying a note saying what was rolled back. A half-restamped store
    would carry a ladder whose
    thresholds and ``selector`` disagree about their units, which nothing
    downstream can detect.

    Parameters
    ----------
    path
        An uncompressed ``.luxar.zarr`` / ``.gsplats.zarr`` DIRECTORY. A
        ``.zarr.zip`` is rejected: an archive read is temp-dir based, so there is
        nothing to write back to in place.
    dry_run
        Report the per-group old→new ladder and write nothing (the store is
        opened read-only).
    groups
        Restrict the pass to these group paths, in the store's own spelling
        (``"tiled/part_0"``; the root is ``"/"``). A path that matches no
        ``kind=lod`` group is an error, not a silent no-op.
    finest_anchor
        Optional whole-object finest-level screen-area fraction in ``(0, 1]``.
        Supplying it explicitly re-derives already-``screen-area`` ladders as
        well as legacy ones. Partition-bound ladders remain at ``1.0``.

    Returns
    -------
    RestampReport
        Every group restamped, skipped-as-current, skipped-as-unsupported and
        skipped-as-unresolved, plus the new ``content_hash`` (with a
        ``content_hash_status`` saying whether there was one to move) and any
        re-verification residual.

    Raises
    ------
    ValueError
        The path is not a directory, is not a Luxar store, or ``groups`` names a
        path that is not a ``kind=lod`` group in it.
    """
    store_path = Path(path)
    if not store_path.is_dir():
        raise ValueError(
            f"restamp-lod requires an uncompressed .zarr directory; got "
            f"{store_path} (unpack a .zip/.tar.gz store first — an attrs rewrite "
            f"of a compressed archive cannot happen in place)"
        )
    _validate_finest_anchor(finest_anchor)

    root = open_group(store_path, mode="r" if dry_run else "r+")
    # `optimize.ensure_luxar_store` is the same gate but its message names
    # `--generic`, an escape hatch this command does not offer (there is no
    # kind=lod group in a foreign store to restamp), so the CLASSIFICATION is
    # reused and the wording is this command's own.
    if not _is_luxar_store(root):
        close(root)
        raise ValueError(
            f"{store_path} does not look like a Luxar scene or a .gsplats.zarr "
            f"tree, so it carries no kind=lod ladder to restamp"
        )

    selected = None if groups is None else {_normalise(g) for g in groups}
    if selected is not None:
        # Validated BEFORE the mutating walk, not after it: an unmatched --group
        # is an error, and an error that fires halfway through leaves the store
        # partly restamped with nothing saying which half.
        available = _lod_paths(root)
        unmatched = sorted(selected - set(available))
        if unmatched:
            close(root)
            raise ValueError(
                f"--group named {unmatched}, which "
                f"{'is' if len(unmatched) == 1 else 'are'} not a kind=lod group "
                f"in {store_path}. Available: {sorted(available) or '(none)'}"
            )

    was_consolidated = is_consolidated(store_path)
    report = RestampReport(
        path=str(store_path), dry_run=dry_run, was_consolidated=was_consolidated
    )
    try:
        with asection(
            f"{'🔎 Dry run: ' if dry_run else '🪜 '}restamp-lod {store_path}"
        ):
            # PLAN first, in one read-only walk, so every refusal is known before
            # a single attr moves and the apply below is the only fallible phase.
            plans: List[_PlannedRestamp] = []
            _walk(
                root,
                under_partition=False,
                selected=selected,
                finest_anchor=finest_anchor,
                report=report,
                plans=plans,
            )

            if not report.restamped:
                if report.unsupported or report.unresolved:
                    aprint("  Nothing was restamped — see below.")
                else:
                    aprint(
                        "  Nothing to restamp — all inspected LOD ladders are "
                        "already current."
                    )

            if dry_run and plans and not _carries_restampable_digest(root):
                report.content_hash_status = HASH_UNSTAMPABLE

            # Only when something CHANGED: a clean no-op store must not have its
            # hash moved, and a dry run must not write at all.
            if not dry_run and plans:
                _apply(
                    root,
                    plans,
                    report,
                    store_path=store_path,
                    was_consolidated=was_consolidated,
                )

            _summarise(report)
            if dry_run:
                aprint("  Nothing was written.")
    finally:
        close(root)

    return report
