"""Re-derive a stored LOD ladder's thresholds under the current selector.

Every ``kind=lod`` group on disk writes two things that are really one decision:
the per-child ``coverage_fraction`` thresholds, and the group-level ``selector``
naming the UNITS they are in. The whole shipped demo corpus predates the
``"screen-area"`` selector, so its ladders carry ``selector="coverage"`` — the
legacy diagonal metric — over thresholds that were *derived* rather than
authored (issue #1727). Re-deriving them under the current metric is a pure
ATTRIBUTE rewrite: no chunk data moves, no array is touched, so a store already
on disk can be upgraded in place instead of regenerated.

**Why this is not part of ``luxar optimise``.** That pass documents "every
attribute is preserved" and refuses same-path work outright; this one changes
attributes and nothing else, in place. They are siblings, not one command with a
flag.

**Why it is never automatic.** An authored ``coverage_fractions=[...]`` list and
a legacy derived one are INDISTINGUISHABLE on disk — the point
``io/_compiler/finalize/lod_backfill.py::warn_one_part_partition_anchors``
makes normatively, which is why that check only ever warns. Running this command
IS the opt-in: nothing else may trigger it, and it prints the old→new ladder for
every group it rewrites so the audit trail exists even when the rewrite is
overriding a deliberate choice.

**Which anchor a ladder gets is decided from the STORE**, mirroring the tree
writers' rule (``io/_compiler/gsplat_tree.py::write_gsplat_node``) rather than
trusting the stamped values: a ladder is tile-anchored only when some enclosing
``kind=partition`` is a real tiling (>1 part). One consequence is deliberate — a
store whose one-part partition holds a tile-anchored ladder, exactly what
``warn_one_part_partition_anchors`` reports and cannot repair, comes out of this
pass re-anchored at whole-object. That is a rewrite, which is why it happens only
here and only when asked for.

The public entry point is :func:`restamp_lod_store`; the CLI wrapper is
``luxar.cli.restamp_lod_command``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import zarr
from arbol import aprint, asection

from .._zarr_compat import close, consolidate, open_group, read_consolidated_attrs
from ..core.group.lod.group import coverage_fractions, partitioned_coverage_fractions
from ..typing_utils.constants import DERIVED_LOD_SELECTOR, LOD_SELECTORS

# Private imports, deliberately. Both express rules this pass must MIRROR
# EXACTLY rather than re-state: `_lod_children` is the three-tier coarsest→finest
# ordering (child_index, then a `child_<i>` numeric suffix, then sorted name) and
# `_child_nodes` is the "which subgroups are scene nodes" filter that keeps the
# reserved fitting/provenance/pipeline buckets out. A local copy of either would
# be one more place for the ordering convention to drift.
from ._compiler.finalize.amplitude_window import _child_nodes, _lod_children
from .optimise import _is_luxar_store, _restamp_content_hash

__all__ = [
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


@dataclass(frozen=True)
class RestampedGroup:
    """One ``kind=lod`` group whose ladder was (or would be) re-derived."""

    path: str
    partition_bound: bool
    old_selector: Optional[str]
    old_thresholds: List[Optional[float]]
    new_thresholds: List[float]
    element_counts: List[Optional[int]]

    @property
    def anchor(self) -> str:
        """Human name of the anchor the re-derivation used."""
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
    residual: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Was every ``kind=lod`` group either restamped or already current?

        False when a group was skipped for a reason the caller must act on — an
        out-of-vocabulary selector, an unresolvable finest element count, or a
        re-verification residual. Nothing is ever silently ignored, so the CLI
        keys its exit code on this.
        """
        return not (self.unsupported or self.unresolved or self.residual)


def _count_of(group: "zarr.Group", attrs: Dict[str, Any]) -> Optional[int]:
    """This node's element count, or ``None`` when the store does not record one.

    A LEAF is read from the one attr its geometry stamps
    (:data:`_ELEMENT_COUNT_ATTR`); a WRAPPER child — a ``kind=partition``, a
    nested ``kind=lod``, or a plain group, all of which are real ladder-child
    shapes — carries no single count and is summed over its descendant leaves.

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


def _visit_lod(
    group: "zarr.Group",
    attrs: Dict[str, Any],
    *,
    under_partition: bool,
    report: RestampReport,
    dry_run: bool,
) -> None:
    """Classify one ``kind=lod`` group and, when it is legacy, re-derive it."""
    path = group.path or "/"
    selector = attrs.get("selector")

    if selector == DERIVED_LOD_SELECTOR:
        report.already_current.append(
            SkippedGroup(path, "already-current", f"selector={DERIVED_LOD_SELECTOR!r}")
        )
        return

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
        return

    children = _lod_children(group)
    if not children:
        report.unresolved.append(
            SkippedGroup(path, "no-children", "kind=lod group has no ladder children")
        )
        return

    old = [_threshold_of(child_attrs) for _, _, child_attrs in children]
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
        return
    if finest <= 0:
        report.unresolved.append(
            SkippedGroup(
                path,
                "empty-finest-level",
                f"the finest child holds {finest} elements — a broken ladder, "
                "not one to re-anchor",
            )
        )
        return

    derive = partitioned_coverage_fractions if under_partition else coverage_fractions
    # Only the LENGTH and the finest entry are consumed (see the two derivation
    # docstrings), and the finest is checked above — so a coarser level whose
    # count the store never recorded is passed as 0 rather than blocking a
    # re-derivation it cannot affect. The report keeps the honest `None`.
    new = derive([0 if c is None else c for c in counts])

    entry = RestampedGroup(
        path=path,
        partition_bound=under_partition,
        old_selector=None if selector is None else str(selector),
        old_thresholds=old,
        new_thresholds=[float(v) for v in new],
        element_counts=counts,
    )
    report.restamped.append(entry)

    aprint(
        f"  🪜 {path}: anchor {entry.anchor}, "
        f"selector {entry.old_selector or '<absent>'} → {DERIVED_LOD_SELECTOR}"
    )
    aprint(
        f"       {_format_ladder(old)} → {_format_ladder(entry.new_thresholds)}"
        f"  (elements {_format_counts(counts)})"
    )

    if dry_run:
        return
    for (_, child, _), value in zip(children, entry.new_thresholds):
        child.attrs["coverage_fraction"] = float(value)
    group.attrs["selector"] = DERIVED_LOD_SELECTOR


def _lod_paths(group: "zarr.Group") -> List[str]:
    """Every ``kind=lod`` group path in the store, in walk order.

    A read-only pre-pass so ``--group`` can be validated before anything is
    written.
    """
    out: List[str] = []
    if dict(group.attrs).get("kind") == "lod":
        out.append(group.path or "/")
    for name in group.group_keys():
        out.extend(_lod_paths(group[str(name)]))
    return out


def _walk(
    group: "zarr.Group",
    *,
    under_partition: bool,
    selected: Optional[set],
    report: RestampReport,
    dry_run: bool,
) -> None:
    """Recurse the store, handling every ``kind=lod`` group ``selected`` allows.

    ``under_partition`` is threaded with the WRITERS' rule
    (``io/_compiler/gsplat_tree.py::write_gsplat_node``): a ``kind=partition``
    binds its children only when it holds more than one part — a one-part
    partition is not a tiling, its single part's bbox IS the whole object — and
    the flag ORs in going down, never clears, so a lone wrapper nested inside a
    real tiling is still inside one tile.

    Child GROUPS only (``group_keys()``, never ``keys()``): a group's arrays are
    listed by the latter and recursing into a ``zarr.Array`` dies on
    ``Array.keys()``.
    """
    attrs = dict(group.attrs)
    kind = attrs.get("kind")
    if kind == "lod" and (selected is None or (group.path or "/") in selected):
        _visit_lod(
            group,
            attrs,
            under_partition=under_partition,
            report=report,
            dry_run=dry_run,
        )
    child_under = under_partition or (
        kind == "partition" and len(_child_nodes(group)) > 1
    )
    for name in group.group_keys():
        _walk(
            group[str(name)],
            under_partition=child_under,
            selected=selected,
            report=report,
            dry_run=dry_run,
        )


def _normalise(path: str) -> str:
    """A user-supplied group path in the store's own spelling (root is ``"/"``)."""
    return path.strip().strip("/") or "/"


def _verify(store_path: Path, report: RestampReport) -> List[str]:
    """Re-read the written store and confirm every restamp is really there.

    Both readers are consulted, because they can disagree and the disagreement is
    the failure this exists to catch: :func:`~luxar._zarr_compat.open_group`
    (``use_consolidated=False``) reports the per-node documents on disk, while
    :func:`~luxar._zarr_compat.read_consolidated_attrs` reports the ROOT index —
    which is the only thing the viewer ever fetches. A consolidation mistake
    leaves the first correct and the second stale, and nothing raises.

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
    """Print the tail of the run: what was skipped, and why."""
    for entry in report.unsupported:
        aprint(f"  ⚠️  {entry.path}: {entry.detail}")
    for entry in report.unresolved:
        aprint(f"  ⚠️  {entry.path}: {entry.detail}")
    aprint(
        f"  {len(report.restamped)} restamped, "
        f"{len(report.already_current)} already {DERIVED_LOD_SELECTOR}, "
        f"{len(report.unsupported)} unsupported, "
        f"{len(report.unresolved)} unresolved"
    )
    for message in report.residual:
        aprint(f"  ❌ verify: {message}")


def restamp_lod_store(
    path: "str | Path",
    *,
    dry_run: bool = False,
    groups: Optional[Sequence[str]] = None,
) -> RestampReport:
    """Re-derive every legacy ``kind=lod`` ladder in a store, in place.

    An attrs-only pass: for each ``kind=lod`` group still on the legacy
    ``"coverage"`` selector (or carrying none, which means the same thing), the
    per-child ``coverage_fraction`` thresholds are re-derived by screen-occupancy
    halving — :func:`~luxar.core.group.lod.group.partitioned_coverage_fractions`
    when the group is bound to a real (>1 part) spatial partition,
    :func:`~luxar.core.group.lod.group.coverage_fractions` otherwise — and the
    group is stamped :data:`~luxar.typing_utils.constants.DERIVED_LOD_SELECTOR`.
    A group already on that selector is skipped, so a second run is a no-op down
    to the ``content_hash``.

    No chunk data moves and no array is opened. When anything changed, the
    store's ``content_hash`` is restamped and the metadata re-consolidated, in
    that order — an attrs-only change must still invalidate a warm viewer cache —
    and the result is then read back and verified.

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

    Returns
    -------
    RestampReport
        Every group restamped, skipped-as-current, skipped-as-unsupported and
        skipped-as-unresolved, plus the new ``content_hash`` and any
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

    root = open_group(store_path, mode="r" if dry_run else "r+")
    # `optimise.ensure_luxar_store` is the same gate but its message names
    # `--generic`, an escape hatch this command does not offer (there is no
    # kind=lod group in a foreign store to restamp), so the CLASSIFICATION is
    # reused and the wording is this command's own.
    if not _is_luxar_store(root):
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

    report = RestampReport(path=str(store_path), dry_run=dry_run)
    try:
        with asection(
            f"{'🔎 Dry run: ' if dry_run else '🪜 '}restamp-lod {store_path}"
        ):
            _walk(
                root,
                under_partition=False,
                selected=selected,
                report=report,
                dry_run=dry_run,
            )

            if not report.restamped:
                aprint("  Nothing to restamp — every LOD ladder is already current.")

            if not dry_run and report.restamped:
                # The writers' finalize order: hash BEFORE consolidating, so the
                # new hash lands inside the index too and the viewer's persistent
                # cache invalidates on an attrs-only change it would never see
                # otherwise. Only when something CHANGED: a clean no-op store must
                # not have its hash moved.
                report.content_hash = _restamp_content_hash(root)
                consolidate(root)
                report.residual = _verify(store_path, report)

            _summarise(report)
            if dry_run:
                aprint("  Nothing was written.")
    finally:
        close(root)

    return report
