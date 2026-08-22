"""Diagnose — and, on request, repair — partition metadata in Luxar stores.

A dataset can be perfectly loadable and still be missing something a later
Luxar learned to record, or be carrying metadata that has quietly gone stale
under an edit. Those conditions are invisible in the viewer: the scene renders,
just not as well as it should. The doctor is where that class of problem is
named, explained, and — where the correct value is recoverable from the store
itself — fixed in place, without re-fitting.

Read-only by default: :func:`diagnose_store` reports, and only writes when
``fix=True``. Repairs to standalone gsplat stores and scenes are metadata-level
and go through one finalize
(:func:`~luxar.gsplats.io.save_gsplats._stamp_content_hash` then
``zarr.consolidate_metadata``, in the writer's order) so the consolidated
metadata cannot disagree with the per-node attrs it shadows, and the viewer's
persistent cache invalidates on the change.

The checks live in :mod:`.checks`; see its module docstring for how to add one.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional

import zarr

from luxar._zarr_compat import consolidate, open_group

from .checks import ALL_CHECKS
from .model import Check, DoctorReport, Finding

__all__ = [
    "ALL_CHECKS",
    "Check",
    "DoctorReport",
    "Finding",
    "diagnose_store",
    "resolve_store_kind",
]

StoreKind = Literal["gsplats", "scene"]


def _store_kind_from_attrs(attrs: Mapping[str, Any]) -> Optional[StoreKind]:
    if attrs.get("format_type") == "gsplats_zarr":
        return "gsplats"
    if attrs.get("type") == "scene" and "scene_dimensions" in attrs:
        return "scene"
    return None


def resolve_store_kind(path: "str | Path") -> StoreKind:
    """Classify a Luxar store from its root attrs, including archives."""
    path = Path(path)
    if path.is_dir():
        attrs = open_group(path, mode="r").attrs
    else:
        from luxar.gsplats.io._archive import read_archive_root_attrs

        attrs = read_archive_root_attrs(path)

    kind = _store_kind_from_attrs(attrs)
    if kind is None:
        raise ValueError(
            f"{path} is not a Luxar scene or standalone .gsplats.zarr store "
            f"(type={attrs.get('type')!r}, "
            f"format_type={attrs.get('format_type')!r})."
        )
    return kind


def diagnose_store(
    path: "str | Path",
    *,
    fix: bool = False,
    checks: "Optional[List[Check]]" = None,
) -> DoctorReport:
    """Run every check over a ``.gsplats.zarr`` or ``.luxar.zarr`` store.

    Parameters
    ----------
    path
        A ``.gsplats.zarr`` / ``.luxar.zarr`` directory, or a
        ``.zip``/``.tar.gz`` archive. An
        archive is extracted to a temp directory and read from there, so it can
        be DIAGNOSED but not repaired: with ``fix=True`` it is rejected, for the
        same reason ``annotate-quality`` rejects one — there is nothing to write
        back to in place. Unpack first to repair.
    fix
        Apply the repairs the checks offer. Off by default: a diagnosis should
        never surprise anyone by writing.
    checks
        Override the registry (tests, or a targeted run).

    Returns
    -------
    DoctorReport
        Every finding, each flagged with whether it was repaired. After a
        repairing run it also carries ``residual`` — what a fresh pass of the
        same checks reports afterwards, which is what ``healthy`` keys on.
    """
    path = Path(path)
    if fix and not path.is_dir():
        raise ValueError(
            f"doctor --fix requires an uncompressed zarr directory; got "
            f"{path} (unpack a .zip/.tar.gz first — an archive cannot be repaired "
            f"in place). Without --fix it can still be diagnosed."
        )

    # A compressed store is diagnosable but not repairable: extract to a temp
    # directory and read from there. Most bundled demo datasets ship as .zip, so
    # refusing them outright would put the common case out of reach of a
    # read-only sweep.
    scratch: Optional[Path] = None
    if not path.is_dir():
        from luxar.gsplats.io._archive import extract_compressed_zarr

        target = extract_compressed_zarr(path)
        scratch = target.parent
    else:
        target = path

    try:
        return _diagnose_opened(path, target, fix=fix, checks=checks)
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _diagnose_opened(
    reported_path: Path,
    store_path: Path,
    *,
    fix: bool,
    checks: "Optional[List[Check]]",
) -> DoctorReport:
    """Run the checks against an already-resolved directory store.

    ``reported_path`` is what the user asked about (an archive keeps its own name
    in the report); ``store_path`` is the directory actually read.
    """
    root = open_group(store_path, mode="r+" if fix else "r")
    if _store_kind_from_attrs(root.attrs) is None:
        raise ValueError(
            f"{reported_path} is not a Luxar scene or standalone .gsplats.zarr "
            f"store (type={root.attrs.get('type')!r}, "
            f"format_type={root.attrs.get('format_type')!r})."
        )

    selected = ALL_CHECKS if checks is None else checks
    report = DoctorReport(path=str(reported_path), fix=fix)
    for check in selected:
        report.checks_run.append(getattr(check, "__name__", str(check)))
        report.findings.extend(check(root))

    if not fix:
        return report

    applied = 0
    for finding in report.findings:
        if finding.fix is None:
            continue
        finding.fix()
        finding.fixed = True
        applied += 1

    if applied:
        # The writer's finalize order: hash BEFORE consolidating, so the new hash
        # lands inside .zmetadata too. Consolidated metadata SHADOWS the per-node
        # .zattrs a fix just wrote, so skipping this would leave every repair
        # invisible to readers while looking applied on disk.
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        _stamp_content_hash(root)
        consolidate(root)
        # Then re-diagnose. A repair is not always a cure: removing a misleading
        # tree from parts that cannot be ordered exactly leaves the lesser
        # "no split planes" condition behind, and a run that called every fix
        # would otherwise report a clean bill of health (and exit 0) for a store
        # the very next run condemns.
        report.residual = [f for check in selected for f in check(root)]
    return report
