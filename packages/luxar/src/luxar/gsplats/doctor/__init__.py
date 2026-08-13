"""Diagnose — and, on request, repair — an existing ``.gsplats.zarr`` store.

A dataset can be perfectly loadable and still be missing something a later
Luxar learned to record, or be carrying metadata that has quietly gone stale
under an edit. Those conditions are invisible in the viewer: the scene renders,
just not as well as it should. The doctor is where that class of problem is
named, explained, and — where the correct value is recoverable from the store
itself — fixed in place, without re-fitting.

Read-only by default: :func:`diagnose_store` reports, and only writes when
``fix=True``. Repairs are metadata-level and go through one finalize
(:func:`~luxar.gsplats.io.save_gsplats._stamp_content_hash` then
``zarr.consolidate_metadata``, in the writer's order) so the consolidated
metadata cannot disagree with the per-node attrs it shadows, and the viewer's
persistent cache invalidates on the change.

The checks live in :mod:`.checks`; see its module docstring for how to add one.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import zarr

from .checks import ALL_CHECKS
from .model import Check, DoctorReport, Finding

__all__ = [
    "ALL_CHECKS",
    "Check",
    "DoctorReport",
    "Finding",
    "diagnose_store",
]


def diagnose_store(
    path: "str | Path",
    *,
    fix: bool = False,
    checks: "Optional[List[Check]]" = None,
) -> DoctorReport:
    """Run every check over a ``.gsplats.zarr`` store.

    Parameters
    ----------
    path
        A ``.gsplats.zarr`` **directory**. Compressed ``.zip``/``.tar.gz`` stores
        are rejected when ``fix=True`` for the same reason ``annotate-quality``
        rejects them — extraction is temp-dir based, so there is nothing to
        repair in place; unpack first.
    fix
        Apply the repairs the checks offer. Off by default: a diagnosis should
        never surprise anyone by writing.
    checks
        Override the registry (tests, or a targeted run).

    Returns
    -------
    DoctorReport
        Every finding, each flagged with whether it was repaired.
    """
    from luxar.gsplats.io.save_gsplats import _stamp_content_hash

    path = Path(path)
    if fix and not path.is_dir():
        raise ValueError(
            f"doctor --fix requires an uncompressed .gsplats.zarr directory; got "
            f"{path} (unpack a .zip/.tar.gz first — an archive cannot be repaired "
            f"in place). Without --fix it can still be diagnosed."
        )

    root = zarr.open_group(str(path), mode="r+" if fix else "r")
    fmt = root.attrs.get("format_type")
    if fmt != "gsplats_zarr":
        raise ValueError(
            f"{path} is not a standalone .gsplats.zarr store (format_type={fmt!r}). "
            f"Gsplats embedded in a scene are diagnosed by pointing doctor at the "
            f"source .gsplats.zarr, and repaired by re-exporting the scene."
        )

    selected = ALL_CHECKS if checks is None else checks
    report = DoctorReport(path=str(path), fix=fix)
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
        _stamp_content_hash(root)
        zarr.consolidate_metadata(root.store)
    return report
