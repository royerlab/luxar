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

import shutil
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
        A ``.gsplats.zarr`` directory, or a ``.zip``/``.tar.gz`` archive. An
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
        Every finding, each flagged with whether it was repaired.
    """
    path = Path(path)
    if fix and not path.is_dir():
        raise ValueError(
            f"doctor --fix requires an uncompressed .gsplats.zarr directory; got "
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
    from luxar.gsplats.io.save_gsplats import _stamp_content_hash

    root = zarr.open_group(str(store_path), mode="r+" if fix else "r")
    fmt = root.attrs.get("format_type")
    if fmt != "gsplats_zarr":
        raise ValueError(
            f"{reported_path} is not a standalone .gsplats.zarr store "
            f"(format_type={fmt!r}). "
            f"Gsplats embedded in a scene are diagnosed by pointing doctor at the "
            f"source .gsplats.zarr, and repaired by re-exporting the scene."
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
        _stamp_content_hash(root)
        zarr.consolidate_metadata(root.store)
    return report
