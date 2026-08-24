"""Safe zip-member validation and extraction helpers."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath


# Package-internal: imported by bundles and download.
def _validate_zip_member_path(member: str) -> PurePosixPath:
    """Validate a zip member path before reading it from a bundle.

    Zip files always use POSIX-style separators. Reject absolute paths,
    parent-directory traversal, and backslashes to avoid platform-specific
    traversal surprises when bundles are created on Windows.
    """
    if "\\" in member:
        raise ValueError(f"Unsafe zip path with backslash separator: {member!r}")

    path = PurePosixPath(member)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"Unsafe zip path: {member!r}")
    if not path.parts or path.name in ("", "."):
        raise ValueError(f"Invalid zip path: {member!r}")
    return path


# Package-internal: imported by bundles.
def _safe_extract_zip_member(
    zf: zipfile.ZipFile,
    member: str,
    destination: Path,
    *,
    target_name: str | None = None,
) -> Path:
    """Extract one validated zip member under ``destination``.

    The member's archive path is validated, and the final output path is
    resolved to ensure it remains inside ``destination``. ``target_name`` can
    be used to flatten bundle members into the cache root.
    """
    member_path = _validate_zip_member_path(member)
    output_name = target_name if target_name is not None else member_path.as_posix()
    output_path = destination / output_name
    destination_resolved = destination.resolve()
    output_resolved = output_path.resolve()

    try:
        output_resolved.relative_to(destination_resolved)
    except ValueError as exc:
        raise ValueError(f"Unsafe extraction target: {output_name!r}") from exc

    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, output_resolved.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return output_resolved
