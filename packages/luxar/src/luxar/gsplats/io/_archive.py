"""Safe extraction of a compressed ``.gsplats.zarr`` archive.

Single shared helper used by every reader that accepts a ``.gsplats.zarr.zip``
or ``.gsplats.zarr.tar.gz`` (the live loader, the format migrator, and the CLI
inspect/serve commands). Consolidated here so the security-critical extraction
logic exists in exactly one place and cannot drift between call sites.

Threat model (CVE-2007-4559 and symlink-escape): a ``.gsplats.zarr`` archive
legitimately contains only regular files and directories. Every member is
validated *before* a single byte is extracted; symlinks, hardlinks, devices and
FIFOs are rejected outright (a symlink member pointing outside the extraction
dir followed by a file written "through" it escapes name-only validation), and
member-count / total-uncompressed-size caps bound archive-bomb exposure. On any
validation or extraction failure the temp directory is removed so a rejected
malicious archive leaves nothing behind.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

__all__ = ["extract_compressed_zarr"]

#: Reject archives with more members than this (archive-bomb guard).
_MAX_MEMBERS = 5_000_000
#: Reject archives whose declared total uncompressed size exceeds this (256 GiB).
_MAX_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024**3


def _is_zip(path: Path) -> bool:
    return path.suffix == ".zip" or str(path).endswith(".gsplats.zarr.zip")


def _is_targz(path: Path) -> bool:
    return path.suffix == ".gz" or str(path).endswith(
        (".tar.gz", ".gsplats.zarr.tar.gz")
    )


def _extract_zip(path: Path, temp_dir: Path) -> None:
    temp_dir_resolved = temp_dir.resolve()
    with zipfile.ZipFile(path, "r") as zip_ref:
        members = zip_ref.namelist()
        if len(members) > _MAX_MEMBERS:
            raise ValueError(
                f"Zip archive has {len(members)} members (> {_MAX_MEMBERS}); refusing"
            )
        total = 0
        for info in zip_ref.infolist():
            total += info.file_size
            if total > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "Zip archive's declared uncompressed size exceeds the cap; refusing"
                )
            name = info.filename
            # Reject absolute paths, parent traversal, and backslash separators.
            if "\\" in name or name.startswith("/"):
                raise ValueError(f"Zip member '{name}' has unsafe path separator")
            member_path = (temp_dir / name).resolve()
            try:
                member_path.relative_to(temp_dir_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"Zip member '{name}' would escape extraction directory"
                ) from exc
        # Every member was validated above (safe separators + no escape).
        zip_ref.extractall(temp_dir)  # nosec B202


def _extract_targz(path: Path, temp_dir: Path) -> None:
    temp_dir_resolved = temp_dir.resolve()
    with tarfile.open(path, "r:gz") as tar_ref:
        members = tar_ref.getmembers()
        if len(members) > _MAX_MEMBERS:
            raise ValueError(
                f"Tar archive has {len(members)} members (> {_MAX_MEMBERS}); refusing"
            )
        total = 0
        for member in members:
            total += member.size
            if total > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "Tar archive's total uncompressed size exceeds the cap; refusing"
                )
            # A legitimate .gsplats.zarr archive is only regular files and
            # directories. Reject links/devices/FIFOs outright: a symlink member
            # pointing outside temp_dir followed by a file written "through" it
            # escapes name-only validation (the file's own name resolves inside).
            if (
                member.issym()
                or member.islnk()
                or member.isdev()
                or member.ischr()
                or member.isblk()
                or member.isfifo()
            ):
                raise ValueError(
                    f"Tar member '{member.name}' is a link/device; refusing "
                    "(gsplats archives must contain only regular files)"
                )
            member_path = (temp_dir / member.name).resolve()
            try:
                member_path.relative_to(temp_dir_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"Tar member '{member.name}' would escape extraction directory"
                ) from exc
        # Every member was validated above (no links/devices, no escape).
        tar_ref.extractall(temp_dir)  # nosec B202


def extract_compressed_zarr(compressed_path: Path) -> Path:
    """Extract a compressed ``.gsplats.zarr`` archive to a fresh temp directory.

    Args:
        compressed_path: Path to a ``.gsplats.zarr.zip`` or ``.gsplats.zarr.tar.gz``.

    Returns:
        Path to the extracted ``.gsplats.zarr`` directory (inside a new temp dir
        the caller is responsible for cleaning up on success).

    Raises:
        ValueError: On an unsafe member (link/device, path escape, unsafe
            separator), an oversized archive, an unsupported format, or when no
            ``.gsplats.zarr`` directory is found. The temp directory is removed
            before the error propagates.
    """
    compressed_path = Path(compressed_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="luxar_gsplat_"))
    try:
        if _is_zip(compressed_path):
            _extract_zip(compressed_path, temp_dir)
        elif _is_targz(compressed_path):
            _extract_targz(compressed_path, temp_dir)
        else:
            raise ValueError(f"Unsupported compression format: {compressed_path}")

        # Locate the extracted .gsplats.zarr directory: prefer a .gsplats.zarr
        # suffix, else fall back to the sole top-level directory.
        children = list(temp_dir.iterdir())
        for child in children:
            if child.is_dir() and child.name.endswith(".gsplats.zarr"):
                return child
        if children and children[0].is_dir():
            return children[0]
        raise ValueError(f"No .gsplats.zarr directory found in {compressed_path}")
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
