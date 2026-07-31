"""Atomic directory copy: writes to a sibling temp dir, renames on success.

`shutil.copytree` writes directly to the destination, so a failure mid-copy
(disk full, permission error, SIGKILL) leaves a half-written tree under the
final path. Callers that promise atomicity to their users — `Scene.to_zarr`
and the CLI `luxar export` — must instead copy to a sibling temp directory
and atomically rename on success. This module provides those helpers
(``atomic_copytree`` for directory trees, ``atomic_copy_file`` for single files).
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


def atomic_copytree(src: Path, dst: Path) -> None:
    """Copy the directory tree at ``src`` to ``dst`` atomically.

    Writes to a sibling temp directory ``dst.parent/.tmp_<dst.name>_<uuid>``,
    then performs an atomic ``os.replace(tmp, dst)`` on success. On failure
    the temp directory is removed and the exception is re-raised, so the
    destination either exists in full or does not exist at all.

    Args:
        src: Source directory to copy. Must exist.
        dst: Destination path. Must NOT already exist — the caller is
            responsible for clearing it if overwrite is desired (matches
            ``shutil.copytree``'s default behaviour).

    Raises:
        FileExistsError: ``dst`` already exists.
        FileNotFoundError: ``src`` does not exist or isn't a directory.
        OSError: Any I/O failure during the copy. The temp directory is
            cleaned up before re-raising.

    Note:
        ``os.replace`` is atomic on POSIX when the source and destination
        live on the same filesystem (always true here — we use a sibling
        temp dir). On Windows, ``os.replace`` is atomic for files but the
        directory rename is implemented via ``MoveFileExW``, which gives
        the same atomicity guarantees in practice.
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src}")
    if dst.exists():
        raise FileExistsError(
            f"Destination already exists: {dst}. Remove it first or choose "
            "a different path."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".tmp_{dst.name}_{uuid.uuid4().hex[:8]}"

    try:
        shutil.copytree(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def atomic_copy_file(src: Path, dst: Path) -> Path:
    """Copy the FILE ``src`` onto ``dst`` atomically (temp sibling + rename).

    ``shutil.copy2`` writes straight into ``dst``, so an interruption (Ctrl-C,
    full disk, SIGKILL) leaves a truncated file under the final name — which a
    later run may mistake for a complete cache entry, or must quarantine and
    re-fetch. Copying to a sibling temp file and renaming makes ``dst`` appear
    only once it is complete. The temp file's contents are flushed with
    ``os.fsync`` before the rename, so ``dst`` cannot surface with unwritten
    blocks under a complete-looking name after a crash: ``os.replace`` orders
    the rename against other renames, not against the preceding writes.

    Unlike :func:`atomic_copytree`, an existing ``dst`` IS replaced: every caller
    is refreshing a cache entry and wants overwrite semantics. Metadata is
    preserved (``copy2``), so mtime-based staleness checks keep working.

    Args:
        src: Existing regular file to copy.
        dst: Destination path; replaced if present. Parents are created.

    Returns:
        ``dst``.

    Raises:
        FileNotFoundError: ``src`` is missing or is not a regular file.
        OSError: Any I/O failure. The temp file is removed before re-raising, so
            ``dst`` keeps whatever it had before the call.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".tmp_{dst.name}_{uuid.uuid4().hex[:8]}"

    try:
        shutil.copy2(src, tmp)
        # Flush the copied bytes before the rename so dst cannot point at
        # unwritten blocks after a crash (see docstring). Open read+write, not
        # read-only: os.fsync maps to FlushFileBuffers on Windows, which needs
        # a write-access handle.
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dst
