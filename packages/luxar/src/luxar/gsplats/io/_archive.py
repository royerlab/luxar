"""Safe extraction of a compressed ``.gsplats.zarr`` archive.

Single shared helper used by every reader that accepts a ``.gsplats.zarr.zip``
or ``.gsplats.zarr.tar.gz`` (the live loader, the format migrator, and the CLI
inspect/serve commands). Consolidated here so the security-critical extraction
logic exists in exactly one place and cannot drift between call sites.

Also home to :func:`read_archive_root_attrs`, a read-only *peek* that pulls the
root ``.zattrs`` out of such an archive without extracting it — same layout and
same threat model, so it belongs beside the extractor rather than in a caller.

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

import json
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

__all__ = ["extract_compressed_zarr", "read_archive_root_attrs"]

#: Reject archives with more members than this (archive-bomb guard).
_MAX_MEMBERS = 5_000_000
#: Reject archives whose declared total uncompressed size exceeds this (256 GiB).
_MAX_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024**3
#: Refuse a ``.zattrs`` bigger than this (4 MiB). A zarr group's attrs are a
#: small JSON object; anything this large is not attrs, and reading it into
#: memory during a best-effort peek is not a cost worth paying.
_MAX_ATTRS_BYTES = 4 * 1024**2


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


def _root_zattrs_rank(name: str) -> Optional[int]:
    """How root-like a ``.zattrs`` member is; ``None`` if it cannot be the root.

    Lower sorts better. The two ranks mirror, one for one, the two rules
    :func:`extract_compressed_zarr` uses to locate the store inside an archive,
    so the peek and a real extraction can never disagree about which node's attrs
    they are looking at:

    * ``0`` — ``<name>.gsplats.zarr/.zattrs``: the top-level directory whose name
      ends in ``.gsplats.zarr``. The extractor prefers it, and it is what
      ``_compress_zarr`` always writes.
    * ``1`` — ``<anything-else>/.zattrs``: the extractor's fallback, the sole
      top-level directory whatever it happens to be called (what you get from
      ``tar czf x.gsplats.zarr.tar.gz mystore``).

    Every other member is not a candidate at all. A deeper one is a child group's
    attrs (``<top>/lod_0/.zattrs``) — a different object, and authoring an
    appearance from it is exactly the failure this ranking exists to prevent. A
    depth-0 one is not the store root either: the extractor requires a top-level
    *directory* and raises for a flat archive, and this peek only ever runs
    alongside a load that succeeded, so a loose top-level ``.zattrs`` is a stray.
    Ranking it would let it beat the real ``mystore/.zattrs`` in the fallback
    layout and hand back attrs belonging to something that is not the dataset.
    """
    parts = PurePosixPath(name).parts
    if not parts or parts[-1] != ".zattrs":
        return None
    if len(parts) != 2:
        return None
    return 0 if parts[0].endswith(".gsplats.zarr") else 1


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    """Whether a zip member carries a unix symlink mode (never follow one)."""
    return stat.S_ISLNK(info.external_attr >> 16)


def _parse_attrs(raw: bytes) -> Dict[str, Any]:
    """Decode a ``.zattrs`` payload; anything but a JSON object yields ``{}``."""
    parsed = json.loads(raw.decode("utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def _read_zip_root_attrs(path: Path) -> Dict[str, Any]:
    """Root ``.zattrs`` of a zip archive, reading that one member's bytes only."""
    with zipfile.ZipFile(path, "r") as zip_ref:
        best: Optional[zipfile.ZipInfo] = None
        best_rank = 0
        # infolist() parses the central directory only — no member payload is
        # decompressed by this scan.
        for info in zip_ref.infolist():
            if info.is_dir() or _zip_member_is_symlink(info):
                continue
            rank = _root_zattrs_rank(info.filename)
            if rank is None:
                continue
            if best is None or rank < best_rank:
                best, best_rank = info, rank
        if best is None or best.file_size > _MAX_ATTRS_BYTES:
            return {}
        with zip_ref.open(best, "r") as handle:
            # Bound the read itself rather than trusting the size check above.
            # The extra byte cannot actually arrive: `ZipExtFile` truncates the
            # decompressed stream at the declared `file_size`, and a member whose
            # central-directory size was tampered with raises `BadZipFile` on the
            # CRC instead (absorbed by the best-effort caller).
            raw = handle.read(_MAX_ATTRS_BYTES + 1)
        if len(raw) > _MAX_ATTRS_BYTES:
            return {}
        return _parse_attrs(raw)


#: The best possible ``_root_zattrs_rank``: the ``.zattrs`` of a top-level
#: ``*.gsplats.zarr`` directory. Nothing can outrank it, so the tar walk can stop
#: the moment it sees one — which is what keeps the peek cheap on the layout
#: ``_compress_zarr`` always writes, where the member turns up within the first
#: couple of headers (``tarfile.add`` walks a directory in sorted order, so
#: dotfiles come first). The unnamed fallback layout gets no such stop: a rank-1
#: member can still be superseded by a rank-0 one later in the stream, so the walk
#: has to reach the end of the header stream — and a gzipped tar has no index, so
#: that means inflating the whole file (measured ~0.2 s for a 315 MB archive,
#: ~35% on top of the extraction the caller performs anyway).
_BEST_ROOT_RANK = 0


def _read_targz_root_attrs(path: Path) -> Dict[str, Any]:
    """Root ``.zattrs`` of a tar.gz archive, reading that one member's bytes only.

    Member HEADERS are walked lazily (never ``getmembers()``, which materializes
    the whole archive), and only the winning member's payload is read. The walk
    stops as soon as a member reaches :data:`_BEST_ROOT_RANK`; failing that it
    runs to the end of the header stream (see :data:`_BEST_ROOT_RANK` for what
    that costs), because a better-ranked ``.zattrs`` may appear after a worse one
    and reading the wrong node's attrs as the root's would silently author an
    appearance nobody asked for.
    """
    with tarfile.open(path, "r:gz") as tar_ref:
        best: Optional[tarfile.TarInfo] = None
        best_rank = 0
        for member in tar_ref:
            # Only regular files: a symlink named `.zattrs` is never followed.
            if not member.isfile():
                continue
            rank = _root_zattrs_rank(member.name)
            if rank is None:
                continue
            if best is None or rank < best_rank:
                best, best_rank = member, rank
                if best_rank == _BEST_ROOT_RANK:
                    break
        if best is None or best.size > _MAX_ATTRS_BYTES:
            return {}
        handle = tar_ref.extractfile(best)
        if handle is None:
            return {}
        with handle:
            return _parse_attrs(handle.read())


def read_archive_root_attrs(path: str | Path) -> Dict[str, Any]:
    """Read the ROOT ``.zattrs`` of a compressed ``.gsplats.zarr``, no extraction.

    Only ONE member's payload is read: the archive index (zip central directory /
    tar headers) is scanned for the root ``.zattrs``, and nothing else is
    decompressed. Nothing is ever written to disk and no link is ever followed.
    Scanning the index is free for a zip but not for a gzipped tar, which has no
    index at all — see :data:`_BEST_ROOT_RANK` for when that walk stops early.

    Which member counts as the root follows :func:`_root_zattrs_rank`, which
    mirrors :func:`extract_compressed_zarr`'s own choice of store root: the
    top-level ``*.gsplats.zarr`` directory that ``_compress_zarr`` writes, else
    the sole top-level directory whatever it is named. So a child group's attrs
    (``<top>/fitting/.zattrs``) is never read as the root's, and neither is a
    stray ``.zattrs`` loose at the archive root.

    Reading ``.zattrs`` directly — not consolidated ``.zmetadata`` — is the
    correct source: this repo pins zarr 2.18.x, where ``zarr.open_group`` reads
    per-node ``.zattrs`` and ignores ``.zmetadata`` unless opened via
    ``open_consolidated``. So this matches exactly what the directory-store path
    sees, and preferring ``.zmetadata`` would introduce a divergence, not fix one.

    Args:
        path: Path to a ``.gsplats.zarr.zip`` or ``.gsplats.zarr.tar.gz``.

    Returns:
        The root attrs as a dict. ``{}`` when the path is missing or is not one of
        the two supported archive formats, when no ``.zattrs`` member exists, or
        when the payload is oversized or not a JSON object.

    Raises:
        OSError, zipfile.BadZipFile, tarfile.TarError, UnicodeDecodeError,
        json.JSONDecodeError: A corrupt or unreadable archive propagates; callers
        peeking best-effort (see
        ``luxar.gsplats.io.load_gsplats.read_authored_appearance``) catch it.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    if _is_zip(p):
        return _read_zip_root_attrs(p)
    if _is_targz(p):
        return _read_targz_root_attrs(p)
    return {}
