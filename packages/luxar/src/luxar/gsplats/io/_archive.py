"""Safe extraction of a compressed ``.gsplats.zarr`` archive.

Single shared helper used by every reader that accepts a ``.gsplats.zarr.zip``
or ``.gsplats.zarr.tar.gz`` (the live loader, the format migrator, and the CLI
inspect/serve commands). Consolidated here so the security-critical extraction
logic exists in exactly one place and cannot drift between call sites.

Also home to :func:`read_archive_root_attrs`, a read-only *peek* that pulls the
root attributes out of such an archive without extracting it — from whichever
metadata document the store's format wrote — same layout and same threat model,
so it belongs beside the extractor rather than in a caller.

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

from luxar._zarr_compat import NODE_ATTR_DOCS, NODE_GROUP_DOCS, attrs_from_node_doc

__all__ = ["extract_compressed_zarr", "read_archive_root_attrs", "resolve_store_path"]

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


#: Suffixes a user-supplied path may carry to mean "the store is inside here".
_COMPRESSED_SUFFIXES = (".zip", ".tar.gz")


def _zip_is_flat_store(path: Path) -> bool:
    """Whether a zip holds the store AT ITS ROOT rather than one directory deep.

    Decided from the central directory alone (``infolist()`` parses the index; no
    member payload is decompressed), on two conditions that must BOTH hold:

    * a zarr GROUP document sits at depth 0 — ``.zgroup`` (format 2) or
      ``zarr.json`` (format 3), the names :data:`NODE_GROUP_DOCS` carries. Only a
      group document counts: ``.zattrs`` also sits beside an ARRAY and may be a
      stray, so it says nothing about a store root being here. That name pins a
      group exactly only at format 2 — format 3's ``zarr.json`` marks a node of
      either kind, so a depth-0 ``zarr.json`` describing an ARRAY reads as flat
      here and then fails at ``open_group`` with ``ContainsArrayError``, which is
      what that input already did, so the imprecision is in the naming rather
      than in the behaviour.
    * no member lies under a top-level ``*.gsplats.zarr/`` directory, so the
      nested layout ``_compress_zarr`` writes keeps winning even in the odd
      archive that carries both (the extractor prefers that directory, and this
      resolver must not disagree with it).

    A corrupt archive raises here (``zipfile.BadZipFile``) exactly as it would
    have from the extraction path.
    """
    with zipfile.ZipFile(path, "r") as zip_ref:
        has_root_group_doc = False
        for info in zip_ref.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if not parts:
                continue
            if len(parts) > 1 and parts[0].endswith(".gsplats.zarr"):
                return False
            if (
                len(parts) == 1
                and not info.is_dir()
                and not _zip_member_is_symlink(info)
                and parts[0] in NODE_GROUP_DOCS
            ):
                has_root_group_doc = True
        return has_root_group_doc


def resolve_store_path(
    path: Path, *, flat_zip_in_place: bool = False
) -> tuple[Path, Optional[Path]]:
    """Resolve a user-supplied path to something ``open_group`` can open.

    ``_compress_zarr`` nests the store one directory deep inside the archive
    (``<name>.gsplats.zarr/.zgroup``, …), so handing the archive itself to
    ``open_group`` finds no group at the store root. Extraction resolves that
    root; the caller owns the returned temp directory and must remove it.

    A ``.tar.gz`` is always extracted: zarr has no tar store, so a flat
    ``.tar.gz`` is unresolvable by either route and stays unsupported.

    Args:
        path: An existing ``.gsplats.zarr`` directory, or a ``.zip`` /
            ``.tar.gz`` archive holding one.
        flat_zip_in_place: Opt in to handling a FLAT zip — one whose store sits
            at the archive ROOT, as ``zip -r x.gsplats.zarr.zip .`` from inside a
            store or a zarr-native ``ZipStore`` write produces — by returning the
            archive itself, which ``open_group`` opens as a ``ZipStore``
            (:func:`_zip_is_flat_store` decides, from the zip index only).
            Otherwise a flat zip takes the extraction path, where the
            "sole top-level directory" fallback picks an ARRAY sub-directory and
            ``open_group`` raises ``ContainsArrayError``.

            Only :func:`~luxar.gsplats.io.inspect_gsplats.inspect_gsplats_zarr`
            opts in, and there it is pure regression-avoidance: it reads metadata
            only, which zarr's ``ZipStore`` has always been able to do for a flat
            zip, so refusing it would be a NEW failure. ``load_gsplat_node``
            deliberately does NOT opt in. Accepting a flat zip there would newly
            succeed on a shape two downstream steps do not handle — the
            appearance peek (:func:`read_archive_root_attrs`) refuses a depth-0
            root by design (:func:`_root_attrs_rank`, the #1608 contract), so a
            rebuild would silently drop the authored appearance; and
            ``luxar gsplat info`` uses the loader only as a gate before
            re-resolving with :func:`extract_compressed_zarr`, which would report
            one part of a partition as the whole dataset. Granting the loader
            that capability is left to a separate follow-up.

    Returns:
        ``(store_path, temp_dir)`` — ``temp_dir`` is ``None`` when there is
        nothing to clean up, i.e. for a directory store and for an in-place flat
        zip.

    Raises:
        ValueError: For a regular file that is not one of the two archive
            formats, or on an unsafe/unusable archive (see
            :func:`extract_compressed_zarr`).
    """
    if (
        flat_zip_in_place
        and _is_zip(path)
        and path.is_file()
        and _zip_is_flat_store(path)
    ):
        return path, None
    if any(str(path).endswith(suffix) for suffix in _COMPRESSED_SUFFIXES):
        zarr_path = extract_compressed_zarr(path)
        return zarr_path, zarr_path.parent
    if path.is_file():
        raise ValueError(
            f"Expected a zarr directory or compressed archive (.zip/.tar.gz), "
            f"got regular file: {path}"
        )
    return path, None


def _root_attrs_rank(name: str) -> Optional[int]:
    """How root-like an attrs member is; ``None`` if it cannot be the root.

    BOTH formats' documents count (:data:`NODE_ATTR_DOCS`): format 2 puts the
    root attributes in ``.zattrs``, format 3 inside ``zarr.json``. A store is
    written in one format or the other, so only one of the two names is ever
    present and they can share a rank without competing. Recognising only
    ``.zattrs`` made this peek answer ``{}`` for every format-3 archive — which
    reads as "this dataset authored no appearance" rather than as an error.

    Lower sorts better. The two ranks mirror, one for one, the two rules
    :func:`extract_compressed_zarr` uses to locate the store inside an archive:

    * ``0`` — ``<name>.gsplats.zarr/<doc>``: the top-level directory whose name
      ends in ``.gsplats.zarr``. The extractor prefers it, and it is what
      ``_compress_zarr`` always writes. Nothing else can outrank it, so it needs
      no further qualification.
    * ``1`` — ``<anything-else>/<doc>``: a candidate for the extractor's
      fallback, the *sole* top-level directory whatever it happens to be called
      (what you get from ``tar czf x.gsplats.zarr.tar.gz mystore``). The rank
      alone does NOT establish that: only :func:`_fallback_is_unambiguous`, which
      needs the whole member list, can — a rank-1 member in an archive with
      several top-level directories is just one candidate among several and is
      refused.

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
    if not parts or parts[-1] not in NODE_ATTR_DOCS:
        return None
    if len(parts) != 2:
        return None
    return 0 if parts[0].endswith(".gsplats.zarr") else 1


def _top_level_dir(name: str) -> Optional[str]:
    """Top-level directory a member lies in/under; ``None`` for a loose member.

    Inferred from the member NAME alone: any name with more than one path
    component implies its first component is a top-level directory. That is what
    makes the two formats agree on the same set even though they disagree about
    directories — a zip written by ``_compress_zarr`` carries no directory entries
    at all (only files), while a tar carries explicit ones, and a tar's
    ``mystore/`` entry adds nothing its ``mystore/.zattrs`` member does not.

    A top-level directory with no descendants of any kind therefore contributes
    nothing, deliberately: it holds no store (``zarr.open_group`` on an empty
    directory raises), so counting it could only manufacture false ambiguity and
    suppress a perfectly good carry (``zip -r out.gsplats.zarr.zip mystore notes``
    with an empty ``notes/``).
    """
    parts = PurePosixPath(name).parts
    if len(parts) > 1:
        return parts[0]
    return None


def _fallback_is_unambiguous(rank: int, top_dirs: set[str]) -> bool:
    """Whether a winning candidate of ``rank`` may actually be used as the root.

    Rank 0 always may. It is the extractor's own first preference, so the two
    agree on the KIND of node — and for an archive holding ONE store, which is
    every archive ``_compress_zarr`` writes, on the same node. They need not agree
    on WHICH when an archive holds several ``*.gsplats.zarr`` directories: that is
    not one dataset, the extractor picks among them arbitrarily (``iterdir()``
    order), and the peek picks the first in archive-index order. Rank 1 stands in
    for the extractor's *fallback*, the SOLE top-level directory, and there
    refusing costs nothing — so with two of them the peek refuses rather than
    guessing. Carrying nothing is the safe outcome (the appearance simply is not
    carried, as before the carry existed); carrying a sibling directory's attrs
    would author an appearance from something that is not the dataset.
    """
    if rank == _BEST_ROOT_RANK:
        return True
    return len(top_dirs) == 1


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    """Whether a zip member carries a unix symlink mode (never follow one)."""
    return stat.S_ISLNK(info.external_attr >> 16)


def _parse_attrs(raw: bytes) -> Dict[str, Any]:
    """Decode a node metadata payload; anything but a JSON object yields ``{}``.

    Both formats' documents arrive here, so the unwrapping is the facade's:
    a format-2 ``.zattrs`` IS the attributes, while a format-3 ``zarr.json``
    nests them under ``attributes`` beside the structural fields.
    """
    return attrs_from_node_doc(json.loads(raw.decode("utf-8")))


def _read_zip_root_attrs(path: Path) -> Dict[str, Any]:
    """Root attrs of a zip archive, reading that one member's bytes only."""
    with zipfile.ZipFile(path, "r") as zip_ref:
        best: Optional[zipfile.ZipInfo] = None
        best_rank = 0
        top_dirs: set[str] = set()
        # infolist() parses the central directory only — no member payload is
        # decompressed by this scan.
        for info in zip_ref.infolist():
            is_dir = info.is_dir()
            top = _top_level_dir(info.filename)
            if top is not None:
                top_dirs.add(top)
            if is_dir or _zip_member_is_symlink(info):
                continue
            rank = _root_attrs_rank(info.filename)
            if rank is None:
                continue
            if best is None or rank < best_rank:
                best, best_rank = info, rank
        if best is None or not _fallback_is_unambiguous(best_rank, top_dirs):
            return {}
        if best.file_size > _MAX_ATTRS_BYTES:
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


#: The best possible ``_root_attrs_rank``: the root attrs document of a
#: top-level ``*.gsplats.zarr`` directory. Nothing can outrank it and it needs no
#: whole-archive context to be usable, so the tar walk can stop the moment it
#: sees one.
#:
#: HOW EARLY that stop comes depends on the on-disk format, because
#: ``tarfile.add`` walks a directory in sorted order. A format-2 store's
#: ``.zattrs`` is a dotfile and lands second (measured: member 1 of 24), so the
#: walk really does end after a couple of headers. A format-3 store's document is
#: ``zarr.json``, which sorts AFTER every array sub-directory and lands last
#: (measured: member 26 of 27) — so on the format Luxar now writes, the stop
#: effectively never fires and the peek costs a full inflate. Correct either way,
#: and free for a zip (its central directory is an index); reordering the tar at
#: write time would only help archives written from here on, so it is not done.
#:
#: The unnamed fallback layout gets no such stop in either format, twice over: a
#: rank-1 member can still be superseded by a rank-0 one later in the stream, and
#: whether it may be used at all depends on the archive's full set of top-level
#: directories (:func:`_fallback_is_unambiguous`). So that walk has to reach the
#: end of the header stream — and a gzipped tar has no index, so reaching the end
#: means inflating the whole file.
_BEST_ROOT_RANK = 0


def _read_targz_root_attrs(path: Path) -> Dict[str, Any]:
    """Root attrs of a tar.gz archive, reading that one member's bytes only.

    Member HEADERS are walked lazily (never ``getmembers()``, which materializes
    the whole archive), and only the winning member's payload is read. The walk
    stops as soon as a member reaches :data:`_BEST_ROOT_RANK`; failing that it
    runs to the end of the header stream (see :data:`_BEST_ROOT_RANK` for what
    that costs, and why a format-3 archive nearly always pays it), because a
    better-ranked member may appear after a worse one, the fallback tier cannot
    be settled before the archive's top-level directories are all known, and
    reading the wrong node's attrs as the root's would silently author an
    appearance nobody asked for.
    """
    with tarfile.open(path, "r:gz") as tar_ref:
        best: Optional[tarfile.TarInfo] = None
        best_rank = 0
        top_dirs: set[str] = set()
        for member in tar_ref:
            top = _top_level_dir(member.name)
            if top is not None:
                top_dirs.add(top)
            # Only regular files: a symlink named `.zattrs` is never followed
            # (`extractfile` resolves an in-archive link target, so dropping this
            # check would hand back whatever node the link points at).
            if not member.isfile():
                continue
            rank = _root_attrs_rank(member.name)
            if rank is None:
                continue
            if best is None or rank < best_rank:
                best, best_rank = member, rank
                if best_rank == _BEST_ROOT_RANK:
                    break
        if best is None or not _fallback_is_unambiguous(best_rank, top_dirs):
            return {}
        if best.size > _MAX_ATTRS_BYTES:
            return {}
        handle = tar_ref.extractfile(best)
        if handle is None:
            return {}
        with handle:
            return _parse_attrs(handle.read())


def read_archive_root_attrs(path: str | Path) -> Dict[str, Any]:
    """Read the ROOT attributes of a compressed ``.gsplats.zarr``, no extraction.

    Only ONE member's payload is read: the archive index (zip central directory /
    tar headers) is scanned for the root attrs document, and nothing else is
    decompressed. Nothing is ever written to disk and no link is ever followed.
    Scanning the index is free for a zip but not for a gzipped tar, which has no
    index at all — see :data:`_BEST_ROOT_RANK` for when that walk stops early
    (rarely, at format 3).

    Which member counts as the root follows :func:`_root_attrs_rank` and
    :func:`_fallback_is_unambiguous`, which pick the same KIND of node
    :func:`extract_compressed_zarr` does: a top-level ``*.gsplats.zarr``
    directory, as ``_compress_zarr`` writes, else — only when it is the archive's
    SOLE top-level directory — that directory whatever it is named. For an archive
    holding ONE store, which is every archive ``_compress_zarr`` writes, that is
    the same node the extractor loads. An archive holding SEVERAL
    ``*.gsplats.zarr`` directories is not a single dataset; the extractor picks
    one of them arbitrarily (``iterdir()`` order) and the peek may pick another.
    Either way a child group's attrs (``<top>/fitting/.zattrs``) is never read as
    the root's, and neither is a stray ``.zattrs`` loose at the archive root.

    An archive with several top-level directories and no ``*.gsplats.zarr``-named
    one is ambiguous and yields ``{}``: the extractor resolves that case by
    ``iterdir()`` order, which an archive index cannot predict, so the peek would
    have to guess — and carrying nothing is merely the status quo, while carrying
    a sibling directory's attrs would author an appearance from the wrong node.

    Where a member name occurs twice, the FIRST occurrence wins here while
    ``extractall`` keeps the last; such an archive is malformed (both occurrences
    are the same PATH — the store root's own ``.zattrs`` — so no foreign node's
    attrs can be reached either way), and the early stop is worth keeping.

    Reading ``.zattrs`` directly — not consolidated ``.zmetadata`` — is the
    correct source, for an invariant rather than a version: the peek must see
    exactly what the loader's own group-open sees, and the loader never opens a
    store with consolidated metadata (under zarr 2 that takes an explicit
    ``open_consolidated``; the zarr 3 path disables it deliberately). So per-node
    ``.zattrs`` is authoritative even for a store that also carries a
    ``.zmetadata``, and preferring ``.zmetadata`` would introduce a divergence
    from the directory-store path, not fix one.

    Args:
        path: Path to a ``.gsplats.zarr.zip`` or ``.gsplats.zarr.tar.gz``.

    Returns:
        The root attrs as a dict. ``{}`` when the path is missing or is not one of
        the two supported archive formats, when no root attrs member exists,
        when the store root is ambiguous (above), or when the payload is oversized
        or not a JSON object.

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
