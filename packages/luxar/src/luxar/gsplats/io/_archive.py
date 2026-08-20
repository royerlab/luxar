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
import warnings
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Any, Callable, Dict, Optional

from luxar._zarr_compat import (
    NODE_ATTR_DOCS,
    NODE_GROUP_DOCS,
    V3_NODE_DOC,
    attrs_from_node_doc,
)

__all__ = ["extract_compressed_zarr", "read_archive_root_attrs", "resolve_store_path"]

#: Reject archives with more members than this (archive-bomb guard).
_MAX_MEMBERS = 5_000_000
#: Reject archives whose declared total uncompressed size exceeds this (256 GiB).
_MAX_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024**3
#: Refuse ATTRIBUTES bigger than this (4 MiB) — the user-authored mapping this
#: peek hands back, once any format-3 document has been unwrapped to it. A zarr
#: node's attrs really are a small JSON object (a Luxar root carries a couple of
#: dozen scalars); anything this large is not attrs, and materializing it during
#: a best-effort peek is not a cost worth paying.
#:
#: Applied in two places, which is the point: as the member budget for a
#: format-2 ``.zattrs``, which literally IS the attributes mapping, and again
#: after parsing (:func:`_parse_attrs`) — but there ONLY on the branch that was
#: read under the raised document budget below, so that raising THAT budget
#: cannot raise what this peek can hand a caller. A ``.zattrs`` is not re-checked:
#: its member budget already IS this number, and a second check measuring a
#: re-serialization rather than the bytes on disk can only ever refuse something
#: the member budget admitted (see :func:`_parse_attrs`).
_MAX_ATTRS_BYTES = 4 * 1024**2

#: Refuse a format-3 node DOCUMENT (``zarr.json``) bigger than this (128 MiB).
#:
#: A separate and much larger budget, because a ``zarr.json`` is not the
#: attributes: it carries the node's structural fields beside them and — at a
#: CONSOLIDATED store root, which every Luxar store is — the whole consolidated
#: index of every descendant, nested under ``consolidated_metadata``. So the
#: document grows with the SHAPE OF THE TREE while its ``attributes`` stay a
#: handful of scalars, and the attrs budget above is simply the wrong ruler for
#: it. (This is why the two budgets exist at all: before the move to zarr format
#: 3 the root document WAS ``.zattrs``, so one number covered both.)
#:
#: The index has one entry per NODE, not per part, so the rate depends on the
#: SHAPE of each part and not only on how many there are. Measured on real
#: ``write_gsplats_tree`` partitions at format 3, as bytes of root ``zarr.json``
#: per part:
#:
#: * bare leaf part (one additive sub-LOD): 8,260–8,429 B (a real
#:   ``gsplat partition``: 8,996 B) — 4 MiB crossed at ~450–510 parts, 128 MiB at
#:   ~14,000–16,300.
#: * per-part 6-step ``stream`` ladder: 48,348–48,507 B — 4 MiB at ~87 parts,
#:   128 MiB at ~2,776.
#: * per-part 3 levels x 4 sub-LODs (``adaptive``-shaped): 99,695 B — 4 MiB at
#:   ~42 parts, 128 MiB at ~1,346.
#:
#: So the old single budget was far MORE reachable than a part count suggests:
#: the recommended ``--recipe stream|levels|adaptive`` multiply nodes per part by
#: 6-12x, and a 42-part ``adaptive`` partition already crossed 4 MiB. Past that
#: point this peek's ``{}`` reads as "this dataset authored no appearance" rather
#: than as an error, so every rewriting command silently overwrote a
#: zipped/tarred partition's appearance with the writer's defaults while the SAME
#: tree as a directory store answered correctly. 128 MiB buys ~1,300-2,800 parts
#: of headroom on the laddered shapes people actually publish (and ~16,000 on
#: bare leaves) — real headroom, not unlimited, which is why a refusal now warns
#: loudly (:func:`_warn_size_refusal`) instead of answering ``{}`` in silence.
#:
#: Not larger, because this is still an archive-bomb bound and the document is
#: parsed whole: 128 MiB of JSON materializes on the order of a gigabyte of
#: Python objects, and that parse — not the byte count — is the real ceiling.
#:
#: Raising it does not widen the PRODUCT's exposure, which was verified rather
#: than assumed: every caller of the peek also LOADS the same store, and
#: ``luxar._zarr_compat.open_group`` opts reads out of consolidated metadata but
#: still reads and parses the root document whole. Measured on an 8,384,150-byte
#: consolidated root — one ``LocalStore.get`` returning all 8,384,150 bytes, a
#: single ``json.loads`` over all of them, 32.6 MiB tracemalloc peak (~4x the
#: document) — so zarr materializes what this peek does, and more, for the same
#: store. The exposure that IS new belongs to a caller invoking
#: :func:`read_archive_root_attrs` directly on an untrusted archive and never
#: opening it: that caller gets the raised ceiling with no load behind it.
_MAX_NODE_DOC_BYTES = 128 * 1024**2


def _max_member_bytes(member_name: str) -> int:
    """Byte budget for a root metadata member, chosen by the document's NAME.

    The name is the whole signal, and it is exact — it is also how the member was
    selected in the first place (:func:`_root_attrs_rank`). A format-2
    ``.zattrs`` is the attributes mapping itself and keeps the small
    :data:`_MAX_ATTRS_BYTES` budget; nothing about format 2 changed. A format-3
    ``zarr.json`` is a whole node document carrying the consolidated index and
    gets :data:`_MAX_NODE_DOC_BYTES`.

    The name comes from the facade (:data:`~luxar._zarr_compat.V3_NODE_DOC`),
    which owns metadata-document names, rather than being spelled again here or
    read positionally out of :data:`NODE_ATTR_DOCS` — a positional read would let
    a reordering of that tuple silently swap the two budgets.

    This rides on the SELECTION above, and inherits its one known limitation: a
    root carrying BOTH documents ranks them equally, so the tie-break is archive
    order and the peek budgets whichever one it happened to select, while
    ``open_group`` resolves format 3 either way. No Luxar writer produces that
    state and #1600 deliberately defers changing it (preferring the v3 document
    at equal rank would cost the tar walk its rank-0 early stop), so this is a
    recorded limitation and not an invariant being relied on.

    Raising the document budget does not raise what the peek can hand back:
    :func:`_parse_attrs` re-checks the unwrapped attributes against the small
    budget on exactly that branch.
    """
    return (
        _MAX_NODE_DOC_BYTES
        if PurePosixPath(member_name).name == V3_NODE_DOC
        else _MAX_ATTRS_BYTES
    )


def _warn_size_refusal(
    archive_path: Path,
    member_name: str,
    what: str,
    measured_bytes: int,
    budget_bytes: int,
    budget_name: str,
) -> None:
    """Say out loud that a size budget just cost this archive its appearance.

    Every refusal in this peek returns ``{}``, and ``{}`` is indistinguishable
    from "this dataset authored no appearance" at every layer above:
    ``read_authored_appearance`` filters the mapping and only reports a NON-empty
    carry, so a silent refusal reaches the user as a rebuild that quietly reset
    the look — the exact undiagnosable symptom #1600 point 4 objects to, and the
    reason the budgets themselves had to be revisited. The budgets are larger now
    but still finite, so the next person to cross one gets a message naming the
    archive, the member, the measured size and the budget instead of nothing.

    A warning rather than console output because this is a library read path
    (``luxar.gsplats.io`` warns; it does not print). Note the caller
    ``load_gsplats.read_authored_appearance`` wraps this whole peek in
    ``except Exception``, so under ``-W error`` the promoted warning is swallowed
    there and the carry degrades to the same ``{}`` — louder is not available on
    that path, and turning a best-effort carry into a hard failure is not wanted.
    """
    warnings.warn(
        f"Appearance peek refused the {what} '{member_name}' in {archive_path}: "
        f"{measured_bytes:,} bytes exceeds the {budget_name} budget of "
        f"{budget_bytes:,} bytes. This archive's ROOT ATTRIBUTES will not be "
        "read, which every caller sees as 'no appearance was authored' — a "
        "command rewriting this dataset will therefore write its own appearance "
        "defaults over the authored look. Extract the archive and pass the "
        "directory store to carry it.",
        UserWarning,
        stacklevel=2,
    )


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


def _parse_attrs(
    raw: bytes,
    member_name: str,
    *,
    member_budget: int,
    archive_path: Path,
) -> Dict[str, Any]:
    """Decode a node metadata payload; anything but a JSON object yields ``{}``.

    Both formats' documents arrive here, so the unwrapping is the facade's:
    a format-2 ``.zattrs`` IS the attributes, while a format-3 ``zarr.json``
    nests them under ``attributes`` beside the structural fields. The member's
    own name is what settles which — it is how the member was selected in the
    first place — so the facade is told rather than left to infer it from the
    content.

    The unwrapped ATTRIBUTES are then capped at :data:`_MAX_ATTRS_BYTES`, but
    ONLY when ``member_budget`` was the RAISED one: a format-3 document is
    allowed :data:`_MAX_NODE_DOC_BYTES` because of the consolidated index it
    carries, and that allowance must not become licence to hand a caller a
    100 MiB attrs mapping. A member read under the small budget needs no second
    check — it was already bounded by this very number — and a second check here
    measures a DIFFERENT currency, so on that branch it could only ever refuse
    something the member budget admitted. Concretely: this re-serializes, and a
    1,800,044-byte non-ASCII ``.zattrs`` (comfortably inside 4 MiB, and read fine
    before this peek grew a second budget) re-serializes to 5,400,044 bytes and
    would be refused — a regression straight back into the silent-``{}`` failure
    this whole change exists to fix.

    The re-serialization is therefore COMPACT and faithful — ``ensure_ascii``
    off, no separator padding — so the number it produces is the closest thing to
    "the bytes these attributes are" that is available after ``json.loads`` has
    run. Measuring by re-serializing at all is honest about the cost: the
    document is already parsed and resident by this point, so this is one more
    pass over data we are holding anyway, and no incremental accounting is
    possible once the parse has happened.
    """
    attrs = attrs_from_node_doc(
        json.loads(raw.decode("utf-8")),
        doc_name=PurePosixPath(member_name).name,
    )
    if member_budget <= _MAX_ATTRS_BYTES:
        return attrs
    measured = len(
        json.dumps(attrs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if measured > _MAX_ATTRS_BYTES:
        _warn_size_refusal(
            archive_path,
            member_name,
            "attributes unwrapped from",
            measured,
            _MAX_ATTRS_BYTES,
            "attributes",
        )
        return {}
    return attrs


def _read_selected_member(
    archive_path: Path,
    member_name: str,
    declared_size: int,
    open_payload: Callable[[], Optional[IO[bytes]]],
) -> Dict[str, Any]:
    """Read the ONE member both peeks settled on, under its name's budget.

    The zip and tar walks differ entirely — a central directory versus a lazy
    header stream — but everything after the winner is chosen is identical, so it
    lives here once: budget by DOCUMENT NAME (:func:`_max_member_bytes`; a
    format-3 ``zarr.json`` root carries the whole consolidated index and is
    orders of magnitude bigger than the attrs it nests), refuse on the size the
    ARCHIVE declares, then bound the read itself rather than trusting that
    declaration, then parse.

    ``open_payload`` is a zero-argument callable rather than an already-open
    handle so nothing is decompressed for a member that the declared-size gate is
    about to refuse; it may return ``None``, which is what ``tarfile``'s
    ``extractfile`` gives for a member with no payload.

    The bounding read (``cap + 1``) cannot actually over-deliver in either
    format — ``ZipExtFile`` truncates the decompressed stream at the declared
    ``file_size`` (a tampered central-directory size raises ``BadZipFile`` on the
    CRC instead) and ``ExFileObject`` truncates at the tar header's size — so it
    is belt-and-braces against a size the archive itself supplied.
    """
    cap = _max_member_bytes(member_name)
    budget_name = "document" if cap > _MAX_ATTRS_BYTES else "attributes"
    if declared_size > cap:
        _warn_size_refusal(
            archive_path,
            member_name,
            "metadata member",
            declared_size,
            cap,
            budget_name,
        )
        return {}
    handle = open_payload()
    if handle is None:
        return {}
    with handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        _warn_size_refusal(
            archive_path,
            member_name,
            "metadata member",
            len(raw),
            cap,
            budget_name,
        )
        return {}
    return _parse_attrs(raw, member_name, member_budget=cap, archive_path=archive_path)


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
        winner = best
        # Budgeting, bounded read and parse are shared with the tar path.
        return _read_selected_member(
            path,
            winner.filename,
            winner.file_size,
            lambda: zip_ref.open(winner, "r"),
        )


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
        winner = best
        # Budgeting, bounded read and parse are shared with the zip path;
        # `extractfile` is the tar's payload opener and may answer `None`.
        return _read_selected_member(
            path,
            winner.name,
            winner.size,
            lambda: tar_ref.extractfile(winner),
        )


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

    The member is read under a budget chosen by its document NAME
    (:func:`_max_member_bytes`), because the two formats' documents are not
    remotely the same size: a format-2 ``.zattrs`` IS the attributes, while a
    format-3 ``zarr.json`` at a consolidated root carries the whole consolidated
    index of the tree beside them (measured: ~8 KB per part for BARE-LEAF parts,
    ~48 KB with a 6-step ``stream`` ladder, ~100 KB for an ``adaptive``-shaped
    part — the index has one entry per NODE, not per part). A document read under
    that raised budget has the ATTRIBUTES it unwraps to re-capped at the small
    :data:`_MAX_ATTRS_BYTES`; a ``.zattrs``, already bounded by that same number
    as a member, is not measured twice.

    Reading the store's OWN per-node document — never a consolidated index —
    is the correct source, for an invariant rather than a version: the peek must
    see exactly what the loader's own group-open sees, and the loader never reads
    a store THROUGH consolidated metadata (under zarr 2 that took an explicit
    ``open_consolidated``; the zarr 3 path passes ``use_consolidated=False``
    deliberately). At format 2 that distinction is a choice of file — ``.zattrs``
    over ``.zmetadata``. At format 3 the two live in the SAME file, since the root
    ``zarr.json`` carries both the root's own ``attributes`` and the
    ``consolidated_metadata`` index of its descendants: the peek unwraps
    ``attributes`` and ignores the index (:func:`~luxar._zarr_compat.attrs_from_node_doc`),
    which is precisely what ``open_group`` does with the same bytes. Either way,
    answering from an index would introduce a divergence from the directory-store
    path rather than fix one.

    Args:
        path: Path to a ``.gsplats.zarr.zip`` or ``.gsplats.zarr.tar.gz``.

    Returns:
        The root attrs as a dict. ``{}`` when the path is missing or is not one of
        the two supported archive formats, when no root attrs member exists,
        when the store root is ambiguous (above), when the metadata document
        exceeds its name's budget or the attributes inside it exceed
        :data:`_MAX_ATTRS_BYTES`, or when the payload is not a JSON object.

    Warns:
        UserWarning: On any SIZE refusal, naming the archive, the member, the
            measured size and the budget (:func:`_warn_size_refusal`). ``{}``
            otherwise reads as "this dataset authored no appearance" all the way
            up, so a size refusal is the one ``{}`` that must not be silent. The
            other empty answers above are ordinary — no archive, no root member,
            an ambiguous root — and stay quiet.

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
