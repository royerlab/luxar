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

A zip's symlink member is refused for a second reason on top of the escape:
``extractall`` writes one out as a REGULAR FILE holding the target path, which
escapes nothing but does let a crafted archive show the extractor a depth-0
``.zgroup`` that the index-only resolvers here skip — the loader and the
appearance peek would then read different nodes out of one archive.
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
#: after parsing (:func:`_parse_attrs`) — but there ONLY when the member's own
#: bytes exceeded this number, i.e. only when the raised budget below was
#: actually USED, so that raising THAT budget cannot raise what this peek can
#: hand a caller. A member already inside this number is never measured twice:
#: the second check measures a re-serialization rather than the bytes on disk,
#: so there it could only ever refuse something this very budget admitted (see
#: :func:`_parse_attrs` for why the two are not the same currency). For a
#: ``.zattrs`` that is arithmetic rather than convention — its member budget IS
#: this number, so the re-cap condition is unreachable for it.
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
#: SHAPE of each part and not only on how many there are. Measured twice
#: independently on real ``write_gsplats_tree`` partitions at format 3, as
#: APPROXIMATE bytes of root ``zarr.json`` per part — the exact count moves with
#: the leaf's own shape, so these are ranges spanning both measurements, an order
#: of magnitude and not a closed interval. The crossover part counts take the
#: conservative (largest-per-part) end:
#:
#: * bare leaf part (one additive sub-LOD): ~8-10 KB (a real ``gsplat
#:   partition``: ~9 KB) — 4 MiB crossed at ~400 parts, 128 MiB at ~13,000.
#: * per-part 6-step ``stream`` ladder: ~48-54 KB — 4 MiB at ~78 parts, 128 MiB
#:   at ~2,500.
#: * per-part 3 levels x 4 sub-LODs (``adaptive``-shaped): ~100-110 KB — 4 MiB
#:   at ~37 parts, 128 MiB at ~1,200.
#:
#: So the old single budget was far MORE reachable than a part count suggests:
#: the recommended ``--recipe stream|levels|adaptive`` multiply nodes per part by
#: 6-12x, and a ~37-part ``adaptive`` partition already crossed 4 MiB. Past that
#: point this peek's ``{}`` reads as "this dataset authored no appearance" rather
#: than as an error, so every rewriting command silently overwrote a
#: zipped/tarred partition's appearance with the writer's defaults while the SAME
#: tree as a directory store answered correctly. 128 MiB buys ~1,200-2,500 parts
#: of headroom on the laddered shapes people actually publish (and ~13,000 on
#: bare leaves) — real headroom, not unlimited, which is why a refusal now warns
#: loudly (:func:`_warn_size_refusal`) instead of answering ``{}`` in silence.
#:
#: Not larger, because this is still an archive-bomb bound and the document is
#: parsed whole: at the ~3-4x measured below, 128 MiB of JSON materializes
#: several hundred MiB of Python objects, and that parse — not the byte count —
#: is the real ceiling.
#:
#: Raising it does not widen the PRODUCT's exposure, which was verified rather
#: than assumed: every caller of the peek also LOADS the same store, and
#: ``luxar._zarr_compat.open_group`` opts reads out of consolidated metadata but
#: still reads and parses the root document whole. Measured on ~8.4 MB
#: consolidated roots — one ``LocalStore.get`` returning the whole document, a
#: single ``json.loads`` over all of it, and a tracemalloc peak of ~3-4x the
#: document (24.5 MiB peak on an 8,385,305-byte root, 32.6 MiB on an
#: 8,384,150-byte one; the multiplier moves with the INDEX SHAPE, the structural
#: claims do not) — so zarr materializes what this peek does, and more, for the same
#: store. The exposure that IS new belongs to a caller invoking
#: :func:`read_archive_root_attrs` directly on an untrusted archive and never
#: opening it: that caller gets the raised ceiling with no load behind it.
_MAX_NODE_DOC_BYTES = 128 * 1024**2


def _is_node_document(member_name: str) -> bool:
    """Whether an archive member is a format-3 node document (``zarr.json``).

    THE single classification the NAME-keyed differences key on — the byte
    budget (:func:`_max_member_bytes`) and the word used for it in a refusal
    message. One predicate rather than two, and a predicate over the NAME rather
    than over the budget VALUE, because a budget value is not a stable signal: a
    test (or a future retune) that lowers :data:`_MAX_NODE_DOC_BYTES` below
    :data:`_MAX_ATTRS_BYTES` inverts any ``cap > _MAX_ATTRS_BYTES`` comparison,
    and a ``zarr.json`` then silently describes itself as attributes.

    The post-parse re-cap of the unwrapped attributes is deliberately NOT keyed
    on this. It exists solely to stop the RAISED budget from raising what the
    peek hands back, so its honest trigger is "was the raised budget actually
    used" — a fact about THIS MEMBER'S BYTES, not about its name and not about
    how the two constants happen to be ordered (:func:`_parse_attrs`).

    The name comes from the facade (:data:`~luxar._zarr_compat.V3_NODE_DOC`),
    which owns metadata-document names, rather than being spelled again here or
    read positionally out of :data:`NODE_ATTR_DOCS` — a positional read would let
    a reordering of that tuple silently swap the two formats' treatment.
    """
    return PurePosixPath(member_name).name == V3_NODE_DOC


def _max_member_bytes(member_name: str) -> int:
    """Byte budget for a root metadata member, chosen by the document's NAME.

    The name is the whole signal, and it is exact — it is also how the member was
    selected in the first place (:func:`_root_attrs_rank`). A format-2
    ``.zattrs`` is the attributes mapping itself and keeps the small
    :data:`_MAX_ATTRS_BYTES` budget; nothing about format 2 changed. A format-3
    ``zarr.json`` is a whole node document carrying the consolidated index and
    gets :data:`_MAX_NODE_DOC_BYTES`. :func:`_is_node_document` is the one place
    that distinction is made.

    This rides on the SELECTION above, and inherits its one known limitation: a
    root carrying BOTH documents ranks them equally, so the tie-break is archive
    order and the peek budgets whichever one it happened to select, while
    ``open_group`` resolves format 3 either way. No Luxar writer produces that
    state and #1600 deliberately defers changing it (preferring the v3 document
    at equal rank would cost the tar walk its rank-0 early stop), so this is a
    recorded limitation and not an invariant being relied on.

    Raising the document budget does not raise what the peek can hand back:
    :func:`_parse_attrs` re-checks the unwrapped attributes against the small
    budget whenever a member's bytes actually exceeded it — which only a
    ``zarr.json`` read under the raised budget ever can.
    """
    return _MAX_NODE_DOC_BYTES if _is_node_document(member_name) else _MAX_ATTRS_BYTES


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

    ``stacklevel=1`` deliberately: the warning is attributed to THIS module, which
    is where it is raised and the only stable answer. There is no single correct
    constant to thread it out with — the two callers
    (:func:`_read_selected_member`, :func:`_parse_attrs`) sit at different depths,
    and both are inside this file anyway, so a bumped level would still point at
    an internal frame while merely being wrong about which one. Users suppressing
    or routing this key on ``module=r"luxar\\.gsplats\\.io\\._archive"``, which
    stacklevel does not affect.

    Under the default warning filter this therefore dedupes per ARCHIVE: the
    ``default`` action keys on message+category+module+lineno and the archive path
    is in the message, so peeking the same archive twice in one process warns
    once, while three different oversized archives warn three times.
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
        stacklevel=1,
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
            # Same rule the tar path applies, for the reason in the module
            # docstring: `extractall` materializes a symlink member as a regular
            # file, so this is not an escape — it is a way to hand the extractor
            # a metadata document the index-only resolvers here deliberately
            # skip, and have the two disagree about the store root.
            if _zip_member_is_symlink(info):
                raise ValueError(
                    f"Zip member '{name}' is a symlink; refusing "
                    "(gsplats archives must contain only regular files)"
                )
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


#: Store directory name for a flat archive whose own name strips to nothing.
_FLAT_STORE_FALLBACK_NAME = "store.gsplats.zarr"


def _flat_store_dir_name(archive_name: str) -> str:
    """Store-shaped directory name to re-parent a FLAT archive's tree under.

    Purely cosmetic — nothing validates it — but it is what ``gsplat view`` logs
    as the served store, so it is derived from the archive
    (``flat.gsplats.zarr.zip`` → ``flat.gsplats.zarr``) rather than being a
    constant. ``Path.name`` is a single component, so no separator can appear,
    and the unconditional append below already makes every answer a FRESH single
    component: with the dot guard removed, ``""`` would come out as
    ``.gsplats.zarr`` and ``"."`` as ``..gsplats.zarr``, neither of which is
    ``outer`` or its parent. What the guard buys is cosmetic too — those answers
    are HIDDEN directories, and a leading dot is a poor thing to log as the store
    being served.
    """
    stripped = archive_name
    for suffix in _COMPRESSED_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    if stripped in ("", ".", ".."):
        return _FLAT_STORE_FALLBACK_NAME
    if not stripped.endswith(".gsplats.zarr"):
        stripped += ".gsplats.zarr"
    return stripped


def _reparent_flat_store(temp_dir: Path, compressed_path: Path) -> Path:
    """Move an extracted FLAT tree one level down and return the store path.

    Every caller treats ``zarr_path.parent`` as the removable temp directory
    (:func:`resolve_store_path` hands back exactly that; ``doctor`` and
    ``migrate`` take ``target.parent`` as their scratch). For a flat archive the
    store IS ``temp_dir``, whose parent is the system temp directory — so it is
    moved into a second temp directory instead of being returned as-is, and the
    contract holds unchanged.
    """
    outer = Path(tempfile.mkdtemp(prefix="luxar_gsplat_"))
    target = outer / _flat_store_dir_name(compressed_path.name)
    try:
        shutil.move(str(temp_dir), str(target))
    except BaseException:
        shutil.rmtree(outer, ignore_errors=True)
        raise
    return target


def extract_compressed_zarr(compressed_path: Path) -> Path:
    """Extract a compressed ``.gsplats.zarr`` archive to a fresh temp directory.

    The store root is resolved from the extracted tree in three tiers, in order:

    1. a top-level directory whose name ends in ``.gsplats.zarr`` — what
       ``_compress_zarr`` always writes. It keeps winning even in an odd archive
       that also looks flat, because that is what :func:`_zip_is_flat_store`
       decides and the two resolutions must not disagree.
    2. a zarr GROUP document (:data:`~luxar._zarr_compat.NODE_GROUP_DOCS` —
       ``zarr.json`` at format 3, ``.zgroup`` at format 2) sitting at depth 0: the
       archive root IS the store. This is the FLAT shape, which
       ``zip -r x.gsplats.zarr.zip .`` from inside a store and a zarr-native
       ``ZipStore`` write both produce; without this tier the fallback below
       picked an arbitrary array sub-directory (#1628). A flat ``.tar.gz`` gets
       it too, and for tar this is the only possible route (zarr has no tar
       store).
    3. the first top-level directory, whatever it is called — ``tar czf
       x.gsplats.zarr.tar.gz mystore``. Named "sole" in the peek's mirror of
       these tiers because that is the only case the peek will resolve; here a
       depth-0 FILE beside it (a ``README.md``, a ``.DS_Store`` a macOS zip
       picked up) must not turn the archive into a hard failure, and with
       several directories the pick is arbitrary — ``iterdir()`` order — which
       is pre-existing and is why the peek refuses that case instead.

    A flat tree is MOVED into a second temp directory under a store-shaped name
    (see :func:`_reparent_flat_store`) so the returned path's ``parent`` is
    always a temp directory the caller may remove, exactly as for the other two
    tiers.

    Tier 2 outranking tier 3 REVERSED one shape: an archive carrying a depth-0
    group document AND the real store one level down under a name that is not
    ``*.gsplats.zarr`` (``zip -r x.gsplats.zarr.zip .`` run from inside a parent
    zarr group that merely CONTAINS the store) used to fall through to the
    directory and load; it now resolves to the wrapper and fails with
    ``Invalid format_type: None``. The order is still right: nothing in an
    archive distinguishes "a store whose root happens to have one child group"
    from "a wrapper around a store", and gating tier 2 on "the sole child is not
    itself a group" would break the flat PARTITION that is #1628's own repro.
    The trade is a loud failure on a shape nothing in Luxar writes, in exchange
    for a shape that is common in the wild, and the rescue is to re-archive the
    inner directory on its own.

    Args:
        compressed_path: Path to a ``.gsplats.zarr.zip`` or ``.gsplats.zarr.tar.gz``.

    Returns:
        Path to the extracted store directory (inside a new temp dir the caller
        is responsible for cleaning up on success).

    Raises:
        ValueError: On an unsafe member (link/device, path escape, unsafe
            separator), an oversized archive, an unsupported format, or when none
            of the three tiers finds a store. The temp directory is removed
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

        # Locate the extracted store: prefer a .gsplats.zarr-named top-level
        # directory, then a flat store (group document at depth 0), else fall
        # back to the first top-level directory.
        children = list(temp_dir.iterdir())
        for child in children:
            if child.is_dir() and child.name.endswith(".gsplats.zarr"):
                return child
        for child in children:
            if (
                child.name in NODE_GROUP_DOCS
                and child.is_file()
                and not child.is_symlink()
            ):
                return _reparent_flat_store(temp_dir, compressed_path)
        for child in children:
            # The first DIRECTORY, not `children[0]`: `iterdir()` is unordered,
            # so requiring the very first entry to be one made a stray depth-0
            # file (`README.md`, a `.DS_Store`) a hard failure on some
            # filesystems and a no-op on others, while the peek resolved the
            # store either way.
            if child.is_dir():
                return child
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
      resolver must not disagree with it). A bare directory ENTRY with that name
      and nothing under it counts too: ``zip -r`` emits one for an empty
      subdirectory, ``extractall`` materializes it, and the extractor's first
      tier then wins on a directory this rule could not otherwise see.

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
            if len(parts) == 1 and info.is_dir() and parts[0].endswith(".gsplats.zarr"):
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

    A ``.tar.gz`` is always extracted: zarr has no tar store, so extraction is
    the only route into that container whatever shape the store has inside.

    Args:
        path: An existing ``.gsplats.zarr`` directory, or a ``.zip`` /
            ``.tar.gz`` archive holding one.
        flat_zip_in_place: Opt in to reading a FLAT zip — one whose store sits at
            the archive ROOT, as ``zip -r x.gsplats.zarr.zip .`` from inside a
            store or a zarr-native ``ZipStore`` write produces — WITHOUT
            extracting it, by returning the archive itself for ``open_group`` to
            open as a ``ZipStore`` (:func:`_zip_is_flat_store` decides, from the
            zip index only).

            Both routes work: since #1628 :func:`extract_compressed_zarr`
            resolves a flat archive's store root too, so the default extraction
            path supports the shape for every reader. The flag is therefore about
            TEMP SPACE, not capability — reading a flat zip in place costs none,
            where extraction costs the full uncompressed size.

            Only :func:`~luxar.gsplats.io.inspect_gsplats.inspect_gsplats_zarr`
            opts in, because it reads metadata documents and nothing else, so
            paying for a full extraction would be a pure regression for a call
            whose entire purpose is a cheap metadata peek. ``load_gsplat_node``
            does not: it reads array data through the resolved store, which it
            wants as a directory.

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
    root attributes in ``.zattrs``, format 3 inside ``zarr.json``. They share a
    rank, and a store carrying BOTH — a half-finished in-place migration or a
    half-overwritten copy, the case :data:`NODE_ATTR_DOCS` is ordered
    format-3-first against — is therefore settled by ARCHIVE ORDER, so this peek
    can hand back the stale format-2 view where ``open_group`` answers format 3.
    That is a known limitation of an index-only peek rather than a contract:
    preferring the v3 document at equal rank would cost the tar walk its
    documented rank-0 early stop, and no Luxar writer produces the state.
    Recognising only
    ``.zattrs`` made this peek answer ``{}`` for every format-3 archive — which
    reads as "this dataset authored no appearance" rather than as an error.

    Lower sorts better. The three ranks mirror, one for one, the three tiers
    :func:`extract_compressed_zarr` uses to locate the store inside an archive:

    * ``0`` — ``<name>.gsplats.zarr/<doc>``: the top-level directory whose name
      ends in ``.gsplats.zarr``. The extractor prefers it, and it is what
      ``_compress_zarr`` always writes. Nothing else can outrank it, so it needs
      no further qualification.
    * ``1`` — a depth-0 ``<doc>``: the FLAT store's own root document, the shape
      ``zip -r x.gsplats.zarr.zip .`` from inside a store produces. The rank
      alone does NOT establish that a store root is here — a loose ``.zattrs``
      with no group document beside it is a stray, and reading it would hand back
      attrs belonging to something that is not the dataset. Only
      :func:`_root_is_usable`, which sees the whole member list, can tell the two
      apart.
    * ``2`` — ``<anything-else>/<doc>``: a candidate for the extractor's last
      tier, the *sole* top-level directory whatever it happens to be called
      (what you get from ``tar czf x.gsplats.zarr.tar.gz mystore``). Also only
      usable in whole-archive context: a rank-2 member in an archive with several
      top-level directories is one candidate among several and is refused.

    Every other member is not a candidate at all: a deeper one is a child group's
    attrs (``<top>/lod_0/.zattrs``), a different object, and authoring an
    appearance from it is exactly the failure this ranking exists to prevent.
    """
    parts = PurePosixPath(name).parts
    if not parts or parts[-1] not in NODE_ATTR_DOCS:
        return None
    if len(parts) == 1:
        return _FLAT_ROOT_RANK
    if len(parts) != 2:
        return None
    return _BEST_ROOT_RANK if parts[0].endswith(".gsplats.zarr") else _SOLE_DIR_RANK


def _is_root_group_doc(name: str) -> bool:
    """Whether a member is a zarr GROUP document sitting at depth 0.

    The flat tier's gate. Only a group document counts (:data:`NODE_GROUP_DOCS`):
    ``.zattrs`` also sits beside an ARRAY and may be a stray, so it says nothing
    about a store root being here — the same rule :func:`_zip_is_flat_store`
    applies, and the same one this peek needs so a stray keeps losing.
    """
    parts = PurePosixPath(name).parts
    return len(parts) == 1 and parts[0] in NODE_GROUP_DOCS


def _marks_gsplats_dir(name: str, *, is_dir: bool) -> bool:
    """Whether a member establishes a top-level ``*.gsplats.zarr/`` DIRECTORY.

    Two spellings, mirroring the disqualifying rules of
    :func:`_zip_is_flat_store`: a member lying UNDER such a directory (its first
    of several components), and a bare depth-0 directory ENTRY with that name —
    what ``zip -r`` emits for an empty subdirectory and what a tar records
    explicitly. ``extractall`` materializes either, so the extractor's first tier
    picks the directory either way and the peek has to see both or it calls flat
    an archive the extractor does not.

    Deliberately NOT folded into :func:`_top_level_dir`, whose blindness to a
    bare entry is a separate deliberate choice for a separate tier (an empty
    top-level directory holds no store, so counting it would manufacture false
    ambiguity for the sole-directory rule).
    """
    parts = PurePosixPath(name).parts
    if not parts or not parts[0].endswith(".gsplats.zarr"):
        return False
    return len(parts) > 1 or is_dir


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


def _root_is_usable(
    rank: int, top_dirs: set[str], has_root_group_doc: bool, has_gsplats_dir: bool
) -> bool:
    """Whether a candidate of ``rank`` may actually be used as the store root.

    Rank 0 always may. It is the extractor's own first preference, so the two
    agree on the KIND of node — and for an archive holding ONE store, which is
    every archive ``_compress_zarr`` writes, on the same node. They need not agree
    on WHICH when an archive holds several ``*.gsplats.zarr`` directories: that is
    not one dataset, the extractor picks among them arbitrarily (``iterdir()``
    order), and the peek picks the first in archive-index order.

    Rank 1 (flat) needs a zarr GROUP document at depth 0 and no top-level
    ``*.gsplats.zarr`` directory — the conditions :func:`_zip_is_flat_store`
    uses, all three of them, so the peek and the extractor classify the same
    archives as flat. The directory is taken from ``has_gsplats_dir`` rather than
    from ``top_dirs`` because the two answer different questions: a BARE depth-0
    directory entry still gives the extractor a directory to prefer (see
    :func:`_marks_gsplats_dir`) while contributing nothing to the sole-directory
    tier below (see :func:`_top_level_dir`). Reading it off ``top_dirs`` left the
    peek calling such an archive flat while the extractor resolved the directory.
    The group document is what keeps a STRAY depth-0 ``.zattrs`` refused: on its
    own it marks no store root.

    Rank 2 stands in for the extractor's last tier, the SOLE top-level directory,
    and there refusing costs nothing — so with two of them the peek refuses rather
    than guessing. It is also refused outright once the archive is flat: a flat
    store that authored no root attrs at all has no depth-0 document, and falling
    through would then hand back an ARRAY sub-directory's attrs (``centers/
    .zattrs``) as the dataset's appearance. Carrying nothing is the safe outcome
    (the appearance simply is not carried, as before the carry existed); carrying
    a sibling node's attrs would author an appearance from something that is not
    the dataset.
    """
    if rank == _BEST_ROOT_RANK:
        return True
    is_flat = has_root_group_doc and not has_gsplats_dir
    if rank == _FLAT_ROOT_RANK:
        return is_flat
    return len(top_dirs) == 1 and not is_flat


def _pick_root_member(
    candidates: Dict[int, Any],
    top_dirs: set[str],
    has_root_group_doc: bool,
    has_gsplats_dir: bool,
) -> Optional[Any]:
    """Best-ranked candidate that :func:`_root_is_usable` accepts, else ``None``.

    Best-ranked *usable*, not best-ranked: an unusable rank-1 candidate is a
    stray depth-0 ``.zattrs``, and the archive it sits in may still carry a real
    store one directory down (``mystore/.zattrs``) which the load reads and the
    peek must therefore read too.
    """
    for rank in sorted(candidates):
        if _root_is_usable(rank, top_dirs, has_root_group_doc, has_gsplats_dir):
            return candidates[rank]
    return None


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    """Whether a zip member carries a unix symlink mode (never follow one)."""
    return stat.S_ISLNK(info.external_attr >> 16)


def _parse_attrs(
    raw: bytes,
    member_name: str,
    *,
    recap_attrs: bool,
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
    ONLY when this member's own bytes exceeded that number (``recap_attrs``,
    computed in :func:`_read_selected_member` as ``len(raw) > _MAX_ATTRS_BYTES``).
    The re-cap exists for exactly one reason — a format-3 node document is
    allowed :data:`_MAX_NODE_DOC_BYTES` because of the consolidated index it
    carries, and that allowance must not become licence to hand a caller a
    100 MiB attrs mapping — so the honest trigger is "was the raised budget
    actually used", which is a fact about the bytes, not about the document's
    name and not about how the two constants happen to be ordered.

    Keying it on the bytes is what makes the three cases fall out, one of them
    provably:

    * a ``.zattrs`` can never be re-capped, and that is arithmetic rather than a
      convention: its member budget IS :data:`_MAX_ATTRS_BYTES`
      (:func:`_max_member_bytes`), so the declared-size and length gates refuse
      first and ``len(raw) > _MAX_ATTRS_BYTES`` is unreachable on that branch;
    * a ``zarr.json`` at or under :data:`_MAX_ATTRS_BYTES` is not re-capped
      either, so nothing the single pre-#1600 budget admitted is newly refused —
      which is a real case and not a courtesy, because a re-serialization is not
      bounded by the source bytes (below);
    * a ``zarr.json`` OVER it — the only member that actually used the raised
      budget — is re-capped, which is the whole point.

    That the re-cap can only ever REFUSE, never admit, is a property of the
    CURRENCY it measures in, not of any one input, which is why the two
    mitigations here are independent rather than one covering for the other:

    * the measure is a COMPACT, faithful re-serialization — ``ensure_ascii`` off,
      no separator padding — so it does not inflate non-ASCII attrs the way a
      default ``json.dumps`` does (measured: attrs that are 1,800,029 bytes on
      disk measure 1,800,026 compact but 5,400,029 ASCII-escaped, i.e. a 1.7 MiB
      ``.zattrs`` would cross a 4 MiB cap purely by being escaped). That is the
      whole reason the re-cap that DOES run — on a big format-3 document, whose
      unwrapped attributes may legitimately be non-ASCII too — is not a fresh
      silent-``{}`` trap.
    * a re-serialization still cannot be assumed ``<=`` the source bytes even
      after the escaping is removed, because ``json.loads`` is not
      round-trip-preserving: exponent notation is one verified case
      (``{"a":1e10}`` is 10 bytes and re-serializes to 19). That is precisely
      why the re-cap must not run on a member the raised budget did not admit —
      a 2.61 MiB ``zarr.json`` of such literals measures 4.24 MiB and would be
      refused for nothing, reintroducing the silent ``{}`` this whole change
      exists to remove.

    Measuring by re-serializing at all is honest about the cost: the document is
    already parsed and resident by this point, so this is one more pass over data
    we are holding anyway, and no incremental accounting is possible once the
    parse has happened.
    """
    attrs = attrs_from_node_doc(
        json.loads(raw.decode("utf-8")),
        doc_name=PurePosixPath(member_name).name,
    )
    if not recap_attrs:
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

    The parse re-caps the unwrapped attributes only when the bytes actually READ
    exceeded :data:`_MAX_ATTRS_BYTES` — i.e. only when the raised document budget
    was actually used, which is the one thing that re-cap exists to bound. The
    budget and the refusal LABEL are keyed on the document's name; the re-cap is
    keyed on this member's bytes, and :func:`_parse_attrs` says why the two
    signals are not interchangeable.

    ``open_payload`` is a zero-argument callable rather than an already-open
    handle so nothing is decompressed for a member that the declared-size gate is
    about to refuse; it may return ``None``, which is what ``tarfile``'s
    ``extractfile`` gives for a member with no payload.

    The bounding read (``cap + 1``) cannot actually over-deliver in either
    format — ``ZipExtFile`` truncates the decompressed stream at the declared
    ``file_size`` (a tampered central-directory size raises ``BadZipFile`` on the
    CRC instead) and ``ExFileObject`` truncates at the tar header's size — so it
    is belt-and-braces against a size the archive itself supplied.

    The two member-level refusals word themselves DIFFERENTLY ("declared size"
    versus "bytes read") even though they share a budget. They are not the same
    gate: the declared-size one is the security-relevant half — it refuses
    BEFORE ``open_payload`` is called, so an archive-bomb member is never
    decompressed at all — while the second only catches a payload that outran its
    own header. Identical wording made them indistinguishable to a
    ``pytest.warns(match=…)``, so a test aimed at the first passed just as
    happily when it was deleted and the second fired instead.
    """
    cap = _max_member_bytes(member_name)
    budget_name = "document" if _is_node_document(member_name) else "attributes"
    if declared_size > cap:
        _warn_size_refusal(
            archive_path,
            member_name,
            "metadata member (declared size)",
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
            "metadata member (bytes read)",
            len(raw),
            cap,
            budget_name,
        )
        return {}
    return _parse_attrs(
        raw,
        member_name,
        recap_attrs=len(raw) > _MAX_ATTRS_BYTES,
        archive_path=archive_path,
    )


def _read_zip_root_attrs(path: Path) -> Dict[str, Any]:
    """Root attrs of a zip archive, reading that one member's bytes only."""
    with zipfile.ZipFile(path, "r") as zip_ref:
        candidates: Dict[int, zipfile.ZipInfo] = {}
        top_dirs: set[str] = set()
        has_root_group_doc = False
        has_gsplats_dir = False
        # infolist() parses the central directory only — no member payload is
        # decompressed by this scan.
        for info in zip_ref.infolist():
            is_dir = info.is_dir()
            top = _top_level_dir(info.filename)
            if top is not None:
                top_dirs.add(top)
            has_gsplats_dir |= _marks_gsplats_dir(info.filename, is_dir=is_dir)
            if is_dir or _zip_member_is_symlink(info):
                continue
            if _is_root_group_doc(info.filename):
                has_root_group_doc = True
            rank = _root_attrs_rank(info.filename)
            if rank is None:
                continue
            candidates.setdefault(rank, info)
        best = _pick_root_member(
            candidates, top_dirs, has_root_group_doc, has_gsplats_dir
        )
        if best is None:
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
#: The other two layouts get no such stop in either format, twice over: a worse
#: candidate can still be superseded by a rank-0 one later in the stream, and
#: whether it may be used at all depends on whole-archive context — the set of
#: top-level directories, and whether a depth-0 group document appears anywhere
#: (:func:`_root_is_usable`). So those walks have to reach the end of the header
#: stream — and a gzipped tar has no index, so reaching the end means inflating
#: the whole file.
_BEST_ROOT_RANK = 0

#: A depth-0 attrs document: the FLAT store's own root (see
#: :func:`_root_attrs_rank`). Usable only alongside a depth-0 group document.
_FLAT_ROOT_RANK = 1

#: The root attrs document of a top-level directory that is NOT
#: ``*.gsplats.zarr``-named — usable only when it is the archive's sole one.
_SOLE_DIR_RANK = 2


def _read_targz_root_attrs(path: Path) -> Dict[str, Any]:
    """Root attrs of a tar.gz archive, reading that one member's bytes only.

    Member HEADERS are walked lazily (never ``getmembers()``, which materializes
    the whole archive), and only the winning member's payload is read. The walk
    stops as soon as a member reaches :data:`_BEST_ROOT_RANK`; failing that it
    runs to the end of the header stream (see :data:`_BEST_ROOT_RANK` for what
    that costs, and why a format-3 archive nearly always pays it), because a
    better-ranked member may appear after a worse one, neither of the other two
    tiers can be settled before the archive's whole membership is known (its
    top-level directories, and whether a depth-0 group document appears at all),
    and reading the wrong node's attrs as the root's would silently author an
    appearance nobody asked for.
    """
    with tarfile.open(path, "r:gz") as tar_ref:
        candidates: Dict[int, tarfile.TarInfo] = {}
        top_dirs: set[str] = set()
        has_root_group_doc = False
        has_gsplats_dir = False
        for member in tar_ref:
            top = _top_level_dir(member.name)
            if top is not None:
                top_dirs.add(top)
            # A tar carries explicit directory members where a zip has a
            # trailing-slash entry; `is_dir` is the only part that differs.
            has_gsplats_dir |= _marks_gsplats_dir(member.name, is_dir=member.isdir())
            # Only regular files: a symlink named `.zattrs` is never followed
            # (`extractfile` resolves an in-archive link target, so dropping this
            # check would hand back whatever node the link points at). A symlink
            # named `.zgroup` must not pass for a group document either, which is
            # the same check.
            if not member.isfile():
                continue
            if _is_root_group_doc(member.name):
                has_root_group_doc = True
            rank = _root_attrs_rank(member.name)
            if rank is None:
                continue
            candidates.setdefault(rank, member)
            if rank == _BEST_ROOT_RANK:
                break
        best = _pick_root_member(
            candidates, top_dirs, has_root_group_doc, has_gsplats_dir
        )
        if best is None:
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
    :func:`_root_is_usable`, which mirror the three tiers
    :func:`extract_compressed_zarr` resolves a store by: a top-level
    ``*.gsplats.zarr`` directory, as ``_compress_zarr`` writes; else the archive
    ROOT itself when a zarr group document sits at depth 0 (the FLAT shape); else
    — only when it is the archive's SOLE top-level directory — that directory
    whatever it is named. For an archive holding ONE store, which is every
    archive ``_compress_zarr`` writes, that is the same node the extractor loads.
    An archive holding SEVERAL ``*.gsplats.zarr`` directories is not a single
    dataset; the extractor picks one of them arbitrarily (``iterdir()`` order) and
    the peek may pick another. Either way a child group's attrs
    (``<top>/fitting/.zattrs``) is never read as the root's.

    A depth-0 attrs document is the flat store's own root — but only when a
    depth-0 GROUP document is there with it. Without one it is a stray ``.zattrs``
    (a name that also sits beside an ARRAY), and it loses to a real store one
    directory down, exactly as it did before the flat tier existed.

    An archive with several top-level directories and no ``*.gsplats.zarr``-named
    one is ambiguous and yields ``{}``: the extractor resolves that case by
    ``iterdir()`` order, which an archive index cannot predict, so the peek would
    have to guess — and carrying nothing is merely the status quo, while carrying
    a sibling directory's attrs would author an appearance from the wrong node.
    A flat archive that authored no root attrs at all yields ``{}`` on the same
    principle, by an explicit rule rather than by ambiguity: its top-level
    directories are its own ARRAYS, so the sole-directory tier is refused
    outright once the archive is flat.

    Where a member name occurs twice, the FIRST occurrence wins here while
    ``extractall`` keeps the last; such an archive is malformed (both occurrences
    are the same PATH — the store root's own ``.zattrs`` — so no foreign node's
    attrs can be reached either way), and the early stop is worth keeping.

    The member is read under a budget chosen by its document NAME
    (:func:`_max_member_bytes`), because the two formats' documents are not
    remotely the same size: a format-2 ``.zattrs`` IS the attributes, while a
    format-3 ``zarr.json`` at a consolidated root carries the whole consolidated
    index of the tree beside them (measured: ~8-10 KB per part for BARE-LEAF
    parts, ~48-54 KB with a 6-step ``stream`` ladder, ~100-110 KB for an
    ``adaptive``-shaped part — the index has one entry per NODE, not per part).
    That budget is keyed on the document NAME (:func:`_is_node_document`), never
    on the budget values. A member whose bytes actually exceeded
    :data:`_MAX_ATTRS_BYTES` — only ever a ``zarr.json`` read under the raised
    budget — additionally has the ATTRIBUTES it unwraps to re-capped at that
    small number, so raising the document budget cannot raise what this peek
    hands back. Anything already inside it, every ``.zattrs`` included, is not
    measured twice.

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
