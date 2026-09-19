"""The one module in Luxar that knows which zarr *format* version we write.

Library version and on-disk format are independent axes, and this module is
where they are pinned together. Luxar runs on **zarr-python 3** and writes
**zarr format 3** by default; :data:`ZARR_FORMAT` selects the on-disk format and
:data:`ZARR_FORMAT_ENV_VAR` overrides it, so format 2 remains producible for a
tool that cannot read 3.

READING is not affected by any of this — zarr-python 3 opens both formats — and
that asymmetry is the whole design. Existing `.luxar.zarr` / `.gsplats.zarr`
stores stay format 2 and are never rewritten, so a mixed-format tree is the
expected steady state rather than a migration window. Every helper here that
answers a question ABOUT a store therefore answers it for both formats, while
the helpers that CREATE one follow :data:`ZARR_FORMAT`.

Why a facade at all
-------------------
The TypeScript viewer has exactly one module that imports zarrita
(``src/data/zarr.ts``); everything else speaks in Luxar concepts. That design
made the viewer nearly version-agnostic for free. The Python side had no
equivalent — 53 production modules imported ``zarr`` directly (171 counting
tests) — so a format change meant sweeping all of them. Routing through here is
what made the move to format 3 an edit to *this* file rather than to that whole
surface: the flip itself was one constant, and the work that remained was the
translation layer below (numcodecs objects are format-2 currency; format 3 wants
``zarr.codecs`` ones) plus the bi-format readers.

Five zarr-3 behaviours are actively dangerous here, and all five are neutralised
below rather than left to call sites. Each one fails SILENTLY — none raises:

1. **An omitted compressor is not "no compressor".** zarr's ``compressors="auto"``
   silently applies a real compressor, whereas Luxar has arrays that must be
   stored RAW (the packed label byte-blobs) and others that must carry the
   measured zstd-9 policy. ``None`` is therefore mapped to an explicit
   ``compressors=None`` rather than being confused with "unspecified".

   ``"auto"`` is NOT the same compressor in both formats — it is whatever zarr
   would have chosen for that format, which is Blosc/lz4/clevel-5 at format 2
   and ``zstd`` level 0 at format 3. It exists for test fixtures that never
   named a compressor and do not care; nothing whose bytes matter may rely on
   it. Production code is held to naming one by
   ``test_zarr_compat.py::test_production_create_array_calls_pass_a_compressor``,
   which walks the AST rather than trusting review.
2. **``data=`` and ``shape=`` are mutually exclusive.** zarr 2's
   ``create_dataset`` accepted both, and several Luxar writers passed both.
   Passing both to zarr 3 raises. :func:`create_array` accepts both, casts
   ``data`` to ``dtype`` when given, and forwards only what zarr 3 allows.
3. **``chunks=True``/``False`` are rejected.** zarr 2 spelled "choose for me" and
   "one chunk for everything" that way, ``ChunkSpec`` still carries both, and
   ``create_resizable_dataset`` defaults to ``True`` — so every resizable array
   would have failed. See :func:`_translate_chunks`.
4. **Creation defaults to format 3.** A bare ``zarr.group()`` emits ``zarr.json``
   and ``c/`` chunk keys. Every creating helper here pins :data:`ZARR_FORMAT`
   explicitly rather than leaning on ``zarr.config``, so correctness does not
   depend on import order or on nobody else touching the ambient default.
5. **Reads trust consolidated metadata.** zarr 2 consulted ``.zmetadata`` only
   through the separate ``open_consolidated``; zarr 3 reversed that default, which
   makes a deleted array still read as PRESENT and silently disables Luxar's
   detection of a partially written store. :func:`open_group` opts reads out.

A sixth hazard is specific to ``mode="a"``: because it is create-or-open, pinning
the format unconditionally would assert it against a store that already exists —
and on a v3 store that SHADOWS it rather than failing. See :func:`open_group`.

Everything here is deliberately thin. It is a compatibility seam, not an
abstraction layer: callers still hold real :class:`zarr.Group` and
:class:`zarr.Array` objects and still use ``.attrs``, ``[...]`` indexing and
``.resize()`` directly, because none of those changed between zarr 2 and 3.

It lives at the TOP level of the package, not under ``luxar.io``, because
``luxar.encoding._encoders`` needs it too and ``luxar/io/__init__.py`` imports
the compiler, which imports those encoders — an import from ``luxar.io`` would
close that loop. This module imports nothing from Luxar at runtime, so it is
safe for any layer to depend on.
"""

from __future__ import annotations

import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing import Iterator as _Iterator

import numpy as np
import zarr
import zarr.errors

if TYPE_CHECKING:  # pragma: no cover - typing only
    # The canonical home of the alias — `typing_utils.protocols` only re-imports
    # it under TYPE_CHECKING, so it is not a re-export mypy will follow.
    from luxar.encoding.compression import CompressorLike

__all__ = [
    "DEFAULT_ZARR_FORMAT",
    "NODE_ATTR_DOCS",
    "NODE_GROUP_DOCS",
    "SUPPORTED_ZARR_FORMATS",
    "V3_NODE_DOC",
    "ZARR_FORMAT",
    "ZARR_FORMAT_ENV_VAR",
    "attrs_from_node_doc",
    "close",
    "consolidate",
    "create_array",
    "create_root_group",
    "is_consolidated",
    "is_missing_error",
    "is_zarr_path",
    "list_raw_keys",
    "memory_group",
    "open_group",
    "open_store",
    "read_array_meta",
    "read_consolidated_attrs",
    "read_node_attrs",
    "read_raw_bytes",
    "set_zarr_format",
    "write_raw_bytes",
    "zarr_format",
]

#: The zarr format Luxar writes by DEFAULT. Reading is version-agnostic — zarr
#: -python 3 opens format 2 and format 3 stores alike — so this governs new
#: output only, and a repository holding both formats is the expected steady
#: state rather than a transitional one.
DEFAULT_ZARR_FORMAT = 3

#: The formats this module will write. Reading is not restricted to these.
SUPPORTED_ZARR_FORMATS = (2, 3)

#: Environment variable that overrides :data:`DEFAULT_ZARR_FORMAT`.
#:
#: An ENV VAR rather than only a CLI flag because the processes that do the
#: writing are frequently not the process the user invoked: ``batch-fit run``
#: spawns per-GPU workers, ``batch-fit submit`` writes an sbatch script whose
#: array tasks run hours later on other nodes, and ``fit -j N`` forks tile
#: workers. An exported variable reaches all of them; a flag parsed by one
#: command would have to be threaded through every spawn site to match.
ZARR_FORMAT_ENV_VAR = "LUXAR_ZARR_FORMAT"


def _format_from_env() -> int:
    """Resolve the startup format from the environment, or the default."""
    raw = os.environ.get(ZARR_FORMAT_ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_ZARR_FORMAT
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{ZARR_FORMAT_ENV_VAR}={raw!r} is not an integer; expected one of "
            f"{list(SUPPORTED_ZARR_FORMATS)}"
        ) from None
    if value not in SUPPORTED_ZARR_FORMATS:
        raise ValueError(
            f"{ZARR_FORMAT_ENV_VAR}={raw!r} is not a supported write format; "
            f"expected one of {list(SUPPORTED_ZARR_FORMATS)}"
        )
    return value


#: The zarr format Luxar WRITES right now.
#:
#: Read through :func:`zarr_format` rather than imported by value — a
#: ``from luxar._zarr_compat import ZARR_FORMAT`` snapshots whatever was current
#: at import time and would not see :func:`set_zarr_format`. The helpers in this
#: module all read the module attribute at CALL time, so an override applies to
#: every write that follows it.
ZARR_FORMAT = _format_from_env()

_UNKNOWN_CODEC_ERRORS = (
    KeyError,
    getattr(zarr.errors, "UnknownCodecError", KeyError),
)


def zarr_format() -> int:
    """The format new stores are written at. See :func:`set_zarr_format`."""
    return ZARR_FORMAT


def set_zarr_format(value: int) -> None:
    """Override the write format for the rest of this process.

    The escape hatch for producing v2 for a tool that cannot read v3. Prefer the
    :data:`ZARR_FORMAT_ENV_VAR` environment variable when the write may happen in
    a child process — a setter call does not survive a fork/exec boundary, and
    Luxar's heavier writers routinely cross one.
    """
    global ZARR_FORMAT
    if value not in SUPPORTED_ZARR_FORMATS:
        raise ValueError(
            f"set_zarr_format({value!r}): expected one of "
            f"{list(SUPPORTED_ZARR_FORMATS)}"
        )
    ZARR_FORMAT = value


# Suffixes that mean "this path is a zipped store, not a directory". zarr 2
# sniffed these inside `zarr.open`; zarr 3 requires the store to be chosen
# explicitly, so the sniffing lives here now.
_ZIP_SUFFIXES = (".zip",)


def is_zarr_path(path: str | Path) -> bool:
    """Does ``path`` name a zarr store — a ``.zarr`` directory or a zipped one?

    Input routing rather than store construction, but it lives here because it
    needs exactly the same suffix knowledge as :func:`open_store`, and keeping
    ``.zip`` awareness in one module is the point of that function existing.
    Recognises the compound spellings Luxar actually produces, so
    ``scene.luxar.zarr`` and ``fit.gsplats.zarr.zip`` both answer True.
    """
    p = Path(path)
    if p.suffix.lower() == ".zarr":
        return True
    return p.suffix.lower() in _ZIP_SUFFIXES and p.stem.lower().endswith(".zarr")


def _metadata_docs_exist(path: Path) -> bool:
    """Is there already a zarr node at ``path``, of EITHER format?

    A zipped store is one file, so its existence is the answer. A directory store
    is identified by its root metadata document — v2's ``.zgroup``/``.zarray`` or
    v3's ``zarr.json`` — rather than by the directory merely existing, because a
    caller may well have created an empty output directory first.

    ORDERING NOTE: :func:`open_group` builds its store before calling this, so for
    a zip this answer is only correct while zarr's ``ZipStore`` stays LAZY —
    constructing one must not create the archive, or a fresh store would look like
    an existing one and skip the format pin. Guarded by
    ``test_creating_a_fresh_zipped_store_still_pins_the_format``.
    """
    if path.suffix.lower() in _ZIP_SUFFIXES:
        return path.is_file()
    return any((path / name).exists() for name in (".zgroup", ".zarray", "zarr.json"))


# --------------------------------------------------------------------------
# Reading metadata documents straight off disk, in either format.
#
# Several tools deliberately inspect a store WITHOUT opening it: the batch-fit
# validator walks thousands of tile directories and wants a structural verdict
# without paying a store open per node, and `luxar serve` needs to know which
# filenames are zarr's rather than a user's data. Those sites named v2's
# documents literally, which stops being true the moment anything writes v3.
#
# They are BI-FORMAT rather than switched on :data:`ZARR_FORMAT`. Luxar's steady
# state now holds both — existing `.luxar.zarr` / `.gsplats.zarr` stores stay v2
# while new output is v3 — so a reader that followed the write format would be
# wrong for exactly the stores it did not create. `luxar serve` will be asked
# for a v2 dataset by a viewer that also loads v3 ones; `batch-fit validate`
# will meet a run whose earlier tiles predate the flip.
#
# DIRECTORY STORES ONLY. They read paths, so a `.zarr.zip` answers False /
# None for every one of them — a COMPLETE zipped store reads as "save
# incomplete", which is the wrong answer rather than a missing feature. That is
# reachable only by a future caller: the sole caller today (`batch-fit
# validate`) skips anything failing `is_dir()` before it gets here, and the
# v2-only code these replaced had exactly the same blind spot. Anything that
# needs to ask these questions of a zipped store should open it through
# :func:`open_store` and inspect the group, not extend these.
# --------------------------------------------------------------------------

_V3_METADATA_DOC = "zarr.json"


def _read_json_doc(path: Path) -> dict[str, Any] | None:
    """Parse a JSON metadata document, or ``None`` if absent/unreadable."""
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def read_consolidated_attrs(store_dir: Path) -> dict[str, dict[str, Any]]:
    """Every node's attributes, keyed by node path, from the consolidated index.

    The root is keyed ``"/"``; children by their store-relative path
    (``"points"``, ``"grp/child"``). Reading the consolidated document rather
    than walking the tree is what makes this ONE file read instead of one per
    node, which is why writers' round-trip tests use it.

    The formats key it differently and this is where that is absorbed. Format 2
    lists every metadata document — ``"points/.zattrs"``, ``"points/.zarray"``,
    ``"points/.zgroup"`` — so the attribute documents have to be picked out by
    suffix. Format 3 lists one entry per NODE, whose attributes are nested under
    ``attributes``, and the root's own attributes live outside
    ``consolidated_metadata`` entirely, in the top-level document.

    Empty when there is no consolidated index — that is "nothing to read here",
    not "a store with no nodes"; use :func:`is_consolidated` to tell them apart.
    The two agree on which document to ask, so they never disagree about a store:
    a v3 root, when present, is answered from EXCLUSIVELY, and a leftover
    ``.zmetadata`` beside it (an ``rsync`` over an older v2 store, an interrupted
    migration) describes a store nobody opens. Consulting it would hand back the
    PRE-copy attributes of every node while :func:`read_node_attrs` and every
    opener report the new ones.
    """
    root_doc = store_dir / _V3_METADATA_DOC
    if root_doc.exists():
        root = _read_json_doc(root_doc)
        if root is None or root.get("consolidated_metadata") is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        root_attrs = root.get("attributes")
        out["/"] = root_attrs if isinstance(root_attrs, dict) else {}
        consolidated = root["consolidated_metadata"]
        entries = (
            consolidated.get("metadata") if isinstance(consolidated, dict) else None
        )
        for path, node in (entries or {}).items():
            if not isinstance(node, dict):
                continue
            attrs = node.get("attributes")
            out[str(path).lstrip("/") or "/"] = attrs if isinstance(attrs, dict) else {}
        return out

    v2 = _read_json_doc(store_dir / ".zmetadata")
    if v2 is None:
        return {}
    out = {}
    for key, value in (v2.get("metadata") or {}).items():
        if key.endswith(".zattrs") and isinstance(value, dict):
            out[key[: -len("/.zattrs")] or "/"] = value
    return out


def read_array_meta(array_dir: Path) -> dict[str, Any] | None:
    """Array metadata (``shape``, ``dtype``, …) read straight off disk.

    Returns ``None`` when ``array_dir`` is not an array — no document, corrupt
    JSON, or a v3 document whose ``node_type`` says group. Both formats spell
    ``shape`` the same way, so callers reading that need no branch of their own.

    A format-3 document, when present, is answered from EXCLUSIVELY, for the
    reason :data:`NODE_ATTR_DOCS` gives: zarr resolves a node carrying both
    documents as format 3, and answering from a stale ``.zarray`` here would hand
    a caller the wrong ``shape`` — which ``batch-fit validate`` checks tiles
    against. So a ``zarr.json`` that says GROUP, or one too corrupt to parse,
    reads as "not an array here" rather than falling through to a v2 document
    that describes a store nobody opens: both of those ARE the corruption the
    validator is looking for, and a plausible stale shape would hide it.
    """
    v3_doc = array_dir / _V3_METADATA_DOC
    if v3_doc.exists():
        v3 = _read_json_doc(v3_doc)
        return v3 if v3 is not None and v3.get("node_type") == "array" else None
    return _read_json_doc(array_dir / ".zarray")


#: The per-node documents that can carry a node's user attributes, best first.
#:
#: Exported because a caller may hold the document's BYTES rather than a path
#: and so cannot use :func:`read_node_attrs` — the archive peek in
#: ``luxar.gsplats.io._archive`` reads one member out of a zip/tar without
#: extracting it, and has to recognise the member by name. Keeping the names
#: here means that peek does not have to know them itself.
#:
#: FORMAT 3 FIRST, because that is how zarr itself resolves a node carrying both
#: documents: it warns ("Both zarr.json and .zgroup metadata objects exist …")
#: and uses format 3. Luxar's own writers never produce that state — appending
#: to a foreign v3 store leaves no v2 shadow root, which
#: ``test_append_to_an_existing_v3_store_does_not_shadow_it`` pins — but an
#: interrupted in-place migration or a half-overwritten copy can, and there the
#: reader must
#: not answer with the stale v2 view while every opener sees v3. Ordering only
#: matters in that mixed case: a single-format store has just one of the two.
#: The viewer's ``ROOT_ATTR_DOCS`` is ordered the same way for the same reason.
NODE_ATTR_DOCS: tuple[str, ...] = (_V3_METADATA_DOC, ".zattrs")

#: The per-node documents whose presence marks a GROUP, best first.
#:
#: Exported for the same reason as :data:`NODE_ATTR_DOCS`: a caller holding an
#: archive INDEX rather than a directory tree — the resolver in
#: ``luxar.gsplats.io._archive`` — must recognise "a store root lives here" from
#: member names alone. Note this is deliberately NOT ``NODE_ATTR_DOCS``:
#: ``.zattrs`` may sit beside an array or be a stray. The marking is exact only
#: at format 2, where ``.zgroup`` is a group's document and nothing else's; at
#: format 3 ``zarr.json`` marks a node of EITHER kind (which is why
#: :func:`read_array_meta` has to check ``node_type``), so a ``zarr.json``
#: describing an ARRAY is recognised here too and only fails later, at
#: ``open_group``, with ``ContainsArrayError`` — the same failure that input
#: already produced, so this is a naming caveat and not a behaviour change.
#:
#: Format 3 first, matching :data:`NODE_ATTR_DOCS`. Today both consumers test
#: MEMBERSHIP rather than iterating, so the order is documentation — but stating
#: it consistently is what keeps a future consumer that does iterate from
#: silently preferring a stale v2 document.
NODE_GROUP_DOCS: tuple[str, ...] = (_V3_METADATA_DOC, ".zgroup")

#: The format-3 per-node metadata document's NAME, exported by itself.
#:
#: CLAUDE.md's rule is that this facade owns metadata-document names, and a
#: caller holding archive member NAMES rather than paths cannot use
#: :func:`read_node_attrs` to find out which format it is looking at — the
#: archive peek in ``luxar.gsplats.io._archive`` has to recognise the format-3
#: document from the name alone, because the two formats' documents are not the
#: same KIND of object and it budgets them differently (a ``.zattrs`` IS the
#: attributes; a ``zarr.json`` also carries the node's structure and, at a
#: consolidated root, the whole index of the tree).
#:
#: A separate name rather than ``NODE_ATTR_DOCS[0]``: the tuples above are
#: ordered best-first as DOCUMENTATION for a future iterating consumer, both of
#: today's consumers test membership, and a positional read would silently swap
#: the meaning of anything keyed on it if that order were ever revisited.
V3_NODE_DOC: str = _V3_METADATA_DOC


def attrs_from_node_doc(parsed: Any, *, doc_name: str | None = None) -> dict[str, Any]:
    """User attributes out of an ALREADY-PARSED node metadata document.

    A format-2 ``.zattrs`` *is* the attributes mapping; a format-3 ``zarr.json``
    nests it under ``attributes`` beside the node's structural fields. Answering
    the format-3 document verbatim would hand back ``shape``/``data_type``/
    ``node_type`` as though a user had authored them.

    The content signal is a ``zarr_format: 3`` member, and alone it is a guess:
    a format-2 document whose USER attributes happen to carry a ``zarr_format``
    key is indistinguishable from a format-3 one and would be answered as ``{}``
    instead of verbatim. Unreachable for a Luxar store — its roots carry
    ``kind``/``format_type``, and zarr never writes ``zarr_format`` into
    ``.zattrs`` — but a foreign store is not ours to constrain.

    ``doc_name``, the file name the bytes came from, settles it. It may only
    DEMOTE: a name that is not ``zarr.json`` vetoes the unwrap, but a name that
    IS ``zarr.json`` never forces one on a document that does not look like a v3
    record. That asymmetry is deliberate. Promoting on the name alone would make
    this answer ``{}`` for any non-v3 body served from a ``zarr.json`` address —
    a shape no real server produces, but one that fakes and misconfigured
    proxies do, and turning those into empty attributes trades a reachable
    failure for an unreachable one.

    Anything that is not a JSON object, and a format-3 document whose
    ``attributes`` is missing or not an object, both yield ``{}``.
    """
    if not isinstance(parsed, dict):
        return {}
    looks_v3 = parsed.get("zarr_format") == 3
    named_v2 = doc_name is not None and doc_name != _V3_METADATA_DOC
    if looks_v3 and not named_v2:
        attrs = parsed.get("attributes")
        return attrs if isinstance(attrs, dict) else {}
    return parsed


def read_node_attrs(node_dir: Path) -> dict[str, Any] | None:
    """A node's user attributes, from v2's ``.zattrs`` or v3's ``zarr.json``.

    ``None`` means "no readable node here", which callers treat as corrupt —
    so an EMPTY attributes mapping must stay distinguishable from a missing
    one, and is returned as ``{}``.

    A ``zarr.json`` that is PRESENT but unparseable answers ``None`` rather than
    falling through to ``.zattrs``: zarr reads that node as format 3 and fails, so
    a stale v2 view of it would be attributes nobody else can see.
    """
    for doc in NODE_ATTR_DOCS:
        path = node_dir / doc
        parsed = _read_json_doc(path)
        if parsed is not None:
            # The document's NAME settles the format; nothing is sniffed.
            return attrs_from_node_doc(parsed, doc_name=doc)
        if doc == _V3_METADATA_DOC and path.exists():
            return None
    return None


def is_consolidated(store_dir: Path) -> bool:
    """Does ``store_dir`` carry consolidated metadata — i.e. did the save finish?

    Luxar's writers consolidate LAST, so this doubles as the completion sentinel
    that batch-fit uses to tell a finished tile from an interrupted one.

    The two formats put it in different places: v2 writes a separate
    ``.zmetadata`` document, while v3 embeds a ``consolidated_metadata`` member
    in the root ``zarr.json`` — which is a zarr-python extension rather than part
    of the v3 spec, but one zarrita implements, so it stays load-bearing for the
    viewer either way.

    A v3 root document, when present, is answered from EXCLUSIVELY — not merely
    first. zarr reads such a store as format 3, so a leftover ``.zmetadata``
    beside it describes a store nobody will open; treating it as the answer
    would report an interrupted v3 save as finished, and this is the sentinel
    ``batch-fit`` uses to tell a complete tile from a half-written one. An
    unparseable v3 root likewise reads as unfinished rather than falling back,
    since a corrupt root IS the interrupted case.
    """
    root = store_dir / _V3_METADATA_DOC
    if root.exists():
        root_doc = _read_json_doc(root)
        return (
            root_doc is not None and root_doc.get("consolidated_metadata") is not None
        )
    return (store_dir / ".zmetadata").exists()


#: zarr group modes that map straight onto a ``ZipStore`` mode. ``"w-"`` is
#: handled separately because it needs a guard rather than a translation; see
#: :func:`_zip_mode`.
_ZIP_MODES = {"r": "r", "r+": "a", "a": "a", "w": "w"}


def _zip_mode(path: Path, mode: str) -> str:
    """Translate a zarr group mode into one zarr's ``ZipStore`` accepts.

    ``ZipStore`` speaks only zipfile's ``r``/``w``/``a``, while the vocabulary of
    :func:`open_group` is zarr's — which also has ``"r+"`` (read-write, must
    already exist) and ``"w-"`` (create, must NOT already exist). Passing either
    through unchanged raises zipfile's own ``ValueError: ZipFile requires mode
    'r', 'w', 'x', or 'a'``, from a layer with no idea what the caller asked for.

    ``"r+"`` becomes ``"a"``: zipfile has no read-write-must-exist mode, and the
    "must exist" half is enforced a level up by ``zarr.open_group``, which will
    not create a root under ``"r+"``.

    ``"w-"`` becomes ``"w"`` plus OUR OWN existence check. It cannot lean on
    zarr's, because a ZipStore opened for ``"w"`` TRUNCATES the archive the first
    time it is touched — before ``zarr.open_group`` ever looks for the root it
    would have refused to overwrite. Exclusive creation would then destroy
    precisely the file it exists to protect.
    """
    if mode == "w-":
        if path.is_file():
            raise FileExistsError(
                f"open_store({str(path)!r}, mode='w-'): the archive already exists"
            )
        return "w"
    try:
        return _ZIP_MODES[mode]
    except KeyError:
        raise ValueError(
            f"open_store({str(path)!r}): mode {mode!r} is not supported for a "
            f"zipped store; expected one of 'r', 'r+', 'w', 'w-', 'a'"
        ) from None


def open_store(path: str | Path, *, mode: str = "r") -> Any:
    """Open the right store class for ``path``.

    ``.zarr.zip`` gets a :class:`zarr.storage.ZipStore`; anything else gets a
    :class:`zarr.storage.LocalStore`. In zarr 2 this dispatch happened inside
    ``zarr.open`` via ``normalize_store_arg``; it is explicit in zarr 3.

    ``mode`` is honoured for both branches: a ZipStore is given the equivalent
    zipfile mode (see :func:`_zip_mode` — the two vocabularies are not the same),
    and a LocalStore is marked ``read_only`` for ``"r"``. Handing back a writable
    store for a declared read would make the argument a decoration rather than a
    constraint, and a read path that acquired a write by accident would not be
    caught.
    """
    p = Path(path)
    if p.suffix.lower() in _ZIP_SUFFIXES:
        return zarr.storage.ZipStore(str(p), mode=_zip_mode(p, mode))
    return zarr.storage.LocalStore(str(p), read_only=(mode == "r"))


def open_group(path: str | Path, *, mode: str = "r", **kwargs: Any) -> zarr.Group:
    """Open (or create) a group, writing :data:`ZARR_FORMAT` when creating.

    ``zarr_format`` is only meaningful when the group is actually being CREATED.
    Asserting it against an existing store would defeat the "read v2 and v3
    alike" property, so it is supplied only for the creating modes — and for
    ``"a"``, only when there is nothing there yet (see below).

    Reads deliberately ignore consolidated metadata; see the comment in the body.
    That is a behavioural choice, not an optimisation: it is what keeps Luxar's
    detection of a partially written store working.
    """
    store: Any = path
    p = Path(path)
    if p.suffix.lower() in _ZIP_SUFFIXES:
        store = open_store(p, mode=mode)
    else:
        store = str(p)

    if mode in ("w", "w-", "a") and "zarr_format" not in kwargs:
        # `"a"` is create-OR-open, so pinning the format unconditionally would
        # assert it against a store that already exists. On a zarr-v3 store that
        # does not fail — it SHADOWS it: zarr writes a second, v2 root beside the
        # v3 one, leaving `zarr.json` and `.zgroup` side by side. Everything
        # written afterwards lands in the v2 view, while an auto-detecting reader
        # resolves the v3 one and cannot see it. A successful write that readers
        # miss is worse than an error, and this migration makes foreign v3 stores
        # reachable for the first time, so `"a"` pins the format only when it is
        # genuinely creating. `"w"`/`"w-"` always create, so they always pin.
        if mode != "a" or not _metadata_docs_exist(p):
            kwargs["zarr_format"] = ZARR_FORMAT

    # Do NOT trust consolidated metadata on read. This restores zarr 2 semantics:
    # there, `.zmetadata` was only consulted via the separate
    # `zarr.open_consolidated`, so `open_group` always saw the arrays that were
    # actually ON DISK. zarr 3 reversed the default and consults it automatically.
    #
    # That difference is not cosmetic — it silently disables Luxar's detection of
    # a PARTIALLY WRITTEN store. Delete an array directory from a consolidated
    # store and zarr 3 still reports the array as present (`"x" in group` is True,
    # `array_keys()` still lists it), because it is answering from the stale
    # index. Luxar's writers carry deliberate crash-safety machinery precisely
    # because half-written stores happen (a killed merge used to leave a partial
    # store that the existence-gated batch-merge resume then treated as
    # complete), and the readers' "this required array is missing" guards are the
    # backstop. Those guards must see the filesystem, not a snapshot of it.
    #
    # The cost is real and worth stating rather than waving away: per-node reads
    # instead of one. Measured on local directory stores it is ~0.44 ms per array
    # (a ~7.8x ratio, linear in array count) — so ~23 ms for a 52-array scene,
    # but ~0.5 s at 1200 arrays and seconds for a large partition.
    #
    # EVERY reader pays it, including the merely informational ones, and that is
    # deliberate. An earlier version of this comment argued the opposite — that
    # `luxar info` / `get_zarr_info` "gain no correctness" because they do not
    # guard on a missing array — which was wrong. They REPORT COUNTS, and on a
    # consolidated store whose `pts/positions` never landed, `get_zarr_info`
    # cheerfully reported the full 1000 points and no error, where zarr 2 reported
    # zero. For a command whose entire job is to say what is on disk, an
    # interrupted write being indistinguishable from a finished one IS the
    # correctness bug; half a second on a large store is the cheaper problem.
    #
    # The VIEWER — which is what consolidated metadata is really for, over HTTP —
    # is untouched either way: it fetches the consolidated document itself.
    #
    # ONE CAVEAT, and it is why in-place attribute edits must go through THIS
    # function rather than `zarr.open_group`. Re-opening an ALREADY-consolidated
    # store with plain `zarr.open_group` returns nodes built FROM the root index
    # (it trusts it), so re-consolidating serializes that stale in-memory tree
    # back out as a NESTED index beneath the root. Both formats grow the nested
    # document; only format 3 is CORRUPTED by it, and that asymmetry is the bug:
    # at format 2 a `.zmetadata` is skipped at every level, whereas at format 3
    # the index lives inside each `zarr.json` and only the ROOT one is bypassed,
    # so the nested copy is honoured. Reads then return pre-edit attributes even
    # though every document on disk is correct, and nothing raises. Re-opening
    # HERE avoids it entirely: the tree carries no index to re-serialize, so
    # consolidating leaves exactly one, at the root — the format-2 invariant
    # every Luxar flow already assumes. Pinned by
    # `test_editing_in_place_leaves_exactly_one_index`.
    if mode in ("r", "r+", "a") and "use_consolidated" not in kwargs:
        kwargs["use_consolidated"] = False
    return zarr.open_group(store, mode=mode, **kwargs)


def create_root_group(store: Any, *, overwrite: bool = True) -> zarr.Group:
    """Create a root group on an already-constructed ``store``.

    The zarr-3 spelling of ``zarr.group(store=store, overwrite=True)``.
    """
    return zarr.create_group(store=store, overwrite=overwrite, zarr_format=ZARR_FORMAT)


def memory_group() -> zarr.Group:
    """An in-memory group at :data:`ZARR_FORMAT` — for tests.

    zarr 3's :class:`~zarr.storage.MemoryStore` is an async store rather than a
    ``MutableMapping``, so the zarr-2 idiom ``zarr.group(store=zarr.MemoryStore())``
    does not survive the upgrade unchanged.
    """
    return zarr.create_group(store=zarr.storage.MemoryStore(), zarr_format=ZARR_FORMAT)


def _translate_chunks(
    chunks: bool | int | tuple[int, ...],
    shape: tuple[int, ...] | None,
    data: np.ndarray | None,
) -> Any:
    """Map a zarr-2 ``ChunkSpec`` onto something zarr 3 accepts.

    zarr 2 spelled "choose chunks for me" as ``chunks=True`` and "one chunk for
    the whole array" as ``chunks=False``; :data:`luxar.typing_utils.ChunkSpec`
    still carries both and ``create_resizable_dataset`` DEFAULTS to ``True``.
    zarr 3 rejects a bool outright ("True is not a valid chunk input"), so the
    translation happens here rather than at ~20 call sites.

    Returns ``None`` for the un-inferable ``False`` case (neither shape nor data
    supplied); the caller's own "needs data or shape" error is the useful one to
    raise there, rather than silently auto-chunking.
    """
    if chunks is True:
        return "auto"
    if chunks is not False:
        return chunks
    if shape is not None:
        return tuple(shape)
    if data is not None:
        return tuple(np.asarray(data).shape)
    return None


def _as_stored(
    data: np.ndarray,
    dtype: Any,
    shape: tuple[int, ...] | None,
    name: str,
) -> np.ndarray:
    """Coerce ``data`` to what should land on disk, checking any declared shape.

    zarr 3 derives both shape and dtype from ``data`` and rejects being given
    either alongside it, so an explicit ``dtype=`` has to be applied by casting
    up front — otherwise a caller asking for float32 would silently store the
    float64 it passed in.
    """
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    if shape is not None and tuple(shape) != tuple(arr.shape):
        raise ValueError(
            f"create_array({name!r}): shape={tuple(shape)} contradicts the "
            f"supplied data's shape {tuple(arr.shape)}"
        )
    return arr


# --------------------------------------------------------------------------
# Compressor translation: numcodecs objects (format 2) -> zarr codecs (format 3)
#
# This is the hazard that does NOT announce itself as a format concern. Luxar's
# compressor policy (`luxar.encoding.compression`) is expressed as numcodecs
# `Blosc` instances, which are exactly right for a format-2 array and are
# REJECTED outright by a format-3 one — `TypeError: 'Blosc' object is not
# iterable`, raised deep in zarr's codec-pipeline parsing, because v3 wants a
# sequence of bytes-to-bytes codec objects instead of one numcodecs filter.
#
# The translation lives here rather than in `compression.py` on purpose: the
# policy is a MEASUREMENT (zstd-9, byte shuffle for multi-byte integer codes —
# manuscript supplementary `codec_selection`) and should be stated once, in the
# module that owns it, without a second format-shaped copy to keep in sync. This
# module already owns "what does the current format need"; that is all this is.
# --------------------------------------------------------------------------

#: numcodecs' integer shuffle constants -> the v3 blosc codec's spelling.
#: ``-1`` is numcodecs' AUTOSHUFFLE; v3 spells "decide for me" as ``None``.
_BLOSC_SHUFFLE_NAMES: dict[int, str | None] = {
    0: "noshuffle",
    1: "shuffle",
    2: "bitshuffle",
    -1: None,
}


def _to_v3_compressor(compressor: Any) -> Any:
    """Translate a numcodecs compressor into its format-3 equivalent.

    ``None`` (store raw) and ``"auto"`` (let zarr choose) are format-neutral and
    pass through. An object that is already a v3 codec passes through too, so a
    caller may hand one over directly.

    ``typesize`` is deliberately NOT set: zarr evolves it from the array's dtype
    at creation, which is what numcodecs' Blosc did implicitly from the buffer.
    Pinning it here to the value visible at policy-definition time would silently
    disable the byte shuffle the policy depends on — the shuffle is only worth
    anything when its element width matches the stored codes.

    Whether zarr can FORWARD that evolved ``typesize`` to blosc is a
    ``numcodecs`` version question, and it is why ``pyproject.toml`` carries a
    direct ``numcodecs>=0.16`` floor: below it zarr's format-3 ``BloscCodec``
    hands numcodecs the serialized byte buffer with no width, the byte shuffle
    becomes a no-op, and the metadata still records the shuffle that did not
    happen. Pinned by
    ``test_zarr_compat.py::test_the_recorded_shuffle_is_the_one_the_chunk_got``.
    """
    if compressor is None or isinstance(compressor, str):
        return compressor
    get_config = getattr(compressor, "get_config", None)
    if not callable(get_config):
        return compressor  # already a v3 codec (or something zarr will judge)

    config = dict(get_config())
    codec_id = config.pop("id", None)
    if codec_id == "blosc":
        shuffle = _BLOSC_SHUFFLE_NAMES.get(int(config.get("shuffle", 0)), None)
        return zarr.codecs.BloscCodec(
            cname=config.get("cname", "zstd"),
            clevel=int(config.get("clevel", 5)),
            shuffle=shuffle,
            blocksize=int(config.get("blocksize", 0)),
        )
    if codec_id == "zstd":
        return zarr.codecs.ZstdCodec(level=int(config.get("level", 0)))
    if codec_id == "gzip":
        return zarr.codecs.GzipCodec(level=int(config.get("level", 5)))
    raise ValueError(
        f"no format-3 equivalent is known for the numcodecs compressor "
        f"{codec_id!r}; add one to _zarr_compat._to_v3_compressor rather than "
        f"letting zarr fail with a codec-pipeline TypeError"
    )


def _to_v3_filter(filter_obj: Any) -> Any:
    """Translate one numcodecs FILTER into its format-3 array-to-array codec.

    Resolved through zarr's codec registry BY NAME rather than by importing the
    Luxar codec class, which would close the import cycle this module exists
    outside of (see the module docstring). The name is the numcodecs
    ``codec_id``, and Luxar registers its format-3 twin under exactly that name
    — one wire name per filter, whichever format is being written.

    A filter with no registered format-3 twin is a hard error rather than a
    silent drop: dropping it would write codes that no reader can invert, and
    the resulting store would look valid and decode to garbage.
    """
    get_config = getattr(filter_obj, "get_config", None)
    if not callable(get_config):
        return filter_obj  # already a v3 codec

    config = dict(get_config())
    codec_id = config.pop("id", None)
    try:
        codec_cls = zarr.registry.get_codec_class(str(codec_id))
    except _UNKNOWN_CODEC_ERRORS:
        raise ValueError(
            f"filter {codec_id!r} has no format-3 codec registered under that "
            f"name; register one (entry-point group 'zarr.codecs') before "
            f"writing format 3, or the array becomes undecodable"
        ) from None
    return codec_cls.from_dict({"name": codec_id, "configuration": config})


def create_array(
    group: zarr.Group,
    name: str,
    *,
    compressor: CompressorLike | str = "auto",
    data: np.ndarray | None = None,
    shape: tuple[int, ...] | None = None,
    # Matches `luxar.typing_utils.ChunkSpec`: an explicit shape, or `True`/`int`
    # meaning "let zarr choose". Spelled out rather than imported to keep this
    # module free of Luxar imports at runtime (see the module docstring).
    chunks: bool | int | tuple[int, ...] | None = None,
    dtype: Any = None,
    filters: list[Any] | None = None,
    overwrite: bool = True,
    **kwargs: Any,
) -> zarr.Array:
    """Create an array under ``group`` — the zarr-3 spelling of ``create_dataset``.

    ``compressor`` distinguishes three cases that zarr 3 would otherwise blur:
    a codec object (use it), ``None`` (store RAW), and ``"auto"`` (let zarr pick,
    which is Blosc/lz4/clevel-5 — identical to zarr 2's implicit default). Every
    production caller names one explicitly; ``"auto"`` exists for test fixtures
    that never did.

    ``data`` and ``shape`` may both be supplied (zarr 2 allowed it, several
    Luxar writers rely on it); only what zarr 3 accepts is forwarded.
    """
    # Translate for the format of the GROUP being written, NOT the module-level
    # default. The two differ routinely and in both directions: a legacy v2 store
    # opened for append takes new arrays while `ZARR_FORMAT` is 3, and a test may
    # build a v2 group deliberately. Keying off the default instead put a v3
    # `BloscCodec` into a v2 array's metadata, which zarr rejects with "Invalid
    # compressor. Expected None, a numcodecs.abc.Codec, ..." — the exact mirror
    # of the v3 failure this translation exists to prevent.
    target_format = getattr(
        getattr(group, "metadata", None), "zarr_format", ZARR_FORMAT
    )

    call: dict[str, Any] = {
        "overwrite": overwrite,
        # An explicit `compressors=` every time — never zarr's "auto". At
        # format 3 the policy's numcodecs objects are translated first; see
        # :func:`_to_v3_compressor` for why that is not `compression.py`'s job.
        "compressors": (
            _to_v3_compressor(compressor) if target_format == 3 else compressor
        ),
        # Filters travel the same road: writers (and the delta probe) build
        # numcodecs objects, and format 3 wants array-to-array codecs.
        "filters": (
            [_to_v3_filter(f) for f in filters]
            if (target_format == 3 and filters)
            else filters
        ),
        **kwargs,
    }
    if chunks is not None:
        translated = _translate_chunks(chunks, shape, data)
        if translated is not None:
            call["chunks"] = translated

    if data is not None:
        return group.create_array(
            name, data=_as_stored(data, dtype, shape, name), **call
        )

    if shape is None:
        raise ValueError(f"create_array({name!r}): needs either `data` or `shape`")
    call["shape"] = shape
    if dtype is not None:
        call["dtype"] = dtype
    return group.create_array(name, **call)


def consolidate(target: zarr.Group | Any) -> None:
    """Write consolidated metadata for ``target``'s store.

    Accepts a :class:`zarr.Group` or a bare store, because both spellings occur
    naturally at call sites — a writer holds the group it just built, while a
    fixture that constructed a store directly holds only the store. Taking just
    the group would leave the second kind reaching for ``zarr.*`` and silently
    skipping the warning suppression below.

    This is load-bearing rather than an optimisation, in BOTH formats: the
    TypeScript scene loader builds its graph purely from the store's
    ``contents()`` listing and has no directory-walking fallback. At format 2 it
    is the ``.zmetadata`` document; at format 3 a ``consolidated_metadata``
    member inside the root ``zarr.json``.

    zarr-python warns that format-3 consolidated metadata is a zarr-python
    EXTENSION rather than part of the v3 spec. That warning is suppressed here,
    deliberately and at this one call site only:

    - The warning is advice about portability, not about this store's validity.
      Luxar's only consumer is zarrita, which implements the extension
      (``isConsolidatedV3`` in its ``extension/consolidation`` module), and a
      reader that does not is unaffected — the member is additive, and the
      per-node ``zarr.json`` documents remain complete and standard.
    - Not suppressing it makes every single save emit a warning, and turns any
      save into a hard failure under ``-W error`` (which is what several tests
      assert warning-freedom with, around a whole compile):
      ``LuxarZarrCompiler.finalize`` catches it and re-raises ``Could not
      finalize Zarr store``. Dropping consolidation to silence it is not an
      option — that produces a store the viewer loads as an empty scene.

    ``catch_warnings`` mutates PROCESS-GLOBAL filter state, so this is only
    sound while no other thread is warning concurrently. It holds today because
    Luxar's parallel writers are subprocesses — the ThreadPoolExecutors in
    ``fit_tiled_parallel`` / ``batch.task_pool`` only launch and wait on them —
    so consolidation always runs on a main thread. Consolidating from a worker
    THREAD would need a different mechanism.

    Matching on the message text means a reworded zarr warning would slip
    through; ``test_consolidate_is_silent`` is the backstop, and it asserts on
    the warning CATEGORY rather than the wording so a reword still fails it.
    """
    store = target.store if isinstance(target, zarr.Group) else target
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*[Cc]onsolidated metadata is currently not part.*",
        )
        warnings.filterwarnings("ignore", message=_PAYLOAD_MEMBER_WARNING)
        zarr.consolidate_metadata(store)


#: The zarr 3 warning raised while ENUMERATING a group that holds a plain payload
#: key (an overlay's ``image.png``, a sound node's ``audio.mp3``). Luxar stores
#: carry such keys by design (``finalize/hashing.py::PAYLOAD_FILE_ATTRS``), so the
#: advice is noise here — and under ``-W error`` it turned every finalize of a
#: scene with an overlay image into ``Could not finalize Zarr store``.
_PAYLOAD_MEMBER_WARNING = (
    r"Object at .* is not recognized as a component of a Zarr hierarchy"
)


@contextmanager
def suppress_payload_member_warning() -> "_Iterator[None]":
    """Scope out zarr's plain-payload member warning for a whole store walk.

    For code that enumerates MANY groups through zarr's own API (the compiler's
    finalize pass, ``luxar info``): one boundary instead of a helper call at
    every ``group_keys()`` site. Same process-global caveat as :func:`consolidate`.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_PAYLOAD_MEMBER_WARNING)
        yield


def array_keys(group: zarr.Group) -> list[str]:
    """``group.array_keys()`` without zarr's plain-payload member warning.

    zarr's member enumeration opens every key under the group's prefix and warns
    about each one that is not a node. A Luxar group may legitimately hold one —
    the raw payload file :func:`write_raw_bytes` stores — so callers that walk a
    whole store (the content hasher, the reader) enumerate through here. Same
    scoping caveat as :func:`consolidate`: ``catch_warnings`` is process-global,
    which holds because Luxar's parallel writers are subprocesses.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_PAYLOAD_MEMBER_WARNING)
        return list(group.array_keys())


def group_keys(group: zarr.Group) -> list[str]:
    """``group.group_keys()`` without zarr's plain-payload member warning (see :func:`array_keys`)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_PAYLOAD_MEMBER_WARNING)
        return list(group.group_keys())


def close(group: zarr.Group) -> None:
    """Close the group's underlying store if it holds an OS resource.

    zarr 2's ``Group`` proxied a ``close()`` to its store; zarr 3's does not.
    Only stores that own a file handle (``ZipStore``) need it — for a
    ``LocalStore`` this is a no-op, which is why the zarr-2 call site guarded on
    ``hasattr`` and never noticed the difference.
    """
    store = getattr(group, "store", None)
    closer = getattr(store, "close", None)
    if callable(closer):
        closer()


def read_raw_bytes(group: zarr.Group, key: str) -> bytes | None:
    """Read a raw, non-zarr key stored *inside* ``group``.

    Some Luxar stores carry plain files alongside their zarr nodes — an overlay
    image written straight into its overlay group's directory. Such a file has
    no chunk grid and no zarr metadata, so no array/group API reaches it; only
    the store itself does, and the store API is async.

    zarr 3.3 *does* ship a public sync facade — ``StorePath.get_sync()``, backed
    by ``zarr.abc.store.SupportsGetSync`` — but it is opt-in per store, and
    ``ZipStore`` does not implement it: ``get_sync()`` there raises ``TypeError:
    Store ZipStore does not support synchronous get``. Driving the async
    ``StorePath.get()`` through ``zarr.core.sync.sync`` is therefore what makes
    this READ store-agnostic — the same answer whether the key sits in a
    ``LocalStore``, a ``MemoryStore``, a ``ZipStore`` or an fsspec-backed one —
    which is why it belongs behind this facade with the rest of the
    version-sensitive surface rather than at its one call site.

    Luxar's overlay writer uses the matching store-level facade below, so raw
    payloads are read through the same abstraction they were written through
    rather than assuming a directory-backed store.

    Args:
        group: Group whose own prefix the key is resolved against — ``group``
            may be the root or any subgroup; ``store_path`` already carries its
            prefix either way.
        key: Store key relative to ``group`` (for a payload file, its filename).

    Returns:
        The key's bytes, or ``None`` when it does not exist.
    """
    # Function-local: this private zarr path is the one thing here that a point
    # release could relocate, and `_zarr_compat` is imported by all of luxar —
    # at module scope such a move would break `import luxar` wholesale.
    from zarr.core.sync import sync

    buffer = sync((group.store_path / key).get())
    return None if buffer is None else buffer.to_bytes()


def list_raw_keys(group: zarr.Group) -> frozenset[str]:
    """The key names stored directly under ``group``'s own prefix.

    Answers one question :func:`read_raw_bytes` structurally cannot: *does this
    store hold a key spelled EXACTLY this?* An open-by-name is resolved by the
    filesystem, and on the case-insensitive ones Luxar treats as first-class
    (macOS, Windows) ``Zarr.json`` folds to the node's own ``zarr.json`` — so a
    probe for a payload that does not exist comes back with the METADATA
    DOCUMENT's bytes, and "no such key" is unobservable for precisely the names
    where the distinction decides whether a payload contributes bytes to the
    content hash (``luxar.io._compiler.finalize.hashing._payload_terms``) or is
    safe to copy (``luxar.io.optimize._copy_payload_files``). A directory
    LISTING is not folded: the store reports the names it actually stores, and
    Python compares them case-sensitively on every platform.

    Immediate children only, subdirectories (subgroups, chunk directories) and
    plain files alike, as bare names relative to ``group`` — never a nested path
    and never the group's own prefix. Both zarr formats' metadata documents show
    up here like any other key, which is the point: they are exactly what a
    case-shifted payload name would collide with.

    Same store-agnostic mechanism, and the same reason for it, as
    :func:`read_raw_bytes`: ``Store.list_dir`` is an async iterator with no
    public sync facade, so it is driven through ``zarr.core.sync`` and verified
    against ``LocalStore``, ``MemoryStore`` and ``ZipStore`` alike. Listing is
    one of the capabilities the store ABC leaves optional (``supports_listing``),
    which is why this is a targeted disambiguator and not how payloads are
    ENUMERATED — the attrs name them in both walks, and the listing only
    disambiguates a metadata-document collision
    (``io/_compiler/finalize/README.md``).

    Args:
        group: Group whose own prefix is listed — ``group`` may be the root or
            any subgroup; ``store_path`` already carries its prefix either way.

    Returns:
        The immediate child key names, compared case-sensitively.
    """
    # Function-local for the reason `read_raw_bytes` gives: these private zarr
    # paths are the one thing here a point release could relocate.
    from zarr.core.sync import collect_aiterator

    store_path = group.store_path
    return frozenset(collect_aiterator(store_path.store.list_dir(store_path.path)))


#: zarr's own per-node metadata documents, lowercased, for the write backstop.
#:
#: Restated here rather than imported from
#: ``luxar.io._compiler.finalize.hashing._ZARR_METADATA_DOCS`` (the same five
#: names, built from the same expression): that module imports THIS one at module
#: scope, so the reverse import is a cycle, and ``_zarr_compat`` is imported by
#: all of luxar — a cycle here breaks ``import luxar`` wholesale. The two sets
#: are pinned equal by
#: ``test_zarr_compat.py::test_the_write_backstop_covers_the_hasher_document_set``.
_METADATA_DOC_KEYS: frozenset[str] = frozenset(
    doc.lower() for doc in (*NODE_ATTR_DOCS, *NODE_GROUP_DOCS, ".zarray", ".zmetadata")
)


def write_raw_bytes(group: zarr.Group, key: str, payload: bytes) -> None:
    """Write a raw, non-zarr key *inside* ``group`` — the twin of the read above.

    Same reasoning, same mechanism: the key has no chunk grid and no zarr
    metadata, so only the store reaches it, and the store API is async. Driving
    ``StorePath.set()`` through ``zarr.core.sync.sync`` is what keeps this
    store-agnostic — a ``LocalStore`` and a ``MemoryStore`` take the same call,
    whereas bypassing the store through a filesystem ``Path`` only works for a
    directory-backed store.

    One backstop lives here rather than only at a call site, because this writes
    UNDER a live node's own prefix and is exported: a key whose last path
    component is one of zarr's metadata documents (``zarr.json``, ``.zgroup``,
    ``.zattrs``, ``.zarray``, ``.zmetadata``) would replace the document the node
    is read through, turning a payload write into a destroyed store. Matched
    case-insensitively, since ``Zarr.json`` IS that document on a
    case-insensitive filesystem. Callers still own the REST of the name check —
    that a payload name is a single path component at all, and what to do about a
    refused one — since only they know whether skipping, refusing or renaming is
    the right answer (``luxar.io.optimize._copy_payload_files`` decides all
    three).

    Args:
        group: Group whose own prefix the key is resolved against.
        key: Store key relative to ``group`` (for a payload file, its filename).
        payload: The bytes to store under that key.

    Raises:
        ValueError: If ``key`` names a zarr metadata document.
    """
    if key.rsplit("/", 1)[-1].lower() in _METADATA_DOC_KEYS:
        raise ValueError(
            f"write_raw_bytes refuses the key {key!r}: it names one of zarr's "
            f"own metadata documents, so writing it would replace the document "
            f"the node at {group.path or '/'!r} is read through. Store the "
            f"payload under a different name."
        )
    # Function-local for the reason `read_raw_bytes` gives: these private zarr
    # paths are the one thing here a point release could relocate.
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync((group.store_path / key).set(buffer))


def is_missing_error(exc: BaseException) -> bool:
    """Is ``exc`` zarr's "no store / no node here" error?

    zarr 2 raised ``zarr.errors.PathNotFoundError`` / ``GroupNotFoundError``.
    Only the FIRST is gone in zarr 3 — ``GroupNotFoundError`` survives and now
    subclasses :class:`FileNotFoundError` (its MRO is ``GroupNotFoundError →
    NodeNotFoundError → BaseZarrError → ValueError → FileNotFoundError``), so a
    plain ``except FileNotFoundError`` catches both "no store here" and "there is
    a node here but it is an array, not a group". Callers that used to name the
    zarr-specific pair go through here or catch ``FileNotFoundError`` directly.
    """
    return isinstance(exc, FileNotFoundError | KeyError)
