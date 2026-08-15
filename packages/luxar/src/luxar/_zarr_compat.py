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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

if TYPE_CHECKING:  # pragma: no cover - typing only
    # The canonical home of the alias — `typing_utils.protocols` only re-imports
    # it under TYPE_CHECKING, so it is not a re-export mypy will follow.
    from luxar.encoding.compression import CompressorLike

__all__ = [
    "DEFAULT_ZARR_FORMAT",
    "SUPPORTED_ZARR_FORMATS",
    "ZARR_FORMAT",
    "ZARR_FORMAT_ENV_VAR",
    "close",
    "consolidate",
    "create_array",
    "create_root_group",
    "is_consolidated",
    "is_missing_error",
    "is_zarr_path",
    "memory_group",
    "open_group",
    "open_store",
    "read_array_meta",
    "read_consolidated_attrs",
    "read_node_attrs",
    "set_zarr_format",
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
    """
    v2 = _read_json_doc(store_dir / ".zmetadata")
    if v2 is not None:
        out: dict[str, dict[str, Any]] = {}
        for key, value in (v2.get("metadata") or {}).items():
            if key.endswith(".zattrs") and isinstance(value, dict):
                out[key[: -len("/.zattrs")] or "/"] = value
        return out

    root = _read_json_doc(store_dir / _V3_METADATA_DOC)
    if root is None or root.get("consolidated_metadata") is None:
        return {}
    out = {}
    root_attrs = root.get("attributes")
    out["/"] = root_attrs if isinstance(root_attrs, dict) else {}
    consolidated = root["consolidated_metadata"]
    entries = consolidated.get("metadata") if isinstance(consolidated, dict) else None
    for path, node in (entries or {}).items():
        if not isinstance(node, dict):
            continue
        attrs = node.get("attributes")
        out[str(path).lstrip("/") or "/"] = attrs if isinstance(attrs, dict) else {}
    return out


def read_array_meta(array_dir: Path) -> dict[str, Any] | None:
    """Array metadata (``shape``, ``dtype``, …) read straight off disk.

    Returns ``None`` when ``array_dir`` is not an array — no document, corrupt
    JSON, or a v3 document whose ``node_type`` says group. Both formats spell
    ``shape`` the same way, so callers reading that need no branch of their own.
    """
    v2 = _read_json_doc(array_dir / ".zarray")
    if v2 is not None:
        return v2
    v3 = _read_json_doc(array_dir / _V3_METADATA_DOC)
    if v3 is not None and v3.get("node_type") == "array":
        return v3
    return None


def read_node_attrs(node_dir: Path) -> dict[str, Any] | None:
    """A node's user attributes, from v2's ``.zattrs`` or v3's ``zarr.json``.

    ``None`` means "no readable node here", which callers treat as corrupt —
    so an EMPTY attributes mapping must stay distinguishable from a missing
    one, and is returned as ``{}``.
    """
    v2 = _read_json_doc(node_dir / ".zattrs")
    if v2 is not None:
        return v2
    v3 = _read_json_doc(node_dir / _V3_METADATA_DOC)
    if v3 is not None:
        attrs = v3.get("attributes")
        return attrs if isinstance(attrs, dict) else {}
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
    """
    if (store_dir / ".zmetadata").exists():
        return True
    root_doc = _read_json_doc(store_dir / _V3_METADATA_DOC)
    return root_doc is not None and root_doc.get("consolidated_metadata") is not None


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
    # function rather than `zarr.open_group`. Format 3 allows a consolidated
    # index on ANY group, not just the root, and `use_consolidated=False`
    # bypasses only the ROOT one — a nested index is still honoured. Nested
    # indexes appear when an ALREADY-consolidated store is re-opened with plain
    # `zarr.open_group` (which trusts the root index, so the nodes it returns
    # are built from it) and then re-consolidated: the stale in-memory tree is
    # serialized back out beneath the root. Subsequent reads then see the
    # pre-edit attributes even though every document on disk is correct, and
    # nothing raises. Re-opening HERE avoids it: the tree carries no index to
    # re-serialize, so consolidating leaves exactly one index, at the root —
    # the format-2 invariant every Luxar flow already assumes.
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
    except KeyError:
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


def consolidate(group: zarr.Group) -> None:
    """Write consolidated metadata for ``group``'s store.

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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*[Cc]onsolidated metadata is currently not part.*",
        )
        zarr.consolidate_metadata(group.store)


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
