"""The one module in Luxar that knows which zarr *format* version we write.

Luxar runs on **zarr-python 3** but still writes **zarr format 2** stores. Those
are two independent axes and this module is where they are pinned together:
:data:`ZARR_FORMAT` selects the on-disk format, and every helper below routes a
Luxar call through the zarr-3 API in a way that reproduces the format-2 bytes
the viewer (and every published `.luxar.zarr` / `.gsplats.zarr`) already expects.

Why a facade at all
-------------------
The TypeScript viewer has exactly one module that imports zarrita
(``src/data/zarr.ts``); everything else speaks in Luxar concepts. That design
made the viewer nearly version-agnostic for free. The Python side had no
equivalent — 75 production modules imported ``zarr`` directly — so a format
change meant sweeping all of them. Routing through here means a future move to
format 3 edits *this* file, not that whole surface.

Two zarr-3 behaviours are actively dangerous here, and both are neutralised
below rather than left to call sites:

1. **An omitted compressor is not "no compressor".** zarr's ``compressors="auto"``
   silently applies Blosc/lz4/clevel-5, whereas Luxar has arrays that must be
   stored RAW (the packed label byte-blobs) and others that must carry the
   measured zstd-9 policy. ``None`` is therefore mapped to an explicit
   ``compressors=None`` rather than being confused with "unspecified". The
   ``"auto"`` default here reproduces zarr 2's implicit default exactly, so test
   fixtures that never named a compressor keep their old bytes; production code
   is held to naming one by
   ``test_zarr_compat.py::test_production_create_array_calls_pass_a_compressor``,
   which walks the AST rather than trusting review.
2. **``data=`` and ``shape=`` are mutually exclusive.** zarr 2's
   ``create_dataset`` accepted both, and several Luxar writers passed both.
   Passing both to zarr 3 raises. :func:`create_array` accepts both, casts
   ``data`` to ``dtype`` when given, and forwards only what zarr 3 allows.

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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

if TYPE_CHECKING:  # pragma: no cover - typing only
    # The canonical home of the alias — `typing_utils.protocols` only re-imports
    # it under TYPE_CHECKING, so it is not a re-export mypy will follow.
    from luxar.encoding.compression import CompressorLike

__all__ = [
    "ZARR_FORMAT",
    "close",
    "consolidate",
    "create_array",
    "create_root_group",
    "is_missing_error",
    "memory_group",
    "open_group",
    "open_store",
]

#: The zarr format Luxar WRITES. Reading is version-agnostic — zarr-python 3
#: opens format 2 and format 3 stores alike, which is the whole point of being
#: on 3.x while still emitting 2 (a v3 store from, say, a GEFF-writing tracking
#: tool is readable, but nothing Luxar produces changes shape).
#:
#: Flipping this to 3 is NOT sufficient on its own to migrate the format: the
#: `luxar_delta_v1` filter would have to become a `zarr.codecs` entry-point
#: codec, and the viewer has its own v2 assumptions (raw `.zattrs` fetches in
#: cache validation and the scene-identity watchdog). See the migration plan.
ZARR_FORMAT = 2

# Suffixes that mean "this path is a zipped store, not a directory". zarr 2
# sniffed these inside `zarr.open`; zarr 3 requires the store to be chosen
# explicitly, so the sniffing lives here now.
_ZIP_SUFFIXES = (".zip",)


def open_store(path: str | Path, *, mode: str = "r") -> Any:
    """Open the right store class for ``path``.

    ``.zarr.zip`` gets a :class:`zarr.storage.ZipStore`; anything else gets a
    :class:`zarr.storage.LocalStore`. In zarr 2 this dispatch happened inside
    ``zarr.open`` via ``normalize_store_arg``; it is explicit in zarr 3.
    """
    p = Path(path)
    if p.suffix.lower() in _ZIP_SUFFIXES:
        return zarr.storage.ZipStore(str(p), mode=mode)
    return zarr.storage.LocalStore(str(p))


def open_group(path: str | Path, *, mode: str = "r", **kwargs: Any) -> zarr.Group:
    """Open (or create) a group, writing :data:`ZARR_FORMAT` when creating.

    ``zarr_format`` is only meaningful when the group is being CREATED. Passing
    it for a read of an existing store would assert the format rather than
    detect it, which would defeat the "read v2 and v3 alike" property, so it is
    supplied only for the creating modes.
    """
    store: Any = path
    p = Path(path)
    if p.suffix.lower() in _ZIP_SUFFIXES:
        store = open_store(p, mode=mode)
    else:
        store = str(p)

    if mode in ("w", "w-", "a") and "zarr_format" not in kwargs:
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
    # The cost is per-node metadata reads instead of one — irrelevant for the
    # local directory stores this reader path handles, and the VIEWER (which is
    # what consolidated metadata is really for, over HTTP) is untouched by this:
    # it fetches `.zmetadata` itself and is unaffected.
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
    call: dict[str, Any] = {
        "overwrite": overwrite,
        # An explicit `compressors=` every time — never zarr's "auto".
        "compressors": compressor,
        "filters": filters,
        **kwargs,
    }
    if chunks is not None:
        # zarr 2 spelled "choose chunks for me" as `chunks=True` (and `False` as
        # "one chunk for the whole array"); `luxar.typing_utils.ChunkSpec` still
        # carries those, and `create_resizable_dataset` defaults to True. zarr 3
        # rejects a bool outright — "True is not a valid chunk input" — so it is
        # translated here rather than at ~20 call sites.
        if chunks is True:
            call["chunks"] = "auto"
        elif chunks is False:
            # One chunk spanning the array. Whichever of shape/data was given
            # supplies it; if neither was, the ValueError below is the right
            # error to surface rather than a silently auto-chunked array.
            if shape is not None:
                call["chunks"] = tuple(shape)
            elif data is not None:
                call["chunks"] = tuple(np.asarray(data).shape)
        else:
            call["chunks"] = chunks

    if data is not None:
        # zarr 3 rejects `data=` together with `shape=`/`dtype=`, deriving both
        # from the array instead. Cast first so an explicit `dtype=` still wins,
        # then let zarr read the shape off the (possibly cast) array.
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if shape is not None and tuple(shape) != tuple(arr.shape):
            raise ValueError(
                f"create_array({name!r}): shape={tuple(shape)} contradicts the "
                f"supplied data's shape {tuple(arr.shape)}"
            )
        return group.create_array(name, data=arr, **call)

    if shape is None:
        raise ValueError(f"create_array({name!r}): needs either `data` or `shape`")
    call["shape"] = shape
    if dtype is not None:
        call["dtype"] = dtype
    return group.create_array(name, **call)


def consolidate(group: zarr.Group) -> None:
    """Write consolidated metadata for ``group``'s store.

    At :data:`ZARR_FORMAT` 2 this is the ``.zmetadata`` document the viewer
    fetches once to enumerate a whole scene, so it is load-bearing rather than
    an optimisation: the TypeScript scene loader builds its graph purely from
    the store's ``contents()`` listing and has no directory-walking fallback.
    """
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
