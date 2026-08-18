"""Re-chunk an existing zarr store in one structure-preserving pass.

``luxar optimise`` exists because a store that is already on disk is usually
badly chunked for STREAMING, and regenerating it is not an option: the source
volume may be gone, the fit may have cost GPU-days, and the copy on Zenodo has a
DOI. The demo corpus measured 606,349 files for 2.94 GB — an average file of
5.1 KB, 97.0% of them under 16 KB — against a
:data:`~luxar.typing_utils.constants.TARGET_CHUNK_BYTES` of 64 KB. Cold-loading
one demo from object storage took 245 s / 9,390 requests as written and 51 s /
2,348 requests after this pass. It was never bandwidth-bound (38.6 MB over
245 s); it was round trips.

Fixing the WRITER (which #1713 did) does not subsume this, for three reasons.
It cannot reach data that already exists. It sizes its byte budget from the
array's INPUT dtype — float32 positions, before the encoder quantizes them to
uint16 — so its chunks land 2-4x under target, whereas a post-hoc pass reads the
STORED dtype and sees the real itemsize. And authoring and serving want
different targets, which is a flag rather than a rebuild.

What is preserved, exactly
--------------------------
Everything except the zarr chunk grid: array VALUES bit-for-bit, dtype,
compressor, filters, serializer, ``fill_value``, memory ``order``, every group
and array attribute, and the on-disk zarr FORMAT (a v2 store stays v2 — format
conversion is ``gsplat migrate-format``'s job, not this one). Codecs are reused
from the SOURCE array rather than re-derived, because an omitted compressor is
not "no compressor" (zarr's ``"auto"`` is Blosc/lz4 at format 2 and zstd at
format 3) and some Luxar arrays are deliberately RAW.

The spatial-index contract
--------------------------
``chunk_size`` and ``chunk_bounds`` ARE the viewer's partition grid, and this
pass must not move it. Every emitted chunk is therefore a whole multiple of the
node's ``chunk_size`` atom — rounded DOWN from the byte budget, never below one
atom — so a partition's row range still falls inside a single zarr chunk and no
row-range read straddles a boundary. The bounds arrays themselves are never
re-chunked.

The cache-invalidation hazard
-----------------------------
:func:`~luxar.io._compiler.finalize.hashing.compute_content_hashes` hashes array
values and attrs, NOT chunk shapes, and the viewer's ``MultiLevelCachingStore``
validates its persistent cache by comparing ``content_hash`` against the remote.
A naive re-chunk therefore produces a store whose hash is IDENTICAL while chunk
key ``0/0`` covers a different row range — a client with a warm cache believes
itself up to date and serves chunks that no longer mean what their keys say.
Silent wrong data, on the one path with no error to raise.

The guard is ONE mechanism with two halves, and it is worth being precise about
which half does the work, because a plausible-sounding simplification deletes
the real one. Neither hasher can see a chunk shape — the scene walk hashes
values + attrs, and the ``.gsplats.zarr`` stamp hashes ``(name, shape, dtype)``
+ attrs — so recomputing a hash over an otherwise identical store reproduces the
SOURCE's hash exactly. What moves it is the ``chunk_layout`` summary attr
written on the root: attrs are hashed, so the new layout lands in the digest.
The restamp is what PROPAGATES that attr into the stored ``content_hash`` the
viewer actually compares. Remove either and the output ships the source's hash.
Suppressing the attr alone was measured to leave the output hash byte-identical.

(#1719 additionally folds chunk shapes into the tree hash; once that lands the
restamp becomes independently sufficient and the attr becomes documentation.
Until then, both are load-bearing.)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import xxhash
import zarr
from arbol import aprint, asection

from .._zarr_compat import close, consolidate, create_array, open_group
from ..typing_utils.constants import (
    MAX_CHUNK_BYTES,
    MIN_CHUNK_BYTES,
    TARGET_CHUNK_BYTES,
)
from ._compiler.chunking import _atom_aligned_rows

__all__ = [
    "CHUNK_PROFILES",
    "ArrayPlan",
    "ChunkLayoutSummary",
    "OptimisePlan",
    "optimise_store",
    "plan_optimisation",
    "resolve_target_bytes",
    "summarise_chunk_layout",
]


#: Byte targets selected by ``--profile``.
#:
#: ``local`` is the authoring default: a 64 KB chunk is the balance point
#: between request count and over-fetch when a read costs microseconds.
#: ``hosting`` is the top of the documented 16-256 KB band — over object storage
#: a round trip costs ~100 ms and over-fetching a few hundred KB is free by
#: comparison. ``archive`` deliberately leaves that band: an archived store is
#: not being streamed, and its only real cost is the file COUNT (upload time,
#: inode pressure, per-object storage minimums).
#:
#: The larger targets trade PARTIAL-QUERY bytes for FULL-LOAD requests, so
#: "object storage, therefore ``hosting``" is not unconditional. A Points or
#: GSplats node is not loaded whole — the viewer turns visible index chunks into
#: element ranges of ``chunk_size`` atoms, and one atom-hit costs one zarr chunk
#: whatever its size. Measured on a real store (atom 2340, uint16 ``(N, 3)``):
#: ``local`` 4 atoms / 54.8 KB per partial hit, ``hosting`` 18 atoms / 246.8 KB
#: (4.5x), ``archive`` 74 atoms / 1014.6 KB (18x). Size up when the access
#: pattern is "load the node whole"; stay on ``local`` when the viewer will be
#: slicing into a large one.
CHUNK_PROFILES: dict[str, int] = {
    "local": TARGET_CHUNK_BYTES,
    "hosting": MAX_CHUNK_BYTES,
    "archive": 1024 * 1024,
}

#: Array names that ARE the spatial index. Re-chunking one would move the grid
#: the viewer resolves row ranges against, which is the one thing this pass
#: promises not to do.
_INDEX_ARRAYS = frozenset(
    {"chunk_bounds", "vertex_chunk_bounds", "segment_chunk_bounds"}
)

#: The Lines array indexed by SEGMENT rather than by vertex. Lines carries two
#: independent atoms (``vertex_ordering`` / ``segment_ordering``) and this is the
#: only array on the second grid.
_SEGMENT_ARRAY = "segments"

#: Upper bound on one slab of a value copy. Large enough that the copy is not
#: syscall-bound, small enough that a 629 MB array (one exists in the demo
#: corpus) never lands in memory whole.
_SLAB_BYTES = 64 * 1024 * 1024


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrayPlan:
    """What will happen to one array, and why."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    source_chunks: tuple[int, ...]
    target_chunks: tuple[int, ...]
    source_n_chunks: int
    target_n_chunks: int
    atom: int | None
    #: Empty when the array is being re-chunked; otherwise the reason it is not.
    skip_reason: str = ""

    @property
    def rechunked(self) -> bool:
        """True when this array gets a new chunk grid."""
        return not self.skip_reason

    @property
    def target_chunk_bytes(self) -> int:
        """Nominal payload of the planned chunk, in bytes."""
        return _chunk_bytes(self.target_chunks, np.dtype(self.dtype))


@dataclass(frozen=True)
class OptimisePlan:
    """The whole-store plan — what :func:`optimise_store` will do, in advance."""

    target_bytes: int
    profile: str | None
    arrays: list[ArrayPlan] = field(default_factory=list)

    @property
    def n_rechunked(self) -> int:
        """How many arrays get a new chunk grid."""
        return sum(1 for a in self.arrays if a.rechunked)

    @property
    def source_n_chunks(self) -> int:
        """Chunk files the input store holds — the requests a full load costs now."""
        return sum(a.source_n_chunks for a in self.arrays)

    @property
    def target_n_chunks(self) -> int:
        """Chunk files the output will hold — the requests it will cost instead."""
        return sum(a.target_n_chunks for a in self.arrays)


@dataclass(frozen=True)
class ChunkLayoutSummary:
    """The streaming-shape diagnostic ``luxar info --stats`` reports.

    Computed off the same walk the optimiser plans from, so "is this store worth
    optimising?" is answerable without hosting it first.
    """

    n_arrays: int
    n_chunks: int
    #: Chunk-count-weighted mean of the NOMINAL chunk payload, in bytes.
    mean_chunk_bytes: float
    #: Arrays whose nominal chunk payload is below
    #: :data:`~luxar.typing_utils.constants.MIN_CHUNK_BYTES`. Single-chunk arrays
    #: count: a store of a hundred tiny arrays is request-heavy for exactly the
    #: reason a store of tiny chunks is.
    n_arrays_under_floor: int

    @property
    def share_under_floor(self) -> float:
        """Fraction of arrays chunked below the 16 KB floor, in ``[0, 1]``."""
        return self.n_arrays_under_floor / self.n_arrays if self.n_arrays else 0.0


# --------------------------------------------------------------------------
# Walking
# --------------------------------------------------------------------------


def _walk_arrays(group: zarr.Group, path: str = "") -> Iterator[tuple[str, zarr.Array]]:
    """Yield ``(store-relative path, array)`` for every array in the tree."""
    for name in sorted(group.array_keys()):
        yield (f"{path}/{name}" if path else name), group[name]
    for name in sorted(group.group_keys()):
        child_path = f"{path}/{name}" if path else name
        yield from _walk_arrays(group[name], child_path)


def _chunk_bytes(chunks: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    """The NOMINAL payload of one chunk, uncompressed — the target's units."""
    return int(math.prod(chunks)) * int(dtype.itemsize)


def _n_chunks(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    """How many chunk files this grid produces — i.e. how many HTTP requests.

    A zero-extent axis yields ZERO files, not one: zarr writes no chunk for a
    ``(0, D)`` array, and rounding that up would make this diagnostic overstate
    the request count of every ``array_ref`` placeholder in the store.
    """
    if not shape:
        return 1
    total = 1
    for extent, step in zip(shape, chunks):
        if int(extent) <= 0:
            return 0
        total *= -(-int(extent) // max(1, int(step)))
    return total


def _file_grid(array: zarr.Array) -> tuple[int, ...]:
    """The grid whose cells are FILES — the shard grid when sharded, else chunks.

    A sharded v3 array packs many chunks into one object, so counting its inner
    chunk grid would report 200 requests where 5 files exist. This diagnostic's
    entire job is predicting the request count, so it counts objects.
    """
    shards = getattr(array, "shards", None)
    if shards:
        return tuple(int(s) for s in shards)
    return tuple(int(c) for c in array.chunks)


# --------------------------------------------------------------------------
# Atom discovery
# --------------------------------------------------------------------------


def _atom_candidate(
    attrs: dict[str, Any], array_name: str
) -> tuple[int | None, str | None]:
    """The node's declared atom for ``array_name``, and the array that proves it.

    Points and GSplats put the atom at the top level (``chunk_size``, proven by
    the ``chunk_bounds`` array). Lines nests TWO of them —
    ``vertex_ordering.chunk_size`` for every per-vertex array and
    ``segment_ordering.chunk_size`` for ``segments`` alone — each with its own
    bounds array. Returning the proving array's name rather than a bare int is
    what lets :func:`_resolve_atom` tell a real partition grid from a vestigial
    ``chunk_size`` default.
    """
    if array_name == _SEGMENT_ARRAY:
        nested = attrs.get("segment_ordering")
        if isinstance(nested, dict) and "chunk_size" in nested:
            return _as_atom(nested["chunk_size"]), "segment_chunk_bounds"
        return None, None

    nested = attrs.get("vertex_ordering")
    if isinstance(nested, dict) and "chunk_size" in nested:
        return _as_atom(nested["chunk_size"]), "vertex_chunk_bounds"
    if "chunk_size" in attrs:
        return _as_atom(attrs["chunk_size"]), "chunk_bounds"
    return None, None


def _as_atom(raw: Any) -> int | None:
    """Coerce a stored ``chunk_size`` attr to a positive int, or ``None``."""
    try:
        atom = int(raw)
    except (TypeError, ValueError):
        return None
    return atom if atom > 0 else None


def _resolve_atom(
    group_attrs: dict[str, Any],
    array_names: frozenset[str],
    array_name: str,
) -> int | None:
    """The atom this array's new chunk must be a multiple of, or ``None``.

    A ``chunk_size`` attr is NOT sufficient on its own. A gsplat leaf written
    with ``ordering="none"`` still gets a default ``chunk_size`` stamped
    (``gsplat_tree.py``: ``min(1024, max(64, n_splats))``), and its arrays are
    not on that grid — trusting it would round a chunk down to a boundary that
    indexes nothing while inflating it to at least one bogus atom.

    The BOUNDS array is the only proof accepted. An earlier version also trusted
    a bare ``chunk_size`` when the array's current chunk happened to be a
    multiple of it, which is defeated by the very value the writer emits: the
    vestigial default is a power of two, so ``amplitudes`` chunked ``(16384,)``
    passed the coincidence test while ``centers`` at ``(5461, 3)`` on the SAME
    node did not — one node, two answers, decided by arithmetic luck. Every
    Luxar writer omits the bounds array only when there are zero rows
    (``spatial_ordering/points.py``, ``spatial_ordering/lines.py``,
    ``gsplat_assembly.py`` all guard on ``len(chunk_bounds) > 0``), so a node
    with rows to re-chunk always carries its proof.
    """
    atom, proof = _atom_candidate(group_attrs, array_name)
    if atom is None or proof is None or proof not in array_names:
        return None
    return atom


# --------------------------------------------------------------------------
# Chunk sizing
# --------------------------------------------------------------------------


def _encoding_name(array: zarr.Array) -> str:
    """The array's ``encoding.name`` attr (``""`` when it is stored directly)."""
    enc = array.attrs.get("encoding")
    if isinstance(enc, dict):
        name = enc.get("name")
        if isinstance(name, str):
            return name
    return ""


def _structural_skip(
    name: str,
    array: zarr.Array,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> str:
    """Why this array must keep its chunk grid, or ``""`` if it may be re-chunked.

    Every branch here is a shape that appears in real demo stores and that a
    naive first pass got wrong: the bounds arrays ARE the index, an ``array_ref``
    is a physical ``(0, D)`` placeholder whose values live elsewhere, and a
    broadcast is a single stored value standing in for N of them.
    """
    if name in _INDEX_ARRAYS:
        return "spatial index"
    if not shape:
        return "scalar array"
    if getattr(array, "shards", None) is not None:
        return "sharded"
    encoding = _encoding_name(array)
    if encoding == "array_ref":
        return "array_ref"
    if shape[0] == 0:
        return "empty"
    if encoding == "broadcasted" or shape[0] == 1:
        return "broadcast"
    if chunks[0] >= shape[0]:
        return "single chunk"
    return ""


def _plan_array(
    path: str,
    array: zarr.Array,
    atom: int | None,
    target_bytes: int,
) -> ArrayPlan:
    """Decide this array's new chunk shape (or why it keeps the one it has)."""
    shape = tuple(int(s) for s in array.shape)
    chunks = tuple(int(c) for c in array.chunks)
    dtype = np.dtype(array.dtype)
    name = path.rsplit("/", 1)[-1]
    # Files, not nominal grid cells — a sharded array's objects are its shards.
    file_grid = _file_grid(array)

    def keep(reason: str) -> ArrayPlan:
        """A plan that changes nothing, recording why."""
        n = _n_chunks(shape, file_grid)
        return ArrayPlan(
            path=path,
            shape=shape,
            dtype=str(dtype),
            source_chunks=chunks,
            target_chunks=chunks,
            source_n_chunks=n,
            target_n_chunks=n,
            atom=atom,
            skip_reason=reason,
        )

    structural = _structural_skip(name, array, shape, chunks)
    if structural:
        return keep(structural)

    # Bytes per row of a chunk, not of the array: for a 2D Luxar array the two
    # agree, but an array chunked on a trailing axis too must not be sized as
    # though its rows were whole.
    row_bytes = max(1, int(math.prod(chunks[1:])) * int(dtype.itemsize))
    ideal_rows = max(1, int(target_bytes) // row_bytes)

    if atom is not None:
        rows = _atom_aligned_rows(ideal_rows, atom, shape[0])
    else:
        rows = max(1, min(shape[0], ideal_rows))

    if rows <= chunks[0]:
        # Never shrink. A smaller chunk is strictly more requests, and on a
        # store already chunked above the requested target that is the honest
        # answer rather than an "optimisation".
        return keep("already at or above target")

    target_chunks = (rows, *chunks[1:])
    return ArrayPlan(
        path=path,
        shape=shape,
        dtype=str(dtype),
        source_chunks=chunks,
        target_chunks=target_chunks,
        source_n_chunks=_n_chunks(shape, chunks),
        target_n_chunks=_n_chunks(shape, target_chunks),
        atom=atom,
    )


def plan_optimisation(
    root: zarr.Group,
    *,
    target_bytes: int = TARGET_CHUNK_BYTES,
    profile: str | None = None,
) -> OptimisePlan:
    """Plan the re-chunk of ``root`` without writing anything.

    Every leaf is planned independently — a ``kind=lod`` level, a
    ``kind=partition`` part and an ``additive_<i>`` rung each carry their own
    atom — so nesting needs no special case beyond the recursive walk.
    """
    plans: list[ArrayPlan] = []
    for group_path, group in _walk_groups(root):
        attrs = dict(group.attrs)
        names = frozenset(group.array_keys())
        for name in sorted(names):
            array = group[name]
            path = f"{group_path}/{name}" if group_path else name
            atom = _resolve_atom(attrs, names, name)
            plans.append(_plan_array(path, array, atom, target_bytes))
    return OptimisePlan(target_bytes=target_bytes, profile=profile, arrays=plans)


def _walk_groups(group: zarr.Group, path: str = "") -> Iterator[tuple[str, zarr.Group]]:
    """Yield ``(store-relative path, group)`` for the root and every subgroup."""
    yield path, group
    for name in sorted(group.group_keys()):
        yield from _walk_groups(group[name], f"{path}/{name}" if path else name)


def summarise_chunk_layout(root: zarr.Group) -> ChunkLayoutSummary:
    """Measure a store's streaming shape: chunk sizes and request count."""
    n_arrays = 0
    n_chunks = 0
    total_bytes = 0
    under_floor = 0
    for _path, array in _walk_arrays(root):
        shape = tuple(int(s) for s in array.shape)
        # The FILE grid, so the projected request count matches the objects a
        # loader actually fetches (a shard is one object, however many chunks
        # it packs) and a zero-row array contributes none.
        grid = _file_grid(array)
        dtype = np.dtype(array.dtype)
        count = _n_chunks(shape, grid)
        payload = _chunk_bytes(grid, dtype)
        n_arrays += 1
        n_chunks += count
        total_bytes += payload * count
        if payload < MIN_CHUNK_BYTES:
            under_floor += 1
    return ChunkLayoutSummary(
        n_arrays=n_arrays,
        n_chunks=n_chunks,
        mean_chunk_bytes=(total_bytes / n_chunks) if n_chunks else 0.0,
        n_arrays_under_floor=under_floor,
    )


# --------------------------------------------------------------------------
# Copying
# --------------------------------------------------------------------------


def _copy_array(
    source: zarr.Array,
    dest_group: zarr.Group,
    name: str,
    chunks: tuple[int, ...],
    target_format: int,
) -> zarr.Array:
    """Recreate ``source`` under ``dest_group`` with a new chunk grid.

    Codecs come off the SOURCE array — never re-derived — so a RAW array stays
    raw and a measured zstd-9 policy survives verbatim. ``compressors=()`` is
    what zarr reports for a raw array and is exactly what it accepts back; that
    is a different thing from ``"auto"``, which would silently compress it.

    The SHARD grid is forwarded too. A sharded array is never re-chunked (see
    :func:`_structural_skip`), but recreating it from ``chunks`` alone would
    promote its INNER chunk shape to the top level: a ``(200000, 4)`` float32
    array chunked ``(1000, 4)`` inside shards of ``(50000, 4)`` went from 5
    files to 201 — a 40x increase in round trips, reported by the plan as
    "unchanged".
    """
    extra: dict[str, Any] = {
        "fill_value": source.fill_value,
        "attributes": dict(source.attrs),
    }
    shards = getattr(source, "shards", None)
    if shards:
        extra["shards"] = tuple(int(s) for s in shards)
    if target_format == 3:
        # v3 splits the array-to-bytes step out as a serializer (endianness
        # lives there). v2 has no such concept, and `order` is metadata there
        # rather than a runtime config — passing it to a v3 array only warns.
        extra["serializer"] = source.serializer
        # The chunk KEY layout is not the chunk grid, and "only chunk shapes
        # change" has to cover it too. v3 carries it as a chunk_key_encoding;
        # v2 spells the same thing `dimension_separator`, and a source written
        # with the nested `/` layout (chosen for exactly the per-directory
        # pressure this pass relieves) must not silently come out flat.
        cke = getattr(source.metadata, "chunk_key_encoding", None)
        if cke is not None:
            extra["chunk_key_encoding"] = cke
    else:
        extra["order"] = source.order
        separator = getattr(source.metadata, "dimension_separator", None)
        if separator is not None:
            from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

            extra["chunk_key_encoding"] = V2ChunkKeyEncoding(separator=separator)
    dimension_names = getattr(source.metadata, "dimension_names", None)
    if dimension_names:
        extra["dimension_names"] = tuple(dimension_names)

    dest = create_array(
        dest_group,
        name,
        shape=tuple(int(s) for s in source.shape),
        dtype=source.dtype,
        chunks=chunks,
        compressor=source.compressors,
        filters=list(source.filters),
        **extra,
    )

    shape = tuple(int(s) for s in source.shape)
    if not shape:
        dest[...] = source[...]
        return dest
    if shape[0] == 0:
        return dest

    # Slabs are aligned to the WRITE unit — the shard when there is one, the
    # chunk otherwise — so no slab boundary lands mid-object and forces a
    # read-modify-write.
    rows = max(1, int(shards[0]) if shards else int(chunks[0]))
    row_bytes = max(1, int(math.prod(shape[1:])) * int(np.dtype(source.dtype).itemsize))
    slab_rows = rows * max(1, _SLAB_BYTES // max(1, rows * row_bytes))
    for start in range(0, shape[0], slab_rows):
        stop = min(shape[0], start + slab_rows)
        dest[start:stop] = source[start:stop]
    return dest


def _copy_group(
    source: zarr.Group,
    dest: zarr.Group,
    plans: dict[str, ArrayPlan],
    target_format: int,
    path: str = "",
) -> None:
    """Mirror one group — its attrs, its arrays, and its subgroups — into ``dest``."""
    dest.attrs.update(dict(source.attrs))
    for name in sorted(source.array_keys()):
        child_path = f"{path}/{name}" if path else name
        plan = plans[child_path]
        _copy_array(source[name], dest, name, plan.target_chunks, target_format)
    for name in sorted(source.group_keys()):
        child_path = f"{path}/{name}" if path else name
        _copy_group(
            source[name], dest.create_group(name), plans, target_format, child_path
        )


def _hash_array_streaming(hasher: Any, dataset: zarr.Array) -> None:
    """Feed one array's bytes to ``hasher`` in bounded-memory row slabs.

    Byte-for-byte what ``hasher.update(dataset[:].tobytes())`` feeds it, without
    materialising the array. ``ndarray.tobytes()`` is C-order by default whatever
    the array's memory order, so the concatenation of the row slabs' bytes is
    the whole array's bytes, and an xxhash update is order-preserving over a
    concatenation. Pinned by
    ``test_optimise.py::test_the_streaming_hash_is_byte_identical``.
    """
    shape = tuple(int(s) for s in dataset.shape)
    if not shape:
        hasher.update(np.asarray(dataset[...]).tobytes())
        return
    if shape[0] == 0:
        return
    row_bytes = max(
        1, int(math.prod(shape[1:])) * int(np.dtype(dataset.dtype).itemsize)
    )
    slab_rows = max(1, _SLAB_BYTES // row_bytes)
    for start in range(0, shape[0], slab_rows):
        stop = min(shape[0], start + slab_rows)
        hasher.update(np.asarray(dataset[start:stop]).tobytes())


def _compute_content_hashes_streaming(root: zarr.Group) -> str:
    """:func:`~luxar.io._compiler.finalize.hashing.compute_content_hashes`, but
    slab-wise.

    The digest is identical — same post-order walk, same per-node xxhash64 over
    (sorted array bytes, sorted attrs JSON, sorted child hashes) — and so is the
    stamped ``content_hash`` on every node. Only the peak memory differs: the
    finalize-time version does ``dataset[:].tobytes()``, which holds the ndarray
    AND a full byte copy at once (measured: 200 MB peak for a 100 MB array;
    ~1.26 GB for the 629 MB array in the demo corpus that :data:`_SLAB_BYTES`
    exists to avoid). Re-chunking an existing store is exactly the case where
    the array is already on disk and need not be, so the walk is reimplemented
    here rather than the shared finalize helper being changed under its other
    caller.
    """

    def hash_group(group: zarr.Group) -> str:
        hasher = xxhash.xxh64()
        for name in sorted(group.array_keys()):
            _hash_array_streaming(hasher, group[name])
        attrs = {k: v for k, v in dict(group.attrs).items() if k != "content_hash"}
        hasher.update(json.dumps(attrs, sort_keys=True, default=str).encode())
        for name in sorted(group.group_keys()):
            hasher.update(hash_group(group[name]).encode())
        content_hash = hasher.hexdigest()
        group.attrs["content_hash"] = content_hash
        return content_hash

    root_hash = hash_group(root)
    aprint(f"Scene content hash: {root_hash[:16]}...")
    return root_hash


def _restamp_content_hash(root: zarr.Group) -> str | None:
    """Recompute the output store's content hash, the right way for its kind.

    A compiled scene gets the full value-hashing walk it was built with (in the
    bounded-memory form above). A standalone ``.gsplats.zarr`` gets its own
    metadata-only root stamp — the value walk would additionally stamp per-group
    hashes that format never carries. A store with neither marker is left alone:
    it is not ours to annotate.

    Runs for ``--generic`` too. ``--generic`` describes the INPUT ("I know this
    is not a Luxar store, re-chunk it anyway"), and gating the restamp on it
    disabled the anti-stale-cache guard on the one input where it matters — a
    Luxar scene passed with ``--generic`` came out carrying the SOURCE's
    ``content_hash`` while its chunk keys addressed different rows, which the
    viewer's validation queue answers ``mode: 'content-hash'`` for and never
    falls back to a byte digest on. The marker test below is the real gate: a
    foreign store carries neither marker and is still left untouched.
    """
    attrs = dict(root.attrs)
    if attrs.get("type") == "scene":
        return _compute_content_hashes_streaming(root)
    if "content_hash" in attrs:
        from ..gsplats.io.save_gsplats import _stamp_content_hash

        return _stamp_content_hash(root)
    return None


def _verify_values(path: str, src: zarr.Array, dst: zarr.Array) -> None:
    """Compare one array's BYTES, slab by slab.

    Bytes rather than ``np.array_equal``: the contract is bit-identity, and a
    payload-preserving NaN or a negative zero compares equal under numeric
    equality while differing on disk.
    """
    shape = tuple(int(s) for s in src.shape)
    if not shape:
        if np.asarray(src[...]).tobytes() != np.asarray(dst[...]).tobytes():
            raise ValueError(f"verify: {path!r} differs")
        return
    if shape[0] == 0:
        return
    row_bytes = max(1, int(math.prod(shape[1:])) * int(np.dtype(src.dtype).itemsize))
    slab_rows = max(1, _SLAB_BYTES // row_bytes)
    for start in range(0, shape[0], slab_rows):
        stop = min(shape[0], start + slab_rows)
        a = np.ascontiguousarray(src[start:stop])
        b = np.ascontiguousarray(dst[start:stop])
        if a.tobytes() != b.tobytes():
            raise ValueError(f"verify: {path!r} differs at rows {start}:{stop}")


def _verify(source: zarr.Group, dest: zarr.Group) -> int:
    """Re-read the output and compare it to the source. Returns the array count."""
    dest_arrays = dict(_walk_arrays(dest))
    checked = 0
    for path, src in _walk_arrays(source):
        dst = dest_arrays.get(path)
        if dst is None:
            raise ValueError(f"verify: {path!r} is missing from the output store")
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(
                f"verify: {path!r} shape {tuple(dst.shape)} != {tuple(src.shape)}"
            )
        if np.dtype(src.dtype) != np.dtype(dst.dtype):
            raise ValueError(f"verify: {path!r} dtype {dst.dtype} != {src.dtype}")
        _verify_values(path, src, dst)
        if dict(src.attrs) != dict(dst.attrs):
            raise ValueError(f"verify: {path!r} attrs differ")
        checked += 1
    return checked


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def resolve_target_bytes(
    *,
    target_bytes: int | None = None,
    target_kb: int | None = None,
    profile: str | None = None,
) -> int:
    """Turn the three mutually exclusive size flags into one byte budget."""
    given = [x for x in (target_bytes, target_kb, profile) if x is not None]
    if len(given) > 1:
        raise ValueError(
            "--target-bytes, --target-kb and --profile are alternatives; pass one"
        )
    if profile is not None:
        if profile not in CHUNK_PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; expected one of {sorted(CHUNK_PROFILES)}"
            )
        return CHUNK_PROFILES[profile]
    if target_kb is not None:
        if target_kb <= 0:
            raise ValueError(f"--target-kb must be positive, got {target_kb}")
        return int(target_kb) * 1024
    if target_bytes is not None:
        if target_bytes <= 0:
            raise ValueError(f"--target-bytes must be positive, got {target_bytes}")
        return int(target_bytes)
    return TARGET_CHUNK_BYTES


def _is_luxar_store(root: zarr.Group) -> bool:
    """Does this root look like a Luxar scene or a ``.gsplats.zarr`` tree?"""
    attrs = dict(root.attrs)
    return bool(
        attrs.get("type") == "scene"
        or "format_type" in attrs
        or "kind" in attrs
        or "content_hash" in attrs
    )


def optimise_store(
    source_path: str | Path,
    dest_path: str | Path,
    *,
    target_bytes: int = TARGET_CHUNK_BYTES,
    profile: str | None = None,
    overwrite: bool = False,
    verify: bool = False,
    generic: bool = False,
) -> OptimisePlan:
    """Copy ``source_path`` to ``dest_path``, re-chunked, values untouched.

    Raises rather than writing in place: the pass reads the source while writing
    the destination, so a destination that IS the source — or contains it, or
    lives inside it — is refused (:func:`_check_destination_path`).
    ``overwrite`` only governs replacing a DIFFERENT existing output, and only
    one that is a zarr store or an empty directory; it is opt-in because
    "publish under a new URL prefix" is the real fix for a warm client cache and
    cannot be enforced from here.

    All-or-nothing. The output is built in a hidden sibling directory and
    renamed onto ``dest_path`` only after the copy — and ``verify``, when asked
    for — has succeeded, so a failure leaves neither a half-written store at the
    user's path nor a damaged previous one.
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    _check_destination_path(source_path, dest_path)

    # The SOURCE is opened and fully validated before the destination is
    # touched, and the write goes to a temp sibling that is moved into place
    # last. Neither is fussiness. The earlier order deleted `dest_path` first,
    # so `luxar optimise mydata/s.luxar.zarr mydata --overwrite` removed the
    # whole containing directory and only then raised FileNotFoundError, and
    # `luxar optimise plain.zarr known-good.zarr --overwrite` destroyed a good
    # store before refusing to work for want of `--generic`. `luxar export
    # --native` carries the same pre-validation for the same reason.
    source = open_group(source_path, mode="r")
    try:
        if not generic and not _is_luxar_store(source):
            raise ValueError(
                f"{source_path} does not look like a Luxar scene or a "
                f".gsplats.zarr tree; pass --generic to re-chunk an arbitrary "
                f"zarr store"
            )

        source_format = int(source.metadata.zarr_format)
        plan = plan_optimisation(source, target_bytes=target_bytes, profile=profile)
        by_path = {p.path: p for p in plan.arrays}
        _check_destination_state(dest_path, overwrite=overwrite)

        with asection(f"Re-chunking {source_path.name} -> {dest_path.name}"):
            aprint(
                f"target {target_bytes / 1024:.0f} KB"
                + (f" (profile {profile})" if profile else "")
                + f", zarr format {source_format} preserved"
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            staging = _staging_path(dest_path)
            artifact: Path | None = None
            try:
                _write_store(
                    source, staging, by_path, plan, source_format, target_bytes, profile
                )
                aprint(
                    f"✓ {plan.n_rechunked}/{len(plan.arrays)} arrays re-chunked; "
                    f"{plan.source_n_chunks} → {plan.target_n_chunks} chunks"
                )
                artifact = _package(staging, dest_path)
                if verify:
                    reread = open_group(artifact, mode="r")
                    try:
                        checked = _verify(source, reread)
                    finally:
                        close(reread)
                    aprint(f"✓ Verified {checked} arrays byte-for-byte")
                _replace(artifact, dest_path)
                artifact = None
            finally:
                # The artifact only survives here on a failure path; on success
                # it has already been renamed onto the destination.
                shutil.rmtree(staging, ignore_errors=True)
                if artifact is not None and artifact != staging:
                    artifact.unlink(missing_ok=True)
    finally:
        close(source)
    return plan


# --------------------------------------------------------------------------
# Destination safety and staging
# --------------------------------------------------------------------------

#: Metadata documents whose presence means "there is a zarr node here" — v2's
#: pair and v3's single document. Named here rather than reached for by literal
#: at the call site; see the CLAUDE.md note on format-2 document names.
_ROOT_DOCS = frozenset({"zarr.json", ".zgroup", ".zarray"})


def _check_destination_path(source_path: Path, dest_path: Path) -> None:
    """Refuse a destination that is, contains, or lives inside the source.

    Equality alone is not enough. ``optimise s/scene.luxar.zarr s`` passes an
    equality test while naming the source's own parent, and ``--overwrite`` then
    deletes the source (plus whatever else shares that directory) before the
    source is ever read. Both containment directions are refused: a destination
    INSIDE the source would be copied into itself.
    """
    src = source_path.resolve()
    dst = dest_path.resolve()
    if src == dst:
        raise ValueError(
            "optimise cannot rewrite a store in place; give a different output "
            "path (and prefer a NEW URL prefix when republishing, so warm "
            "client caches cannot serve chunks under keys that moved)"
        )
    if dst in src.parents:
        raise ValueError(
            f"optimise refuses to write to {dest_path}: it CONTAINS the source "
            f"{source_path}, so replacing it would delete the input (and "
            f"everything else in that directory). Give an output path outside "
            f"the source's directory."
        )
    if src in dst.parents:
        raise ValueError(
            f"optimise refuses to write to {dest_path}: it is inside the source "
            f"store {source_path}, which would copy the store into itself."
        )


def _looks_like_a_zarr_store(path: Path) -> bool:
    """Is ``path`` an existing zarr store — a directory or a zipped one?

    Both formats are recognised (see :data:`_ROOT_DOCS`). A zipped store is
    identified by its members rather than its name, so an unrelated ``.zip``
    that happens to sit at the output path is not mistaken for one.
    """
    if path.is_dir():
        return any((path / doc).exists() for doc in _ROOT_DOCS)
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return any(
                name.rsplit("/", 1)[-1] in _ROOT_DOCS for name in archive.namelist()
            )
    except (OSError, zipfile.BadZipFile):
        return False


def _check_destination_state(dest_path: Path, *, overwrite: bool) -> None:
    """Decide whether ``dest_path`` may be replaced — WITHOUT deleting anything.

    ``--overwrite`` means "replace an existing OUTPUT store", not "delete
    whatever is at this path". A destination that exists and is neither a zarr
    store nor an empty directory is refused, so a mistyped path costs an error
    rather than someone's data.
    """
    if not dest_path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"{dest_path} already exists; pass --overwrite to replace it"
        )
    if dest_path.is_dir() and not any(dest_path.iterdir()):
        return
    if not _looks_like_a_zarr_store(dest_path):
        raise ValueError(
            f"--overwrite refuses to replace {dest_path}: it exists and is "
            f"neither a zarr store nor an empty directory. Pick an output path "
            f"that does not already hold something else."
        )


def _staging_path(dest_path: Path) -> Path:
    """A hidden sibling DIRECTORY to build the output in before moving it.

    Always a directory store, even when the destination is a ``.zarr.zip``. A
    ``ZipStore`` appends rather than replaces, and zarr re-serializes a group
    document on every attr write and child creation — measured on a 3-node
    scene: 211 members for 50 unique names, 16% dead payload, 50
    ``UserWarning: Duplicate name:`` (a hard failure under ``-W error``), and a
    first-match unzipper reading the pre-attrs stub of every group, i.e. the
    "loads as an empty scene" failure. ``save_gsplats`` writes a directory and
    compresses it for the same reason.
    """
    return dest_path.parent / (
        f".{dest_path.name}.optimise-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )


def _write_store(
    source: zarr.Group,
    staging: Path,
    by_path: dict[str, ArrayPlan],
    plan: OptimisePlan,
    source_format: int,
    target_bytes: int,
    profile: str | None,
) -> None:
    """Build the whole re-chunked store at ``staging``."""
    # `zarr_format` is passed explicitly so the output keeps the SOURCE's
    # format rather than whatever this process writes by default; format
    # conversion belongs to `gsplat migrate-format`, not here.
    dest = open_group(staging, mode="w", zarr_format=source_format)
    try:
        _copy_group(source, dest, by_path, source_format)
        # Written BEFORE the hash is recomputed, so it is folded into it — the
        # attr is what actually MOVES the hash (see the module docstring).
        dest.attrs["chunk_layout"] = {
            "tool": "luxar optimise",
            "target_bytes": int(target_bytes),
            "profile": profile,
            "arrays_total": len(plan.arrays),
            "arrays_rechunked": plan.n_rechunked,
            "chunks_before": plan.source_n_chunks,
            "chunks_after": plan.target_n_chunks,
        }
        _restamp_content_hash(dest)
        consolidate(dest)
    finally:
        close(dest)


def _package(staging: Path, dest_path: Path) -> Path:
    """The artifact that will BECOME the output, still beside the destination.

    The staging directory itself for a directory destination; for a ``.zip`` one
    a hidden sibling archive whose members are keyed STORE-RELATIVE, which is
    what a :class:`zarr.storage.ZipStore` reads (stored, not deflated — the
    chunks already carry their own codec). Either way ``--verify`` runs against
    this, and only a passing artifact is renamed into place.
    """
    if dest_path.suffix.lower() != ".zip":
        return staging
    archive = Path(f"{staging}.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        for member in sorted(staging.rglob("*")):
            if member.is_file():
                zf.write(member, member.relative_to(staging).as_posix())
    return archive


def _replace(new: Path, dest_path: Path) -> None:
    """Put ``new`` at ``dest_path``, removing whatever is already there."""
    if dest_path.is_dir():
        shutil.rmtree(dest_path)
    elif dest_path.exists():
        dest_path.unlink()
    os.replace(str(new), str(dest_path))
