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
compressor, filters, serializer, ``fill_value``, memory ``order``, the on-disk
zarr FORMAT (a v2 store stays v2 — format conversion is ``gsplat
migrate-format``'s job, not this one), and every group and array attribute
EXCEPT the two the pass is contractually required to move: the root's
``content_hash``, which is restamped, and the ``chunk_layout`` summary written
beside it. Those two are the cache-invalidation guard described below, and
dropping the restamp ships a re-chunked store under the source's hash. The plain
non-zarr files a group's attrs name — an overlay image, which no array or group
API reaches — are copied across byte-for-byte with everything else, or the pass
refuses rather than dropping them (:func:`_copy_payload_files` states which
names it will not write and why). Codecs are
reused from the SOURCE array rather than re-derived, because an omitted
compressor is not "no compressor" (zarr's ``"auto"`` is Blosc/lz4 at format 2
and zstd at format 3) and some Luxar arrays are deliberately RAW.

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
The viewer's ``MultiLevelCachingStore`` validates its persistent cache by
comparing ``content_hash`` against the remote, and that cache holds ENCODED
CHUNKS KEYED BY CHUNK INDEX. A re-chunk that leaves the hash where it was is
therefore the worst thing this pass could do: chunk key ``0/0`` covers a
different row range while a warm client believes itself up to date, so it serves
bytes that no longer mean what their keys say. Silent wrong data, on the one path
with no error to raise.

Both hashers now fold layout in themselves.
:func:`~luxar.io._compiler.finalize.hashing.compute_content_hashes` hashes each
array's STORAGE IDENTITY — name, shape, dtype, chunks, shards, codec ids and the
array's own attrs — before its values, and the ``.gsplats.zarr`` stamp
(``save_gsplats._stamp_content_hash``) folds the same identity terms over
metadata alone; both key each child group by its NAME. Only the value walk also
folds the bytes of a group's plain payload files — the gsplat stamp has no
payload handling at all. So recomputing over the re-chunked output
lands on a different digest by construction. What makes that reach the viewer is
the RESTAMP: nothing else rewrites the stored ``content_hash``, and the stored
one is what gets compared. Suppress it and the output ships under the source's
hash however far the grid moved.

The ``chunk_layout`` summary attr the pass writes on the root is folded in too,
since attrs are hashed. It used to be the ONLY thing moving the digest — back
then, suppressing it was measured to leave the output hash byte-identical to the
source's — and against a layout-aware hasher it is now belt-and-braces there. It
is kept because it documents what the pass did, and because it is still the whole
guard in the hash-less case below.

The attr is written for a NON-Luxar store too, and that is not tidiness either.
A store with a ``kind`` marker but no ``content_hash`` gets no restamp, so the
viewer's ``MultiLevelCachingStore`` validation queue falls back to a SHA-256 of
the raw root document bytes. At format 3 the root ``zarr.json`` carries the
chunk grid, so that token moves whatever we write; at format 2 the probe reads
``.zattrs``, which — without ``chunk_layout`` — would be byte-identical to the
source's, and the warm cache would again believe itself current. "Suppress the
attr on non-Luxar stores" is therefore a plausible-sounding simplification that
silently breaks exactly the hash-less format-2 case.
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

from .._zarr_compat import (
    array_keys,
    close,
    consolidate,
    create_array,
    group_keys,
    list_raw_keys,
    open_group,
    read_raw_bytes,
    write_raw_bytes,
)
from ..typing_utils._format_contract import FORMAT_TYPE_GSPLATS, NODE_TYPES
from ..typing_utils.constants import (
    ENVIRONMENT_GROUP,
    MAX_CHUNK_BYTES,
    MIN_CHUNK_BYTES,
    TARGET_CHUNK_BYTES,
)
from ._compiler.chunking import _atom_aligned_rows
from ._compiler.finalize.hashing import (
    _ZARR_METADATA_DOCS_LOWERCASED,
    PAYLOAD_FILE_ATTRS,
    _is_safe_payload_name,
    _payload_terms,
    _storage_identity,
)

__all__ = [
    "CHUNK_PROFILES",
    "ArrayPlan",
    "ChunkLayoutSummary",
    "OptimisePlan",
    "PlaybackChunkWarning",
    "optimise_store",
    "plan_optimisation",
    "resolve_target_bytes",
    "summarise_chunk_layout",
    "summarise_plan",
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
#:
#: On an ANIMATED node (a non-displayed axis the user plays) the useful measure
#: is frames-per-chunk: group the ordered ``chunk_bounds`` atoms exactly as the
#: planned zarr chunks will group them, then measure each group's inclusive
#: hidden-axis span. The writer already orders slice-major, so a 1 MB
#: ``archive`` chunk of a 250-frame Lines node holds ~6 frames. The viewer now
#: prefetches the nearest next zarr chunk
#: boundary across a node's arrays while those preceding frames play (#2686);
#: before that, every boundary produced a measured 0.4-0.9 s stall on a 7 Mbps
#: link.
#: A LADDERED played node is the opposite case: a coarse rung that fits one
#: chunk stays cache-resident and serves every timepoint (#2377), which is why
#: the 4D splat demos ship at 1 MB. The per-array planner therefore stays byte
#: based, while the whole-store plan warns when an actively played, un-laddered
#: node would exceed two frames per chunk across multiple chunks.
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

#: More than two played frames per chunk creates the bursty boundary-crossing
#: shape measured in #2686. Two or fewer retains at least one frame of lead from
#: the viewer's boundary-aware prefetch and is not worth warning about.
_PLAYBACK_WARNING_FRAMES = 2.0

_SPATIALLY_INDEXED_TYPES = frozenset({"points", "lines", "gsplats"})


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrayPlan:
    """What will happen to one array, and why."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    #: ``dtype.itemsize``, carried rather than re-derived from :attr:`dtype`.
    #: ``str(np.dtype(...))`` is NOT a round trip for every dtype — a structured
    #: dtype stringifies to ``"[('a', '<i4'), ('b', '<f8')]"`` and numpy's
    #: variable-width ``StringDType`` to ``"StringDType()"``, neither of which
    #: ``np.dtype()`` accepts — so re-parsing it to reach the itemsize raised
    #: ``TypeError`` on exactly the third-party stores (AnnData/cellxgene, an
    #: OME-Zarr label table) ``--generic`` exists for: ``luxar info --stats``
    #: exited 1 on a store its own ``--format json`` path handled, and
    #: ``optimise --dry-run`` died with a traceback while the real copy of the
    #: same store succeeded.
    itemsize: int
    source_chunks: tuple[int, ...]
    target_chunks: tuple[int, ...]
    source_n_chunks: int
    target_n_chunks: int
    atom: int | None
    #: The grid whose cells are FILES — the shard grid when the array is
    #: sharded, its chunk grid otherwise. Recorded rather than re-derived so
    #: :func:`summarise_plan` can measure a store off the plan's single walk.
    source_file_grid: tuple[int, ...]
    #: Empty when the array is being re-chunked; otherwise the reason it is not.
    skip_reason: str = ""

    @property
    def rechunked(self) -> bool:
        """True when this array gets a new chunk grid."""
        return not self.skip_reason

    @property
    def target_chunk_bytes(self) -> int:
        """Nominal payload of the planned chunk, in bytes."""
        return _chunk_bytes(self.target_chunks, self.itemsize)

    @property
    def source_file_bytes(self) -> int:
        """Nominal payload of one SOURCE object (a shard when sharded)."""
        return _chunk_bytes(self.source_file_grid, self.itemsize)


@dataclass(frozen=True)
class OptimisePlan:
    """The whole-store plan — what :func:`optimise_store` will do, in advance."""

    target_bytes: int
    profile: str | None
    arrays: list[ArrayPlan] = field(default_factory=list)
    playback_warnings: list[PlaybackChunkWarning] = field(default_factory=list)

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
class PlaybackChunkWarning:
    """An animated flat node whose planned chunks span several frames."""

    node_path: str
    array_path: str
    dimension_name: str
    frames_per_chunk: float
    n_chunks: int

    @property
    def message(self) -> str:
        """Human-facing warning shared by dry-run and real-copy narration."""
        return (
            f"{self.node_path}: {self.array_path} would span up to "
            f"{self.frames_per_chunk:.1f} {self.dimension_name} frames/chunk "
            f"across {self.n_chunks} chunks while playback starts enabled; "
            "consider a smaller target if playback stalls"
        )


@dataclass(frozen=True)
class ChunkLayoutSummary:
    """The streaming-shape diagnostic ``luxar info --stats`` reports.

    Computed off the same walk the optimiser plans from, so "is this store worth
    optimising?" is answerable without hosting it first.
    """

    #: Arrays that produce at least one object. A ``(0, D)`` ``array_ref``
    #: placeholder writes no chunk and costs no request, so counting it would
    #: put arrays nobody fetches in the denominator of the floor share below —
    #: measured on a 6-node scene with deduplicated positions, that reported
    #: "23/24 arrays (96%)" where the honest answer is 13/14 (93%).
    n_arrays: int
    n_chunks: int
    #: Chunk-count-weighted mean of the NOMINAL chunk payload, in bytes.
    mean_chunk_bytes: float
    #: Arrays whose nominal chunk payload is below
    #: :data:`~luxar.typing_utils.constants.MIN_CHUNK_BYTES`. Single-chunk arrays
    #: count: a store of a hundred tiny arrays is request-heavy for exactly the
    #: reason a store of tiny chunks is. Arrays that fetch nothing do not.
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
    for name in sorted(array_keys(group)):
        yield (f"{path}/{name}" if path else name), group[name]
    for name in sorted(group_keys(group)):
        child_path = f"{path}/{name}" if path else name
        yield from _walk_arrays(group[name], child_path)


def _chunk_bytes(chunks: tuple[int, ...], itemsize: int) -> int:
    """The NOMINAL payload of one chunk, uncompressed — the target's units."""
    return int(math.prod(chunks)) * int(itemsize)


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

    A bounds array is on no element grid at all — it is indexed by CHUNK — so it
    gets no atom. Without that first test it fell through to the per-vertex
    branch, and ``ArrayPlan.atom`` for ``segment_chunk_bounds`` reported the
    node's VERTEX atom (3276 where the segment grid is 4096). Harmless only
    because :func:`_structural_skip` refuses to re-chunk it before the atom is
    used, which is not a property worth depending on for a public field.
    """
    if array_name in _INDEX_ARRAYS:
        return None, None
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

    Each reason is user-visible (``--dry-run`` groups its "Left alone" tally by
    it), so it must not claim more than it knows. Two did. ``shape[0] == 1`` was
    reported as "broadcast", which is a statement about a Luxar encoding rather
    than a shape — an OME-Zarr level array ``(1, 1, Z, Y, X)`` is not one; and
    the last branch was "single chunk", which is only true for an array chunked
    on axis 0 alone. This pass merges rows and nothing else, so what both
    branches actually establish is that axis 0 is already one chunk. The
    ``shape[0] == 1`` test is dropped rather than relabelled: it is subsumed by
    the axis-0 test below, since ``chunks[0] >= 1`` always.
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
    if encoding == "broadcasted":
        return "broadcast"
    if chunks[0] >= shape[0]:
        return "rows already in one chunk"
    return ""


def _plan_array(
    path: str,
    array: zarr.Array,
    atom: int | None,
    target_bytes: int,
    *,
    is_environment_map: bool,
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
            itemsize=int(dtype.itemsize),
            source_chunks=chunks,
            target_chunks=chunks,
            source_n_chunks=n,
            target_n_chunks=n,
            atom=atom,
            source_file_grid=file_grid,
            skip_reason=reason,
        )

    # A baked environment map (`environment/faces-<digest>`) is one chunk per
    # cube face by construction and the viewer reads it whole; it is not spatial
    # data and nothing about streaming applies. Copied verbatim.
    if is_environment_map:
        return keep("baked environment map")

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
        itemsize=int(dtype.itemsize),
        source_chunks=chunks,
        target_chunks=target_chunks,
        source_n_chunks=_n_chunks(shape, chunks),
        target_n_chunks=_n_chunks(shape, target_chunks),
        atom=atom,
        source_file_grid=file_grid,
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
        names = frozenset(array_keys(group))
        for name in sorted(names):
            array = group[name]
            path = f"{group_path}/{name}" if group_path else name
            atom = _resolve_atom(attrs, names, name)
            is_environment_map = (
                group_path == ENVIRONMENT_GROUP and attrs.get("faces") == name
            )
            plans.append(
                _plan_array(
                    path,
                    array,
                    atom,
                    target_bytes,
                    is_environment_map=is_environment_map,
                )
            )
    by_path = {plan.path: plan for plan in plans}
    warnings = _plan_playback_warnings(root, by_path)
    return OptimisePlan(
        target_bytes=target_bytes,
        profile=profile,
        arrays=plans,
        playback_warnings=warnings,
    )


def _playing_dimensions(root: zarr.Group) -> list[tuple[int, str, float]]:
    """Played dimensions as ``(index, name, positive tick step)`` tuples.

    Auto playback on a continuous dimension has no fixed tick step and is
    therefore not diagnosable unless the animation supplies ``step_size``.
    """
    attrs = dict(root.attrs)
    viewer_config = attrs.get("viewer_config")
    scene_dimensions = attrs.get("scene_dimensions")
    if not isinstance(viewer_config, dict) or not isinstance(scene_dimensions, dict):
        return []
    animations = viewer_config.get("animation")
    dimensions = scene_dimensions.get("dimensions")
    if not isinstance(animations, list) or not isinstance(dimensions, list):
        return []
    played: list[tuple[int, str, float]] = []
    for index, animation in enumerate(animations):
        if not isinstance(animation, dict) or animation.get("playing") is not True:
            continue
        if index >= len(dimensions) or not isinstance(dimensions[index], dict):
            continue
        dimension = dimensions[index]
        base_step = _positive_float(dimension.get("step"))
        numeric_step = _positive_float(animation.get("step_size"))
        if numeric_step is not None and dimension.get("discrete") is True:
            grid_step = base_step or 1.0
            cells = math.floor(numeric_step / grid_step + 0.5)
            numeric_step = max(grid_step, cells * grid_step)
        elif numeric_step is None:
            numeric_step = base_step
        if numeric_step is None:
            continue
        name = dimension.get("name")
        played.append(
            (index, name if isinstance(name, str) else str(index), numeric_step)
        )
    return played


def _positive_float(raw: Any) -> float | None:
    """A finite positive float, or ``None`` when ``raw`` is not one."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _is_ladder(attrs: dict[str, Any]) -> bool:
    """Whether this group starts either supported LOD ladder shape."""
    if attrs.get("kind") == "lod":
        return True
    try:
        return int(attrs.get("n_additive_sublods", 1)) > 1
    except (TypeError, ValueError):
        return False


def _ordering_for_array(
    attrs: dict[str, Any], array_name: str, dimension: int
) -> tuple[str, tuple[int, ...]] | None:
    """Bounds array and bounds dimensions for one played scene dimension."""
    if array_name == _SEGMENT_ARRAY:
        ordering = attrs.get("segment_ordering")
        raw_ndim = attrs.get("ndim")
        if not isinstance(raw_ndim, (int, str)):
            return None
        try:
            ndim = int(raw_ndim)
        except (TypeError, ValueError):
            return None
        # ``slice_dims`` addresses the 2×D segment endpoint coordinates, while
        # ``segment_chunk_bounds`` stores one D-wide union bound per atom.
        bounds_dims: tuple[int, ...] = (dimension, dimension + ndim)
        bounds_name = "segment_chunk_bounds"
    else:
        ordering = attrs.get("vertex_ordering", attrs)
        bounds_dims = (dimension,)
        bounds_name = (
            "vertex_chunk_bounds"
            if isinstance(attrs.get("vertex_ordering"), dict)
            else "chunk_bounds"
        )
    if not isinstance(ordering, dict):
        return None
    slice_dims = ordering.get("slice_dims")
    if not isinstance(slice_dims, list) or not any(
        dim in slice_dims for dim in bounds_dims
    ):
        return None
    return bounds_name, bounds_dims


def _chunk_frame_span(
    group: zarr.Group,
    bounds_name: str,
    bounds_dims: tuple[int, ...],
    step: float,
    atoms_per_chunk: int,
) -> float | None:
    """Largest inclusive frame span of one planned first-axis chunk."""
    if atoms_per_chunk < 1:
        return None
    if bounds_name not in frozenset(array_keys(group)):
        return None
    bounds = group[bounds_name]
    if len(bounds.shape) != 3 or bounds.shape[2] != 2:
        return None
    valid_dims = tuple(dim for dim in bounds_dims if 0 <= dim < bounds.shape[1])
    if not valid_dims:
        return None
    values = np.asarray(bounds[...], dtype=np.float64)[:, valid_dims, :]
    if values.size == 0 or not np.isfinite(values).all():
        return None
    lows = values[..., 0].min(axis=1)
    highs = values[..., 1].max(axis=1)
    spans = [
        (
            highs[start : start + atoms_per_chunk].max()
            - lows[start : start + atoms_per_chunk].min()
        )
        / step
        + 1.0
        for start in range(0, len(lows), atoms_per_chunk)
    ]
    return float(max(spans)) if spans else None


def _group_playback_warning(
    group_path: str,
    group: zarr.Group,
    plans: dict[str, ArrayPlan],
    played: list[tuple[int, str, float]],
) -> PlaybackChunkWarning | None:
    """Worst multi-frame planned array for one flat indexed node."""
    attrs = dict(group.attrs)
    if attrs.get("type") not in _SPATIALLY_INDEXED_TYPES:
        return None
    candidates: list[PlaybackChunkWarning] = []
    span_cache: dict[tuple[str, tuple[int, ...], float, int], float | None] = {}
    for dimension, dimension_name, step in played:
        for array_name in sorted(array_keys(group)):
            path = f"{group_path}/{array_name}" if group_path else array_name
            plan = plans.get(path)
            if plan is None or not plan.shape or plan.atom is None:
                continue
            n_chunks = -(-plan.shape[0] // plan.target_chunks[0])
            if n_chunks <= 1:
                continue
            resolved = _ordering_for_array(attrs, array_name, dimension)
            if resolved is None:
                continue
            bounds_name, bounds_dims = resolved
            atoms_per_chunk = plan.target_chunks[0] // plan.atom
            key = (bounds_name, bounds_dims, step, atoms_per_chunk)
            if key not in span_cache:
                span_cache[key] = _chunk_frame_span(
                    group, bounds_name, bounds_dims, step, atoms_per_chunk
                )
            frames = span_cache[key]
            if frames is None:
                continue
            if frames <= _PLAYBACK_WARNING_FRAMES:
                continue
            candidates.append(
                PlaybackChunkWarning(
                    node_path=group_path or "/",
                    array_path=path,
                    dimension_name=dimension_name,
                    frames_per_chunk=frames,
                    n_chunks=n_chunks,
                )
            )
    return max(candidates, key=lambda warning: warning.frames_per_chunk, default=None)


def _plan_playback_warnings(
    root: zarr.Group, plans: dict[str, ArrayPlan]
) -> list[PlaybackChunkWarning]:
    """Warn once per un-laddered node or partition root for wide chunks."""
    played = _playing_dimensions(root)
    if not played:
        return []
    warnings: dict[tuple[str, str], PlaybackChunkWarning] = {}

    def visit(
        group: zarr.Group,
        path: str,
        inside_ladder: bool,
        partition_root: str | None,
    ) -> None:
        attrs = dict(group.attrs)
        laddered = inside_ladder or _is_ladder(attrs)
        if partition_root is None and attrs.get("kind") == "partition":
            partition_root = path or "/"
        if not laddered:
            warning = _group_playback_warning(path, group, plans, played)
            if warning is not None:
                key = (
                    ("partition", partition_root)
                    if partition_root is not None
                    else ("node", path)
                )
                previous = warnings.get(key)
                if (
                    previous is None
                    or warning.frames_per_chunk > previous.frames_per_chunk
                ):
                    warnings[key] = warning
        for name in sorted(group_keys(group)):
            child_path = f"{path}/{name}" if path else name
            visit(group[name], child_path, laddered, partition_root)

    visit(root, "", False, None)
    return list(warnings.values())


def _walk_groups(group: zarr.Group, path: str = "") -> Iterator[tuple[str, zarr.Group]]:
    """Yield ``(store-relative path, group)`` for the root and every subgroup."""
    yield path, group
    for name in sorted(group_keys(group)):
        yield from _walk_groups(group[name], f"{path}/{name}" if path else name)


def summarise_chunk_layout(root: zarr.Group) -> ChunkLayoutSummary:
    """Measure a store's streaming shape: chunk sizes and request count."""
    return summarise_plan(plan_optimisation(root))


def summarise_plan(plan: OptimisePlan) -> ChunkLayoutSummary:
    """The same diagnostic, off a plan that has already been walked.

    ``luxar info --stats`` wants both the summary and a real plan (the "try
    ``luxar optimise``" hint is gated on one), and the corpus this tool exists
    for holds 606,349 files — so the two share one walk rather than opening
    every array twice.

    Counts OBJECTS, not nominal grid cells: a shard is one file however many
    chunks it packs, and a ``(0, D)`` placeholder is none at all.
    """
    n_arrays = 0
    n_chunks = 0
    total_bytes = 0
    under_floor = 0
    for array_plan in plan.arrays:
        count = array_plan.source_n_chunks
        if count == 0:
            # Fetches nothing, so it is neither a request nor a badly sized
            # chunk — booking it "under the floor" only dilutes the diagnostic.
            continue
        payload = array_plan.source_file_bytes
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
    if dest_group.attrs.get("type") == "sound":
        extra["config"] = {"write_empty_chunks": True}

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
    """Mirror a group — attrs, payload files, arrays, subgroups — into ``dest``."""
    dest.attrs.update(dict(source.attrs))
    _copy_payload_files(source, dest)
    for name in sorted(array_keys(source)):
        child_path = f"{path}/{name}" if path else name
        plan = plans[child_path]
        _copy_array(source[name], dest, name, plan.target_chunks, target_format)
    for name in sorted(group_keys(source)):
        child_path = f"{path}/{name}" if path else name
        _copy_group(
            source[name], dest.create_group(name), plans, target_format, child_path
        )


def _read_payload_or_refuse(
    group: zarr.Group, attr_key: str, filename: str
) -> bytes | None:
    """The payload's bytes (``None`` when the source holds nothing there).

    Where this diverges from the hasher is the READ OUTCOME, not the name gate.
    :func:`_payload_terms` degrades a payload it cannot read to a deterministic
    ``unreadable:`` term, because a compile must still finish and the alternative
    there is a store stamped ``incomplete``. This refuses instead, with an error
    naming the file: shipping a re-chunked store whose overlay silently vanished
    is exactly the failure this pass exists to prevent, and refusing costs
    nothing, because the destination is still in staging and
    :func:`optimise_store` removes it on any raise. So the same store the hasher
    completes over can stop this pass — deliberately.

    Shared by the copy and by ``--verify`` so that a source read failing on the
    verify pass (an NFS ``ESTALE``, a concurrent ``chmod`` during a long run)
    produces the same worded refusal rather than a bare ``OSError`` from the
    layer underneath.
    """
    try:
        return read_raw_bytes(group, filename)
    except (OSError, ValueError) as unreadable:
        raise ValueError(
            f"optimise cannot read the payload file {filename!r} named by "
            f"{attr_key!r} on group {group.path or '/'!r}: {unreadable}. "
            f"The re-chunk is refused rather than shipping a store whose "
            f"overlay silently vanished — repair the file (permissions, or "
            f"the name it is stored under), remove it, or clear the "
            f"{attr_key!r} attr, then run optimise again."
        ) from unreadable


def _copy_payload_files(source: zarr.Group, dest: zarr.Group) -> None:
    """Copy the plain non-zarr files ``source``'s attrs name into ``dest``.

    An overlay image is written straight into its group's own directory
    (``core.scene.overlays.internals.write_overlay``), so it is neither an array
    nor a subgroup and the rest of :func:`_copy_group` is blind to it: without
    this the output store carries an ``image_file`` attr naming a file that does
    not exist, and the overlay silently disappears from a re-chunked scene. The
    attr keys are :data:`PAYLOAD_FILE_ATTRS` and the name test is the hasher's
    own :func:`_is_safe_payload_name` — so the names this WRITES are the names
    the digest reads, minus the metadata-document collision below. The copy
    materialises BYTES, so a payload that is a symlink in the source comes out as
    a real file in the output, which is what a self-contained store needs (a
    symlink would not survive the ``.zarr.zip`` packaging either).

    The hasher's name gate remains exact: a case-shifted collision reaches a
    listing probe, and a genuinely held key contributes its bytes. The copy is
    stricter because it must be portable: a name colliding case-insensitively
    with a zarr metadata document (:data:`_ZARR_METADATA_DOCS_LOWERCASED`) is
    decided by whether the SOURCE really holds a key spelled EXACTLY that, not
    by the name alone. On a case-SENSITIVE filesystem a file called
    ``Zarr.json`` is an ordinary distinct file: skipping it would produce
    exactly the state this function exists to prevent — an
    ``image_file`` attr naming a file the output does not hold — and do it under
    an exit code of 0. So a name that really is there REFUSES the re-chunk,
    because the copy cannot write it faithfully: on the macOS or Windows machine
    the output may be read on, that key IS the group's own metadata document. The
    way out is to rename the payload file and the attr that names it. A DANGLING
    attr of that shape is skipped with a notice, since there is nothing to be
    unfaithful to.

    Which of the two it is comes from :func:`list_raw_keys` — a directory listing
    compared case-sensitively in Python — and not from trying to read the key.
    An open-by-name is resolved by the OS, so on the very filesystems this branch
    exists for, a probe for ``Zarr.json`` hands back the group's own
    ``zarr.json``: the dangling case would be invisible, every such store would
    be refused, and the refusal would quote the metadata document's byte count as
    if it were the payload's. The listing therefore GATES the read rather than
    replacing it — the bytes are still read, for the count the refusal quotes,
    but only for a name the store really lists, so a fold can no longer make a
    dangling attr look held. A key that lists and yet reads back nothing is not a
    payload at all (a child group or a plain subdirectory of that name), and
    takes the skip too: there are no bytes to be unfaithful to and no file to
    rename.

    A named file that is simply ABSENT is skipped with a notice rather than
    raising, for the same reason: the source has no bytes to hand over, so
    refusing the whole re-chunk would only strand a store that is already in that
    state, and the output is exactly as complete as its input.
    """
    attrs = dict(source.attrs)
    for attr_key in sorted(PAYLOAD_FILE_ATTRS):
        filename = attrs.get(attr_key)
        if not isinstance(filename, str) or not filename:
            continue
        if not _is_safe_payload_name(filename):
            # Two classes, and the notice names both: a name that is not a
            # single path component, and one that IS zarr's own metadata
            # document spelled exactly (``.zattrs``), which the hasher's gate
            # rejects too. Only a CASE-shifted collision reaches the branch
            # below, which refuses when the source really holds that key.
            aprint(
                f"⚠ skipping payload {filename!r}: not a plain file name, or "
                f"one of zarr's own metadata documents"
            )
            continue
        if filename.lower() in _ZARR_METADATA_DOCS_LOWERCASED:
            # The listing GATES the read, and a listing is not folded: a name
            # the store does not really hold never reaches the read that would
            # resolve onto the node document, so the dangling case stays visible
            # on every platform. A key that lists but reads back `None` is not a
            # payload either — a child group or a plain subdirectory of that
            # name has no bytes to drop and nothing to rename — so it takes the
            # same skip.
            held = (
                _read_payload_or_refuse(source, attr_key, filename)
                if filename in list_raw_keys(source)
                else None
            )
            if held is None:
                aprint(
                    f"⚠ skipping payload {filename!r}: it names a zarr metadata "
                    f"document and the source holds no file there"
                )
                continue
            raise ValueError(
                f"optimise cannot copy the payload file {filename!r} named by "
                f"{attr_key!r} on group {source.path or '/'!r}: on a "
                f"case-insensitive filesystem that name resolves to the group's "
                f"own zarr metadata document, so writing it would replace the "
                f"document the whole store is read through. The re-chunk is "
                f"refused rather than dropping the {len(held)} bytes the source "
                f"really holds — rename the payload file and the {attr_key!r} "
                f"attr that names it, then run optimise again."
            )
        payload = _read_payload_or_refuse(source, attr_key, filename)
        if payload is None:
            aprint(f"⚠ payload {filename!r} named by {attr_key!r} is missing")
            continue
        write_raw_bytes(dest, filename, payload)


def _hash_array_streaming(hasher: Any, dataset: zarr.Array) -> None:
    """Feed one array's bytes to ``hasher`` in bounded-memory row slabs.

    Byte-for-byte what ``hasher.update(dataset[:].tobytes())`` feeds it, without
    materialising the array. ``ndarray.tobytes()`` is C-order by default whatever
    the array's memory order, so the concatenation of the row slabs' bytes is
    the whole array's bytes, and an xxhash update is order-preserving over a
    concatenation. Pinned by
    ``test_optimise.py::test_the_streaming_hash_is_byte_identical``.

    That holds for every FIXED-WIDTH dtype, i.e. every store the value walk can
    hash reproducibly at all. A variable-width dtype (numpy's ``StringDType``, a
    ``vlen-utf8`` v2 array, an object array) puts descriptors into the buffer
    rather than characters, so ``tobytes()`` there hashes an allocation: the
    reference itself yields a different digest on each read of the same array,
    and slabbing changes the byte stream again. Nothing is done about it here —
    no Luxar writer emits such an array (the reference could not hash one
    stably either), and ``--verify`` is unaffected because
    :func:`_slabs_differ` compares those element-wise. Noted rather than
    silently claimed away.
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

    The digest is identical wherever the reference produces one REPRODUCIBLY —
    same post-order walk, same per-node xxhash64 over the same terms in the same
    order: each array's storage identity followed by its values, the group's own
    attrs as sorted JSON, the bytes of whatever plain payload file those attrs
    name, then each child group's name and hash — and so is the stamped
    ``content_hash`` on every node. Only the VALUE step is local; the identity
    and payload terms are the reference's own helpers, imported rather than
    restated, so a future term lands in both walks at once and cannot drift.
    Two documented departures, both narrow: a
    0-d array makes the reference raise ``IndexError`` (``dataset[:]`` on a
    scalar) while the slab walk hashes it via ``dataset[...]``, and a
    variable-width dtype has no stable digest under EITHER walk (see
    :func:`_hash_array_streaming`). So "identical" is a statement about every
    store the reference can hash stably, not about every store. Pinned by
    ``test_optimise.py::TestCacheInvalidation``.

    Only the peak memory differs otherwise: the
    finalize-time version does ``dataset[:].tobytes()``, which holds the ndarray
    AND a full byte copy at once (measured: 200 MB peak for a 100 MB array;
    ~1.26 GB for the 629 MB array in the demo corpus that :data:`_SLAB_BYTES`
    exists to avoid). That bound is over ARRAY values only: a payload file is
    read WHOLE here, exactly as the reference reads it, since the store hands
    back a key's complete bytes and there is no slabbed read for one.
    Re-chunking an existing store is exactly the case where
    the array is already on disk and need not be, so the walk is reimplemented
    here rather than the shared finalize helper being changed under its other
    caller.
    """

    def hash_group(group: zarr.Group, *, is_root: bool) -> str:
        hasher = xxhash.xxh64()
        for name in sorted(array_keys(group)):
            dataset = group[name]
            identity = _storage_identity(name, dataset)
            hasher.update(json.dumps(identity, sort_keys=True, default=str).encode())
            _hash_array_streaming(hasher, dataset)
        attrs = {k: v for k, v in dict(group.attrs).items() if k != "content_hash"}
        hasher.update(json.dumps(attrs, sort_keys=True, default=str).encode())
        for term in _payload_terms(group, attrs):
            hasher.update(term)
        for name in sorted(group_keys(group)):
            child_hash = hash_group(group[name], is_root=False)
            # The root's baked-environment group is stamped but not folded in —
            # the same exception the reference walk makes, for the same reason.
            if is_root and name == ENVIRONMENT_GROUP:
                continue
            hasher.update(f"{name}:{child_hash}".encode())
        content_hash = hasher.hexdigest()
        group.attrs["content_hash"] = content_hash
        return content_hash

    root_hash = hash_group(root, is_root=True)
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


def _baked_environment_group(root: zarr.Group) -> zarr.Group | None:
    """Return the baked sidecar, never ordinary data that only shares its name."""
    if ENVIRONMENT_GROUP not in root:
        return None
    candidate = root[ENVIRONMENT_GROUP]
    if not isinstance(candidate, zarr.Group):
        return None
    faces = dict(candidate.attrs).get("faces")
    if not isinstance(faces, str) or faces not in array_keys(candidate):
        return None
    return candidate


#: dtype kinds whose memory buffer IS the stored payload, so comparing
#: ``tobytes()`` compares the data. Everything outside this set (numpy's
#: variable-width ``StringDType``, kind ``T``; an object array, kind ``O``) puts
#: POINTERS in the buffer and must be compared element-wise instead.
_BYTEWISE_KINDS = frozenset("biufcmMSUV")


def _slabs_differ(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> bool:
    """Do these two slabs differ AS STORED?

    Bytes rather than ``np.array_equal`` for every fixed-width dtype: the
    contract is bit-identity, and a payload-preserving NaN or a negative zero
    compares equal under numeric equality while differing on disk.

    A variable-width dtype is the one case where that idiom is not merely
    stricter but WRONG. ``numpy.dtypes.StringDType`` — what a zarr-v3
    ``dtype=str`` array decodes to, reachable here via ``--generic`` on an
    OME-Zarr label table or an AnnData/cellxgene store — keeps its characters in
    an arena and its buffer holds descriptors into it, so ``tobytes()`` compares
    two allocations. Measured: ``['x'*60, 'x'*60]`` and ``['x'*60, 'y'*60]``
    compare IDENTICAL. Those fall back to an exact element-wise comparison, so
    ``--verify`` stays a real check rather than a vacuous one.

    (The same idiom in :func:`_hash_array_streaming` is deliberately left alone:
    it must stay byte-for-byte what ``compute_content_hashes`` feeds its hasher.)
    """
    if a.dtype.kind in _BYTEWISE_KINDS:
        return a.tobytes() != b.tobytes()
    return not np.array_equal(a, b)


def _verify_values(path: str, src: zarr.Array, dst: zarr.Array) -> None:
    """Compare one array's stored values, slab by slab."""
    shape = tuple(int(s) for s in src.shape)
    if not shape:
        if _slabs_differ(np.asarray(src[...]), np.asarray(dst[...])):
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
        if _slabs_differ(a, b):
            raise ValueError(f"verify: {path!r} differs at rows {start}:{stop}")


def _verify_payloads(path: str, src: zarr.Group, dst: zarr.Group) -> int:
    """Compare this group's named payload files byte for byte; count them.

    Walking arrays alone leaves the one thing :func:`_copy_payload_files` moves
    outside ``--verify`` entirely, so a truncated or dropped overlay image
    passed a run that printed a byte-for-byte promise. The name gate is the copy
    site's, exactly: a name the copy skipped has no bytes in the output by
    design and must not be reported as a loss. A payload the SOURCE does not
    have is skipped for the same reason — the copy skipped it too, and the
    output is as complete as its input. A metadata-document name reaches here
    only in the two cases the copy SKIPS — a dangling attr, and a name held by
    a directory rather than a file — since the copy refuses the one with bytes.

    The count is what the run REPORTS, so it is the number actually compared —
    a name skipped by any of those gates is not one of them.
    """
    compared = 0
    attrs = dict(src.attrs)
    for attr_key in sorted(PAYLOAD_FILE_ATTRS):
        filename = attrs.get(attr_key)
        if not isinstance(filename, str) or not filename:
            continue
        if not _is_safe_payload_name(filename):
            continue
        if filename.lower() in _ZARR_METADATA_DOCS_LOWERCASED:
            continue
        expected = _read_payload_or_refuse(src, attr_key, filename)
        if expected is None:
            continue
        actual = read_raw_bytes(dst, filename)
        if actual is None:
            raise ValueError(
                f"verify: payload {filename!r} named by {attr_key!r} on {path!r} "
                f"is missing from the output store"
            )
        if actual != expected:
            raise ValueError(
                f"verify: payload {filename!r} named by {attr_key!r} on {path!r} "
                f"differs ({len(actual)} bytes, expected {len(expected)})"
            )
        compared += 1
    return compared


def _verify(source: zarr.Group, dest: zarr.Group) -> tuple[int, int]:
    """Re-read the output and compare it to the source.

    Returns ``(arrays compared, payload files compared)``. The two are counted
    separately because a payload file is attached to a GROUP rather than to an
    array, and because the run reports both: almost every store has zero payload
    files, and saying so is more honest than a blanket "and every payload file".
    """
    dest_groups = dict(_walk_groups(dest))
    payloads = 0
    for group_path, src_group in _walk_groups(source):
        dst_group = dest_groups.get(group_path)
        if dst_group is None:
            raise ValueError(
                f"verify: group {group_path or '/'!r} is missing from the output store"
            )
        payloads += _verify_payloads(group_path or "/", src_group, dst_group)
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
    return checked, payloads


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


#: The node ``kind`` values a gsplat tree stores. ``lod`` and ``partition`` are
#: written explicitly (``gsplat_tree.py``, ``save_gsplats.py``); ``leaf`` is the
#: spec's name for the third, which the writers spell as the ABSENCE of the attr
#: — accepted here because a store that names it is naming a Luxar kind.
_LUXAR_NODE_KINDS = frozenset({"leaf", "lod", "partition"})


def _is_luxar_store(root: zarr.Group) -> bool:
    """Does this root look like a Luxar scene or a ``.gsplats.zarr`` tree?

    Recognized VALUES, not merely attribute PRESENCE. ``kind`` and
    ``content_hash`` are generic enough for a foreign store to carry either, and
    testing for presence let one straight past the documented ``--generic``
    opt-in — after which its ``content_hash`` is rewritten under Luxar's hashing
    semantics and a ``chunk_layout`` attr is added, which is precisely what the
    gate exists to make the caller ask for.

    ``content_hash`` is dropped as a marker rather than tightened: it carries no
    value to recognise, and it was redundant. Every Luxar root declares one of
    the other three — a scene or scene-subtree root a node ``type``, a
    standalone ``.gsplats.zarr`` the ``format_type`` header, a gsplat node a
    ``kind``.
    """
    attrs = dict(root.attrs)
    return bool(
        _as_marker(attrs.get("type")) in NODE_TYPES
        or _as_marker(attrs.get("format_type")) == FORMAT_TYPE_GSPLATS
        or _as_marker(attrs.get("kind")) in _LUXAR_NODE_KINDS
    )


def _as_marker(raw: Any) -> str | None:
    """A stored attr as a comparable marker string, or ``None``.

    Attrs come from arbitrary JSON, so the value may be a dict or a list —
    unhashable, and ``x in frozenset(...)`` raises ``TypeError`` on those rather
    than answering "not a marker". The gate must not turn a foreign store into a
    crash.
    """
    return raw if isinstance(raw, str) else None


def ensure_luxar_store(source_path: Path, root: zarr.Group, *, generic: bool) -> None:
    """Refuse a foreign store unless ``generic`` says the caller meant it.

    Shared by :func:`optimise_store` and the CLI's ``--dry-run`` path, which
    must not diverge: a dry run that happily plans a store the real run refuses
    is a plan the user cannot act on.
    """
    if generic or _is_luxar_store(root):
        return
    raise ValueError(
        f"{source_path} does not look like a Luxar scene or a .gsplats.zarr "
        f"tree; pass --generic to re-chunk an arbitrary zarr store"
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
    user's path nor a damaged previous one. A previous store is moved ASIDE
    rather than deleted first (:func:`_replace`), so a failed swap restores it
    instead of costing both copies.
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
        ensure_luxar_store(source_path, source, generic=generic)

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
            for warning in plan.playback_warnings:
                aprint(f"⚠ {warning.message}")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            staging = _staging_path(dest_path)
            # Computed BEFORE anything is written, so the cleanup below can
            # remove it however far the run got. Binding it to `_package`'s
            # RETURN left a hidden partial `.<name>.optimise-<pid>-<uuid>.zip`
            # beside the destination whenever the archive write itself failed
            # (ENOSPC on the last member) — the one window the all-or-nothing
            # promise did not cover.
            artifact = _artifact_path(staging, dest_path)
            consumed = False
            try:
                _write_store(
                    source, staging, by_path, plan, source_format, target_bytes, profile
                )
                aprint(
                    f"✓ {plan.n_rechunked}/{len(plan.arrays)} arrays re-chunked; "
                    f"{plan.source_n_chunks} → {plan.target_n_chunks} chunks"
                )
                _package(staging, artifact)
                if verify:
                    reread = open_group(artifact, mode="r")
                    try:
                        checked, payloads = _verify(source, reread)
                    finally:
                        close(reread)
                    aprint(
                        f"✓ Verified {checked} array{'' if checked == 1 else 's'} "
                        f"and {payloads} payload "
                        f"file{'' if payloads == 1 else 's'} byte-for-byte"
                    )
                _replace(artifact, dest_path)
                consumed = True
            finally:
                # The artifact only survives here on a failure path; on success
                # it has already been renamed onto the destination.
                shutil.rmtree(staging, ignore_errors=True)
                if not consumed and artifact != staging:
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

    A destination that IS a symlink is refused outright, before any work.
    ``Path.exists()`` follows the link and ``Path.is_symlink()`` does not, and
    every mutator here acts on the link itself: a link to a real store passed
    every check and then died in ``shutil.rmtree`` ("Cannot call rmtree on a
    symbolic link") with the finished output already built, and a DANGLING link
    reads as "nothing here", skipping the ``--overwrite`` requirement entirely
    before ``os.replace`` raised ``NotADirectoryError``. Both wasted the whole
    copy. Symlinked output paths are ordinary (small home, big ``/mnt``), so the
    error names the target and the two ways forward.
    """
    if dest_path.is_symlink():
        raise ValueError(
            f"optimise refuses to write to {dest_path}: it is a symlink (→ "
            f"{os.readlink(dest_path)}). The output is renamed into place, "
            f"which would replace the LINK rather than what it points at. "
            f"Give the link's target as the output path, or remove the link "
            f"first."
        )
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
        scene_hash = _restamp_content_hash(dest)
        environment = _baked_environment_group(dest)
        if scene_hash is not None and environment is not None:
            environment.attrs["scene_content_hash"] = scene_hash
        consolidate(dest)
    finally:
        close(dest)


def _artifact_path(staging: Path, dest_path: Path) -> Path:
    """Where the thing that will BECOME the output lives before the rename.

    The staging directory itself for a directory destination; a hidden sibling
    archive for a ``.zip`` one. Derived from the paths alone so the caller can
    register it for cleanup BEFORE any of it exists.
    """
    if dest_path.suffix.lower() != ".zip":
        return staging
    return Path(f"{staging}.zip")


def _package(staging: Path, artifact: Path) -> None:
    """Compress ``staging`` into ``artifact``, or do nothing when they are one.

    A ``.zip`` destination gets an archive whose members are keyed
    STORE-RELATIVE, which is what a :class:`zarr.storage.ZipStore` reads
    (stored, not deflated — the chunks already carry their own codec). Either
    way ``--verify`` runs against the artifact, and only a passing one is
    renamed into place.
    """
    if artifact == staging:
        return
    with zipfile.ZipFile(artifact, "w", zipfile.ZIP_STORED) as zf:
        for member in sorted(staging.rglob("*")):
            if member.is_file():
                zf.write(member, member.relative_to(staging).as_posix())


def _replace(new: Path, dest_path: Path) -> None:
    """Put ``new`` at ``dest_path``, moving whatever is there ASIDE first.

    Not "delete the old one, then rename". That order can lose BOTH copies:
    a ``shutil.rmtree`` that fails partway (a read-only child, EBUSY on a mount
    point, an interrupt) propagates before the caller can mark the artifact as
    consumed, so its ``finally`` then deletes the freshly built output too —
    and for a directory destination the artifact IS the staging tree. A single
    ``--overwrite`` could take out a good previous store and its replacement,
    which is precisely what :func:`optimise_store` promises cannot happen.

    So: rename the old one out of the way (atomic, and it works for a directory,
    a file or a symlink), swap the new one in, and only then delete the aside —
    restoring it if the swap fails. Deleting last also means a failure to clean
    up costs a hidden leftover rather than the output.
    """
    if not os.path.lexists(dest_path):
        os.replace(str(new), str(dest_path))
        return
    aside = dest_path.parent / (
        f".{dest_path.name}.optimise-old-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    os.replace(str(dest_path), str(aside))
    try:
        os.replace(str(new), str(dest_path))
    except BaseException:
        try:
            os.replace(str(aside), str(dest_path))
        except OSError as restore_failed:
            raise OSError(
                f"optimise could not swap the new store into {dest_path}, and "
                f"could not move the previous one back either. It is intact at "
                f"{aside} — rename it to {dest_path.name} by hand."
            ) from restore_failed
        raise
    if aside.is_dir() and not aside.is_symlink():
        shutil.rmtree(aside, ignore_errors=True)
    else:
        aside.unlink(missing_ok=True)
