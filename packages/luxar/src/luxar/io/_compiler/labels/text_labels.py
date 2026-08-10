"""Per-element string-label CSR serializer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np
import zarr
from arbol import aprint

from ....encoding.compression import resolve_compressor

if TYPE_CHECKING:
    from ....encoding.compression import CompressorLike


def write_labels_csr(
    group: zarr.Group,
    labels: Sequence[str],
    n_elements: int,
    compressor: "CompressorLike",
    sort_order: Optional[np.ndarray] = None,
) -> None:
    """Write per-element string labels using CSR-style encoding.

    Stores two zarr arrays:
    - ``label_offsets``: uint64 of shape (N+1,) — byte offset of each label
    - ``label_bytes``: uint8 — concatenated UTF-8 encoded label strings

    Label ``i`` is decoded as ``label_bytes[offsets[i]:offsets[i+1]]``.
    Empty strings (null labels) have ``offsets[i] == offsets[i+1]``.

    Args:
        group: Zarr group to write to
        labels: Sequence of strings, one per element. Length must equal n_elements.
        n_elements: Expected element count (for validation)
        compressor: Scene default compressor for both CSR arrays.
        sort_order: Optional index array to reorder labels (e.g. from spatial ordering).
            For points: ``ordering_data["sort_order"]``
            For lines: ``ordering_data["vertex_sort_indices"]``
            For gsplats: ``ordering_data["sort_order"]``
    """
    if len(labels) != n_elements:
        raise ValueError(
            f"Labels length ({len(labels)}) must match element count ({n_elements})"
        )

    # Apply spatial reordering if present
    ordered_labels: Sequence[str] = labels
    if sort_order is not None:
        ordered_labels = [labels[i] for i in sort_order]

    # Build CSR arrays
    offsets = np.zeros(n_elements + 1, dtype=np.uint64)
    encoded_parts: list[bytes] = []
    for i, label in enumerate(ordered_labels):
        encoded = label.encode("utf-8") if label else b""
        encoded_parts.append(encoded)
        offsets[i + 1] = offsets[i] + len(encoded)

    total_bytes = int(offsets[-1])
    label_bytes = np.zeros(max(total_bytes, 1), dtype=np.uint8)
    pos = 0
    for encoded in encoded_parts:
        if encoded:
            label_bytes[pos : pos + len(encoded)] = np.frombuffer(
                encoded, dtype=np.uint8
            )
            pos += len(encoded)

    # Write to zarr
    group.create_dataset(
        "label_offsets",
        data=offsets,
        chunks=(min(n_elements + 1, 65536),),
        compressor=resolve_compressor(compressor, offsets.dtype),
        overwrite=True,
    )
    group.create_dataset(
        "label_bytes",
        data=label_bytes,
        chunks=(min(total_bytes, 65536) if total_bytes > 0 else 1,),
        compressor=resolve_compressor(compressor, label_bytes.dtype),
        overwrite=True,
    )
    group.attrs["has_labels"] = True
    n_nonempty = sum(1 for lbl in ordered_labels if lbl)
    aprint(
        f"  ✓ Wrote labels ({n_nonempty}/{n_elements} non-empty, {total_bytes:,} bytes)"
    )


def validate_ladder_labels(
    levels: Sequence[Mapping[str, Any]],
    positions_key: str,
) -> bool:
    """Pre-write gate for an additive ladder's labels; returns whether it is labelled.

    PURE — reads only ``levels``, touches no store — so the multi-LOD writers can
    call it BEFORE ``require_group`` creates the parent node. A rejected ladder
    must not leave an empty group behind (the writers' documented fail-fast
    contract).

    Enforces two rules:

    - **All-or-nothing**: labels on every level or on none. A partially-labelled
      ladder cannot produce a correct union, and silently labelling only part of
      it would misalign every slot after the first unlabelled level.
    - **Per-level length**: each level's label count must equal that level's own
      element count. The flat writers check this themselves; a laddered write
      hands them ``labels=None``, so the check has to happen here instead.
      Skipped when any level's element array is not ``(N, D)``: that is a
      geometry fault, and the per-level writer's positions validator names it
      properly. Diagnosing it here would both report the wrong fault and index
      ``shape[0]`` on a scalar.

    Args:
        levels: The per-level dicts handed to a multi-LOD writer.
        positions_key: The key holding each level's ``(N, D)`` element array —
            ``"positions"`` for Points, ``"vertices"`` for Lines.

    Returns:
        ``True`` when the ladder carries labels (so the caller should build the
        parent union CSR), ``False`` when no level does.

    Raises:
        ValueError: On mixed label presence (the message names the first
            unlabelled level) or a per-level length mismatch.
    """
    from ....validation.base import validate_labels_for_writing

    labelled_flags = [lvl.get("labels") is not None for lvl in levels]
    if not any(labelled_flags):
        return False
    if not all(labelled_flags):
        missing = labelled_flags.index(False)
        raise ValueError(
            f"labels must be provided for every additive LOD level or for none; "
            f"level {missing} (additive_{missing}) has no labels"
        )
    level_shapes = [np.shape(lvl[positions_key]) for lvl in levels]
    if any(len(shape) != 2 for shape in level_shapes):
        # A level whose element array is not (N, D) has no well-defined element
        # count, so leave the whole per-level label check to the writers: the
        # first level write raises the positions validator's guided message
        # ("Got 1D array with N elements ..."), which is the real fault and the
        # same error the caller would see with no labels at all.
        return True
    for lvl, shape in zip(levels, level_shapes):
        validate_labels_for_writing(lvl["labels"], int(shape[0]))
    return True


def write_ladder_union_labels_csr(
    group: zarr.Group,
    level_labels: Sequence[Sequence[str]],
    level_sort_orders: Sequence[Optional[np.ndarray]],
    n_elements: int,
    compressor: "CompressorLike",
) -> None:
    """Write ONE label CSR spanning an additive ladder's levels, on the parent.

    An additive LOD ladder stores its data in ``additive_<i>/`` subgroups, but
    the viewer's progressive loader concatenates the levels it has loaded into a
    single buffer — so no one level's array is the thing a pick index addresses.
    The label CSR therefore lives on the PARENT ladder node and spans the levels;
    the ``additive_<i>`` subgroups carry no label arrays at all.

    **Index-space contract**: index ``k`` of the parent CSR is the ``k``-th
    element of the concatenation ``additive_0 || additive_1 || …``, with each
    level in its own **stored** (spatially reordered) order. This is the SAME
    on-disk index space a flat labelled leaf's CSR uses, just spanning the
    ladder's levels rather than one array.

    The viewer commits levels coarsest-first, so a fully-loaded ladder maps
    straight through: index ``k`` is committed slot ``k``. The COMMITTED BUFFER is
    not in general a prefix of this union, though — the per-level loader compacts
    out elements culled by the current nD slice and fetches only the chunk ranges
    a query intersects, so slots shift. A labelled ladder therefore resolves at
    the RAW committed slot: exact for a fully-loaded 3D scene (no per-element
    slice culling), and otherwise carrying the slot shift that issue #1421 /
    PR #1425 removed for FLAT nodes by publishing a visible-slot -> on-disk-index
    map. That map is deliberately not published across a ladder — each level's map
    is in that level's own on-disk space, so the concatenation clears it —
    and extending it (offsetting each level by the preceding levels' on-disk
    counts) is the remaining piece of work, tracked as issue #1439.

    Under the ``partition=``-outer + ``additive_lod=``-inner composition the CSR
    lands on each ``part_<i>`` ladder parent, which is exactly where the viewer
    looks: since #1415 / PR #1420 the label lookup path is the HIT LEAF scene node
    (``result.mainNode.name``), and a laddered part's scene node IS its ladder
    parent — the outermost ``kind=partition`` wrapper is the *reported* path only.
    So that composition resolves too, with the same raw-committed-slot caveat as
    an unpartitioned ladder.

    For **Lines** the CSR is per-VERTEX (matching the flat Lines writer), while
    the viewer's Lines pick id is a per-SEGMENT storage slot. Issue #1424 bridged
    that granularity for FLAT lines nodes — the picked segment's slot resolves back
    to that segment's START vertex row in the stored ordering — but it does so
    through the same visible-slot -> on-disk-index map no ladder publishes, so
    across a ladder only a broadcast (one-string-per-node) label set resolves until
    #1439 carries that map over the levels.

    Args:
        group: The PARENT ladder zarr group (not a subgroup).
        level_labels: One label sequence per level, in ``additive_0 …``
            (coarsest → finest) order, each in that level's INPUT order.
        level_sort_orders: One spatial permutation per level, aligned with
            ``level_labels``. ``None`` for a level that was not spatially
            reordered (identity).
        n_elements: Total element count across all levels (for validation).
        compressor: Scene default compressor for both CSR arrays.

    Raises:
        ValueError: If ``level_labels`` and ``level_sort_orders`` differ in
            length, or (via :func:`write_labels_csr`) if the union length does
            not match ``n_elements``.
    """
    if len(level_labels) != len(level_sort_orders):
        raise ValueError(
            f"level_labels has {len(level_labels)} entries but level_sort_orders "
            f"has {len(level_sort_orders)} — one permutation per level is required"
        )

    union: list[str] = []
    for labels, sort_order in zip(level_labels, level_sort_orders):
        if sort_order is None:
            union.extend(labels)
        else:
            union.extend(labels[i] for i in sort_order)

    # The union is already in final (committed) order — no further permutation.
    write_labels_csr(group, union, n_elements, compressor, None)
