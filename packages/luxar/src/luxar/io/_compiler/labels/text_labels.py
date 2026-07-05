"""Per-element string-label CSR serializer."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import zarr
from arbol import aprint

from ....encoding.compression import resolve_compressor
from ....typing_utils.protocols import CompressorProtocol


def write_labels_csr(
    group: zarr.Group,
    labels: Sequence[str],
    n_elements: int,
    compressor: Optional[CompressorProtocol],
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
