"""Positions (``COORDINATE`` semantic type) zarr serializer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ....encoding import SemanticType
from ..chunking import calculate_intelligent_chunks
from ..context import DatasetCtx


def write_positions(
    group: zarr.Group,
    positions: NDArray[np.float32],
    spatial_index_data: Optional[Dict[str, Any]],
    ctx: DatasetCtx,
) -> None:
    """Write positions dataset to Zarr using ArrayEncoder.

    Args:
        group: Zarr group to write to
        positions: Positions array (may be reordered by spatial index)
        spatial_index_data: Optional spatial index data for chunk optimization
        ctx: Encoder configuration (encoder, mode, compressor).
    """
    # Calculate intelligent chunks (aligned with spatial index if available)
    chunks = calculate_intelligent_chunks(
        positions.shape,
        spatial_index_data=spatial_index_data,
        dtype=positions.dtype,
        per_array_bytes=True,
    )

    # Use ArrayEncoder for positions (COORDINATE semantic type)
    ctx.encoder.encode(
        data=positions,
        zarr_group=group,
        name="positions",
        semantic_type=SemanticType.COORDINATE,
        mode=ctx.encoding_mode,
        chunks=chunks,
        compressor=ctx.compressor,
    )

    # Log encoding result (e.g. linear_perchannel_u16 under AUTO, float32
    # under PRECISION or the large-extent fallback)
    enc = group["positions"].attrs.get("encoding", {})
    enc_name = enc.get("name", "unknown")
    aprint(f"  ✓ Wrote positions ({enc_name})")
