"""Colors (``COLOR`` semantic type) zarr serializer.

Canonical writer for the ``COLOR`` semantic type across all three geometry types
(Points, Lines, GSplats), keeping default-precision selection identical everywhere.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ....encoding import SemanticType
from ..chunking import calculate_intelligent_chunks
from ..context import DatasetCtx


def write_colors(
    group: zarr.Group,
    colors: Union[NDArray[np.float32], tuple, list],
    spatial_index_data: Optional[Dict[str, Any]],
    n_elements: int,
    ctx: DatasetCtx,
    per_array_bytes: bool = False,
) -> None:
    """Write colors dataset to Zarr using ArrayEncoder.

    Args:
        group: Zarr group to write to
        colors: Colors array or tuple/list
        spatial_index_data: Optional spatial index for chunk optimization
        n_elements: Logical element count (points / vertices / splats).
            This must not be inferred from the
            positions zarr array because duplicate positions may be stored as
            an array_ref with physical shape ``(0, D)``.
        ctx: Encoder configuration (encoder, mode, compressor).
        per_array_bytes: Opt-in per-array dtype byte-budget chunking (points
            only; defaults ``False`` so lines/gsplats keep atom-sized chunks).
    """
    n_points = n_elements  # local alias keeps the rest of the body unchanged
    # RGBA colors: the alpha column is per-element opacity in [0, 1], not
    # emission — HDR detection and the display data range look at RGB only
    # (alpha <= 1 can never trip HDR, but it must not skew the range either).
    # Handle scalar vs array
    color_mode: Optional[Literal["sdr", "hdr"]] = None
    if isinstance(colors, (tuple, list)):
        # Detect HDR vs SDR from values (RGB components only)
        max_val = max(colors[:3])
        color_mode = "hdr" if max_val > 1.0 else "sdr"
        if color_mode == "hdr":
            aprint("  ✓ Detected HDR colors (values > 1.0)")
        n_elems = n_points
        color_chunks = None
    else:
        if colors.shape[0] == 1:
            n_elems = n_points
            color_chunks = None
        else:
            n_elems = None
            color_chunks = calculate_intelligent_chunks(
                colors.shape,
                spatial_index_data=spatial_index_data,
                dtype=colors.dtype,
                per_array_bytes=per_array_bytes,
            )

        # Detect color_mode for float arrays
        if np.issubdtype(colors.dtype, np.floating):
            # Float colors require explicit color_mode
            if np.any(colors[:, :3] > 1.0):
                color_mode = "hdr"
                aprint("  ✓ Detected HDR colors (values > 1.0)")
            else:
                color_mode = "sdr"

    # Use ArrayEncoder with all optimizations
    ctx.encoder.encode(
        data=colors,
        zarr_group=group,
        name="colors",
        semantic_type=SemanticType.COLOR,
        mode=ctx.encoding_mode,
        color_mode=color_mode,
        n_elements=n_elems,
        chunks=color_chunks,
        compressor=ctx.compressor,
    )

    # Log encoding result
    enc = group["colors"].attrs.get("encoding", {})
    enc_name = enc.get("name", "unknown")
    if enc_name == "broadcasted":
        aprint("  ✓ Wrote colors (broadcasted - uniform)")
    elif enc_name == "array_ref":
        aprint(f"  ✓ Wrote colors (reference to {enc['target']})")
    elif enc_name in ("lut_uint8", "lut_uint16"):
        aprint(f"  ✓ Wrote colors (LUT with {len(enc['lut'])} unique values)")
    elif enc_name in ("rgb_uint8", "uint8"):
        aprint("  ✓ Wrote colors (uint8)")
    elif enc_name in ("rgb_uint16", "uint16"):
        aprint("  ✓ Wrote colors (uint16)")
    elif enc_name == "float32":
        aprint("  ✓ Wrote HDR colors (float32)")
    else:
        aprint(f"  ✓ Wrote colors ({enc_name})")

    # Store color data range for layer controls (min/max of original data,
    # RGB only — alpha is opacity, not a display value)
    if isinstance(colors, np.ndarray) and colors.size > 0:
        group.attrs["color_data_range"] = [
            float(colors[:, :3].min()),
            float(colors[:, :3].max()),
        ]
    elif isinstance(colors, (tuple, list)):
        group.attrs["color_data_range"] = [
            float(min(colors[:3])),
            float(max(colors[:3])),
        ]
