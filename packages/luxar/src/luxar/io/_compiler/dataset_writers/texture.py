"""Mesh texture zarr serializer.

The Mesh peer of :mod:`.colors`, and it exists as its own writer rather than an
arm of that one because a texture is the first **2-D image** Luxar stores. Every
other pixel-like payload in the format is per-element ``(N, C)``; a texture is
``(H, W, C)``, and the two differ in the three places that matter — chunking
(rows of an image, not runs of elements), the encoding decision (an image may
arrive already compressed by a codec we do not own), and what the viewer must
know before it fetches anything.

## Two payloads, one array

``raw`` writes the ``(H, W, C)`` array. Everything else writes a 1-D ``uint8``
array of encoded bytes — the same shape ``image_label_bytes`` already uses for
per-vertex hover thumbnails, so an encoded texture is a **fourth** use of a
shipped mechanism rather than a new one.

Encoded bytes deliberately go into a zarr *array* rather than a loose file beside
the store. The store abstraction is what makes ``.zarr.zip``, remote HTTP,
consolidated metadata and ``luxar optimise`` work, and — the decisive one —
``content_hash`` covers arrays. A loose PNG would not be hashed, so editing a
texture would not invalidate a warm viewer cache and users would keep seeing the
old image with no way to tell.

## Why the encoded path is written RAW

An encoded payload is already compressed by its own codec. Running blosc over a
JPEG buys nothing measurable and costs decode time on every load, so the encoded
branch passes ``compressor=None`` and writes through :func:`create_array`
directly, bypassing the encoder — there is no semantic type for an opaque blob,
since the encoder's types all describe *numbers* it may requantize. This mirrors
``image_labels``, whose module docstring records the same reasoning.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import numpy as np
import zarr
from arbol import aprint
from numpy.typing import NDArray

from ...._zarr_compat import create_array
from ....encoding import SemanticType
from ..chunking import calculate_intelligent_chunks
from ..context import DatasetCtx


def write_texture(
    group: zarr.Group,
    texture: NDArray[Any],
    encoding: str,
    width: Optional[int],
    height: Optional[int],
    channels: Optional[int],
    ctx: DatasetCtx,
) -> Tuple[int, int, int]:
    """Write a mesh texture and return its resolved ``(height, width, channels)``.

    Args:
        group: The mesh node's zarr group.
        texture: ``(H, W, C)`` array for ``raw``, else 1-D ``uint8`` encoded bytes.
        encoding: ``raw`` | ``png`` | ``webp`` | ``jpeg``.
        width: Declared width; required for encoded payloads.
        height: Declared height; required for encoded payloads.
        channels: Declared channel count; required for encoded payloads.
        ctx: Dataset write context (encoder, mode, compressor).

    Returns:
        ``(height, width, channels)`` as validated.
    """
    from ....validation.base import validate_texture_for_writing

    res_h, res_w, res_c = validate_texture_for_writing(
        texture, encoding, width, height, channels
    )
    arr = np.asarray(texture)

    if encoding != "raw":
        # Straight to `create_array`, bypassing the encoder — exactly what
        # `image_labels` does for `image_label_bytes`, and for the same two
        # reasons. There is no SemanticType for an opaque blob (the encoder's
        # types all describe NUMBERS it may requantize), and blosc over a JPEG
        # buys nothing measurable while costing decode time on every load. 1 MB
        # chunks, capped rather than tuned: an encoded texture is decoded whole,
        # since the browser's image decoder takes a complete buffer, so there is
        # no partial-read case to chunk for.
        create_array(
            group,
            "texture",
            data=arr,
            chunks=(min(int(arr.size), 1_048_576),),
            compressor=None,
            overwrite=True,
        )
        aprint(
            f"  ✓ Wrote texture ({encoding}, {arr.size:,} bytes → "
            f"{res_w}x{res_h}x{res_c})"
        )
        return res_h, res_w, res_c

    # `raw`: reuse the COLOR semantic type deliberately. A texture IS colour, and
    # keying it the same way means HDR follows the element-colour path for free —
    # float with any value > 1.0 quantizes through `geolog_perchannel_u16` under
    # AUTO and stays float32 under PRECISION, decoding to float32 either way.
    color_mode: Optional[Literal["sdr", "hdr"]] = None
    if np.issubdtype(arr.dtype, np.floating):
        color_mode = "hdr" if bool(np.any(arr[..., : min(res_c, 3)] > 1.0)) else "sdr"

    ctx.encoder.encode(
        data=arr,
        zarr_group=group,
        name="texture",
        semantic_type=SemanticType.COLOR,
        mode=ctx.encoding_mode,
        color_mode=color_mode,
        # Chunked by ROWS: the natural access unit of an image, and it keeps a
        # chunk contiguous in memory so a decode does not stride.
        chunks=calculate_intelligent_chunks((res_h, res_w, res_c), dtype=arr.dtype),
        compressor=ctx.compressor,
        deduplicate=False,
        allow_lut=False,
    )
    # RGB range only, and only for HDR — the same quantity and the same exclusion
    # of alpha as `write_colors`' `color_data_range`, so the viewer's window
    # derivation reads one shape whatever the source.
    if color_mode == "hdr":
        rgb = arr[..., : min(res_c, 3)]
        group.attrs["texture_data_range"] = [float(rgb.min()), float(rgb.max())]
    aprint(
        f"  ✓ Wrote texture (raw {arr.dtype}, {res_w}x{res_h}x{res_c}"
        f"{', HDR' if color_mode == 'hdr' else ''})"
    )
    return res_h, res_w, res_c
