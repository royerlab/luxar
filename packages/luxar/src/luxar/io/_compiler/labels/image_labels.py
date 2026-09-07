"""Per-element image-label normalization for the compiler.

Private support module for :class:`luxar.io.compiler.LuxarZarrCompiler`. Converts
heterogeneous image-label inputs (bytes, PIL images, numpy arrays, file paths) into
encoded image bytes ready for CSR-style storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import zarr
from arbol import aprint

from luxar._zarr_compat import create_array

from ....encoding.compression import resolve_compressor
from ....validation.writing import (
    check_image_label_type,
    validate_image_labels_for_writing,
)

if TYPE_CHECKING:
    from ....encoding.compression import CompressorLike


def normalize_image_label(item: Any) -> bytes:
    """Convert a single image label input to encoded bytes.

    Accepts:
    - ``bytes`` / ``bytearray`` — used as-is (pre-encoded JPEG/WebP/PNG)
    - ``PIL.Image.Image`` — encoded to WebP (quality 85)
    - ``numpy.ndarray`` (H, W, C) uint8 — converted to PIL, then WebP
    - ``pathlib.Path`` / ``str`` — file read as raw bytes

    Runs :func:`check_image_label_type` FIRST (#1491), so the type-AND-shape
    dispatch lives in exactly one place; everything below is then free to
    assume ``item`` is one of the accepted types (and, for an ``ndarray``,
    one of the accepted shapes) and focus on the part no pure check can do —
    actually reading a file or calling PIL.

    Returns:
        Encoded image bytes, or ``b""`` for None / empty inputs.
    """
    check_image_label_type(item)

    if item is None:
        return b""
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, Path):
        return item.read_bytes()
    if isinstance(item, str):
        return Path(item).read_bytes()

    # PIL Image
    from PIL import Image as PILImage

    if isinstance(item, PILImage.Image):
        import io

        buf = io.BytesIO()
        item.save(buf, format="webp", quality=85)
        return buf.getvalue()

    # numpy array (H, W, C) uint8 — check_image_label_type has already
    # confirmed Pillow is importable, the type is right, AND the shape is one
    # of (H, W) / (H, W, 3) / (H, W, 4) (raising ValueError otherwise), so the
    # only two live shapes left here are 2-D (grayscale) and 3-channel — a
    # 4-channel ndarray falls to the `else` and gets RGBA.
    import io

    if item.ndim == 2:
        pil_img = PILImage.fromarray(item, mode="L")
    elif item.shape[2] == 3:
        pil_img = PILImage.fromarray(item, mode="RGB")
    else:
        pil_img = PILImage.fromarray(item, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="webp", quality=85)
    return buf.getvalue()


def write_image_labels_csr(
    group: zarr.Group,
    image_labels: Any,
    n_elements: int,
    compressor: "CompressorLike",
    sort_order: Optional[np.ndarray] = None,
) -> None:
    """Write per-element image labels using CSR-style encoding.

    Stores two zarr arrays:
    - ``image_label_offsets``: uint64 of shape (N+1,) — byte offset of each image
    - ``image_label_bytes``: uint8 — concatenated encoded image blobs

    Image ``i`` is decoded as ``image_label_bytes[offsets[i]:offsets[i+1]]``.
    Empty entries (no image) have ``offsets[i] == offsets[i+1]``.

    The ``image_label_bytes`` array uses **no compression** (``compressor=None``)
    because the image blobs are already compressed (JPEG/WebP/PNG). The offsets
    array uses the scene's default compressor since it is small.

    Args:
        group: Zarr group to write to.
        image_labels: Per-element images. Accepted types:
            - ``List[bytes]``: pre-encoded blobs
            - ``List[PIL.Image.Image]``: auto-encoded to WebP
            - ``List[numpy.ndarray]``: (H,W,C) uint8, auto-encoded to WebP
            - ``List[Path]`` or ``List[str]``: file paths, read as bytes
            - ``Dict[int, Any]``: sparse — missing indices get empty blobs
        n_elements: Expected element count (for validation).
        compressor: Scene default compressor for the small offsets array.
        sort_order: Optional index array to reorder (from spatial ordering).

    Note:
        The dense (non-``dict``) form is materialised into a ``list`` exactly
        ONCE, up front, before either validating or encoding it — so a
        single-pass iterable (one whose ``__iter__`` keeps returning the same,
        already-advanced iterator, unlike ``list`` / ``tuple`` / ``ndarray`` /
        ``pandas.Series``, which are all re-iterable) passed directly to this
        call is walked exactly once and works.

        A single-pass iterable is only supported on a call that comes STRAIGHT
        here, though. Every gate above this function (a writer's step-0 sweep,
        a ``substitutive_lod=`` wrapper's pre-split gate) walks the value
        itself and cannot hand its own materialised copy back to its caller,
        so :func:`validate_image_labels_for_writing` refuses a one-shot dense
        iterable outright — before anything is written — rather than draining
        it and leaving this function nothing to encode. A bare generator is
        refused one step earlier still, by that gate's ``len()``:
        ``TypeError: object of type 'generator' has no len()``.

    Raises:
        ValueError: Also raised (via :func:`validate_image_labels_for_writing`)
            if ``image_labels`` is a single-pass iterable the CALLER had
            already drained before calling this — the materialisation then
            sees zero items and the length check reports ``Image labels
            length (0) must match element count (N)`` instead of writing an
            all-empty CSR.
    """
    # Length/index/type checks now shared with the callers' pre-write gates —
    # see validate_image_labels_for_writing. Materialise the dense form ONCE
    # before that call (not after) so validating and encoding walk the exact
    # same concrete list: a Sized single-pass iterable would otherwise be
    # drained by the validator's own per-entry loop, leaving the blob-encoding
    # loop below nothing to see and silently writing an all-empty CSR (#1491).
    if not isinstance(image_labels, dict):
        image_labels = list(image_labels)
    validate_image_labels_for_writing(image_labels, n_elements)
    if isinstance(image_labels, dict):
        normalized: List[bytes] = [b""] * n_elements
        for idx, item in image_labels.items():
            normalized[idx] = normalize_image_label(item)
        blob_list = normalized
    else:
        blob_list = [normalize_image_label(item) for item in image_labels]

    # Apply spatial reordering if present
    if sort_order is not None:
        blob_list = [blob_list[i] for i in sort_order]

    # Build CSR arrays
    offsets = np.zeros(n_elements + 1, dtype=np.uint64)
    for i, blob in enumerate(blob_list):
        offsets[i + 1] = offsets[i] + len(blob)

    total_bytes = int(offsets[-1])
    image_bytes = np.zeros(max(total_bytes, 1), dtype=np.uint8)
    pos = 0
    for blob in blob_list:
        if blob:
            image_bytes[pos : pos + len(blob)] = np.frombuffer(blob, dtype=np.uint8)
            pos += len(blob)

    # Write offsets (small, compressible)
    create_array(
        group,
        "image_label_offsets",
        data=offsets,
        chunks=(min(n_elements + 1, 65536),),
        compressor=resolve_compressor(compressor, offsets.dtype),
        overwrite=True,
    )
    # Write image bytes — NO compression (already compressed blobs), 1MB chunks
    create_array(
        group,
        "image_label_bytes",
        data=image_bytes,
        chunks=(min(total_bytes, 1_048_576) if total_bytes > 0 else 1,),
        compressor=None,
        overwrite=True,
    )
    group.attrs["has_image_labels"] = True
    n_nonempty = sum(1 for b in blob_list if b)
    avg_size = total_bytes / n_nonempty if n_nonempty > 0 else 0
    aprint(
        f"  ✓ Wrote image labels ({n_nonempty}/{n_elements} non-empty, "
        f"{total_bytes:,} bytes, avg {avg_size:.0f} bytes/image)"
    )
