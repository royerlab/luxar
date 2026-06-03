"""Per-element image-label normalization for the compiler.

Private support module for :class:`luxar.io.compiler.LuxarZarrCompiler`. Converts
heterogeneous image-label inputs (bytes, PIL images, numpy arrays, file paths) into
encoded image bytes ready for CSR-style storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import zarr
from arbol import aprint

from ....typing_utils.protocols import CompressorProtocol


def normalize_image_label(item: Any) -> bytes:
    """Convert a single image label input to encoded bytes.

    Accepts:
    - ``bytes`` / ``bytearray`` — used as-is (pre-encoded JPEG/WebP/PNG)
    - ``PIL.Image.Image`` — encoded to WebP (quality 85)
    - ``numpy.ndarray`` (H, W, C) uint8 — converted to PIL, then WebP
    - ``pathlib.Path`` / ``str`` — file read as raw bytes

    Returns:
        Encoded image bytes, or ``b""`` for None / empty inputs.
    """
    if item is None:
        return b""
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, Path):
        return item.read_bytes()
    if isinstance(item, str):
        return Path(item).read_bytes()

    # PIL Image
    try:
        from PIL import Image as PILImage

        if isinstance(item, PILImage.Image):
            import io

            buf = io.BytesIO()
            item.save(buf, format="webp", quality=85)
            return buf.getvalue()
    except ImportError:
        raise ImportError(
            "Pillow is required to encode PIL Image objects as image labels. "
            "Install it with: pip install Pillow"
        )

    # numpy array (H, W, C) uint8
    if isinstance(item, np.ndarray):
        try:
            import io

            from PIL import Image as PILImage

            if item.ndim == 2:
                pil_img = PILImage.fromarray(item, mode="L")
            elif item.ndim == 3 and item.shape[2] == 3:
                pil_img = PILImage.fromarray(item, mode="RGB")
            elif item.ndim == 3 and item.shape[2] == 4:
                pil_img = PILImage.fromarray(item, mode="RGBA")
            else:
                raise ValueError(
                    f"Unsupported ndarray shape for image label: {item.shape}. "
                    f"Expected (H, W), (H, W, 3), or (H, W, 4)."
                )
            buf = io.BytesIO()
            pil_img.save(buf, format="webp", quality=85)
            return buf.getvalue()
        except ImportError:
            raise ImportError(
                "Pillow is required to encode numpy arrays as image labels. "
                "Install it with: pip install Pillow"
            )

    raise TypeError(
        f"Unsupported image label type: {type(item).__name__}. "
        f"Expected bytes, PIL.Image, numpy.ndarray, or file path."
    )


def write_image_labels_csr(
    group: zarr.Group,
    image_labels: Any,
    n_elements: int,
    compressor: CompressorProtocol,
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
    """
    # Normalize dict (sparse) to list
    if isinstance(image_labels, dict):
        normalized: List[bytes] = [b""] * n_elements
        for idx, item in image_labels.items():
            if idx < 0 or idx >= n_elements:
                raise ValueError(
                    f"Image label index {idx} out of range [0, {n_elements})"
                )
            normalized[idx] = normalize_image_label(item)
        blob_list = normalized
    else:
        if len(image_labels) != n_elements:
            raise ValueError(
                f"Image labels length ({len(image_labels)}) must match "
                f"element count ({n_elements})"
            )
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
    group.create_dataset(
        "image_label_offsets",
        data=offsets,
        chunks=(min(n_elements + 1, 65536),),
        compressor=compressor,
        overwrite=True,
    )
    # Write image bytes — NO compression (already compressed blobs), 1MB chunks
    group.create_dataset(
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
