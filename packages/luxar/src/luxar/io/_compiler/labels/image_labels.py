"""Per-element image-label normalization for the compiler.

Private support module for :class:`luxar.io.compiler.LuxarZarrCompiler`. Converts
heterogeneous image-label inputs (bytes, PIL images, numpy arrays, file paths) into
encoded image bytes ready for CSR-style storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


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
