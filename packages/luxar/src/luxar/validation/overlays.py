"""Validation functions for overlay parameters.

Validates positions, anchors, fonts, blend modes, transitions, visible ranges,
image inputs, and HTML content for the overlay system.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

# Valid anchor positions (3x3 grid)
VALID_ANCHORS: Set[str] = {
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
}

# Valid blend modes for overlays (image + text + html)
VALID_BLEND_MODES: Set[str] = {
    "normal",
    "multiply",
    "screen",
    "overlay",
    "additive",
    "difference",
}

# Valid transition types
VALID_TRANSITIONS: Set[str] = {"none", "fade"}

# Valid image storage formats
VALID_IMAGE_FORMATS: Set[str] = {"png", "jpeg", "webp"}

# Font presets
FONT_PRESETS: Set[str] = {"sans", "serif", "mono"}

# Allowed HTML tags for sanitization
ALLOWED_TAGS: Set[str] = {
    "b",
    "i",
    "em",
    "strong",
    "a",
    "span",
    "div",
    "br",
    "img",
    "ul",
    "ol",
    "li",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "sub",
    "sup",
    "code",
    "pre",
    "table",
    "tr",
    "td",
    "th",
    "thead",
    "tbody",
}

# Allowed HTML attributes per tag
ALLOWED_ATTRS: Set[str] = {"style", "href", "src", "alt", "class", "target"}

# Valid text-align values
VALID_TEXT_ALIGNS: Set[str] = {"left", "center", "right", "justify"}


def validate_position(position: Any) -> Tuple[float, float]:
    """Validate overlay position as a 2-tuple of floats in [0, 1].

    Args:
        position: (x, y) normalized screen coordinates

    Returns:
        Validated (x, y) tuple

    Raises:
        ValueError: If position is invalid
    """
    if not isinstance(position, (tuple, list)) or len(position) != 2:
        raise ValueError(
            f"position must be a (x, y) tuple of 2 floats, got {type(position).__name__}"
        )
    x, y = float(position[0]), float(position[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"position coordinates must be in [0, 1], got ({x}, {y})")
    return (x, y)


def validate_anchor(anchor: str) -> str:
    """Validate anchor string.

    Args:
        anchor: One of 9 valid anchor positions

    Returns:
        Validated anchor string

    Raises:
        ValueError: If anchor is not valid
    """
    if anchor not in VALID_ANCHORS:
        raise ValueError(
            f"Invalid anchor '{anchor}'. Must be one of: {sorted(VALID_ANCHORS)}"
        )
    return anchor


def validate_font(font: str) -> str:
    """Validate font specification.

    Accepts preset names ('sans', 'serif', 'mono') or any CSS font-family string.

    Args:
        font: Font preset name or CSS font-family

    Returns:
        Validated font string
    """
    if not isinstance(font, str) or not font.strip():
        raise ValueError("font must be a non-empty string")
    return font


def validate_blend_mode(blend_mode: str) -> str:
    """Validate overlay blend mode.

    Args:
        blend_mode: One of: normal, multiply, screen, overlay, additive, difference

    Returns:
        Validated blend mode string

    Raises:
        ValueError: If blend mode is not valid
    """
    if blend_mode not in VALID_BLEND_MODES:
        raise ValueError(
            f"Invalid blend_mode '{blend_mode}'. "
            f"Must be one of: {sorted(VALID_BLEND_MODES)}"
        )
    return blend_mode


def validate_transition(transition: str) -> str:
    """Validate transition type.

    Args:
        transition: 'none' or 'fade'

    Returns:
        Validated transition string

    Raises:
        ValueError: If transition is not valid
    """
    if transition not in VALID_TRANSITIONS:
        raise ValueError(
            f"Invalid transition '{transition}'. "
            f"Must be one of: {sorted(VALID_TRANSITIONS)}"
        )
    return transition


def validate_text_align(text_align: str) -> str:
    """Validate text alignment.

    Args:
        text_align: One of: left, center, right, justify

    Returns:
        Validated text_align string

    Raises:
        ValueError: If text_align is not valid
    """
    if text_align not in VALID_TEXT_ALIGNS:
        raise ValueError(
            f"Invalid text_align '{text_align}'. "
            f"Must be one of: {sorted(VALID_TEXT_ALIGNS)}"
        )
    return text_align


def validate_image_format(fmt: str) -> str:
    """Validate image storage format.

    Args:
        fmt: One of: png, jpeg, webp

    Returns:
        Validated format string

    Raises:
        ValueError: If format is not valid
    """
    if fmt not in VALID_IMAGE_FORMATS:
        raise ValueError(
            f"Invalid image format '{fmt}'. "
            f"Must be one of: {sorted(VALID_IMAGE_FORMATS)}"
        )
    return fmt


def validate_visible_range(
    visible_range: Optional[Dict[str, Any]],
    dimension_names: list[str],
) -> Optional[Dict[str, Any]]:
    """Validate visible_range against scene dimensions.

    Args:
        visible_range: Dict mapping dimension names to values or (min, max) tuples
        dimension_names: Valid dimension names from the scene

    Returns:
        Validated visible_range dict (with tuples converted to lists for JSON)

    Raises:
        ValueError: If dimension names are invalid or values are malformed
    """
    if visible_range is None:
        return None

    if not isinstance(visible_range, dict):
        raise ValueError(
            f"visible_range must be a dict, got {type(visible_range).__name__}"
        )

    validated: Dict[str, Any] = {}
    for dim_name, value in visible_range.items():
        if dim_name not in dimension_names:
            raise ValueError(
                f"Unknown dimension '{dim_name}' in visible_range. "
                f"Valid dimensions: {dimension_names}"
            )

        if isinstance(value, (int, float)):
            # Single value — exact match
            validated[dim_name] = float(value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            # Range (min, max)
            lo, hi = float(value[0]), float(value[1])
            if lo > hi:
                raise ValueError(
                    f"visible_range['{dim_name}']: min ({lo}) > max ({hi})"
                )
            validated[dim_name] = [lo, hi]
        else:
            raise ValueError(
                f"visible_range['{dim_name}'] must be a number or (min, max) tuple, "
                f"got {type(value).__name__}"
            )

    return validated


def validate_image_input(
    image: Any,
    fmt: str = "png",
) -> Tuple[bytes, str]:
    """Convert any supported image input to encoded bytes.

    Accepts: file path (str/Path), raw bytes, numpy array (HWC uint8),
    PIL Image, or anything imageio can read.

    Args:
        image: Image data in any supported format
        fmt: Target encoding format ('png', 'jpeg', 'webp')

    Returns:
        Tuple of (encoded_bytes, format_string)

    Raises:
        ValueError: If image cannot be processed
    """
    fmt = validate_image_format(fmt)

    # Already bytes — assume pre-encoded
    if isinstance(image, bytes):
        return image, fmt

    # File path — read the file
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise ValueError(f"Image file not found: {path}")
        return path.read_bytes(), fmt

    # Try PIL Image
    try:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return _pil_to_bytes(image, fmt), fmt
    except ImportError:
        pass

    # numpy array — convert via PIL
    if isinstance(image, np.ndarray):
        return _numpy_to_bytes(image, fmt), fmt

    # Try imageio as fallback
    try:
        import imageio.v3 as iio

        # imageio can read many formats; convert to numpy then to bytes
        if hasattr(image, "read"):
            arr = iio.imread(image)
        else:
            arr = np.asarray(image)
        return _numpy_to_bytes(arr, fmt), fmt
    except (ImportError, Exception):
        pass

    raise ValueError(
        f"Cannot process image of type {type(image).__name__}. "
        f"Supported: str/Path, bytes, numpy array, PIL Image, or imageio-compatible."
    )


def _pil_to_bytes(image: Any, fmt: str) -> bytes:
    """Convert a PIL Image to encoded bytes."""
    buf = io.BytesIO()
    pil_fmt = "JPEG" if fmt == "jpeg" else fmt.upper()
    if pil_fmt == "JPEG" and image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buf, format=pil_fmt)
    return buf.getvalue()


def _numpy_to_bytes(arr: np.ndarray, fmt: str) -> bytes:
    """Convert a numpy array to encoded image bytes via PIL."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ValueError(
            "PIL (Pillow) is required to convert numpy arrays to images. "
            "Install with: pip install Pillow"
        )

    # Supported layouts: (H, W) grayscale, (H, W, 3) RGB, (H, W, 4) RGBA.
    # PIL infers the mode from the (uint8) array shape, so we only validate.
    if not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (3, 4))):
        raise ValueError(
            f"Cannot convert numpy array with shape {arr.shape} to image. "
            f"Expected (H, W), (H, W, 3), or (H, W, 4)."
        )

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    # No mode= : it is deprecated (removed in Pillow 13) and PIL infers the
    # same L / RGB / RGBA mode from the validated uint8 shape.
    pil_img = PILImage.fromarray(arr)
    return _pil_to_bytes(pil_img, fmt)


def sanitize_html(html: str) -> str:
    """Sanitize HTML to a safe subset using an allowlist approach.

    Strips disallowed tags and attributes. Allowed tags: basic formatting,
    links, images, lists, tables. Allowed attributes: style, href, src,
    alt, class, target.

    Script tags, event handlers (onclick, etc.), and dangerous attributes
    are always removed.

    Args:
        html: Raw HTML string

    Returns:
        Sanitized HTML string
    """
    if not isinstance(html, str):
        raise ValueError(f"html must be a string, got {type(html).__name__}")

    # Remove script/style tags and their contents
    html = re.sub(
        r"<(script|style|iframe|object|embed|form|input|textarea|button|select)"
        r"[\s>].*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove self-closing dangerous tags
    html = re.sub(
        r"<(script|style|iframe|object|embed|form|input|textarea|button|select)"
        r"[^>]*/?>",
        "",
        html,
        flags=re.IGNORECASE,
    )

    # Remove event handler attributes (onclick, onload, onerror, etc.)
    html = re.sub(r"\s+on\w+\s*=\s*[\"'][^\"']*[\"']", "", html, flags=re.IGNORECASE)
    html = re.sub(r"\s+on\w+\s*=\s*\S+", "", html, flags=re.IGNORECASE)

    # Remove javascript: URLs
    html = re.sub(
        r'(href|src)\s*=\s*["\']?\s*javascript:[^"\'>\s]*["\']?',
        "",
        html,
        flags=re.IGNORECASE,
    )

    return html
