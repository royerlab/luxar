"""Validation functions for overlay parameters.

Validates positions, anchors, fonts, blend modes, transitions, visible ranges,
image inputs, and HTML content for the overlay system.
"""

from __future__ import annotations

import io
import re
from html import escape as _escape_html
from html.parser import HTMLParser
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

_IMAGE_SUFFIX_FORMATS = {
    ".jpeg": "jpeg",
    ".jpg": "jpeg",
    ".png": "png",
    ".webp": "webp",
}

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
ALLOWED_ATTRS: Set[str] = {
    "style",
    "href",
    "src",
    "alt",
    "class",
    "target",
    "title",
    "rel",
    "colspan",
    "rowspan",
    "width",
    "height",
}

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


_SUPPORTED_IMAGE_INPUTS = (
    "Supported: str/Path, bytes, numpy array, PIL Image, or imageio-compatible."
)


def _try_imageio_to_bytes(image: Any, fmt: str) -> Tuple[bytes, str] | None:
    """Decode an optional imageio input, preserving runtime decoder failures."""
    try:
        import imageio.v3 as iio
    except ImportError:
        return None

    if hasattr(image, "read"):
        try:
            arr = iio.imread(image)
        except Exception as exc:
            raise ValueError(
                f"Cannot process image of type {type(image).__name__}: imageio failed "
                f"with {type(exc).__name__}: {exc}. {_SUPPORTED_IMAGE_INPUTS}"
            ) from exc
    else:
        try:
            arr = np.asarray(image)
        except Exception as exc:
            raise ValueError(
                f"Cannot process image of type {type(image).__name__}: array conversion "
                f"failed with {type(exc).__name__}: {exc}. {_SUPPORTED_IMAGE_INPUTS}"
            ) from exc
        if arr.ndim not in (2, 3):
            return None
        try:
            return _numpy_to_bytes(arr, fmt), fmt
        except Exception as exc:
            raise ValueError(
                f"Cannot process image of type {type(image).__name__}: image conversion "
                f"failed with {type(exc).__name__}: {exc}. {_SUPPORTED_IMAGE_INPUTS}"
            ) from exc

    return _numpy_to_bytes(arr, fmt), fmt


def validate_image_input(
    image: Any,
    fmt: str = "png",
) -> Tuple[bytes, str]:
    """Convert any supported image input to encoded bytes.

    Accepts: file path (str/Path), raw bytes, numpy array (HWC uint8),
    PIL Image, or anything imageio can read.

    Args:
        image: Image data in any supported format. Pre-encoded bytes and path
            payloads must be PNG, JPEG, or WebP; a recognized path extension
            must match the payload.
        fmt: Target encoding format for decoded inputs ('png', 'jpeg', 'webp')

    Returns:
        Tuple of (encoded_bytes, format_string)

    Raises:
        ValueError: If the image cannot be processed, a pre-encoded payload is
            unsupported, or a recognized path extension contradicts its payload.
    """
    fmt = validate_image_format(fmt)

    # Pre-encoded inputs keep their actual format rather than being relabelled.
    if isinstance(image, bytes):
        return image, _detect_encoded_image_format(image)

    # Recognized file suffixes must agree with the payload format.
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise ValueError(f"Image file not found: {path}")
        image_bytes = path.read_bytes()
        detected_fmt = _detect_encoded_image_format(image_bytes, context=path)
        suffix_fmt = _IMAGE_SUFFIX_FORMATS.get(path.suffix.lower())
        if suffix_fmt is not None and suffix_fmt != detected_fmt:
            raise ValueError(
                f"Image file extension '{path.suffix}' does not match "
                f"the {detected_fmt} payload"
            )
        return image_bytes, detected_fmt

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

    imageio_result = _try_imageio_to_bytes(image, fmt)
    if imageio_result is not None:
        return imageio_result

    raise ValueError(
        f"Cannot process image of type {type(image).__name__}. {_SUPPORTED_IMAGE_INPUTS}"
    )


VALID_VIDEO_FORMATS: Set[str] = {"webm", "mp4"}

#: How a video overlay carries transparency. ``"stacked"``: the frame is the
#: colour on top and the alpha channel as a grey matte of the same size below
#: (an ordinary opaque clip twice as tall), which the viewer recombines in a
#: shader. Chosen over a VP9 alpha plane because Safari / WKWebView decode that
#: and silently drop the alpha.
VALID_VIDEO_ALPHA_MATTES: Set[str] = {"stacked"}


def validate_video_alpha_matte(alpha_matte: Any) -> None:
    """Refuse an ``alpha_matte`` layout the viewer does not recombine."""
    if alpha_matte is None:
        return
    if alpha_matte not in VALID_VIDEO_ALPHA_MATTES:
        raise ValueError(
            f"alpha_matte must be one of {sorted(VALID_VIDEO_ALPHA_MATTES)} or None, "
            f"got {alpha_matte!r}"
        )


def _detect_encoded_video_format(data: bytes) -> str:
    """Sniff a pre-encoded video payload: WebM (EBML header) or MP4 (``ftyp`` box).

    Raises:
        ValueError: for anything else — an overlay video must be one the
            browser can play; Ogg, AVI or a mislabelled PNG are refused here
            rather than failing silently on the display.
    """
    if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mp4"
    raise ValueError(
        "Unsupported video payload: expected WebM (EBML header) or MP4 ('ftyp' "
        "box). Encode turntables as VP9 WebM (with alpha) or H.264/AV1 MP4."
    )


def validate_video_input(video: Any) -> Tuple[bytes, str]:
    """Return ``(bytes, format)`` for a video overlay payload.

    Accepts raw bytes or a file path. Unlike images there is no decode/re-encode
    path: the payload is stored verbatim, so it must already be a browser-playable
    container (:data:`VALID_VIDEO_FORMATS`). A recognized path suffix must agree
    with the sniffed payload.
    """
    if isinstance(video, bytes):
        return video, _detect_encoded_video_format(video)
    if isinstance(video, (str, Path)):
        path = Path(video)
        if not path.is_file():
            raise ValueError(f"Video file not found: {path}")
        data = path.read_bytes()
        fmt = _detect_encoded_video_format(data)
        suffix = path.suffix.lower().lstrip(".")
        if suffix in VALID_VIDEO_FORMATS and suffix != fmt:
            raise ValueError(
                f"Video path suffix '.{suffix}' does not match its payload ({fmt})"
            )
        return data, fmt
    raise ValueError(f"video must be bytes or a file path, got {type(video).__name__}")


def _detect_encoded_image_format(
    image: bytes,
    context: Optional[Path] = None,
) -> str:
    """Detect a supported encoded image format from its signature."""
    if image.startswith(b"\x89PNG"):
        return "png"
    if image.startswith(b"\xff\xd8"):
        return "jpeg"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "webp"
    message = "Unsupported encoded image format. Expected PNG, JPEG, or WebP payload."
    if context is None:
        raise ValueError(message)
    raise ValueError(f"{message.removesuffix('.')}: {context}")


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


# HTML void (self-closing) elements: they never carry content.
_VOID_TAGS: Set[str] = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

# Disallowed tags whose TEXT CONTENT must also be dropped (raw-text or
# script-like elements). For non-void members we skip everything until the
# matching end tag; the void members (input, embed) are simply dropped.
_DROP_CONTENT_TAGS: Set[str] = {
    "script",
    "style",
    "iframe",
    "object",
    "embed",
    "form",
    "input",
    "textarea",
    "button",
    "select",
    "template",
    "noscript",
    "xmp",
}


def _is_dangerous_url(value: str) -> bool:
    """Return True if a decoded href/src value uses a dangerous URL scheme.

    The value has already been HTML-entity-decoded by the parser, so
    ``&#106;avascript:`` arrives as ``javascript:``. Browsers strip ASCII
    whitespace and control characters from within a scheme, so we collapse
    those out (matching ``jav&#9;ascript:``) before comparing case-insensitively.
    """
    collapsed = re.sub(r"[\x00-\x20]", "", value).lower()
    if collapsed.startswith(("javascript:", "vbscript:")):
        return True
    if collapsed.startswith("data:"):
        # Only raster image data URIs are safe; svg is scriptable.
        if not collapsed.startswith("data:image/"):
            return True
        if collapsed.startswith("data:image/svg"):
            return True
    return False


class _HtmlSanitizer(HTMLParser):
    """Allowlist HTML sanitizer built on the stdlib ``HTMLParser``.

    Only tags in ``ALLOWED_TAGS`` and attributes in ``ALLOWED_ATTRS`` survive;
    everything else is dropped. Raw-text/script-like tags additionally have
    their text content removed. ``href``/``src`` values with a dangerous scheme
    are stripped (the tag is kept).
    """

    def __init__(self) -> None:
        """Initialise the parser with an empty output buffer and skip state.

        ``convert_charrefs=True`` so character references in text are decoded
        before escaping (attribute values are decoded by the parser regardless).
        ``_skip_tag`` / ``_skip_depth`` track the currently-open content-dropping
        element (0 = not skipping).
        """
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        # Name of the disallowed content-dropping tag we are inside, and its
        # nesting depth (0 = not currently dropping content).
        self._skip_tag: Optional[str] = None
        self._skip_depth: int = 0

    def get_output(self) -> str:
        """Return the accumulated sanitized HTML as a single string."""
        return "".join(self._out)

    @staticmethod
    def _render_attrs(attrs: list[Tuple[str, Optional[str]]]) -> str:
        """Render an allowlisted attribute string for a start tag.

        Drops attributes not in ``ALLOWED_ATTRS`` and ``href``/``src`` values
        with a dangerous scheme; escapes the surviving values. Returns a
        leading-space-prefixed string, or ``""`` when nothing survives.
        """
        parts: list[str] = []
        for name, value in attrs:
            if name not in ALLOWED_ATTRS:
                continue
            if name in ("href", "src") and value is not None:
                if _is_dangerous_url(value):
                    continue
            if value is None:
                parts.append(name)
            else:
                escaped = (
                    value.replace("&", "&amp;")
                    .replace('"', "&quot;")
                    .replace("<", "&lt;")
                )
                parts.append(f'{name}="{escaped}"')
        return (" " + " ".join(parts)) if parts else ""

    def _emit_start(self, tag: str, attrs: list[Tuple[str, Optional[str]]]) -> None:
        """Append a start tag with its sanitized attributes to the output."""
        self._out.append(f"<{tag}{self._render_attrs(attrs)}>")

    def handle_starttag(self, tag: str, attrs: list[Tuple[str, Optional[str]]]) -> None:
        """Handle a start tag: skip content-dropping tags, emit allowed ones.

        While inside a content-dropping element the tag is swallowed (and its
        nesting depth tracked); a disallowed non-raw tag is dropped but its
        inner text is kept; an allowed tag is emitted with sanitized attrs.
        """
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in _DROP_CONTENT_TAGS:
            # Void raw-text tags (input, embed) have no content to skip.
            if tag not in _VOID_TAGS:
                self._skip_tag = tag
                self._skip_depth = 1
            return
        if tag in ALLOWED_TAGS:
            self._emit_start(tag, attrs)
        # Disallowed non-raw tag: drop the markup, keep any inner text.

    def handle_startendtag(
        self, tag: str, attrs: list[Tuple[str, Optional[str]]]
    ) -> None:
        if self._skip_depth:
            return
        if tag in _DROP_CONTENT_TAGS:
            return
        if tag in ALLOWED_TAGS:
            self._emit_start(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag in ALLOWED_TAGS and tag not in _VOID_TAGS:
            self._out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        """Append HTML-escaped text, unless inside a content-dropping element."""
        if not self._skip_depth:
            self._out.append(
                data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )

    def handle_comment(self, data: str) -> None:
        # Comments are dropped (may hide conditional or scripted content).
        return


def sanitize_html(html: str) -> str:
    """Sanitize HTML to a safe subset using an allowlist approach.

    Parses the input with the standard-library ``html.parser.HTMLParser`` and
    re-emits only tags in ``ALLOWED_TAGS`` with only attributes in
    ``ALLOWED_ATTRS``. Any other tag is dropped; raw-text/script-like tags
    (``script``, ``style``, ``iframe``, ``form``, ``template``, ...) have their
    text content dropped as well. On ``href``/``src`` the value is
    HTML-entity-decoded and any ``javascript:``/``vbscript:``/``data:`` scheme
    (except raster ``data:image/...``, not ``svg``) is stripped. Safe markup is
    preserved as closely as possible (original attribute order); text content is
    HTML-escaped so ``&``/``<``/``>`` round-trip.

    If the stdlib parser cannot handle the input at all (e.g. it raises
    ``AssertionError`` on malformed marked sections like ``<![bogus]>`` on
    CPython builds without the 2025 ``html.parser`` security patch), the whole
    input is escaped as plain text instead — arbitrary overlay HTML must never
    abort validation.

    This is defense-in-depth, not a hard security boundary: the viewer
    re-sanitizes overlay HTML client-side before rendering it.

    Args:
        html: Raw HTML string

    Returns:
        Sanitized HTML string

    Raises:
        ValueError: If ``html`` is not a string
    """
    if not isinstance(html, str):
        raise ValueError(f"html must be a string, got {type(html).__name__}")

    sanitizer = _HtmlSanitizer()
    try:
        sanitizer.feed(html)
        sanitizer.close()
    except Exception:
        # Unpatched CPython's HTMLParser raises AssertionError on unknown
        # marked-section keywords ("<![bogus]>"); fail closed as plain text.
        return _escape_html(html, quote=False)
    return sanitizer.get_output()
