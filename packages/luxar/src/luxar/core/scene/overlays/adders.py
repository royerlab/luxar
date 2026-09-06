"""Overlay adder impls: add_text / add_image / add_html.

Pure functions called by Scene's add_text / add_image / add_html
method delegates. Each takes a ``scene: Scene`` first arg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from arbol import aprint

from ...overlay import Overlay
from .internals import next_overlay_name, write_overlay

if TYPE_CHECKING:
    from ..scene import Scene


def add_text_impl(
    scene: "Scene",
    text: str,
    position: Tuple[float, float],
    *,
    name: Optional[str] = None,
    font_size: float = 0.025,
    font: str = "sans",
    color: str = "white",
    opacity: float = 1.0,
    anchor: str = "top-left",
    width: Optional[float] = None,
    text_align: str = "left",
    line_height: float = 1.4,
    background: Optional[str] = None,
    padding: float = 0.005,
    stroke_color: Optional[str] = None,
    stroke_width: float = 0.002,
    visible_range: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    transition: str = "none",
    transition_duration: float = 0.3,
    interactive: bool = False,
    blend_mode: str = "normal",
    hover: bool = False,
) -> Overlay:
    from ....validation.overlays import (
        validate_anchor,
        validate_font,
        validate_position,
        validate_text_align,
        validate_transition,
        validate_visible_range,
    )
    from ....validation.overlays import (
        validate_blend_mode as validate_overlay_blend_mode,
    )

    name = next_overlay_name(scene, name)
    position = validate_position(position)
    validate_anchor(anchor)
    validate_font(font)
    validate_text_align(text_align)
    validate_transition(transition)
    validate_overlay_blend_mode(blend_mode)
    validated_range = validate_visible_range(visible_range, scene._dimensions.names)

    attrs: Dict[str, Any] = {
        "type": "overlay_text",
        "text": str(text),
        "position": list(position),
        "font_size": float(font_size),
        "font": font,
        "color": color,
        "opacity": float(opacity),
        "anchor": anchor,
        "text_align": text_align,
        "line_height": float(line_height),
        "padding": float(padding),
        "stroke_width": float(stroke_width),
        "transition": transition,
        "transition_duration": float(transition_duration),
        "interactive": bool(interactive),
        "z_index": len(scene._overlays),
    }
    if width is not None:
        attrs["width"] = float(width)
    if background is not None:
        attrs["background"] = background
    if stroke_color is not None:
        attrs["stroke_color"] = stroke_color
    if blend_mode != "normal":
        attrs["blend_mode"] = blend_mode
    if hover:
        attrs["hover"] = True
    if validated_range is not None:
        attrs["visible_range"] = validated_range

    overlay = write_overlay(scene, name, "overlay_text", position, attrs)
    aprint(f"✓ Text overlay '{name}' added at ({position[0]:.2f}, {position[1]:.2f})")
    return overlay


def add_image_impl(
    scene: "Scene",
    image: Any,
    position: Tuple[float, float],
    *,
    name: Optional[str] = None,
    size: Optional[Tuple[float, float]] = None,
    opacity: float = 1.0,
    anchor: str = "top-left",
    blend_mode: str = "normal",
    format: str = "png",
    visible_range: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    transition: str = "none",
    transition_duration: float = 0.3,
    interactive: bool = False,
) -> Overlay:
    from ....validation.overlays import (
        validate_anchor,
        validate_image_input,
        validate_position,
        validate_transition,
        validate_visible_range,
    )
    from ....validation.overlays import (
        validate_blend_mode as validate_overlay_blend_mode,
    )

    name = next_overlay_name(scene, name)
    position = validate_position(position)
    validate_anchor(anchor)
    validate_overlay_blend_mode(blend_mode)
    validate_transition(transition)
    validated_range = validate_visible_range(visible_range, scene._dimensions.names)

    image_bytes, fmt = validate_image_input(image, fmt=format)
    image_filename = f"image.{fmt}"

    attrs: Dict[str, Any] = {
        "type": "overlay_image",
        "position": list(position),
        "image_file": image_filename,
        "opacity": float(opacity),
        "anchor": anchor,
        "blend_mode": blend_mode,
        "transition": transition,
        "transition_duration": float(transition_duration),
        "interactive": bool(interactive),
        "z_index": len(scene._overlays),
    }
    if size is not None:
        attrs["size"] = list(size)
    if validated_range is not None:
        attrs["visible_range"] = validated_range

    overlay = write_overlay(
        scene,
        name,
        "overlay_image",
        position,
        attrs,
        image_data=image_bytes,
        image_filename=image_filename,
    )
    aprint(f"✓ Image overlay '{name}' added at ({position[0]:.2f}, {position[1]:.2f})")
    return overlay


def add_video_impl(
    scene: "Scene",
    video: Any,
    position: Tuple[float, float],
    *,
    name: Optional[str] = None,
    size: Optional[Tuple[float, Optional[float]]] = None,
    opacity: float = 1.0,
    anchor: str = "top-left",
    blend_mode: str = "normal",
    loop: bool = True,
    autoplay: bool = True,
    muted: bool = True,
    playback_rate: float = 1.0,
    poster: Any = None,
    visible_range: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    transition: str = "none",
    transition_duration: float = 0.3,
    interactive: bool = False,
) -> Overlay:
    from ....validation.overlays import (
        validate_anchor,
        validate_image_input,
        validate_position,
        validate_transition,
        validate_video_input,
        validate_visible_range,
    )
    from ....validation.overlays import (
        validate_blend_mode as validate_overlay_blend_mode,
    )

    name = next_overlay_name(scene, name)
    position = validate_position(position)
    validate_anchor(anchor)
    validate_overlay_blend_mode(blend_mode)
    validate_transition(transition)
    validated_range = validate_visible_range(visible_range, scene._dimensions.names)
    if not (0.0 < float(playback_rate) <= 16.0):
        raise ValueError(f"playback_rate must be in (0, 16], got {playback_rate}")
    if not muted and autoplay:
        # Browsers refuse un-muted autoplay without a user gesture; a video that
        # never starts is worse than one that starts silent.
        raise ValueError("autoplay=True requires muted=True (browser autoplay policy)")
    if size is not None:
        if len(size) != 2 or size[0] is None or size[0] <= 0:
            raise ValueError(
                f"size must be (width, height-or-None) with width > 0, got {size}"
            )
        if size[1] is not None and size[1] <= 0:
            raise ValueError(f"size height must be > 0 or None, got {size[1]}")

    video_bytes, fmt = validate_video_input(video)
    video_filename = f"video.{fmt}"
    files: Dict[str, bytes] = {video_filename: video_bytes}

    attrs: Dict[str, Any] = {
        "type": "overlay_video",
        "position": list(position),
        "video_file": video_filename,
        "loop": bool(loop),
        "autoplay": bool(autoplay),
        "muted": bool(muted),
        "playback_rate": float(playback_rate),
        "opacity": float(opacity),
        "anchor": anchor,
        "blend_mode": blend_mode,
        "transition": transition,
        "transition_duration": float(transition_duration),
        "interactive": bool(interactive),
        "z_index": len(scene._overlays),
    }
    if poster is not None:
        poster_bytes, poster_fmt = validate_image_input(poster)
        poster_filename = f"poster.{poster_fmt}"
        files[poster_filename] = poster_bytes
        attrs["poster_file"] = poster_filename
    if size is not None:
        # A None height keeps the video's own aspect ratio (CSS height:auto).
        attrs["size"] = [float(size[0]), None if size[1] is None else float(size[1])]
    if validated_range is not None:
        attrs["visible_range"] = validated_range

    overlay = write_overlay(scene, name, "overlay_video", position, attrs, files=files)
    aprint(f"✓ Video overlay '{name}' added at ({position[0]:.2f}, {position[1]:.2f})")
    return overlay


def add_html_impl(
    scene: "Scene",
    html: str,
    position: Tuple[float, float],
    *,
    name: Optional[str] = None,
    width: Optional[float] = None,
    opacity: float = 1.0,
    anchor: str = "top-left",
    visible_range: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    transition: str = "none",
    transition_duration: float = 0.3,
    interactive: bool = False,
    blend_mode: str = "normal",
    hover: bool = False,
    hover_image_size: Optional[Tuple[float, float]] = None,
) -> Overlay:
    from ....validation.overlays import (
        sanitize_html,
        validate_anchor,
        validate_position,
        validate_transition,
        validate_visible_range,
    )
    from ....validation.overlays import (
        validate_blend_mode as validate_overlay_blend_mode,
    )

    name = next_overlay_name(scene, name)
    position = validate_position(position)
    validate_anchor(anchor)
    validate_transition(transition)
    validate_overlay_blend_mode(blend_mode)
    validated_range = validate_visible_range(visible_range, scene._dimensions.names)

    sanitized = sanitize_html(html)

    attrs: Dict[str, Any] = {
        "type": "overlay_html",
        "position": list(position),
        "html": sanitized,
        "opacity": float(opacity),
        "anchor": anchor,
        "transition": transition,
        "transition_duration": float(transition_duration),
        "interactive": bool(interactive),
        "z_index": len(scene._overlays),
    }
    if width is not None:
        attrs["width"] = float(width)
    if blend_mode != "normal":
        attrs["blend_mode"] = blend_mode
    if hover:
        attrs["hover"] = True
    if hover_image_size is not None:
        attrs["hover_image_size"] = list(hover_image_size)
    if validated_range is not None:
        attrs["visible_range"] = validated_range

    overlay = write_overlay(scene, name, "overlay_html", position, attrs)
    aprint(f"✓ HTML overlay '{name}' added at ({position[0]:.2f}, {position[1]:.2f})")
    return overlay
