"""Hover-overlay auto-injection.

Called by the compiler during finalize() if any node had labels or
image_labels and no user-defined hover overlay exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arbol import aprint

from .internals import write_overlay

if TYPE_CHECKING:
    from ..scene import Scene


def auto_inject_hover_overlay(scene: "Scene") -> None:
    """Auto-inject a default hover overlay if labels/image_labels exist.

    Called by the compiler during finalize(). Checks:
    1. At least one node has labels or image_labels
    2. No existing overlay has hover=True
    3. suppress_hover_overlay is False

    When image_labels are present, uses an HTML overlay (for ``<img>`` tag).
    When only text labels exist, uses a text overlay (current behavior).
    """
    if not scene._has_labels and not scene._has_image_labels:
        return
    if scene._suppress_hover_overlay:
        return
    # Check if user already defined a hover overlay
    if any(o.attrs.get("hover") for o in scene._overlays):
        return

    has_text = scene._has_labels
    has_img = scene._has_image_labels

    # Inject separate overlays for image and text so they don't
    # interfere (image loading would cause layout shift in a
    # combined overlay). Both anchor top-right; demos can suppress
    # auto-injection and define custom hover overlays for
    # different layouts.
    z = len(scene._overlays)

    if has_img:
        aprint("  Auto-injecting hover image overlay")
        write_overlay(
            scene,
            name="__hover_image",
            overlay_type="overlay_html",
            position=(0.98, 0.02),
            attrs={
                "type": "overlay_html",
                "hover": True,
                "html": "{hover_image_label}",
                "position": [0.98, 0.02],
                "anchor": "top-right",
                "background": "rgba(0,0,0,0.7)",
                "padding": 0.008,
                "opacity": 1.0,
                "transition": "fade",
                "transition_duration": 0.15,
                "interactive": False,
                "z_index": z,
            },
        )
        z += 1

    if has_text:
        aprint("  Auto-injecting hover text overlay")
        write_overlay(
            scene,
            name="__hover_text",
            overlay_type="overlay_text",
            position=(0.98, 0.02),
            attrs={
                "type": "overlay_text",
                "hover": True,
                "text": "{hover_label}",
                "position": [0.98, 0.02],
                "anchor": "top-right",
                "font_size": 0.018,
                "font": "sans",
                "color": "white",
                "background": "rgba(0,0,0,0.7)",
                "padding": 0.008,
                "opacity": 1.0,
                "transition": "fade",
                "transition_duration": 0.15,
                "interactive": False,
                "z_index": z,
            },
        )
