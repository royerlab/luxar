"""Shared house style for example *explainer* overlays.

Every example in this folder drops a single, consistently-styled card over
the canvas by calling :func:`add_explainer`. The card always reads
top-to-bottom in the same order:

1. a bold **title** naming the feature being demonstrated,
2. an **explanation** paragraph (what the scene shows and why),
3. a labelled **"look for" list** — concrete things to verify / observe /
   notice in the viewer to confirm the feature works.

Centralising the styling here is deliberate: it guarantees a uniform,
elegant look across the whole example gallery, and lets every example
script stay focused on the *feature* it teaches rather than on overlay
boilerplate. The styling is plain inline CSS inside a sanitised HTML
overlay (see ``luxar.validation.overlays.ALLOWED_TAGS`` — ``div``/``span``/
``strong``/``ul``/``li``/``code`` with ``style`` survive sanitisation).

Usage
-----
>>> from _overlay_style import add_explainer
>>> add_explainer(
...     scene,
...     title="Per-point radii",
...     body="Each point's on-screen size comes from the per-point "
...          "<code>radii</code> array.",
...     observe=[
...         "Sizes increase smoothly from left to right.",
...         "The smallest point is 0.1 units; the largest 1.0.",
...     ],
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from luxar.core.overlay import Overlay
    from luxar.core.scene import Scene

# ---------------------------------------------------------------------------
# House palette — calm, high-contrast over both bright and dark scenes.
# ---------------------------------------------------------------------------
_CARD_BG = "rgba(14, 17, 23, 0.80)"  # near-black, lets the scene glow through
_BORDER = "rgba(255, 255, 255, 0.10)"
_ACCENT = "#5cc8ff"  # cool cyan accent bar + label colour
_TITLE_COLOR = "#ffffff"
_BODY_COLOR = "rgba(255, 255, 255, 0.86)"
_BULLET_COLOR = "rgba(255, 255, 255, 0.80)"


def _li(item: str) -> str:
    """Render one bullet as a flex row with an accent tick."""
    return (
        '<li style="position:relative; padding-left:1.15em; '
        'margin:0.28em 0; list-style:none;">'
        f'<span style="position:absolute; left:0; top:0; color:{_ACCENT};">'
        "&#8226;</span>"
        f"{item}</li>"
    )


def add_explainer(
    scene: "Scene",
    *,
    title: str,
    body: str,
    observe: Sequence[str],
    observe_label: str = "Look for",
    position: Tuple[float, float] = (0.022, 0.032),
    anchor: str = "top-left",
    width: float = 0.30,
    font_scale: float = 1.75,
    name: Optional[str] = None,
    visible_range: Optional[dict] = None,
) -> "Overlay":
    """Add a single house-style *explainer* card to ``scene``.

    Args:
        scene: The scene to annotate.
        title: Short feature name (rendered bold, largest).
        body: One or two sentences of explanation. May contain the inline
            tags allowed by the sanitiser (``<code>``, ``<strong>``, ``<br>``).
        observe: Bullet items — concrete things to look for / verify in the
            viewer. May contain inline tags.
        observe_label: Heading for the bullet list. One of
            ``"Look for"``, ``"Verify"``, ``"Observe"``, ``"Notice"``.
        position: Normalised (x, y) in [0, 1]; default top-left inset.
        anchor: Overlay anchor; default ``"top-left"``. Use a different
            corner when the default would cover the geometry.
        width: Card width as a fraction of viewport width (vw).
        font_scale: Base font size in vh units; the title, label, and
            bullets scale relative to this via ``em``.
        name: Optional explicit overlay name.
        visible_range: Optional dimension-gated visibility (see
            ``Scene.add_html``).

    Returns:
        The created :class:`~luxar.core.overlay.Overlay` descriptor.
    """
    label = observe_label.strip()

    bullets = "".join(_li(item) for item in observe)
    list_block = (
        (
            f'<div style="margin-top:0.7em; color:{_ACCENT}; '
            "font-size:0.72em; font-weight:600; letter-spacing:0.09em; "
            f'text-transform:uppercase;">{label}</div>'
            '<ul style="margin:0.35em 0 0 0; padding:0; '
            f'color:{_BULLET_COLOR}; font-size:0.9em; line-height:1.45;">'
            f"{bullets}</ul>"
        )
        if observe
        else ""
    )

    html = (
        # Outer card: rounded, subtle border, accent bar down the left edge.
        # Base font size is set here in vh so the whole card scales with the
        # viewport; every inner size is expressed in em relative to it.
        '<div style="'
        f"box-sizing:border-box; width:100%; font-size:{font_scale}vh; "
        "font-family:system-ui,-apple-system,sans-serif; "
        f"background:{_CARD_BG}; "
        f"border:1px solid {_BORDER}; border-left:3px solid {_ACCENT}; "
        "border-radius:9px; padding:0.85em 1.0em; "
        "box-shadow:0 6px 22px rgba(0,0,0,0.35); "
        "backdrop-filter:blur(3px); -webkit-backdrop-filter:blur(3px);"
        '">'
        # Title
        f'<div style="color:{_TITLE_COLOR}; font-size:1.3em; '
        "font-weight:650; line-height:1.2; margin-bottom:0.35em; "
        'letter-spacing:0.01em;">'
        f"{title}</div>"
        # Body / explanation
        f'<div style="color:{_BODY_COLOR}; font-size:0.95em; '
        'line-height:1.5;">'
        f"{body}</div>"
        # Look-for list
        f"{list_block}"
        "</div>"
    )

    return scene.add_html(
        html,
        position=position,
        anchor=anchor,
        width=width,
        name=name,
        visible_range=visible_range,
    )
