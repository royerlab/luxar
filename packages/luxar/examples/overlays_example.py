#!/usr/bin/env python3
"""Overlays Example — text, image, and HTML overlays on top of the canvas.

This example demonstrates:
- ``scene.add_text(...)`` — text rendered as HTML in viewport coords.
- ``scene.add_image(...)`` — image overlays (PNG, JPEG, WebP); numpy
  array, file path, or PIL Image accepted.
- ``scene.add_html(...)`` — sanitized HTML overlay with optional hover
  templating (``{hover_label}``, ``{hover_node}``, ``{hover_index}``).
- Hover interactivity: when ``hover=True``, the overlay text updates
  with the currently-hovered point's metadata.

A small 4-point scene gives the overlays something to interact with.
Per-point ``labels=[...]`` populates the hover templates.

Educational value:
- Discover the overlay API surface; nothing in the existing examples
  exercises it.
- See how to anchor overlays at viewport corners (``anchor='top-right'``,
  etc.) and how positions are normalized to [0, 1].
- Learn the hover-template pattern for live tooltips driven by GPU
  picking.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_inline_image(width: int = 64, height: int = 64) -> np.ndarray:
    """Build a small RGB gradient as a numpy array.

    ``scene.add_image`` accepts a numpy HWC uint8 array directly, so this
    avoids needing an external file. The gradient is just visually
    obvious — red increases left to right, green top to bottom.
    """
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    r = (xx / max(width - 1, 1) * 255).astype(np.uint8)
    g = (yy / max(height - 1, 1) * 255).astype(np.uint8)
    b = np.full_like(r, 128, dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


def main() -> None:
    """Build a tiny scene with text, image, and HTML overlays."""
    output_path = get_examples_output_dir() / "overlays_example.luxar.zarr"
    aprint(f"Writing overlays example to {output_path}")

    positions = np.array(
        [
            [-2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.5, 1.0],
            [1.0, 0.9, 0.3],
        ],
        dtype=np.float32,
    )
    labels = ["alpha", "beta", "gamma", "delta"]

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        scene.add_points(
            "labelled_points",
            positions,
            colors=colors,
            radii=0.4,
            sharpness=0.5,
            # Per-point labels feed the {hover_label} template below.
            labels=labels,
        )

        # 1. Static text overlay anchored to the top-left.
        scene.add_text(
            "Overlays demo: hover any point to see a live label.",
            position=(0.02, 0.02),
            anchor="top-left",
            font_size=0.025,
            color="white",
            background="rgba(0,0,0,0.6)",
            padding=0.01,
        )

        # 2. Inline-numpy image overlay in the top-right corner.
        scene.add_image(
            make_inline_image(),
            position=(0.98, 0.02),
            anchor="top-right",
            size=(0.1, 0.1),  # 10% × 10% of viewport
            opacity=0.85,
        )

        # 3. Hover-templated HTML overlay anchored to the bottom-center.
        # The ``{hover_label}`` placeholder is replaced at runtime with
        # the ``labels[i]`` entry of the currently-hovered point.
        scene.add_html(
            (
                "<div style='text-align:center'>"
                "Hovered point: <code>{hover_label}</code><br/>"
                "<small>node: {hover_node}, index: {hover_index}</small>"
                "</div>"
            ),
            position=(0.5, 0.98),
            anchor="bottom-center",
            opacity=0.95,
            hover=True,
        )

        # Explainer card placed bottom-left so it clears the demo overlays
        # (text top-left, image top-right, hover tooltip bottom-center).
        add_explainer(
            scene,
            title="Canvas Overlays",
            body=(
                "Three overlay kinds layer over the scene: "
                "<code>add_text</code>, <code>add_image</code> (inline numpy "
                "array), and a hover-templated <code>add_html</code> tooltip."
            ),
            observe=[
                "A text banner sits in the top-left corner.",
                "A red-green gradient image sits in the top-right.",
                "Hovering a point updates the bottom-center tooltip.",
            ],
            observe_label="Look for",
            anchor="bottom-left",
            position=(0.022, 0.968),
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")


if __name__ == "__main__":
    main()
