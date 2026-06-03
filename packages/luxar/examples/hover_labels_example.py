#!/usr/bin/env python3
"""Hover Labels Example — per-point text + image labels for hover tooltips.

This example demonstrates:
- ``labels=[...]`` on ``add_points`` — per-point text strings keyed by
  point index. The viewer's built-in tooltip uses them, and overlay
  templates (``add_text(hover=True, …)``, ``add_html(hover=True, …)``)
  can reference them with ``{hover_label}``.
- ``image_labels=...`` — per-point thumbnail images. Accepts
  ``List[bytes]`` (PNG/WebP/JPEG bytes), ``List[PIL.Image]``, or a
  sparse ``Dict[int, bytes]``. Images appear in the hover tooltip when
  using an overlay template that includes ``{hover_image_label}``.
- The combination: an HTML overlay that interpolates both text and
  image labels, giving rich per-point tooltips.

Scene: 8 points arranged in a circle. Each carries a text label
("species_0" … "species_7"). Only the first three ALSO carry a small
numpy-generated PNG image label — image labels are sparse and optional,
so hovering points 3–7 shows text only and NO thumbnail (by design).
Each point's text label spells out whether a thumbnail is expected, so
the absence of an image on the later points reads as intentional rather
than a missing-data bug.

Educational value:
- Discover how to attach hover metadata to points without authoring an
  external image asset (PNG bytes are built in-memory).
- See the difference between the default built-in tooltip (just shows
  ``labels[i]``) and a custom HTML overlay tooltip that styles the
  hovered metadata however you like.
"""

import io

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def encode_png_from_array(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array to PNG bytes (in-memory, no file I/O).

    Uses the ``Pillow`` package, which ``arbol`` and other Luxar deps
    already pull in transitively. If Pillow isn't available in your
    environment, install it with ``pip install pillow``.
    """
    from PIL import Image  # local import keeps the example's top-level light

    return _png_bytes(Image.fromarray(rgb, mode="RGB"))


def _png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_thumbnail(hue: float, size: int = 48) -> bytes:
    """Generate a small radial-gradient PNG with the given hue [0, 1)."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2.0)
    radius = np.clip(radius, 0.0, 1.0)
    # Simple HSV→RGB with S=V=1 just for visual variety.
    h6 = hue * 6.0
    sector = int(h6) % 6
    f = h6 - sector
    table = {
        0: (1.0, f, 0.0),
        1: (1.0 - f, 1.0, 0.0),
        2: (0.0, 1.0, f),
        3: (0.0, 1.0 - f, 1.0),
        4: (f, 0.0, 1.0),
        5: (1.0, 0.0, 1.0 - f),
    }
    base = np.array(table[sector], dtype=np.float32)
    rgb = (base[None, None, :] * (1.0 - radius)[..., None] * 255.0).astype(np.uint8)
    return encode_png_from_array(rgb)


def main() -> None:
    """Build a small scene with hover-driven text and image overlays."""
    output_path = get_examples_output_dir() / "hover_labels_example.zarr"
    aprint(f"Writing hover-labels example to {output_path}")

    n_points = 8
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float32)
    radius = 4.0
    positions = np.column_stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(n_points, dtype=np.float32),
        ]
    ).astype(np.float32)

    # Per-point HSV-driven colors so the labels feel distinct.
    hues = np.linspace(0.0, 1.0, n_points, endpoint=False, dtype=np.float32)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    for i, h in enumerate(hues):
        h6 = float(h) * 6.0
        sector = int(h6) % 6
        f = h6 - sector
        table = {
            0: (1.0, f, 0.0),
            1: (1.0 - f, 1.0, 0.0),
            2: (0.0, 1.0, f),
            3: (0.0, 1.0 - f, 1.0),
            4: (f, 0.0, 1.0),
            5: (1.0, 0.0, 1.0 - f),
        }
        colors[i] = table[sector]

    # Only the first few points carry an image label; image labels are
    # sparse and optional. Drive both the labels and the image dict from
    # one count so they can never drift out of sync.
    n_image_points = 3

    # Spell out image expectation in each point's text label, so when you
    # hover a point with no thumbnail it's obviously by design — not a
    # missing-data bug. Points 0–2 say "(thumbnail below)"; 3–7 say
    # "(no thumbnail)".
    labels = [
        f"species_{i} ({'thumbnail below' if i < n_image_points else 'no thumbnail'})"
        for i in range(n_points)
    ]

    # The dict form keeps image labels sparse — only points 0, 1, 2 carry
    # images, matching the "(thumbnail below)" note in their text labels.
    image_labels = {i: make_thumbnail(float(hues[i])) for i in range(n_image_points)}

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        scene.add_points(
            "ring",
            positions,
            colors=colors,
            radii=0.5,
            sharpness=2.0,
            labels=labels,
            image_labels=image_labels,
        )

        # A bottom-anchored HTML overlay that interpolates the hovered
        # point's text label, index, and (when present) thumbnail.
        scene.add_html(
            (
                "<div style='text-align:center; min-width:14em'>"
                "<strong>{hover_label}</strong> (index {hover_index})"
                "<br/>{hover_image_label}"
                "</div>"
            ),
            position=(0.5, 0.97),
            anchor="bottom-center",
            hover=True,
            hover_image_size=(0.08, 0.08),
            opacity=0.95,
        )

        # Explainer card top-left so it clears the bottom-center hover tooltip.
        add_explainer(
            scene,
            title="Hover Labels",
            body=(
                "Each point carries a per-point text <code>labels</code> entry; "
                "the first three also carry a sparse <code>image_labels</code> "
                "thumbnail. Hovering interpolates both into the HTML overlay."
            ),
            observe=[
                "Hovering any point shows its <code>species_N</code> label.",
                "Points 0-2 also show a radial-gradient thumbnail.",
                "Points 3-7 show text only, by design (no thumbnail).",
            ],
            observe_label="Verify",
            anchor="top-left",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")


if __name__ == "__main__":
    main()
