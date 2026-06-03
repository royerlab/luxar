#!/usr/bin/env python3
"""Scalars and Colormap Example — map a scalar field through built-in LUTs.

This example demonstrates:
- The ``scalars=`` kwarg on ``add_points`` (and ``add_lines`` / ``add_gsplats``):
  per-point float values used as an index into a colormap LUT.
- The ``colormap=`` rendering attribute, accepting any of the 20 built-in
  colormap names (or a custom ``(N, 3)`` LUT array).
- How using ``scalars`` differs from supplying explicit ``colors`` — the
  viewer can change the colormap at runtime without re-uploading colors.

The scene contains three side-by-side spirals carrying the same scalar
field (turn count along the spiral) but rendered with three different
built-in colormaps so the difference is obvious.

Educational value:
- Master the ``scalars`` + ``colormap`` pattern for parametric colouring.
- See what the 20 built-in colormaps actually look like — pick favourites.
- Avoid the common trap of computing colors yourself when a colormap LUT
  would be both smaller on disk and runtime-tunable.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.colormaps import BUILTIN_COLORMAP_NAMES
from luxar.utils.paths import get_examples_output_dir


def spiral(n_points: int, x_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 3D conical spiral and a scalar field along its length.

    The scalar runs from 0.0 at the bottom of the spiral to 1.0 at the
    top. Feeding it through a colormap LUT produces a visible gradient.
    """
    t = np.linspace(0.0, 4.0 * np.pi, n_points, dtype=np.float32)
    radius = np.linspace(0.5, 3.0, n_points, dtype=np.float32)
    positions = np.column_stack(
        [
            radius * np.cos(t) + x_offset,
            radius * np.sin(t),
            np.linspace(-5.0, 5.0, n_points, dtype=np.float32),
        ]
    ).astype(np.float32)
    scalars = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    return positions, scalars


def main() -> None:
    """Render three identical spirals with three different colormaps."""
    output_path = get_examples_output_dir() / "scalars_and_colormap_example.zarr"
    aprint(f"Writing scalars + colormap example to {output_path}")
    aprint(f"Built-in colormaps available: {', '.join(BUILTIN_COLORMAP_NAMES)}")

    # Three colormaps with very different visual styles. Each spiral
    # uses the same scalar field — only the LUT differs.
    showcase = [
        ("viridis_spiral", "viridis", -8.0),
        ("turbo_spiral", "turbo", 0.0),
        ("inferno_spiral", "inferno", 8.0),
    ]

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        for name, colormap_name, x_offset in showcase:
            positions, scalars = spiral(n_points=3000, x_offset=x_offset)
            scene.add_points(
                name,
                positions,
                scalars=scalars,
                colormap=colormap_name,
                radii=0.06,
                sharpness=2.0,
                # layer=True surfaces the spiral in the layers panel so
                # the user can toggle each colormap variant independently.
                layer=True,
            )
            aprint(f"  Added {name} using colormap='{colormap_name}'")

        # Explainer overlay describing what to look for in the viewer.
        add_explainer(
            scene,
            title="Scalars + Colormap",
            body=(
                "Three identical spirals carry the same <code>scalars</code> "
                "field but use different <code>colormap</code> LUTs. The viewer "
                "can recolor from a LUT at runtime — <strong>no need</strong> to "
                "precompute or re-upload per-point colors."
            ),
            observe=[
                "Left spiral uses <code>viridis</code> (blue-green-yellow).",
                "Center uses <code>turbo</code> (rainbow blue-to-red).",
                "Right uses <code>inferno</code> (black-purple-orange).",
                "All three share an identical bottom-to-top scalar ramp.",
            ],
            observe_label="Look for",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")


if __name__ == "__main__":
    main()
