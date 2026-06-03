#!/usr/bin/env python3
"""Radius Basic Example - Introduction to point size control.

This example demonstrates:
- How the radius parameter controls point visual size
- Three distinct radius values for clear comparison
- World-space sizing (size independent of camera distance)
- Default radius behavior when not specified

Educational value:
- Understand the radius parameter fundamentals
- See immediate visual effect of different radii
- Learn that radius is in world-space units
- Perfect starting point before radius_showcase

Key principle:
- Radius defines point size in scene units (not pixels)
- Larger radius = larger visual appearance
- Independent of camera position or viewport
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main():
    """Create a simple test scene with three groups of different-sized points."""
    output_path = get_examples_output_dir() / "radius_basic_example.zarr"

    aprint(f"Creating radius test scene at {output_path}")

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Create three rows of points with different sizes
        n_points = 10
        y_positions = [-2, 0, 2]
        radii_values = [0.05, 0.2, 0.5]
        colors_rgb = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]  # Red, Green, Blue
        labels = ["Small", "Medium", "Large"]

        for i, (y, radius, color, label) in enumerate(
            zip(y_positions, radii_values, colors_rgb, labels)
        ):
            # Create a line of points
            x = np.linspace(-5, 5, n_points)
            y = np.full(n_points, y)
            z = np.zeros(n_points)

            positions = np.column_stack([x, y, z]).astype(np.float32)
            radii = np.full(n_points, radius, dtype=np.float32)
            colors = np.tile(color, (n_points, 1)).astype(np.float32)

            scene.add_points(f"{label}Points", positions, colors, radii=radii)
            aprint(f"Added {label} points with radius {radius}")

        add_explainer(
            scene,
            title="Per-point radii",
            body="Three rows of points, each row a different <code>radii</code> "
            "value. Radius is in <strong>world-space units</strong>, so on-screen "
            "size is independent of camera distance and viewport.",
            observe=[
                "Top row: large blue points (<code>radii=0.5</code>).",
                "Middle row: medium green points (<code>radii=0.2</code>).",
                "Bottom row: small red points (<code>radii=0.05</code>).",
                "Sizes stay consistent as you zoom in and out.",
            ],
            observe_label="Look for",
        )

        aprint("\n✓ Test scene created successfully!")
        aprint("\nExpected result when viewing:")
        aprint("- Top row: Large blue points (radius=0.5)")
        aprint("- Middle row: Medium green points (radius=0.2)")
        aprint("- Bottom row: Small red points (radius=0.05)")
        aprint(f"\nTo view: luxar serve {output_path}")


if __name__ == "__main__":
    main()
