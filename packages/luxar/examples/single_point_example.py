#!/usr/bin/env python3
"""Single Point Example - The simplest possible Luxar scene.

This minimal example demonstrates:
- Creating the most basic scene with just one point
- Using default values for all optional parameters
- Understanding the minimal requirements for a Luxar scene
- Perfect starting point for beginners
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main():
    """Create the simplest possible Luxar scene with a single point."""
    output_path = get_examples_output_dir() / "single_point_example.zarr"

    aprint(f"Creating single point example at {output_path}")
    aprint("This is the simplest possible Luxar scene!")

    # Create scene with progressive writer
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Create a single point at the origin
        position = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        # Add the point with minimal parameters
        scene.add_points(
            "SinglePoint",
            position,
            # Everything else uses defaults:
            # - colors: None (will be white)
            # - radii: None (will use default radius)
            # - sharpness: None (will use default sharpness)
            # - opacity: 1.0
            # - gamma: 1.0
            # - blending_mode: "additive"
        )

        # Alternative: Specify some properties explicitly
        scene.add_points(
            "ColoredPoint",
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),  # Position at (1, 0, 0)
            colors=[1.0, 0.0, 0.0],  # Red color
            radii=0.2,  # Larger radius
        )

        add_explainer(
            scene,
            title="Single point scene",
            body="The minimal Luxar scene: a default white point at the origin "
            "and a second point given an explicit color and <code>radii</code>. "
            "Only a <strong>name</strong> and <strong>positions</strong> are "
            "required; everything else falls back to defaults.",
            observe=[
                "A white point sits at the origin <code>(0, 0, 0)</code>.",
                "A red point sits to its side at <code>(1, 0, 0)</code>.",
                "The red point is visibly larger (<code>radii=0.2</code>).",
            ],
            observe_label="Look for",
        )

    aprint("\n" + "=" * 60)
    aprint("SINGLE POINT EXAMPLE")
    aprint("=" * 60)
    aprint("Created the simplest possible Luxar scene!")
    aprint("  - White point at origin (0, 0, 0)")
    aprint("  - Red point at (1, 0, 0)")
    aprint("\nThis example shows:")
    aprint("  - Minimal required parameters (name and positions)")
    aprint("  - Default values for all optional parameters")
    aprint("  - How to specify individual properties")
    aprint(f"\nTo view: luxar serve {output_path}")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
