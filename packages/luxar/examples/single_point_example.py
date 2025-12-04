#!/usr/bin/env python3
"""Single Point Example - The simplest possible Luxar scene.

This minimal example demonstrates:
- Creating the most basic scene with just one point
- Using default values for all optional parameters
- Understanding the minimal requirements for a Luxar scene
- Perfect starting point for beginners
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler


def main():
    """Create the simplest possible Luxar scene with a single point."""
    output_path = Path(__file__).parent / "single_point_example.zarr"

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
