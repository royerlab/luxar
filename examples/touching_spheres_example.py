#!/usr/bin/env python3
"""Touching Spheres Example - Visualize world-space point relationships.

This example demonstrates:
- Creating pairs of spheres that should exactly touch
- Testing world-space sizing with different configurations
- Verifying point size calculations with various radii
- Visual validation of the touching relationship

The fundamental principle: Two spheres of radius r separated by
distance 2r should just touch at a single point.
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Scene


def create_touching_pairs():
    """Create various configurations of touching sphere pairs.

    Returns:
        Tuple of (positions, colors, radii) arrays
    """
    positions = []
    colors = []
    radii = []

    # Test case 1: Horizontal pair (along X axis)
    radius = 0.1
    spacing = 2 * radius
    positions.extend(
        [
            [-spacing / 2, 0, 0],  # Left sphere
            [spacing / 2, 0, 0],  # Right sphere
        ]
    )
    colors.extend(
        [
            [1.0, 0.0, 0.0],  # Red
            [1.0, 0.0, 0.0],  # Red
        ]
    )
    radii.extend([radius, radius])

    # Test case 2: Vertical pair (along Y axis)
    positions.extend(
        [
            [0, -spacing / 2 - 0.5, 0],  # Bottom sphere
            [0, spacing / 2 - 0.5, 0],  # Top sphere
        ]
    )
    colors.extend(
        [
            [0.0, 1.0, 0.0],  # Green
            [0.0, 1.0, 0.0],  # Green
        ]
    )
    radii.extend([radius, radius])

    # Test case 3: Depth pair (along Z axis)
    positions.extend(
        [
            [0.5, 0, -spacing / 2],  # Back sphere
            [0.5, 0, spacing / 2],  # Front sphere
        ]
    )
    colors.extend(
        [
            [0.0, 0.0, 1.0],  # Blue
            [0.0, 0.0, 1.0],  # Blue
        ]
    )
    radii.extend([radius, radius])

    # Test case 4: Different sizes touching
    radius_large = 0.15
    radius_small = 0.05
    diagonal_spacing = radius_large + radius_small
    positions.extend(
        [
            [-diagonal_spacing / 2, 0.5, 0],  # Large sphere
            [diagonal_spacing / 2, 0.5, 0],  # Small sphere
        ]
    )
    colors.extend(
        [
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
        ]
    )
    radii.extend([radius_large, radius_small])

    # Test case 5: Grid of touching spheres
    grid_size = 3
    grid_start = -0.3
    for i in range(grid_size):
        for j in range(grid_size):
            x = grid_start + i * spacing
            y = grid_start + j * spacing
            z = -0.5
            positions.append([x, y, z])
            # Checkerboard coloring
            if (i + j) % 2 == 0:
                colors.append([0.0, 1.0, 1.0])  # Cyan
            else:
                colors.append([1.0, 0.5, 0.0])  # Orange
            radii.append(radius)

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(radii, dtype=np.float32),
    )


def main():
    """Create a scene with various touching sphere configurations."""
    output_path = Path(__file__).parent / "touching_spheres_example.zarr"

    aprint(f"Creating touching spheres example at {output_path}")
    aprint("This example demonstrates various touching sphere configurations")
    aprint("")
    aprint("Configurations included:")
    aprint("- Red pair: Horizontal touching along X axis")
    aprint("- Green pair: Vertical touching along Y axis")
    aprint("- Blue pair: Depth touching along Z axis")
    aprint("- Yellow/Magenta: Different sizes touching")
    aprint("- Cyan/Orange grid: 3x3 grid of touching spheres")

    # Create scene
    scene = Scene(output_path)

    # Add metadata
    scene.attrs["description"] = """
    Touching Spheres Visualization
    ===============================

    This scene demonstrates the fundamental principle of world-space
    point sizing: spheres should maintain their touching relationships
    regardless of viewing conditions.

    Test cases:
    1. Red pair - horizontal alignment
    2. Green pair - vertical alignment
    3. Blue pair - depth alignment
    4. Yellow/Magenta - different sizes
    5. Cyan/Orange grid - multiple touching pairs

    All pairs should just touch at a single point.
    """

    # Create the touching spheres
    positions, colors, radii = create_touching_pairs()

    # Use high sharpness for crisp edges
    sharpness = np.full(len(positions), 8.0, dtype=np.float32)

    # Add the spheres
    scene.add_points(
        "TouchingSpheresDemo",
        positions,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
        opacity=1.0,
        blending_mode="additive",
    )

    scene.finalize()

    aprint(f"\n✓ Scene created with {len(positions)} spheres")
    aprint("\n" + "=" * 60)
    aprint("VIEWING INSTRUCTIONS:")
    aprint("1. Run: luxar serve touching_spheres_example.zarr")
    aprint("2. Rotate the view to see different perspectives")
    aprint("3. Verify all pairs are just touching")
    aprint("4. Press SPACE for fullscreen - touching maintained")
    aprint("5. Use Shift+Wheel to change FOV - touching maintained")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
