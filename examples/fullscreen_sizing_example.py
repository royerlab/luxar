#!/usr/bin/env python3
"""Fullscreen Sizing Example - Test world-space point sizing consistency.

This example demonstrates:
- World-space point sizing that remains consistent across resolution changes
- Creating a grid of touching spheres for visual verification
- Testing that point sizing is independent of window size and fullscreen state
- Verifying the mathematical relationship between point radius and distance

Key concept: Two spheres of radius r at distance 2r should always just touch,
regardless of viewport size, fullscreen state, or field of view.
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Scene


def create_test_grid(grid_size=5, sphere_radius=0.1):
    """Create a 3D grid of touching spheres for testing.

    Args:
        grid_size: Number of spheres along each axis
        sphere_radius: Radius of each sphere

    Returns:
        Tuple of (positions, colors, radii) arrays
    """
    # Distance between sphere centers (they should just touch)
    spacing = 2 * sphere_radius

    positions = []
    colors = []
    radii = []

    # Create grid centered at origin
    start = -(grid_size - 1) * spacing / 2

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x = start + i * spacing
                y = start + j * spacing
                z = start + k * spacing

                positions.append([x, y, z])

                # Color based on position for easy identification
                r = (i + 1) / grid_size
                g = (j + 1) / grid_size
                b = (k + 1) / grid_size
                colors.append([r, g, b])

                radii.append(sphere_radius)

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(radii, dtype=np.float32),
    )


def main():
    """Create a test scene for verifying world-space point sizing."""
    output_path = Path(__file__).parent / "fullscreen_sizing_example.zarr"

    aprint(f"Creating fullscreen sizing test at {output_path}")
    aprint("This example tests world-space point sizing consistency")
    aprint("")
    aprint("Testing Instructions:")
    aprint("1. Run: luxar serve fullscreen_sizing_example.zarr")
    aprint("2. Verify spheres are touching (no gaps, no overlaps)")
    aprint("3. Press SPACE to toggle fullscreen")
    aprint("4. Spheres should maintain touching relationship")
    aprint("5. Resize window - spheres should still touch")
    aprint("6. Use Shift+Wheel to change FOV - relationship maintained")

    # Create scene
    scene = Scene(output_path)

    # Add metadata with instructions
    scene.attrs["description"] = """
    World-Space Point Sizing Test
    ==============================
    
    This scene contains a 5x5x5 grid of spheres that should just touch.
    The touching relationship should be maintained regardless of:
    - Window size
    - Fullscreen state  
    - Field of view
    
    If two spheres of radius r are at distance 2r, they should always
    just touch - this is the fundamental test of correct world-space sizing.
    """

    # Create test grid
    positions, colors, radii = create_test_grid(grid_size=5, sphere_radius=0.1)

    # Use high sharpness for clear sphere boundaries
    sharpness = np.full(len(positions), 10.0, dtype=np.float32)

    # Add the test grid
    scene.add_points(
        "TouchingSpheres",
        positions,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
        opacity=1.0,
        blending_mode="additive",
    )

    # Add axis markers for reference
    axis_length = 1.0
    axis_positions = np.array(
        [
            [0, 0, 0],
            [axis_length, 0, 0],  # X axis
            [0, 0, 0],
            [0, axis_length, 0],  # Y axis
            [0, 0, 0],
            [0, 0, axis_length],  # Z axis
        ],
        dtype=np.float32,
    )

    axis_colors = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],  # Red for X
            [0, 1, 0],
            [0, 1, 0],  # Green for Y
            [0, 0, 1],
            [0, 0, 1],  # Blue for Z
        ],
        dtype=np.float32,
    )

    scene.add_points(
        "Axes", axis_positions, colors=axis_colors, radii=0.02, sharpness=10.0
    )

    scene.finalize()

    aprint(f"\n✓ Test scene created with {len(positions)} spheres")
    aprint("  Grid size: 5x5x5")
    aprint("  Sphere radius: 0.1 units")
    aprint("  Spacing: 0.2 units (spheres should just touch)")
    aprint("\n" + "=" * 60)
    aprint("EXPECTED BEHAVIOR:")
    aprint("- Spheres should touch but not overlap")
    aprint("- Relationship maintained in fullscreen")
    aprint("- Relationship maintained when resizing")
    aprint("- Relationship maintained at different FOVs")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
