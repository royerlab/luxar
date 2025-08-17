#!/usr/bin/env python3
"""World Sizing Example - Verify world-space point sizing fundamentals.

This example demonstrates:
- The fundamental principle of world-space point sizing
- Creating points that should exactly touch based on their radii
- Testing the mathematical relationship: points of radius r at distance 2r touch
- Providing reference points at various distances for perspective testing

This is a minimal test case for verifying correct world-space point sizing.
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Scene


def create_touching_points():
    """Create two points that should just touch.

    Returns:
        Tuple of (positions, colors, radii)
    """
    # Define the radius for our test points
    radius = 1.0

    # Create two points separated by exactly 2*radius
    # They should just touch when rendered with world-space sizing
    positions = np.array(
        [
            [-radius, 0, 0],  # Left point
            [radius, 0, 0],  # Right point
        ],
        dtype=np.float32,
    )

    # Use distinct colors to see them clearly
    colors = np.array(
        [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
        ],
        dtype=np.float32,
    )

    # Both points have the same radius
    radii = np.full(2, radius, dtype=np.float32)

    return positions, colors, radii


def create_reference_points():
    """Create reference points at various distances.

    Returns:
        Tuple of (positions, colors, radii)
    """
    positions = []
    colors = []
    radii = []

    # Add reference points at different Z distances
    for i, z_dist in enumerate([5, 10, 20]):
        positions.append([0, 2, z_dist])
        # Gradient from yellow to blue based on distance
        colors.append([1.0 - i / 3, 1.0 - i / 3, i / 3])
        radii.append(0.5)

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(radii, dtype=np.float32),
    )


def main():
    """Create a minimal world-space sizing test scene."""
    output_path = Path(__file__).parent / "world_sizing_example.zarr"

    aprint(f"Creating world sizing test at {output_path}")
    aprint("This example tests the fundamental world-space sizing principle:")
    aprint("Two points of radius r at distance 2r should just touch")

    # Create scene
    scene = Scene(output_path)

    # Add metadata
    scene.attrs["description"] = """
    World-Space Sizing Test
    =======================

    This minimal scene tests world-space point sizing:

    Main test: Red and green points
    - Both have radius 1.0
    - Centers are 2.0 units apart
    - They should just touch at the origin

    Reference points: Yellow to blue gradient
    - Located at different depths (5, 10, 20 units)
    - Help verify perspective scaling

    Expected behavior:
    - Red and green points should touch but not overlap
    - This relationship should be maintained at any:
      * Window size
      * Fullscreen state
      * Field of view
      * Viewing angle
    """

    # Create main test points
    positions, colors, radii = create_touching_points()

    scene.add_points(
        "TouchingPoints",
        positions,
        colors=colors,
        radii=radii,
        sharpness=10.0,  # High sharpness for clear boundaries
        opacity=1.0,
        blending_mode="normal",
    )

    # Add reference points
    ref_positions, ref_colors, ref_radii = create_reference_points()

    scene.add_points(
        "ReferencePoints",
        ref_positions,
        colors=ref_colors,
        radii=ref_radii,
        sharpness=5.0,
        opacity=0.8,
        blending_mode="additive",
    )

    # Add axis indicators
    axis_positions = np.array(
        [
            # X axis
            [0, 0, 0],
            [3, 0, 0],
            # Y axis
            [0, 0, 0],
            [0, 3, 0],
            # Z axis
            [0, 0, 0],
            [0, 0, 3],
        ],
        dtype=np.float32,
    )

    axis_colors = np.array(
        [
            [0.5, 0, 0],
            [1, 0, 0],  # Dark to bright red for X
            [0, 0.5, 0],
            [0, 1, 0],  # Dark to bright green for Y
            [0, 0, 0.5],
            [0, 0, 1],  # Dark to bright blue for Z
        ],
        dtype=np.float32,
    )

    scene.add_points(
        "Axes",
        axis_positions,
        colors=axis_colors,
        radii=0.05,
        sharpness=10.0,
        opacity=1.0,
        blending_mode="additive",
    )

    scene.finalize()

    aprint("\n✓ Scene created successfully")
    aprint("  Main test: Red and green points (radius 1.0, separation 2.0)")
    aprint("  Reference: 3 points at different depths")
    aprint("  Axes: RGB indicators for orientation")
    aprint("\n" + "=" * 60)
    aprint("VERIFICATION STEPS:")
    aprint("1. Run: luxar serve world_sizing_example.zarr")
    aprint("2. Verify red and green points are just touching")
    aprint("3. Rotate view - touching relationship maintained")
    aprint("4. Press SPACE for fullscreen - still touching")
    aprint("5. Use Shift+Wheel to change FOV - still touching")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
