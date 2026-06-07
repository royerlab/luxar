#!/usr/bin/env python3
"""Point Spacing Example - Understanding the relationship between point size and spacing.

This example demonstrates:
- The fundamental principle: spacing = 2 × radius for touching points
- How to calculate proper spacing for points of different sizes
- Visual verification of world-space sizing correctness
- Multiple test configurations (horizontal, vertical, depth, mixed sizes, grid)
- Why this matters: ensures points don't overlap or have gaps

Educational value:
- Shows the mathematical relationship between radius and spacing
- Demonstrates world-space sizing (size independent of camera/viewport)
- Provides visual test cases for size calculation verification
- Essential for understanding point density and packing
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_touching_pairs():
    """Create various configurations of touching sphere pairs.

    Returns:
        Tuple of (positions, colors, radii) arrays
    """
    positions = []
    colors = []
    radii = []

    # Test case 1: Horizontal pair (along X axis)
    # Demonstrates: For points of radius r, spacing = 2r makes them touch
    radius = 0.1
    spacing = 2 * radius  # KEY: spacing = 2 × radius for touching
    positions.extend(
        [
            [-spacing / 2, 0, 0],  # Left sphere
            [spacing / 2, 0, 0],  # Right sphere (should just touch left)
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
    """Create a scene demonstrating point size and spacing relationships."""
    output_path = get_examples_output_dir() / "point_spacing_example.zarr"

    aprint(f"Creating point spacing example at {output_path}")
    aprint("This example demonstrates the relationship between point size and spacing")
    aprint("KEY PRINCIPLE: For points of radius r, use spacing = 2r for touching")
    aprint("")
    aprint("Configurations included:")
    aprint("- Red pair: Horizontal touching along X axis")
    aprint("- Green pair: Vertical touching along Y axis")
    aprint("- Blue pair: Depth touching along Z axis")
    aprint("- Yellow/Magenta: Different sizes touching")
    aprint("- Cyan/Orange grid: 3x3 grid of touching spheres")

    # Create scene

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Add metadata
        scene.attrs["description"] = """
Point Size and Spacing Relationships
====================================

This scene demonstrates the fundamental mathematical relationship between
point radius and spacing for proper point packing.

KEY PRINCIPLE: spacing = 2 × radius for touching points

Test cases demonstrating this principle:
1. Red pair - horizontal touching (r=0.1, spacing=0.2)
2. Green pair - vertical touching (same radius)
3. Blue pair - depth touching (same radius)
4. Yellow/Magenta - different sizes touching (spacing = r1 + r2)
5. Cyan/Orange grid - systematic 3×3 grid with proper spacing

Educational value:
- Understand how to calculate proper point spacing
- Verify world-space sizing works correctly
- See how different radii require different spacing
- Learn point packing principles for dense visualizations
        """

        # Create the touching spheres
        positions, colors, radii = create_touching_pairs()

        # Use high sharpness for crisp edges
        sharpness = np.full(len(positions), 0.85, dtype=np.float32)

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

        add_explainer(
            scene,
            title="Radius and spacing",
            body="Pairs and a grid of spheres placed so they <strong>just "
            "touch</strong>. For equal radii, spacing <code>= 2 × radius</code>; "
            "for mixed sizes, spacing <code>= r1 + r2</code>. This verifies "
            "world-space sizing is correct.",
            observe=[
                "Red, green, and blue pairs touch along X, Y, and Z.",
                "The yellow/magenta pair touches despite different sizes.",
                "The cyan/orange 3x3 grid tiles without gaps or overlap.",
                "Contacts hold as you rotate, zoom, or change FOV.",
            ],
            observe_label="Verify",
        )

        aprint(f"\n✓ Scene created with {len(positions)} spheres")
        aprint("\n" + "=" * 60)
        aprint("VIEWING INSTRUCTIONS:")
        aprint("1. Run: luxar serve point_spacing_example.zarr")
        aprint("2. Observe the touching relationships in all test cases")
        aprint("3. Rotate view to verify touching from all angles")
        aprint("4. Notice: spacing = 2 × radius makes spheres touch perfectly")
        aprint("5. Test: Press SPACE for fullscreen - relationship maintained")
        aprint("6. Test: Use Shift+Wheel to change FOV - relationship maintained")
        aprint("")
        aprint("EDUCATIONAL NOTES:")
        aprint("- This principle is crucial for dense point clouds")
        aprint("- Proper spacing prevents overlapping and gaps")
        aprint("- Works regardless of camera, viewport, or FOV")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
