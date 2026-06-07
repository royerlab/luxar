#!/usr/bin/env python3
"""Radius Slicing Example - Demonstrates radius-based visibility in nD.

This example demonstrates:
- How point radius affects visibility across dimension slices
- Points appearing as nD hyperspheres that intersect with viewing hyperplanes
- Larger radii making points visible across wider dimension ranges
- Visual size changes as points move away from slice center
- Practical applications for uncertainty visualization

Educational value:
- Understand the nD hypersphere slicing model for point visibility
- Learn how radius controls visibility range across dimensions
- See why larger radii make points visible in more slices
- Essential concept for working with nD point cloud data
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_test_points(time_positions: list, n_points_per_time: int = 30) -> tuple:
    """Create test points at specific time positions with varying radii.

    Args:
        time_positions: List of (time, radius, color, label) tuples
        n_points_per_time: Number of points to create at each time

    Returns:
        Tuple of (positions, colors, radii) arrays
    """
    positions = []
    colors = []
    radii = []

    for time, radius, color, label in time_positions:
        aprint(f"  Creating {label} at t={time}")

        # Create a circular arrangement of points
        angles = np.linspace(0, 2 * np.pi, n_points_per_time, endpoint=False)

        # Vary the circle size based on radius for visual clarity
        circle_radius = 2 + radius * 2

        for i, angle in enumerate(angles):
            # Create points in a circle
            x = circle_radius * np.cos(angle)
            y = circle_radius * np.sin(angle)
            z = 0  # Keep all points in the same z-plane

            positions.append([time, x, y, z])
            colors.append(color)
            radii.append(radius)

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(radii, dtype=np.float32),
    )


def main():
    """Create a demonstration of radius-based slicing behavior."""
    output_path = get_examples_output_dir() / "radius_slicing_example.zarr"
    aprint(f"Creating radius slicing example at {output_path}")

    # Define test cases with different radii
    test_cases = [
        # (time, radius, color, label)
        (0.0, 0.1, [1.0, 0.39, 0.39], "Small red points (r=0.1)"),
        (1.0, 0.3, [0.39, 1.0, 0.39], "Medium green points (r=0.3)"),
        (2.0, 0.6, [0.39, 0.59, 1.0], "Large blue points (r=0.6)"),
        (3.0, 1.0, [1.0, 0.78, 0.39], "Extra large orange points (r=1.0)"),
    ]

    # Also add a continuous line of points with gradually changing radius
    aprint("\nCreating gradient radius demonstration:")
    gradient_positions = []
    gradient_colors = []
    gradient_radii = []

    n_gradient = 50
    for i in range(n_gradient):
        t = i / (n_gradient - 1) * 4  # Time from 0 to 4
        radius = 0.1 + 0.4 * (i / n_gradient)  # Radius from 0.1 to 0.5

        # Create a vertical line of points
        gradient_positions.append([t, -8, i / 5 - 5, 0])

        # Color gradient from purple to yellow
        color = [
            0.78 + 0.22 * (i / n_gradient),  # Red: 0.78 to 1.0
            0.39 + 0.39 * (i / n_gradient),  # Green: 0.39 to 0.78
            1.0 - 0.61 * (i / n_gradient),  # Blue: 1.0 to 0.39
        ]
        gradient_colors.append(color)
        gradient_radii.append(radius)

    aprint("  Created gradient with radii from 0.1 to 0.5")

    # Create scene with proper dimensions
    dimensions = Dimensions(
        [
            Dimension(
                "w", unit="units", range=(0, 4), step=0.1, display=False, spatial=True
            ),
            Dimension("x", unit="units", range=(-10, 10)),
            Dimension("y", unit="units", range=(-10, 10)),
            Dimension("z", unit="units", range=(-2, 2)),
        ]
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)

        # Create test points
        aprint("\nCreating test point clusters:")
        positions, colors, radii = create_test_points(test_cases)

        # Combine with gradient
        all_positions = np.vstack([positions, gradient_positions])
        all_colors = np.vstack([colors, gradient_colors])
        all_radii = np.hstack([radii, gradient_radii])

        # Add sharpness variation for additional demonstration
        # Sharper points will have more defined boundaries
        sharpness = np.ones(len(all_positions), dtype=np.float32) * 0.5
        sharpness[len(positions) :] = 0.8  # Make gradient points sharper

        # Add points to scene
        scene.add_points(
            "RadiusSlicingDemo",
            all_positions,
            colors=all_colors,
            radii=all_radii,
            sharpness=sharpness,
        )

        add_explainer(
            scene,
            title="Radius-Based nD Slicing",
            body=(
                "Points act as nD hyperspheres: a point is visible only where "
                "its <code>radii</code> intersects the current slice. Step the "
                "hidden time dimension with <code>1</code> then <code>[</code>/"
                "<code>]</code>."
            ),
            observe=[
                "Larger-radius clusters stay visible across more time slices.",
                "Points shrink as the slice moves off their center.",
                "The small red cluster (r=0.1) vanishes almost immediately.",
            ],
            observe_label="Observe",
        )

        # Print detailed instructions
        aprint(f"\n✓ Created radius slicing example with {len(all_positions):,} points")
        aprint("\n" + "=" * 70)
        aprint("RADIUS-BASED SLICING DEMONSTRATION")
        aprint("=" * 70)
        aprint("\nThis example shows how point radius affects visibility in nD:")
        aprint("\n1. Start the server:")
        aprint(f"   luxar serve {output_path}")

        aprint("\n2. Navigation:")
        aprint("   - Press '1' to select time dimension")
        aprint("   - Use '[' and ']' to step through time (0.1s increments)")

        aprint("\n3. What to observe:")
        aprint("\n   CIRCULAR CLUSTERS (right side):")
        aprint("   - t=0.0: Small red points (r=0.1) - visible range ±0.1")
        aprint("   - t=1.0: Medium green points (r=0.3) - visible range ±0.3")
        aprint("   - t=2.0: Large blue points (r=0.6) - visible range ±0.6")
        aprint("   - t=3.0: XL orange points (r=1.0) - visible range ±1.0")

        aprint("\n   VERTICAL GRADIENT (left side):")
        aprint("   - Bottom: Small radius points (appear/disappear quickly)")
        aprint("   - Top: Large radius points (visible across many time slices)")

        aprint("\n4. Key concepts:")
        aprint("   - Points are nD hyperspheres intersecting the viewing hyperplane")
        aprint("   - Larger radius = visible across more dimension slices")
        aprint("   - Points shrink as slice moves away from their center")
        aprint("   - Useful for visualizing uncertainty or influence ranges")
        aprint("=" * 70)


if __name__ == "__main__":
    main()
