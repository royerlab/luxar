#!/usr/bin/env python3
"""Time Series 4D Example - Animated rotating spiral in 3D space over time.

This example demonstrates:
- 4D point cloud data (time + 3D space)
- Scene-level dimension definitions with proper units and ranges
- Smooth animation through time dimension
- Color coding by time for visual clarity
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Dimension, Dimensions, Scene


def create_rotating_spiral(
    n_points: int, n_times: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a rotating spiral that evolves over time.

    Args:
        n_points: Number of points in the spiral
        n_times: Number of time steps

    Returns:
        positions: Array of shape (n_points * n_times, 4) with [t, x, y, z]
        colors: Array of shape (n_points * n_times, 3) with RGB values
    """
    positions = np.zeros((n_times * n_points, 4), dtype=np.float32)
    colors = np.zeros((n_times * n_points, 3), dtype=np.uint8)

    for t in range(n_times):
        for i in range(n_points):
            idx = t * n_points + i

            # Create rotating spiral
            angle = i * 0.1 + t * 0.5  # Rotation over time
            radius = 5 + i * 0.05  # Expanding radius
            height = i * 0.3 - 15  # Vertical progression

            positions[idx] = [
                t,  # time coordinate
                radius * np.cos(angle),  # x coordinate
                radius * np.sin(angle),  # y coordinate
                height,  # z coordinate
            ]

            # Color gradient based on time (blue to red)
            colors[idx] = [
                int(255 * t / (n_times - 1)),  # Red increases with time
                50,  # Constant green
                int(255 * (1 - t / (n_times - 1))),  # Blue decreases with time
            ]

    return positions, colors


def main():
    """Create a 4D time series example with a rotating spiral."""
    output_path = Path("time_series_4d_example.zarr")
    aprint(f"Creating 4D time series example at {output_path}")

    # Parameters
    n_times = 10
    n_points = 200

    # Define scene dimensions
    dimensions = Dimensions(
        [
            Dimension(
                "t",
                unit="frame",
                range=(0, n_times - 1),
                step=1,
                display=False,
                discrete=True,
            ),
            Dimension("x", unit="units", range=(-15, 15)),
            Dimension("y", unit="units", range=(-15, 15)),
            Dimension("z", unit="units", range=(-20, 10)),
        ]
    )

    # Create scene with dimension definitions
    scene = Scene(output_path, dimensions=dimensions)

    # Generate rotating spiral data
    aprint(f"Generating {n_times} time steps with {n_points} points each...")
    positions, colors = create_rotating_spiral(n_points, n_times)

    # Add point cloud
    scene.add_points(
        "RotatingSpiral",
        positions,
        colors=colors,
        radii=np.full(len(positions), 0.15, dtype=np.float32),
    )

    scene.finalize()

    aprint(f"✓ Created 4D time series with {len(positions):,} total points")
    aprint(f"  Time steps: {n_times}")
    aprint(f"  Points per frame: {n_points}")
    aprint("  Navigate through time with keyboard controls")
    aprint(f"\nTo view: luxar serve {output_path}")


if __name__ == "__main__":
    main()
