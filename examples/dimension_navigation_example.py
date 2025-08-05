#!/usr/bin/env python3
"""Dimension Navigation Example - Interactive shapes that change through dimensions.

This example demonstrates:
- Clear visual feedback for dimension navigation
- Distinct shapes at each dimension slice for easy verification
- Using discrete dimensions for frame-based navigation
- Color coding for additional visual clarity
- Proper scene-level dimension configuration
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Dimension, Dimensions, Scene


def create_circle(radius: float = 5, n_points: int = 100) -> np.ndarray:
    """Create points arranged in a circle."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(n_points)
    return np.column_stack([x, y, z])


def create_square(size: float = 10, n_points: int = 100) -> np.ndarray:
    """Create points arranged in a square."""
    points_per_side = n_points // 4
    positions = []

    # Create four edges
    edges = [
        (
            np.linspace(-size / 2, size / 2, points_per_side),
            np.full(points_per_side, size / 2),
        ),  # Top
        (
            np.full(points_per_side, size / 2),
            np.linspace(size / 2, -size / 2, points_per_side),
        ),  # Right
        (
            np.linspace(size / 2, -size / 2, points_per_side),
            np.full(points_per_side, -size / 2),
        ),  # Bottom
        (
            np.full(points_per_side, -size / 2),
            np.linspace(-size / 2, size / 2, points_per_side),
        ),  # Left
    ]

    for x_coords, y_coords in edges:
        z_coords = np.zeros(points_per_side)
        positions.extend(zip(x_coords, y_coords, z_coords))

    return np.array(positions)


def create_triangle(size: float = 10, n_points: int = 90) -> np.ndarray:
    """Create points arranged in an equilateral triangle."""
    points_per_side = n_points // 3
    positions = []

    # Define vertices of equilateral triangle
    vertices = np.array(
        [
            [0, size * np.sqrt(3) / 3, 0],  # Top
            [-size / 2, -size * np.sqrt(3) / 6, 0],  # Bottom left
            [size / 2, -size * np.sqrt(3) / 6, 0],  # Bottom right
        ]
    )

    # Create edges between vertices
    for i in range(3):
        v1, v2 = vertices[i], vertices[(i + 1) % 3]
        for j in range(points_per_side):
            t = j / points_per_side
            pos = (1 - t) * v1 + t * v2
            positions.append(pos)

    return np.array(positions)


def create_star(size: float = 5, n_points: int = 100, n_spikes: int = 5) -> np.ndarray:
    """Create points arranged in a star pattern."""
    positions = []
    points_per_spike = n_points // (n_spikes * 2)

    for i in range(n_spikes):
        # Outer and inner angles for star shape
        outer_angle = i * 2 * np.pi / n_spikes
        inner_angle = outer_angle + np.pi / n_spikes

        # Outer point
        outer_x = size * np.cos(outer_angle)
        outer_y = size * np.sin(outer_angle)

        # Inner point (closer to center)
        inner_x = (size * 0.4) * np.cos(inner_angle)
        inner_y = (size * 0.4) * np.sin(inner_angle)

        # Line from center to outer point
        for j in range(points_per_spike):
            t = j / points_per_spike
            positions.append([outer_x * t, outer_y * t, 0])

        # Line from outer to inner point
        for j in range(points_per_spike):
            t = j / points_per_spike
            x = (1 - t) * outer_x + t * inner_x
            y = (1 - t) * outer_y + t * inner_y
            positions.append([x, y, 0])

    return np.array(positions)


def create_cross(size: float = 10, n_points: int = 100) -> np.ndarray:
    """Create points arranged in a cross/plus pattern."""
    half_points = n_points // 2

    # Horizontal line
    h_x = np.linspace(-size / 2, size / 2, half_points)
    h_y = np.zeros(half_points)
    h_z = np.zeros(half_points)

    # Vertical line
    v_x = np.zeros(half_points)
    v_y = np.linspace(-size / 2, size / 2, half_points)
    v_z = np.zeros(half_points)

    horizontal = np.column_stack([h_x, h_y, h_z])
    vertical = np.column_stack([v_x, v_y, v_z])

    return np.vstack([horizontal, vertical])


def main():
    """Create an interactive dimension navigation example."""
    output_path = Path("dimension_navigation_example.zarr")
    aprint(f"Creating dimension navigation example at {output_path}")

    # Define shapes and their properties
    shape_configs = [
        ("Circle", create_circle(), [255, 50, 50]),  # Red
        ("Square", create_square(), [50, 255, 50]),  # Green
        ("Triangle", create_triangle(), [50, 100, 255]),  # Blue
        ("Star", create_star(), [255, 200, 50]),  # Gold
        ("Cross", create_cross(), [255, 50, 255]),  # Magenta
    ]

    # Create scene with proper dimensions
    n_frames = len(shape_configs)
    dimensions = Dimensions(
        [
            Dimension(
                "frame",
                unit="shape",
                range=(0, n_frames - 1),
                step=1,
                display=False,
                discrete=True,
                description="Shape selector",
            ),
            Dimension("x", unit="units", range=(-6, 6)),
            Dimension("y", unit="units", range=(-6, 6)),
            Dimension("z", unit="units", range=(-1, 1)),
        ]
    )

    scene = Scene(output_path, dimensions=dimensions)

    # Combine all shapes with frame dimension
    all_positions = []
    all_colors = []
    all_radii = []

    aprint("Creating shapes:")
    for frame_idx, (name, shape_3d, color) in enumerate(shape_configs):
        n_points = len(shape_3d)

        # Add frame dimension to positions [frame, x, y, z]
        frame_column = np.full((n_points, 1), frame_idx)
        positions_4d = np.hstack([frame_column, shape_3d])
        all_positions.append(positions_4d)

        # Set colors
        color_array = np.tile(color, (n_points, 1))
        all_colors.append(color_array)

        # Vary radius slightly for visual interest
        radii = np.full(n_points, 0.15 + 0.05 * np.sin(frame_idx), dtype=np.float32)
        all_radii.append(radii)

        aprint(f"  Frame {frame_idx}: {name} ({n_points} points) - RGB{color}")

    # Stack all arrays
    positions = np.vstack(all_positions).astype(np.float32)
    colors = np.vstack(all_colors).astype(np.uint8)
    radii = np.hstack(all_radii)

    # Add to scene
    scene.add_points(
        "ShapeSequence",
        positions,
        colors=colors,
        radii=radii,
        sharpness=np.full(
            len(positions), 3.0, dtype=np.float32
        ),  # Sharp edges for clarity
    )

    scene.finalize()

    # Print instructions
    aprint(
        f"\n✓ Created dimension navigation example with {len(positions):,} total points"
    )
    aprint("\n" + "=" * 60)
    aprint("DIMENSION NAVIGATION INSTRUCTIONS")
    aprint("=" * 60)
    aprint("1. Start the data server:")
    aprint(f"   luxar serve {output_path}")
    aprint("\n2. Navigation controls:")
    aprint("   - Press '1' to select the frame dimension")
    aprint("   - Press '[' to go to previous shape")
    aprint("   - Press ']' to go to next shape")
    aprint("\n3. Expected sequence:")
    aprint("   - Frame 0: RED CIRCLE")
    aprint("   - Frame 1: GREEN SQUARE")
    aprint("   - Frame 2: BLUE TRIANGLE")
    aprint("   - Frame 3: GOLD STAR")
    aprint("   - Frame 4: MAGENTA CROSS")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
