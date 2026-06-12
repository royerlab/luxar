#!/usr/bin/env python3
"""Dimension Navigation Example - Interactive shapes that change through dimensions.

This example demonstrates:
- Clear visual feedback for dimension navigation
- Distinct shapes at each dimension slice for easy verification
- Using discrete dimensions for frame-based navigation
- Color coding for additional visual clarity
- Proper scene-level dimension configuration

Educational value:
- Learn how dimension navigation works with distinct visual shapes
- Understand discrete frame-based dimension stepping
- See how shapes change when navigating through a hidden dimension
- Good test case for verifying nD navigation in the viewer
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_circle(radius: float = 5, n_points: int = 100) -> np.ndarray:
    """Create points arranged in a circle.

    Args:
        radius: Radius of the circle
        n_points: Number of points around the circle

    Returns:
        Array of 3D positions forming a circle in the XY plane
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(n_points)
    return np.column_stack([x, y, z]).astype(np.float32)


def create_square(size: float = 10, n_points: int = 100) -> np.ndarray:
    """Create points arranged in a square.

    Args:
        size: Side length of the square
        n_points: Total number of points (distributed across 4 edges)

    Returns:
        Array of 3D positions forming a square in the XY plane
    """
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

    return np.array(positions, dtype=np.float32)


def create_triangle(size: float = 10, n_points: int = 90) -> np.ndarray:
    """Create points arranged in an equilateral triangle.

    Args:
        size: Side length of the triangle
        n_points: Total number of points (distributed across 3 edges)

    Returns:
        Array of 3D positions forming an equilateral triangle in the XY plane
    """
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

    return np.array(positions, dtype=np.float32)


def create_star(size: float = 5, n_points: int = 100, n_spikes: int = 5) -> np.ndarray:
    """Create points arranged in a star pattern.

    Args:
        size: Outer radius of the star
        n_points: Total number of points
        n_spikes: Number of star points/spikes

    Returns:
        Array of 3D positions forming a star in the XY plane
    """
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

    return np.array(positions, dtype=np.float32)


def create_cross(size: float = 10, n_points: int = 100) -> np.ndarray:
    """Create points arranged in a cross/plus pattern.

    Args:
        size: Total length of each cross arm
        n_points: Total number of points (split between horizontal and vertical)

    Returns:
        Array of 3D positions forming a cross in the XY plane
    """
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

    return np.vstack([horizontal, vertical]).astype(np.float32)


def main():
    """Create an interactive dimension navigation example."""
    output_path = get_examples_output_dir() / "dimension_navigation_example.luxar.zarr"
    aprint(f"Creating dimension navigation example at {output_path}")

    # Define shapes and their properties
    shape_configs = [
        ("Circle", create_circle(), [1.0, 0.2, 0.2]),  # Red
        ("Square", create_square(), [0.2, 1.0, 0.2]),  # Green
        ("Triangle", create_triangle(), [0.2, 0.39, 1.0]),  # Blue
        ("Star", create_star(), [1.0, 0.78, 0.2]),  # Gold
        ("Cross", create_cross(), [1.0, 0.2, 1.0]),  # Magenta
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

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)

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
        colors = np.vstack(all_colors).astype(np.float32)
        radii = np.hstack(all_radii)

        # Add to scene
        scene.add_points(
            "ShapeSequence",
            positions,
            colors=colors,
            radii=radii,
            sharpness=np.full(
                len(positions), 0.6, dtype=np.float32
            ),  # Crisp edges for clarity
        )

        # Per-frame colour legend (bottom-left): the frame dimension is
        # *sliced* so only one shape shows at a time — this persistent legend
        # names the full shape+colour sequence so you know what each step
        # reveals. Each line is tinted to match its frame's points.
        for frame_idx, (name, _shape, color) in enumerate(shape_configs):
            r, g, b = (int(round(c * 255)) for c in color)
            scene.add_text(
                f"● frame {frame_idx}: {name.lower()}",
                position=(0.02, 0.72 + 0.05 * frame_idx),
                font_size=0.022,
                font="mono",
                color=f"rgb({r},{g},{b})",
                stroke_color="black",
                stroke_width=0.0018,
            )

        add_explainer(
            scene,
            title="Dimension Navigation",
            body=(
                "The hidden discrete <code>frame</code> dimension selects one "
                "of five distinct shapes; stepping it swaps the whole shape "
                "and its colour at once."
            ),
            observe=[
                "Press <code>1</code> for frame, then <code>[</code> / "
                "<code>]</code> to step.",
                "Sequence: red circle, green square, blue triangle, gold "
                "star, magenta cross.",
                "Exactly one shape is visible per frame.",
            ],
        )

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
