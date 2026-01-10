#!/usr/bin/env python
"""
Lines Basic Example

Creates a simple 3D scene with lines to test line rendering in the viewer.
Demonstrates:
- Basic line creation with segments
- Polylines (connected vertices)
- Per-vertex colors and widths
- Varying sharpness values

Run with:
    hatch run python packages/luxar/examples/lines_basic_example.py
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir

# Output path
output_path = get_examples_output_dir() / "lines_basic_example.zarr"


def create_spiral_line(n_points: int = 100, radius: float = 5.0, height: float = 10.0):
    """Create a 3D spiral polyline."""
    t = np.linspace(0, 4 * np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, height, n_points)
    return np.column_stack([x, y, z]).astype(np.float32)


def create_grid_lines(size: float = 10.0, n_lines: int = 10):
    """Create a grid of line segments on the XY plane."""
    vertices = []

    # Horizontal lines
    for i in range(n_lines + 1):
        y = -size / 2 + i * size / n_lines
        vertices.append([-size / 2, y, 0])
        vertices.append([size / 2, y, 0])

    # Vertical lines
    for i in range(n_lines + 1):
        x = -size / 2 + i * size / n_lines
        vertices.append([x, -size / 2, 0])
        vertices.append([x, size / 2, 0])

    return np.array(vertices, dtype=np.float32)


def create_star_burst(
    n_rays: int = 20, inner_radius: float = 1.0, outer_radius: float = 8.0
):
    """Create a star burst pattern of line segments."""
    vertices = []
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    for angle in angles:
        # Inner point
        x_in = inner_radius * np.cos(angle)
        y_in = inner_radius * np.sin(angle)
        # Outer point
        x_out = outer_radius * np.cos(angle)
        y_out = outer_radius * np.sin(angle)

        vertices.append([x_in, y_in, 0])
        vertices.append([x_out, y_out, 0])

    return np.array(vertices, dtype=np.float32)


def main():
    """Create lines example scene."""
    aprint(f"Creating lines example at {output_path}")

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # === 1. Spiral polyline with rainbow colors ===
        spiral_vertices = create_spiral_line(n_points=200, radius=3.0, height=8.0)
        n_spiral = len(spiral_vertices)

        # Rainbow colors along the spiral
        t = np.linspace(0, 1, n_spiral)
        spiral_colors = np.zeros((n_spiral, 3), dtype=np.float32)
        spiral_colors[:, 0] = np.sin(t * np.pi * 2) * 0.5 + 0.5  # Red
        spiral_colors[:, 1] = np.sin(t * np.pi * 2 + np.pi * 2 / 3) * 0.5 + 0.5  # Green
        spiral_colors[:, 2] = np.sin(t * np.pi * 2 + np.pi * 4 / 3) * 0.5 + 0.5  # Blue

        # Varying width along the spiral
        spiral_widths = 0.1 + 0.2 * np.sin(t * np.pi * 4) ** 2
        spiral_widths = spiral_widths.astype(np.float32)

        # Varying sharpness
        spiral_sharpness = 0.5 + 0.5 * np.cos(t * np.pi * 2)
        spiral_sharpness = spiral_sharpness.astype(np.float32)

        scene.add_lines(
            "rainbow_spiral",
            vertices=spiral_vertices,
            widths=spiral_widths,
            colors=spiral_colors,
            sharpness=spiral_sharpness,
            line_type="polyline",
        )

        # === 2. Grid lines (segments) ===
        grid_vertices = create_grid_lines(size=12.0, n_lines=12)
        n_grid = len(grid_vertices)

        # Move grid down and make it blue-ish
        grid_vertices[:, 2] = -2.0
        grid_colors = np.full((n_grid, 3), [0.2, 0.4, 0.8], dtype=np.float32)

        scene.add_lines(
            "floor_grid",
            vertices=grid_vertices,
            widths=0.05,  # Uniform width
            colors=grid_colors,
            sharpness=1.0,  # Sharp edges
            line_type="segments",
        )

        # === 3. Star burst (segments) ===
        star_vertices = create_star_burst(n_rays=30, inner_radius=0.5, outer_radius=6.0)
        n_star = len(star_vertices)

        # Move star up
        star_vertices[:, 2] = 10.0

        # Gold/orange colors
        star_colors = np.zeros((n_star, 3), dtype=np.float32)
        star_colors[:, 0] = 1.0  # Red
        star_colors[:, 1] = 0.7  # Green
        star_colors[:, 2] = 0.2  # Blue

        # Varying widths - thicker at center, thinner at edges
        star_widths = np.zeros(n_star, dtype=np.float32)
        star_widths[0::2] = 0.15  # Inner points (even indices)
        star_widths[1::2] = 0.05  # Outer points (odd indices)

        scene.add_lines(
            "star_burst",
            vertices=star_vertices,
            widths=star_widths,
            colors=star_colors,
            sharpness=0.5,  # Soft edges
            line_type="segments",
        )

        # === 4. Loop (closed polyline) ===
        n_loop = 50
        loop_angles = np.linspace(0, 2 * np.pi, n_loop, endpoint=False)
        loop_radius = 2.0 + 0.5 * np.sin(5 * loop_angles)  # Flower shape
        loop_vertices = np.column_stack(
            [
                loop_radius * np.cos(loop_angles),
                loop_radius * np.sin(loop_angles),
                np.full(n_loop, 12.0),  # Z = 12
            ]
        ).astype(np.float32)

        # Magenta/pink colors
        loop_colors = np.full((n_loop, 3), [1.0, 0.2, 0.8], dtype=np.float32)

        scene.add_lines(
            "flower_loop",
            vertices=loop_vertices,
            widths=0.12,
            colors=loop_colors,
            sharpness=0.7,
            line_type="loop",
        )

        aprint("Created scene with 4 lines objects:")
        aprint(f"  - rainbow_spiral: {len(spiral_vertices)} vertices (polyline)")
        aprint(f"  - floor_grid: {len(grid_vertices) // 2} segments")
        aprint(f"  - star_burst: {len(star_vertices) // 2} segments")
        aprint(f"  - flower_loop: {len(loop_vertices)} vertices (loop)")

    aprint(f"Scene saved to: {output_path}")
    aprint(f"\nTo view: luxar serve --viewer {output_path}")


if __name__ == "__main__":
    main()
