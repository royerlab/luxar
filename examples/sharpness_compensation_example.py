#!/usr/bin/env python3
"""
Example demonstrating sharpness compensation for consistent point sizes.

This example shows how the shader compensation maintains consistent apparent
point sizes across different sharpness values. All points have the same radius
but varying sharpness, and should appear roughly the same size.
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Scene


def create_compensation_demo():
    """Create a grid of points with same radius but different sharpness values."""
    output_path = Path("sharpness_compensation_example.zarr")

    # Create scene
    scene = Scene(output_path)

    # Grid parameters
    rows = 2
    cols = 6
    spacing = 3.0

    # Sharpness values to test
    sharpness_values = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

    # Create two rows: one with compensation (top), one without (if we had old shader)
    all_positions = []
    all_colors = []
    all_radii = []
    all_sharpness = []

    for row in range(rows):
        for col, sharp in enumerate(sharpness_values):
            # Position
            x = (col - cols / 2) * spacing
            y = (row - rows / 2) * spacing * 2
            z = 0

            # Generate a small cluster of points
            n_points = 100
            positions = np.random.randn(n_points, 3) * 0.3
            positions[:, 0] += x
            positions[:, 1] += y
            positions[:, 2] += z

            # Color: gradient from blue (low sharpness) to red (high sharpness)
            t = col / (cols - 1)
            color = np.array([t, 0.2, 1 - t]) * 255
            colors = np.tile(color, (n_points, 1))

            # All points have the same radius
            radii = np.full(n_points, 0.5, dtype=np.float32)

            # Varying sharpness
            sharpness = np.full(n_points, sharp, dtype=np.float32)

            all_positions.append(positions)
            all_colors.append(colors)
            all_radii.append(radii)
            all_sharpness.append(sharpness)

    # Combine all arrays
    positions = np.vstack(all_positions).astype(np.float32)
    colors = np.vstack(all_colors).astype(np.uint8)
    radii = np.hstack(all_radii)
    sharpness = np.hstack(all_sharpness)

    # Add to scene
    scene.add_points(
        "SharpnessCompensationTest",
        positions=positions,
        colors=colors,
        radii=radii,
        sharpness=sharpness,
    )

    # Add labels as single points
    label_positions = []
    label_colors = []
    label_radii = []
    label_sharpness = []

    for col, sharp in enumerate(sharpness_values):
        x = (col - cols / 2) * spacing
        y = -4.0
        label_positions.append([x, y, 0])
        label_colors.append([255, 255, 255])
        label_radii.append(0.1)
        label_sharpness.append(2.0)

    scene.add_points(
        "Labels",
        positions=np.array(label_positions, dtype=np.float32),
        colors=np.array(label_colors, dtype=np.uint8),
        radii=np.array(label_radii, dtype=np.float32),
        sharpness=np.array(label_sharpness, dtype=np.float32),
    )

    scene.finalize()

    aprint(f"✓ Created sharpness compensation test at {output_path}")
    aprint("\nSharpness values (left to right):")
    aprint(" | ".join([f"{s:4.1f}" for s in sharpness_values]))
    aprint("\nExpected result:")
    aprint("- All point clusters should appear roughly the same size")
    aprint("- Colors: Blue (soft, left) → Red (sharp, right)")
    aprint("\nTo view: luxar serve sharpness_compensation_example.zarr")
    return output_path


if __name__ == "__main__":
    create_compensation_demo()
