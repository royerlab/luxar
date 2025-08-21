#!/usr/bin/env python3
"""
Comprehensive example showcasing the point sharpness feature in Luxar.

This script generates several point cloud examples with varying sharpness values
to demonstrate how the sharpness parameter controls the edge falloff of points.
Low sharpness values create soft, glowing points while high values create sharp edges.
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import LuxarZarrCompiler


def create_sharpness_gradient_example(scene, n_points: int = 5000) -> None:
    """Create a grid showing gradual sharpness transition."""
    aprint("Creating sharpness gradient example...")

    # Create a grid of points
    grid_size = int(np.sqrt(n_points))
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-5, 5, grid_size)
    xx, yy = np.meshgrid(x, y)

    positions = np.column_stack([xx.flatten(), yy.flatten(), np.zeros(xx.size)]).astype(
        np.float32
    )

    # Sharpness increases from left to right
    normalized_x = (positions[:, 0] + 10) / 20  # 0 to 1
    sharpness = 0.5 + normalized_x * 9.5  # 0.5 to 10.0
    sharpness = sharpness.astype(np.float32)

    # All points same size for fair comparison
    radii = np.full(positions.shape[0], 0.3, dtype=np.float32)

    # Color gradient to visualize sharpness
    colors = np.zeros((positions.shape[0], 3), dtype=np.float32)
    colors[:, 0] = normalized_x  # Red increases
    colors[:, 2] = 1 - normalized_x  # Blue decreases

    # Offset vertically
    positions[:, 1] += 10

    scene.add_points(
        "SharpnessGradient", positions, colors, radii=radii, sharpness=sharpness
    )


def create_sharpness_comparison_example(scene) -> None:
    """Create rows of points with different fixed sharpness values."""
    aprint("Creating sharpness comparison example...")

    n_points_per_row = 20
    sharpness_values = [0.5, 1.0, 2.0, 4.0, 10.0]
    labels = [
        "Very Soft (0.5)",
        "Linear (1.0)",
        "Quadratic (2.0)",
        "Quartic (4.0)",
        "Very Sharp (10.0)",
    ]

    for i, (sharp_val, label) in enumerate(zip(sharpness_values, labels)):
        # Create a row of points
        x = np.linspace(-8, 8, n_points_per_row)
        y = np.full(n_points_per_row, i * 2.5 - 5)
        z = np.zeros(n_points_per_row)

        positions = np.column_stack([x, y, z]).astype(np.float32)

        # All points have the same sharpness value
        sharpness = np.full(n_points_per_row, sharp_val, dtype=np.float32)
        radii = np.full(n_points_per_row, 0.4, dtype=np.float32)

        # Different color for each row
        hue = i / len(sharpness_values)
        if hue < 1 / 6:
            r, g, b = 1.0, hue * 6, 0
        elif hue < 2 / 6:
            r, g, b = 2 - hue * 6, 1.0, 0
        elif hue < 3 / 6:
            r, g, b = 0, 1.0, hue * 6 - 2
        elif hue < 4 / 6:
            r, g, b = 0, 4 - hue * 6, 1.0
        else:
            r, g, b = hue * 6 - 4, 0, 1.0

        colors = np.tile([r, g, b], (n_points_per_row, 1)).astype(np.float32)

        # Offset to the right
        positions[:, 0] += 15

        scene.add_points(
            f"Sharpness_{label}", positions, colors, radii=radii, sharpness=sharpness
        )


def create_mixed_sharpness_example(scene, n_points: int = 10000) -> None:
    """Create a sphere with mixed sharpness values."""
    aprint("Creating mixed sharpness cloud example...")

    rng = np.random.default_rng(789)

    # Generate points in a sphere
    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = np.arccos(rng.uniform(-1, 1, n_points))
    r = rng.uniform(3, 5, n_points)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    positions = np.column_stack([x, y, z]).astype(np.float32)

    # Mix of sharpness values - create clusters
    sharpness = np.zeros(n_points, dtype=np.float32)
    # 1/3 soft points
    sharpness[: n_points // 3] = rng.uniform(0.5, 1.5, n_points // 3)
    # 1/3 medium points
    sharpness[n_points // 3 : 2 * n_points // 3] = rng.uniform(2.0, 4.0, n_points // 3)
    # 1/3 sharp points
    sharpness[2 * n_points // 3 :] = rng.uniform(
        6.0, 10.0, n_points - 2 * n_points // 3
    )

    # Shuffle to mix them
    rng.shuffle(sharpness)

    # Size varies with sharpness (sharp points are smaller)
    radii = 0.4 - (sharpness - 0.5) * 0.03  # Larger soft points, smaller sharp points
    radii = np.clip(radii, 0.1, 0.4).astype(np.float32)

    # Color based on sharpness
    normalized_sharp = (sharpness - 0.5) / 9.5
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 0] = normalized_sharp  # Red for sharp
    colors[:, 1] = 0.5 * (1 - np.abs(normalized_sharp - 0.5) * 2)  # Green for medium
    colors[:, 2] = 1 - normalized_sharp  # Blue for soft

    # Offset down and left
    positions[:, 1] -= 10
    positions[:, 0] -= 15

    scene.add_points(
        "MixedSharpnessCloud", positions, colors, radii=radii, sharpness=sharpness
    )


def create_sharpness_wave_example(scene, n_points: int = 4000) -> None:
    """Create a wave pattern where sharpness varies sinusoidally."""
    aprint("Creating sharpness wave example...")

    # Create a wave grid
    grid_size = int(np.sqrt(n_points))
    x = np.linspace(-10, 10, grid_size)
    z = np.linspace(-5, 5, grid_size)
    xx, zz = np.meshgrid(x, z)

    # Wave height based on distance from origin
    distance = np.sqrt(xx**2 + zz**2)
    y = 2 * np.sin(distance * 0.5)

    positions = np.column_stack([xx.flatten(), y.flatten(), zz.flatten()]).astype(
        np.float32
    )

    # Sharpness varies with the wave
    sharpness = 5.0 + 4.5 * np.sin(distance.flatten() * 0.5)
    sharpness = sharpness.astype(np.float32)

    # Radii also vary slightly
    radii = 0.2 + 0.1 * np.cos(distance.flatten() * 0.5)
    radii = radii.astype(np.float32)

    # Colors based on height
    normalized_y = (y.flatten() + 2) / 4
    colors = np.zeros((positions.shape[0], 3), dtype=np.float32)
    colors[:, 0] = normalized_y
    colors[:, 1] = 1 - np.abs(normalized_y - 0.5) * 2
    colors[:, 2] = 1 - normalized_y

    # Offset down
    positions[:, 1] -= 20

    scene.add_points(
        "SharpnessWave", positions, colors, radii=radii, sharpness=sharpness
    )


def main():
    """Run the sharpness feature example."""
    output_path = Path(__file__).parent / "sharpness_showcase_example.zarr"

    aprint(f"Creating sharpness example scene at {output_path}")
    aprint("\nThis example showcases the per-point sharpness feature:")
    aprint("- Gradient: Smooth transition from soft (0.5) to sharp (10.0)")
    aprint("- Comparison: Fixed sharpness values side by side")
    aprint("- Mixed cloud: Sphere with varying sharpness values")
    aprint("- Wave: Sinusoidal sharpness variation")

    # Create scene
    
    with LuxarZarrCompiler(output_path) as compiler:

        scene = compiler.create_scene()

        # Add all examples
        create_sharpness_gradient_example(scene)
        create_sharpness_comparison_example(scene)
        create_mixed_sharpness_example(scene)
        create_sharpness_wave_example(scene)

        # Finalize

        aprint(f"\n✓ Demo scene created successfully at {output_path}")
        aprint("\nTo view the example:")
        aprint("1. Start the viewer: cd packages/luxar-player && npm run dev")
        aprint(f"2. Serve the data: luxar serve {output_path}")
        aprint("3. Open http://localhost:5173 in your browser")
        aprint("\nWhat to look for:")
        aprint("- Top: Gradient from soft glowing points (left) to sharp points (right)")
        aprint("- Right: 5 rows showing different fixed sharpness values")
        aprint(
            "- Bottom left: Mixed cloud with soft blue, medium green, and sharp red points"
        )
        aprint("- Bottom: Wave pattern with varying sharpness creating visual depth")


if __name__ == "__main__":
    main()
