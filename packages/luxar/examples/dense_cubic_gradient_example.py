#!/usr/bin/env python3
"""Dense Cubic Gradient Example - Million-point cube with depth-based colors.

This example demonstrates:
- High-density visualization (1,000,000 points in 100×100×100 cubic lattice)
- Beautiful depth-based color gradients for perspective visualization
- Performance testing with dense regular grids
- Crystalline/volumetric structures with sharp disc-like points
- Background star field (500k points) providing depth context
- Creating evenly-spaced 3D grids using np.meshgrid

Educational value:
- Shows how to generate large regular grids efficiently
- Demonstrates depth perception through color gradients
- Illustrates performance characteristics with million-point datasets
- Perfect for testing rendering quality, camera controls, and navigation
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_cubic_array(
    grid_size: int = 100,
    spacing: float = 0.5,
    radius: float = 0.05,
    sharpness: float = 0.9,
) -> tuple:
    """Create a dense cubic lattice with depth-based color gradient.

    This function generates a perfect cubic grid of points with a beautiful
    warm-to-cool color gradient based on depth (Z coordinate), creating
    a stunning visualization of perspective and depth.

    Args:
        grid_size: Number of points along each axis (100 → 1 million points)
        spacing: Distance between adjacent points in the grid
        radius: Radius of each point (small creates disc-like appearance)
        sharpness: Sharpness value (high creates sharp edges)

    Returns:
        Tuple of (positions, colors, radii, sharpness) arrays
    """
    aprint(f"Generating {grid_size}³ = {grid_size**3:,} points...")

    # Create 3D grid coordinates
    # Use linspace to ensure even spacing
    axis = np.linspace(
        -(grid_size - 1) * spacing / 2,
        (grid_size - 1) * spacing / 2,
        grid_size,
        dtype=np.float32,
    )

    # Create meshgrid for all 3 dimensions
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")

    # Flatten to create position array
    positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Create depth-based color gradient for better visualization
    # Normalize z-coordinates to [0, 1]
    z_normalized = (z.ravel() - z.min()) / (z.max() - z.min() + 1e-8)

    # Color scheme: Near points are warm (red/yellow), far points are cool (blue/cyan)
    colors = np.zeros((positions.shape[0], 3), dtype=np.float32)

    # Red channel: high for near points
    colors[:, 0] = 1.0 - z_normalized * 0.7  # Red: 1.0 to 0.3

    # Green channel: peaks in middle distance
    colors[:, 1] = 0.3 + 0.4 * np.sin(z_normalized * np.pi)  # Green: varies

    # Blue channel: high for far points
    colors[:, 2] = 0.2 + z_normalized * 0.8  # Blue: 0.2 to 1.0

    # Add slight color variation based on position for visual interest
    # This helps distinguish individual points
    position_hash = np.sin(x.ravel() * 12.345) * np.cos(y.ravel() * 67.89)
    color_variation = position_hash * 0.1  # ±0.1 variation

    colors[:, 0] = np.clip(colors[:, 0] + color_variation, 0, 1)
    colors[:, 1] = np.clip(colors[:, 1] - color_variation * 0.5, 0, 1)
    colors[:, 2] = np.clip(colors[:, 2] + color_variation * 0.5, 0, 1)

    # All points have the same small radius for disc-like appearance
    radii = np.full(positions.shape[0], radius, dtype=np.float32)

    # All points have high sharpness for sharp edges
    sharpness_array = np.full(positions.shape[0], sharpness, dtype=np.float32)

    return positions, colors, radii, sharpness_array


def main():
    """Create a dense cubic gradient visualization scene."""
    output_path = get_examples_output_dir() / "dense_cubic_gradient_example.zarr"

    aprint(f"Creating dense cubic gradient at {output_path}")
    aprint("This example creates a stunning 100×100×100 grid visualization")
    aprint("")
    aprint("Scene features:")
    aprint("- 1,000,000 sharp disc-like points in perfect cubic lattice")
    aprint("- Beautiful depth gradient: warm (near) → cool (far)")
    aprint("- Plus 500,000 background stars for depth context")
    aprint("- Optimized spacing preventing overlap while maintaining density")

    # Create scene
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Add metadata
        scene.attrs["description"] = """
Dense Cubic Gradient Visualization
===================================

A stunning high-density visualization featuring:

- 1,000,000 points in a perfect 100×100×100 cubic lattice
- Beautiful depth-based color gradient:
  * Warm colors (red/yellow) for points near the camera
  * Cool colors (blue/cyan) for points far from the camera
- Sharp disc-like points creating a crystalline appearance
- 500,000 background stars providing depth context
- Additive blending for realistic light accumulation

Educational features:
- Demonstrates high-density visualization techniques
- Shows depth perception through color gradients
- Illustrates efficient grid generation with np.meshgrid
- Perfect for testing camera fly controls and navigation
- Good performance benchmark (1M+ points)

Viewing tips:
- Press 'C' to switch to fly controls
- Use WASD to fly through the cubic structure
- Use mouse to look around
- Observe how color gradient enhances depth perception
- Notice the crystalline lattice structure

Performance notes:
- This is a stress test with 1.5M total points
- GPU performance dependent
- Excellent test for rendering optimization
        """

        # Create the main cubic array
        aprint("\nGenerating cubic array...")
        positions, colors, radii, sharpness = create_cubic_array(
            grid_size=100,  # 100x100x100 grid
            spacing=0.5,  # Distance between points
            radius=0.05,  # Small radius for disc-like appearance
            sharpness=0.9,  # High sharpness for sharp edges
        )

        # Add main points
        scene.add_points(
            "CubicArray",
            positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            opacity=1.0,
            blending_mode="additive",  # Use additive blending for points
        )

        aprint(f"✓ Added {len(positions):,} points to cubic array")

        aprint("Creating background star field...")
        n_stars = 500000
        # Seed so the random star field is reproducible run-to-run.
        np.random.seed(0)
        star_positions = np.random.uniform(-500, 500, (n_stars, 3)).astype(np.float32)

        # Variable star colors and sizes
        star_colors = []
        star_radii = []
        for _ in range(n_stars):
            # Random star colors (white to yellow to red)
            temp = np.random.random()
            if temp < 0.3:
                color = [1.0, 1.0, 1.0]  # White
            elif temp < 0.6:
                color = [1.0, 1.0, 0.78]  # Yellow-white
            elif temp < 0.8:
                color = [1.0, 0.86, 0.59]  # Yellow
            else:
                color = [1.0, 0.71, 0.47]  # Orange-red

            star_colors.append(color)
            star_radii.append(np.random.uniform(0.01, 0.05))

        scene.add_points(
            "BackgroundStars",
            star_positions,
            colors=np.array(star_colors, dtype=np.float32),
            radii=np.array(star_radii, dtype=np.float32),
            sharpness=0.5,
            opacity=0.4,
            gamma=1.0,
            blending_mode="normal",
        )

        # Explainer card describing the million-point stress test.
        add_explainer(
            scene,
            title="Dense Cubic Gradient",
            body=(
                "A 100x100x100 cubic lattice (1,000,000 points) colored by a "
                "depth gradient, plus a 500,000-point background star field "
                "for spatial context."
            ),
            observe=[
                "The lattice runs warm (near) to cool (far) along Z.",
                "Points stay disc-like and crisp at high <code>sharpness</code>.",
                "Press <code>V</code> to cycle to fly mode, then WASD through the cube.",
            ],
            observe_label="Notice",
        )

        # Print statistics
        aprint("\n✓ Scene created successfully!")
        aprint(f"  Total points: {len(positions):,}")
        aprint("  Grid dimensions: 100×100×100")
        aprint("  Point spacing: 0.5 units")
        aprint("  Point radius: 0.05 units")
        aprint("  Point sharpness: 0.9 (sharp edges)")

        aprint("\n" + "=" * 60)
        aprint("VIEWING INSTRUCTIONS:")
        aprint("1. Run: luxar serve dense_cubic_gradient_example.zarr --viewer")
        aprint("2. Observe the beautiful depth gradient (warm → cool)")
        aprint("3. Press 'C' to switch to fly controls")
        aprint("4. Use WASD to fly through the crystalline cube")
        aprint("5. Use mouse to look around inside the structure")
        aprint("")
        aprint("COLOR GUIDE:")
        aprint("- Red/Yellow points: Near to camera (front of cube)")
        aprint("- Green points: Middle distance")
        aprint("- Blue/Cyan points: Far from camera (back of cube)")
        aprint("- Background stars: Depth context and orientation")
        aprint("")
        aprint("EDUCATIONAL NOTES:")
        aprint("- Notice how color gradient enhances depth perception")
        aprint("- Observe the sharp crystalline disc structure")
        aprint("- This demonstrates million-point rendering capabilities")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
