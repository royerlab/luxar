#!/usr/bin/env python3
"""DOF Cubic Array Example - Test depth-of-field with a dense 3D grid.

This example demonstrates:
- Creating a 100x100x100 cubic array of small sharp disc-like points
- Optimized for testing depth-of-field blur effects
- Color gradient indicating depth for better visualization
- Proper spacing to avoid overlapping while maintaining density

This scene is specifically designed for testing DOF post-processing effects,
with high point density and sharp edges that clearly show blur effects.
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import LuxarZarrCompiler


def create_cubic_array(
    grid_size: int = 100,
    spacing: float = 0.5,
    radius: float = 0.05,
    sharpness: float = 10.0,
) -> tuple:
    """Create a cubic array of sharp disc-like points.

    Args:
        grid_size: Number of points along each axis (100 for 100x100x100)
        spacing: Distance between adjacent points
        radius: Radius of each point (small for disc-like appearance)
        sharpness: Sharpness value (high for sharp edges)

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
    """Create a DOF test cubic array scene."""
    output_path = Path(__file__).parent / "dof_cubic_array_example.zarr"

    aprint(f"Creating DOF cubic array at {output_path}")
    aprint("This example creates a dense 100×100×100 grid for DOF testing")
    aprint("")
    aprint("Scene features:")
    aprint("- 1,000,000 sharp disc-like points")
    aprint("- Color gradient from warm (near) to cool (far)")
    aprint("- Optimized spacing to prevent overlap")

    # Create scene
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene()

        # Add metadata
        scene.attrs["description"] = """
DOF Cubic Array Test Scene
===========================

This scene provides a dense cubic array specifically designed for 
testing depth-of-field post-processing effects:

- 100×100×100 grid = 1,000,000 points
- Small radius (0.05) for disc-like appearance
- High sharpness (10.0) for sharp edges
- Additive blending for realistic light accumulation
- Color gradient: warm colors (red/yellow) near, cool colors (blue/cyan) far
- Reference markers at near/middle/far planes

Use this scene to test:
- DOF blur quality with dense point clouds
- Focus distance transitions
- Bokeh quality and shape
- Performance with high point counts

Recommended settings:
- Enable DOF in rendering controls (press 'R')
- Start with focus distance = 0 (middle of grid)
- Try DOF strength = 0.5 to 1.0
- Adjust focus distance to see blur transitions
- Use fly controls (press 'C') to move through the grid

Performance notes:
- This is a stress test with 1M points
- Reduce grid size if performance is poor
- Consider enabling FXAA for smoother edges
        """

        # Create the main cubic array
        aprint("\nGenerating cubic array...")
        positions, colors, radii, sharpness = create_cubic_array(
            grid_size=100,  # 100x100x100 grid
            spacing=0.5,  # Distance between points
            radius=0.05,  # Small radius for disc-like appearance
            sharpness=10.0,  # High sharpness for sharp edges
        )

        # Add main point cloud
        scene.add_points(
            "CubicArray",
            positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            opacity=1.0,
            blending_mode="additive",  # Use additive blending for point clouds
        )

        aprint(f"✓ Added {len(positions):,} points to cubic array")

        aprint("Creating background star field...")
        n_stars = 500000
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
            sharpness=2.0,
            opacity=0.4,
            gamma=1.0,
            blending_mode="normal",
        )


        # Print statistics
        aprint(f"\n✓ Scene created successfully!")
        aprint(f"  Total points: {len(positions):,}")
        aprint(f"  Grid dimensions: 100×100×100")
        aprint(f"  Point spacing: 0.5 units")
        aprint(f"  Point radius: 0.05 units")
        aprint(f"  Point sharpness: 10.0 (sharp edges)")

        aprint("\n" + "=" * 60)
        aprint("VIEWING INSTRUCTIONS:")
        aprint("1. Run: luxar serve dof_cubic_array_example.zarr --viewer")
        aprint("2. Press 'R' to open rendering controls")
        aprint("3. Navigate to 'Post-Processing Effects' → 'Depth of Field'")
        aprint("4. Enable DOF and experiment with:")
        aprint("   - Focus Distance: -25 to +25 (grid depth range)")
        aprint("   - DOF Strength: 0.5 to 1.0 (blur amount)")
        aprint("5. Press 'C' to switch to fly controls for navigation")
        aprint("6. Use WASD to fly through the grid")
        aprint("")
        aprint("COLOR GUIDE:")
        aprint("- Red/Yellow points: Near to camera")
        aprint("- Blue/Cyan points: Far from camera")
        aprint("- Red markers: Near reference plane")
        aprint("- Green markers: Middle reference plane")
        aprint("- Blue markers: Far reference plane")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
