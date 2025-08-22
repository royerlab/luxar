#!/usr/bin/env python3
"""Depth of Field Grid Example - Test depth-based rendering effects.

This example demonstrates:
- Creating a 3D grid with clear depth layers
- Testing depth-based visual effects and occlusion
- Verifying perspective rendering with multiple depth planes
- Providing a reference for depth-of-field and fog effects

This scene is useful for testing rendering features that depend on
depth, such as depth-of-field blur, fog, and proper occlusion.
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import LuxarZarrCompiler


def create_depth_grid(grid_size=10, layers=5, spacing=1.0):
    """Create a multi-layer grid for depth testing.

    Args:
        grid_size: Number of points per layer in X and Y
        layers: Number of depth layers
        spacing: Spacing between points

    Returns:
        Tuple of (positions, colors, radii, labels) arrays
    """
    positions = []
    colors = []
    radii = []

    # Create layers at different depths
    for layer in range(layers):
        z = layer * spacing * 2  # Space layers apart

        # Color gradient from red (near) to blue (far)
        layer_color = [
            1.0 - layer / (layers - 1),  # Red decreases with depth
            0.0,
            layer / (layers - 1),  # Blue increases with depth
        ]

        # Create grid for this layer
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size / 2) * spacing
                y = (j - grid_size / 2) * spacing

                positions.append([x, y, z])
                colors.append(layer_color)

                # Vary radius slightly based on position for variety
                base_radius = 0.1
                radius_variation = 0.02 * np.sin(i * 0.5) * np.cos(j * 0.5)
                radii.append(base_radius + radius_variation)

    # Add depth markers (large spheres at specific depths)
    marker_positions = []
    marker_colors = []
    for layer in range(layers):
        z = layer * spacing * 2
        # Place marker at edge of grid
        marker_positions.append([grid_size * spacing / 2 + 1, 0, z])
        marker_colors.append([1.0, 1.0, 1.0])  # White markers

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(radii, dtype=np.float32),
        np.array(marker_positions, dtype=np.float32),
        np.array(marker_colors, dtype=np.float32),
    )


def create_depth_labels(layers=5, spacing=1.0):
    """Create label points to indicate depth values.

    Args:
        layers: Number of depth layers
        spacing: Spacing between layers

    Returns:
        Tuple of (positions, colors) for labels
    """
    positions = []
    colors = []

    for layer in range(layers):
        z = layer * spacing * 2
        # Create small points forming the layer number
        # Place them to the left of the grid
        base_x = -8
        base_y = 4 - layer * 1.5

        # Create a simple marker for each layer
        for i in range(layer + 1):  # Number of dots indicates layer
            positions.append([base_x - i * 0.3, base_y, z])
            colors.append([1.0, 1.0, 0.0])  # Yellow for visibility

    return (np.array(positions, dtype=np.float32), np.array(colors, dtype=np.float32))


def main():
    """Create a depth-of-field test grid scene."""
    output_path = Path(__file__).parent / "depth_of_field_grid_example.zarr"

    aprint(f"Creating depth-of-field grid at {output_path}")
    aprint("This example creates a multi-layer grid for depth testing")
    aprint("")
    aprint("Scene features:")
    aprint("- 5 layers at different depths")
    aprint("- Color gradient from red (near) to blue (far)")
    aprint("- White depth markers at each layer")
    aprint("- Yellow layer indicators on the left")

    # Create scene

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene()

        # Add metadata
        scene.attrs["description"] = """
Depth of Field Test Grid
        ========================

        This scene provides a reference for testing depth-based rendering:

        - 5 distinct depth layers
        - Red-to-blue color gradient indicating depth
        - White markers showing exact depth positions
        - Yellow indicators showing layer numbers

        Use this scene to test:
        - Depth of field blur effects
        - Fog and atmospheric effects
        - Proper occlusion and depth sorting
        - Perspective scaling with depth

        Controls:
        - Press 'R' to toggle rendering controls
        - Adjust DOF focal distance and aperture
        - Test different fog densities
        """

        # Create the depth grid
        positions, colors, radii, marker_pos, marker_colors = create_depth_grid(
            grid_size=10, layers=5, spacing=1.0
        )

        # Add main grid points
        scene.add_points(
            "DepthGrid",
            positions,
            colors=colors,
            radii=radii,
            sharpness=5.0,
            opacity=0.9,
            blending_mode="normal",
        )

        # Add depth markers
        scene.add_points(
            "DepthMarkers",
            marker_pos,
            colors=marker_colors,
            radii=0.3,
            sharpness=10.0,
            opacity=1.0,
            blending_mode="additive",
        )

        # Add depth labels
        label_pos, label_colors = create_depth_labels(layers=5, spacing=1.0)
        scene.add_points(
            "DepthLabels",
            label_pos,
            colors=label_colors,
            radii=0.08,
            sharpness=10.0,
            opacity=1.0,
            blending_mode="additive",
        )

        aprint(f"\n✓ Scene created with {len(positions)} grid points")
        aprint("  Grid: 10x10 points per layer")
        aprint("  Layers: 5 depth planes")
        aprint("  Depth range: 0 to 8 units")
        aprint("\n" + "=" * 60)
        aprint("VIEWING TIPS:")
        aprint("1. Run: luxar serve depth_of_field_grid_example.zarr")
        aprint("2. Press 'R' to open rendering controls")
        aprint("3. Enable DOF and adjust focal distance")
        aprint("4. Try different aperture sizes for blur amount")
        aprint("5. Notice color gradient from red (near) to blue (far)")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
