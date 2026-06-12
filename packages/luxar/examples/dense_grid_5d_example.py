#!/usr/bin/env python3
"""Dense Grid 5D Example - Multi-dimensional grid with time and channels.

This example demonstrates:
- Creating dense 5D grid data (X, Y, Z, Time, Channel)
- Regular grid structure (10x10x10) across 10 timepoints and 3 channels
- Scene-level dimension configuration for nD navigation
- How points change appearance across time and channels
- Discrete dimensions for frame-based navigation

Educational value:
- Learn to structure regular 5D grid data
- Understand time and channel as additional dimensions
- See how dense data behaves with dimension navigation
- Master configuration for multi-channel temporal data
- Good test case for spatial index performance
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

import luxar
from luxar import LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main():
    """Create a dense 5D grid scene with time and channel dimensions."""
    scene_path = get_examples_output_dir() / "dense_grid_5d_example.luxar.zarr"

    # Create a 5D scene (X, Y, Z, Time, Channel)
    dimensions = luxar.Dimensions(
        [
            luxar.Dimension(name="X", unit="μm", range=(-30, 30), display=True),
            luxar.Dimension(name="Y", unit="μm", range=(-30, 30), display=True),
            luxar.Dimension(name="Z", unit="μm", range=(-30, 30), display=True),
            luxar.Dimension(
                name="Time",
                unit="frame",
                range=(0, 9),
                display=False,
                step=1.0,
                discrete=True,
            ),
            luxar.Dimension(
                name="Channel",
                unit="",
                range=(0, 2),
                display=False,
                discrete=True,
                step=1.0,
            ),
        ]
    )

    with LuxarZarrCompiler(scene_path) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)

        # Create a dense 3D grid that changes over time and channels
        # Grid spacing
        grid_size = 10  # 10x10x10 grid
        spacing = 4.0  # 4 μm between points

        # Time and channel parameters
        n_time_points = 10  # 0 to 9
        n_channels = 3  # 0, 1, 2

        # Channel colors
        channel_colors = [
            [1.0, 0.3, 0.3],  # Red
            [0.3, 1.0, 0.3],  # Green
            [0.3, 0.3, 1.0],  # Blue
        ]

        all_positions = []
        all_colors = []
        all_radii = []

        # Create grid for each time point and channel
        for t in range(n_time_points):
            for c in range(n_channels):
                # Create 3D grid
                x = np.linspace(
                    -spacing * (grid_size - 1) / 2,
                    spacing * (grid_size - 1) / 2,
                    grid_size,
                )
                y = np.linspace(
                    -spacing * (grid_size - 1) / 2,
                    spacing * (grid_size - 1) / 2,
                    grid_size,
                )
                z = np.linspace(
                    -spacing * (grid_size - 1) / 2,
                    spacing * (grid_size - 1) / 2,
                    grid_size,
                )

                # Create meshgrid
                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

                # Flatten to get point positions
                x_flat = xx.flatten()
                y_flat = yy.flatten()
                z_flat = zz.flatten()

                # Apply time-based transformation
                # Rotate around center based on time
                angle = t * 10  # degrees
                angle_rad = np.radians(angle)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)

                # Rotate around Z axis
                x_rot = x_flat * cos_a - y_flat * sin_a
                y_rot = x_flat * sin_a + y_flat * cos_a
                z_rot = z_flat

                # Create 5D positions (X, Y, Z, Time, Channel)
                positions_5d = np.column_stack(
                    [
                        x_rot,
                        y_rot,
                        z_rot,
                        np.full_like(x_flat, t),  # Time coordinate
                        np.full_like(x_flat, c),  # Channel coordinate
                    ]
                ).astype(np.float32)

                all_positions.append(positions_5d)

                # Colors based on channel
                n_points = len(x_flat)
                # Apply time-based intensity modulation
                intensity = 0.5 + 0.5 * np.sin(angle_rad)
                modulated_color = [c * intensity for c in channel_colors[c]]
                colors = np.tile(modulated_color, (n_points, 1)).astype(np.float32)
                all_colors.append(colors)

                # Radii that vary with time
                base_radius = 0.15
                time_factor = 1.0 + 0.3 * np.sin(angle_rad)
                radii = np.full(n_points, base_radius * time_factor, dtype=np.float32)
                all_radii.append(radii)

        # Combine all data
        all_positions = np.vstack(all_positions)
        all_colors = np.vstack(all_colors)
        all_radii = np.concatenate(all_radii)

        # Add points to scene
        aprint(f"Adding {len(all_positions):,} 5D points to scene...")
        scene.add_points(
            "DenseGrid5D",
            all_positions,
            colors=all_colors,
            radii=all_radii,
            sharpness=0.5,
            opacity=0.8,
            blending_mode="additive",
            # Explicitly set extend_to_all=[] - points only appear at their defined time/channel values
            extend_to_all=[],
        )

        # Add axis markers for spatial reference
        # Note: These are only defined for time=0, channel=0, but we explicitly
        # extend them to appear at all time/channel combinations
        axis_length = 20.0
        axis_positions = []
        axis_colors = []

        # X axis markers (red)
        for i in range(5):
            x_pos = -axis_length + (i * axis_length / 2)
            # 5D position: (x, y, z, time=0, channel=0)
            axis_positions.append([x_pos, 0, 0, 0, 0])
            axis_colors.append([1.0, 0.2, 0.2])

        # Y axis markers (green)
        for i in range(5):
            y_pos = -axis_length + (i * axis_length / 2)
            axis_positions.append([0, y_pos, 0, 0, 0])
            axis_colors.append([0.2, 1.0, 0.2])

        # Z axis markers (blue)
        for i in range(5):
            z_pos = -axis_length + (i * axis_length / 2)
            axis_positions.append([0, 0, z_pos, 0, 0])
            axis_colors.append([0.2, 0.2, 1.0])

        scene.add_points(
            "AxisMarkers",
            np.array(axis_positions, dtype=np.float32),
            colors=np.array(axis_colors, dtype=np.float32),
            radii=0.3,
            sharpness=0.9,
            opacity=1.0,
            blending_mode="normal",
            # Explicitly extend axis markers to all time points and channels
            extend_to_all=["Time", "Channel"],
        )

        add_explainer(
            scene,
            title="Dense 5D Grid",
            body=(
                "A regular 10×10×10 grid replicated across 10 timepoints and "
                "3 channels; the hidden <code>Time</code> and "
                "<code>Channel</code> dimensions are navigated as sliders."
            ),
            observe=[
                "Press <code>4</code> for Time, then <code>[</code> / "
                "<code>]</code>: the grid rotates 10° per frame.",
                "Press <code>5</code> for Channel: colour switches red, green, blue.",
                "Point sizes oscillate with time.",
                "The R/G/B axis markers stay visible at every time and channel.",
            ],
        )

    aprint(f"✓ 5D scene created at {scene_path}")
    aprint("\nScene summary:")
    aprint(f"- Total points: {len(all_positions):,}")
    aprint("- Dimensions: X, Y, Z (displayed), Time, Channel (hidden)")
    aprint(f"- Time points: {n_time_points} (0-9)")
    aprint(f"- Channels: {n_channels} (Red, Green, Blue)")
    aprint(f"- Grid size: {grid_size}x{grid_size}x{grid_size} per time/channel")
    aprint("\n" + "=" * 60)
    aprint("VIEWING INSTRUCTIONS:")
    aprint(f"1. Run: luxar serve {scene_path}")
    aprint("2. Use keyboard to navigate time dimension:")
    aprint("   - Press '4' to select Time dimension")
    aprint("   - Press '[' and ']' to navigate through time")
    aprint("3. Use keyboard to navigate channel dimension:")
    aprint("   - Press '5' to select Channel dimension")
    aprint("   - Press '[' and ']' to switch channels")
    aprint("")
    aprint("Expected behavior:")
    aprint("- Grid rotates over time (10 deg per frame)")
    aprint("- Colors change based on channel (R/G/B)")
    aprint("- Point sizes oscillate with time")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
