#!/usr/bin/env python3
"""Dimension Sliders 5D Example - Demonstrates 5D data visualization with sliders.

This educational example demonstrates:
- Creating 5D scenes with time and channel dimensions
- Setting up dimension sliders for navigation
- Discrete vs continuous dimension handling
- Temporal animation data
- Multi-channel data visualization
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Dimension, Dimensions, Scene


def main():
    """Create a 5D scene with slider navigation."""
    output_path = Path(__file__).parent / "dimension_sliders_5d_example.zarr"

    aprint(f"Creating 5D dimension sliders example at {output_path}")
    aprint("This example demonstrates:")
    aprint("- 5D data: X, Y, Z, Time, Channel")
    aprint("- Time dimension: Continuous with step navigation")
    aprint("- Channel dimension: Discrete values")
    aprint("- Moving point animation over time")
    aprint("- Multi-channel color coding")

    # Create a 5D scene (X, Y, Z, Time, Channel)
    scene = Scene(
        output_path,
        dimensions=Dimensions(
            [
                Dimension(name="X", unit="μm", range=(-50, 50), display=True),
                Dimension(name="Y", unit="μm", range=(-50, 50), display=True),
                Dimension(name="Z", unit="μm", range=(-50, 50), display=True),
                Dimension(
                    name="Time", unit="s", range=(0, 10), display=False, step=1.0
                ),
                Dimension(
                    name="Channel",
                    unit="",
                    range=(0, 2),
                    display=False,
                    discrete=True,
                    step=1.0,
                ),
            ]
        ),
    )

    # Create some test data - moving points over time and channels
    aprint("\nGenerating 5D point data...")
    n_time_points = 11  # 0 to 10 seconds
    n_channels = 3  # 0, 1, 2
    n_points_per_frame = 1000

    # Generate base positions (spiral)
    aprint("Creating animated spiral pattern...")
    theta = np.linspace(0, 4 * np.pi, n_points_per_frame)
    base_radius = np.linspace(10, 40, n_points_per_frame)

    all_positions = []
    all_colors = []
    all_radii = []

    # Channel colors
    channel_colors = [
        [255, 100, 100],  # Red
        [100, 255, 100],  # Green
        [100, 100, 255],  # Blue
    ]

    for t in range(n_time_points):
        for c in range(n_channels):
            # Animate the spiral over time
            time_offset = t * 0.2 * np.pi
            channel_offset = c * 2 * np.pi / 3

            x = base_radius * np.cos(theta + time_offset + channel_offset)
            y = base_radius * np.sin(theta + time_offset + channel_offset)
            z = np.linspace(-30, 30, n_points_per_frame) + 5 * np.sin(time_offset)

            # Create 5D positions (X, Y, Z, Time, Channel)
            positions = np.zeros((n_points_per_frame, 5), dtype=np.float32)
            positions[:, 0] = x
            positions[:, 1] = y
            positions[:, 2] = z
            positions[:, 3] = t  # Time coordinate
            positions[:, 4] = c  # Channel coordinate

            all_positions.append(positions)

            # Colors based on channel
            colors = np.tile(channel_colors[c], (n_points_per_frame, 1)).astype(
                np.uint8
            )
            all_colors.append(colors)

            # Radii - smaller for higher channels
            radii = np.full(n_points_per_frame, 2.0 - c * 0.5, dtype=np.float32)
            all_radii.append(radii)

    # Combine all data
    positions = np.vstack(all_positions)
    colors = np.vstack(all_colors)
    radii = np.concatenate(all_radii)

    aprint(
        f"Generated {len(positions):,} points across {n_time_points} time steps and {n_channels} channels"
    )

    # Add points to scene
    scene.add_points(
        "AnimatedSpiral",
        positions,
        colors=colors,
        radii=radii,
        sharpness=2.0,
        opacity=0.9,
        gamma=1.0,
        blending_mode="normal",
    )

    scene.finalize()

    # Print educational summary
    aprint("\n" + "=" * 60)
    aprint("5D DIMENSION SLIDER DEMONSTRATION")
    aprint("=" * 60)
    aprint("Scene Structure:")
    aprint("- Total dimensions: 5 (X, Y, Z, Time, Channel)")
    aprint("- Displayed dimensions: X, Y, Z")
    aprint("- Navigation dimensions: Time (0-10s), Channel (0-2)")
    aprint(f"- Total points: {len(positions):,}")
    aprint(f"- Time frames: {n_time_points}")
    aprint(f"- Channels: {n_channels}")
    aprint(f"- Points per time/channel: {n_points_per_frame}")

    aprint("\nDimension Configuration:")
    for i, dim in enumerate(scene.dimensions.dimensions):
        displayed = "displayed" if dim.display else "slider/navigation"
        aprint(f"  {dim.name}: {dim.unit} - {displayed}")

    aprint("\nAnimation Features:")
    aprint("- Spiral rotates over time")
    aprint("- Different colors per channel (Red, Green, Blue)")
    aprint("- Z-axis oscillation with time")
    aprint("- Different point sizes per channel")

    aprint(f"\nTo view: luxar serve {output_path}")
    aprint("Navigation controls:")
    aprint("- Press '4' to select Time dimension, then use [ ] to navigate")
    aprint("- Press '5' to select Channel dimension, then use [ ] to navigate")
    aprint("- Use sliders in the UI for smooth navigation")
    aprint("=" * 60)


if __name__ == "__main__":
    main()
