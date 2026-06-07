#!/usr/bin/env python3
"""Dimension Sliders 5D Example - Interactive 5D navigation with slider UI.

This example demonstrates:
- Creating 5D scenes (time + 3D space + channel)
- Dimension slider UI for intuitive navigation
- Discrete vs continuous dimension handling
- Temporal animation with rotating spiral
- Multi-channel data visualization

Educational value:
- Learn to configure dimension sliders in the viewer
- Understand discrete dimension navigation
- Master 5D data structure and organization
- See how sliders enhance nD navigation UX
- Good example for UI-based dimension control

Key principle:
- Sliders provide visual feedback for dimension navigation
- Non-displayed dimensions become navigable via sliders
- Discrete dimensions snap to integer values
- Essential UI pattern for nD data exploration
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main():
    """Create a 5D scene with slider navigation."""
    output_path = get_examples_output_dir() / "dimension_sliders_5d_example.zarr"

    aprint(f"Creating 5D dimension sliders example at {output_path}")
    aprint("This example demonstrates:")
    aprint("- 5D data: X, Y, Z, Time, Channel")
    aprint("- Time dimension: Continuous with step navigation")
    aprint("- Channel dimension: Discrete values")
    aprint("- Moving point animation over time")
    aprint("- Multi-channel color coding")

    # Create a 5D scene (X, Y, Z, Time, Channel)
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension(name="X", unit="μm", range=(-50, 50), display=True),
                    Dimension(name="Y", unit="μm", range=(-50, 50), display=True),
                    Dimension(name="Z", unit="μm", range=(-50, 50), display=True),
                    Dimension(
                        name="W",
                        unit="μm",
                        range=(0, 10),
                        display=False,
                        spatial=True,
                        step=1.0,
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
            [1.0, 0.39, 0.39],  # Red
            [0.39, 1.0, 0.39],  # Green
            [0.39, 0.39, 1.0],  # Blue
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
                    np.float32
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

        add_explainer(
            scene,
            title="5D Dimension Sliders",
            body=(
                "An animated spiral spanning 11 time frames and 3 channels; "
                "the hidden <code>W</code> (time) and <code>Channel</code> "
                "dimensions are driven by the bottom slider UI."
            ),
            observe=[
                "Press <code>4</code> for W, then <code>[</code> / "
                "<code>]</code>: the spiral rotates and Z oscillates.",
                "Press <code>5</code> for Channel: colour cycles red, green, "
                "blue with smaller points per channel.",
                "Dragging the bottom sliders gives the same navigation.",
            ],
        )

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
