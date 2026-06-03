#!/usr/bin/env python3
"""nD Points Example - Comprehensive demonstration of high-dimensional point data.

This example demonstrates:
- Creating 5D point data (time + 3D space + channel)
- Defining scene-level dimensions for proper nD visualization
- Using non-displayed dimensions for temporal and channel data
- Navigating through high-dimensional data with keyboard controls
- How scene dimensions enable automatic validation and navigation

Educational value:
- Understand nD point clouds (beyond 3D)
- Learn scene-level dimension configuration
- See how displayed vs non-displayed dimensions work
- Master nD navigation with dimension selection and stepping
- Understand spatial indexing for efficient nD queries

Key principles:
- Scene dimensions define the coordinate system for all objects
- Non-displayed dimensions (time, channel) must be discrete
- Displayed dimensions (x, y, z) show the 3D slice
- Spatial indexing enables efficient loading of relevant points only
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_5d_time_series(
    n_timepoints: int = 10, n_channels: int = 3, n_points: int = 1000
) -> np.ndarray:
    """Create 5D point data with temporal and channel dimensions.

    Generates a spiral pattern that evolves over time with multiple channels,
    demonstrating how to structure high-dimensional point data.

    Args:
        n_timepoints: Number of time steps
        n_channels: Number of data channels
        n_points: Points per channel per timepoint

    Returns:
        Array of 5D positions [time, x, y, z, channel]
    """
    positions = np.zeros((n_timepoints * n_channels * n_points, 5), dtype=np.float32)

    idx = 0
    for t in range(n_timepoints):
        time_val = t * 0.1  # Time in seconds

        for c in range(n_channels):
            # Different dynamics for each channel
            phase_offset = c * np.pi / 3

            for i in range(n_points):
                # Spiral motion that evolves over time
                angle = i * 0.1 + t * 0.5 + phase_offset
                radius = 10 + 5 * np.sin(t * 0.2 + phase_offset)

                positions[idx] = [
                    time_val,  # Time dimension
                    radius * np.cos(angle),  # X
                    radius * np.sin(angle),  # Y
                    i * 0.1 - 50,  # Z (vertical spread)
                    c,  # Channel
                ]
                idx += 1

    return positions


def main():
    """Create a single 5D scene demonstrating nD points with scene-level dimensions."""
    aprint("Creating 5D nD points demonstration...")

    output_path = get_examples_output_dir() / "nd_points_example.zarr"

    with LuxarZarrCompiler(output_path) as compiler:
        # Define 5D dimensions: time, x, y, z, channel
        # Only x, y, z are displayed (visualized), time and channel are non-displayed
        aprint("Defining 5D scene dimensions...")
        dims_5d = Dimensions(
            [
                Dimension(
                    "time",
                    unit="s",
                    range=(0.0, 1.0),
                    step=0.1,
                    display=False,
                    discrete=True,  # Non-displayed dimensions must be discrete
                    scale=1.0,
                    description="Time evolution of the system",
                ),
                Dimension(
                    "x",
                    unit="um",
                    display=True,
                    scale=0.5,
                    description="Spatial X coordinate",
                ),
                Dimension(
                    "y",
                    unit="um",
                    display=True,
                    scale=0.5,
                    description="Spatial Y coordinate",
                ),
                Dimension(
                    "z",
                    unit="um",
                    display=True,
                    scale=1.0,
                    description="Spatial Z coordinate",
                ),
                Dimension(
                    "channel",
                    unit="au",
                    range=(0, 2),
                    step=1.0,
                    display=False,
                    discrete=True,
                    scale=1.0,
                    description="Color channel (R=0, G=1, B=2)",
                ),
            ]
        )

        scene = compiler.create_scene(dimensions=dims_5d)

        # Create 5D time series data
        aprint("Generating 5D time series data...")
        positions_5d = create_5d_time_series(
            n_timepoints=10, n_channels=3, n_points=500
        )

        # Create colors based on time and channel
        n_points_5d = positions_5d.shape[0]
        colors_5d = np.zeros((n_points_5d, 3), dtype=np.float32)

        # Color by channel: R, G, B for channels 0, 1, 2
        channel_indices = positions_5d[:, 4].astype(int)
        colors_5d[channel_indices == 0, 0] = 1.0  # Red for channel 0
        colors_5d[channel_indices == 1, 1] = 1.0  # Green for channel 1
        colors_5d[channel_indices == 2, 2] = 1.0  # Blue for channel 2

        # Add intensity variation based on time
        time_norm = positions_5d[:, 0] / positions_5d[:, 0].max()
        for i in range(3):
            colors_5d[:, i] = colors_5d[:, i] * (0.5 + 0.5 * time_norm)

        # Add 5D points to scene
        aprint(f"Adding {n_points_5d:,} 5D points to scene...")
        scene.add_points(
            "TimeSeries5D",
            positions_5d,
            colors=colors_5d,
        )

        # Add some reference 5D points with different patterns
        aprint("Adding reference 5D points with different patterns...")

        # Create a simple 5D grid pattern
        n_ref = 200
        positions_ref = np.zeros((n_ref, 5), dtype=np.float32)

        # Grid in time and channel dimensions
        t_vals = np.linspace(0.2, 0.8, 5)  # 5 time points
        c_vals = np.array([0, 1, 2])  # 3 channels

        idx = 0
        for t in t_vals:
            for c in c_vals:
                # Create spatial points in a small sphere for this time/channel combo
                n_spatial = n_ref // (len(t_vals) * len(c_vals))
                if n_spatial == 0:
                    continue

                # Random points in a sphere
                theta = np.random.uniform(0, 2 * np.pi, n_spatial)
                phi = np.random.uniform(0, np.pi, n_spatial)
                r = np.random.uniform(0, 15, n_spatial)

                x = r * np.sin(phi) * np.cos(theta) + 30  # Offset from main data
                y = r * np.sin(phi) * np.sin(theta) + 30
                z = r * np.cos(phi)

                for i in range(n_spatial):
                    if idx >= n_ref:
                        break
                    positions_ref[idx] = [t, x[i], y[i], z[i], c]
                    idx += 1

        # Truncate if we didn't fill all positions
        positions_ref = positions_ref[:idx]

        # Create colors for reference points (white/gray)
        colors_ref = np.full((positions_ref.shape[0], 3), 0.6, dtype=np.float32)

        scene.add_points(
            "Reference5D",
            positions_ref,
            colors=colors_ref,
        )

        add_explainer(
            scene,
            title="High-Dimensional Points",
            body=(
                "A 5D point cloud (<code>time</code>, x, y, z, "
                "<code>channel</code>) where only x/y/z are displayed and the "
                "other two dimensions are sliced via the keyboard."
            ),
            observe=[
                "Press <code>1</code> for time, then <code>[</code> / "
                "<code>]</code>: the spiral evolves and brightens.",
                "Press <code>5</code> for channel: red, green, blue subsets appear.",
                "Grey reference points sit offset near (30, 30) at discrete "
                "time/channel values.",
            ],
        )

        aprint(f"✓ 5D nD scene created at {output_path}")
        aprint("\nScene summary:")
        aprint(
            f"- Main time series: {n_points_5d:,} 5D points across time and channels"
        )
        aprint(f"- Reference pattern: {positions_ref.shape[0]:,} 5D reference points")
        aprint("\n5D Dimensions defined:")
        aprint("  - time: [0.0, 1.0] s (non-displayed, discrete)")
        aprint("  - x, y, z: spatial coordinates in μm (displayed)")
        aprint("  - channel: [0, 1, 2] au (non-displayed, discrete)")
        aprint(
            "\nIn the viewer:"
            "\n  - Use keys 1-5 to select dimension for navigation"
            "\n  - Use [ and ] to step through non-displayed dimensions"
            "\n  - Only 3D spatial view is displayed, other dims are sliced"
        )


if __name__ == "__main__":
    main()
