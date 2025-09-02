#!/usr/bin/env python
"""
Example demonstrating nD points support in Luxar.

This example shows how to:
1. Create 5D points with time and channel dimensions
2. Add dimension metadata for proper interpretation
3. Mix different dimensionalities in the same scene
"""

from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import LuxarZarrCompiler
from luxar.types import DimensionMetadata


def create_5d_time_series(
    n_timepoints: int = 10, n_channels: int = 3, n_points: int = 1000
) -> np.ndarray:
    """Create a 5D points representing a time series with multiple channels."""
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


def create_2d_projection(n_points: int = 5000) -> np.ndarray:
    """Create a 2D points for a projection view."""
    # Create a 2D Lissajous curve
    t = np.linspace(0, 4 * np.pi, n_points)
    x = 50 * np.sin(3 * t + np.pi / 4)
    y = 50 * np.sin(2 * t)

    positions = np.column_stack([x, y]).astype(np.float32)
    return positions


def main():
    """Create a scene with mixed-dimensionality points."""
    aprint("Creating nD points demonstration...")

    # Create scene
    output_path = Path(__file__).parent / "nd_points_example.zarr"

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene()

        # Create 5D time series data
        aprint("Generating 5D time series data...")
        positions_5d = create_5d_time_series(
            n_timepoints=10, n_channels=3, n_points=500
        )

        # Define dimension metadata for 5D data
        metadata_5d = [
            DimensionMetadata(name="time", unit="s", scale=1.0, range=(0.0, 1.0)),
            DimensionMetadata(name="x", unit="um", scale=0.5),
            DimensionMetadata(name="y", unit="um", scale=0.5),
            DimensionMetadata(name="z", unit="um", scale=1.0),
            DimensionMetadata(name="channel", unit="au", scale=1.0, range=(0, 2)),
        ]

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
            dimension_metadata=metadata_5d,
        )

        # Create 2D projection data
        aprint("Generating 2D projection data...")
        positions_2d = create_2d_projection(n_points=2000)

        # Define dimension metadata for 2D data
        metadata_2d = [
            DimensionMetadata(name="x", unit="px", scale=1.0),
            DimensionMetadata(name="y", unit="px", scale=1.0),
        ]

        # Create gradient colors for 2D data
        n_points_2d = positions_2d.shape[0]
        colors_2d = np.zeros((n_points_2d, 3), dtype=np.float32)
        gradient = np.linspace(0, 1, n_points_2d)
        colors_2d[:, 0] = gradient  # Red gradient
        colors_2d[:, 1] = 1.0 - gradient  # Inverse green gradient
        colors_2d[:, 2] = 128  # Constant blue

        # Add 2D points to scene
        aprint(f"Adding {n_points_2d:,} 2D points to scene...")
        scene.add_points(
            "Projection2D",
            positions_2d,
            colors=colors_2d,
            dimension_metadata=metadata_2d,
        )

        # Add standard 3D points for reference
        aprint("Adding 3D reference points...")
        n_points_3d = 1000
        positions_3d = np.random.randn(n_points_3d, 3).astype(np.float32) * 20

        metadata_3d = [
            DimensionMetadata(name="x", unit="um", scale=1.0),
            DimensionMetadata(name="y", unit="um", scale=1.0),
            DimensionMetadata(name="z", unit="um", scale=1.0),
        ]

        # Simple white color for 3D points
        colors_3d = np.full((n_points_3d, 3), 0.78, dtype=np.float32)

        scene.add_points(
            "Reference3D",
            positions_3d,
            colors=colors_3d,
            dimension_metadata=metadata_3d,
        )

        # Finalize scene

        aprint(f"✓ nD demo scene created at {scene.get_store_path()}")
        aprint("\nScene summary:")
        aprint(f"- 5D time series: {n_points_5d:,} points across time and channels")
        aprint(f"- 2D projection: {n_points_2d:,} points in a Lissajous pattern")
        aprint(f"- 3D reference: {n_points_3d:,} randomly distributed points")
        aprint(
            "\nDimension metadata has been stored for proper visualization in the viewer."
        )


if __name__ == "__main__":
    main()
