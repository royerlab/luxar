#!/usr/bin/env python3
"""Test example for large radius slicing with spatial index support.

This example creates a 4D dataset with some points having very large radii
that are far from the slice plane. This tests that the lazy loading system
correctly loads chunks containing these points even when their centers are
far from the current slice position.

When spatial index is enabled (default), it demonstrates efficient querying
of only relevant points based on their spatial locality.
"""

import argparse
from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def main():
    """Create test dataset with large radius points."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Create large radius slicing test dataset"
    )
    parser.add_argument(
        "--no-spatial-index",
        action="store_true",
        help="Disable spatial index (for comparison)",
    )
    args = parser.parse_args()

    # Output path with suffix based on options
    suffix = "_no_index" if args.no_spatial_index else ""
    output_path = Path(__file__).parent / f"large_radius_slicing{suffix}_example.zarr"
    aprint(f"Creating test dataset: {output_path}")
    aprint(f"Spatial index: {'DISABLED' if args.no_spatial_index else 'ENABLED'}")

    # Create scene with 4D dimensions (x, y, z, time)
    # Note: time is discrete and non-spatial - points don't extend through time
    dimensions = Dimensions(
        [
            Dimension(
                name="x",
                unit="μm",
                range=[0, 100],
                step=0.1,
                display=True,
                spatial=True,
            ),
            Dimension(
                name="y",
                unit="μm",
                range=[0, 100],
                step=0.1,
                display=True,
                spatial=True,
            ),
            Dimension(
                name="z",
                unit="μm",
                range=[0, 100],
                step=0.1,
                display=True,
                spatial=True,
            ),
            Dimension(
                name="time",
                unit="s",
                range=[0, 100],
                step=1.0,
                display=False,
                discrete=True,
                spatial=False,
            ),
        ]
    )

    with LuxarZarrCompiler(
        output_path, enable_spatial_index=not args.no_spatial_index
    ) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)

        # Create points at different time slices
        n_time_points = 11  # 0 to 10 seconds
        points_per_time = 100

        all_positions = []
        all_colors = []
        all_radii = []

        for t in range(n_time_points):
            # Regular points with small radii
            regular_positions = np.random.uniform(20, 80, (points_per_time - 10, 3))
            regular_times = np.full((points_per_time - 10, 1), 10 * t)
            regular_4d = np.hstack([regular_positions, regular_times])

            # Regular radii (small) - appropriate for time step of 1.0
            regular_radii = np.random.uniform(0.1, 2, points_per_time - 10)

            # Add a few points with VERY LARGE radii at this time slice
            # These are positioned at the edges but their large radii should make them
            # visible even when we're looking at adjacent time slices
            large_radius_positions = np.array(
                [
                    [10, 50, 50],  # Left edge
                    [90, 50, 50],  # Right edge
                    [50, 10, 50],  # Bottom edge
                    [50, 90, 50],  # Top edge
                    [50, 50, 10],  # Front edge
                    [50, 50, 90],  # Back edge
                    [20, 20, 20],  # Corner 1
                    [80, 80, 80],  # Corner 2
                    [20, 80, 20],  # Corner 3
                    [80, 20, 80],  # Corner 4
                ]
            )
            large_radius_times = np.full((10, 1), 10 * t)
            large_radius_4d = np.hstack([large_radius_positions, large_radius_times])

            # VERY LARGE radii - these should be visible across multiple time slices!
            # NOTE: The radius applies to ALL dimensions uniformly!
            # With time step=1.0s, a radius of 0.8 means visible when within 0.8 time units
            # So a point at t=5 with radius 0.8 is visible from t=4.2 to t=5.8
            large_radii = np.array([2, 2, 2, 2, 3, 8, 4, 5, 8, 8])

            # Combine regular and large radius points
            time_positions = np.vstack([regular_4d, large_radius_4d])
            time_radii = np.hstack([regular_radii, large_radii])

            # Colors: regular points are blue, large radius points are red
            regular_colors = np.tile([0.2, 0.5, 1.0], (points_per_time - 10, 1))  # Blue
            large_colors = np.tile([1.0, 0.2, 0.2], (10, 1))  # Red
            time_colors = np.vstack([regular_colors, large_colors])

            all_positions.append(time_positions)
            all_colors.append(time_colors)
            all_radii.append(time_radii)

        # Stack all time points
        positions = np.vstack(all_positions).astype(np.float32)
        colors = np.vstack(all_colors).astype(np.float32)
        radii = np.hstack(all_radii).astype(np.float32)

        # Shuffle to make chunk loading more interesting
        indices = np.random.permutation(len(positions))
        positions = positions[indices]
        colors = colors[indices]
        radii = radii[indices]

        aprint(f"Total points: {len(positions)}")
        aprint(f"Max radius: {np.max(radii):.1f}")
        aprint(f"Min radius: {np.min(radii):.1f}")

        # Add to scene with spatial index support
        # For 4D data, use a reasonable grid shape
        grid_shape = None if args.no_spatial_index else (10, 10, 10, 5)

        if not args.no_spatial_index:
            aprint(f"Using spatial index grid: {grid_shape}")

        scene.add_points(
            "test_points",
            positions=positions,
            colors=colors,
            radii=radii,
            grid_shape=grid_shape,
        )

    aprint(f"✅ Created test dataset: {output_path}")
    aprint("\nTest instructions:")
    aprint(f"1. Serve the dataset: luxar serve {output_path}")
    aprint(
        "2. Open the viewer and navigate through time (press 4 to select time dimension, then [ and ])"
    )
    aprint(
        "3. Red points with large radii should be visible across multiple time slices"
    )
    aprint(
        "4. Points with radius 8 are visible when within ±8 seconds of their time position"
    )
    aprint("5. Watch how red points fade in/out smoothly as you navigate through time")

    if not args.no_spatial_index:
        aprint("\n🔍 Spatial index benefits:")
        aprint("   - Only loads points near current time slice")
        aprint("   - Efficient radius-based queries")
        aprint("   - Better performance for large datasets")
        aprint("\n💡 Compare with non-indexed version:")
        aprint(f"   python {Path(__file__).name} --no-spatial-index")

    return output_path


if __name__ == "__main__":
    main()
