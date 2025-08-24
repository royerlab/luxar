#!/usr/bin/env python3
"""Spatial index demonstration example.

This example creates a 5D dataset with spatial indexing enabled to demonstrate
efficient nD point cloud navigation and querying.

The dataset contains multiple clusters of points distributed across a 5D space
(x, y, z, time, channel), with the spatial index enabling efficient loading
of only the relevant points for any given slice position.

Run with:
    python spatial_index_demo_example.py [--no-spatial-index]

Then view with:
    luxar serve spatial_index_demo_example.zarr
"""

import argparse
import time
from pathlib import Path

import numpy as np
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def create_5d_clusters(n_clusters: int = 10, points_per_cluster: int = 500) -> tuple:
    """Create clustered 5D point data.

    Creates clusters distributed across 5D space to demonstrate the
    benefits of spatial indexing for nD queries.

    Args:
        n_clusters: Number of clusters to create
        points_per_cluster: Points in each cluster

    Returns:
        Tuple of (positions, colors, radii)
    """
    positions = []
    colors = []
    radii = []

    # Create random cluster centers across 5D space
    cluster_centers = np.random.uniform(-50, 50, (n_clusters, 5)).astype(np.float32)

    # Ensure clusters are distributed across time and channel dimensions
    cluster_centers[:, 3] = np.linspace(0, 10, n_clusters)  # Time: 0-10
    cluster_centers[:, 4] = np.random.choice([0, 1, 2], n_clusters)  # Channel: 0-2

    for i, center in enumerate(cluster_centers):
        # Create points around this cluster center
        cluster_points = np.random.randn(points_per_cluster, 5).astype(np.float32)
        cluster_points *= [2, 2, 2, 0.2, 0.1]  # Different spreads per dimension
        cluster_points += center

        # Assign colors based on cluster
        hue = i / n_clusters
        cluster_colors = np.ones((points_per_cluster, 3), dtype=np.float32)
        cluster_colors[:, 0] = hue * 255  # Red channel varies
        cluster_colors[:, 1] = (1 - hue) * 255  # Green inverse
        cluster_colors[:, 2] = 128  # Blue constant

        # Vary radii within cluster
        cluster_radii = np.random.uniform(0.3, 1.5, points_per_cluster).astype(
            np.float32
        )

        # Add some large radius points that span multiple time slices
        if i % 3 == 0:
            # Every third cluster has some large-radius points
            large_indices = np.random.choice(points_per_cluster, size=50, replace=False)
            cluster_radii[large_indices] = np.random.uniform(3.0, 5.0, 50)
            cluster_colors[large_indices] *= 2.0  # Make them brighter (HDR)

        positions.append(cluster_points)
        colors.append(cluster_colors)
        radii.append(cluster_radii)

    # Combine all clusters
    positions = np.vstack(positions)
    colors = np.vstack(colors)
    radii = np.concatenate(radii)

    # Shuffle to test spatial index reordering
    shuffle_idx = np.random.permutation(len(positions))
    positions = positions[shuffle_idx]
    colors = colors[shuffle_idx]
    radii = radii[shuffle_idx]

    return positions, colors, radii


def main():
    """Create spatial index demonstration dataset."""
    parser = argparse.ArgumentParser(description="Create spatial index demo dataset")
    parser.add_argument(
        "--no-spatial-index",
        action="store_true",
        help="Disable spatial index for comparison",
    )
    parser.add_argument(
        "--clusters", type=int, default=20, help="Number of clusters (default: 20)"
    )
    parser.add_argument(
        "--points-per-cluster",
        type=int,
        default=1000,
        help="Points per cluster (default: 1000)",
    )
    args = parser.parse_args()

    # Output filename based on whether spatial index is enabled
    suffix = "_no_index" if args.no_spatial_index else ""
    output_path = Path(__file__).parent / Path(
        f"spatial_index_demo{suffix}_example.zarr"
    )

    # Clean up existing file if present
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    aprint(f"🎯 Creating spatial index demo with {args.clusters} clusters")
    aprint(f"   Each cluster has {args.points_per_cluster} points")
    aprint(f"   Spatial index: {'DISABLED' if args.no_spatial_index else 'ENABLED'}")

    # Define 5D dimensions
    dims = Dimensions(
        [
            Dimension("x", unit="um", display=True, range=(-60, 60)),
            Dimension("y", unit="um", display=True, range=(-60, 60)),
            Dimension("z", unit="um", display=True, range=(-60, 60)),
            Dimension(
                "time", unit="s", display=False, range=(0, 10), step=0.5, discrete=True
            ),
            Dimension(
                "channel", unit="ch", display=False, range=(0, 2), step=1, discrete=True
            ),
        ]
    )

    # Create 5D clustered data
    aprint("🔨 Generating 5D clustered point data...")
    positions, colors, radii = create_5d_clusters(
        args.clusters, args.points_per_cluster
    )

    total_points = len(positions)
    aprint(f"📊 Total points: {total_points:,}")
    aprint(f"   Position range: [{positions.min():.1f}, {positions.max():.1f}]")
    aprint(f"   Radii range: [{radii.min():.2f}, {radii.max():.2f}]")
    aprint(f"   HDR color max: {colors.max():.1f}")

    # Build scene with or without spatial index
    start_time = time.time()

    with LuxarZarrCompiler(
        str(output_path), enable_spatial_index=not args.no_spatial_index
    ) as compiler:
        scene = compiler.create_scene(dimensions=dims)

        # Add all points as a single cloud
        # The spatial index will automatically reorder them for efficiency
        if not args.no_spatial_index:
            # Use a grid that balances query performance and memory
            # For 5D data: 8×8×8×5×3 = 7680 potential cells
            grid_shape = (8, 8, 8, 5, 3)
            aprint(f"🔍 Using spatial index grid: {grid_shape}")
        else:
            grid_shape = None

        scene.add_points(
            "clustered_points",
            positions,
            colors=colors,
            radii=radii,
            grid_shape=grid_shape,
            opacity=0.8,
            gamma=1.2,
        )

    build_time = time.time() - start_time

    # Report statistics
    file_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    file_size_mb = file_size / (1024 * 1024)

    aprint("✅ Dataset created successfully!")
    aprint(f"   Output: {output_path}")
    aprint(f"   Build time: {build_time:.2f} seconds")
    aprint(f"   File size: {file_size_mb:.2f} MB")
    aprint(f"   Points/MB: {total_points / file_size_mb:.0f}")

    if not args.no_spatial_index:
        aprint("\n🔍 Spatial index benefits:")
        aprint("   - Efficient nD range queries")
        aprint("   - Only loads relevant point clusters")
        aprint("   - Smooth navigation through time/channel dims")
        aprint("   - Better cache utilization")

    aprint("\n📡 To view this dataset:")
    aprint(f"   luxar serve {output_path}")
    aprint("\n🎮 Navigation tips:")
    aprint("   - Use [ ] keys to navigate through time")
    aprint("   - Press 4 then [ ] to navigate channels")
    aprint("   - Notice how large-radius points remain visible across slices")

    if not args.no_spatial_index:
        aprint("\n💡 Try comparing with non-indexed version:")
        aprint(f"   python {__file__} --no-spatial-index")
        aprint("   Then compare loading performance in the viewer!")


if __name__ == "__main__":
    main()
