#!/usr/bin/env python3
"""Self-Contained Demo: Network Performance Testing with Large 4D Dataset

This demo demonstrates:
- Testing viewer performance under realistic network conditions
- Generating a large, complex 4D dataset with time dimension (10M+ points)
- Using network simulation profiles (slow broadband by default)
- Comparing performance with different network profiles
- Progressive loading behavior under bandwidth constraints
- Interactive bandwidth testing via time dimension navigation

The demo creates a visually interesting 4D dataset that stresses the viewer's
loading and caching systems. Navigate through time ([4] then [ / ]) to trigger
additional data loading and test network performance interactively.

Usage:
    python demo_network_performance.py [--points N] [--profile NAME]

    --points N        Number of points (default: 1000000)
    --profile NAME    Network profile (default: slow-broadband)
                      Options: 3g, 4g, 5g, slow-broadband, broadband,
                               fast-broadband, satellite, rural, congested
    --no-serve        Generate only, don't serve
    --no-simulation   Serve without network simulation (full speed)

Examples:
    # Test with slow broadband (default)
    python demo_network_performance.py

    # Test with 3G mobile connection
    python demo_network_performance.py --profile 3g

    # Large dataset with satellite latency
    python demo_network_performance.py --points 2000000 --profile satellite

    # Compare: full speed vs throttled
    python demo_network_performance.py --no-simulation
    python demo_network_performance.py --profile slow-broadband

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
    - Watch browser DevTools Network tab to see throttling in action
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def generate_performance_test_dataset(
    output_path: Path, n_points: int = 1000000, n_timesteps: int = 10, seed: int = 42
) -> None:
    """Generate a large, complex 4D dataset for performance testing.

    Creates a visually interesting "particle cloud" with:
    - Multiple clustered regions (tests spatial queries)
    - Varying densities (tests LOD and culling)
    - Color gradients (tests color encoding/decoding)
    - Mixed radii (tests radius-based slicing)
    - Time dimension (tests nD navigation and progressive loading)

    This dataset is designed to stress-test:
    - Progressive loading under bandwidth constraints
    - Cache effectiveness with limited bandwidth
    - Spatial index query performance
    - UI responsiveness during slow loading
    - nD navigation bandwidth usage (navigating time loads new data)

    Args:
        output_path: Where to write the zarr store
        n_points: Total number of points per timestep
        n_timesteps: Number of timesteps (default: 10)
        seed: Random seed for reproducibility
    """
    with asection(f"Generating Performance Test Dataset ({n_points:,} points × {n_timesteps} timesteps)"):
        rng = np.random.default_rng(seed)

        aprint(f"Seed: {seed}")
        aprint(f"Timesteps: {n_timesteps}")
        aprint(
            "Creating multi-cluster particle system with varying densities and time evolution..."
        )

        # Generate multiple clusters with different characteristics
        n_clusters = 8
        points_per_cluster = n_points // n_clusters
        all_positions = []
        all_colors = []
        all_radii = []

        for i in range(n_clusters):
            with asection(f"Cluster {i+1}/{n_clusters}"):
                # Random cluster center (3D spatial)
                center = rng.uniform(-100, 100, size=3).astype(np.float32)

                # Varying cluster sizes (some tight, some spread out)
                spread = rng.uniform(5, 30)

                # Generate 3D spatial points in this cluster (Gaussian distribution)
                cluster_points_3d = rng.normal(
                    loc=center, scale=spread, size=(points_per_cluster, 3)
                ).astype(np.float32)

                # Add time dimension: replicate points across time with slight variation
                # Shape will be (points_per_cluster * n_timesteps, 4)
                cluster_points_4d = []
                for t in range(n_timesteps):
                    # Add small time-based variation (animate the cluster slightly)
                    time_offset = rng.normal(0, 0.5, size=(points_per_cluster, 3)).astype(np.float32)
                    spatial_coords = cluster_points_3d + time_offset * (t / n_timesteps)

                    # Add time coordinate as 4th dimension
                    time_coords = np.full((points_per_cluster, 1), t, dtype=np.float32)
                    points_with_time = np.hstack([time_coords, spatial_coords])

                    cluster_points_4d.append(points_with_time)

                cluster_points = np.vstack(cluster_points_4d)  # Now (points_per_cluster * n_timesteps, 4)

                # Color based on cluster position and distance from center (3D spatial only)
                spatial_positions = cluster_points[:, 1:4]  # Extract x, y, z (skip time dimension)
                distances = np.linalg.norm(spatial_positions - center, axis=1)
                max_dist = distances.max()

                # Create color gradient from cluster center (blue) to edge (red)
                t = distances / max_dist  # 0 at center, 1 at edge

                # Use HSV color space for smooth gradients
                hue = (i / n_clusters + t * 0.3) % 1.0  # Vary hue by cluster and distance
                saturation = 0.8 + 0.2 * t  # More saturated at edges
                value = 0.6 + 0.4 * (1 - t)  # Brighter at center

                # Convert HSV to RGB (vectorized)
                h_i = (hue * 6).astype(int)
                f = hue * 6 - h_i
                p = value * (1 - saturation)
                q = value * (1 - f * saturation)
                t_val = value * (1 - (1 - f) * saturation)

                h_i = h_i % 6
                r = np.where(
                    h_i == 0,
                    value,
                    np.where(h_i == 1, q, np.where(h_i == 2, p, np.where(h_i == 3, p, np.where(h_i == 4, t_val, value))))
                )
                g = np.where(
                    h_i == 0,
                    t_val,
                    np.where(h_i == 1, value, np.where(h_i == 2, value, np.where(h_i == 3, q, np.where(h_i == 4, p, p))))
                )
                b = np.where(
                    h_i == 0,
                    p,
                    np.where(h_i == 1, p, np.where(h_i == 2, t_val, np.where(h_i == 3, value, np.where(h_i == 4, value, q))))
                )

                cluster_colors = np.column_stack([r, g, b]).astype(np.float32)

                # Varying radii (smaller at edges, larger at center)
                cluster_radii = (0.5 + 2.0 * (1 - t)).astype(np.float32)

                all_positions.append(cluster_points)
                all_colors.append(cluster_colors)
                all_radii.append(cluster_radii)

                aprint(
                    f"✓ Cluster center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), "
                    f"spread: {spread:.1f}, points: {points_per_cluster:,}"
                )

        # Combine all clusters
        positions = np.vstack(all_positions)
        colors = np.vstack(all_colors)
        radii = np.concatenate(all_radii)

        aprint(
            f"✓ Combined {n_clusters} clusters into {len(positions):,} total points"
        )

    # Write to zarr with performance-optimized settings
    with asection("Writing to Zarr (optimized for large datasets)"):
        dims = Dimensions(
            [
                Dimension("time", unit="frame", range=(0, n_timesteps - 1), step=1, discrete=True, display=False),
                Dimension("x", unit="units", range=(-150, 150), display=True),
                Dimension("y", unit="units", range=(-150, 150), display=True),
                Dimension("z", unit="units", range=(-150, 150), display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "PerformanceTestData",
                positions,
                colors=colors,
                radii=radii,
                opacity=0.8,
                blending_mode="additive",
            )

        total_points = len(positions)
        aprint(f"✓ Written {total_points:,} points ({n_points:,} × {n_timesteps} timesteps) to {output_path}")
        aprint(f"✓ Dataset size: ~{total_points * 48 / 1_000_000:.1f} MB (4D positions)")
        aprint(f"✓ Time dimension: Navigate with keys [4] then [ / ] to load different timesteps")


def main() -> None:
    """Main demo entry point."""
    # Parse command line arguments
    n_points = 1000000  # 1M points by default
    network_profile = "slow-broadband"  # Slow broadband by default
    no_serve = False
    no_simulation = False

    for arg in sys.argv[1:]:
        if arg.startswith("--points="):
            n_points = int(arg.split("=")[1])
        elif arg.startswith("--profile="):
            network_profile = arg.split("=")[1]
        elif arg == "--no-serve":
            no_serve = True
        elif arg == "--no-simulation":
            no_simulation = True

    # Display demo information
    aprint("=" * 70)
    aprint("NETWORK PERFORMANCE TESTING DEMO (4D Dataset)")
    aprint("=" * 70)
    aprint("")
    aprint("This demo generates a large 4D dataset and serves it with network")
    aprint("simulation to test viewer performance under realistic conditions.")
    aprint("")
    aprint(f"Dataset: {n_points:,} points × 10 timesteps = {n_points * 10:,} total points")
    aprint(f"Size: ~{n_points * 10 * 48 / 1_000_000:.1f} MB (4D positions with time)")
    aprint(f"Use [4] then [ / ] keys to navigate time and trigger progressive loading")

    if not no_simulation:
        aprint(f"Network Profile: {network_profile}")
        aprint("")
        aprint("Watch browser DevTools (Network tab) to see throttling in action!")
        aprint("Compare loading times with different profiles to test performance.")
    else:
        aprint("Network Simulation: DISABLED (full speed)")

    aprint("")

    # Use temporary directory for demo data
    with tempfile.TemporaryDirectory(prefix="luxar_demo_network_perf_") as tmpdir:
        output_path = Path(tmpdir) / "performance_test.zarr"

        # Generate the dataset
        generate_performance_test_dataset(output_path, n_points=n_points)

        if no_serve:
            aprint("")
            aprint("✓ Dataset generated successfully (--no-serve mode)")
            return

        # Display serving information
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER WITH NETWORK SIMULATION")
        aprint("=" * 70)
        aprint("")

        if no_simulation:
            aprint("Serving at full speed (no network simulation)")
            aprint("")
            cmd = ["luxar", "serve", str(output_path), "--viewer", "--open"]
        else:
            aprint(f"Simulating {network_profile} network conditions:")
            aprint("")
            aprint("What to observe:")
            aprint("  1. Browser DevTools Network tab shows throttled requests")
            aprint("  2. Chunks load progressively (not all at once)")
            aprint("  3. UI remains responsive during loading")
            aprint("  4. Cache reduces redundant requests")
            aprint("")
            aprint("Try different profiles to compare:")
            aprint("  python demo_network_performance.py --profile 3g")
            aprint("  python demo_network_performance.py --profile broadband")
            aprint("  python demo_network_performance.py --no-simulation")
            aprint("")

            cmd = [
                "luxar",
                "serve",
                str(output_path),
                "--viewer",
                "--open",
                "--profile",
                network_profile,
            ]

        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")

        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            aprint("\n🛑 Stopping demo...")
        except subprocess.CalledProcessError as e:
            aprint(f"\n❌ Error: {e}")
            aprint(
                "💡 Make sure the luxar viewer is built: cd packages/luxar-viewer && pnpm build"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
