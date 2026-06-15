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

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def generate_performance_test_dataset(
    output_path: Path, n_points: int = 1000000, seed: int = 42
) -> None:
    """Generate a large, complex 4D spatial dataset for performance testing.

    Creates a visually interesting "particle cloud" with:
    - Multiple clustered regions (tests spatial queries)
    - Varying densities (tests LOD and culling)
    - Color gradients (tests color encoding/decoding)
    - Mixed radii (tests radius-based slicing)
    - Continuous 4D spatial distribution (tests nD navigation and slicing)

    This dataset is designed to stress-test:
    - Progressive loading under bandwidth constraints
    - Cache effectiveness with limited bandwidth
    - Spatial index query performance in 4D
    - UI responsiveness during slow loading
    - nD navigation bandwidth usage (slider movement loads new data progressively)
    - Smooth continuous slicing through 4D space

    Args:
        output_path: Where to write the zarr store
        n_points: Total number of 4D points
        seed: Random seed for reproducibility
    """
    with asection(f"Generating 4D Performance Test Dataset ({n_points:,} points)"):
        rng = np.random.default_rng(seed)

        aprint(f"Seed: {seed}")
        aprint("Points continuously distributed across 4D space (W, X, Y, Z)")
        aprint("W dimension: [-50, 50] (use slider for smooth navigation)")
        aprint("Creating multi-cluster particle system...")

        # Generate multiple clusters with different characteristics
        n_clusters = 8
        points_per_cluster = n_points // n_clusters
        all_positions = []
        all_colors = []
        all_radii = []

        for i in range(n_clusters):
            with asection(f"Cluster {i + 1}/{n_clusters}"):
                # Random cluster center (3D spatial)
                center = rng.uniform(-100, 100, size=3).astype(np.float32)

                # Varying cluster sizes (some tight, some spread out)
                spread = rng.uniform(5, 30)

                # Generate 4D spatial points in this cluster
                # W dimension: uniformly distributed across full range for continuous slicing
                w_coords = rng.uniform(-50, 50, size=(points_per_cluster, 1)).astype(
                    np.float32
                )

                # X, Y, Z dimensions: Gaussian distribution around cluster center
                xyz_coords = rng.normal(
                    loc=center, scale=spread, size=(points_per_cluster, 3)
                ).astype(np.float32)

                # Combine into 4D positions (W, X, Y, Z)
                cluster_points = np.hstack(
                    [w_coords, xyz_coords]
                )  # Shape: (points_per_cluster, 4)

                # Color based on cluster position and distance from center (3D spatial only)
                spatial_positions = cluster_points[
                    :, 1:4
                ]  # Extract x, y, z (skip time dimension)
                distances = np.linalg.norm(spatial_positions - center, axis=1)
                max_dist = distances.max()

                # Create color gradient from cluster center (blue) to edge (red)
                t = distances / max_dist  # 0 at center, 1 at edge

                # Use HSV color space for smooth gradients
                hue = (
                    i / n_clusters + t * 0.3
                ) % 1.0  # Vary hue by cluster and distance
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
                    np.where(
                        h_i == 1,
                        q,
                        np.where(
                            h_i == 2,
                            p,
                            np.where(h_i == 3, p, np.where(h_i == 4, t_val, value)),
                        ),
                    ),
                )
                g = np.where(
                    h_i == 0,
                    t_val,
                    np.where(
                        h_i == 1,
                        value,
                        np.where(
                            h_i == 2,
                            value,
                            np.where(h_i == 3, q, np.where(h_i == 4, p, p)),
                        ),
                    ),
                )
                b = np.where(
                    h_i == 0,
                    p,
                    np.where(
                        h_i == 1,
                        p,
                        np.where(
                            h_i == 2,
                            t_val,
                            np.where(h_i == 3, value, np.where(h_i == 4, value, q)),
                        ),
                    ),
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

        aprint(f"✓ Combined {n_clusters} clusters into {len(positions):,} total points")

    # Write to zarr with performance-optimized settings
    with asection("Writing to Zarr (optimized for large datasets)"):
        dims = Dimensions(
            [
                # W is the slicing dimension (slider appears because display=False)
                # IMPORTANT: spatial=True for radius-based slicing to work!
                Dimension(
                    "w",
                    unit="units",
                    range=(-50, 50),
                    step=0.01,
                    discrete=False,
                    spatial=True,
                    display=False,
                ),
                # X, Y, Z are the displayed dimensions (shown in 3D space)
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
                intensity=0.1,
            )

            # --- Overlays ---
            # Title
            scene.add_text(
                "Network Performance Test",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info
            scene.add_text(
                "10M+ points \u2022 Stress test",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        total_points = len(positions)
        aprint(f"✓ Written {total_points:,} points (4D: W, X, Y, Z) to {output_path}")
        aprint(
            f"✓ Dataset size: ~{total_points * 48 / 1_000_000:.1f} MB (4D positions + colors + radii)"
        )
        aprint("✓ W range: [-50, 50] continuously distributed")
        aprint(
            "✓ 4D Navigation: Use slider or press [1] then [ / ] to navigate through W dimension"
        )
        aprint("✓ Moving slider will show smooth continuous slicing through 4D space")


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
    aprint(
        f"Dataset: {n_points:,} points × 10 timesteps = {n_points * 10:,} total points"
    )
    aprint(f"Size: ~{n_points * 10 * 48 / 1_000_000:.1f} MB (4D positions with time)")
    aprint("Use [4] then [ / ] keys to navigate time and trigger progressive loading")

    if not no_simulation:
        aprint(f"Network Profile: {network_profile}")
        aprint("")
        aprint("Watch browser DevTools (Network tab) to see throttling in action!")
        aprint("Compare loading times with different profiles to test performance.")
    else:
        aprint("Network Simulation: DISABLED (full speed)")

    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if no_serve:
        output_path = get_demos_output_dir() / "performance_test.luxar.zarr"
        generate_performance_test_dataset(output_path, n_points=n_points)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_network_perf_") as tmpdir:
        output_path = Path(tmpdir) / "performance_test.luxar.zarr"

        # Generate the dataset
        generate_performance_test_dataset(output_path, n_points=n_points)

        # Display serving information
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER WITH NETWORK SIMULATION")
        aprint("=" * 70)
        aprint("")

        if no_simulation:
            aprint("Serving at full speed (no network simulation)")
            aprint("")
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

        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")

        launch_viewer(output_path)


if __name__ == "__main__":
    main()
