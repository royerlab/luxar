#!/usr/bin/env python3
"""Performance Benchmark Example - Stress test for large scenes with many nodes and materials.

This benchmark example demonstrates:
- Creating large scenes with many points nodes
- Testing material combination caching efficiency
- Measuring scene creation performance
- Generating systematic test data for performance analysis
- Educational timing and profiling techniques

Educational value:
- Learn how to benchmark scene creation performance
- Understand material combination caching and reuse patterns
- See systematic test data generation for performance testing
- Understand how to profile and measure creation speed
- Provides reference performance numbers for large scenes
"""

import time

import numpy as np
from arbol import aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def create_node_positions(
    node_index: int,
    points_per_node: int,
    grid_spacing: float = 3.0,
    cluster_size: float = 0.5,
) -> np.ndarray:
    """Create positioned point cluster for a specific node in 3D grid layout.

    Args:
        node_index: Index of the node (for grid positioning)
        points_per_node: Number of points in this cluster
        grid_spacing: Distance between node centers
        cluster_size: Standard deviation of points within cluster

    Returns:
        Array of 3D positions for this node's points
    """
    # Calculate 3D grid position for this node
    grid_size = 10  # 10x10x10 grid can hold 1000 nodes
    x_grid = (node_index % grid_size) * grid_spacing
    y_grid = ((node_index // grid_size) % grid_size) * grid_spacing
    z_grid = (node_index // (grid_size * grid_size)) * grid_spacing

    # Create cluster center position
    center = np.array([x_grid, y_grid, z_grid])

    # Generate points around center with normal distribution
    positions = np.random.randn(points_per_node, 3) * cluster_size
    positions += center

    return positions.astype(np.float32)


def generate_test_colors(points_per_node: int, color_seed: int) -> np.ndarray:
    """Generate deterministic but varied colors for consistent benchmarking.

    Args:
        points_per_node: Number of color values needed
        color_seed: Seed for deterministic color generation

    Returns:
        Array of RGB colors

    NOTE: This uses np.random.seed (legacy global RNG) intentionally — the
    resulting zarr output is consumed by viewer E2E tests
    (worker-wasm-integration.spec.ts uses it as DATASET_LARGE). Changing
    the RNG mechanism would alter every color value and require updating
    the E2E baselines. See packages/luxar/examples/README.md for the
    "examples are also test fixtures" convention.
    """
    np.random.seed(color_seed)
    colors = np.random.uniform(0.2, 1.0, (points_per_node, 3)).astype(np.float32)
    return colors


def main():
    """Run performance benchmark creating large scene with many materials."""
    output_path = get_examples_output_dir() / "performance_benchmark_example.zarr"

    with asection("Benchmark Setup and Configuration"):
        aprint(f"Starting performance benchmark at {output_path}")
        aprint("This benchmark tests:")
        aprint("- Large scene creation performance")
        aprint("- Material combination caching efficiency")
        aprint("- Memory usage patterns")
        aprint("- Systematic node generation")

        # Benchmark parameters
        num_nodes = 100
        points_per_node = 1000
        total_points = num_nodes * points_per_node

        aprint("Benchmark Configuration:")
        aprint(f"- Nodes: {num_nodes}")
        aprint(f"- Points per node: {points_per_node:,}")
        aprint(f"- Total points: {total_points:,}")
        aprint(
            f"- Expected file size: ~{(total_points * 20) // 1024 // 1024}MB"
        )  # Rough estimate

    # Create scene
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        with asection("Material Property Setup"):
            # Define rendering property cycles for material variety
            blending_modes = ["normal", "additive"]
            opacities = [0.3, 0.5, 0.7, 0.9, 1.0]
            gammas = [0.8, 1.0, 1.2, 1.5]
            radii_values = [0.05, 0.1, 0.15]  # Different point sizes

            # Track unique material combinations for cache analysis
            material_combinations = set()
            aprint("Material variation combinations available:")
            aprint(f"- Blending modes: {len(blending_modes)}")
            aprint(f"- Opacity levels: {len(opacities)}")
            aprint(f"- Gamma levels: {len(gammas)}")
            aprint(f"- Radius sizes: {len(radii_values)}")

        with asection("Node Creation Performance Test"):
            aprint(f"Creating {num_nodes} nodes with varied material properties...")
            start_time = time.time()

            # Progress tracking
            progress_interval = max(1, num_nodes // 10)

            for i in range(num_nodes):
                # Show progress
                if i % progress_interval == 0:
                    aprint(
                        f"  Progress: {i}/{num_nodes} nodes ({i / num_nodes * 100:.1f}%)"
                    )

                # Generate deterministic positions for this node
                positions = create_node_positions(i, points_per_node)

                # Cycle through different rendering properties systematically
                blending = blending_modes[i % len(blending_modes)]
                opacity = opacities[i % len(opacities)]
                gamma = gammas[i % len(gammas)]
                radius = radii_values[i % len(radii_values)]

                # Track unique material combinations for analysis
                material_key = (
                    blending,
                    round(opacity, 2),
                    round(gamma, 2),
                    round(radius, 3),
                )
                material_combinations.add(material_key)

                # Generate deterministic colors using node index as seed.
                # Uses the legacy global RNG by design — see helper docstring.
                colors = generate_test_colors(points_per_node, i)

                # Create the node
                scene.add_points(
                    f"BenchmarkNode_{i:03d}",
                    positions,
                    colors=colors,
                    radii=radius,
                    blending_mode=blending,
                    opacity=opacity,
                    gamma=gamma,
                )

            creation_time = time.time() - start_time

    # The context manager finalizes on exit; measure the full block above
    # rather than a no-op interval inside it.

    with asection("Performance Analysis and Results"):
        aprint("=" * 60)
        aprint("PERFORMANCE BENCHMARK RESULTS")
        aprint("=" * 60)
        aprint("Scene Creation Performance:")
        aprint(f"- Node creation time (inside compiler): {creation_time:.2f} seconds")
        aprint(f"- Points per second: {total_points / creation_time:,.0f}")
        aprint(f"- Nodes per second: {num_nodes / creation_time:.1f}")

        aprint("\nMaterial Combination Analysis:")
        aprint(f"- Unique material combinations: {len(material_combinations)}")
        aprint(
            f"- Theoretical maximum: {len(blending_modes) * len(opacities) * len(gammas) * len(radii_values)}"
        )
        aprint(
            f"- Cache efficiency: {len(material_combinations)}/{len(blending_modes) * len(opacities) * len(gammas) * len(radii_values)} combinations used"
        )

        aprint("\nScene Statistics:")
        aprint(f"- Total nodes: {num_nodes}")
        aprint(f"- Total points: {total_points:,}")
        aprint(f"- Average points per node: {points_per_node}")
        aprint("- Spatial distribution: 10×10×10 grid layout")

        aprint("\nRendering Performance Test:")
        aprint("This scene is designed to stress-test the renderer with:")
        aprint("- High point density")
        aprint("- Many different material combinations")
        aprint("- Spatial clustering for occlusion testing")
        aprint("- Mixed transparency and blending modes")

        aprint("\nTo test rendering performance:")
        aprint(f"  luxar serve {output_path}")
        aprint("Monitor frame rates and memory usage while navigating!")
        aprint("=" * 60)


if __name__ == "__main__":
    main()
