#!/usr/bin/env python3
"""Memory optimization example using different data types.

This example demonstrates how to use different data types for Luxar points
to optimize memory usage and bandwidth. It shows the trade-offs between precision
and memory efficiency.
"""

import os
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import LuxarZarrCompiler
from luxar.typing_utils.datatypes import DataTypeConfig, DataTypeMode


def create_test_data(n_points: int = 100000):
    """Create test points data with known ranges."""
    # Create positions in a reasonable range (can use float16)
    positions = np.random.randn(n_points, 3).astype(np.float32) * 10

    # Create SDR colors (0-1 range, suitable for uint8)
    colors = np.random.rand(n_points, 3).astype(np.float32)

    # Create radii in a small range (suitable for uint8 with normalization)
    radii = np.random.rand(n_points).astype(np.float32) * 0.5 + 0.1

    # Create sharpness values (small range, suitable for uint8)
    sharpness = np.random.rand(n_points).astype(np.float32) * 3 + 0.5

    return positions, colors, radii, sharpness


def get_zarr_size(path: Path) -> int:
    """Calculate the total size of a zarr store in bytes."""
    total_size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = Path(root) / file
            total_size += file_path.stat().st_size
    return total_size


def create_dataset_with_dtype_config(
    output_path: Path,
    dtype_config: DataTypeConfig,
    positions: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    sharpness: np.ndarray,
    description: str,
):
    """Create a dataset with specific dtype configuration."""
    aprint(f"\n{'=' * 60}")
    aprint(f"Creating dataset: {description}")
    aprint(f"Output: {output_path}")

    # Create the dataset
    with LuxarZarrCompiler(output_path, dtype_config=dtype_config) as compiler:
        scene = compiler.create_scene()
        scene.add_points(
            "optimized_points",
            positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            opacity=0.9,
            gamma=1.2,
        )

    # Report file size
    size_mb = get_zarr_size(output_path) / (1024 * 1024)
    aprint(f"✓ Dataset created: {size_mb:.2f} MB")

    # Report data types used
    import zarr

    store = zarr.open_group(output_path, mode="r")
    points_group = store["optimized_points"]

    aprint("Data types used:")
    aprint(f"  - positions: {points_group.attrs.get('position_dtype', 'float32')}")
    aprint(f"  - colors: {points_group.attrs.get('color_dtype', 'float32')}")
    aprint(f"  - radii: {points_group.attrs.get('radius_dtype', 'float32')}")
    aprint(f"  - sharpness: {points_group.attrs.get('sharpness_dtype', 'float32')}")

    return size_mb


def main():
    """Run memory optimization comparison."""
    with asection("Memory Optimization Setup"):
        aprint("=" * 70)
        aprint("LUXAR MEMORY OPTIMIZATION EXAMPLE")
        aprint("=" * 70)

        # Generate test data
        n_points = 100000
        aprint(f"Generating test data with {n_points:,} points...")
        positions, colors, radii, sharpness = create_test_data(n_points)

        # Create output directory
        output_dir = Path("delme")
        output_dir.mkdir(exist_ok=True)

    with asection("DataType Configuration Comparison"):
        # Test different dtype configurations
        configs = [
            (
                DataTypeConfig(mode=DataTypeMode.PRECISION),
                "memory_precision_example.zarr",
                "Maximum Precision (all float32)",
            ),
            (
                DataTypeConfig(mode=DataTypeMode.MEMORY),
                "memory_efficient_example.zarr",
                "Memory Efficient (mixed types)",
            ),
            (
                DataTypeConfig(mode=DataTypeMode.AUTO),
                "memory_auto_example.zarr",
                "Auto-detected types",
            ),
            (
                DataTypeConfig(
                    mode=DataTypeMode.CUSTOM,
                    position_dtype="float32",
                    color_dtype="uint8",
                    radius_dtype="uint8",
                    sharpness_dtype="uint8",
                ),
                "memory_custom_example.zarr",
                "Custom (float32 positions, uint8 attributes)",
            ),
        ]

        aprint(f"Testing {len(configs)} different dtype configurations...")
        sizes = []
        for config, filename, description in configs:
            output_path = output_dir / filename
            size_mb = create_dataset_with_dtype_config(
                output_path, config, positions, colors, radii, sharpness, description
            )
            sizes.append((description, size_mb))

    with asection("Memory Usage Comparison Results"):
        # Print comparison summary
        aprint("=" * 60)
        aprint("MEMORY USAGE COMPARISON")
        aprint("=" * 60)

        baseline_size = sizes[0][1]  # Precision mode as baseline
        for description, size_mb in sizes:
            reduction = ((baseline_size - size_mb) / baseline_size) * 100
            if reduction > 0:
                aprint(
                    f"{description:50s}: {size_mb:6.2f} MB ({reduction:+.1f}% reduction)"
                )
            else:
                aprint(f"{description:50s}: {size_mb:6.2f} MB (baseline)")

    with asection("Theoretical Memory Analysis"):
        # Calculate theoretical memory usage
        aprint("=" * 60)
        aprint("THEORETICAL MEMORY USAGE (uncompressed)")
        aprint("=" * 60)

        # Float32 everything (baseline)
        float32_size = (
            n_points * (3 * 4 + 3 * 4 + 4 + 4) / (1024 * 1024)
        )  # positions + colors + radius + sharpness
        aprint(f"All float32:                     {float32_size:6.2f} MB")

        # Mixed precision
        mixed_size = (
            n_points * (3 * 4 + 3 * 1 + 1 + 1) / (1024 * 1024)
        )  # float32 pos + uint8 rest
        aprint(f"Float32 pos + uint8 attributes:  {mixed_size:6.2f} MB")
        reduction = ((float32_size - mixed_size) / float32_size) * 100
        aprint(f"Theoretical reduction:            {reduction:.1f}%")

    with asection("Usage Notes and Viewing Instructions"):
        aprint("=" * 60)
        aprint("NOTES:")
        aprint("- Actual file sizes are smaller due to compression")
        aprint("- uint8 types use normalization for 0-1 range in WebGL")
        aprint("- Float16 support depends on browser/hardware capabilities")
        aprint("- Choose dtype based on your precision requirements")
        aprint("=" * 60)

        # Serve the most memory-efficient example
        aprint("To view the memory-efficient example, run:")
        aprint(f"  luxar serve {output_dir}/memory_efficient_example.zarr")


if __name__ == "__main__":
    main()
