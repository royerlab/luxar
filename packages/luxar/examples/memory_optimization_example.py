#!/usr/bin/env python3
"""Memory optimization example using different encoding modes.

This example demonstrates how to use different encoding modes for Luxar scenes
to optimize memory usage and bandwidth. It shows the trade-offs between precision
and memory efficiency.

The new encoding system uses EncodingMode from luxar.encoding:
- AUTO: Automatically analyze data and select appropriate encoding
- PRECISION: Full float32 precision for all arrays
- MEMORY: Aggressive quantization for minimum storage
"""

import os
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import LuxarZarrCompiler
from luxar.encoding import EncodingMode


def create_test_data(n_points: int = 100000):
    """Create test points data with known ranges."""
    # Create positions in a reasonable range
    positions = np.random.randn(n_points, 3).astype(np.float32) * 10

    # Create SDR colors (0-1 range)
    colors = np.random.rand(n_points, 3).astype(np.float32)

    # Create radii in a small range
    radii = np.random.rand(n_points).astype(np.float32) * 0.5 + 0.1

    # Create sharpness values
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


def create_dataset_with_encoding_mode(
    output_path: Path,
    encoding_mode: EncodingMode,
    positions: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    sharpness: np.ndarray,
    description: str,
):
    """Create a dataset with specific encoding mode."""
    aprint(f"\n{'=' * 60}")
    aprint(f"Creating dataset: {description}")
    aprint(f"Output: {output_path}")
    aprint(f"Encoding mode: {encoding_mode.value}")

    # Create the dataset
    with LuxarZarrCompiler(output_path, encoding_mode=encoding_mode) as compiler:
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
    aprint(f"Dataset created: {size_mb:.2f} MB")

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

    with asection("Encoding Mode Comparison"):
        # Test different encoding modes
        configs = [
            (
                EncodingMode.PRECISION,
                "memory_precision_example.zarr",
                "Maximum Precision (float32 for all arrays)",
            ),
            (
                EncodingMode.MEMORY,
                "memory_efficient_example.zarr",
                "Memory Efficient (aggressive quantization)",
            ),
            (
                EncodingMode.AUTO,
                "memory_auto_example.zarr",
                "Auto-detected encoding",
            ),
        ]

        aprint(f"Testing {len(configs)} different encoding modes...")
        sizes = []
        for encoding_mode, filename, description in configs:
            output_path = output_dir / filename
            size_mb = create_dataset_with_encoding_mode(
                output_path,
                encoding_mode,
                positions,
                colors,
                radii,
                sharpness,
                description,
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

    with asection("Encoding Mode Guide"):
        aprint("=" * 60)
        aprint("ENCODING MODE GUIDE")
        aprint("=" * 60)
        aprint("")
        aprint("EncodingMode.PRECISION:")
        aprint("  - Uses float32 for all arrays")
        aprint("  - Maximum precision, no quantization error")
        aprint("  - Best for scientific accuracy requirements")
        aprint("")
        aprint("EncodingMode.MEMORY:")
        aprint("  - Aggressive quantization (uint8, uint16, float16)")
        aprint("  - Smallest file size")
        aprint("  - May introduce small quantization errors")
        aprint("  - Best for large datasets and streaming")
        aprint("")
        aprint("EncodingMode.AUTO:")
        aprint("  - Analyzes data and selects appropriate encoding")
        aprint("  - Balanced approach between precision and size")
        aprint("  - Good default choice")
        aprint("")
        aprint("Note: Broadcasting and LUT optimizations apply in ALL modes")
        aprint("as they are lossless. Mode only affects quantization.")
        aprint("=" * 60)

        # Serve the most memory-efficient example
        aprint("To view the memory-efficient example, run:")
        aprint(f"  luxar serve {output_dir}/memory_efficient_example.zarr")


if __name__ == "__main__":
    main()
