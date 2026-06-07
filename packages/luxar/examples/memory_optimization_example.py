#!/usr/bin/env python3
"""Memory Optimization Example - Comparing encoding modes for storage efficiency.

This example demonstrates:
- Three encoding modes: AUTO, PRECISION, MEMORY
- How encoding affects file size and data quality
- Trade-offs between precision and storage/bandwidth
- Measuring and comparing zarr store sizes
- When to use each encoding mode in production

Educational value:
- Understand encoding mode implications for real-world use
- Learn to make informed precision vs size trade-offs
- See actual file size differences with identical data
- Master the EncodingMode API for production deployments

Use cases by mode:
- AUTO: Smart default for most cases (automatic compression)
- PRECISION: Scientific data where accuracy is critical
- MEMORY: Web delivery where bandwidth/storage is limited

Key principle:
- Encoding happens during write, decoding during read
- Quality loss only in MEMORY mode (quantization)
- AUTO mode intelligently chooses based on data characteristics
"""

import os
from pathlib import Path

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.utils.paths import get_examples_output_dir

# Per-mode explainer content: each scene notes the encoding-mode trade-off.
_MODE_EXPLAINERS = {
    EncodingMode.PRECISION: {
        "title": "Encoding: PRECISION",
        "body": (
            "<code>EncodingMode.PRECISION</code> stores every array as "
            "<code>float32</code> with no quantization — the largest file, "
            "but bit-exact data. Use it when scientific accuracy is critical."
        ),
        "observe": [
            "This is the size baseline the other modes are compared against.",
            "Colors and radii match the source data exactly.",
            "No quantization banding appears anywhere in the cloud.",
        ],
    },
    EncodingMode.MEMORY: {
        "title": "Encoding: MEMORY",
        "body": (
            "<code>EncodingMode.MEMORY</code> applies aggressive quantization "
            "(uint8/uint16/float16) for the smallest file and least bandwidth, "
            "at the cost of small quantization errors. Best for web delivery."
        ),
        "observe": [
            "The store is markedly smaller than the PRECISION baseline.",
            "Geometry looks the same; tiny color/radius rounding may appear.",
            "Ideal for streaming large clouds over the network.",
        ],
    },
    EncodingMode.AUTO: {
        "title": "Encoding: AUTO",
        "body": (
            "<code>EncodingMode.AUTO</code> inspects the data and picks an "
            "encoding per array, balancing precision against size. It is the "
            "recommended default for most scenes."
        ),
        "observe": [
            "File size lands between the PRECISION and MEMORY extremes.",
            "Lossless broadcasting and LUT optimizations still apply.",
            "Visual quality is close to PRECISION with smaller storage.",
        ],
    },
}


def create_test_data(n_points: int = 100000):
    """Create test points data with known ranges.

    Args:
        n_points: Number of points to generate

    Returns:
        Tuple of (positions, colors, radii, sharpness) arrays
    """
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
    """Calculate the total size of a zarr store in bytes.

    Args:
        path: Path to the zarr store directory

    Returns:
        Total size in bytes of all files in the store
    """
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
    """Create a dataset with specific encoding mode.

    Args:
        output_path: Path to write the zarr store
        encoding_mode: Encoding mode to use (AUTO, PRECISION, or MEMORY)
        positions: Point positions array
        colors: Point colors array
        radii: Point radii array
        sharpness: Point sharpness array
        description: Human-readable description of this configuration

    Returns:
        File size of the created dataset in MB
    """
    aprint(f"\n{'=' * 60}")
    aprint(f"Creating dataset: {description}")
    aprint(f"Output: {output_path}")
    aprint(f"Encoding mode: {encoding_mode.value}")

    # Create the dataset
    with LuxarZarrCompiler(output_path, encoding_mode=encoding_mode) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "optimized_points",
            positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            opacity=0.9,
            gamma=1.2,
        )

        spec = _MODE_EXPLAINERS[encoding_mode]
        add_explainer(
            scene,
            title=spec["title"],
            body=spec["body"],
            observe=spec["observe"],
            observe_label="Look for",
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
        output_dir = get_examples_output_dir()

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
