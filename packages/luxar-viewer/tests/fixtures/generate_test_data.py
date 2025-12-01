#!/usr/bin/env python3
"""Generate test datasets for TypeScript-Python encoding compatibility tests.

This script creates small zarr datasets with all encoding modes to verify
that the TypeScript ArrayDecoder can correctly read Python-encoded data.

IMPORTANT: Uses NO compression (compressor=None) to avoid blosc/numcodecs
WASM binding issues in Node.js test environment.

Run from project root:
    hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py
"""

from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode

# Output directory
FIXTURES_DIR = Path(__file__).parent
FIXTURES_DIR.mkdir(exist_ok=True)


def generate_broadcasting_test():
    """Test dataset with broadcasted (uniform) values."""
    with asection("Generating Broadcasting Test"):
        output = FIXTURES_DIR / "test_broadcasting.zarr"

        # Create 1000 points at different positions
        # But all with the SAME color (perfect for broadcasting)
        positions = np.random.randn(1000, 3).astype(np.float32) * 10

        # Single uniform color (will be broadcasted)
        uniform_color = np.array([[1.0, 0.5, 0.25]], dtype=np.float32)

        # Single uniform radius
        uniform_radius = np.array([0.5], dtype=np.float32)

        dims = Dimensions([
            Dimension("x", unit="units", display=True),
            Dimension("y", unit="units", display=True),
            Dimension("z", unit="units", display=True),
        ])

        # Disable compression for Node.js compatibility (blosc has WASM issues)
        with LuxarZarrCompiler(
            output, encoding_mode=EncodingMode.MEMORY, compressor=None
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "points",
                positions,
                colors=uniform_color,  # (1, 3) - will broadcast to (1000, 3)
                radii=uniform_radius,  # (1,) - will broadcast to (1000,)
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Colors: {uniform_color.shape} (broadcasted)")
        aprint(f"  Radii: {uniform_radius.shape} (broadcasted)")


def generate_lut_test():
    """Test dataset with LUT encoding (≤256 unique values)."""
    with asection("Generating LUT Encoding Test"):
        output = FIXTURES_DIR / "test_lut.zarr"

        # 1000 points with only 10 unique colors (perfect for LUT)
        positions = np.random.randn(1000, 3).astype(np.float32) * 10

        # Create 10 unique colors
        unique_colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
            [0.0, 0.5, 0.5],  # Teal
            [0.5, 0.5, 0.5],  # Gray
        ], dtype=np.float32)

        # Randomly assign colors (indices 0-9)
        color_indices = np.random.randint(0, 10, size=1000)
        colors = unique_colors[color_indices]

        dims = Dimensions([
            Dimension("x", unit="units", display=True),
            Dimension("y", unit="units", display=True),
            Dimension("z", unit="units", display=True),
        ])

        with LuxarZarrCompiler(output, encoding_mode=EncodingMode.MEMORY, compressor=None) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "points",
                positions,
                colors=colors,  # Will use LUT (only 10 unique values)
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Colors: {colors.shape} ({len(unique_colors)} unique - LUT)")


def generate_quantization_test():
    """Test dataset with quantized arrays (uint8/uint16)."""
    with asection("Generating Quantization Test"):
        output = FIXTURES_DIR / "test_quantization.zarr"

        # 1000 points with bounded radii (perfect for quantization)
        positions = np.random.randn(1000, 3).astype(np.float32) * 10

        # Radii in range [0.1, 2.0] - will be quantized to uint8
        radii = np.random.uniform(0.1, 2.0, size=1000).astype(np.float32)

        # Colors in [0, 1] - will be quantized to uint8
        colors = np.random.rand(1000, 3).astype(np.float32)

        dims = Dimensions([
            Dimension("x", unit="units", display=True),
            Dimension("y", unit="units", display=True),
            Dimension("z", unit="units", display=True),
        ])

        with LuxarZarrCompiler(output, encoding_mode=EncodingMode.MEMORY, compressor=None) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "points",
                positions,
                colors=colors,  # Will quantize to uint8
                radii=radii,    # Will quantize to uint8
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Colors: {colors.shape} (quantized)")
        aprint(f"  Radii: {radii.shape} (quantized)")


def generate_array_refs_test():
    """Test dataset with array references (deduplication)."""
    with asection("Generating Array References Test"):
        output = FIXTURES_DIR / "test_array_refs.zarr"

        # Create two point groups with IDENTICAL colors
        # (should deduplicate via array references)
        positions1 = np.random.randn(500, 3).astype(np.float32) * 10
        positions2 = np.random.randn(500, 3).astype(np.float32) * 10

        # Same colors for both groups
        shared_colors = np.random.rand(500, 3).astype(np.float32)

        dims = Dimensions([
            Dimension("x", unit="units", display=True),
            Dimension("y", unit="units", display=True),
            Dimension("z", unit="units", display=True),
        ])

        with LuxarZarrCompiler(output, encoding_mode=EncodingMode.MEMORY, compressor=None) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add both point clouds with same colors (should deduplicate via array ref)
            scene.add_points("points1", positions1, colors=shared_colors)
            scene.add_points("points2", positions2, colors=shared_colors)

        aprint(f"✓ Created {output}")
        aprint(f"  Points1: {positions1.shape}")
        aprint(f"  Points2: {positions2.shape}")
        aprint(f"  Shared colors: {shared_colors.shape} (deduplicated)")


def generate_mixed_encoding_test():
    """Test dataset with mixed encoding modes in same scene."""
    with asection("Generating Mixed Encoding Test"):
        output = FIXTURES_DIR / "test_mixed.zarr"

        # Group 1: Broadcasting
        pos1 = np.random.randn(500, 3).astype(np.float32) * 10
        uniform_color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        # Group 2: LUT
        pos2 = np.random.randn(500, 3).astype(np.float32) * 10
        lut_colors = np.tile(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32), (250, 1))

        # Group 3: Direct (no encoding)
        pos3 = np.random.randn(500, 3).astype(np.float32) * 10
        direct_colors = np.random.rand(500, 3).astype(np.float32)

        dims = Dimensions([
            Dimension("x", unit="units", display=True),
            Dimension("y", unit="units", display=True),
            Dimension("z", unit="units", display=True),
        ])

        with LuxarZarrCompiler(output, encoding_mode=EncodingMode.MEMORY, compressor=None) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add points with different encoding modes
            scene.add_points("uniform", pos1, colors=uniform_color)  # Broadcasting
            scene.add_points("lut", pos2, colors=lut_colors)  # LUT

            # For direct encoding, temporarily switch mode
            saved_mode = compiler._encoding_mode
            compiler._encoding_mode = EncodingMode.PRECISION
            scene.add_points("direct", pos3, colors=direct_colors)  # Direct
            compiler._encoding_mode = saved_mode

        aprint(f"✓ Created {output}")
        aprint(f"  Points 'uniform': Broadcasting")
        aprint(f"  Points 'lut': LUT encoding")
        aprint(f"  Points 'direct': No encoding")


def generate_4d_test():
    """Test dataset with 4D data (time dimension) for nD slicing tests."""
    with asection("Generating 4D nD Slicing Test"):
        output = FIXTURES_DIR / "test_4d.zarr"

        # 4D data: 500 points × 10 time steps = 5000 total
        num_points = 500
        num_time_steps = 10

        # Create positions in 4D (time, x, y, z)
        positions_4d = []
        colors_4d = []

        for t in range(num_time_steps):
            # Spatial positions (vary slightly with time)
            pos_3d = np.random.randn(num_points, 3).astype(np.float32) * 10
            pos_3d += t * 0.1  # Slight drift over time

            # Add time dimension
            time_col = np.full((num_points, 1), t, dtype=np.float32)
            pos_4d = np.column_stack([time_col, pos_3d])
            positions_4d.append(pos_4d)

            # Colors that change with time
            hue = t / num_time_steps
            color = np.array([hue, 1-hue, 0.5], dtype=np.float32)
            colors_4d.append(np.tile(color, (num_points, 1)))

        positions = np.vstack(positions_4d)
        colors = np.vstack(colors_4d)

        dims = Dimensions([
            Dimension("time", unit="frame", range=(0, num_time_steps-1),
                     step=1, display=False, discrete=True),
            Dimension("x", unit="units", display=True),
            Dimension("y", unit="units", display=True),
            Dimension("z", unit="units", display=True),
        ])

        with LuxarZarrCompiler(output, encoding_mode=EncodingMode.MEMORY, compressor=None) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points("points", positions, colors=colors)

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape} (4D)")
        aprint(f"  Time steps: {num_time_steps}")
        aprint(f"  Points per step: {num_points}")


def main():
    """Generate all test datasets."""
    aprint("=" * 70)
    aprint("GENERATING TYPESCRIPT-PYTHON COMPATIBILITY TEST DATASETS")
    aprint("=" * 70)
    aprint("")

    try:
        generate_broadcasting_test()
        aprint("")

        generate_lut_test()
        aprint("")

        generate_quantization_test()
        aprint("")

        generate_array_refs_test()
        aprint("")

        generate_mixed_encoding_test()
        aprint("")

        generate_4d_test()
        aprint("")

        aprint("=" * 70)
        aprint("✓ ALL TEST DATASETS GENERATED")
        aprint("=" * 70)
        aprint("")
        aprint("Generated datasets:")
        aprint(f"  {FIXTURES_DIR}/test_broadcasting.zarr")
        aprint(f"  {FIXTURES_DIR}/test_lut.zarr")
        aprint(f"  {FIXTURES_DIR}/test_quantization.zarr")
        aprint(f"  {FIXTURES_DIR}/test_array_refs.zarr")
        aprint(f"  {FIXTURES_DIR}/test_mixed.zarr")
        aprint(f"  {FIXTURES_DIR}/test_4d.zarr")
        aprint("")
        aprint("Run TypeScript tests with:")
        aprint("  cd packages/luxar-viewer && pnpm test array-decoder")

    except Exception as e:
        aprint(f"❌ Error generating test data: {e}")
        raise


if __name__ == "__main__":
    main()
