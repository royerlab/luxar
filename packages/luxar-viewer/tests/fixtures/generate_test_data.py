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

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Disable compression for Node.js compatibility (blosc has WASM issues)
        # Disable float16 for TypeScript/Zarrita compatibility
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
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
        unique_colors = np.array(
            [
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
            ],
            dtype=np.float32,
        )

        # Randomly assign colors (indices 0-9)
        color_indices = np.random.randint(0, 10, size=1000)
        colors = unique_colors[color_indices]

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Add uniform radii so points are visible
        radii = np.ones(1000, dtype=np.float32) * 0.5

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "points",
                positions,
                colors=colors,  # Will use LUT (only 10 unique values)
                radii=radii,
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Colors: {colors.shape} ({len(unique_colors)} unique - LUT)")
        aprint(f"  Radii: {radii.shape}")


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

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "points",
                positions,
                colors=colors,  # Will quantize to uint8
                radii=radii,  # Will quantize to uint8
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Colors: {colors.shape} (quantized)")
        aprint(f"  Radii: {radii.shape} (quantized)")


def generate_array_refs_test():
    """Test dataset with array references (deduplication)."""
    with asection("Generating Array References Test"):
        output = FIXTURES_DIR / "test_array_refs.zarr"

        # CRITICAL: Use SAME positions for both groups!
        # Morton ordering must produce SAME sort order for deduplication to work
        shared_positions = np.random.randn(500, 3).astype(np.float32) * 10

        # Same colors for both groups
        shared_colors = np.random.rand(500, 3).astype(np.float32)

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Disable spatial index to prevent Morton ordering from creating new arrays
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
            enable_spatial_index=False,  # CRITICAL: Disable to preserve array identity
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add both point clouds with SAME colors
            # With spatial index disabled, colors passed directly to encoder
            # Second encoding will detect duplicate and create array ref
            scene.add_points("points1", shared_positions, colors=shared_colors)
            scene.add_points("points2", shared_positions, colors=shared_colors)

        aprint(f"✓ Created {output}")
        aprint(f"  Shared positions: {shared_positions.shape}")
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
        lut_colors = np.tile(
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32), (250, 1)
        )

        # Group 3: Direct (no encoding)
        pos3 = np.random.randn(500, 3).astype(np.float32) * 10
        direct_colors = np.random.rand(500, 3).astype(np.float32)

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
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
        aprint("  Points 'uniform': Broadcasting")
        aprint("  Points 'lut': LUT encoding")
        aprint("  Points 'direct': No encoding")


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
            color = np.array([hue, 1 - hue, 0.5], dtype=np.float32)
            colors_4d.append(np.tile(color, (num_points, 1)))

        positions = np.vstack(positions_4d)
        colors = np.vstack(colors_4d)

        # Add radii so points are visible
        radii = np.ones(num_points * num_time_steps, dtype=np.float32) * 0.5

        dims = Dimensions(
            [
                Dimension(
                    "time",
                    unit="frame",
                    range=(0, num_time_steps - 1),
                    step=1,
                    display=False,
                    discrete=True,
                ),
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points("points", positions, colors=colors, radii=radii)

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape} (4D)")
        aprint(f"  Time steps: {num_time_steps}")
        aprint(f"  Points per step: {num_points}")


def generate_hierarchical_transforms_test():
    """Test dataset with hierarchical scene graph and nested transforms.

    CRITICAL: This test verifies transform composition and hierarchy:
    - Parent transforms affect children
    - Transforms are stored in correct format (column-major for THREE.js)
    - Matrix multiplication order is correct
    """
    with asection("Generating Hierarchical Transforms Test"):
        output = FIXTURES_DIR / "test_hierarchical_transforms.zarr"

        # Create a simple hierarchy:
        # Scene
        #   └─ parent_group (translated by [10, 0, 0])
        #       └─ child_points (translated by [0, 5, 0])
        # Final position should be [10, 5, 0] due to transform composition

        # Import transform functions
        from luxar.transforms import translate

        # Child points at origin initially
        positions = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )

        # Add radii so points are visible
        radii = np.ones(3, dtype=np.float32) * 0.5

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Create parent group with translation [10, 0, 0]
            parent_transform = translate(10, 0, 0)
            parent_group = scene.add_group("parent_group", transform=parent_transform)

            # Create child points with translation [0, 5, 0] relative to parent
            child_transform = translate(0, 5, 0)
            scene.add_points(
                "child_points",
                positions,
                parent=parent_group,
                transform=child_transform,
                radii=radii,
            )

        aprint(f"✓ Created {output}")
        aprint("  Hierarchy: Scene → parent_group [10,0,0] → child_points [0,5,0]")
        aprint("  Expected final position: [10, 5, 0]")
        aprint("  CRITICAL: Verifies transform composition and matrix format")


def generate_hdr_colors_test():
    """Test dataset with HDR colors (values > 1.0) to verify float32 color handling.

    CRITICAL: This test verifies that HDR colors are preserved through the pipeline:
    - Python stores colors as float32 with values > 1.0
    - TypeScript loads and preserves float32 colors
    - Rendering pipeline handles HDR values correctly
    """
    with asection("Generating HDR Colors Test"):
        output = FIXTURES_DIR / "test_hdr_colors.zarr"

        # Create 20 points with HDR colors ranging from [0, 10]
        num_points = 20
        positions = np.zeros((num_points, 3), dtype=np.float32)

        # Arrange points in a line along X axis
        positions[:, 0] = np.arange(num_points, dtype=np.float32)

        # HDR colors: Red channel from 0 to 10 (HDR range)
        colors = np.zeros((num_points, 3), dtype=np.float32)
        colors[:, 0] = np.linspace(0, 10, num_points)  # Red: 0 to 10 (HDR)
        colors[:, 1] = 0.5  # Green: constant
        colors[:, 2] = 0.5  # Blue: constant

        # Add radii so points are visible
        radii = np.ones(num_points, dtype=np.float32) * 0.5

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Use PRECISION mode to force float32 storage (no quantization)
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "hdr_points",
                positions,
                colors=colors,  # HDR colors stored as float32
                radii=radii,
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Colors: {colors.shape} (HDR, max={colors.max():.1f})")
        aprint("  CRITICAL: Verifies float32 HDR color preservation")


def generate_log_scalar_test():
    """Test dataset with log-space encoded radii (wide dynamic range).

    CRITICAL: This test verifies log_scalar_uint8 encoding which is used for
    positive scalars with wide dynamic range (e.g., radii from 0.001 to 100.0).

    Encoding: log1p(value)/max_log → uint8
    Decoding: expm1(normalized * max_log)
    """
    with asection("Generating Log-Scalar Encoding Test"):
        output = FIXTURES_DIR / "test_log_scalar.zarr"

        # Create 100 points with radii spanning wide dynamic range
        num_points = 100
        positions = np.zeros((num_points, 3), dtype=np.float32)

        # Arrange points in a line along X axis
        positions[:, 0] = np.arange(num_points, dtype=np.float32)

        # Radii with wide dynamic range: 0.01 to 100.0 (log scale)
        # This range benefits from log-space encoding
        radii = np.logspace(-2, 2, num_points, dtype=np.float32)  # 0.01 to 100

        # Simple colors
        colors = np.zeros((num_points, 3), dtype=np.float32)
        colors[:, 0] = np.linspace(0, 1, num_points)  # Red gradient

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Use MEMORY mode which uses log_scalar for wide-range scalars
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "log_radii_test",
                positions,
                colors=colors,
                radii=radii,  # Wide range - will use log_scalar encoding
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Radii range: [{radii.min():.4f}, {radii.max():.4f}] (log-space)")
        aprint("  CRITICAL: Verifies log_scalar_uint8 encoding/decoding")


def generate_4d_scalar_lut_test():
    """Test dataset with 4D positions using scalar LUT encoding.

    CRITICAL: This test verifies scalar LUT mode on multi-dimensional positions,
    which is the exact scenario that caused bugs in quantum orbitals (4D data).

    Key aspects tested:
    - 4D positions stored as scalar LUT (flattened indices)
    - original_shape metadata preserves [n_points, 4]
    - Partial range extraction works correctly
    - actualElementsPerPoint is correctly calculated from original_shape

    The quantum orbitals bug: When scalar LUT is used on 4D positions,
    the decoder returns the full array but range extraction needs to use
    actualElementsPerPoint = 4 (not 1 from indices shape).
    """
    with asection("Generating 4D Scalar LUT Encoding Test"):
        output = FIXTURES_DIR / "test_4d_scalar_lut.zarr"

        # Create 200 points in 4D with LIMITED unique coordinate values
        # This triggers scalar LUT encoding (≤256 unique values per coordinate)
        num_points = 200

        # Use only 10 unique values per dimension to guarantee LUT encoding
        unique_values = np.array(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32
        )

        # Generate positions by randomly selecting from unique values
        np.random.seed(42)  # Reproducible for testing
        positions = np.zeros((num_points, 4), dtype=np.float32)
        for i in range(4):
            positions[:, i] = np.random.choice(unique_values, size=num_points)

        # Make positions identifiable: point i at position i has predictable values
        # First 10 points have sequential patterns for easy verification
        for i in range(min(10, num_points)):
            positions[i, 0] = float(i)  # First dim = point index
            positions[i, 1] = float(i * 2)  # Second dim = 2x index
            positions[i, 2] = float(i * 0.5)  # Third dim = 0.5x index
            positions[i, 3] = float(i % 5)  # Fourth dim = modulo pattern

        # Add radii and colors
        radii = np.ones(num_points, dtype=np.float32) * 0.5
        colors = np.random.rand(num_points, 3).astype(np.float32)

        dims = Dimensions(
            [
                Dimension("time", unit="frame", range=(0, 9), step=1, display=False),
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Use MEMORY mode which enables LUT encoding for arrays with ≤256 unique values
        with (
            LuxarZarrCompiler(
                output,
                encoding_mode=EncodingMode.MEMORY,
                compressor=None,
                float16_allowed=False,
                enable_spatial_index=False,  # Disable to preserve exact positions for testing
            ) as compiler
        ):
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points("points", positions, colors=colors, radii=radii)

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape} (4D)")
        aprint(f"  Unique values per dim: {len(unique_values)}")
        aprint("  First 10 points have predictable patterns for verification")
        aprint("  CRITICAL: Tests scalar LUT + 4D + partial range extraction")


def generate_uint16_quantization_test():
    """Test dataset with uint16 quantization (wide dynamic range).

    CRITICAL: This test verifies that uint16 bounded_scalar encoding is correctly
    handled by the TypeScript decoder. The bug was that attrs.dtype defaults to
    'uint8' when undefined, causing 256x amplitude error for uint16 data!

    Scenario:
    - Python encoder detects dynamic range > 256:1 → uses uint16
    - zarr array.dtype = uint16
    - attrs.encoding.original_dtype = float32 (input dtype, NOT storage dtype)
    - attrs.dtype = undefined (Python encoder doesn't write this!)

    The fix: Use zarr array.dtype for max_int calculation (65535 for uint16),
    NOT attrs.dtype which defaults to uint8.
    """
    with asection("Generating uint16 Quantization Test"):
        output = FIXTURES_DIR / "test_uint16_quantization.zarr"

        # Create data with WIDE dynamic range (> 256:1) to trigger uint16 encoding
        # This mimics gsplat amplitudes: small values with ~6000:1 dynamic range
        num_points = 1000
        positions = np.random.randn(num_points, 3).astype(np.float32) * 10

        # Radii with wide dynamic range: 0.001 to 1.0 (1000:1 ratio)
        # This MUST trigger uint16 encoding (> 256:1 range)
        # Using linspace ensures exact 1000:1 ratio for reliable uint16 triggering
        radii = np.linspace(0.001, 1.0, num_points).astype(np.float32)
        dynamic_range = radii.max() / radii.min()
        assert dynamic_range > 256, (
            f"Need >256:1 range for uint16, got {dynamic_range:.1f}:1"
        )

        # Simple colors (use uint8 encoding as comparison)
        colors = np.random.rand(num_points, 3).astype(np.float32)

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Use MEMORY mode which uses dynamic range-based dtype selection
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "uint16_test",
                positions,
                colors=colors,
                radii=radii,  # Will be encoded as uint16 (range > 256:1)
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Radii range: [{radii.min():.4f}, {radii.max():.4f}]")
        aprint(f"  Dynamic range: {dynamic_range:.1f}:1 (triggers uint16)")
        aprint("  CRITICAL: Verifies uint16 bounded_scalar decoding uses max_int=65535")


def generate_sharpness_range_test():
    """Test dataset with full sharpness range [0, 31] to verify decoding.

    CRITICAL: This test verifies the bug fix where TypeScript was using
    scale factor 15.0 instead of 31.0 (matching Python's SHARPNESS_MAX).
    """
    with asection("Generating Sharpness Range Test"):
        output = FIXTURES_DIR / "test_sharpness_range.zarr"

        # Create 31 points, each with a different sharpness value from 1 to 31
        # Note: Skipping 0.0 because validation requires strictly positive values
        num_points = 31
        positions = np.zeros((num_points, 3), dtype=np.float32)

        # Arrange points in a line along X axis for easy visualization
        positions[:, 0] = np.arange(num_points, dtype=np.float32)

        # Sharpness values: [1.0, 2.0, 3.0, ..., 31.0]
        sharpness = np.arange(1, num_points + 1, dtype=np.float32)

        # Assign colors based on sharpness (gradient from blue to red)
        colors = np.zeros((num_points, 3), dtype=np.float32)
        colors[:, 0] = sharpness / 31.0  # Red increases with sharpness
        colors[:, 2] = 1.0 - (sharpness / 31.0)  # Blue decreases with sharpness

        # Add radii so points are visible
        radii = np.ones(num_points, dtype=np.float32) * 0.5

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "sharpness_test",
                positions,
                colors=colors,
                sharpness=sharpness,  # Full range [0, 31]
                radii=radii,
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Sharpness range: [{sharpness.min()}, {sharpness.max()}]")
        aprint("  CRITICAL: Verifies TypeScript uses scale factor 31.0 (not 15.0)")


def generate_nd_transforms_test():
    """Test dataset with nd_transforms for verifying inverse-query in viewer.

    Creates a 4D scene (X, Y, Z, Time) with two groups:
    - GroupA: no nd_transform (baseline). 50 points at time=0 only.
    - GroupB: nd_transform={"Time": {"offset": 5}}. 50 points at local time=0,
      which should appear at world time=5 in the viewer.

    At world time=0: only GroupA points visible.
    At world time=5: only GroupB points visible (due to offset).
    """
    with asection("Generating nD Transforms Test"):
        output = FIXTURES_DIR / "test_nd_transforms.zarr"

        dims = Dimensions(
            [
                Dimension("X", unit="u", range=(-10, 10), display=True),
                Dimension("Y", unit="u", range=(-10, 10), display=True),
                Dimension("Z", unit="u", range=(-10, 10), display=True),
                Dimension(
                    "Time",
                    unit="frame",
                    range=(0, 10),
                    display=False,
                    discrete=True,
                    step=1.0,
                ),
            ]
        )

        rng = np.random.default_rng(42)

        # GroupA: 50 points at time=0, no nd_transform
        pos_a = np.zeros((50, 4), dtype=np.float32)
        pos_a[:, :3] = rng.uniform(-5, 5, (50, 3)).astype(np.float32)
        pos_a[:, 3] = 0  # time = 0
        col_a = np.tile([1.0, 0.0, 0.0], (50, 1)).astype(np.float32)  # Red

        # GroupB: 50 points at local time=0, with nd_transform offset=5
        # So these should appear at world time=5
        pos_b = np.zeros((50, 4), dtype=np.float32)
        pos_b[:, :3] = rng.uniform(-5, 5, (50, 3)).astype(np.float32)
        pos_b[:, 3] = 0  # local time = 0 (world time = 5 after offset)
        col_b = np.tile([0.0, 0.0, 1.0], (50, 1)).astype(np.float32)  # Blue

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            group_a = scene.add_group("GroupA")
            group_a.add_points(
                "points_a",
                pos_a,
                colors=col_a,
                radii=0.5,
                extend_to_all=[],
            )

            group_b = scene.add_group(
                "GroupB",
                nd_transform={"Time": {"offset": 5.0}},
            )
            group_b.add_points(
                "points_b",
                pos_b,
                colors=col_b,
                radii=0.5,
                extend_to_all=[],
            )

        aprint(f"  Generated: {output}")
        aprint("  GroupA: 50 red points at time=0 (no nd_transform)")
        aprint("  GroupB: 50 blue points at local time=0 (nd_transform offset=5)")
        aprint("  Expected: time=0 → 50 red, time=5 → 50 blue")


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

        generate_hierarchical_transforms_test()
        aprint("")

        generate_hdr_colors_test()
        aprint("")

        generate_sharpness_range_test()
        aprint("")

        generate_log_scalar_test()
        aprint("")

        generate_4d_scalar_lut_test()
        aprint("")

        generate_uint16_quantization_test()
        aprint("")

        generate_nd_transforms_test()
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
        aprint(f"  {FIXTURES_DIR}/test_hierarchical_transforms.zarr")
        aprint(f"  {FIXTURES_DIR}/test_hdr_colors.zarr")
        aprint(f"  {FIXTURES_DIR}/test_sharpness_range.zarr")
        aprint(f"  {FIXTURES_DIR}/test_log_scalar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_4d_scalar_lut.zarr")
        aprint(f"  {FIXTURES_DIR}/test_uint16_quantization.zarr")
        aprint(f"  {FIXTURES_DIR}/test_nd_transforms.zarr")
        aprint("")
        aprint("Run TypeScript tests with:")
        aprint("  cd packages/luxar-viewer && pnpm test array-decoder")

    except Exception as e:
        aprint(f"❌ Error generating test data: {e}")
        raise


if __name__ == "__main__":
    main()
