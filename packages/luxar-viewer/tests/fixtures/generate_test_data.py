#!/usr/bin/env python3
"""Generate test datasets for TypeScript-Python encoding compatibility tests.

This script creates small zarr datasets with all encoding modes to verify
that the TypeScript ArrayDecoder can correctly read Python-encoded data.

IMPORTANT: Uses NO compression (compressor=None) to avoid blosc/numcodecs
WASM binding issues in Node.js test environment.

Run from project root:
    hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py
"""

import shutil
from pathlib import Path

import numpy as np
import zarr
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import ArrayEncoder, EncodingMode, SemanticType

# Output directory
FIXTURES_DIR = Path(__file__).parent
FIXTURES_DIR.mkdir(exist_ok=True)

# Audit C2 (viewer-integration-fixtures) guardrails — see
# delme/test-audit-luxar-codebase/findings-viewer-integration-fixtures.md.
#
# These two flags MUST stay at the values below for ALL fixtures created
# by this script. They are passed to every LuxarZarrCompiler invocation
# in this file:
#
#   compressor=COMPRESSOR_DISABLED  (None) — blosc/numcodecs has known WASM
#       binding issues under Node.js (jsdom + Vitest). Compressed fixtures
#       would silently fail to load and the entire unit-test suite would
#       skip every fixture-backed test.
#   float16_allowed=FLOAT16_ALLOWED (False) — Node.js zarrita does not
#       support float16 reads (no native Float16Array yet). float16 in a
#       fixture would surface as decode failures across the unit suite.
#
# If a future change needs to flip these defaults, FIRST verify the
# Node.js side can round-trip the new format (see
# packages/luxar-viewer/src/tests/unit/data/array-decoder/decoder.test.ts).
COMPRESSOR_DISABLED = None  # blosc is incompatible with Node.js test env
FLOAT16_ALLOWED = False  # zarrita-js cannot read float16 in Node.js


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


def generate_array_ref_broadcasting_test():
    """Test array_ref positions combined with scalar/broadcast attributes.

    This catches a subtle encoder/metadata bug: when a duplicate positions array is
    stored as an array_ref, the physical zarr array shape is ``(0, D)``. Scalar
    attributes on the same node must still be broadcast to the logical point count,
    not to the physical array_ref shape.
    """
    with asection("Generating Array Ref + Broadcasting Test"):
        output = FIXTURES_DIR / "test_array_ref_broadcasting.zarr"

        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        first_colors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        first_radii = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)

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
            enable_spatial_index=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "source_points",
                positions,
                colors=first_colors,
                radii=first_radii,
            )
            scene.add_points(
                "ref_points_with_colors",
                positions,
                colors=(0.25, 0.5, 0.75),
                radii=0.5,
                sharpness=2.0,
            )
            scene.add_points(
                "ref_points_with_scalars",
                positions,
                radii=0.5,
                scalars=1.25,
                colormap="viridis",
            )

        aprint(f"✓ Created {output}")
        aprint("  ref_points_with_colors/positions: array_ref to source_points")
        aprint("  ref_points_with_scalars/positions: array_ref to source_points")
        aprint("  Scalar color/radius/sharpness/scalars broadcast to 4 logical points")


def generate_encoding_edge_cases_test():
    """Raw ArrayEncoder fixture covering edge cases outside scene validation."""
    with asection("Generating Raw Encoding Edge Cases Test"):
        output = FIXTURES_DIR / "test_encoding_edge_cases.zarr"
        if output.exists():
            shutil.rmtree(output)

        root = zarr.open_group(str(output), mode="w")
        encoder = ArrayEncoder(float16_allowed=False)

        # Empty passthrough array. Scene validation disallows empty geometries,
        # but the encoder/decoder contract should still handle empty arrays.
        encoder.encode(
            np.empty((0, 3), dtype=np.float32),
            root,
            "empty_colors",
            SemanticType.COLOR,
            mode=EncodingMode.PRECISION,
            color_mode="sdr",
            compressor=None,
        )

        # Scalar and singleton broadcasting through metadata n_elements.
        encoder.encode(
            (0.25, 0.5, 0.75),
            root,
            "singleton_color",
            SemanticType.COLOR,
            mode=EncodingMode.MEMORY,
            n_elements=1,
            color_mode="sdr",
            compressor=None,
        )
        encoder.encode(
            1.5,
            root,
            "singleton_radius",
            SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.MEMORY,
            n_elements=1,
            compressor=None,
        )

        # Full uniform arrays should encode as broadcasted with inferred n_elements.
        encoder.encode(
            np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (4, 1)),
            root,
            "uniform_color_array",
            SemanticType.COLOR,
            mode=EncodingMode.MEMORY,
            color_mode="sdr",
            compressor=None,
        )

        # Explicit scalar LUT mode for 2D non-color data.
        scalar_lut_values = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        encoder.encode(
            scalar_lut_values,
            root,
            "scalar_lut_matrix",
            SemanticType.BOUNDED_SCALAR,
            mode=EncodingMode.MEMORY,
            compressor=None,
        )

        # Explicit log-space positive scalar encoding. The scene compiler uses
        # linear positive-scalar encoding by default, so keep a raw fixture for
        # decoder compatibility with this valid encoder mode.
        encoder.encode(
            np.logspace(-2, 2, 32, dtype=np.float32),
            root,
            "log_scalar_radii",
            SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.MEMORY,
            positive_scalar_encoding="log",
            compressor=None,
        )

        # Known 4D packed lower-triangular Cholesky values.
        cholesky = np.array(
            [
                [1.0, 0.1, 1.1, 0.2, 0.3, 1.2, 0.4, 0.5, 0.6, 1.3],
                [2.0, 0.0, 2.1, 0.0, 0.0, 2.2, 0.0, 0.0, 0.0, 2.3],
            ],
            dtype=np.float32,
        )
        encoder.encode(
            cholesky,
            root,
            "cholesky_4d_packed",
            SemanticType.CHOLESKY,
            mode=EncodingMode.PRECISION,
            compressor=None,
        )

        zarr.consolidate_metadata(str(output))
        aprint(f"✓ Created {output}")
        aprint(
            "  Covers empty arrays, singleton broadcasts, scalar LUT, log scalar, 4D Cholesky"
        )


def _tag_contract_case(
    array: zarr.Array,
    case_id: str,
    semantic_type: SemanticType | str,
    description: str,
) -> None:
    """Attach machine-readable coverage metadata to raw contract arrays."""
    array.attrs["contract_case"] = {
        "case_id": case_id,
        "semantic_type": str(getattr(semantic_type, "value", semantic_type)),
        "description": description,
    }


def generate_encoding_contract_matrix_test():
    """Generate a declarative raw ArrayEncoder contract matrix fixture.

    Scene fixtures are realistic, but they do not exhaustively pin the encoder
    surface. This fixture intentionally creates small arrays for every semantic
    type and every decoder-supported encoding family, including dtype and LUT
    thresholds that are awkward to trigger from normal scene validation.
    """
    with asection("Generating Encoding Contract Matrix Test"):
        output = FIXTURES_DIR / "test_encoding_contract_matrix.zarr"
        if output.exists():
            shutil.rmtree(output)

        root = zarr.open_group(str(output), mode="w")

        def encode_case(
            case_id: str,
            data: np.ndarray | float | int | tuple[float, ...],
            semantic_type: SemanticType,
            description: str,
            *,
            mode: EncodingMode = EncodingMode.MEMORY,
            n_elements: int | None = None,
            bounds: tuple[float, float] | None = None,
            positive_scalar_encoding: str = "linear",
            custom_encoder: str | None = None,
            color_mode: str | None = None,
            chunks: tuple[int, ...] | None = None,
        ) -> None:
            # Use a fresh encoder per case so array_ref coverage remains explicit
            # in scene fixtures and does not accidentally appear in this matrix.
            encoder = ArrayEncoder(float16_allowed=False)
            encoder.encode(
                data,
                root,
                case_id,
                semantic_type,
                mode=mode,
                n_elements=n_elements,
                bounds=bounds,
                positive_scalar_encoding=positive_scalar_encoding,  # type: ignore[arg-type]
                custom_encoder=custom_encoder,
                color_mode=color_mode,  # type: ignore[arg-type]
                chunks=chunks,
                compressor=None,
            )
            _tag_contract_case(root[case_id], case_id, semantic_type, description)

        # COORDINATE: direct float32, empty passthrough, and 4D scalar LUT.
        encode_case(
            "coordinate_empty_0x3",
            np.empty((0, 3), dtype=np.float32),
            SemanticType.COORDINATE,
            "empty coordinate passthrough",
            mode=EncodingMode.PRECISION,
        )
        encode_case(
            "coordinate_float32_chunked_2d",
            np.array(
                [[-1.0, 0.0], [1.0, 2.0], [3.5, -4.0], [5.0, 6.0], [7.0, -8.0]],
                dtype=np.float32,
            ),
            SemanticType.COORDINATE,
            "2D coordinate float32 with row chunks",
            mode=EncodingMode.PRECISION,
            chunks=(2, 2),
        )
        unique = np.arange(16, dtype=np.float32)
        coordinate_4d = np.column_stack(
            [unique % 4, (unique * 2) % 7, (unique * 3) % 11, (unique * 5) % 13]
        ).astype(np.float32)
        coordinate_4d = np.tile(coordinate_4d, (4, 1))
        encode_case(
            "coordinate_lut_scalar_4d",
            coordinate_4d,
            SemanticType.COORDINATE,
            "4D coordinate scalar LUT with original_shape metadata",
            mode=EncodingMode.MEMORY,
            chunks=(5, 4),
        )

        # COLOR: RGB quantization, direct integer colors, row LUT, and RGB uint16.
        color_gradient = np.linspace(0.0, 1.0, 17 * 3, dtype=np.float32).reshape(17, 3)
        encode_case(
            "color_rgb_uint8_sdr",
            color_gradient,
            SemanticType.COLOR,
            "SDR float color quantized to rgb_uint8",
            mode=EncodingMode.MEMORY,
            color_mode="sdr",
            chunks=(4, 3),
        )
        encode_case(
            "color_rgb_uint16_custom",
            color_gradient,
            SemanticType.COLOR,
            "SDR float color explicitly quantized to rgb_uint16",
            mode=EncodingMode.CUSTOM,
            custom_encoder="rgb_uint16",
            color_mode="sdr",
            chunks=(4, 3),
        )
        encode_case(
            "color_direct_uint8",
            np.array([[0, 127, 255], [255, 64, 0], [1, 2, 3]], dtype=np.uint8),
            SemanticType.COLOR,
            "direct uint8 color storage",
            mode=EncodingMode.PRECISION,
            color_mode="sdr",
        )
        encode_case(
            "color_direct_uint16",
            np.array(
                [[0, 32768, 65535], [65535, 4096, 0], [100, 200, 300]], dtype=np.uint16
            ),
            SemanticType.COLOR,
            "direct uint16 color storage",
            mode=EncodingMode.PRECISION,
            color_mode="sdr",
        )
        lut_rows = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        encode_case(
            "color_lut_row_chunked",
            np.tile(lut_rows, (16, 1)),
            SemanticType.COLOR,
            "row-mode color LUT with chunked indices",
            mode=EncodingMode.MEMORY,
            color_mode="sdr",
            chunks=(7, 3),
        )
        encode_case(
            "color_broadcast_rgba_singleton",
            (0.25, 0.5, 0.75, 1.0),
            SemanticType.COLOR,
            "RGBA singleton broadcast",
            mode=EncodingMode.MEMORY,
            n_elements=1,
            color_mode="sdr",
        )

        # BOUNDED_SCALAR: precision, uint8, and uint16 quantization.
        encode_case(
            "bounded_scalar_float32_precision",
            np.array([-1.0, -0.25, 0.0, 0.25, 1.0], dtype=np.float32),
            SemanticType.BOUNDED_SCALAR,
            "bounded scalar precision/direct float32",
            mode=EncodingMode.PRECISION,
            chunks=(2,),
        )
        encode_case(
            "bounded_scalar_uint8_custom",
            np.linspace(-2.0, 2.0, 33, dtype=np.float32),
            SemanticType.BOUNDED_SCALAR,
            "explicit bounded_scalar_uint8 over negative-to-positive range",
            mode=EncodingMode.CUSTOM,
            custom_encoder="bounded_scalar_uint8",
            bounds=(-2.0, 2.0),
            chunks=(5,),
        )
        encode_case(
            "bounded_scalar_uint16_custom",
            np.linspace(0.001, 10.0, 257, dtype=np.float32),
            SemanticType.BOUNDED_SCALAR,
            "explicit bounded_scalar_uint16 crossing uint8 precision needs",
            mode=EncodingMode.CUSTOM,
            custom_encoder="bounded_scalar_uint16",
            bounds=(0.0, 10.0),
            chunks=(17,),
        )

        # POSITIVE_SCALAR: linear uint16 and log uint8/uint16.
        encode_case(
            "positive_scalar_linear_uint16",
            np.linspace(0.001, 1.0, 257, dtype=np.float32),
            SemanticType.POSITIVE_SCALAR,
            "positive scalar linear uint16 chosen by dynamic range",
            mode=EncodingMode.MEMORY,
            chunks=(17,),
        )
        encode_case(
            "positive_scalar_log_uint8",
            np.logspace(-6, 3, 64, dtype=np.float32),
            SemanticType.POSITIVE_SCALAR,
            "positive scalar log uint8",
            mode=EncodingMode.MEMORY,
            positive_scalar_encoding="log",
            chunks=(9,),
        )
        encode_case(
            "positive_scalar_log_uint16_custom",
            np.logspace(-9, 6, 257, dtype=np.float32),
            SemanticType.POSITIVE_SCALAR,
            "positive scalar explicit log uint16",
            mode=EncodingMode.CUSTOM,
            custom_encoder="log_scalar_uint16",
            chunks=(17,),
        )

        # CHOLESKY packed lower-triangular shapes for D=1,2,3,4.
        encode_case(
            "cholesky_d1_packed",
            np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            SemanticType.CHOLESKY,
            "D=1 packed Cholesky",
            mode=EncodingMode.PRECISION,
        )
        encode_case(
            "cholesky_d2_packed",
            np.array([[1.0, 0.1, 1.1], [2.0, 0.2, 2.2]], dtype=np.float32),
            SemanticType.CHOLESKY,
            "D=2 packed Cholesky",
            mode=EncodingMode.PRECISION,
        )
        encode_case(
            "cholesky_d4_packed_chunked",
            np.array(
                [
                    [1.0, 0.1, 1.1, 0.2, 0.3, 1.2, 0.4, 0.5, 0.6, 1.3],
                    [2.0, 0.0, 2.1, 0.0, 0.0, 2.2, 0.0, 0.0, 0.0, 2.3],
                    [3.0, 0.2, 3.1, 0.3, 0.4, 3.2, 0.5, 0.6, 0.7, 3.3],
                ],
                dtype=np.float32,
            ),
            SemanticType.CHOLESKY,
            "D=4 packed Cholesky with row chunks",
            mode=EncodingMode.PRECISION,
            chunks=(2, 10),
        )

        # INDEX: direct integer dtype selection, including uint64.
        encode_case(
            "index_uint8",
            np.array([0, 1, 2, 254, 255], dtype=np.uint16),
            SemanticType.INDEX,
            "index values fitting uint8",
            mode=EncodingMode.MEMORY,
            chunks=(2,),
        )
        encode_case(
            "index_uint16",
            np.array([0, 255, 256, 4096, 65535], dtype=np.uint32),
            SemanticType.INDEX,
            "index values fitting uint16",
            mode=EncodingMode.MEMORY,
            chunks=(2,),
        )
        encode_case(
            "index_uint32",
            np.array([0, 65536, 2**24 + 1, 2**32 - 1], dtype=np.uint64),
            SemanticType.INDEX,
            "index values fitting uint32",
            mode=EncodingMode.MEMORY,
            chunks=(2,),
        )
        encode_case(
            "index_uint64",
            np.array([0, 2**32, 2**40 + 12345], dtype=np.uint64),
            SemanticType.INDEX,
            "index values requiring uint64 storage",
            mode=EncodingMode.MEMORY,
        )

        # UNIT_VECTOR: direct float path with signed normalized rows.
        unit_vectors = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.57735026] * 3],
            dtype=np.float32,
        )
        encode_case(
            "unit_vector_float32",
            unit_vectors,
            SemanticType.UNIT_VECTOR,
            "signed 3D unit vectors direct float32",
            mode=EncodingMode.PRECISION,
            chunks=(2, 3),
        )

        # Manual LUT threshold cases, including lut_uint16 which normal scenes do
        # not currently emit automatically (the decoder contract still supports it).
        def manual_lut_case(
            case_id: str,
            unique_count: int,
            *,
            row_mode: bool = False,
            dtype: np.dtype = np.dtype("uint8"),
        ) -> None:
            name = "lut_uint16" if dtype == np.dtype("uint16") else "lut_uint8"
            if row_mode:
                lut = np.column_stack(
                    [
                        np.linspace(0.0, 1.0, unique_count, dtype=np.float32),
                        np.linspace(1.0, 0.0, unique_count, dtype=np.float32),
                        np.full(unique_count, 0.5, dtype=np.float32),
                    ]
                )
                indices = np.arange(unique_count * 2, dtype=np.uint32) % unique_count
                indices = indices.astype(dtype)
                root.create_dataset(
                    case_id, data=indices, chunks=(17,), compressor=None
                )
                root[case_id].attrs["encoding"] = {
                    "name": name,
                    "lut": lut.tolist(),
                    "lut_mode": "row",
                    "original_dtype": "float32",
                    "original_shape": [int(indices.shape[0]), 3],
                }
                description = f"manual row LUT with {unique_count} unique rows"
                semantic = SemanticType.COLOR
            else:
                lut = np.linspace(-1.0, 1.0, unique_count, dtype=np.float32)
                indices = np.arange(unique_count * 2, dtype=np.uint32) % unique_count
                indices = indices.astype(dtype)
                root.create_dataset(
                    case_id, data=indices, chunks=(17,), compressor=None
                )
                root[case_id].attrs["encoding"] = {
                    "name": name,
                    "lut": lut.tolist(),
                    "lut_mode": "scalar",
                    "original_dtype": "float32",
                    "original_shape": [int(indices.shape[0])],
                }
                description = f"manual scalar LUT with {unique_count} unique values"
                semantic = SemanticType.BOUNDED_SCALAR
            _tag_contract_case(root[case_id], case_id, semantic, description)

        manual_lut_case("lut_scalar_1_unique", 1)
        manual_lut_case("lut_scalar_2_unique", 2)
        manual_lut_case("lut_scalar_255_unique", 255)
        manual_lut_case("lut_scalar_256_unique", 256)
        manual_lut_case("lut_scalar_257_unique_uint16", 257, dtype=np.dtype("uint16"))
        manual_lut_case("lut_row_256_unique", 256, row_mode=True)
        manual_lut_case(
            "lut_row_257_unique_uint16", 257, row_mode=True, dtype=np.dtype("uint16")
        )

        zarr.consolidate_metadata(str(output))
        aprint(f"✓ Created {output}")
        aprint(
            "  Covers all semantic types, uint32/uint64, lut_uint16, rgb_uint16, log_uint16"
        )


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


def generate_integer_colors_test():
    """Test dataset with direct uint8 and uint16 SDR color arrays."""
    with asection("Generating Integer Colors Test"):
        output = FIXTURES_DIR / "test_integer_colors.zarr"

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        radii = np.full(3, 0.5, dtype=np.float32)
        colors_u8 = np.array([[255, 0, 0], [0, 128, 255], [64, 32, 16]], dtype=np.uint8)
        colors_u16 = np.array(
            [[65535, 0, 0], [0, 32768, 65535], [16384, 8192, 4096]],
            dtype=np.uint16,
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points("uint8_points", positions, colors=colors_u8, radii=radii)
            scene.add_points("uint16_points", positions, colors=colors_u16, radii=radii)

        aprint(f"✓ Created {output}")
        aprint("  uint8_points/colors: direct uint8")
        aprint("  uint16_points/colors: direct uint16")


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
    """Test scene dataset with wide dynamic range radii.

    The scene compiler currently uses linear positive-scalar encoding by default,
    so this fixture exercises uint16 bounded-scalar radii. The raw
    ``test_encoding_edge_cases.zarr`` fixture below covers explicit log-scalar
    encoder compatibility.
    """
    with asection("Generating Wide-Range Scalar Encoding Test"):
        output = FIXTURES_DIR / "test_log_scalar.zarr"

        # Create 100 points with radii spanning wide dynamic range
        num_points = 100
        positions = np.zeros((num_points, 3), dtype=np.float32)

        # Arrange points in a line along X axis
        positions[:, 0] = np.arange(num_points, dtype=np.float32)

        # Radii with wide dynamic range: 0.01 to 100.0.
        # The scene compiler stores these with bounded_scalar_uint16 by default.
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

        # Use MEMORY mode, which selects uint16 for this dynamic range.
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
        aprint(f"  Radii range: [{radii.min():.4f}, {radii.max():.4f}] (wide range)")
        aprint("  CRITICAL: Verifies bounded_scalar_uint16 encoding/decoding")


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


def generate_lines_test():
    """Test dataset with Lines geometry type.

    Verifies that the TypeScript viewer can load and render Lines,
    including vertices, widths, colors, and segment auto-generation.
    """
    with asection("Generating Lines Test"):
        output = FIXTURES_DIR / "test_lines.zarr"

        # Create a simple zigzag line with 10 vertices
        num_vertices = 10
        vertices = np.zeros((num_vertices, 3), dtype=np.float32)
        vertices[:, 0] = np.arange(num_vertices, dtype=np.float32)  # X: 0,1,2,...
        vertices[:, 1] = np.array(
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32
        )  # Y: zigzag

        # Widths per vertex
        widths = np.linspace(0.1, 0.5, num_vertices).astype(np.float32)

        # Colors per vertex (gradient red to blue)
        colors = np.zeros((num_vertices, 3), dtype=np.float32)
        colors[:, 0] = np.linspace(1, 0, num_vertices).astype(np.float32)
        colors[:, 2] = np.linspace(0, 1, num_vertices).astype(np.float32)

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_lines(
                "zigzag_line",
                vertices,
                widths=widths,
                colors=colors,
            )

        aprint(f"  Created {output}")
        aprint(f"  Vertices: {vertices.shape}, Widths: {widths.shape}")


def generate_gsplats_test():
    """Test dataset with GSplats (Gaussian Splats) geometry type.

    Verifies that the TypeScript viewer can load and render GSplats,
    including centers, amplitudes, cholesky_factors, and colors.
    """
    with asection("Generating GSplats Test"):
        output = FIXTURES_DIR / "test_gsplats.zarr"

        # Create 20 Gaussian splats at grid positions
        num_splats = 20
        centers = np.zeros((num_splats, 3), dtype=np.float32)
        centers[:, 0] = np.arange(num_splats, dtype=np.float32) % 5
        centers[:, 1] = np.arange(num_splats, dtype=np.float32) // 5

        # Amplitudes (brightness)
        amplitudes = np.linspace(0.5, 2.0, num_splats).astype(np.float32)

        # Cholesky factors (lower-triangular covariance factors).
        # For 3D: 6 elements per splat (L11, L21, L22, L31, L32, L33).
        # IMPORTANT: Luxar stores covariance Cholesky factors, not precision
        # factors. Using 1/sigma here makes projected splats enormous; the
        # viewer's screen-coverage safety fade then legitimately culls the
        # entire fixture, so browser smoke tests see a black canvas.
        cholesky = np.zeros((num_splats, 6), dtype=np.float32)
        for i in range(num_splats):
            # Diagonal covariance factor: sigma (isotropic-ish with slight variation)
            sigma = 0.3 + 0.1 * (i / num_splats)
            cholesky[i, 0] = sigma  # L11
            cholesky[i, 2] = sigma  # L22
            cholesky[i, 5] = sigma  # L33

        # Colors (rainbow gradient)
        colors = np.zeros((num_splats, 3), dtype=np.float32)
        colors[:, 0] = np.linspace(1, 0, num_splats).astype(np.float32)
        colors[:, 1] = np.linspace(0, 1, num_splats).astype(np.float32)
        colors[:, 2] = 0.5

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_gsplats(
                "test_splats",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )

        aprint(f"  Created {output}")
        aprint(f"  Centers: {centers.shape}, Cholesky: {cholesky.shape}")


def generate_labelled_points_test():
    """Small labelled-points dataset for the hover-tooltip E2E spec.

    Each point gets a string label "Point 0", "Point 1", ... The scene
    auto-injects a default hover overlay because `has_labels=True`
    (see `luxar/core/scene.py::_inject_default_hover_overlay`). The
    Playwright spec `hover-tooltip.spec.ts` loads this fixture, stops
    the cursor over a point, and verifies the tooltip appears.

    Kept deliberately small (8 points, fixed layout) so the spec can
    predict pixel-coords without any randomness.
    """
    with asection("Generating Labelled Points Test (E2E hover-tooltip)"):
        output = FIXTURES_DIR / "test_labelled_points.zarr"

        # 8 points in a regular pattern across the visible volume.
        # Stay near the origin (the viewer's default camera centres on (0,0,0));
        # spread out enough that mouse can hover one without hitting another.
        positions = np.array(
            [
                [-5.0, -5.0, 0.0],
                [5.0, -5.0, 0.0],
                [-5.0, 5.0, 0.0],
                [5.0, 5.0, 0.0],
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 5.0],
                [-5.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        n = positions.shape[0]

        colors = np.tile(np.array([[1.0, 0.5, 0.25]], dtype=np.float32), (n, 1))
        radii = np.full(n, 1.0, dtype=np.float32)
        labels = [f"Point {i}" for i in range(n)]

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
                "labelled_points",
                positions=positions,
                colors=colors,
                radii=radii,
                labels=labels,
            )

        aprint(f"  Created {output}")
        aprint(f"  {n} labelled points; default hover overlay auto-injected")


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

        generate_array_ref_broadcasting_test()
        aprint("")

        generate_encoding_edge_cases_test()
        aprint("")

        generate_encoding_contract_matrix_test()
        aprint("")

        generate_mixed_encoding_test()
        aprint("")

        generate_4d_test()
        aprint("")

        generate_hierarchical_transforms_test()
        aprint("")

        generate_hdr_colors_test()
        aprint("")

        generate_integer_colors_test()
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

        generate_lines_test()
        aprint("")

        generate_gsplats_test()
        aprint("")

        generate_labelled_points_test()
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
        aprint(f"  {FIXTURES_DIR}/test_array_ref_broadcasting.zarr")
        aprint(f"  {FIXTURES_DIR}/test_encoding_edge_cases.zarr")
        aprint(f"  {FIXTURES_DIR}/test_encoding_contract_matrix.zarr")
        aprint(f"  {FIXTURES_DIR}/test_mixed.zarr")
        aprint(f"  {FIXTURES_DIR}/test_4d.zarr")
        aprint(f"  {FIXTURES_DIR}/test_hierarchical_transforms.zarr")
        aprint(f"  {FIXTURES_DIR}/test_hdr_colors.zarr")
        aprint(f"  {FIXTURES_DIR}/test_integer_colors.zarr")
        aprint(f"  {FIXTURES_DIR}/test_sharpness_range.zarr")
        aprint(f"  {FIXTURES_DIR}/test_log_scalar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_4d_scalar_lut.zarr")
        aprint(f"  {FIXTURES_DIR}/test_uint16_quantization.zarr")
        aprint(f"  {FIXTURES_DIR}/test_nd_transforms.zarr")
        aprint(f"  {FIXTURES_DIR}/test_lines.zarr")
        aprint(f"  {FIXTURES_DIR}/test_gsplats.zarr")
        aprint(f"  {FIXTURES_DIR}/test_labelled_points.zarr")
        aprint("")
        aprint("Run TypeScript tests with:")
        aprint("  cd packages/luxar-viewer && pnpm test array-decoder")

    except Exception as e:
        aprint(f"❌ Error generating test data: {e}")
        raise


if __name__ == "__main__":
    main()
