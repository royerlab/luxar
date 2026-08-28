#!/usr/bin/env python3
"""Generate test datasets for TypeScript-Python encoding compatibility tests.

This script creates small zarr datasets with all encoding modes to verify
that the TypeScript ArrayDecoder can correctly read Python-encoded data.

IMPORTANT: Uses NO compression (compressor=None) to avoid blosc/numcodecs
WASM binding issues in Node.js test environment.

Editing this file or Luxar's production Python sources makes every existing
fixture store stale. Vitest regenerates stale fixtures automatically; Playwright
fails fast with the regeneration command rather than serving old writer output.

These direct commands run only the generators; record the input stamps afterward
from ``packages/luxar-viewer/`` with
``pnpm exec tsx tools/fixture-freshness.ts``.

Run the generators from project root:
    hatch run fixtures:python packages/luxar-viewer/tests/fixtures/generate_test_data.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr
from arbol import aprint, asection

from luxar import CameraConfig, Dimension, Dimensions, LuxarZarrCompiler, ViewerConfig
from luxar._zarr_compat import consolidate as zarr_consolidate
from luxar._zarr_compat import create_array
from luxar._zarr_compat import open_group as zarr_open_group
from luxar.encoding import ArrayEncoder, EncodingMode, SemanticType
from luxar.io._compiler.finalize.hashing import compute_content_hashes

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

# Audit C1 (viewer-integration-fixtures) — declarative manifest read by
# the TypeScript global-setup (packages/luxar-viewer/src/tests/global-setup.ts).
# Previously the TS side regex-scraped `FIXTURES_DIR / "..."` usages,
# which broke if a generator function switched to single quotes, f-strings,
# path concatenation, etc. A single canonical list here means the TS
# parser only needs to match THIS one declaration.
#
# When you add a new generate_*() function below, add the output filename
# here too. The Python self-check in main() asserts that every declared
# name was actually written to disk.
FIXTURE_NAMES: list[str] = [
    "test_4d.luxar.zarr",
    "test_4d_scalar_lut.luxar.zarr",
    "test_array_ref_broadcasting.luxar.zarr",
    "test_array_refs.luxar.zarr",
    "test_blending_inherited.luxar.zarr",
    "test_broadcasting.luxar.zarr",
    "test_delta_filter.luxar.zarr",
    "test_encoding_contract_matrix.luxar.zarr",
    "test_encoding_edge_cases.luxar.zarr",
    "test_extend_to_all_4d.luxar.zarr",
    "test_gsplats.luxar.zarr",
    "test_gsplats_2d.luxar.zarr",
    "test_gsplats_normal_overlap.luxar.zarr",
    "test_gsplats_normal_overlap_reversed.luxar.zarr",
    "test_gsplats_volumetric.luxar.zarr",
    "test_gsplats_volumetric_reversed.luxar.zarr",
    "test_gsplats_rgba_occlusion.luxar.zarr",
    "test_gsplats_rgba_hdr.luxar.zarr",
    "test_gsplats_rgba_uint8.luxar.zarr",
    "test_gsplats_rgba_lut.luxar.zarr",
    "test_hdr_colors.luxar.zarr",
    "test_hierarchical_transforms.luxar.zarr",
    "test_image_overlay.luxar.zarr",
    "test_integer_colors.luxar.zarr",
    "test_labelled_partitioned_points.luxar.zarr",
    "test_labelled_points.luxar.zarr",
    "test_line_joins.luxar.zarr",
    "test_lines.luxar.zarr",
    "test_lines_blending_modes.luxar.zarr",
    "test_lines_categorical.luxar.zarr",
    "test_lines_volumetric_reversed.luxar.zarr",
    "test_linked_points.luxar.zarr",
    "test_lod_group.luxar.zarr",
    "test_lod_group_additive_finest.luxar.zarr",
    "test_lod_group_volumetric.luxar.zarr",
    "test_log_scalar.luxar.zarr",
    "test_lift_parity.luxar.zarr",
    "test_layer_4d_gsplats.luxar.zarr",
    "test_lut.luxar.zarr",
    "test_lut_u16.luxar.zarr",
    "test_mesh.luxar.zarr",
    "test_mesh_nd.luxar.zarr",
    "test_mesh_reveal_ladder.luxar.zarr",
    "test_mixed.luxar.zarr",
    "test_nd_transforms.luxar.zarr",
    "test_overview.gsplats.zarr",
    "test_partition_layer.luxar.zarr",
    "test_partition_wrong_frame.luxar.zarr",
    "test_points_blending_modes.luxar.zarr",
    "test_points_normal_overlap.luxar.zarr",
    "test_points_normal_overlap_reversed.luxar.zarr",
    "test_points_volumetric_reversed.luxar.zarr",
    "test_quantization.luxar.zarr",
    "test_sharpness_range.luxar.zarr",
    "test_standalone_gsplats.gsplats.zarr",
    "test_uint16_quantization.luxar.zarr",
]


def generate_broadcasting_test() -> None:
    """Test dataset with broadcasted (uniform) values."""
    with asection("Generating Broadcasting Test"):
        output = FIXTURES_DIR / "test_broadcasting.luxar.zarr"

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


def generate_lut_test() -> None:
    """Test dataset with LUT encoding (≤256 unique values)."""
    with asection("Generating LUT Encoding Test"):
        output = FIXTURES_DIR / "test_lut.luxar.zarr"

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


def generate_lut_u16_test() -> None:
    """Test dataset with the uint16 LUT tier (>256 unique colors).

    100,000 points with exactly 300 unique HDR colors: above the uint8 cap,
    large enough to clear the byte-modeled benefit rule (at 50k the doubled
    LUT JSON would exceed half the savings and be rejected), so the encoder
    emits `lut_uint16` in row mode — the in-browser path for Uint16Array
    LUT indices (range loader + worker + WASM row kernel).
    """
    with asection("Generating LUT uint16 Encoding Test"):
        output = FIXTURES_DIR / "test_lut_u16.luxar.zarr"

        n_points = 100_000
        rng = np.random.default_rng(7)
        positions = (rng.standard_normal((n_points, 3)) * 10).astype(np.float32)

        # Exactly 300 unique HDR colors, tiled so every palette row appears.
        palette = np.stack(
            [
                np.linspace(0.05, 9.5, 300),
                np.linspace(9.5, 0.05, 300),
                np.linspace(0.2, 4.0, 300),
            ],
            axis=1,
        ).astype(np.float32)
        colors = np.tile(palette, (-(-n_points // 300), 1))[:n_points]

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )
        # Radius 0.1, not the 0.5 the small fixtures use (#1746). The reasoning is in
        # SCREEN pixels, not world-space area: the point shader clamps
        # `pointSize = clamp(basePointSize, 1.5, maxPointSize)`
        # (packages/luxar-viewer/src/rendering/materials/point/shader-glsl.ts),
        # so shrinking a radius stops buying fill rate the moment the sprite
        # reaches that floor. At the default auto-fit framing of this cloud's
        # ~±45 bounds (fov 47, fit ratio 0.75, a 720 px-tall viewport) a
        # 0.5-radius point rasterizes 9.4 px across and a 0.1-radius one
        # 1.9 px — at/above the floor everywhere, the far side of the cloud
        # landing right on it — so the sprite is still the size the radius asks
        # for while shedding (1.9 / 9.4)^2 ≈ 25x of the overdraw that made this
        # the one fixture heavy enough to starve a shared box under the E2E
        # suite's parallel workers.
        #
        # Not smaller: 0.08 already sits on the 1.5 px floor, and below it the
        # clamp caps any further fill-rate win while the fragment shader
        # compensates the enforced sprite area with
        # `sizeScale = min(vPointSize / 1.5, 1.0)` squared into alpha — 0.4x at
        # radius 0.05 — so peak alpha drops by more than half for no saving at
        # all. Do NOT reduce the point count instead: 100k is what puts the
        # encoder in the lut_uint16 tier (see the docstring). Nothing asserts on
        # the radius — the E2E test reads colors out of the element texture, the
        # unit test reads `points/colors` only, and both readiness helpers gate
        # on `initialized` / `totalPoints`, never pixels.
        radii = np.ones(n_points, dtype=np.float32) * 0.1

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.AUTO,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "points",
                positions,
                colors=colors,  # 300 uniques -> lut_uint16 row mode
                radii=radii,
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Colors: {colors.shape} ({len(palette)} unique - lut_uint16)")


def generate_quantization_test() -> None:
    """Test dataset with quantized arrays (uint8/uint16)."""
    with asection("Generating Quantization Test"):
        output = FIXTURES_DIR / "test_quantization.luxar.zarr"

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


def generate_array_refs_test() -> None:
    """Test dataset with array references (deduplication)."""
    with asection("Generating Array References Test"):
        output = FIXTURES_DIR / "test_array_refs.luxar.zarr"

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


def generate_array_ref_broadcasting_test() -> None:
    """Test array_ref positions combined with scalar/broadcast attributes.

    This catches a subtle encoder/metadata bug: when a duplicate positions array is
    stored as an array_ref, the physical zarr array shape is ``(0, D)``. Scalar
    attributes on the same node must still be broadcast to the logical point count,
    not to the physical array_ref shape.
    """
    with asection("Generating Array Ref + Broadcasting Test"):
        output = FIXTURES_DIR / "test_array_ref_broadcasting.luxar.zarr"

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
                # Sharpness is a normalized [0, 1] knob; 2.0 exploited the
                # (now closed) scalar-broadcast validation hole.
                sharpness=0.8,
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


def generate_encoding_edge_cases_test() -> None:
    """Raw ArrayEncoder fixture covering edge cases outside scene validation."""
    with asection("Generating Raw Encoding Edge Cases Test"):
        output = FIXTURES_DIR / "test_encoding_edge_cases.luxar.zarr"
        if output.exists():
            shutil.rmtree(output)

        root = zarr_open_group(output, mode="w")
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

        # Explicit log-space positive scalar encoding — the "log" opt-in now
        # selects the geometric-log (geolog) family: MEMORY -> geolog_scalar_uint8.
        encoder.encode(
            np.logspace(-2, 2, 32, dtype=np.float32),
            root,
            "log_scalar_radii",
            SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.MEMORY,
            positive_scalar_encoding="log",
            compressor=None,
        )

        # Wide dynamic range (> 65536:1) AUTO positive scalar -> the writer's
        # geolog_scalar_uint16 (rescale-first, reserved zero level). Includes
        # exact zeros to pin the reserved level 0 round-trip.
        wide = np.logspace(-4, 9, 32, dtype=np.float32)
        wide[::7] = 0.0
        encoder.encode(
            wide,
            root,
            "geolog_amplitudes",
            SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.AUTO,
            compressor=None,
        )

        # HDR colors (values > 1) -> per-channel TRUE-log: AUTO ->
        # geolog_perchannel_u16, MEMORY -> geolog_perchannel_u8. Exact zeros
        # (whole rows and lone entries) pin the reserved zero level per column.
        hdr = np.stack(
            [
                np.logspace(-4, 1, 24, dtype=np.float32),
                np.logspace(-2, 1, 24, dtype=np.float32),
                np.logspace(-3, 0, 24, dtype=np.float32),
            ],
            axis=1,
        )
        hdr[::5] = 0.0
        hdr[1, 2] = 0.0
        encoder.encode(
            hdr,
            root,
            "hdr_colors_auto",
            SemanticType.COLOR,
            mode=EncodingMode.AUTO,
            color_mode="hdr",
            compressor=None,
        )
        # Distinct values: identical data would dedup into an array_ref and
        # drop the geolog_perchannel_u8 coverage this array exists to provide.
        hdr_mem = (hdr * np.float32(1.7)).astype(np.float32)
        encoder.encode(
            hdr_mem,
            root,
            "hdr_colors_memory",
            SemanticType.COLOR,
            mode=EncodingMode.MEMORY,
            color_mode="hdr",
            compressor=None,
        )

        # LEGACY 0-anchored log encodings: no longer produced by any policy,
        # but old stores carry them — keep decode coverage via the CUSTOM path.
        encoder.encode(
            np.logspace(-1, 2, 32, dtype=np.float32),
            root,
            "legacy_log_scalar_u8",
            SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.CUSTOM,
            custom_encoder="log_scalar_uint8",
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


def generate_encoding_contract_matrix_test() -> None:
    """Generate a declarative raw ArrayEncoder contract matrix fixture.

    Scene fixtures are realistic, but they do not exhaustively pin the encoder
    surface. This fixture intentionally creates small arrays for every semantic
    type and every decoder-supported encoding family, including dtype and LUT
    thresholds that are awkward to trigger from normal scene validation.
    """
    with asection("Generating Encoding Contract Matrix Test"):
        output = FIXTURES_DIR / "test_encoding_contract_matrix.luxar.zarr"
        if output.exists():
            shutil.rmtree(output)

        root = zarr_open_group(output, mode="w")

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
        # Encoder-EMITTED uint16 LUT tier (row-mode COLORS only,
        # 257..65,536 uniques with the byte-modeled benefit rule; scalar
        # mode never emits u16 — indices would cost what quantized scalars
        # cost, making the LUT JSON pure overhead): sized to clear the
        # break-even so the required lut_uint16 coverage is organic
        # producer output, not just the manual threshold cases below.
        u16_palette = np.stack(
            [
                np.linspace(0.05, 9.5, 300),
                np.linspace(9.5, 0.05, 300),
                np.linspace(0.2, 4.0, 300),
            ],
            axis=1,
        ).astype(np.float32)
        encode_case(
            "color_lut_row_uint16_emitted",
            np.tile(u16_palette, (334, 1))[:100_000],
            SemanticType.COLOR,
            "row-mode uint16 LUT colors (300 uniques, encoder-emitted)",
            mode=EncodingMode.AUTO,
            color_mode="hdr",
            chunks=(8192, 3),
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

        # Manual LUT threshold cases. The encoder now emits lut_uint16 itself
        # (see the *_uint16_emitted cases above); these hand-written variants
        # stay to pin the DECODE contract independent of producer behavior.
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
                create_array(root, case_id, data=indices, chunks=(17,), compressor=None)
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
                create_array(root, case_id, data=indices, chunks=(17,), compressor=None)
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


def generate_mixed_encoding_test() -> None:
    """Test dataset with mixed encoding modes in same scene."""
    with asection("Generating Mixed Encoding Test"):
        output = FIXTURES_DIR / "test_mixed.luxar.zarr"

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


def generate_4d_test() -> None:
    """Test dataset with 4D data (time dimension) for nD slicing tests."""
    with asection("Generating 4D nD Slicing Test"):
        output = FIXTURES_DIR / "test_4d.luxar.zarr"

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


def generate_layer_4d_gsplats_test() -> None:
    """Tiny time-sliced GSplat fixture for the LuxarLayer host E2E test."""
    from luxar.gsplats.lift import lift_points_to_gsplats

    with asection("Generating LuxarLayer 4D GSplat Test"):
        output = FIXTURES_DIR / "test_layer_4d_gsplats.luxar.zarr"
        rng = np.random.default_rng(2293)
        counts = (24, 72)
        positions = []
        colors = []

        for timepoint, count in enumerate(counts):
            spatial = rng.normal(0.0, 0.7, (count, 3)).astype(np.float32)
            time = np.full((count, 1), timepoint, dtype=np.float32)
            positions.append(np.column_stack([time, spatial]))
            color = np.array(
                [1.0 - 0.6 * timepoint, 0.4, 0.5 + 0.5 * timepoint], np.float32
            )
            colors.append(np.tile(color, (count, 1)))

        positions_4d = np.vstack(positions)
        colors_4d = np.vstack(colors)
        lifted = lift_points_to_gsplats(
            positions_4d,
            np.full(len(positions_4d), 0.18, dtype=np.float32),
            colors_4d,
        )
        dims = Dimensions(
            [
                Dimension("time", range=(0, 1), step=1, display=False, discrete=True),
                Dimension("x", display=True),
                Dimension("y", display=True),
                Dimension("z", display=True),
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
                "time_splats",
                lifted.centers,
                lifted.amplitudes,
                lifted.cholesky_factors,
                colors=lifted.colors,
            )

        aprint(f"✓ Created {output}")
        aprint(f"  GSplats per time step: {counts[0]} / {counts[1]}")


def generate_hierarchical_transforms_test() -> None:
    """Test dataset with hierarchical scene graph and nested transforms.

    CRITICAL: This test verifies transform composition and hierarchy:
    - Parent transforms affect children
    - Transforms are stored in correct format (column-major for THREE.js)
    - Matrix multiplication order is correct
    """
    with asection("Generating Hierarchical Transforms Test"):
        output = FIXTURES_DIR / "test_hierarchical_transforms.luxar.zarr"

        # Create a simple hierarchy:
        # Scene
        #   └─ parent_group (translated by [10, 0, 0])
        #       └─ child_points (translated by [0, 5, 0])
        # Final position should be [10, 5, 0] due to transform composition

        # Import transform functions
        from luxar.core.transforms import translate

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


def generate_integer_colors_test() -> None:
    """Test dataset with direct uint8 and uint16 SDR color arrays."""
    with asection("Generating Integer Colors Test"):
        output = FIXTURES_DIR / "test_integer_colors.luxar.zarr"

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


def generate_hdr_colors_test() -> None:
    """Test dataset with HDR colors (values > 1.0) to verify float32 color handling.

    CRITICAL: This test verifies that HDR colors are preserved through the pipeline:
    - Python stores colors as float32 with values > 1.0
    - TypeScript loads and preserves float32 colors
    - Rendering pipeline handles HDR values correctly
    """
    with asection("Generating HDR Colors Test"):
        output = FIXTURES_DIR / "test_hdr_colors.luxar.zarr"

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

        # AUTO mode: HDR colors take the PRODUCTION path
        # (geolog_perchannel_u16 -> decoded back to float32 by the viewer),
        # so the E2E render exercises the real quantized decode. Raw-float32
        # HDR storage stays covered by the PRECISION-mode unit tests.
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.AUTO,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "hdr_points",
                positions,
                colors=colors,  # HDR colors (AUTO -> geolog_perchannel_u16)
                radii=radii,
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Colors: {colors.shape} (HDR, max={colors.max():.1f})")
        aprint("  CRITICAL: Verifies float32 HDR color preservation")


def generate_log_scalar_test() -> None:
    """Test scene dataset with wide dynamic range radii.

    The scene compiler currently uses linear positive-scalar encoding by default,
    so this fixture exercises uint16 bounded-scalar radii. The raw
    ``test_encoding_edge_cases.luxar.zarr`` fixture below covers explicit log-scalar
    encoder compatibility.
    """
    with asection("Generating Wide-Range Scalar Encoding Test"):
        output = FIXTURES_DIR / "test_log_scalar.luxar.zarr"

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


def generate_4d_scalar_lut_test() -> None:
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
        output = FIXTURES_DIR / "test_4d_scalar_lut.luxar.zarr"

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


def generate_delta_filter_test() -> None:
    """Cross-language fixture for the ``luxar_delta_v1`` zarr filter (v3.3).

    Positions are a smooth 3D random walk, so after spatial ordering the
    uint16 codes ramp and the encode-time probe ENABLES the delta filter on
    the positions array. The compressor is **gzip**, not the COMPRESSOR_DISABLED
    default: the probe requires a real compressor (no compressor -> no filter),
    blosc cannot run under Node.js (see the audit note at the top of this
    file), and zarrita ships a pure-JS gzip codec — so the Node unit suite
    decodes this fixture end-to-end THROUGH the registered
    ``luxar_delta_v1`` codec (`zarr-delta-fixture.test.ts` + the generic
    round-trip expectations).

    It was zlib until the move to zarr format 3, which has no zlib codec at all:
    zarrita implements one, but zarr-python does not, so Python could no longer
    WRITE the array. gzip is the compressor both sides support in both formats
    and Node can still decode, which is the whole requirement here.

    The generator asserts the filter actually engaged — if probe gating or
    the encoder wiring regresses, fixture generation fails loudly instead of
    the TS test silently exercising an unfiltered array.
    """
    with asection("Generating Delta Filter Test"):
        import json

        from numcodecs import GZip

        output = FIXTURES_DIR / "test_delta_filter.luxar.zarr"

        rng = np.random.default_rng(1234)
        num_points = 20000
        # Smooth random walk normalized to a few-hundred-unit extent: after
        # Hilbert/Morton ordering the per-axis uint16 codes are locally
        # coherent, which is exactly what the delta filter exploits.
        walk = np.cumsum(rng.normal(0.0, 1.0, size=(num_points, 3)), axis=0)
        lo, hi = walk.min(axis=0), walk.max(axis=0)
        positions = ((walk - lo) / (hi - lo) * [300.0, 500.0, 800.0]).astype(np.float32)
        # Smooth colors too, so the u8 rgb_uint8 delta path is exercised by
        # the Node suite alongside the u16 positions path.
        cwalk = np.cumsum(rng.normal(0.0, 0.01, size=(num_points, 3)), axis=0)
        clo, chi = cwalk.min(axis=0), cwalk.max(axis=0)
        colors = ((cwalk - clo) / (chi - clo)).astype(np.float32)

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.AUTO,
            compressor=GZip(level=6),  # Node-decodable; probe needs a compressor
            float16_allowed=FLOAT16_ALLOWED,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points("points", positions, colors=colors)

        # Fail fast if the filter did not engage (probe/wiring regression) —
        # on BOTH the u16 positions and the u8 colors arrays.
        for arr_name in ("positions", "colors"):
            # Read whichever metadata document the format writes, and pull the
            # filter names out of whichever field holds them: format 2 lists
            # them under `filters` keyed by `id`, format 3 as array-to-array
            # members of the `codecs` chain keyed by `name`.
            arr_dir = output / "points" / arr_name
            v2_doc = arr_dir / ".zarray"
            meta = json.loads(
                (v2_doc if v2_doc.exists() else arr_dir / "zarr.json").read_text()
            )
            if "filters" in meta:
                filters = [f.get("id") for f in (meta.get("filters") or [])]
            else:
                filters = [c.get("name") for c in meta.get("codecs", [])]
            if "luxar_delta_v1" not in filters:
                raise RuntimeError(
                    f"test_delta_filter fixture: {arr_name} array did not receive "
                    f"the luxar_delta_v1 filter (filters={filters}). The "
                    "encode-time probe or the encoder wiring regressed."
                )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape} (uint16 + luxar_delta_v1 + gzip)")
        aprint(f"  Colors: {colors.shape} (uint8 + luxar_delta_v1 + gzip)")


def generate_uint16_quantization_test() -> None:
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
        output = FIXTURES_DIR / "test_uint16_quantization.luxar.zarr"

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


def generate_sharpness_range_test() -> None:
    """Test dataset spanning the full normalized sharpness range [0, 1].

    Verifies that the bounded_scalar_uint8 encode/decode round-trip preserves
    sharpness across its valid range. Sharpness is now a normalized [0, 1] knob
    (SHARPNESS_MAX = 1.0); the viewer maps it to the super-Gaussian exponent
    beta = 2^(6s - 2). The uint8 quantization step is 1/255 ≈ 0.0039.
    """
    with asection("Generating Sharpness Range Test"):
        output = FIXTURES_DIR / "test_sharpness_range.luxar.zarr"

        # Create 32 points sampling the full [0, 1] sharpness range, including
        # the s=0 endpoint (now valid -> beta=0.25) and s=1 (-> beta=16).
        num_points = 32
        positions = np.zeros((num_points, 3), dtype=np.float32)

        # Arrange points in a line along X axis for easy visualization
        positions[:, 0] = np.arange(num_points, dtype=np.float32)

        # Sharpness values: [0.0, 1/31, 2/31, ..., 1.0]
        sharpness = np.linspace(0.0, 1.0, num_points, dtype=np.float32)

        # Assign colors based on sharpness (gradient from blue to red)
        colors = np.zeros((num_points, 3), dtype=np.float32)
        colors[:, 0] = sharpness  # Red increases with sharpness (already [0,1])
        colors[:, 2] = 1.0 - sharpness  # Blue decreases with sharpness

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
                sharpness=sharpness,  # Normalized range [0, 1]
                radii=radii,
            )

        aprint(f"✓ Created {output}")
        aprint(f"  Positions: {positions.shape}")
        aprint(f"  Sharpness range: [{sharpness.min()}, {sharpness.max()}]")
        aprint("  Verifies bounded_scalar round-trip over the [0, 1] knob")


def generate_nd_transforms_test() -> None:
    """Test dataset with nd_transforms for verifying inverse-query in viewer.

    Creates a 4D scene (X, Y, Z, Time) with two groups:
    - GroupA: no nd_transform (baseline). 50 points at time=0 only.
    - GroupB: nd_transform={"Time": {"offset": 5}}. 50 points at local time=0,
      which should appear at world time=5 in the viewer.

    At world time=0: only GroupA points visible.
    At world time=5: only GroupB points visible (due to offset).
    """
    with asection("Generating nD Transforms Test"):
        output = FIXTURES_DIR / "test_nd_transforms.luxar.zarr"

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


def generate_lines_test() -> None:
    """Test dataset with Lines geometry type.

    Verifies that the TypeScript viewer can load and render Lines,
    including vertices, widths, colors, and segment auto-generation.
    """
    with asection("Generating Lines Test"):
        output = FIXTURES_DIR / "test_lines.luxar.zarr"

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


def generate_line_joins_test() -> None:
    """Polyline-joint artifact verification set (issues #780 / #785 / #790).

    The measurement fixture behind
    ``src/tests/e2e/line-join-artifact.spec.ts``. Five joint cases, each a
    separate node in its OWN horizontal band of world Y so the bands never
    touch and can be measured independently on one frame:

    ``curve_smooth``
        120-segment sinusoidal polyline, gentle (~10 degree) turns. The
        headline #790 case: 4.94% dark and 3.52% bright outliers before the
        miter landed, 0% and 0% after it. Both figures come from the same
        pinned frame — headless Chromium, ``dpr=1``, both columns measured
        2026-08-07 (the pre-miter one via ``&lineJoin=none``) — as does
        every other number in this docstring.
    ``zigzag_right_angle``
        16-segment 90-degree zigzag; sharp bends, still inside a miter
        limit. Its wedge is far worse than the curve's but also far wider,
        and the local-median metric only counts the part of a wedge that is
        still a couple of pixels across, so it barely registered this band
        even unmitred (0.076% dark, 0.076% bright; 0% and 0% mitred). The
        axial flux dip is the measure that sees it: p05 0.780 unmitred
        against 0.985 mitred, with the straight bands at 1.000. See the
        sensitivity envelope in ``src/tests/helpers/line-join-metrics.ts``.
    ``straight_thin``
        Straight polyline with free ends at the base width. Segment length
        0.5 against width 0.15 gives ``L/w = 3.3``, comfortably clear of the
        ``L/w >= 2`` a #780 per-joint notch (axial length ``2 x width``)
        needs to sit between joints rather than merge with its neighbours.
        This is the more sensitive of the two straight guards.
    ``straight_thick``
        Straight polyline at 4x the base width, 20 segments, ``L/w = 1.67``.
        That is below the ``L/w >= 2`` separation criterion, so a #780
        regression here would partly merge into a broad ripple rather than
        resolve into discrete notches — the guard still fires (a full
        regression models to p05 ~0.735 against the 0.9 gate) but it is the
        weaker of the pair. What must NOT happen is subdividing it further:
        the first draft used 199 segments (``L/w = 0.17``), where the
        notches merge into near-uniform dimming that normalising the flux
        profile by its own median removes entirely — a fully #780-regressed
        dense band still scores p05 = 0.98, i.e. the guard is disabled.
    ``hub_9ray``
        Nine rays meeting at ONE shared hub vertex (a degree-9 branch
        point). Authored with ``line_type="indexed"`` because joints are
        matched by vertex INDEX: nine two-vertex chains that merely repeat
        the hub coordinate would be nine unrelated free ends, not a branch
        point. This case must never be mitered — it is the control.

    All geometry lies in the ``z = 0`` plane so the pinned face-on camera
    sees it flat, and every line is flat achromatic so Rec.709 luminance is
    exactly the rendered intensity. The viewer config is photometry-grade
    (identity tone response, no bloom / AA / noise / adaptive DPR, camera
    pinned) for the same reason as ``test_lift_parity``: without it the spec
    would measure through ACES plus bloom plus jitter.

    On-screen width is four times what the authored width suggests, which
    matters because the metric's sensitivity depends on pixel sizes. The
    shader computes ``rawPixelWidth = width * uPerspectiveLineScale / dist``
    with ``uPerspectiveLineScale = res.y / tan(fov / 2)``, and expands the
    quad by ``rawPixelWidth`` on EACH side, so the rendered full width is
    ``4 * authored_width * px_per_world_unit``. At the pinned framing below
    that is 36 px per world unit, giving 21.6 px for ``straight_thin``,
    86.4 px for ``straight_thick`` and 57.6 px (28.8 px half-width) for the
    two bend bands. Measured cross-sections agree: 21.95 and 82.0 px, the
    latter a few percent under nominal where the perpendicular falloff drops
    below the background cutoff.

    Band world-space AABBs (z = 0) — the measurement rectangles the E2E spec
    projects through the live camera. All four horizontal bands span
    ``x in [-10, 10]`` and their AABB X range is inset exactly 1.0 unit from
    those ends, so the free-end cap ramps stay OUT of the measured region;
    ``hub_9ray`` is measured whole. ``curve_smooth`` gets a taller box than
    its siblings because its 28.8 px half-width plus its 0.7-unit amplitude
    would otherwise reach the box edge exactly. Keep this table in sync with
    ``LINE_JOIN_BANDS`` in ``src/tests/e2e/line-join-artifact.spec.ts``, and
    the segment counts in sync with ``EXPECTED_LINE_SEGMENTS`` there.

    ==================== ============== ==============
    node                 x range        y range
    ==================== ============== ==============
    curve_smooth         [-9.0,   9.0]  [ 6.3,   9.7]
    zigzag_right_angle   [-9.0,   9.0]  [ 2.5,   5.5]
    straight_thin        [-9.0,   9.0]  [-1.5,   1.5]
    straight_thick       [-9.0,   9.0]  [-5.5,  -2.5]
    hub_9ray             [-1.5,   1.5]  [-9.5,  -6.5]
    ==================== ============== ==============
    """
    with asection("Generating Line-Joints Artifact Test"):
        output = FIXTURES_DIR / "test_line_joins.luxar.zarr"

        # Geometry spans x in [-10, 10]; band centers are 4 units apart so
        # the AABBs above leave a gutter of at least 0.6 units between
        # neighbours (1.0 everywhere except below curve_smooth's taller box).
        x_min, x_max = -10.0, 10.0
        base_width = 0.15
        thick_width = 4.0 * base_width
        bend_width = 0.4

        # Flat achromatic: R == G == B, so Rec.709 luminance IS the rendered
        # intensity. 0.6 rather than 1.0 keeps the tube core off the 8-bit
        # ceiling under the linear (tone_mapping="None") photometry config —
        # a clipped core would hide exactly the flux dips this fixture
        # exists to detect.
        grey = (0.6, 0.6, 0.6)

        # curve_smooth: 121 vertices -> 120 segments, 4 periods over the span.
        curve_x = np.linspace(x_min, x_max, 121, dtype=np.float32)
        curve = np.column_stack(
            [
                curve_x,
                8.0 + 0.7 * np.sin(2.0 * np.pi * curve_x / 5.0),
                np.zeros_like(curve_x),
            ]
        ).astype(np.float32)

        # zigzag_right_angle: consecutive deltas are (+1.25, +1.25) and
        # (+1.25, -1.25) — dot product exactly 0, i.e. a true 90-degree turn.
        # 16 segments of 1.25 span exactly [-10, 10], so the documented
        # 1.0-unit AABB inset holds for this band like the others.
        zig_i = np.arange(17)
        zig_x = (x_min + 1.25 * zig_i).astype(np.float32)
        zig_y = (4.0 + 0.625 * np.where(zig_i % 2 == 0, -1.0, 1.0)).astype(np.float32)
        zig_z = np.zeros_like(zig_x)
        zigzag = np.column_stack([zig_x, zig_y, zig_z]).astype(np.float32)

        # straight_thin / straight_thick: collinear chains with free ends.
        # A #780 notch is 2 x width long, so segment length wants L/w >= 2
        # for notches to stay separated (3.3 here for thin, 1.67 for thick).
        # See the docstring: over-subdivision merges the notches into uniform
        # dimming and the flux normalisation then cancels it entirely.
        thin_x = np.linspace(x_min, x_max, 41, dtype=np.float32)
        thin = np.column_stack(
            [thin_x, np.zeros_like(thin_x), np.zeros_like(thin_x)]
        ).astype(np.float32)

        thick_x = np.linspace(x_min, x_max, 21, dtype=np.float32)
        thick = np.column_stack(
            [thick_x, np.full_like(thick_x, -4.0), np.zeros_like(thick_x)]
        ).astype(np.float32)

        # hub_9ray: one shared hub row + nine tips, wired by index so all
        # nine edges reference the SAME hub vertex (degree-9 branch point).
        hub_center = np.array([0.0, -8.0, 0.0], dtype=np.float32)
        ray_angles = np.arange(9) * (2.0 * np.pi / 9.0)
        ray_tips = np.column_stack(
            [
                hub_center[0] + 1.2 * np.cos(ray_angles),
                hub_center[1] + 1.2 * np.sin(ray_angles),
                np.zeros(9),
            ]
        ).astype(np.float32)
        hub_vertices = np.vstack([hub_center[None, :], ray_tips]).astype(np.float32)
        hub_indices = np.column_stack(
            [np.zeros(9, dtype=np.uint32), np.arange(1, 10, dtype=np.uint32)]
        )

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Photometry-grade viewer config (same rationale as
        # test_lift_parity): identity tone response and every non-linear or
        # stochastic post-effect off, camera pinned face-on. fov=47 at
        # distance 23 puts +-10 world units of Y across the viewport
        # height — 36 px per world unit at 720p. See the docstring for the
        # 4x factor between authored width and rendered pixel width: the
        # thin band renders 21.6 px across and the thick one 86.4 px.
        viewer_config = ViewerConfig(
            camera=CameraConfig(
                position=(0.0, 0.0, 23.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0),
                fov=47.0,
            ),
            background_color="#000000",
            tone_mapping="None",
            exposure=0.0,
            global_offset=0.0,
            global_gamma=1.0,
            bloom_enabled=False,
            vignette_enabled=False,
            detector_noise_enabled=False,
            fxaa_enabled=False,
            msaa_enabled=False,
            ssaa_enabled=False,
            chromatic_lens_distortion_enabled=False,
            adaptive_dpr_enabled=False,
            control_type="orbit",
            auto_rotate=False,
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)

            # Default blending (additive) is what production lines use — it
            # is deliberately NOT overridden here.
            scene.add_lines(
                "curve_smooth",
                curve,
                widths=bend_width,
                colors=grey,
                line_type="polyline",
            )
            scene.add_lines(
                "zigzag_right_angle",
                zigzag,
                widths=bend_width,
                colors=grey,
                line_type="polyline",
            )
            scene.add_lines(
                "straight_thin",
                thin,
                widths=base_width,
                colors=grey,
                line_type="polyline",
            )
            scene.add_lines(
                "straight_thick",
                thick,
                widths=thick_width,
                colors=grey,
                line_type="polyline",
            )
            scene.add_lines(
                "hub_9ray",
                hub_vertices,
                widths=bend_width,
                colors=grey,
                indices=hub_indices,
                line_type="indexed",
            )

        aprint(f"  Created {output}")
        aprint(f"  curve_smooth:       {len(curve)} vertices, {len(curve) - 1} segs")
        aprint(f"  zigzag_right_angle: {len(zigzag)} vertices, {len(zigzag) - 1} segs")
        aprint(f"  straight_thin:      {len(thin)} vertices, width {base_width}")
        aprint(f"  straight_thick:     {len(thick)} vertices, width {thick_width}")
        aprint(f"  hub_9ray:           {len(hub_vertices)} vertices, 9 indexed rays")


def generate_lines_categorical_test() -> None:
    """Lines + Points sliced by a non-displayed categorical dimension.

    Regression fixture for the "Lines are not re-culled when scrubbing a
    non-displayed dimension" bug: a 4D scene with a hidden categorical
    dim `sel` (categories A/B). One Lines node and one Points node sit at
    sel=0, a second pair at sel=1. Scrubbing `sel` must SWAP each pair
    (A xor B) for Lines exactly as it does for Points — pre-fix, Lines
    accumulated (A ∪ B) because the empty-slice projection threw and the
    stale mesh was never cleared.

    The Points pair is the known-good control: any E2E assertion on the
    Lines pair should hold for the Points pair in the same scene.
    """
    with asection("Generating Lines Categorical (nD scrub) Test"):
        output = FIXTURES_DIR / "test_lines_categorical.luxar.zarr"

        def helix(cx: float, phase: float, n: int = 400) -> np.ndarray:
            t = np.linspace(0, 6 * np.pi, n)
            return np.column_stack(
                [
                    cx + 0.3 * np.cos(t + phase),
                    0.3 * np.sin(t + phase),
                    t / (6 * np.pi) - 0.5,
                ]
            ).astype(np.float32)

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
                Dimension("sel", display=False, categories=["A", "B"]),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            for slot, phase in [(0, 0.0), (1, np.pi)]:
                color = [1.0, 0.4, 0.2] if slot == 0 else [0.2, 0.6, 1.0]

                v = helix(-0.6, phase)
                v4 = np.column_stack([v, np.full(len(v), slot, dtype=np.float32)])
                scene.add_lines(
                    f"line_{'ab'[slot]}",
                    vertices=v4,
                    widths=0.02,
                    colors=np.tile(color, (len(v), 1)).astype(np.float32),
                    line_type="polyline",
                    extend_to_all=[],  # intentionally sliced by `sel`
                )

                p = helix(0.6, phase)
                p4 = np.column_stack([p, np.full(len(p), slot, dtype=np.float32)])
                scene.add_points(
                    f"pts_{'ab'[slot]}",
                    p4,
                    radii=0.03,
                    colors=np.tile(color, (len(p), 1)).astype(np.float32),
                    extend_to_all=[],  # intentionally sliced by `sel`
                )

        aprint(f"  Created {output}")
        aprint(
            "  line_a/pts_a at sel=0, line_b/pts_b at sel=1 (399 segments / 400 points each)"
        )
        aprint("  Scrubbing `sel` must swap the pairs for Lines exactly as for Points")


def generate_extend_to_all_test() -> None:
    """Fully-extended ``extend_to_all`` nodes in a 4D scene.

    Regression fixture for "a node whose ``extend_to_all`` covers ALL
    non-displayed dims is never queried": such a node was hard-skipped
    before the per-node tolerance override ran, so it fetched nothing, its
    additive ladder stayed frozen, and gsplat levels were filtered out
    during nD->3D projection.

    Two fixture properties do the discriminating work, and both matter:

    1. The extended nodes are authored at ``time=2`` via ``fill``, which is
       deliberately NOT the scene's initial slice (``time=0``). A layer whose
       ``fill`` happens to equal the opening slice renders correctly even
       when fully broken — that coincidence is what hid the bug in the demos.
    2. ``sliced_pts`` carries a real ``time`` column at ``time=3`` and is the
       known-good control: it must be EMPTY wherever the extended nodes are
       full, so an assertion that passes vacuously (everything visible
       everywhere) cannot survive.

    All three geometry kinds are extended, because they reach the extended
    query through different code: Points/GSplats take the tolerance override
    directly, Lines opt out of the PARTIAL override (their segment bounds
    already encode the extent) and so exercise the full-extend path alone,
    and GSplats additionally derive their `extendToAllDims` set from the
    `1e10` tolerance sentinel during projection.

    ``ext_pts`` carries an additive ladder so a frozen ladder (a node that
    loads its first rung and then never advances) is distinguishable from a
    converged one: only a node re-queried per sweep reaches all 1200 points.
    """
    with asection("Generating extend_to_all (fully-extended) Test"):
        output = FIXTURES_DIR / "test_extend_to_all_4d.luxar.zarr"

        rng = np.random.default_rng(1157)

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
                Dimension(
                    "time",
                    unit="frame",
                    range=(0.0, 4.0),
                    step=1.0,
                    discrete=True,
                    display=False,
                ),
            ]
        )

        n_pts = 1200
        ext_positions = (rng.normal(0.0, 0.35, (n_pts, 3)) - [0.8, 0.0, 0.0]).astype(
            np.float32
        )

        t = np.linspace(0.0, 4 * np.pi, 200)
        ext_vertices = np.column_stack(
            [0.8 + 0.25 * np.cos(t), 0.25 * np.sin(t), t / (4 * np.pi) - 0.5]
        ).astype(np.float32)

        n_splats = 40
        ext_centers = np.column_stack(
            [
                np.linspace(-0.4, 0.4, n_splats),
                np.full(n_splats, 0.7),
                np.zeros(n_splats),
            ]
        ).astype(np.float32)
        # Covariance Cholesky factors (L11, L21, L22, L31, L32, L33) — see
        # generate_gsplats_test on why these are covariance, not precision.
        # `dim_order=` expands them to the scene's 4D packing.
        ext_cholesky = np.zeros((n_splats, 6), dtype=np.float32)
        ext_cholesky[:, 0] = 0.06  # L11
        ext_cholesky[:, 2] = 0.06  # L22
        ext_cholesky[:, 5] = 0.06  # L33

        n_control = 400
        control = np.column_stack(
            [
                rng.normal(0.0, 0.25, (n_control, 3)) + [0.0, -0.8, 0.0],
                np.full(n_control, 3.0),
            ]
        ).astype(np.float32)

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # extend_to_all=['time'] is INFERRED from the unmapped `time` dim.
            scene.add_points(
                "ext_pts",
                ext_positions,
                radii=0.02,
                colors=np.tile([1.0, 0.5, 0.1], (n_pts, 1)).astype(np.float32),
                dim_order=["x", "y", "z"],
                fill={"time": 2.0},
                additive_lod=dict(counts=[300, 700], method="random", seed=0),
            )

            scene.add_lines(
                "ext_lines",
                vertices=ext_vertices,
                widths=0.02,
                colors=np.tile([0.2, 0.9, 0.4], (len(ext_vertices), 1)).astype(
                    np.float32
                ),
                line_type="polyline",
                dim_order=["x", "y", "z"],
                fill={"time": 2.0},
            )

            scene.add_gsplats(
                "ext_gsplats",
                ext_centers,
                amplitudes=np.full(n_splats, 1.0, dtype=np.float32),
                cholesky_factors=ext_cholesky,
                colors=np.tile([0.4, 0.6, 1.0], (n_splats, 1)).astype(np.float32),
                dim_order=["x", "y", "z"],
                fill={"time": 2.0},
            )

            # Control: a real `time` column, visible ONLY at time=3.
            scene.add_points(
                "sliced_pts",
                control,
                radii=0.02,
                colors=np.tile([0.2, 0.7, 1.0], (n_control, 1)).astype(np.float32),
                extend_to_all=[],  # intentionally sliced by `time`
            )

        aprint(f"  Created {output}")
        aprint(
            f"  ext_pts ({n_pts} pts, ladder 300/700/{n_pts}), ext_lines "
            f"({len(ext_vertices)} verts), ext_gsplats ({n_splats} splats): "
            "extend_to_all=['time'], authored at time=2"
        )
        aprint(
            f"  sliced_pts ({n_control} pts) at time=3 only — the control that "
            "must be empty where the extended nodes are full"
        )


# ─── Blending-mode E2E fixtures ─────────────────────────────────────────
#
# Shared by blending-modes.spec.ts / lines-blending-modes.spec.ts. All
# six canonical viewer blending modes, in fixture order. Points/lines
# render `volumetric` through the phase-1 ADDITIVE fallback state
# (VOLUMETRIC_BLENDING_SPEC.md §5.1) — the layer still exercises the
# stored-mode round-trip and the fallback assertion.
BLENDING_MODES = ["normal", "additive", "max", "opaque", "luminous", "volumetric"]

# Pure-channel (or channel-union) saturated colors, one per mode. Chosen
# so the E2E pixel discriminator is sound: `additive` (green) and
# `luminous` (red) are single-channel, so a pixel with BOTH r and g high
# and b low can only come from cross-layer additive accumulation — no
# single layer's base color can fake it (cyan/blue/white/magenta all
# carry a high blue channel and are excluded by the b < min(r,g)/2 term).
BLENDING_MODE_COLORS = {
    "normal": [0.0, 1.0, 1.0],  # cyan
    "additive": [0.0, 1.0, 0.0],  # green
    "max": [0.0, 0.0, 1.0],  # blue
    "opaque": [1.0, 1.0, 1.0],  # white
    "luminous": [1.0, 0.0, 0.0],  # red
    "volumetric": [1.0, 0.0, 1.0],  # magenta (high blue ⇒ discriminator-excluded)
}

# Cloud/line centers on a circle of radius 0.7 (Venn-style): adjacent
# layers overlap near the view center; `additive` (18°) and `luminous`
# (-54°) are ADJACENT so their overlap lens contains ONLY those two
# layers — the E2E spec projects the two centers and samples between
# them. Angles in degrees. `volumetric` (126°) slots into the
# normal↔max gap, far from the additive/luminous lens, so the
# discriminator geometry is undisturbed.
BLENDING_MODE_ANGLES = {
    "normal": 162.0,
    "additive": 18.0,
    "max": 90.0,
    "opaque": -126.0,
    "luminous": -54.0,
    "volumetric": 126.0,
}

# Small per-layer depth stagger so depth-writing modes (opaque, opaque-
# normal) are exercised without changing the XY overlap layout.
BLENDING_MODE_Z = {
    "normal": 0.0,
    "additive": 0.1,
    "max": 0.2,
    "opaque": -0.2,
    "luminous": -0.1,
    "volumetric": 0.3,
}


def _sunflower_disk(n: int, radius: float) -> np.ndarray:
    """Deterministic golden-angle (sunflower) disk of ``n`` 2D points."""
    indices = np.arange(n, dtype=np.float64) + 0.5
    r = radius * np.sqrt(indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)]).astype(np.float32)


# Point radii spanning sub-pixel to comfortably-resolved at the parity spec's
# camera distance. The two ends matter: effect B (uncompensated 2D dilation)
# only bites below ~1 px of screen sigma, effect A (the tau chord factor)
# scales as 1/radius, so a single radius would miss one of them.
LIFT_PARITY_RADII: list[float] = [0.02, 0.05, 0.15, 0.40]
LIFT_PARITY_COLUMNS: list[float] = [-7.5, -2.5, 2.5, 7.5]
LIFT_PARITY_ROW_Y: dict[str, float] = {"points": 3.5, "gsplats": -3.5}
LIFT_PARITY_N = 20000
LIFT_PARITY_BLOB_R = 1.5
# Low enough that the additive stack through a blob stays well below clipping,
# so the spec's photometry runs in the linear part of the response.
LIFT_PARITY_OPACITY = 0.003


def _lift_parity_blob(rng: "np.random.Generator", cx: float, cy: float) -> np.ndarray:
    """Uniform-in-ball cluster centred at (cx, cy, 0)."""
    v = rng.normal(size=(LIFT_PARITY_N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = LIFT_PARITY_BLOB_R * rng.random(LIFT_PARITY_N) ** (1.0 / 3.0)
    p = (v * r[:, None]).astype(np.float32)
    p[:, 0] += cx
    p[:, 1] += cy
    return p


def generate_lift_parity_test() -> None:
    """Points vs their ``lift_points_to_gsplats`` twin, at four radii.

    A substitutive-LOD points ladder is MIXED geometry: the coarse levels are
    lifted gsplats, the finest level stays the original Points node. The two
    families must therefore render the same scene identically in EVERY blending
    mode, or the ladder visibly changes character as it switches levels.

    Layout is a 4x2 grid — one column per radius, top row Points, bottom row the
    lifted twin, identical cluster geometry — so a spec can crop each cell and
    compare brightness directly. Everything sits under ONE ``layer=True`` group
    so a single Blend control drives all eight nodes.

    Four distinct defects have been measured with this shape. Three come from
    VOLUMETRIC_BLENDING_SPEC.md (2026-08-02): the tau chord factor (volumetric,
    fixed), uncompensated 2D dilation (every sum mode, including additive,
    fixed), and the peak-vs-sum lift calibration (max/normal/opaque, still
    open). The fourth was issue #1993 — ``opaque`` dropped the points' alpha and
    all their alpha-carried photometry until #1994 restored alpha-over while
    preserving the mode's depth semantics (fixed).
    """
    from luxar.gsplats.lift import lift_points_to_gsplats

    with asection("Generating Points-vs-lifted-GSplats Parity Test"):
        output = FIXTURES_DIR / "test_lift_parity.luxar.zarr"
        rng = np.random.default_rng(0)

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Photometry-grade viewer config: identity tone response and every
        # non-linear / stochastic post-effect off, camera pinned. Without this
        # the spec measures through ACES + bloom + TAA jitter and no brightness
        # ratio is trustworthy.
        viewer_config = ViewerConfig(
            camera=CameraConfig(
                position=(0.0, 0.0, 33.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0),
                fov=47.0,
            ),
            background_color="#000000",
            tone_mapping="None",
            exposure=0.0,
            global_offset=0.0,
            global_gamma=1.0,
            bloom_enabled=False,
            vignette_enabled=False,
            detector_noise_enabled=False,
            fxaa_enabled=False,
            msaa_enabled=False,
            ssaa_enabled=False,
            chromatic_lens_distortion_enabled=False,
            adaptive_dpr_enabled=False,
            control_type="orbit",
            auto_rotate=False,
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)
            grp = scene.add_group("probe", layer=True)

            for i, radius in enumerate(LIFT_PARITY_RADII):
                pos_p = _lift_parity_blob(
                    rng, LIFT_PARITY_COLUMNS[i], LIFT_PARITY_ROW_Y["points"]
                )
                pos_g = pos_p.copy()
                pos_g[:, 1] += (
                    LIFT_PARITY_ROW_Y["gsplats"] - LIFT_PARITY_ROW_Y["points"]
                )

                colors = np.full((LIFT_PARITY_N, 3), 255, np.uint8)
                radii = np.full(LIFT_PARITY_N, radius, np.float32)

                grp.add_points(
                    f"pts_r{i}",
                    pos_p,
                    colors=colors,
                    radii=radii,
                    sharpness=np.full(LIFT_PARITY_N, 0.5, np.float32),
                    opacity=LIFT_PARITY_OPACITY,
                )

                # opacity=1.0 into the lift, LIFT_PARITY_OPACITY on the node —
                # the same split the points row uses, so the two rows differ
                # only in geometry family.
                lifted = lift_points_to_gsplats(pos_g, radii, colors, opacity=1.0)
                grp.add_gsplats(
                    f"gsp_r{i}",
                    lifted.centers,
                    lifted.amplitudes,
                    lifted.cholesky_factors,
                    colors=colors,
                    opacity=LIFT_PARITY_OPACITY,
                )

        aprint(f"✅ Created: {output}")


def generate_points_blending_modes_test() -> None:
    """Five overlapping point-cloud layers, one per blending mode.

    Node names are literally ``points_<mode>`` with ``blending_mode``
    set accordingly and ``layer=True``, so the E2E spec can assert the
    exact per-mode THREE material state by node name (see
    ``getCompleteBlendingState`` in blending-state.ts). Layout and
    colors are deterministic (sunflower disks on a Venn circle, no
    RNG) — see the module-level constants above for the geometry that
    the pixel discriminators rely on.
    """
    with asection("Generating Points Blending-Modes Test"):
        output = FIXTURES_DIR / "test_points_blending_modes.luxar.zarr"

        n = 600
        disk = _sunflower_disk(n, radius=0.8)

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

            for mode in BLENDING_MODES:
                angle = np.deg2rad(BLENDING_MODE_ANGLES[mode])
                center = np.array(
                    [0.7 * np.cos(angle), 0.7 * np.sin(angle), BLENDING_MODE_Z[mode]],
                    dtype=np.float32,
                )
                positions = np.column_stack(
                    [disk[:, 0], disk[:, 1], np.zeros(n, dtype=np.float32)]
                )
                positions = (positions + center).astype(np.float32)
                colors = np.tile(
                    np.array([BLENDING_MODE_COLORS[mode]], dtype=np.float32), (n, 1)
                )
                scene.add_points(
                    f"points_{mode}",
                    positions,
                    colors=colors,
                    radii=0.035,
                    layer=True,
                    blending_mode=mode,
                )

        aprint(f"  Created {output}")
        aprint(
            f"  {len(BLENDING_MODES)} layers × {n} points, one per mode: "
            f"{', '.join(BLENDING_MODES)}"
        )


def generate_lines_blending_modes_test() -> None:
    """Five crossing polyline layers, one per blending mode.

    The lines twin of :func:`generate_points_blending_modes_test`
    (three-geometry symmetry): node names ``lines_<mode>`` with
    ``blending_mode`` set and ``layer=True``. Each layer is one straight
    polyline through the origin at a distinct angle so all five cross in
    the view center; widths are generous so the lines are visible.
    """
    with asection("Generating Lines Blending-Modes Test"):
        output = FIXTURES_DIR / "test_lines_blending_modes.luxar.zarr"

        n_vertices = 24
        t = np.linspace(-1.2, 1.2, n_vertices).astype(np.float32)

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

            for mode in BLENDING_MODES:
                angle = np.deg2rad(BLENDING_MODE_ANGLES[mode])
                vertices = np.column_stack(
                    [
                        t * np.cos(angle),
                        t * np.sin(angle),
                        np.full(n_vertices, BLENDING_MODE_Z[mode], dtype=np.float32),
                    ]
                ).astype(np.float32)
                colors = np.tile(
                    np.array([BLENDING_MODE_COLORS[mode]], dtype=np.float32),
                    (n_vertices, 1),
                )
                scene.add_lines(
                    f"lines_{mode}",
                    vertices,
                    widths=0.08,
                    colors=colors,
                    line_type="polyline",
                    layer=True,
                    blending_mode=mode,
                )

        aprint(f"  Created {output}")
        aprint(
            f"  5 polyline layers crossing at the origin: {', '.join(BLENDING_MODES)}"
        )


def generate_blending_inherited_test() -> None:
    """Group with blending_mode='max' + a child leaf that does NOT set it.

    Cross-stack regression fixture for blending-mode inheritance:
    Python writers no longer stamp a default ``blending_mode`` on
    leaves, so ``child_points``' .zattrs genuinely OMITS the attr on
    disk, and the viewer must compose the effective mode from the
    nearest ancestor (``surface_group``'s ``max``) — both on the
    material (MaxEquation blend state) and in the Layers panel's
    initial layer state.
    """
    with asection("Generating Blending-Inherited Test"):
        output = FIXTURES_DIR / "test_blending_inherited.luxar.zarr"

        n = 200
        disk = _sunflower_disk(n, radius=0.8)
        positions = np.column_stack(
            [disk[:, 0], disk[:, 1], np.zeros(n, dtype=np.float32)]
        ).astype(np.float32)
        colors = np.tile(np.array([[1.0, 0.6, 0.1]], dtype=np.float32), (n, 1))

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
            group = scene.add_group("surface_group", blending_mode="max", layer=True)
            scene.add_points(
                "child_points",
                positions,
                colors=colors,
                radii=0.05,
                parent=group,
                layer=True,
                # NO blending_mode — must inherit 'max' from surface_group.
            )

        # Pin the PR1 contract end-to-end: the child leaf's attributes must
        # NOT carry a blending_mode key (unset ⇒ inherited in the viewer).
        # Read through the facade so this works whichever format was written —
        # format 2 keeps attributes in a separate `.zattrs`, format 3 inside
        # `zarr.json`, and naming either one directly makes the check silently
        # unrunnable on the other (it raised FileNotFoundError at format 3).
        from luxar._zarr_compat import read_node_attrs

        child_attrs = read_node_attrs(output / "surface_group" / "child_points")
        if child_attrs is None:
            raise RuntimeError(
                "test_blending_inherited: child_points has no readable metadata "
                "document — the fixture did not write the node at all."
            )
        if "blending_mode" in child_attrs:
            raise RuntimeError(
                "test_blending_inherited: child_points attributes unexpectedly "
                "carry a blending_mode key — the writer stamped a default "
                "again, breaking the inheritance regression fixture."
            )

        aprint(f"  Created {output}")
        aprint("  surface_group(blending_mode=max) → child_points (attr omitted)")


def generate_gsplats_test() -> None:
    """Test dataset with GSplats (Gaussian Splats) geometry type.

    Verifies that the TypeScript viewer can load and render GSplats,
    including centers, amplitudes, cholesky_factors, and colors.
    """
    with asection("Generating GSplats Test"):
        output = FIXTURES_DIR / "test_gsplats.luxar.zarr"

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
        # Keep this aligned with the Group.add_gsplats and AdditiveSubLOD
        # authoring docstrings, which own the public covariance convention.
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


def generate_gsplats_2d_test() -> None:
    """GSplats in a 2-DIMENSIONAL scene (only two displayed dims).

    The configuration that used to crash the WASM projection kernel: with
    ``displayDims.length == 2`` the display-marginal Cholesky read
    ``display_dims[2]`` out of bounds and the ``panic = "abort"`` crate trapped,
    so every 2D gsplats node failed to load. The renderer's Cholesky buffer is
    always the 6-element packed-3D layout, so the third row is synthesized: zero
    off-diagonals plus a phantom diagonal equal to the geometric mean of the real
    pivots (NOT an epsilon — an epsilon-thin splat is invisible in sum blending).

    Deliberately authored with ANISOTROPIC, correlated 2D factors: an isotropic
    fixture would pass even if the marginal were computed wrongly, since every
    candidate phantom value coincides when the two pivots are equal.

    No unit test consumes this fixture yet — it stages a future decoder /
    marginal-Cholesky test that loads a real 2D gsplats scene and asserts the
    synthesized phantom row (2D display-dims path), rather than exercising the
    kernel in isolation as the WASM parity tests do.
    """
    with asection("Generating GSplats 2D Test"):
        output = FIXTURES_DIR / "test_gsplats_2d.luxar.zarr"

        num_splats = 16
        # 4x4 grid in the XY plane; 2 columns only, no z.
        centers = np.zeros((num_splats, 2), dtype=np.float32)
        centers[:, 0] = np.arange(num_splats, dtype=np.float32) % 4
        centers[:, 1] = np.arange(num_splats, dtype=np.float32) // 4

        amplitudes = np.linspace(0.6, 1.6, num_splats).astype(np.float32)

        # Packed 2D Cholesky: 3 elements per splat, [L00, L10, L11]. Covariance
        # factors (sigma-like), NOT precision factors — same convention as
        # generate_gsplats_test.
        cholesky = np.zeros((num_splats, 3), dtype=np.float32)
        for i in range(num_splats):
            cholesky[i, 0] = 0.30 + 0.02 * i  # L00
            cholesky[i, 1] = 0.10 - 0.01 * (i % 5)  # L10 (correlation)
            cholesky[i, 2] = 0.45 - 0.01 * i  # L11 (anisotropic vs L00)

        colors = np.zeros((num_splats, 3), dtype=np.float32)
        colors[:, 0] = np.linspace(1, 0, num_splats).astype(np.float32)
        colors[:, 1] = np.linspace(0, 1, num_splats).astype(np.float32)
        colors[:, 2] = 0.4

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
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
                "test_splats_2d",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )

        aprint(f"  Created {output}")
        aprint(f"  Centers: {centers.shape}, Cholesky: {cholesky.shape} (2D packed)")


def generate_points_normal_overlap_test() -> None:
    """Two large overlapping points at staggered depth, 'normal' blending.

    The points sibling of :func:`generate_gsplats_normal_overlap_test`
    (three-geometry symmetry — points are depth-sorted too): a back red
    point and a front green point whose sprites overlap in screen space
    under the viewer's auto-framed camera, plus a small off-axis blue
    reference point. Radii are generous relative to the 0.5-unit center
    separation so the two sprites genuinely overlap;
    ``blending_mode='normal'`` with ``opacity=0.5`` makes the
    compositing ORDER visible in the overlap pixels (green-over-red vs
    the mirror image).
    """
    with asection("Generating Points Normal-Overlap Test"):
        output = FIXTURES_DIR / "test_points_normal_overlap.luxar.zarr"

        # Same layout as the gsplat overlap fixture: back point at z=0,
        # front point at z=1 overlapping in screen space, blue reference
        # off-axis (an anchor outside the overlap).
        positions = np.array(
            [
                [-0.25, 0.0, 0.0],  # back point (red)
                [0.25, 0.0, 1.0],  # front point (green), overlaps in screen space
                [3.0, 2.0, 0.0],  # small reference point (blue), no overlap
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [1.0, 0.1, 0.1],  # red
                [0.1, 1.0, 0.1],  # green
                [0.1, 0.1, 1.0],  # blue
            ],
            dtype=np.float32,
        )
        # Sprite radii ≈ the gsplat fixture's ~1σ extent: the two big
        # sprites (0.7 > half the 0.5-unit separation) overlap solidly.
        radii = np.array([0.7, 0.7, 0.3], dtype=np.float32)

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

            scene.add_points(
                "overlap_points",
                positions,
                colors=colors,
                radii=radii,
                blending_mode="normal",
                opacity=0.5,
            )

        aprint(f"  Created {output}")


def generate_points_normal_overlap_reversed_test() -> None:
    """The points normal-overlap scene declared FRONT-TO-BACK (sort gate).

    Identical to :func:`generate_points_normal_overlap_test` except the
    points are DECLARED in reversed (front-first) order — the points
    twin of :func:`generate_gsplats_normal_overlap_reversed_test`. NOTE:
    the compiler Morton-reorders storage, so the on-disk order is
    spatial, not the declaration order — what makes this fixture a
    depth-sort gate is its geometry: under the viewer's auto-framed
    camera the near (green) point sits BETWEEN the two far points in
    storage order, so the identity ordering is not back-to-front and the
    E2E assertion (aSortedIndex non-identity + view-z monotone, see
    blending-modes.spec.ts) fails without a working SortWorker.
    """
    with asection("Generating Points Normal-Overlap-Reversed Test"):
        output = FIXTURES_DIR / "test_points_normal_overlap_reversed.luxar.zarr"

        # Same points as the canonical overlap fixture, front point FIRST.
        positions = np.array(
            [
                [0.25, 0.0, 1.0],  # front point (green) stored first
                [-0.25, 0.0, 0.0],  # back point (red) stored second
                [3.0, 2.0, 0.0],  # small reference point (blue), no overlap
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [0.1, 1.0, 0.1],  # green (front)
                [1.0, 0.1, 0.1],  # red (back)
                [0.1, 0.1, 1.0],  # blue
            ],
            dtype=np.float32,
        )
        radii = np.array([0.7, 0.7, 0.3], dtype=np.float32)

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

            scene.add_points(
                "overlap_points_reversed",
                positions,
                colors=colors,
                radii=radii,
                blending_mode="normal",
                opacity=0.5,
            )

        aprint(f"  Created {output}")


def generate_points_volumetric_reversed_test() -> None:
    """The points overlap scene in `volumetric`, declared FRONT-TO-BACK.

    The points twin of :func:`generate_gsplats_volumetric_reversed_test`.
    Identical geometry to
    :func:`generate_points_normal_overlap_reversed_test` — only
    ``blending_mode`` differs — so the pair isolates the blending mode as
    the single variable behind the depth-sort decision.

    `needsDepthSort` is `normal ∪ volumetric` and the coordinator judges
    order-dependence on it for ALL four geometry types, but until this
    fixture existed the only E2E gate on the volumetric arm filtered
    ``nodeType === 'gsplats'``: nothing would have caught points (or
    lines) regressing to unsorted volumetric compositing.

    NOTE: as in the normal-mode twin, the compiler Morton-reorders
    storage, so the on-disk order is spatial rather than the declaration
    order — declaring front-first is not itself what makes this a gate.
    Its GEOMETRY is: the auto-framed camera looks straight down -z
    (`camera-framing.ts`), and the near (green) point lands BETWEEN the
    two far points in storage order, so the identity ordering is not
    back-to-front and the E2E assertion (aSortedIndex non-identity +
    view-z monotone) fails unless volumetric points reach the SortWorker.
    """
    with asection("Generating Points Volumetric-Reversed Test"):
        output = FIXTURES_DIR / "test_points_volumetric_reversed.luxar.zarr"

        # Same points as the normal-overlap-reversed twin, front point FIRST.
        positions = np.array(
            [
                [0.25, 0.0, 1.0],  # front point (green) stored first
                [-0.25, 0.0, 0.0],  # back point (red) stored second
                [3.0, 2.0, 0.0],  # small reference point (blue), no overlap
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [0.1, 1.0, 0.1],  # green (front)
                [1.0, 0.1, 0.1],  # red (back)
                [0.1, 0.1, 1.0],  # blue
            ],
            dtype=np.float32,
        )
        radii = np.array([0.7, 0.7, 0.3], dtype=np.float32)

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

            scene.add_points(
                "volumetric_points_reversed",
                positions,
                colors=colors,
                radii=radii,
                blending_mode="volumetric",
                absorption=1.0,
                opacity=0.5,
            )

        aprint(f"  Created {output}")


def generate_lines_volumetric_reversed_test() -> None:
    """The lines twin of :func:`generate_points_volumetric_reversed_test`.

    Three independent segments — a front one, a back one overlapping it in
    screen space, and an off-axis reference — declared FRONT-FIRST with
    ``blending_mode="volumetric"``. Lines sort on SEGMENT MIDPOINTS
    (`commit-lines-geometry.ts`), which is the delta from the points
    fixture: the same gate, exercised through the other centers provider.

    Same caveat as the points twin: the compiler Hilbert-reorders segments,
    so the declaration order is not what lands on disk. The gate is the
    z-structure, which survives any reordering of these three — the front
    segment stores BETWEEN the two z=0 ones, so under the -z auto-framed
    camera the identity ordering is not back-to-front. Keep every segment
    at constant z if you edit this: the E2E check reconstructs the midpoint
    sort key from the line texture, so a segment slanted in z would need
    the fixture's overlap reasoning redone.
    """
    with asection("Generating Lines Volumetric-Reversed Test"):
        output = FIXTURES_DIR / "test_lines_volumetric_reversed.luxar.zarr"

        # Segment pairs, front segment FIRST. Midpoints stagger in z the
        # same way the points fixture's centers do.
        vertices = np.array(
            [
                [-0.6, 0.0, 1.0],  # front segment (green)
                [0.9, 0.0, 1.0],
                [-0.9, 0.0, 0.0],  # back segment (red), overlaps on screen
                [0.6, 0.0, 0.0],
                [2.6, 2.0, 0.0],  # off-axis reference (blue)
                [3.4, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [0.1, 1.0, 0.1],
                [0.1, 1.0, 0.1],
                [1.0, 0.1, 0.1],
                [1.0, 0.1, 0.1],
                [0.1, 0.1, 1.0],
                [0.1, 0.1, 1.0],
            ],
            dtype=np.float32,
        )
        widths = np.array([0.7, 0.7, 0.7, 0.7, 0.3, 0.3], dtype=np.float32)

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
                "volumetric_lines_reversed",
                vertices,
                widths,
                colors=colors,
                line_type="segments",
                blending_mode="volumetric",
                absorption=1.0,
                opacity=0.5,
            )

        aprint(f"  Created {output}")


def generate_gsplats_normal_overlap_test() -> None:
    """Two large overlapping splats at staggered depth, 'normal' blending.

    Exercises the gsplat premultiplied coverage-alpha path
    (LUXAR_NORMAL_PREMULT; see GSPLAT_DEPTH_SORTING_SPEC.md Phase 0):
    with real alpha-over compositing the framebuffer behind the front
    splat must show through (pre-fix, gsplat 'normal' emitted alpha=1.0
    and occluded everything behind it). The two splats overlap in
    screen space from the default camera; a third small reference splat
    sits outside the overlap as an anchor for visual assertions.
    """
    with asection("Generating GSplats Normal-Overlap Test"):
        output = FIXTURES_DIR / "test_gsplats_normal_overlap.luxar.zarr"

        # Two overlapping splats staggered in z, plus one small off-axis
        # reference splat. Sigmas are deliberately modest relative to the
        # scene span: the viewer's screen-coverage safety fade
        # (uMaxExtentFactor, default 0.33 of the viewport) starts fading
        # splats at HALF that fraction — oversized splats would be
        # legitimately culled and the fixture would render black (see the
        # warning in generate_gsplats_test).
        centers = np.array(
            [
                [-0.25, 0.0, 0.0],  # back splat (red)
                [0.25, 0.0, 1.0],  # front splat (green), overlaps in screen space
                [3.0, 2.0, 0.0],  # small reference splat (blue), no overlap
            ],
            dtype=np.float32,
        )
        amplitudes = np.array([2.0, 2.0, 1.0], dtype=np.float32)

        cholesky = np.zeros((3, 6), dtype=np.float32)
        for i, sigma in enumerate((0.35, 0.35, 0.3)):
            cholesky[i, 0] = sigma  # L11
            cholesky[i, 2] = sigma  # L22
            cholesky[i, 5] = sigma  # L33

        colors = np.array(
            [
                [1.0, 0.1, 0.1],  # red
                [0.1, 1.0, 0.1],  # green
                [0.1, 0.1, 1.0],  # blue
            ],
            dtype=np.float32,
        )

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
                "overlap_splats",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
                blending_mode="normal",
                opacity=0.5,
            )

        aprint(f"  Created {output}")


def generate_gsplats_normal_overlap_reversed_test() -> None:
    """The normal-overlap scene declared FRONT-TO-BACK (depth-sort gate).

    Identical to :func:`generate_gsplats_normal_overlap_test` except the
    splats are DECLARED in reversed (front-first) order. NOTE: the
    compiler Morton-reorders storage, so the on-disk order is spatial,
    not the declaration order — what makes this fixture a depth-sort
    gate is its geometry: under the viewer's auto-framed camera the near
    (green) splat sits BETWEEN the two far splats in storage order, so
    the identity ordering is not back-to-front and the E2E assertion
    (aSortedIndex non-identity + view-z monotone, see
    blending-modes.spec.ts Phase 2 test) fails without a working
    SortWorker (spec §5).
    """
    with asection("Generating GSplats Normal-Overlap-Reversed Test"):
        output = FIXTURES_DIR / "test_gsplats_normal_overlap_reversed.luxar.zarr"

        # Same splats as the canonical overlap fixture, front splat FIRST.
        centers = np.array(
            [
                [0.25, 0.0, 1.0],  # front splat (green) stored first
                [-0.25, 0.0, 0.0],  # back splat (red) stored second
                [3.0, 2.0, 0.0],  # small reference splat (blue), no overlap
            ],
            dtype=np.float32,
        )
        amplitudes = np.array([2.0, 2.0, 1.0], dtype=np.float32)

        cholesky = np.zeros((3, 6), dtype=np.float32)
        for i, sigma in enumerate((0.35, 0.35, 0.3)):
            cholesky[i, 0] = sigma  # L11
            cholesky[i, 2] = sigma  # L22
            cholesky[i, 5] = sigma  # L33

        colors = np.array(
            [
                [0.1, 1.0, 0.1],  # green (front)
                [1.0, 0.1, 0.1],  # red (back)
                [0.1, 0.1, 1.0],  # blue
            ],
            dtype=np.float32,
        )

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
                "overlap_splats_reversed",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
                blending_mode="normal",
                opacity=0.5,
            )

        aprint(f"  Created {output}")


def generate_gsplats_volumetric_test() -> None:
    """The overlap scene in `volumetric` mode (emission-absorption).

    Same geometry as :func:`generate_gsplats_normal_overlap_test` (back
    red splat, front green splat overlapping in screen space, off-axis
    blue reference) but ``blending_mode="volumetric"`` at full opacity
    with the default absorption 1.0. The E2E suite drives kappa at
    runtime through the real material path (``updateAbsorption``):

    - I1 (additive limit): kappa=0 then a mode switch to `additive`
      must render pixel-identical (same session, same camera).
    - Absorption darkening: a high kappa must darken the back splat
      seen through the front one relative to the kappa=0 frame.
    """
    with asection("Generating GSplats Volumetric Test"):
        output = FIXTURES_DIR / "test_gsplats_volumetric.luxar.zarr"

        centers = np.array(
            [
                [-0.25, 0.0, 0.0],  # back splat (red)
                [0.25, 0.0, 1.0],  # front splat (green), overlaps in screen space
                [3.0, 2.0, 0.0],  # small reference splat (blue), no overlap
            ],
            dtype=np.float32,
        )
        amplitudes = np.array([2.0, 2.0, 1.0], dtype=np.float32)

        cholesky = np.zeros((3, 6), dtype=np.float32)
        for i, sigma in enumerate((0.35, 0.35, 0.3)):
            cholesky[i, 0] = sigma  # L11
            cholesky[i, 2] = sigma  # L22
            cholesky[i, 5] = sigma  # L33

        colors = np.array(
            [
                [1.0, 0.1, 0.1],  # red
                [0.1, 1.0, 0.1],  # green
                [0.1, 0.1, 1.0],  # blue
            ],
            dtype=np.float32,
        )

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
                "volumetric_splats",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
                blending_mode="volumetric",
                absorption=1.0,
                layer=True,
            )

        aprint(f"  Created {output}")


def generate_gsplats_volumetric_reversed_test() -> None:
    """The volumetric scene declared FRONT-TO-BACK (depth-sort gate).

    The volumetric twin of
    :func:`generate_gsplats_normal_overlap_reversed_test`: identical
    geometry declared front-first, so the identity storage ordering is
    not back-to-front. The E2E assertion (aSortedIndex non-identity +
    view-z monotone) fails unless `needsDepthSort` routes volumetric
    commits through the SortWorker exactly like `normal`.
    """
    with asection("Generating GSplats Volumetric-Reversed Test"):
        output = FIXTURES_DIR / "test_gsplats_volumetric_reversed.luxar.zarr"

        centers = np.array(
            [
                [0.25, 0.0, 1.0],  # front splat (green) stored first
                [-0.25, 0.0, 0.0],  # back splat (red) stored second
                [3.0, 2.0, 0.0],  # small reference splat (blue), no overlap
            ],
            dtype=np.float32,
        )
        amplitudes = np.array([2.0, 2.0, 1.0], dtype=np.float32)

        cholesky = np.zeros((3, 6), dtype=np.float32)
        for i, sigma in enumerate((0.35, 0.35, 0.3)):
            cholesky[i, 0] = sigma  # L11
            cholesky[i, 2] = sigma  # L22
            cholesky[i, 5] = sigma  # L33

        colors = np.array(
            [
                [0.1, 1.0, 0.1],  # green (front)
                [1.0, 0.1, 0.1],  # red (back)
                [0.1, 0.1, 1.0],  # blue
            ],
            dtype=np.float32,
        )

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
                "volumetric_splats_reversed",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
                blending_mode="volumetric",
                absorption=1.0,
            )

        aprint(f"  Created {output}")


def generate_gsplats_rgba_occlusion_test() -> None:
    """RGBA per-element opacity: a black high-α front splat occludes.

    A bright back splat (white, α=1) with a BLACK front splat (RGB≈0)
    that has high per-element opacity α=0.95, overlapping in screen
    space, plus an off-axis white reference. The front splat emits no
    light, so:

    - In `volumetric`, its high α maps to a large optical depth
      (w = −ln(1−0.95) ≈ 3.0): it ABSORBS the back splat's light — the
      overlap darkens relative to the reference.
    - In `additive` (κ=0 limit), a black splat contributes nothing and
      does not occlude — the overlap is as bright as the back splat
      alone (α is a linear contribution scale; black × α = black added).

    This is the RGBA occlusion discriminator: the same fixture renders
    visibly different between the two modes ONLY because the alpha
    channel drives optical depth in volumetric. `layer=True` so the E2E
    suite can flip the mode at runtime through the real material path.
    """
    with asection("Generating GSplats RGBA-Occlusion Test"):
        output = FIXTURES_DIR / "test_gsplats_rgba_occlusion.luxar.zarr"

        centers = np.array(
            [
                [-0.25, 0.0, 0.0],  # back splat (white, opaque)
                [0.25, 0.0, 1.0],  # front splat (black, high α), overlaps
                [3.0, 2.0, 0.0],  # white reference, no overlap
            ],
            dtype=np.float32,
        )
        amplitudes = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        cholesky = np.zeros((3, 6), dtype=np.float32)
        for i, sigma in enumerate((0.35, 0.35, 0.3)):
            cholesky[i, 0] = sigma  # L11
            cholesky[i, 2] = sigma  # L22
            cholesky[i, 5] = sigma  # L33

        # RGBA: the 4th column is per-element opacity. The front splat is
        # near-black (no emission) but nearly opaque (α=0.95) — a pure-ink
        # occluder that only matters in volumetric.
        colors = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],  # back: white, opaque
                [0.0, 0.0, 0.0, 0.95],  # front: black, high opacity
                [1.0, 1.0, 1.0, 1.0],  # reference: white, opaque
            ],
            dtype=np.float32,
        )

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
                "rgba_occlusion_splats",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
                blending_mode="volumetric",
                absorption=1.0,
                layer=True,
            )

        aprint(f"  Created {output}")


def generate_gsplats_rgba_hdr_test() -> None:
    """RGBA colors with HDR RGB (values > 1) — the geolog per-channel path.

    Real-bytes coverage for the one RGBA combination the unit matrix never
    round-tripped: HDR float32 RGBA under the default AUTO encoding, where
    each column (including alpha, [0,1]) gets its own geolog anchors. A
    stride or per-channel-anchor bug shows up as corrupted alpha at decode.
    """
    with asection("Generating GSplats RGBA-HDR Test"):
        output = FIXTURES_DIR / "test_gsplats_rgba_hdr.luxar.zarr"
        rng = np.random.default_rng(7)
        n = 2000  # > LUT palette cap, so AUTO picks geolog per-channel
        centers = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
        amplitudes = np.ones(n, dtype=np.float32)
        cholesky = np.zeros((n, 6), dtype=np.float32)
        cholesky[:, [0, 2, 5]] = 0.3
        colors = np.empty((n, 4), dtype=np.float32)
        colors[:, :3] = rng.uniform(0.0, 8.0, (n, 3))  # HDR: > 1.0
        colors[:, 3] = rng.uniform(0.05, 1.0, n)  # opacity stays [0, 1]
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )
        with LuxarZarrCompiler(output, compressor=None) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_gsplats(
                "rgba_hdr_splats",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )
        aprint(f"Generated {output}")


def generate_gsplats_rgba_uint8_test() -> None:
    """Native uint8 RGBA colors — integer passthrough with a 4th column.

    Alpha is stored at full scale (255 = opaque); the decoder must preserve
    the integer dtype and the worker normalizes /255.
    """
    with asection("Generating GSplats RGBA-uint8 Test"):
        output = FIXTURES_DIR / "test_gsplats_rgba_uint8.luxar.zarr"
        rng = np.random.default_rng(11)
        n = 24
        centers = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
        amplitudes = np.ones(n, dtype=np.float32)
        cholesky = np.zeros((n, 6), dtype=np.float32)
        cholesky[:, [0, 2, 5]] = 0.3
        colors = rng.integers(0, 256, (n, 4), dtype=np.uint8)
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )
        with LuxarZarrCompiler(output, compressor=None) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_gsplats(
                "rgba_uint8_splats",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )
        aprint(f"Generated {output}")


def generate_gsplats_rgba_lut_test() -> None:
    """Few unique RGBA rows over many splats — the 4-wide LUT row encode.

    The LUT planner accepts rows up to width 4 (structural.py); this pins
    that an RGBA palette round-trips through lut encoding with alpha intact.
    """
    with asection("Generating GSplats RGBA-LUT Test"):
        output = FIXTURES_DIR / "test_gsplats_rgba_lut.luxar.zarr"
        rng = np.random.default_rng(13)
        n = 64
        centers = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
        amplitudes = np.ones(n, dtype=np.float32)
        cholesky = np.zeros((n, 6), dtype=np.float32)
        cholesky[:, [0, 2, 5]] = 0.3
        palette = np.array(
            [
                [1.0, 0.0, 0.0, 0.25],
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.75],
                [1.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        colors = palette[rng.integers(0, 4, n)]
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )
        with LuxarZarrCompiler(output, compressor=None) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_gsplats(
                "rgba_lut_splats",
                centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )
        aprint(f"Generated {output}")


def generate_standalone_gsplats_test() -> None:
    """Standalone v3.1 ``.gsplats.zarr`` — a *detached* gsplats leaf node.

    Exercises the viewer's bare-node load path (Phase 4): opening this file via
    ``?src=`` must dispatch the root as a gsplats leaf, auto-frame it, and
    render it. The cluster is offset well away from the origin so a *working*
    auto-frame is observable — if framing breaks, the camera stays at the
    default origin and the offset cluster falls off-screen (black canvas).

    Written node-safe (PRECISION float32, no blosc) like the other fixtures so
    zarrita-js can read it; ``ordering="morton"`` exercises the spatial-index
    path. There is no scene wrapper — the file root IS the gsplats node.
    """
    with asection("Generating Standalone GSplats Test"):
        from luxar.gsplats import GSplatData

        output = FIXTURES_DIR / "test_standalone_gsplats.gsplats.zarr"
        rng = np.random.default_rng(0)
        n = 40
        # Cluster offset from the origin so auto-framing is observable.
        centers = (
            np.array([12.0, 8.0, 5.0], dtype=np.float32)
            + rng.uniform(-2.0, 2.0, size=(n, 3))
        ).astype(np.float32)
        amplitudes = np.linspace(0.6, 1.5, n).astype(np.float32)
        cholesky = np.zeros((n, 6), dtype=np.float32)
        sigma = 0.6
        cholesky[:, 0] = sigma  # L11
        cholesky[:, 2] = sigma  # L22
        cholesky[:, 5] = sigma  # L33
        colors = np.zeros((n, 3), dtype=np.float32)
        colors[:, 0] = np.linspace(1.0, 0.0, n)
        colors[:, 1] = np.linspace(0.0, 1.0, n)
        colors[:, 2] = 0.6

        GSplatData(
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky,
            colors=colors,
        ).save(
            output,
            ordering="morton",
            encoding_mode=EncodingMode.PRECISION,
            compressor=COMPRESSOR_DISABLED,
        )
        aprint(f"  Created {output} ({n} splats, standalone v3.0 leaf)")


def generate_lod_group_test() -> None:
    """Three-level ``lod_group`` fixture for the lod-group E2E spec.

    Builds an LODGroup with three gsplats children at exponentially
    increasing splat counts (8 / 32 / 128 — small enough to render
    instantly, large enough that the auto-derivation produces a real
    threshold spread). The lod_group carries ``layer=True`` so the
    Layers panel exposes the Active-level dropdown.

    Each child writes its own ``coverage_fraction`` attr (viewport-relative,
    coarsest 0.0 → finest 1.0), so the runtime selector has the data it needs
    even though this fixture's explicit thresholds are not the same as what a
    real-world authoring path (auto-derived by screen-occupancy halving, finest
    0.5 for a whole-object ladder) would supply.
    """
    with asection("Generating LODGroup Test"):
        output = FIXTURES_DIR / "test_lod_group.luxar.zarr"

        def _make_level_splats(n: int, rng: np.random.RandomState) -> dict:
            centers = rng.rand(n, 3).astype(np.float32) * 5.0
            amplitudes = np.full(n, 1.0, dtype=np.float32)
            cholesky = np.tile(
                np.array([0.3, 0, 0.3, 0, 0, 0.3], dtype=np.float32), (n, 1)
            )
            return dict(
                centers=centers, amplitudes=amplitudes, cholesky_factors=cholesky
            )

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        rng = np.random.RandomState(42)
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            lod = scene.add_lod_group("multires", layer=True)
            lod.add_gsplats(
                "child_0", **_make_level_splats(8, rng), coverage_fraction=0.0
            )
            lod.add_gsplats(
                "child_1", **_make_level_splats(32, rng), coverage_fraction=0.5
            )
            lod.add_gsplats(
                "child_2", **_make_level_splats(128, rng), coverage_fraction=1.0
            )

        aprint(f"  Created {output}")
        aprint("  3 levels: 8 / 32 / 128 splats, coverage_fraction 0.0 / 0.5 / 1.0")


def generate_lod_group_volumetric_test() -> None:
    """Volumetric twin of ``generate_lod_group_test`` (same 3-level shape).

    The lod_group node authors ``blending_mode="volumetric"`` and
    ``absorption=1.0`` (nearest-setter-wins composition covers the three
    children), so the E2E spec can exercise the LOD coverage cross-fade on a
    blendable VOLUMETRIC group: two adjacent levels simultaneously visible
    mid-band, emission-absorption blend state on the displayed material.
    """
    with asection("Generating LODGroup Volumetric Test"):
        output = FIXTURES_DIR / "test_lod_group_volumetric.luxar.zarr"

        def _make_level_splats(n: int, rng: np.random.RandomState) -> dict:
            centers = rng.rand(n, 3).astype(np.float32) * 5.0
            amplitudes = np.full(n, 1.0, dtype=np.float32)
            cholesky = np.tile(
                np.array([0.3, 0, 0.3, 0, 0, 0.3], dtype=np.float32), (n, 1)
            )
            return dict(
                centers=centers, amplitudes=amplitudes, cholesky_factors=cholesky
            )

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        rng = np.random.RandomState(42)
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            lod = scene.add_lod_group(
                "multires",
                layer=True,
                blending_mode="volumetric",
                absorption=1.0,
            )
            lod.add_gsplats(
                "child_0", **_make_level_splats(8, rng), coverage_fraction=0.0
            )
            lod.add_gsplats(
                "child_1", **_make_level_splats(32, rng), coverage_fraction=0.5
            )
            lod.add_gsplats(
                "child_2", **_make_level_splats(128, rng), coverage_fraction=1.0
            )

        aprint(f"  Created {output}")
        aprint("  3 volumetric levels: 8 / 32 / 128 splats, kappa 1.0")


def generate_lod_group_additive_finest_test() -> None:
    """``lod_group`` whose FINEST level is itself a progressive Points node.

    The composition under test: a node that is BOTH a level of a substitutive
    LOD group AND additively laddered (``stream:``-style progressive). On disk::

        composed/                kind=lod, selector=coverage, default_level=0
          child_0/               gsplats, coverage_fraction=0.0    (coarse)
          child_1/               gsplats, coverage_fraction=0.5    (coarse)
          child_2/               points,  coverage_fraction=1.0,
                                 n_additive_sublods=3
            additive_0/ additive_1/ additive_2/

    Hand-composed through the PUBLIC API (``add_lod_group`` + two
    ``add_gsplats`` + one ``add_points(..., additive_lod=...)``) rather than
    via ``substitutive_lod=``: the latter synthesises the coarse levels through
    the torch-backed gsplat reduction pipeline, which would pull torch into the
    TypeScript fixture path and make the fixture bytes depend on the local
    torch/BLAS build.

    Three additive levels (not two) on purpose: two levels complete the ladder
    on a single streaming pass (``shouldStopAfterLevel`` in
    ``src/data/loaders/progressive/streaming-policy.ts`` only breaks AFTER a
    level past the first-paint floor, so level 0 + level 1 already exhaust a
    2-level ladder), which makes any "streams progressively" assertion vacuous.
    With three, a first pass lands 2 of 3 and a second pass has real work left.
    """
    with asection("Generating LODGroup-with-Additive-Finest Test"):
        output = FIXTURES_DIR / "test_lod_group_additive_finest.luxar.zarr"

        def _make_level_splats(n: int, rng: np.random.RandomState) -> dict:
            centers = rng.rand(n, 3).astype(np.float32) * 5.0
            amplitudes = np.full(n, 1.0, dtype=np.float32)
            cholesky = np.tile(
                np.array([0.3, 0, 0.3, 0, 0, 0.3], dtype=np.float32), (n, 1)
            )
            return dict(
                centers=centers, amplitudes=amplitudes, cholesky_factors=cholesky
            )

        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        rng = np.random.RandomState(42)
        positions = (rng.rand(2000, 3) * 5.0).astype(np.float32)
        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            lod = scene.add_lod_group("composed", layer=True)
            lod.add_gsplats(
                "child_0", **_make_level_splats(8, rng), coverage_fraction=0.0
            )
            lod.add_gsplats(
                "child_1", **_make_level_splats(32, rng), coverage_fraction=0.5
            )
            # Cumulative counts [50, 250] over 2000 points → per-level
            # 50 / 200 / 1750 (three additive_<i> subgroups).
            lod.add_points(
                "child_2",
                positions=positions,
                coverage_fraction=1.0,
                additive_lod=dict(counts=[50, 250], method="random", seed=0),
            )

        aprint(f"  Created {output}")
        aprint("  3 levels: 8 / 32 gsplats + 2000-point additive-laddered Points")
        aprint("  finest child_2: 3 additive sub-LODs (50 / 200 / 1750 points)")


def generate_overview_test() -> None:
    """Standalone ``overview``-recipe fixture — a lod_group with a GROUP child.

    Built through the REAL recipe pipeline (``build_recipe(data, "overview")``)
    so the on-disk shape is authoritative: a ``kind=lod`` group whose children
    are ``[coarse merged leaf, kind=partition of additive-ladder part leaves]``
    (coarsest→finest, ``default_level=0``). This is the one recipe whose fine
    child loads through the viewer's deferred-GROUP path
    (``load-lod-group-node.ts::canDeferGroup``), exercising:

    - the never-downgrade display gate's SUBTREE aggregation
      (``scene/lod-display-gate.ts::subtreeDisplayProgress``), and
    - the post-activation refinement kick
      (``SceneLoader.kickRefinementIfIdle``) — without it the partition's
      part ladders stall at chunk-1 until the next slice change.

    600 splats, ``max_elements=200`` → 4 spatial parts, 3 additive LODs per
    part; node-safe encoding (PRECISION float32, no blosc) like every fixture.
    """
    with asection("Generating Overview Recipe Test"):
        from luxar.gsplats import GSplatData
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.lod.recipes import RecipeParams, build_recipe

        output = FIXTURES_DIR / "test_overview.gsplats.zarr"
        rng = np.random.default_rng(7)
        n = 600
        centers = rng.uniform(0.0, 10.0, size=(n, 3)).astype(np.float32)
        amplitudes = np.linspace(0.5, 1.5, n).astype(np.float32)
        cholesky = np.zeros((n, 6), dtype=np.float32)
        sigma = 0.25
        cholesky[:, 0] = sigma  # L11
        cholesky[:, 2] = sigma  # L22
        cholesky[:, 5] = sigma  # L33

        data = GSplatData(
            centers=centers, amplitudes=amplitudes, cholesky_factors=cholesky
        )
        tree = build_recipe(
            data,
            "overview",
            RecipeParams(max_elements=200, n_lods=3),
        )
        write_gsplats_tree(
            output,
            tree,
            ordering="morton",
            encoding_mode=EncodingMode.PRECISION,
            compressor=COMPRESSOR_DISABLED,
        )
        aprint(f"  Created {output}")
        aprint("  overview: coarse cap + kind=partition of 4 laddered parts")


def generate_partition_layer_test() -> None:
    """A SCENE whose single layer is a grafted kind=partition gsplats node.

    The shape that shipped broken: `add_gsplats_from_file` on a nested
    (kind=partition) `.gsplats.zarr` grafts the subtree and marks the WRAPPER
    `layer=True`, so the layers panel shows one row that fans out to N parts.

    `blending_mode` is nearest-setter-wins, so the graft must stamp it on the
    wrapper only — a copy on each part shadows the wrapper and makes the
    layer's Blend control inert. `layers-panel.spec.ts` loads this fixture,
    switches Blend on the partition layer, and asserts EVERY part's material
    followed. The existing blending case deliberately picks a non-group layer,
    which is why the partition path had no coverage.

    60 splats, `max_elements=30` → 2 spatial parts; node-safe encoding
    (PRECISION float32, no blosc) like every fixture.
    """
    with asection("Generating Partition-Layer Test (E2E layers panel)"):
        from luxar.gsplats import GSplatData
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        output = FIXTURES_DIR / "test_partition_layer.luxar.zarr"

        # Two well-separated clusters so the BSP split is unambiguous.
        rng = np.random.default_rng(11)
        a = rng.uniform(-6.0, -2.0, size=(30, 3)).astype(np.float32)
        b = rng.uniform(2.0, 6.0, size=(30, 3)).astype(np.float32)
        centers = np.concatenate([a, b], axis=0)
        n = centers.shape[0]
        cholesky = np.zeros((n, 6), dtype=np.float32)
        cholesky[:, [0, 2, 5]] = 0.6
        data = GSplatData(
            centers=centers,
            amplitudes=np.linspace(0.5, 1.0, n).astype(np.float32),
            cholesky_factors=cholesky,
        )

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "parts.gsplats.zarr"
            write_gsplats_tree(
                src,
                data.to_spatial_partition(max_elements=30),
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
                compressor=COMPRESSOR_DISABLED,
            )

            dims = Dimensions(
                [
                    Dimension("x", display=True),
                    Dimension("y", display=True),
                    Dimension("z", display=True),
                ]
            )
            with LuxarZarrCompiler(
                output,
                encoding_mode=EncodingMode.PRECISION,
                compressor=COMPRESSOR_DISABLED,
            ) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_gsplats_from_file(
                    "tiles",
                    str(src),
                    opacity=1.0,
                    absorption=1.0,
                    blending_mode="volumetric",
                    layer=True,
                )

        aprint(f"  Created {output}")
        aprint("  partition layer: 1 layer row → 2 parts, blending_mode on the wrapper")

        wrong_frame = FIXTURES_DIR / "test_partition_wrong_frame.luxar.zarr"
        if wrong_frame.exists():
            shutil.rmtree(wrong_frame)
        shutil.copytree(output, wrong_frame)
        root = zarr_open_group(wrong_frame, mode="r+")
        partition = root["tiles"]

        def shift_tree(node: dict) -> dict:
            if "part" in node:
                return dict(node)
            return {
                "axis": node["axis"],
                "split": float(node["split"]) + 1000.0,
                "left": shift_tree(node["left"]),
                "right": shift_tree(node["right"]),
            }

        partition.attrs["bsp_tree"] = shift_tree(dict(partition.attrs["bsp_tree"]))
        compute_content_hashes(root)
        zarr_consolidate(root)
        aprint(f"  Created {wrong_frame}")
        aprint("  wrong-frame partition: valid Python scene with shifted BSP planes")


def _icosphere(subdivisions: int = 2, radius: float = 1.0) -> tuple:
    """A welded, closed icosphere: vertices, faces, and per-vertex unit normals.

    Chosen over a cube or a grid for the mesh fixtures because it is the shape that
    actually exercises the mesh path:

    * **Welded** — vertices are SHARED between adjacent faces, so `gl_VertexID` is a
      genuine many-to-one pick target rather than incidentally per-triangle. A
      de-indexed mesh would make the shared-vertex pick semantics untestable.
    * **Closed** — an nD slab cull removes front faces and EXPOSES the interior back
      faces, which is what the `gl_FrontFacing` normal flip and the pick pass's
      `side` sync exist for. A flat sheet never shows a back face.
    * **Smooth normals that are not axis-aligned** — every vertex normal is its own
      direction, so a build that ignored `shading` and always shaded from derivatives
      would render visibly differently. On a cube the two agree per face.

    Returns `(vertices, faces, normals)` — the normals are exactly the unit positions,
    which is the analytic answer for a sphere and therefore a fixture whose expected
    shading can be reasoned about rather than merely recorded.
    """
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    # Subdivide, sharing each new edge midpoint between the two faces that own it —
    # this is what keeps the result WELDED.
    for _ in range(subdivisions):
        cache: dict = {}
        out = []

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in cache:
                nonlocal verts
                m = (verts[i] + verts[j]) / 2.0
                verts = np.vstack([verts, m[None, :]])
                cache[key] = len(verts) - 1
            return cache[key]

        for a_i, b_i, c_i in faces:
            ab = midpoint(a_i, b_i)
            bc = midpoint(b_i, c_i)
            ca = midpoint(c_i, a_i)
            out += [[a_i, ab, ca], [b_i, bc, ab], [c_i, ca, bc], [ab, bc, ca]]
        faces = out

    # Project onto the sphere; the unit position IS the analytic normal.
    lengths = np.linalg.norm(verts, axis=1, keepdims=True)
    normals = (verts / lengths).astype(np.float32)
    return (
        (normals * radius).astype(np.float32),
        np.asarray(faces, dtype=np.uint32),
        normals,
    )


def generate_mesh_test() -> None:
    """Baseline 3D mesh fixture: a shaded, labelled, coloured icosphere.

        Covers, in one node, everything the mesh render path has that the other three
        geometry types do not:

        * stored per-vertex `normals` + `normal_dims` (the smooth-shading variant);
        * per-vertex **RGBA** where the alpha varies — a band of vertices sits at 0.25,
          BELOW the 0.5 `opaque` cutout default, so the fixture has a visible hole and the
          cutout has something to act on. An all-opaque fixture would make the cutout
          branch untestable;
        * per-vertex `scalars` for the colormap path;
        * per-vertex `labels`, so hover/pick resolves through the CSR at exactly the
          vertex granularity the pick shader reports.

    Four nodes, one axis each — `colors` and `colormap` are mutually exclusive on a
    single node, so the direct-colour and LUT paths cannot share one:

    * `sphere` — RGBA (with the sub-cutoff cap) + labels + smooth stored normals;
    * `scalar_sphere` — the same geometry with `scalars` + `colormap` instead;
    * `flat_patch` — authored `shading="flat"`, so a test can compare the two shading
      variants inside ONE scene rather than across two fixtures. Nearly EDGE-ON to the
      opening camera;
    * `flat_facing` — also `shading="flat"`, but FACE-ON, which is what makes the
      derivative normal's forced viewer-facing sign observable at all. See the comment
      on its vertices for why edge-on cannot show it.
    """
    with asection("Generating Mesh Test"):
        output = FIXTURES_DIR / "test_mesh.luxar.zarr"

        vertices, faces, normals = _icosphere(subdivisions=2, radius=1.0)
        n_v = len(vertices)

        # RGBA. Hue follows the vertex position so the surface is readable; alpha is
        # 1.0 except for the +z cap, which drops to 0.25 — under the cutout default
        # (0.5) that cap is discarded, so the sphere has a hole you can see through.
        colors = np.empty((n_v, 4), dtype=np.float32)
        colors[:, 0] = (vertices[:, 0] + 1.0) / 2.0
        colors[:, 1] = (vertices[:, 1] + 1.0) / 2.0
        colors[:, 2] = (vertices[:, 2] + 1.0) / 2.0
        colors[:, 3] = np.where(vertices[:, 2] > 0.75, 0.25, 1.0)

        # Latitude, so the colormap has a monotone field to map.
        scalars = ((vertices[:, 2] + 1.0) / 2.0).astype(np.float32)
        labels = [f"vertex {i} (z={vertices[i, 2]:.2f})" for i in range(n_v)]

        # A flat 2x2 quad patch, offset in x, authored shading="flat". Its normal is
        # along X, which puts it nearly EDGE-ON to the opening camera: the two spheres
        # stack vertically on screen and this patch sits to their right, so screen-up is
        # ~+Y, screen-right ~+X, and the view axis ~Z.
        patch_v = np.array(
            [[2.0, -1, -1], [2.0, 1, -1], [2.0, 1, 1], [2.0, -1, 1]], dtype=np.float32
        )
        patch_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        # A second flat quad, this one FACE-ON: it lies in a z = const plane, so its
        # normal is along Z — parallel to the view axis.
        #
        # That distinction is the whole reason this node exists, and it is not
        # cosmetic. The derivative-normal path forces its result viewer-facing
        # (`z >= 0`) because GLSL's `dFdy` is bottom-up where WGSL's `dpdy` is
        # top-down, so `cross(dFdx, dFdy)` carries opposite sign on the two backends.
        # EDGE-ON, that forcing is unobservable: `N.z ~= 0`, and flipping the sign of
        # ~0 leaves `wrap = clamp(0 * 0.5 + 0.5) = 0.5` unchanged — measured on real
        # WebGPU as byte-identical pixels with the flip removed, which is why the
        # edge-on patch alone could not verify it. FACE-ON, `N.z ~= +/-1` and the flip
        # is the difference between full brightness and the ambient floor.
        facing_v = np.array(
            [[-1.0, -1.0, 1.5], [1.0, -1.0, 1.5], [1.0, 1.0, 1.5], [-1.0, 1.0, 1.5]],
            dtype=np.float32,
        )
        facing_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

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
            # Direct-colour node: RGBA + labels + smooth stored normals.
            scene.add_mesh(
                "sphere",
                vertices,
                faces,
                normals=normals,
                normal_dims=[0, 1, 2],
                colors=colors,
                labels=labels,
                shading="smooth",
                layer=True,
            )
            # The colormap path needs its OWN node: `colors` and `colormap` are
            # mutually exclusive by design (the shader reads the colour attribute or
            # the LUT, never both), and `scalars` without a `colormap` is refused
            # outright — a scalar field with nothing to map it through would leave
            # USE_COLORMAP unset and the attribute unread.
            scene.add_mesh(
                "scalar_sphere",
                vertices + np.array([0.0, 2.5, 0.0], np.float32),
                faces,
                normals=normals,
                normal_dims=[0, 1, 2],
                scalars=scalars,
                colormap="viridis",
                shading="smooth",
                layer=True,
            )
            scene.add_mesh(
                "flat_patch",
                patch_v,
                patch_f,
                shading="flat",
                layer=True,
            )
            scene.add_mesh(
                "flat_facing",
                facing_v,
                facing_f,
                shading="flat",
                layer=True,
            )

        aprint(f"  Created {output}")
        aprint(f"  Sphere: {n_v} vertices, {len(faces)} faces (welded, closed)")
        aprint(
            f"  Cutout cap: {int((colors[:, 3] < 0.5).sum())} vertices below alpha 0.5"
        )


def generate_mesh_reveal_ladder_test() -> None:
    """A mesh reveal ladder beside an UNLADDERED copy of the same surface.

    The control node is the whole point. Every defect the ladder shipped with was
    a difference between the laddered node and what an ordinary mesh does with the
    same data — buffers sized from the committed prefix instead of the node total
    (#1521), a colour tail never written after level 0 (#1522) — and none of them
    is visible without something correct to compare against in the same scene,
    under the same camera, at the same moment. Two nodes, identical geometry,
    offset in x.

    Why these attrs specifically:

    * **per-vertex RGBA, fully opaque.** Alpha is the whole coverage term for a
      mesh, so an unwritten colour slot reads `(0, 0, 0, 0)` — invisible. Holding
      alpha at 1.0 everywhere makes "any transparency at all" a failure signal
      rather than something to disentangle from an authored cutout (which is what
      `test_mesh` deliberately has, and why this cannot reuse it).
    * **a hue ramp along +x**, so the LAST vertices of the ladder — the ones only
      the final level reveals — carry a colour distinguishable from both black and
      from the first level's.
    * **stored normals**, because the parent group of a ladder has to carry
      `has_normals` / `normal_dims` for the geometry to bind a `normal` attribute
      at all; a ladder whose parent omits them renders faceted and forced
      double-sided beside a smooth control, which is exactly how that bug was
      found.

    Four levels: enough that a prefix is a genuinely partial surface and that the
    concat runs three times, few enough to stay a small fixture.
    """
    with asection("Generating Mesh Reveal Ladder Test"):
        output = FIXTURES_DIR / "test_mesh_reveal_ladder.luxar.zarr"

        vertices, faces, normals = _icosphere(subdivisions=3, radius=1.0)
        n_v = len(vertices)

        colors = np.empty((n_v, 4), dtype=np.float32)
        colors[:, 0] = (vertices[:, 0] + 1.0) / 2.0
        colors[:, 1] = (vertices[:, 1] + 1.0) / 2.0
        colors[:, 2] = (vertices[:, 2] + 1.0) / 2.0
        # Opaque everywhere — see the docstring.
        colors[:, 3] = 1.0

        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
            ]
        )

        laddered_v = vertices.copy()
        laddered_v[:, 0] -= 1.3
        plain_v = vertices.copy()
        plain_v[:, 0] += 1.3

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.MEMORY,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_mesh(
                "laddered",
                laddered_v,
                faces,
                normals=normals,
                normal_dims=[0, 1, 2],
                colors=colors,
                additive_lod={"n_lods": 4},
                shading="smooth",
                layer=True,
            )
            scene.add_mesh(
                "plain",
                plain_v,
                faces,
                normals=normals,
                normal_dims=[0, 1, 2],
                colors=colors,
                shading="smooth",
                layer=True,
            )

        aprint(f"  Created {output}")
        aprint(f"  Source: {n_v} vertices, {len(faces)} faces; ladder = 4 levels")


def generate_mesh_nd_test() -> None:
    """4D mesh fixture: two spheres separated along a hidden categorical dimension.

    The mesh counterpart of `test_lines_categorical`, and it exercises the two things
    that only an nD mesh can:

    * the §5.4 **whole-triangle slab cull** — scrubbing `sel` must SWAP the two spheres
      (A xor B), never accumulate both. Mesh has no interpolation and no per-element
      extent, so its tolerance arm is its own (§5.2.1) and cannot be assumed from the
      Lines behaviour;
    * `drawRange` narrowing — the vertex arrays stay whole while the index buffer is
      rewritten, which is exactly why the debug surface reports triangles from the
      draw range rather than from `index.count`.

    `normal_dims` is `[0, 1, 2]` while the scene has 4 dimensions, so the fixture also
    covers the §3.4 rule that stored normals are used only when `normal_dims` equals
    the DISPLAYED axes — displaying `(x, y, sel)` instead must fall back to derivatives.
    """
    with asection("Generating Mesh nD (categorical scrub) Test"):
        output = FIXTURES_DIR / "test_mesh_nd.luxar.zarr"

        vertices, faces, normals = _icosphere(subdivisions=1, radius=0.8)

        def at(cx: float, sel: float) -> np.ndarray:
            out = np.zeros((len(vertices), 4), dtype=np.float32)
            out[:, 0] = vertices[:, 0] + cx
            out[:, 1] = vertices[:, 1]
            out[:, 2] = vertices[:, 2]
            out[:, 3] = sel
            return out

        # Normals are 3-vectors over dims (0, 1, 2) even though the mesh is 4D — the
        # `sel` axis has no orientation to describe.
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
                Dimension("sel", display=False, categories=["A", "B"]),
            ]
        )

        with LuxarZarrCompiler(
            output,
            encoding_mode=EncodingMode.PRECISION,
            compressor=None,
            float16_allowed=False,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_mesh(
                "sphere_a",
                at(-1.2, 0.0),
                faces,
                normals=normals,
                normal_dims=[0, 1, 2],
                colors=np.tile(
                    np.array([[1.0, 0.35, 0.2]], np.float32), (len(vertices), 1)
                ),
                layer=True,
            )
            scene.add_mesh(
                "sphere_b",
                at(1.2, 1.0),
                faces,
                normals=normals,
                normal_dims=[0, 1, 2],
                colors=np.tile(
                    np.array([[0.2, 0.6, 1.0]], np.float32), (len(vertices), 1)
                ),
                layer=True,
            )

        aprint(f"  Created {output}")
        aprint(f"  Two spheres at sel=A / sel=B, {len(vertices)} vertices each")


def generate_labelled_points_test() -> None:
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
        output = FIXTURES_DIR / "test_labelled_points.luxar.zarr"

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


# The element-interaction fixture asserts on these exact strings, so they are
# named constants rather than inline literals (issue #1917).
LINKED_LABEL = "Linked point"
LINKED_URL_PREFIX = "https://example.org/entry/"


def generate_linked_points_test() -> None:
    """Points carrying `link` / `copy` templates, for the click-actions E2E spec.

    One clickable point at the world origin — `element-actions.spec.ts` clicks
    the canvas centre and asserts the popup URL, so a single centred target
    removes any ambiguity about what was hit — plus four far-off corner points
    that exist only to give the scene a non-degenerate bounding box. Without
    them the auto-frame bails ("zero extent") and the near/far planes collapse,
    which is noisy at best and flaky at worst.

    Only the centre point is labelled; the corners carry empty labels, which is
    also free coverage for the rule that an empty substitution SUPPRESSES the
    link rather than opening `https://example.org/entry/`.

    The label contains a space so the spec can prove that a substituted value
    is percent-encoded into the URL rather than interpolated raw — the property
    that stops a label from restructuring the link.
    """
    with asection("Generating Linked Points Test (E2E element actions)"):
        output = FIXTURES_DIR / "test_linked_points.luxar.zarr"

        # Centre point first (the click target), then bounds-giving corners
        # far enough out that the cursor cannot land on one by accident.
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [-20.0, -20.0, -20.0],
                [20.0, -20.0, 20.0],
                [-20.0, 20.0, 20.0],
                [20.0, 20.0, -20.0],
            ],
            dtype=np.float32,
        )
        n = positions.shape[0]
        colors = np.tile(np.array([[1.0, 0.5, 0.25]], dtype=np.float32), (n, 1))
        radii = np.concatenate(
            [np.array([2.0], dtype=np.float32), np.full(n - 1, 0.5, dtype=np.float32)]
        )

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
                "linked_points",
                positions=positions,
                colors=colors,
                radii=radii,
                labels=[LINKED_LABEL] + [""] * (n - 1),
                link=LINKED_URL_PREFIX + "{hover_label}",
                copy="id={hover_label}",
            )

        aprint(f"  Created {output}")
        aprint(f"  1 linked point at the origin + {n - 1} unlabelled corners")


def generate_image_overlay_test() -> None:
    """Scene with a Python-authored image overlay for zipped-store E2E parity.

    Keep the STORED archive above and DEFLATE archive below the 65,557-byte
    end-of-directory search window asserted by zipped-store-loading.spec.ts.
    """
    with asection("Generating Image Overlay Test"):
        output = FIXTURES_DIR / "test_image_overlay.luxar.zarr"
        rng = np.random.default_rng(1157)
        positions = rng.normal(0.0, 0.35, (1200, 3)).astype(np.float32)
        line_t = np.linspace(0.0, 4 * np.pi, 200)
        line_vertices = np.column_stack(
            [
                0.8 + 0.25 * np.cos(line_t),
                0.25 * np.sin(line_t),
                line_t / (4 * np.pi) - 0.5,
            ]
        ).astype(np.float32)
        splat_centers = np.column_stack(
            [
                np.linspace(-0.4, 0.4, 40),
                np.full(40, 0.7),
                np.zeros(40),
            ]
        ).astype(np.float32)
        splat_cholesky = np.zeros((40, 6), dtype=np.float32)
        splat_cholesky[:, 0] = 0.06
        splat_cholesky[:, 2] = 0.06
        splat_cholesky[:, 5] = 0.06
        colors = rng.random((len(positions), 3), dtype=np.float32)
        radii = np.full(len(positions), 0.02, dtype=np.float32)
        image_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c02"
            "0000000b4944415478da63fcff1f0002eb01f569769f7b0000000049454e44ae426082"
        )
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
            compressor=COMPRESSOR_DISABLED,
            float16_allowed=FLOAT16_ALLOWED,
        ) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "points",
                positions=positions,
                colors=colors,
                radii=radii,
            )
            scene.add_lines(
                "lines",
                vertices=line_vertices,
                widths=0.02,
                colors=np.tile([0.2, 0.9, 0.4], (len(line_vertices), 1)).astype(
                    np.float32
                ),
                line_type="polyline",
            )
            scene.add_gsplats(
                "gsplats",
                splat_centers,
                amplitudes=np.full(40, 1.0, dtype=np.float32),
                cholesky_factors=splat_cholesky,
                colors=np.tile([0.4, 0.6, 1.0], (40, 1)).astype(np.float32),
            )
            scene.add_image(
                image_bytes,
                (0.02, 0.02),
                name="archive-image",
                size=(0.05, 0.05),
            )

        aprint(f"  Created {output}")
        aprint(
            "  1200 points, 200 line vertices, 40 splats, and a Python-authored 1x1 PNG"
        )


# Label of the single hover target in test_labelled_partitioned_points. Kept as
# a named constant because hover-tooltip.spec.ts asserts this exact string.
MARKER_LABEL = "Origin marker"


def generate_labelled_partitioned_points_test() -> None:
    """Labelled POINTS behind a ``kind=partition`` wrapper, for hover-tooltip E2E.

    The partitioned sibling of ``generate_labelled_points_test``. That fixture
    is a flat leaf, so the pick-result handler's reported path and its lookup
    path are the same string and the #1415 regression is invisible to it. Here
    ``add_points(partition=...)`` splits the node into ``part_<i>`` children and
    slices ``labels`` per part, so the label CSR (``label_offsets`` /
    ``label_bytes``) lives on the LEAVES while the layer the viewer reports is
    the wrapper. Looking the label up on the wrapper — the pre-fix behaviour —
    finds no array at all and yields a silently empty tooltip.

    Layout is built for a *predictable* hover assertion:

    * One isolated marker point at the world origin, with a large radius, whose
      label is unique. Every other point sits at ``|x| >= 6``, so the marker is
      the only element anywhere near the centre of the framed scene.
    * Two well-separated clusters either side of it, which is what forces the
      median BSP to actually split (a single part would fall through to the
      plain-leaf write and defeat the point of the fixture).
    * All three dimensions displayed, so the pick's element id is the on-disk
      index and the label the loader returns is the one the marker was authored
      with (a hidden dimension would make the id a visible-buffer slot instead).

    The marker is appended LAST, so its global index (``2 * cluster_n``) differs
    from the part-local index the loader must use — a lookup that reached the
    right node with the wrong index space would still return the wrong label.
    """
    with asection("Generating Labelled Partitioned Points Test (E2E hover-tooltip)"):
        output = FIXTURES_DIR / "test_labelled_partitioned_points.luxar.zarr"

        cluster_n = 150
        rng = np.random.default_rng(1415)
        left = np.stack(
            [
                rng.uniform(-12.0, -6.0, cluster_n),
                rng.uniform(-4.0, 4.0, cluster_n),
                rng.uniform(-4.0, 4.0, cluster_n),
            ],
            axis=1,
        )
        right = np.stack(
            [
                rng.uniform(6.0, 12.0, cluster_n),
                rng.uniform(-4.0, 4.0, cluster_n),
                rng.uniform(-4.0, 4.0, cluster_n),
            ],
            axis=1,
        )
        marker = np.array([[0.0, 0.0, 0.0]])
        positions = np.concatenate([left, right, marker], axis=0).astype(np.float32)
        n = positions.shape[0]

        # The marker is the hover target: big enough to dominate the picking
        # system's 5x5 readback window, and a distinct colour so a failure is
        # legible in a screenshot.
        radii = np.full(n, 0.35, dtype=np.float32)
        radii[-1] = 1.2
        colors = np.tile(np.array([[0.25, 0.45, 1.0]], dtype=np.float32), (n, 1))
        colors[-1] = (1.0, 0.5, 0.25)

        labels = [f"Cluster point {i}" for i in range(n - 1)]
        labels.append(MARKER_LABEL)

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
                "labelled_parts",
                positions=positions,
                colors=colors,
                radii=radii,
                labels=labels,
                # Well below n, so the median BSP is guaranteed to produce
                # more than one part and take the wrapper path.
                partition={"max_elements": 100},
            )

        aprint(f"  Created {output}")
        aprint(f"  {n} labelled points behind a kind=partition wrapper")
        aprint(f"  hover target at the origin, labelled {MARKER_LABEL!r}")


def main() -> None:
    """Generate all test datasets."""
    aprint("=" * 70)
    aprint("GENERATING TYPESCRIPT-PYTHON COMPATIBILITY TEST DATASETS")
    aprint("=" * 70)
    aprint("")

    try:
        generate_broadcasting_test()
        aprint("")

        generate_lut_test()
        generate_lut_u16_test()
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
        generate_layer_4d_gsplats_test()
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

        generate_delta_filter_test()
        aprint("")

        generate_nd_transforms_test()
        aprint("")

        generate_lines_test()
        aprint("")

        generate_lines_categorical_test()
        aprint("")

        generate_line_joins_test()
        aprint("")

        generate_extend_to_all_test()
        aprint("")

        generate_points_blending_modes_test()
        generate_lift_parity_test()
        generate_lines_blending_modes_test()
        generate_blending_inherited_test()
        generate_points_normal_overlap_test()
        generate_points_normal_overlap_reversed_test()
        generate_points_volumetric_reversed_test()
        generate_lines_volumetric_reversed_test()
        aprint("")

        generate_gsplats_test()
        generate_gsplats_2d_test()
        generate_gsplats_normal_overlap_test()
        generate_gsplats_normal_overlap_reversed_test()
        generate_gsplats_volumetric_test()
        generate_gsplats_volumetric_reversed_test()
        generate_gsplats_rgba_occlusion_test()
        generate_gsplats_rgba_hdr_test()
        generate_gsplats_rgba_uint8_test()
        generate_gsplats_rgba_lut_test()
        aprint("")

        generate_standalone_gsplats_test()
        aprint("")

        generate_lod_group_test()
        aprint("")

        generate_lod_group_additive_finest_test()
        aprint("")

        generate_lod_group_volumetric_test()
        aprint("")

        generate_overview_test()
        aprint("")

        generate_partition_layer_test()
        aprint("")

        generate_labelled_points_test()
        aprint("")

        generate_linked_points_test()
        aprint("")

        generate_image_overlay_test()
        aprint("")

        generate_labelled_partitioned_points_test()
        aprint("")

        generate_mesh_test()
        aprint("")

        generate_mesh_nd_test()
        generate_mesh_reveal_ladder_test()
        aprint("")

        aprint("=" * 70)
        aprint("✓ ALL TEST DATASETS GENERATED")
        aprint("=" * 70)
        aprint("")
        aprint("Generated datasets:")
        aprint(f"  {FIXTURES_DIR}/test_broadcasting.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_lut.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_quantization.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_array_refs.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_array_ref_broadcasting.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_encoding_edge_cases.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_encoding_contract_matrix.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_mixed.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_4d.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_hierarchical_transforms.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_hdr_colors.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_integer_colors.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_sharpness_range.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_log_scalar.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_4d_scalar_lut.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_uint16_quantization.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_nd_transforms.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_lines.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_gsplats.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_gsplats_2d.luxar.zarr")
        aprint(f"  {FIXTURES_DIR}/test_labelled_points.luxar.zarr")
        aprint("")
        aprint("Run TypeScript tests with:")
        aprint("  cd packages/luxar-viewer && pnpm test array-decoder")

        # Audit C1 self-check — confirm every name declared in
        # FIXTURE_NAMES at the top of the file was actually written.
        # A mismatch means either a generate_*() function was renamed/
        # removed without updating the manifest, OR the manifest grew
        # ahead of the generators. Either way the TS side would skip
        # tests silently — fail fast here instead.
        missing = [n for n in FIXTURE_NAMES if not (FIXTURES_DIR / n).exists()]
        if missing:
            raise RuntimeError(
                f"FIXTURE_NAMES declares {len(missing)} fixture(s) that "
                f"no generate_*() function produced: {missing}. "
                "Update FIXTURE_NAMES or add the missing generator."
            )

    except Exception as e:
        aprint(f"❌ Error generating test data: {e}")
        raise


if __name__ == "__main__":
    main()
