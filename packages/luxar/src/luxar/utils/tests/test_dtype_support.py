"""Tests for encoding mode support with compiler.

Old DataTypeConfig/datatypes.py tests removed - those functions no longer exist.
The new encoding system uses SemanticType and ArrayEncoder instead.
"""

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode


class TestCompilerWithDtypes:
    """Test LuxarZarrCompiler with different encoding modes."""

    def test_compiler_with_memory_config(self) -> None:
        """Test compiler with memory-efficient encoding mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            # Fixed seed: the encoder's uint8 vs uint16 decision depends on
            # max/min_nonzero ratio (threshold=256). Without a seed, random
            # minimum values can push the ratio above 256, selecting uint16.
            rng = np.random.RandomState(42)
            positions = rng.randn(100, 3).astype(np.float32)
            colors = rng.rand(100, 3).astype(np.float32)
            radii = rng.rand(100).astype(np.float32) * 0.5

            # Write with MEMORY encoding mode
            with LuxarZarrCompiler(
                zarr_path, encoding_mode=EncodingMode.MEMORY
            ) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("test", positions, colors=colors, radii=radii)

            # Check encoding metadata (check encoding, not dtype attributes)
            store = zarr.open_group(zarr_path, mode="r")
            points = store["test"]

            # Check positions encoding: COORDINATE is uint16 per-axis fixed-point
            # in MEMORY (and AUTO) — never float16, never uint8.
            pos_enc = points["positions"].attrs.get("encoding", {})
            assert pos_enc["name"] == "linear_perchannel_u16", (
                "MEMORY mode uses uint16 per-axis fixed-point for positions"
            )

            # Check colors encoding (should be uint8, may be LUT/broadcasted/quantized)
            assert points["colors"].dtype == np.uint8, (
                "Colors should be uint8 in MEMORY mode"
            )

            # Check radii encoding (small range [0, 0.5] should be quantized to uint8)
            assert points["radii"].dtype == np.uint8, (
                "Radii with small range should be uint8"
            )

    def test_compiler_with_precision_mode(self) -> None:
        """Test compiler with PRECISION mode (float32 everywhere)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            # Create test data
            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32)  # SDR colors
            radii = np.random.rand(100).astype(np.float32) * 0.5  # Small range
            sharpness = np.random.rand(100).astype(np.float32)  # [0, 1] range

            # Write with PRECISION mode
            with LuxarZarrCompiler(
                zarr_path, encoding_mode=EncodingMode.PRECISION
            ) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points(
                    "test", positions, colors=colors, radii=radii, sharpness=sharpness
                )

            # Check encoding results (PRECISION mode)
            store = zarr.open_group(zarr_path, mode="r")
            points = store["test"]

            # Check encodings match PRECISION mode behavior (float32 everywhere)
            assert points["positions"].dtype == np.float32, (
                "PRECISION uses float32 for positions"
            )
            assert points["colors"].dtype == np.float32, (
                "PRECISION preserves float32 for colors"
            )
            assert points["radii"].dtype == np.float32, (
                "PRECISION uses float32 for radii"
            )
            assert points["sharpnesses"].dtype == np.float32, (
                "PRECISION uses float32 for sharpness"
            )

    def test_hdr_color_detection(self) -> None:
        """HDR colors are auto-detected and quantized to geolog_perchannel_u16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            # Create HDR colors
            positions = np.random.randn(100, 3).astype(np.float32)
            hdr_colors = np.random.rand(100, 3).astype(np.float32) * 2.0  # HDR values

            # Write with AUTO encoding mode (default)
            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("test", positions, colors=hdr_colors)

            # Check that HDR was detected (check encoding metadata)
            store = zarr.open_group(zarr_path, mode="r")
            points = store["test"]
            colors_arr = points["colors"]

            # HDR detection routes to the per-channel true-log encoding
            # (2026-07 policy: AUTO -> geolog_perchannel_u16, decode -> f32).
            assert colors_arr.dtype == np.uint16
            assert colors_arr.attrs["encoding"]["name"] == "geolog_perchannel_u16"

            # Check encoding metadata
            enc = colors_arr.attrs.get("encoding", {})
            assert enc.get("original_dtype") == "float32", "Original dtype preserved"
            assert enc["zero_level"] is True

    def test_backward_compatibility(self) -> None:
        """Test that default behavior (no encoding_mode specified) still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            # Create test data
            positions = np.random.randn(100, 3).astype(np.float32)

            # Write without encoding_mode (should use AUTO as default)
            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("test", positions)

            # Check that it worked
            store = zarr.open_group(zarr_path, mode="r")
            assert "test" in store
            assert "test/positions" in store
