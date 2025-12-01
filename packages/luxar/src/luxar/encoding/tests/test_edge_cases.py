"""Edge case tests for encoding system.

Tests cover error paths, boundary conditions, and unusual inputs.
"""

import tempfile

import numpy as np
import pytest
import zarr

from luxar.encoding import ArrayEncoder, EncodingMode, SemanticType


class TestErrorPaths:
    """Test error handling paths."""

    def test_bounded_scalar_out_of_bounds(self):
        """Test error when data exceeds specified bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.array([0.0, 5.0, 10.0], dtype=np.float32)

            # Specify bounds that data exceeds
            with pytest.raises(ValueError, match="outside specified bounds"):
                encoder.encode(
                    data,
                    group,
                    "test",
                    SemanticType.BOUNDED_SCALAR,
                    bounds=(0.0, 8.0),  # Max is 10, exceeds bound of 8
                )

    def test_custom_encoder_requires_bounds(self):
        """Test custom bounded_scalar encoders require bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            with pytest.raises(ValueError, match="requires bounds"):
                encoder.encode(
                    data,
                    group,
                    "test",
                    SemanticType.BOUNDED_SCALAR,
                    mode=EncodingMode.CUSTOM,
                    custom_encoder="bounded_scalar_uint8",
                    # bounds=None - missing!
                )

    def test_unknown_custom_encoder(self):
        """Test error for unknown custom encoder name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            with pytest.raises(ValueError, match="Unknown custom encoder"):
                encoder.encode(
                    data,
                    group,
                    "test",
                    SemanticType.POSITIVE_SCALAR,
                    mode=EncodingMode.CUSTOM,
                    custom_encoder="nonexistent_encoder",
                )

    def test_negative_index_rejected(self):
        """Test negative indices are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.array([0, 5, -1, 10], dtype=np.int32)

            with pytest.raises(ValueError, match="non-negative"):
                encoder.encode(data, group, "test", SemanticType.INDEX)

    def test_non_integer_index_rejected(self):
        """Test non-integer index dtype is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            with pytest.raises(ValueError, match="integer dtype"):
                encoder.encode(data, group, "test", SemanticType.INDEX)


class TestModeEdgeCases:
    """Test edge cases in different modes."""

    def test_coordinate_invalid_mode(self):
        """Test invalid mode for coordinate raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.random.randn(100, 3).astype(np.float32)

            # CUSTOM mode without custom_encoder
            with pytest.raises(ValueError, match="custom_encoder"):
                encoder.encode(
                    data,
                    group,
                    "test",
                    SemanticType.COORDINATE,
                    mode=EncodingMode.CUSTOM,
                )

    def test_color_integer_input(self):
        """Test integer color input doesn't require color_mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            # Integer colors (already quantized)
            data = np.random.randint(0, 256, (100, 3), dtype=np.uint8)

            # Should work without color_mode
            encoder.encode(data, group, "test", SemanticType.COLOR)

            arr = group["test"]
            assert arr.dtype == np.uint8

    def test_hdr_color_invalid_mode(self):
        """Test HDR colors with wrong mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.random.rand(100, 3).astype(np.float32) * 2.0

            # This should work - CUSTOM mode with HDR
            encoder.encode(
                data,
                group,
                "test",
                SemanticType.COLOR,
                mode=EncodingMode.CUSTOM,
                custom_encoder="float32",
                color_mode="hdr",
            )

            assert group["test"].dtype == np.float32

    def test_sdr_color_out_of_range_rejected(self):
        """Test SDR color_mode rejects values > 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            # SDR colors with values > 1.0
            data = np.array([[1.5, 0.5, 0.0]], dtype=np.float32)

            with pytest.raises(ValueError, match="SDR.*requires values in"):
                encoder.encode(
                    data, group, "test", SemanticType.COLOR, color_mode="sdr"
                )


class TestCustomEncoders:
    """Test all custom encoder types."""

    def test_custom_log_scalar_uint16(self):
        """Test custom log_scalar_uint16 encoder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.random.rand(100).astype(np.float32) * 1000

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.POSITIVE_SCALAR,
                mode=EncodingMode.CUSTOM,
                custom_encoder="log_scalar_uint16",
            )

            arr = group["test"]
            assert arr.dtype == np.uint16
            enc = arr.attrs["encoding"]
            assert enc["name"] == "log_scalar_uint16"
            assert enc["bits"] == 16

    def test_custom_bounded_scalar_uint16(self):
        """Test custom bounded_scalar_uint16 encoder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.random.rand(100).astype(np.float32) * 100

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.BOUNDED_SCALAR,
                mode=EncodingMode.CUSTOM,
                custom_encoder="bounded_scalar_uint16",
                bounds=(0.0, 100.0),
            )

            arr = group["test"]
            assert arr.dtype == np.uint16
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint16"
            assert enc["bits"] == 16

    def test_custom_rgb_uint16(self):
        """Test custom rgb_uint16 encoder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            data = np.random.rand(100, 3).astype(np.float32)

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.COLOR,
                mode=EncodingMode.CUSTOM,
                custom_encoder="rgb_uint16",
                color_mode="sdr",
            )

            arr = group["test"]
            assert arr.dtype == np.uint16
            enc = arr.attrs["encoding"]
            assert enc["name"] == "rgb_uint16"

    def test_custom_uint_direct(self):
        """Test custom uint8/uint16/uint32/uint64 encoders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            # Test uint16
            data = np.array([0, 100, 60000], dtype=np.uint32)

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.INDEX,
                mode=EncodingMode.CUSTOM,
                custom_encoder="uint16",
            )

            assert group["test"].dtype == np.uint16


class TestBoundaryConditions:
    """Test boundary conditions and degenerate cases."""

    def test_bounded_scalar_degenerate_range(self):
        """Test bounded scalar with zero range (all values equal)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            # All values equal
            data = np.full(100, 5.0, dtype=np.float32)

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.BOUNDED_SCALAR,
                bounds=(5.0, 5.0),  # Zero range
            )

            # Should be broadcasted instead of bounded_scalar
            enc = group["test"].attrs["encoding"]
            assert enc["name"] == "broadcasted"

    def test_index_large_values(self):
        """Test INDEX with large values selects appropriate dtype."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            # Test uint32 range
            data = np.array([0, 100000, 4294967295], dtype=np.uint64)

            encoder.encode(data, group, "test", SemanticType.INDEX)

            assert group["test"].dtype == np.uint32

    def test_cholesky_memory_mode(self):
        """Test CHOLESKY in MEMORY mode uses float32 by default (TypeScript compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()  # float16_allowed=False by default

            # Cholesky factors for 2D (3 elements: L00, L10, L11)
            data = np.random.rand(100, 3).astype(np.float32)

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.CHOLESKY,
                mode=EncodingMode.MEMORY,
            )

            assert group["test"].dtype == np.float32  # Default is float32

    def test_cholesky_memory_mode_float16_enabled(self):
        """Test CHOLESKY in MEMORY mode uses float16 when explicitly enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder(float16_allowed=True)  # Explicitly enable float16

            # Cholesky factors for 2D (3 elements: L00, L10, L11)
            data = np.random.rand(100, 3).astype(np.float32)

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.CHOLESKY,
                mode=EncodingMode.MEMORY,
            )

            assert group["test"].dtype == np.float16

    def test_unit_vector_memory_mode(self):
        """Test UNIT_VECTOR in MEMORY mode uses float32 by default (TypeScript compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()  # float16_allowed=False by default

            # Unit vectors
            data = np.random.randn(100, 3).astype(np.float32)
            # Normalize to unit length
            data = data / np.linalg.norm(data, axis=1, keepdims=True)

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.UNIT_VECTOR,
                mode=EncodingMode.MEMORY,
            )

            assert group["test"].dtype == np.float32  # Default is float32

    def test_unit_vector_memory_mode_float16_enabled(self):
        """Test UNIT_VECTOR in MEMORY mode uses float16 when explicitly enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder(float16_allowed=True)  # Explicitly enable float16

            # Unit vectors
            data = np.random.randn(100, 3).astype(np.float32)
            # Normalize to unit length
            data = data / np.linalg.norm(data, axis=1, keepdims=True)

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.UNIT_VECTOR,
                mode=EncodingMode.MEMORY,
            )

            assert group["test"].dtype == np.float16
