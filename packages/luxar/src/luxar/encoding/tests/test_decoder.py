"""Tests for ArrayDecoder class.

Tests cover all decoding methods including broadcasting, LUT, array references,
and all quantized encodings.
"""

import tempfile

import numpy as np
import pytest
import zarr

from luxar.encoding import ArrayDecoder, ArrayEncoder, SemanticType


class TestBroadcastDecoding:
    """Test decoding of broadcasted arrays."""

    def test_decode_broadcasted_1d(self):
        """Test decoding 1D broadcasted array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Encode uniform data
            data = np.full(1000, 5.0, dtype=np.float32)
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Decode and verify
            decoded = decoder.decode(group["test"])
            assert decoded.shape == (1000,)
            assert np.all(decoded == 5.0)

    def test_decode_broadcasted_2d(self):
        """Test decoding 2D broadcasted array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Encode uniform colors
            data = np.full((500, 3), [1.0, 0.5, 0.0], dtype=np.float32)
            encoder.encode(data, group, "test", SemanticType.COLOR, color_mode="sdr")

            # Decode and verify
            decoded = decoder.decode(group["test"])
            assert decoded.shape == (500, 3)
            assert np.allclose(decoded[0], [1.0, 0.5, 0.0])
            assert np.all(decoded == decoded[0])


class TestLUTDecoding:
    """Test decoding of LUT-encoded arrays."""

    def test_decode_lut_1d(self):
        """Test decoding 1D LUT array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Create data with few unique values
            data = np.random.choice([0.0, 0.5, 1.0, 2.0], size=1000).astype(np.float32)
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Decode and verify
            decoded = decoder.decode(group["test"])
            assert decoded.shape == (1000,)
            assert decoded.dtype == np.float32
            assert set(decoded) == {0.0, 0.5, 1.0, 2.0}

    def test_decode_lut_color_row_mode(self):
        """Test decoding color LUT in row mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Create palette with few unique colors
            palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
            indices = np.random.choice([0, 1, 2], size=1000)
            data = palette[indices]

            encoder.encode(data, group, "test", SemanticType.COLOR)

            # Decode and verify
            decoded = decoder.decode(group["test"])
            assert decoded.shape == (1000, 3)
            assert decoded.dtype == np.uint8
            # Check all colors are from palette
            unique_colors = np.unique(decoded, axis=0)
            assert len(unique_colors) == 3


class TestQuantizedDecoding:
    """Test decoding of quantized encodings."""

    def test_decode_bounded_scalar_uint8(self):
        """Test decoding bounded scalar uint8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Create bounded data
            data = np.random.rand(100).astype(np.float32) * 31  # [0, 31] range

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.BOUNDED_SCALAR,
                bounds=(0.0, 31.0),
            )

            # Decode and verify
            decoded = decoder.decode(group["test"])
            assert decoded.shape == (100,)
            assert decoded.dtype == np.float32
            # Should be close to original (with quantization error)
            assert np.allclose(decoded, data, atol=0.2)  # ~1/256 of range

    def test_decode_log_scalar_uint8(self):
        """Test decoding log scalar uint8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Create data with wide range, starting at 1.0 to avoid near-zero values
            # which have higher relative error with log encoding
            np.random.seed(42)  # Reproducibility
            data = np.random.rand(100).astype(np.float32) * 99 + 1.0  # Range [1, 100]

            encoder.encode(
                data,
                group,
                "test",
                SemanticType.POSITIVE_SCALAR,
                positive_scalar_encoding="log",
            )

            # Decode and verify
            decoded = decoder.decode(group["test"])
            assert decoded.shape == (100,)
            assert decoded.dtype == np.float32
            # Log encoding preserves relative accuracy
            rel_error = np.abs(decoded - data) / (data + 1e-8)
            assert np.max(rel_error) < 0.05  # <5% relative error

    def test_decode_color_uint8(self):
        """Test decoding SDR color uint8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Create SDR colors
            data = np.random.rand(100, 3).astype(np.float32)

            encoder.encode(data, group, "test", SemanticType.COLOR, color_mode="sdr")

            # Decode and verify
            decoded = decoder.decode(group["test"])
            assert decoded.shape == (100, 3)
            assert decoded.dtype == np.float32
            # Should be close to original (with quantization error)
            assert np.allclose(decoded, data, atol=1 / 255)


class TestArrayRefDecoding:
    """Test decoding of array references."""

    def test_decode_array_ref(self):
        """Test decoding array reference (follows pointer)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Encode original array
            data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            encoder.encode(data1, group, "original", SemanticType.POSITIVE_SCALAR)

            # Encode duplicate (should create array_ref)
            data2 = data1.copy()
            encoder.encode(data2, group, "duplicate", SemanticType.POSITIVE_SCALAR)

            # Decode reference (should give same data as original)
            decoded = decoder.decode(group["duplicate"], group)
            assert decoded.shape == (4,)
            # Dtype may be different due to encoding (e.g., float16 in AUTO mode)
            # What matters is the values are correct
            # With rounding, uint8 quantization error is < 0.5% (0.5/255)
            assert np.allclose(decoded, data1, rtol=0.005)

    def test_decode_array_ref_recursive(self):
        """Test that array_ref decoding is recursive (target may also be encoded)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            decoder = ArrayDecoder()

            # Encode LUT-encoded original
            data1 = np.random.choice([1.0, 2.0, 3.0], size=1000).astype(np.float32)
            encoder.encode(data1, group, "original", SemanticType.POSITIVE_SCALAR)

            # Encode duplicate (array_ref to LUT-encoded data)
            data2 = data1.copy()
            encoder.encode(data2, group, "duplicate", SemanticType.POSITIVE_SCALAR)

            # Decode should work recursively
            decoded = decoder.decode(group["duplicate"], group)
            assert decoded.shape == (1000,)
            assert np.array_equal(decoded, data1)


class TestPassthroughDecoding:
    """Test decoding of passthrough (unencoded) arrays."""

    def test_decode_float32_passthrough(self):
        """Test decoding float32 array with no encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            # Write directly without encoding metadata
            data = np.random.rand(100, 3).astype(np.float32)
            group.create_dataset("test", data=data)

            # Decode should return data as-is
            decoded = decoder.decode(group["test"])
            assert np.array_equal(decoded, data)

    def test_decode_with_none_encoding(self):
        """Test decoding array with encoding name = 'none'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            # Write with explicit 'none' encoding
            data = np.random.rand(100).astype(np.float32)
            group.create_dataset("test", data=data)
            group["test"].attrs["encoding"] = {"name": "none"}

            # Decode should return data as-is
            decoded = decoder.decode(group["test"])
            assert np.array_equal(decoded, data)


class TestErrorHandling:
    """Test decoder error handling."""

    def test_array_ref_missing_target(self):
        """Test error when array_ref target not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            # Create array_ref with invalid target
            empty = np.array([], dtype=np.float32)
            group.create_dataset("test", data=empty)
            group["test"].attrs["encoding"] = {
                "name": "array_ref",
                "target": "nonexistent/path",
                "hash": "xxh64:abc123",
                "original_shape": [100],
                "original_dtype": "float32",
            }

            # Should raise error
            with pytest.raises(ValueError, match="not found"):
                decoder.decode(group["test"], group)

    def test_array_ref_requires_zarr_root(self):
        """Test error when array_ref decoded without zarr_root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            # Create array_ref
            empty = np.array([], dtype=np.float32)
            group.create_dataset("test", data=empty)
            group["test"].attrs["encoding"] = {
                "name": "array_ref",
                "target": "other/path",
                "hash": "xxh64:abc123",
                "original_shape": [100],
                "original_dtype": "float32",
            }

            # Should raise error when zarr_root is None
            with pytest.raises(ValueError, match="zarr_root required"):
                decoder.decode(group["test"], zarr_root=None)
