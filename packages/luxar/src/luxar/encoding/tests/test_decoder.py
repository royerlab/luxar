"""Tests for ArrayDecoder class.

Tests cover all decoding methods including broadcasting, LUT, array references,
and all quantized encodings.
"""

import tempfile

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import create_array
from luxar.encoding import ArrayDecoder, ArrayEncoder, EncodingMode, SemanticType
from luxar.encoding.decoder import decode_coordinate_columns


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

    def test_decode_lut_uint16_external(self):
        """Decoder must dispatch lut_uint16 even though the encoder only emits uint8.

        Some external producers (and the TypeScript decoder) recognise lut_uint16,
        so the Python decoder must keep parity with the cross-language contract.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            indices = np.array([0, 1, 2, 1, 0], dtype=np.uint16)
            create_array(group, "test", data=indices)
            group["test"].attrs["encoding"] = {
                "name": "lut_uint16",
                "lut": [10.0, 20.0, 30.0],
                "original_dtype": "float32",
            }

            decoded = decoder.decode(group["test"])
            assert decoded.dtype == np.float32
            assert decoded.tolist() == [10.0, 20.0, 30.0, 20.0, 10.0]

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

    @pytest.mark.parametrize(
        ("encoding", "stored", "expected"),
        [
            (
                {
                    "name": "bounded_scalar_uint8",
                    "min": 1.0,
                    "max": 3.0,
                    "bits": 8,
                    "original_dtype": "uint8",
                },
                np.array([100], dtype=np.uint8),
                np.array([2], dtype=np.uint8),
            ),
            (
                {
                    "name": "log_scalar_uint8",
                    "max_log": np.log(5.0),
                    "bits": 8,
                    "original_dtype": "uint8",
                },
                np.array([200], dtype=np.uint8),
                np.array([3], dtype=np.uint8),
            ),
            (
                {
                    "name": "geolog_scalar_uint8",
                    "min_log": 0.0,
                    "max_log": np.log(3.0),
                    "bits": 8,
                    "original_dtype": "uint8",
                },
                np.array([220], dtype=np.uint8),
                np.array([3], dtype=np.uint8),
            ),
            (
                {
                    "name": "log_perchannel_u8",
                    "col_lo": [np.log1p(1.0)],
                    "col_hi": [np.log1p(3.0)],
                    "bits": 8,
                    "original_dtype": "uint8",
                },
                np.array([[100]], dtype=np.uint8),
                np.array([[2]], dtype=np.uint8),
            ),
            (
                {
                    "name": "signed_log_perchannel_u8",
                    "col_lo": [np.log1p(1.0)],
                    "col_hi": [np.log1p(3.0)],
                    "bits": 8,
                    "original_dtype": "int8",
                },
                np.array([[100]], dtype=np.uint8),
                np.array([[2]], dtype=np.int8),
            ),
            (
                {
                    "name": "linear_perchannel_u8",
                    "col_lo": [1.0],
                    "col_hi": [3.0],
                    "bits": 8,
                    "original_dtype": "uint8",
                },
                np.array([[100]], dtype=np.uint8),
                np.array([[2]], dtype=np.uint8),
            ),
            (
                {
                    "name": "geolog_perchannel_u8",
                    "col_lo": [0.0],
                    "col_hi": [np.log(3.0)],
                    "bits": 8,
                    "zero_level": True,
                    "original_dtype": "uint8",
                },
                np.array([[220]], dtype=np.uint8),
                np.array([[3]], dtype=np.uint8),
            ),
            (
                {"name": "rgb_uint8", "original_dtype": "uint8"},
                np.array([[200]], dtype=np.uint8),
                np.array([[1]], dtype=np.uint8),
            ),
        ],
        ids=[
            "bounded-scalar",
            "log-scalar",
            "geolog-scalar",
            "log-perchannel",
            "signed-log-perchannel",
            "linear-perchannel",
            "geolog-perchannel",
            "color",
        ],
    )
    def test_quantized_integer_restoration_rounds_before_cast(
        self,
        tmp_path,
        encoding: dict,
        stored: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        group = zarr.open_group(tmp_path, mode="w")
        create_array(group, "test", data=stored)
        group["test"].attrs["encoding"] = encoding

        decoded = ArrayDecoder().decode(group["test"], group)

        np.testing.assert_array_equal(decoded, expected)

    def test_coordinate_column_integer_restoration_rounds_before_cast(
        self, tmp_path
    ) -> None:
        group = zarr.open_group(tmp_path, mode="w")
        create_array(group, "test", data=np.array([[100, 200]], dtype=np.uint8))
        group["test"].attrs["encoding"] = {
            "name": "linear_perchannel_u8",
            "col_lo": [1.0, 1.0],
            "col_hi": [3.0, 3.0],
            "bits": 8,
            "original_dtype": "uint8",
        }

        decoded = decode_coordinate_columns(group["test"], [0], group)

        np.testing.assert_array_equal(decoded, np.array([[2]], dtype=np.uint8))

    def test_integer_roundtrip_is_exact_below_half_unit_quantum(self, tmp_path) -> None:
        data = np.arange(1, 33, dtype=np.uint8)
        group = zarr.open_group(tmp_path, mode="w")
        ArrayEncoder().encode(
            data,
            group,
            "test",
            SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.AUTO,
            allow_lut=False,
            deduplicate=False,
        )

        encoding = group["test"].attrs["encoding"]
        quantum = (encoding["max"] - encoding["min"]) / (2 ** encoding["bits"] - 1)
        assert quantum < 0.5

        decoded = ArrayDecoder().decode(group["test"], group)

        np.testing.assert_array_equal(decoded, data)

    def test_coarse_integer_roundtrip_stays_within_quantization_bound(
        self, tmp_path
    ) -> None:
        data = np.arange(100, 25_501, dtype=np.uint16)
        group = zarr.open_group(tmp_path, mode="w")
        ArrayEncoder().encode(
            data,
            group,
            "test",
            SemanticType.POSITIVE_SCALAR,
            mode=EncodingMode.AUTO,
            allow_lut=False,
            deduplicate=False,
        )

        encoding = group["test"].attrs["encoding"]
        assert encoding["name"] == "bounded_scalar_uint8"
        quantum = (encoding["max"] - encoding["min"]) / (2 ** encoding["bits"] - 1)

        decoded = ArrayDecoder().decode(group["test"], group)
        error = decoded.astype(np.int64) - data.astype(np.int64)

        assert np.max(np.abs(error)) <= quantum / 2 + 0.5
        assert error.min() < 0 < error.max()
        assert abs(float(np.mean(error))) < 0.01

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
            create_array(group, "test", data=data)

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
            create_array(group, "test", data=data)
            group["test"].attrs["encoding"] = {"name": "none"}

            # Decode should return data as-is
            decoded = decoder.decode(group["test"])
            assert np.array_equal(decoded, data)


class TestErrorHandling:
    """Test decoder error handling."""

    def test_missing_encoding_name_raises(self):
        """Encoding metadata must explicitly name a known encoder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            create_array(group, "test", data=np.array([1, 2, 3], dtype=np.uint8))
            group["test"].attrs["encoding"] = {"bounds": [0, 1]}

            with pytest.raises(ValueError, match="encoding.name is required"):
                decoder.decode(group["test"])

    def test_unknown_encoding_name_raises(self):
        """Unknown encoder names must fail instead of falling through."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            create_array(group, "test", data=np.array([1, 2, 3], dtype=np.uint8))
            group["test"].attrs["encoding"] = {"name": "quantized_uint8"}

            with pytest.raises(ValueError, match="Unknown encoding name"):
                decoder.decode(group["test"])

    def test_target_without_array_ref_raises(self):
        """Only array_ref metadata may contain a target path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            create_array(group, "test", data=np.array([1, 2, 3], dtype=np.uint8))
            group["test"].attrs["encoding"] = {"name": "uint8", "target": "other"}

            with pytest.raises(ValueError, match="target is only valid for array_ref"):
                decoder.decode(group["test"])

    def test_array_ref_missing_target_metadata(self):
        """array_ref metadata must include a target path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            create_array(group, "test", data=np.array([], dtype=np.float32))
            group["test"].attrs["encoding"] = {
                "name": "array_ref",
                "hash": "xxh64:abc123",
                "original_shape": [100],
                "original_dtype": "float32",
            }

            with pytest.raises(ValueError, match="array_ref encoding requires"):
                decoder.decode(group["test"], group)

    def test_array_ref_missing_target(self):
        """Test error when array_ref target not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()

            # Create array_ref with invalid target
            empty = np.array([], dtype=np.float32)
            create_array(group, "test", data=empty)
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
            create_array(group, "test", data=empty)
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

    def test_bounded_scalar_rejects_non_finite_bounds(self):
        """Bounded scalar metadata must have finite min/max."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()
            create_array(group, "test", data=np.array([0, 128, 255], dtype=np.uint8))
            group["test"].attrs["encoding"] = {
                "name": "bounded_scalar_uint8",
                "min": float("nan"),
                "max": 1.0,
                "bits": 8,
                "original_dtype": "float32",
            }
            with pytest.raises(ValueError, match="finite min/max"):
                decoder.decode(group["test"])

    def test_bounded_scalar_rejects_max_le_min(self):
        """Bounded scalar metadata must have max > min."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()
            create_array(group, "test", data=np.array([0, 128, 255], dtype=np.uint8))
            group["test"].attrs["encoding"] = {
                "name": "bounded_scalar_uint8",
                "min": 5.0,
                "max": 5.0,
                "bits": 8,
                "original_dtype": "float32",
            }
            with pytest.raises(ValueError, match="max > min"):
                decoder.decode(group["test"])

    def test_bounded_scalar_rejects_zero_bits(self):
        """Bounded scalar metadata must have bits > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()
            create_array(group, "test", data=np.array([0, 128, 255], dtype=np.uint8))
            group["test"].attrs["encoding"] = {
                "name": "bounded_scalar_uint8",
                "min": 0.0,
                "max": 1.0,
                "bits": 0,
                "original_dtype": "float32",
            }
            with pytest.raises(ValueError, match="bits > 0"):
                decoder.decode(group["test"])

    def test_log_scalar_rejects_non_positive_max_log(self):
        """Log scalar metadata must have positive, finite max_log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()
            create_array(group, "test", data=np.array([0, 128, 255], dtype=np.uint8))
            group["test"].attrs["encoding"] = {
                "name": "log_scalar_uint8",
                "max_log": -1.0,
                "bits": 8,
                "original_dtype": "float32",
            }
            with pytest.raises(ValueError, match="finite, positive max_log"):
                decoder.decode(group["test"])

    def test_log_scalar_rejects_inf_max_log(self):
        """Log scalar metadata must have finite max_log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            decoder = ArrayDecoder()
            create_array(group, "test", data=np.array([0, 128, 255], dtype=np.uint8))
            group["test"].attrs["encoding"] = {
                "name": "log_scalar_uint8",
                "max_log": float("inf"),
                "bits": 8,
                "original_dtype": "float32",
            }
            with pytest.raises(ValueError, match="finite, positive max_log"):
                decoder.decode(group["test"])
