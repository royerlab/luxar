"""Tests for dynamic range-based dtype selection in encoding.

These tests verify that the encoder correctly:
1. Computes dynamic range as max_val / min_nonzero_val
2. Selects appropriate dtype based on dynamic range
3. Maintains acceptable quantization error bounds
4. Handles edge cases correctly
"""

import tempfile

import numpy as np
import zarr

from luxar.encoding.encoder import ArrayEncoder
from luxar.encoding.modes import EncodingMode
from luxar.encoding.semantic_types import SemanticType


class TestDynamicRangeComputation:
    """Test _compute_quantization_bits helper function."""

    def test_narrow_dynamic_range_selects_uint8(self):
        """Dynamic range <= 256 should select 8 bits (uint8)."""
        encoder = ArrayEncoder()

        # Dynamic range = 100 / 1 = 100 (< 256)
        data = np.array([1.0, 50.0, 100.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 8, f"Expected 8 bits for range 100:1, got {bits}"

        # Dynamic range = 255 / 1 = 255 (still < 256)
        data = np.array([1.0, 128.0, 255.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 8, f"Expected 8 bits for range 255:1, got {bits}"

    def test_medium_dynamic_range_selects_uint16(self):
        """Dynamic range 256 < x <= 65536 should select 16 bits (uint16)."""
        encoder = ArrayEncoder()

        # Dynamic range = 1000 / 1 = 1000 (> 256, < 65536)
        data = np.array([1.0, 500.0, 1000.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 16, f"Expected 16 bits for range 1000:1, got {bits}"

        # Dynamic range = 65535 / 1 = 65535 (still <= 65536)
        data = np.array([1.0, 32768.0, 65535.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 16, f"Expected 16 bits for range 65535:1, got {bits}"

    def test_wide_dynamic_range_selects_float(self):
        """Dynamic range > 65536 should select 0 bits (float)."""
        encoder = ArrayEncoder()

        # Dynamic range = 100000 / 1 = 100000 (> 65536)
        data = np.array([1.0, 50000.0, 100000.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 0, f"Expected 0 bits (float) for range 100000:1, got {bits}"

    def test_boundary_at_256(self):
        """Test boundary condition at dynamic range = 256."""
        encoder = ArrayEncoder()

        # Exactly 256:1 should still use uint8
        data = np.array([1.0, 256.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 8, f"Expected 8 bits for range exactly 256:1, got {bits}"

        # Just over 256:1 should use uint16
        data = np.array([1.0, 257.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 16, f"Expected 16 bits for range 257:1, got {bits}"

    def test_boundary_at_65536(self):
        """Test boundary condition at dynamic range = 65536."""
        encoder = ArrayEncoder()

        # Exactly 65536:1 should still use uint16
        data = np.array([1.0, 65536.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 16, f"Expected 16 bits for range exactly 65536:1, got {bits}"

        # Just over 65536:1 should use float
        data = np.array([1.0, 65537.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 0, f"Expected 0 bits (float) for range 65537:1, got {bits}"

    def test_all_zeros_returns_uint8(self):
        """All zeros should return 8 bits (uint8 is sufficient)."""
        encoder = ArrayEncoder()
        data = np.zeros(100, dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 8, f"Expected 8 bits for all zeros, got {bits}"

    def test_single_nonzero_value_returns_uint8(self):
        """Single non-zero value has dynamic range 1:1, should use uint8."""
        encoder = ArrayEncoder()
        data = np.array([0.0, 0.0, 5.0, 5.0, 5.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 8, f"Expected 8 bits for single value, got {bits}"

    def test_small_values_with_wide_range(self):
        """Small absolute values can have wide dynamic range."""
        encoder = ArrayEncoder()

        # Values in [0.00001, 0.1] have dynamic range 10000:1
        data = np.array([0.00001, 0.001, 0.01, 0.1], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 16, f"Expected 16 bits for range 10000:1, got {bits}"

    def test_large_values_with_narrow_range(self):
        """Large absolute values can have narrow dynamic range."""
        encoder = ArrayEncoder()

        # Values in [1000000, 2000000] have dynamic range 2:1
        data = np.array([1000000.0, 1500000.0, 2000000.0], dtype=np.float32)
        bits = encoder._compute_quantization_bits(data)
        assert bits == 8, f"Expected 8 bits for range 2:1, got {bits}"


class TestPositiveScalarDynamicRange:
    """Test POSITIVE_SCALAR encoding with dynamic range selection."""

    def test_uint8_for_narrow_range(self):
        """POSITIVE_SCALAR with narrow dynamic range uses uint8."""
        # Data with dynamic range ~10:1
        data = np.linspace(0.1, 1.0, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint8"

    def test_uint16_for_medium_range(self):
        """POSITIVE_SCALAR with medium dynamic range uses uint16."""
        # Data with dynamic range ~1000:1
        data = np.linspace(0.001, 1.0, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint16, f"Expected uint16, got {arr.dtype}"
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint16"

    def test_geolog_for_wide_range(self):
        """POSITIVE_SCALAR with very wide dynamic range uses geometric-log
        uint16 (rescale-first, min/max-anchored) instead of float32."""
        # Data with dynamic range ~1000000:1
        data = np.array([0.000001, 0.001, 1.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint16, f"Expected uint16, got {arr.dtype}"
            enc = arr.attrs["encoding"]
            assert enc["name"] == "geolog_scalar_uint16"
            assert "min_log" in enc and "max_log" in enc

    def test_max_value_irrelevant_to_dtype_selection(self):
        """Max value alone should not determine dtype - only dynamic range matters."""
        # Both datasets have same dynamic range (~10:1) but different max values
        data_small_max = np.linspace(0.01, 0.1, 100).astype(np.float32)  # max=0.1
        data_large_max = np.linspace(100.0, 1000.0, 100).astype(np.float32)  # max=1000

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()

            encoder.encode(data_small_max, group, "small", SemanticType.POSITIVE_SCALAR)
            encoder.encode(data_large_max, group, "large", SemanticType.POSITIVE_SCALAR)

            # Both should use uint8 because dynamic range is the same
            assert group["small"].dtype == np.uint8
            assert group["large"].dtype == np.uint8


class TestBoundedScalarDynamicRange:
    """Test BOUNDED_SCALAR encoding with dynamic range selection."""

    def test_uint8_for_narrow_range(self):
        """BOUNDED_SCALAR with narrow dynamic range uses uint8."""
        # Data with dynamic range ~5:1
        data = np.linspace(10.0, 50.0, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.BOUNDED_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint8"

    def test_uint16_for_medium_range(self):
        """BOUNDED_SCALAR with medium dynamic range uses uint16."""
        # Data with dynamic range ~10000:1
        data = np.linspace(0.01, 100.0, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.BOUNDED_SCALAR)

            arr = group["test"]
            assert arr.dtype == np.uint16, f"Expected uint16, got {arr.dtype}"
            enc = arr.attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint16"


class TestQuantizationErrorBounds:
    """Test that quantization error stays within acceptable bounds.

    Quantization error is ABSOLUTE relative to the range, not RELATIVE to each value.
    For uint8: max absolute error = range / 255 / 2 (with rounding)
    For uint16: max absolute error = range / 65535 / 2 (with rounding)

    The relative error for small values (near min) will be larger than for large
    values (near max). This is expected behavior for linear quantization.
    """

    def test_uint8_absolute_error(self):
        """uint8 absolute error should be <= range / 255 / 2 (with rounding)."""
        data = np.linspace(1.0, 10.0, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Decode manually
            arr = np.array(group["test"])
            enc = group["test"].attrs["encoding"]
            max_val = enc["max"]
            decoded = arr / 255.0 * max_val

            # Max absolute error should be <= range / 255 / 2 (with rounding)
            # For [0, 10], max error = 10 / 255 / 2 ≈ 0.0196
            abs_error = np.abs(decoded - data)
            max_abs_error = np.max(abs_error)
            expected_max_error = max_val / 255.0 / 2
            assert max_abs_error <= expected_max_error + 1e-6, (
                f"Max absolute error {max_abs_error:.6f} > expected {expected_max_error:.6f}"
            )

    def test_uint16_absolute_error(self):
        """uint16 absolute error should be <= range / 65535 / 2 (with rounding)."""
        # Data with wide range requiring uint16
        data = np.linspace(0.001, 1.0, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Decode manually
            arr = np.array(group["test"])
            enc = group["test"].attrs["encoding"]
            max_val = enc["max"]
            decoded = arr / 65535.0 * max_val

            # Max absolute error should be <= range / 65535 / 2 (with rounding)
            # Allow 1e-7 tolerance for float32 precision
            abs_error = np.abs(decoded - data)
            max_abs_error = np.max(abs_error)
            expected_max_error = max_val / 65535.0 / 2
            assert max_abs_error <= expected_max_error + 1e-7, (
                f"Max absolute error {max_abs_error:.8f} > expected {expected_max_error:.8f}"
            )

    def test_no_data_loss_with_correct_dtype(self):
        """No values should be quantized to zero if dtype is correctly selected."""
        # Wide dynamic range data - should use uint16
        data = np.linspace(0.0001, 0.1, 8000).astype(np.float32)
        dynamic_range = data.max() / data.min()
        assert dynamic_range > 256, (
            f"Test data should have wide range, got {dynamic_range}"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            arr = np.array(group["test"])

            # With uint16 and proper selection, NO values should be zero
            # (all original values are non-zero)
            zero_count = np.sum(arr == 0)
            assert zero_count == 0, (
                f"Expected 0 zeros, got {zero_count} ({100 * zero_count / len(arr):.1f}%)"
            )


class TestGSplatAmplitudeScenario:
    """Test the specific gsplat amplitude scenario that motivated this fix."""

    def test_gsplat_amplitude_range_preserved(self):
        """Simulate gsplat amplitudes: small values with ~6000:1 dynamic range."""
        # Simulate real gsplat amplitudes after 0.1x scaling
        np.random.seed(42)
        amplitudes = np.random.exponential(0.005, 8000).astype(np.float32)
        amplitudes = np.clip(amplitudes, 0.000008, 0.05)  # Similar to real data

        dynamic_range = amplitudes.max() / amplitudes[amplitudes > 0].min()
        assert dynamic_range > 1000, (
            f"Test should have wide range, got {dynamic_range:.0f}"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(
                amplitudes, group, "amplitudes", SemanticType.POSITIVE_SCALAR
            )

            # Should use uint16 for this dynamic range
            assert group["amplitudes"].dtype == np.uint16

            # Verify NO data loss
            arr = np.array(group["amplitudes"])
            zero_count = np.sum(arr == 0)
            assert zero_count == 0, (
                f"GSplat amplitudes: {zero_count} zeros ({100 * zero_count / len(arr):.1f}%) - "
                f"this would cause invisible splats!"
            )


class TestEdgeCases:
    """Test edge cases in dynamic range encoding."""

    def test_all_same_nonzero_value(self):
        """All identical non-zero values should use uint8."""
        data = np.full(1000, 0.5, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Should be broadcasted, not quantized
            enc = group["test"].attrs["encoding"]
            assert enc["name"] == "broadcasted"

    def test_two_values_narrow_range(self):
        """Two values with narrow range should use uint8."""
        data = np.array([1.0, 2.0] * 500, dtype=np.float32)  # range 2:1

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)

            # Should use LUT encoding (only 2 unique values)
            enc = group["test"].attrs["encoding"]
            assert enc["name"] == "lut_uint8"

    def test_precision_mode_ignores_dynamic_range(self):
        """PRECISION mode should always use float32 regardless of dynamic range."""
        data = np.linspace(1.0, 10.0, 1000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(
                data,
                group,
                "test",
                SemanticType.POSITIVE_SCALAR,
                mode=EncodingMode.PRECISION,
            )

            assert group["test"].dtype == np.float32
            enc = group["test"].attrs["encoding"]
            assert enc["name"] == "float32"
