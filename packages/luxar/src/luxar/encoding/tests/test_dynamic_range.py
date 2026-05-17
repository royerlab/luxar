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


class TestPositiveScalarDefault:
    """POSITIVE_SCALAR always emits float16 in AUTO/MEMORY modes.

    Replaces the previous `TestPositiveScalarDynamicRange` suite that
    asserted uint8/uint16/float32 dtype selection by dynamic range.
    Phase-3 attribute packing fixes the linear-encoding dtype at
    float16 across all dynamic ranges; the decoder retains the old
    `bounded_scalar_uint8` / `bounded_scalar_uint16` branches for
    legacy zarr files, but the encoder no longer emits them.
    """

    def test_float16_for_narrow_range(self):
        data = np.linspace(0.1, 1.0, 1000).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)
            assert group["test"].dtype == np.float16
            assert group["test"].attrs["encoding"]["name"] == "float16"

    def test_float16_for_medium_range(self):
        data = np.linspace(0.001, 1.0, 1000).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)
            assert group["test"].dtype == np.float16

    def test_float16_for_wide_range(self):
        data = np.array([0.000001, 0.001, 1.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)
            # Float16's normal range is 6e-5 to 65504; the very-small
            # value rounds to a denormal but is still preserved with
            # enough precision for radii / amplitudes.
            assert group["test"].dtype == np.float16


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


class TestPositiveScalarFloat16ErrorBounds:
    """Float16 POSITIVE_SCALAR round-trip error bounds.

    Phase-3 replaced the uint8/uint16 quantization paths with a flat
    float16 encoding for POSITIVE_SCALAR. Float16 has 11 bits of
    mantissa precision and a non-uniform error profile (relative,
    not absolute), unlike the prior linear-quantization scheme:
        - For values around 1.0 the ULP is ~1e-3.
        - For values around 1e-6 the ULP is ~1e-9 (denormal range).
    The tests below pin the round-trip accuracy to confirm Luxar's
    scalar fields (radii, widths, amplitudes) survive narrowing.
    """

    def test_narrow_range_relative_error(self):
        data = np.linspace(1.0, 10.0, 1000).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)
            decoded = np.array(group["test"]).astype(np.float32)
            # Float16 ULP at this range is ~5e-4; allow 2x for ordering.
            np.testing.assert_allclose(decoded, data, rtol=1e-3)

    def test_medium_range_relative_error(self):
        data = np.linspace(0.001, 1.0, 1000).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.POSITIVE_SCALAR)
            decoded = np.array(group["test"]).astype(np.float32)
            np.testing.assert_allclose(decoded, data, rtol=1e-3)

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
    """Float16 preserves the gsplat amplitude scenario without invisible splats.

    The motivating real-world case: ~6000:1 dynamic range amplitudes
    that the prior uint8 path would have quantized many to zero.
    Float16 stores each value directly; no value rounds to zero
    unless it's already below ~6e-8 (the float16 subnormal floor).
    """

    def test_gsplat_amplitude_range_preserved(self):
        np.random.seed(42)
        amplitudes = np.random.exponential(0.005, 8000).astype(np.float32)
        amplitudes = np.clip(amplitudes, 0.000008, 0.05)

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

            assert group["amplitudes"].dtype == np.float16

            arr = np.array(group["amplitudes"])
            zero_count = np.sum(arr == 0)
            assert zero_count == 0, (
                f"GSplat amplitudes: {zero_count} zeros "
                f"({100 * zero_count / len(arr):.1f}%) — invisible splats."
            )

            # Round-trip accuracy: float16 ULP is relative and grows
            # near the subnormal floor (~6e-8). Allow 2e-3 rtol for
            # the smallest amplitudes; normal-range values land well
            # inside 5e-4.
            np.testing.assert_allclose(
                arr.astype(np.float32), amplitudes, rtol=2e-3
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
