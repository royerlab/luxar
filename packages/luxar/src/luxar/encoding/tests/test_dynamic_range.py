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

    def test_min_anchoring_tightens_far_from_zero_data(self):
        """Rescale-first: data far from zero gets the full code space over
        its own span — [10, 11] must resolve ~10x better than the old
        0-anchored grid could (max/255/2 = 0.022 vs span/255/2 = 0.002)."""
        data = np.linspace(10.0, 11.0, 500).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            ArrayEncoder().encode(data, group, "t", SemanticType.POSITIVE_SCALAR)
            enc = group["t"].attrs["encoding"]
            assert enc["name"] == "bounded_scalar_uint8"
            assert enc["min"] == 10.0  # anchored at the data's own minimum
            from luxar.encoding.decoder import ArrayDecoder

            decoded = ArrayDecoder().decode(group["t"], group)
            max_err = float(np.abs(decoded - data).max())
            assert max_err <= (11.0 - 10.0) / 255.0 / 2 + 1e-6

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


class TestBoundedScalarSignedData:
    """Regression tests for signed BOUNDED_SCALAR data (issue #730 fix).

    Colormap scalars became BOUNDED_SCALAR and are legitimately signed, so
    the bit-selection and float fallback in ``_encode_bounded_scalar`` must be
    sign-aware. Bit depth is chosen from MAGNITUDES so negating data does not
    silently degrade precision, and the wide-range float path never overflows
    float16 to inf.
    """

    def test_bit_depth_is_negation_invariant(self):
        """An all-negative array and its positive mirror pick the SAME dtype.

        A wide magnitude range (~65000:1) must select uint16 for both. Before
        the fix, the all-negative array had an empty ``data > 0`` mask and
        collapsed to uint8 — a 256x precision loss under negation.
        """
        pos = np.linspace(1.0, 65000.0, 5000).astype(np.float32)
        neg = -pos  # same magnitudes, opposite sign

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(pos, group, "pos", SemanticType.BOUNDED_SCALAR)
            encoder.encode(neg, group, "neg", SemanticType.BOUNDED_SCALAR)

            pos_enc = group["pos"].attrs["encoding"]["name"]
            neg_enc = group["neg"].attrs["encoding"]["name"]
            assert pos_enc == "bounded_scalar_uint16", pos_enc
            assert neg_enc == pos_enc, f"{neg_enc} != {pos_enc}"

    def test_all_negative_wide_range_precision(self):
        """An all-negative wide array decodes with uint16 (~0.5), not uint8 error.

        Directly exercises Defect A: before the fix it stored as uint8 with a
        max absolute error ~128; after the fix uint16 gives ~0.5.
        """
        from luxar.encoding.decoder import ArrayDecoder

        data = np.linspace(-65536.0, -1.0, 300000).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.BOUNDED_SCALAR)

            assert group["test"].attrs["encoding"]["name"] == "bounded_scalar_uint16"
            decoded = ArrayDecoder().decode(group["test"])
            max_err = float(np.max(np.abs(decoded - data)))
            # uint16 over a span of 65535 → step ≈ 65535/65535 = 1.0, err ≤ 0.5.
            assert max_err < 1.0, f"max abs error {max_err} — degraded to uint8?"

    def test_mixed_sign_span_selects_uint16(self):
        """Mixed-sign data whose MAGNITUDE range needs uint16 gets uint16.

        A tiny positive lobe (range ~8:1) beside a wide negative lobe. Before
        the fix, bits were sized from the positive lobe only → uint8.
        """
        neg_lobe = np.linspace(-65000.0, -1000.0, 5000).astype(np.float32)
        pos_lobe = np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float32)
        data = np.concatenate([neg_lobe, pos_lobe])

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.BOUNDED_SCALAR)

            enc = group["test"].attrs["encoding"]["name"]
            assert enc == "bounded_scalar_uint16", enc

    def test_non_negative_encoding_unchanged(self):
        """np.abs is a no-op for non-negative data: encodings stay identical.

        Guards the claim that sharpness/opacity and existing non-negative
        colormap scalars are byte-identical after the fix.
        """
        narrow = np.linspace(10.0, 50.0, 1000).astype(np.float32)  # range ~5:1
        medium = np.linspace(0.01, 100.0, 1000).astype(np.float32)  # range ~1e4:1

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(narrow, group, "narrow", SemanticType.BOUNDED_SCALAR)
            encoder.encode(medium, group, "medium", SemanticType.BOUNDED_SCALAR)

            assert group["narrow"].attrs["encoding"]["name"] == "bounded_scalar_uint8"
            assert group["medium"].attrs["encoding"]["name"] == "bounded_scalar_uint16"

    def test_signed_integer_minimum_is_abs_safe(self):
        """int64 min must not wrap under np.abs and corrupt bit selection.

        ``np.abs(int64 min)`` overflows and stays negative, so the magnitude
        range would exclude the array's largest value: bits would be sized
        from the small positives only (uint16) while the real span is ~9.2e18
        — a catastrophic quantization error. Magnitudes are computed in
        float64 for signed integers, so the wide range selects float32.
        """
        from luxar.encoding.decoder import ArrayDecoder

        int_min = np.iinfo(np.int64).min
        # > 256 unique values so the LUT fast path does not intercept.
        data = np.concatenate(
            [
                np.array([int_min], dtype=np.int64),
                np.arange(1, 1001, dtype=np.int64),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder()
            encoder.encode(data, group, "test", SemanticType.BOUNDED_SCALAR)

            enc = group["test"].attrs["encoding"]["name"]
            assert enc == "float32", f"expected float32 for ~9.2e18 span, got {enc}"
            decoded = ArrayDecoder().decode(group["test"])
            # |int64 min| = 2**63 and 1..1000 are all exact in float32.
            np.testing.assert_array_equal(decoded, data.astype(np.float32))

    def test_wide_range_float16_no_overflow_to_inf(self):
        """Defect B: wide-range values > 65504 must not overflow float16 → inf.

        With ``float16_allowed=True`` the wide-range float branch would cast to
        float16 (max 65504) and write inf. The guard falls back to float32.
        """
        from luxar.encoding.decoder import ArrayDecoder

        # Magnitude dynamic range > 65536 forces the float branch; max 1e6
        # exceeds float16's 65504 ceiling. Strictly increasing → no LUT.
        data = np.linspace(1.0, 1_000_000.0, 1000).astype(np.float32)

        # Document that an unguarded float16 cast WOULD produce inf.
        assert np.any(~np.isfinite(data.astype(np.float16)))

        with tempfile.TemporaryDirectory() as tmpdir:
            group = zarr.open_group(tmpdir, mode="w")
            encoder = ArrayEncoder(float16_allowed=True)
            encoder.encode(data, group, "test", SemanticType.BOUNDED_SCALAR)

            enc = group["test"].attrs["encoding"]["name"]
            assert enc == "float32", f"expected float32 fallback, got {enc}"
            decoded = ArrayDecoder().decode(group["test"])
            assert np.all(np.isfinite(decoded)), "float16 overflow leaked inf to disk"


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

            # Decode manually with the stored [min, max] anchors
            # (rescale-first: the grid spans the array's own range).
            arr = np.array(group["test"])
            enc = group["test"].attrs["encoding"]
            min_val, max_val = enc["min"], enc["max"]
            decoded = min_val + arr / 255.0 * (max_val - min_val)

            # Max absolute error should be <= span / 255 / 2 (with rounding) —
            # a TIGHTER bound than the old 0-anchored max/255/2.
            abs_error = np.abs(decoded - data)
            max_abs_error = np.max(abs_error)
            expected_max_error = (max_val - min_val) / 255.0 / 2
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

            # Decode manually with the stored [min, max] anchors
            arr = np.array(group["test"])
            enc = group["test"].attrs["encoding"]
            min_val, max_val = enc["min"], enc["max"]
            decoded = min_val + arr / 65535.0 * (max_val - min_val)

            # Max absolute error should be <= span / 65535 / 2 (with rounding)
            # Allow 1e-7 tolerance for float32 precision
            abs_error = np.abs(decoded - data)
            max_abs_error = np.max(abs_error)
            expected_max_error = (max_val - min_val) / 65535.0 / 2
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

            # Rescale-first anchoring: raw code 0 now decodes to MIN (the
            # smallest value), not to zero — so assert on DECODED values.
            from luxar.encoding.decoder import ArrayDecoder

            decoded = ArrayDecoder().decode(group["test"], group)
            zero_count = int(np.sum(decoded == 0))
            assert zero_count == 0, (
                f"Expected 0 decoded zeros, got {zero_count} "
                f"({100 * zero_count / len(decoded):.1f}%)"
            )
            # and the smallest value survives with the tighter span grid
            assert decoded.min() > 0


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

            # Verify NO data loss: no nonzero amplitude may DECODE to zero
            # (raw code 0 legitimately appears — it decodes to min, and for
            # geolog it is the reserved exact-zero level).
            from luxar.encoding.decoder import ArrayDecoder

            decoded = ArrayDecoder().decode(group["amplitudes"], group)
            zeroed = int(np.sum((decoded == 0) & (amplitudes > 0)))
            assert zeroed == 0, (
                f"GSplat amplitudes: {zeroed} splats zeroed - "
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
