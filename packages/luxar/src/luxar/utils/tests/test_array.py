"""Tests for array utility functions.

Tests cover:
- ensure_float32: dtype conversion
- validate_array_shape: shape validation with single and multiple expected shapes
"""

import numpy as np
import pytest

import luxar.utils.array as array_utils
from luxar.utils.array import ensure_float32, validate_array_shape


def test_obsolete_broadcast_helpers_are_not_reintroduced() -> None:
    """Scalar broadcasting should stay centralized in ArrayEncoder.

    Guard each removed symbol so any reintroduction surfaces as a test failure.
    """
    assert not hasattr(array_utils, "broadcast_color_to_points")
    assert not hasattr(array_utils, "broadcast_scalar_to_points")
    assert not hasattr(array_utils, "broadcast_radii_to_points")
    assert not hasattr(array_utils, "broadcast_sharpness_to_points")


class TestEnsureFloat32:
    """Tests for ensure_float32 function."""

    def test_already_float32(self) -> None:
        """Test that float32 arrays are returned unchanged."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = ensure_float32(arr)
        assert result is arr  # Same object, not a copy
        assert result.dtype == np.float32

    def test_float64_to_float32(self) -> None:
        """Test conversion from float64 to float32."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = ensure_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_int32_to_float32(self) -> None:
        """Test conversion from int32 to float32."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = ensure_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_int64_to_float32(self) -> None:
        """Test conversion from int64 to float32."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = ensure_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_uint8_to_float32(self) -> None:
        """Test conversion from uint8 to float32."""
        arr = np.array([0, 127, 255], dtype=np.uint8)
        result = ensure_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [0.0, 127.0, 255.0])

    def test_float16_to_float32(self) -> None:
        """Test conversion from float16 to float32."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result = ensure_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_2d_array(self) -> None:
        """Test conversion of 2D arrays."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ensure_float32(arr)
        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_3d_array(self) -> None:
        """Test conversion of 3D arrays."""
        arr = np.ones((2, 3, 4), dtype=np.float64)
        result = ensure_float32(arr)
        assert result.dtype == np.float32
        assert result.shape == (2, 3, 4)


class TestValidateArrayShape:
    """Tests for validate_array_shape function."""

    def test_valid_single_shape(self) -> None:
        """Test validation passes for matching single shape."""
        arr = np.zeros((10, 3))
        validate_array_shape(arr, (10, 3), name="positions")
        # No exception means success

    def test_invalid_single_shape(self) -> None:
        """Test validation fails for non-matching single shape."""
        arr = np.zeros((10, 4))
        with pytest.raises(ValueError, match="Positions must have shape"):
            validate_array_shape(arr, (10, 3), name="positions")

    def test_valid_multiple_shapes_first_match(self) -> None:
        """Test validation passes when array matches first of multiple shapes."""
        arr = np.zeros((10, 3))
        validate_array_shape(arr, [(10, 3), (10, 4)], name="colors")
        # No exception means success

    def test_valid_multiple_shapes_second_match(self) -> None:
        """Test validation passes when array matches second of multiple shapes."""
        arr = np.zeros((10, 4))
        validate_array_shape(arr, [(10, 3), (10, 4)], name="colors")
        # No exception means success

    def test_invalid_multiple_shapes(self) -> None:
        """Test validation fails when array matches none of multiple shapes."""
        arr = np.zeros((10, 5))
        with pytest.raises(ValueError, match="Colors must have shape.*or"):
            validate_array_shape(arr, [(10, 3), (10, 4)], name="colors")

    def test_default_name(self) -> None:
        """Test that default name 'array' is used in error message."""
        arr = np.zeros((5,))
        with pytest.raises(ValueError, match="Array must have shape"):
            validate_array_shape(arr, (10,))

    # [Python-R2/D-G5] Optional `check_finite=True` rejects NaN / ±Inf.
    # Default behaviour (check_finite=False) is backward-compatible —
    # the function still accepts non-finite content.
    def test_check_finite_false_accepts_nan(self) -> None:
        arr = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        # No exception: default does NOT validate content.
        validate_array_shape(arr, (1, 3))

    def test_check_finite_true_rejects_nan(self) -> None:
        arr = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            validate_array_shape(arr, (1, 3), check_finite=True)

    def test_check_finite_true_rejects_positive_inf(self) -> None:
        arr = np.array([np.inf, 1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            validate_array_shape(arr, (3,), check_finite=True)

    def test_check_finite_true_rejects_negative_inf(self) -> None:
        arr = np.array([-np.inf, 1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            validate_array_shape(arr, (3,), check_finite=True)

    def test_check_finite_true_accepts_all_finite(self) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # No exception.
        validate_array_shape(arr, (3,), check_finite=True)

    def test_check_finite_true_skips_integer_dtype(self) -> None:
        # Integers can't hold NaN/Inf — the finiteness check is a no-op
        # for integer dtypes. (np.isfinite would either work or raise
        # TypeError depending on numpy version; the guard makes the
        # function dtype-safe.)
        arr = np.array([1, 2, 3], dtype=np.int32)
        validate_array_shape(arr, (3,), check_finite=True)

    def test_1d_array(self) -> None:
        """Test validation of 1D arrays."""
        arr = np.zeros(100)
        validate_array_shape(arr, (100,), name="radii")
        # No exception means success

    def test_3d_array(self) -> None:
        """Test validation of 3D arrays."""
        arr = np.zeros((2, 3, 4))
        validate_array_shape(arr, (2, 3, 4), name="tensor")
        # No exception means success

    def test_empty_array(self) -> None:
        """Test validation of empty arrays."""
        arr = np.zeros((0, 3))
        validate_array_shape(arr, (0, 3), name="empty_positions")
        # No exception means success

    def test_error_message_contains_actual_shape(self) -> None:
        """Test that error message contains actual array shape."""
        arr = np.zeros((15, 7))
        with pytest.raises(ValueError) as exc_info:
            validate_array_shape(arr, (10, 3), name="data")
        assert "(15, 7)" in str(exc_info.value)

    def test_multiple_shapes_error_message_lists_all(self) -> None:
        """Test that error message for multiple shapes lists all valid options."""
        arr = np.zeros((10, 5))
        with pytest.raises(ValueError) as exc_info:
            validate_array_shape(arr, [(10, 3), (10, 4), (10, 6)], name="data")
        error_msg = str(exc_info.value)
        assert "(10, 3)" in error_msg
        assert "(10, 4)" in error_msg
        assert "(10, 6)" in error_msg
