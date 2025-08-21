"""Comprehensive tests for array_utils module."""

import numpy as np
import pytest

from luxar.array_utils import (
    broadcast_color_to_points,
    broadcast_radii_to_points,
    broadcast_scalar_to_points,
    broadcast_sharpness_to_points,
    ensure_float32,
    validate_array_shape,
)


class TestBroadcastColorToPoints:
    """Test color broadcasting functionality."""

    def test_none_colors(self):
        """Test that None colors returns None."""
        result = broadcast_color_to_points(None, 100)
        assert result is None

    def test_single_rgb_tuple(self):
        """Test broadcasting single RGB tuple."""
        colors = (1.0, 0.5, 0.0)
        result = broadcast_color_to_points(colors, 100)
        assert result.shape == (100, 3)
        assert np.all(result == [1.0, 0.5, 0.0])

    def test_single_rgb_list(self):
        """Test broadcasting single RGB list."""
        colors = [0.2, 0.3, 0.4]
        result = broadcast_color_to_points(colors, 50)
        assert result.shape == (50, 3)
        np.testing.assert_array_almost_equal(result[0], [0.2, 0.3, 0.4])
        assert np.all(result == result[0])  # All rows should be the same

    def test_single_rgb_array(self):
        """Test broadcasting single RGB numpy array."""
        colors = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = broadcast_color_to_points(colors, 75)
        assert result.shape == (75, 3)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        assert np.all(result == result[0])

    def test_full_color_array(self):
        """Test passing full color array."""
        n_points = 100
        colors = np.random.rand(n_points, 3).astype(np.float32)
        result = broadcast_color_to_points(colors, n_points)
        assert result.shape == (n_points, 3)
        np.testing.assert_array_equal(result, colors)

    def test_wrong_shape_error(self):
        """Test error for wrong color shape."""
        colors = np.random.rand(50, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Colors must have shape"):
            broadcast_color_to_points(colors, 100)  # Wrong number of points

        colors = np.random.rand(100, 4).astype(np.float32)  # 4 channels
        with pytest.raises(ValueError, match="Colors must have shape"):
            broadcast_color_to_points(colors, 100)


class TestBroadcastScalarToPoints:
    """Test scalar broadcasting functionality."""

    def test_none_values(self):
        """Test that None values returns None."""
        result = broadcast_scalar_to_points(None, 100)
        assert result is None

    def test_scalar_float(self):
        """Test broadcasting scalar float."""
        result = broadcast_scalar_to_points(0.5, 100, "test")
        assert result.shape == (100,)
        assert np.all(result == 0.5)

    def test_scalar_int(self):
        """Test broadcasting scalar int."""
        result = broadcast_scalar_to_points(2, 50, "test")
        assert result.shape == (50,)
        assert np.all(result == 2.0)

    def test_full_array(self):
        """Test passing full array."""
        values = np.random.rand(100).astype(np.float32)
        result = broadcast_scalar_to_points(values, 100, "test")
        assert result.shape == (100,)
        np.testing.assert_array_equal(result, values)

    def test_negative_scalar_with_positive_required(self):
        """Test error for negative scalar when positive required."""
        with pytest.raises(ValueError, match="must be positive"):
            broadcast_scalar_to_points(-0.5, 100, "test", require_positive=True)

    def test_negative_scalar_allowed(self):
        """Test negative scalar when allowed."""
        result = broadcast_scalar_to_points(-0.5, 100, "test", require_positive=False)
        assert np.all(result == -0.5)

    def test_negative_in_array_with_positive_required(self):
        """Test error for negative values in array."""
        values = np.array([0.5, -0.1, 0.3], dtype=np.float32)
        with pytest.raises(ValueError, match="must be positive"):
            broadcast_scalar_to_points(values, 3, "test", require_positive=True)

    def test_wrong_array_shape(self):
        """Test error for wrong array shape."""
        values = np.random.rand(50).astype(np.float32)
        with pytest.raises(ValueError, match="must be scalar or have shape"):
            broadcast_scalar_to_points(values, 100, "test")


class TestBroadcastRadiiToPoints:
    """Test radii broadcasting."""

    def test_none_radii(self):
        """Test that None radii returns None."""
        result = broadcast_radii_to_points(None, 100)
        assert result is None

    def test_scalar_radius(self):
        """Test broadcasting scalar radius."""
        result = broadcast_radii_to_points(0.5, 100)
        assert result.shape == (100,)
        assert np.all(result == 0.5)

    def test_full_radii_array(self):
        """Test passing full radii array."""
        radii = np.random.uniform(0.1, 2.0, 100).astype(np.float32)
        result = broadcast_radii_to_points(radii, 100)
        np.testing.assert_array_equal(result, radii)

    def test_negative_radius_error(self):
        """Test error for negative radius."""
        with pytest.raises(ValueError, match="must be positive"):
            broadcast_radii_to_points(-0.5, 100)


class TestBroadcastSharpnessToPoints:
    """Test sharpness broadcasting."""

    def test_none_sharpness(self):
        """Test that None sharpness returns None."""
        result = broadcast_sharpness_to_points(None, 100)
        assert result is None

    def test_scalar_sharpness(self):
        """Test broadcasting scalar sharpness."""
        result = broadcast_sharpness_to_points(2.0, 100)
        assert result.shape == (100,)
        assert np.all(result == 2.0)

    def test_full_sharpness_array(self):
        """Test passing full sharpness array."""
        sharpness = np.random.uniform(0.5, 10.0, 100).astype(np.float32)
        result = broadcast_sharpness_to_points(sharpness, 100)
        np.testing.assert_array_equal(result, sharpness)

    def test_out_of_range_warning(self):
        """Test warning for out-of-range sharpness."""
        with pytest.warns(UserWarning, match="outside typical range"):
            result = broadcast_sharpness_to_points(0.1, 100)  # Too low
        assert result is not None

        with pytest.warns(UserWarning, match="outside typical range"):
            result = broadcast_sharpness_to_points(20.0, 100)  # Too high
        assert result is not None

    def test_out_of_range_array_warning(self):
        """Test warning for out-of-range values in array."""
        sharpness = np.array([0.1, 5.0, 20.0], dtype=np.float32)
        with pytest.warns(UserWarning, match="outside typical range"):
            result = broadcast_sharpness_to_points(sharpness, 3)
        assert result is not None

    def test_no_warning_suppression(self):
        """Test that warnings can be suppressed."""
        result = broadcast_sharpness_to_points(0.1, 100, warn_on_out_of_range=False)
        assert result is not None  # Should work without warning


class TestEnsureFloat32:
    """Test float32 conversion."""

    def test_already_float32(self):
        """Test array already float32 is unchanged."""
        array = np.random.rand(10, 3).astype(np.float32)
        result = ensure_float32(array)
        assert result is array  # Same object
        assert result.dtype == np.float32

    def test_float64_conversion(self):
        """Test float64 to float32 conversion."""
        array = np.random.rand(10, 3).astype(np.float64)
        result = ensure_float32(array)
        assert result is not array  # Different object
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, array.astype(np.float32))

    def test_int_conversion(self):
        """Test int to float32 conversion."""
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        result = ensure_float32(array)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, array.astype(np.float32))

    def test_uint8_conversion(self):
        """Test uint8 to float32 conversion."""
        array = np.array([0, 128, 255], dtype=np.uint8)
        result = ensure_float32(array)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [0.0, 128.0, 255.0])


class TestValidateArrayShape:
    """Test array shape validation."""

    def test_exact_shape_match(self):
        """Test exact shape match passes."""
        array = np.zeros((10, 3))
        validate_array_shape(array, (10, 3))  # Should not raise

    def test_exact_shape_mismatch(self):
        """Test exact shape mismatch raises."""
        array = np.zeros((10, 3))
        with pytest.raises(ValueError, match="must have shape"):
            validate_array_shape(array, (10, 4))

    def test_multiple_valid_shapes(self):
        """Test multiple valid shapes."""
        array = np.zeros((10, 3))
        validate_array_shape(array, [(10, 3), (10, 4)])  # Should pass

        array = np.zeros((10, 4))
        validate_array_shape(array, [(10, 3), (10, 4)])  # Should pass

    def test_multiple_shapes_all_invalid(self):
        """Test error when none of multiple shapes match."""
        array = np.zeros((10, 5))
        with pytest.raises(ValueError, match="must have shape.*or"):
            validate_array_shape(array, [(10, 3), (10, 4)])

    def test_custom_name_in_error(self):
        """Test custom name appears in error message."""
        array = np.zeros((10, 3))
        with pytest.raises(ValueError, match="Positions must have shape"):
            validate_array_shape(array, (10, 4), name="positions")
