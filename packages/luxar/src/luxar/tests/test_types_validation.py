"""Test validation functions in types.py module."""

import numpy as np
import pytest

from luxar.types import (
    validate_blending_mode,
    validate_gamma,
    validate_node_type,
    validate_opacity,
    validate_physical_unit,
    validate_transform,
)


class TestTransformValidation:
    """Test validate_transform function."""

    def test_valid_transform(self):
        """Test that a valid 4x4 transform matrix is accepted."""
        transform = np.eye(4, dtype=np.float32)
        result = validate_transform(transform)
        assert result.shape == (4, 4)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, transform)

    def test_transform_wrong_type(self):
        """Test that non-numpy array input raises ValueError."""
        with pytest.raises(ValueError, match="Transform must be a numpy array"):
            validate_transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def test_transform_wrong_shape(self):
        """Test that non-4x4 matrix raises ValueError."""
        # Test 3x3 matrix
        transform_3x3 = np.eye(3)
        with pytest.raises(ValueError, match="Transform must be a 4x4 matrix"):
            validate_transform(transform_3x3)

        # Test 4x3 matrix
        transform_4x3 = np.ones((4, 3))
        with pytest.raises(ValueError, match="Transform must be a 4x4 matrix"):
            validate_transform(transform_4x3)

        # Test 1D array
        transform_1d = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Transform must be a 4x4 matrix"):
            validate_transform(transform_1d)

    def test_transform_dtype_conversion(self):
        """Test that transform is converted to float32."""
        # Test float64 conversion
        transform_f64 = np.eye(4, dtype=np.float64)
        result = validate_transform(transform_f64)
        assert result.dtype == np.float32

        # Test int conversion
        transform_int = np.eye(4, dtype=np.int32)
        result = validate_transform(transform_int)
        assert result.dtype == np.float32


class TestNodeTypeValidation:
    """Test validate_node_type function."""

    def test_valid_node_types(self):
        """Test that valid node types are accepted."""
        assert validate_node_type("points") == "points"
        assert validate_node_type("group") == "group"
        assert validate_node_type("scene") == "scene"

    def test_invalid_node_type(self):
        """Test that invalid node types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid node type 'invalid'"):
            validate_node_type("invalid")

        with pytest.raises(ValueError, match="Invalid node type 'mesh'"):
            validate_node_type("mesh")

        with pytest.raises(ValueError, match="Invalid node type ''"):
            validate_node_type("")


class TestPhysicalUnitValidation:
    """Test validate_physical_unit function."""

    def test_valid_units(self):
        """Test that valid physical units are accepted."""
        # Test all valid units
        valid_units = [
            "nm",
            "um",
            "mm",
            "cm",
            "m",
            "metre",
            "meter",
            "km",
            "inch",
            "foot",
            "px",
            "au",
        ]
        for unit in valid_units:
            assert validate_physical_unit(unit) == unit

    def test_invalid_unit(self):
        """Test that invalid units raise ValueError."""
        with pytest.raises(ValueError, match="Invalid unit 'angstrom'"):
            validate_physical_unit("angstrom")

        with pytest.raises(ValueError, match="Invalid unit 'mile'"):
            validate_physical_unit("mile")

        with pytest.raises(ValueError, match="Invalid unit ''"):
            validate_physical_unit("")

        with pytest.raises(ValueError, match="Invalid unit 'yards'"):
            validate_physical_unit("yards")


class TestOpacityValidation:
    """Test validate_opacity function."""

    def test_valid_opacity(self):
        """Test that valid opacity values are accepted."""
        assert validate_opacity(0.0) == 0.0
        assert validate_opacity(0.5) == 0.5
        assert validate_opacity(1.0) == 1.0
        assert validate_opacity(0.25) == 0.25
        assert validate_opacity(0.99) == 0.99

    def test_opacity_type_conversion(self):
        """Test that opacity values are converted to float."""
        assert validate_opacity(1) == 1.0
        assert validate_opacity(0) == 0.0
        assert validate_opacity("0.5") == 0.5
        assert validate_opacity(np.float32(0.7)) == pytest.approx(0.7)

    def test_invalid_opacity(self):
        """Test that invalid opacity values raise ValueError."""
        with pytest.raises(ValueError, match="Opacity must be between 0.0 and 1.0"):
            validate_opacity(-0.1)

        with pytest.raises(ValueError, match="Opacity must be between 0.0 and 1.0"):
            validate_opacity(1.1)

        with pytest.raises(ValueError, match="Opacity must be between 0.0 and 1.0"):
            validate_opacity(2.0)

        with pytest.raises(TypeError, match="Opacity must be convertible to float"):
            validate_opacity("invalid")


class TestGammaValidation:
    """Test validate_gamma function."""

    def test_valid_gamma(self):
        """Test that valid gamma values are accepted."""
        assert validate_gamma(0.2) == 0.2
        assert validate_gamma(1.0) == 1.0
        assert validate_gamma(2.0) == 2.0
        assert validate_gamma(0.5) == 0.5
        assert validate_gamma(1.5) == 1.5

    def test_gamma_type_conversion(self):
        """Test that gamma values are converted to float."""
        assert validate_gamma(1) == 1.0
        assert validate_gamma(2) == 2.0
        assert validate_gamma("1.5") == 1.5
        assert validate_gamma(np.float32(1.8)) == pytest.approx(1.8)

    def test_invalid_gamma(self):
        """Test that invalid gamma values raise ValueError."""
        with pytest.raises(ValueError, match="Gamma must be between 0.2 and 2.0"):
            validate_gamma(0.0)

        with pytest.raises(ValueError, match="Gamma must be between 0.2 and 2.0"):
            validate_gamma(0.1)

        with pytest.raises(ValueError, match="Gamma must be between 0.2 and 2.0"):
            validate_gamma(2.5)

        with pytest.raises(ValueError, match="Gamma must be between 0.2 and 2.0"):
            validate_gamma(3.0)

        with pytest.raises(TypeError, match="Gamma must be convertible to float"):
            validate_gamma("not_a_number")


class TestBlendingModeValidation:
    """Test validate_blending_mode function."""

    def test_valid_blending_modes(self):
        """Test that valid blending modes are accepted."""
        assert validate_blending_mode("normal") == "normal"
        assert validate_blending_mode("additive") == "additive"
        assert validate_blending_mode("multiply") == "multiply"
        assert validate_blending_mode("minimum") == "minimum"
        assert validate_blending_mode("maximum") == "maximum"

    def test_invalid_blending_mode(self):
        """Test that invalid blending modes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid blending mode 'overlay'"):
            validate_blending_mode("overlay")

        with pytest.raises(ValueError, match="Invalid blending mode 'screen'"):
            validate_blending_mode("screen")

        with pytest.raises(ValueError, match="Invalid blending mode ''"):
            validate_blending_mode("")

        with pytest.raises(ValueError, match="Invalid blending mode 'NORMAL'"):
            validate_blending_mode("NORMAL")  # Case sensitive
