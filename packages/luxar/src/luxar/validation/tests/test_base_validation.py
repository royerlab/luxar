"""Direct unit tests for validation/base.py functions."""

import numpy as np
import pytest

from luxar.validation.base import (
    ValidationError,
    validate_colors_for_writing,
    validate_positions_for_writing,
    validate_radii_for_writing,
    validate_sharpness_for_writing,
    validate_zarr_attributes,
)


class TestValidationError:
    """Test ValidationError class."""

    def test_validation_error_basic(self) -> None:
        """Test ValidationError with just a message."""
        error = ValidationError("Something went wrong")
        assert "Something went wrong" in str(error)
        assert "💡" not in str(error)  # No suggestion

    def test_validation_error_with_suggestion(self) -> None:
        """Test ValidationError with message and suggestion."""
        error = ValidationError(
            "Invalid array shape", suggestion="Reshape your array to (N, 3)"
        )
        assert "Invalid array shape" in str(error)
        assert "💡 Suggestion: Reshape your array to (N, 3)" in str(error)

    def test_validation_error_can_be_caught_as_value_error(self) -> None:  # type: ignore[no-untyped-def]
        """Test that ValidationError is a ValueError subclass."""
        with pytest.raises(ValueError):
            raise ValidationError("test error")

    def test_validation_error_preserves_suggestion(self) -> None:  # type: ignore[no-untyped-def]
        """Test that suggestion is accessible in error message."""
        try:
            raise ValidationError("problem", "try this fix")
        except ValidationError as e:
            assert "problem" in str(e)
            assert "try this fix" in str(e)


class TestValidatePositionsForWriting:
    """Test validate_positions_for_writing function."""

    def test_valid_positions_2d(self) -> None:
        """Test valid 2D positions."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        n_points, n_dims = validate_positions_for_writing(positions)
        assert n_points == 2
        assert n_dims == 2

    def test_valid_positions_3d(self) -> None:
        """Test valid 3D positions."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        n_points, n_dims = validate_positions_for_writing(positions)
        assert n_points == 2
        assert n_dims == 3

    def test_not_numpy_array(self) -> None:
        """Test non-numpy array raises ValidationError."""
        with pytest.raises(ValidationError, match="Expected numpy array"):
            validate_positions_for_writing([[0.0, 0.0], [1.0, 1.0]])

        # Check suggestion is present
        try:
            validate_positions_for_writing([1, 2, 3])
        except ValidationError as e:
            assert "Convert your data to a numpy array" in str(e)

    def test_1d_array_with_helpful_error(self) -> None:
        """Test 1D array raises ValidationError with reshape suggestion."""
        positions_1d = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValidationError, match="1D array"):
            validate_positions_for_writing(positions_1d)

        # Check suggestion
        try:
            validate_positions_for_writing(positions_1d)
        except ValidationError as e:
            assert "reshape" in str(e)
            assert "positions.reshape(-1, 1)" in str(e)

    def test_3d_array_with_helpful_error(self) -> None:
        """Test 3D array raises ValidationError with streaming suggestion."""
        positions_3d = np.random.rand(10, 5, 3)

        with pytest.raises(ValidationError, match="3D array"):
            validate_positions_for_writing(positions_3d)

        # Check suggestion
        try:
            validate_positions_for_writing(positions_3d)
        except ValidationError as e:
            assert "flatten" in str(e) or "StreamingPoints" in str(e)

    def test_empty_positions(self) -> None:
        """Test empty positions (0 points) raises ValidationError."""
        positions_empty = np.empty((0, 3), dtype=np.float32)

        with pytest.raises(ValidationError, match="empty points"):
            validate_positions_for_writing(positions_empty)

        # Check suggestion
        try:
            validate_positions_for_writing(positions_empty)
        except ValidationError as e:
            assert "at least one point" in str(e)

    def test_custom_context(self) -> None:
        """Test custom context in error messages."""
        positions = [[1, 2, 3]]  # Not numpy array

        with pytest.raises(ValidationError, match="mydata"):
            validate_positions_for_writing(positions, context="mydata")


class TestValidateColorsForWriting:
    """Test validate_colors_for_writing function."""

    def test_valid_colors(self) -> None:
        """Test valid colors."""
        colors = np.random.rand(10, 3).astype(np.float32)
        # Should not raise - function returns None
        validate_colors_for_writing(colors, n_points=10)

    def test_not_numpy_array(self) -> None:
        """Test non-numpy array raises ValidationError."""
        with pytest.raises(ValidationError, match="Expected numpy array"):
            validate_colors_for_writing([[1.0, 0.0, 0.0]], n_points=1)

    def test_wrong_shape_suggestions(self) -> None:
        """Test wrong shape error includes helpful suggestions."""
        # Wrong number of points
        colors = np.random.rand(5, 3).astype(np.float32)

        with pytest.raises(ValidationError):
            validate_colors_for_writing(colors, n_points=10)

        # Wrong number of channels
        colors_rgba = np.random.rand(10, 4).astype(np.float32)

        try:
            validate_colors_for_writing(colors_rgba, n_points=10)
        except ValidationError as e:
            # Should have helpful suggestion about RGB vs RGBA
            assert "3" in str(e)  # Mentions need for 3 channels

    def test_negative_colors(self) -> None:
        """Test negative colors raise ValidationError."""
        colors_neg = np.array([[1.0, -0.5, 0.0]], dtype=np.float32)

        with pytest.raises(ValidationError, match="negative"):
            validate_colors_for_writing(colors_neg, n_points=1)

        # Check suggestion
        try:
            validate_colors_for_writing(colors_neg, n_points=1)
        except ValidationError as e:
            assert "Suggestion" in str(e) or "💡" in str(e)


class TestValidateRadiiForWriting:
    """Test validate_radii_for_writing function."""

    def test_valid_radii(self) -> None:
        """Test valid radii."""
        radii = np.ones(10, dtype=np.float32)
        # Should not raise - function returns None
        validate_radii_for_writing(radii, n_points=10)

    def test_not_numpy_array(self) -> None:
        """Test non-numpy array raises ValidationError."""
        with pytest.raises(ValidationError, match="Expected numpy array"):
            validate_radii_for_writing([1.0, 2.0], n_points=2)

    def test_wrong_shape(self) -> None:
        """Test wrong shape raises ValidationError."""
        # 2D instead of 1D
        radii_2d = np.ones((5, 1), dtype=np.float32)

        with pytest.raises(ValidationError, match="shape"):
            validate_radii_for_writing(radii_2d, n_points=5)

    def test_wrong_length(self) -> None:
        """Test wrong length raises ValidationError."""
        radii = np.ones(5, dtype=np.float32)

        with pytest.raises(ValidationError):
            validate_radii_for_writing(radii, n_points=10)

    def test_negative_radii(self) -> None:
        """Test negative radii raise ValidationError."""
        radii_neg = np.array([1.0, -0.5, 1.0], dtype=np.float32)

        with pytest.raises(ValidationError, match="positive"):
            validate_radii_for_writing(radii_neg, n_points=3)

        # Check suggestion
        try:
            validate_radii_for_writing(radii_neg, n_points=3)
        except ValidationError as e:
            assert "💡" in str(e) or "Suggestion" in str(e)

    def test_zero_radii(self) -> None:
        """Test zero radii raise ValidationError."""
        radii_zero = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        with pytest.raises(ValidationError, match="positive"):
            validate_radii_for_writing(radii_zero, n_points=3)


class TestValidateSharpnessForWriting:
    """Test validate_sharpness_for_writing function."""

    def test_valid_sharpness(self) -> None:
        """Test valid sharpness values."""
        sharpness = np.ones(10, dtype=np.float32) * 2.0
        # Should not raise - function returns None
        validate_sharpness_for_writing(sharpness, n_points=10)

    def test_not_numpy_array(self) -> None:
        """Test non-numpy array raises ValidationError."""
        with pytest.raises(ValidationError, match="Expected numpy array"):
            validate_sharpness_for_writing([1.0, 2.0], n_points=2)

    def test_wrong_shape(self) -> None:
        """Test wrong shape raises ValidationError."""
        # 2D instead of 1D
        sharpness_2d = np.ones((5, 1), dtype=np.float32)

        with pytest.raises(ValidationError, match="shape"):
            validate_sharpness_for_writing(sharpness_2d, n_points=5)

    def test_wrong_length(self) -> None:
        """Test wrong length raises ValidationError."""
        sharpness = np.ones(5, dtype=np.float32)

        with pytest.raises(ValidationError):
            validate_sharpness_for_writing(sharpness, n_points=10)

    def test_negative_sharpness(self) -> None:
        """Test negative sharpness raise ValidationError."""
        sharpness_neg = np.array([1.0, -0.5, 1.0], dtype=np.float32)

        with pytest.raises(ValidationError, match="positive"):
            validate_sharpness_for_writing(sharpness_neg, n_points=3)


class TestValidateZarrAttributes:
    """Test validate_zarr_attributes function."""

    def test_valid_attributes(self) -> None:
        """Test valid attributes pass validation."""
        attrs = {
            "type": "points",
        }
        # Should not raise
        validate_zarr_attributes(attrs, is_root=False)

    def test_valid_attributes_with_optional(self) -> None:
        """Test valid attributes with optional rendering attributes."""
        attrs = {
            "type": "points",
            "opacity": 1.0,  # Optional, not validated by this function
            "gamma": 1.0,
            "blending_mode": "normal",
        }
        # Should not raise
        validate_zarr_attributes(attrs, is_root=False)

    def test_root_attributes_with_version(self) -> None:
        """Test root-specific attributes with version."""
        attrs = {
            "type": "scene",
            "luxar_version": "0.1",  # Use supported version format
        }
        # Should not raise
        validate_zarr_attributes(attrs, is_root=True)

    def test_root_missing_type(self) -> None:
        """Test root missing type attribute raises error."""
        attrs = {"luxar_version": "0.1.0"}

        with pytest.raises(ValidationError, match="type"):
            validate_zarr_attributes(attrs, is_root=True)

    def test_root_missing_version(self) -> None:
        """Test root missing version attribute raises error."""
        attrs = {"type": "scene"}

        with pytest.raises(ValidationError, match="luxar_version"):
            validate_zarr_attributes(attrs, is_root=True)

    def test_missing_type(self) -> None:
        """Test missing type attribute raises error."""
        attrs = {"opacity": 1.0}

        with pytest.raises(ValidationError, match="type"):
            validate_zarr_attributes(attrs, is_root=False)

    def test_invalid_type_value(self) -> None:
        """Test invalid type value raises error."""
        attrs = {"type": "invalid_type"}

        with pytest.raises(ValidationError, match="Invalid node type"):
            validate_zarr_attributes(attrs, is_root=False)

    def test_valid_types(self) -> None:
        """Test all valid node types."""
        for node_type in ["scene", "group", "points"]:
            attrs = {"type": node_type}
            # Should not raise for any valid type
            if node_type == "scene":
                attrs["luxar_version"] = "0.1"  # Use supported format
                validate_zarr_attributes(attrs, is_root=True)
            else:
                validate_zarr_attributes(attrs, is_root=False)

    def test_unsupported_version(self) -> None:
        """Test unsupported version raises error."""
        from luxar.typing_utils.config import SUPPORTED_VERSIONS

        # Use a version that's definitely not supported
        attrs = {
            "type": "scene",
            "luxar_version": "999.999.999",  # Invalid version
        }

        if "999.999.999" not in SUPPORTED_VERSIONS:
            with pytest.raises(ValidationError, match="Unsupported"):
                validate_zarr_attributes(attrs, is_root=True)

    def test_validation_error_has_suggestions(self) -> None:
        """Test that ValidationErrors include helpful suggestions."""
        attrs_missing_type = {}

        try:
            validate_zarr_attributes(attrs_missing_type, is_root=False)
        except ValidationError as e:
            assert "💡" in str(e) or "Suggestion" in str(e)
            assert "type" in str(e)
