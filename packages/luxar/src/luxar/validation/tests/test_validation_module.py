"""Tests for the validation module with helpful error messages."""

from typing import Any, cast

import numpy as np
import pytest

from luxar.validation import (
    ValidationError,
    validate_colors_for_writing,
    validate_positions_for_writing,
    validate_radii_for_writing,
    validate_sharpness_for_writing,
)


class TestValidationError:
    """Test the custom ValidationError class."""

    def test_validation_error_with_suggestion(self) -> None:
        """Test ValidationError with suggestion."""
        error = ValidationError("Something went wrong", "Try this instead")
        assert "Something went wrong" in str(error)
        assert "💡 Suggestion: Try this instead" in str(error)

    def test_validation_error_without_suggestion(self) -> None:
        """Test ValidationError without suggestion."""
        error = ValidationError("Something went wrong")
        assert "Something went wrong" in str(error)
        assert "Suggestion" not in str(error)


class TestPositionValidation:
    """Test position validation with helpful errors."""

    def test_valid_positions(self) -> None:
        """Test valid positions pass validation."""
        positions = np.random.randn(100, 3).astype(np.float32)
        n_points, n_dims = validate_positions_for_writing(positions)
        assert n_points == 100
        assert n_dims == 3

    def test_non_numpy_positions(self) -> None:
        """Test error for non-numpy positions."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positions_for_writing(cast(Any, [[1, 2, 3], [4, 5, 6]]))

        assert "Expected numpy array, got list" in str(exc_info.value)
        assert "np.array(data)" in str(exc_info.value)

    def test_1d_positions(self) -> None:
        """Test error for 1D positions."""
        positions = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValidationError) as exc_info:
            validate_positions_for_writing(positions)

        assert "Got 1D array with 5 elements" in str(exc_info.value)
        assert "reshape(-1, 1)" in str(exc_info.value)

    def test_3d_positions(self) -> None:
        """Test error for 3D positions."""
        positions = np.zeros((10, 10, 3))
        with pytest.raises(ValidationError) as exc_info:
            validate_positions_for_writing(positions)

        assert "Got 3D array" in str(exc_info.value)
        assert "split them into separate nodes" in str(exc_info.value)

    def test_empty_positions(self) -> None:
        """Test error for empty positions."""
        positions = np.zeros((0, 3), dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_positions_for_writing(positions)

        assert "Cannot write empty points" in str(exc_info.value)
        assert "at least one point" in str(exc_info.value)

    def test_zero_dimensions(self) -> None:
        """Test error for zero dimensions."""
        positions = np.zeros((10, 0), dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_positions_for_writing(positions)

        assert "Points have 0 dimensions" in str(exc_info.value)
        assert "at least 1 dimension" in str(exc_info.value)

    def test_high_dimensional_warning(self) -> None:
        """Test warning for high-dimensional data."""
        positions = np.random.randn(10, 15).astype(np.float32)
        with pytest.warns(UserWarning, match="15D points.*first 3 dimensions"):
            n_points, n_dims = validate_positions_for_writing(positions)
        assert n_dims == 15


class TestColorValidation:
    """Test color validation with helpful errors."""

    def test_valid_colors(self) -> None:
        """Test valid colors pass validation."""
        colors = np.random.rand(100, 3).astype(np.float32)
        validate_colors_for_writing(colors, 100)  # Should not raise

    def test_single_color_error(self) -> None:
        """Test error for single color when expecting per-point."""
        colors = np.array([1.0, 0.5, 0.0])
        with pytest.raises(ValidationError) as exc_info:
            validate_colors_for_writing(colors, 100)

        assert "Got single RGB color" in str(exc_info.value)
        assert "(1, 3) for broadcasting" in str(exc_info.value)

    def test_wrong_channel_count(self) -> None:
        """Test error for wrong number of color channels."""
        colors = np.random.rand(100, 4).astype(np.float32)  # RGBA instead of RGB
        with pytest.raises(ValidationError) as exc_info:
            validate_colors_for_writing(colors, 100)

        assert "must have 3 channels" in str(exc_info.value)
        assert "got 4 channels" in str(exc_info.value)

    def test_mismatched_count(self) -> None:
        """Test error for mismatched color count."""
        colors = np.random.rand(50, 3).astype(np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_colors_for_writing(colors, 100)

        assert "Number of colors (50)" in str(exc_info.value)
        assert "number of points (100)" in str(exc_info.value)

    def test_negative_colors(self) -> None:
        """Test error for negative color values."""
        colors = np.random.randn(100, 3).astype(np.float32)  # Can be negative
        colors[0, 0] = -0.5
        with pytest.raises(ValidationError) as exc_info:
            validate_colors_for_writing(colors, 100)

        assert "Colors cannot be negative" in str(exc_info.value)
        assert "np.clip(colors, 0, None)" in str(exc_info.value)

    def test_hdr_color_warning(self) -> None:
        """Test warning for extreme HDR values."""
        colors = np.random.rand(100, 3).astype(np.float32) * 20  # Very bright
        with pytest.warns(UserWarning, match="HDR colors.*maximum value"):
            validate_colors_for_writing(colors, 100)


class TestRadiiValidation:
    """Test radii validation with helpful errors."""

    def test_valid_radii(self) -> None:
        """Test valid radii pass validation."""
        radii = np.random.uniform(0.1, 2.0, 100).astype(np.float32)
        validate_radii_for_writing(radii, 100)  # Should not raise

    def test_2d_radii_error(self) -> None:
        """Test error for 2D radii array."""
        radii = np.random.uniform(0.1, 2.0, (100, 1)).astype(np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_radii_for_writing(radii, 100)

        assert "Got 2D array" in str(exc_info.value)
        assert "radii.ravel()" in str(exc_info.value)

    def test_mismatched_radii_count(self) -> None:
        """Test error for mismatched radii count."""
        radii = np.random.uniform(0.1, 2.0, 50).astype(np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_radii_for_writing(radii, 100)

        assert "Number of radii (50)" in str(exc_info.value)
        assert "number of points (100)" in str(exc_info.value)

    def test_zero_radii(self) -> None:
        """Test error for zero radii."""
        radii = np.ones(100, dtype=np.float32)
        radii[0] = 0
        with pytest.raises(ValidationError) as exc_info:
            validate_radii_for_writing(radii, 100)

        assert "must be positive" in str(exc_info.value)
        assert "Found zero values" in str(exc_info.value)
        assert "radii[radii == 0] = 0.01" in str(exc_info.value)

    def test_negative_radii(self) -> None:
        """Test error for negative radii."""
        radii = np.ones(100, dtype=np.float32)
        radii[0] = -0.5
        with pytest.raises(ValidationError) as exc_info:
            validate_radii_for_writing(radii, 100)

        assert "must be positive" in str(exc_info.value)
        assert "Found minimum value: -0.500" in str(exc_info.value)
        assert "np.abs(radii)" in str(exc_info.value)


class TestSharpnessValidation:
    """Test sharpness validation with helpful errors."""

    def test_valid_sharpness(self) -> None:
        """Test valid sharpness passes validation."""
        sharpness = np.random.uniform(0.5, 10.0, 100).astype(np.float32)
        validate_sharpness_for_writing(sharpness, 100)  # Should not raise

    def test_2d_sharpness_error(self) -> None:
        """Test error for 2D sharpness array."""
        sharpness = np.random.uniform(0.5, 10.0, (100, 1)).astype(np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_sharpness_for_writing(sharpness, 100)

        assert "Got 2D array" in str(exc_info.value)
        assert "sharpness.ravel()" in str(exc_info.value)

    def test_mismatched_sharpness_count(self) -> None:
        """Test error for mismatched sharpness count."""
        sharpness = np.random.uniform(0.5, 10.0, 50).astype(np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_sharpness_for_writing(sharpness, 100)

        assert "Number of sharpness values (50)" in str(exc_info.value)
        assert "number of points (100)" in str(exc_info.value)

    def test_negative_sharpness(self) -> None:
        """Test error for negative sharpness."""
        sharpness = np.ones(100, dtype=np.float32)
        sharpness[0] = -1.0
        with pytest.raises(ValidationError) as exc_info:
            validate_sharpness_for_writing(sharpness, 100)

        assert "must be positive" in str(exc_info.value)
        assert "Found minimum value: -1.000" in str(exc_info.value)
        assert "between 0.001 and 31.0" in str(
            exc_info.value
        )  # Matches SHARPNESS_MIN constant

    def test_out_of_range_sharpness_warning(self) -> None:
        """Test warning for out-of-range sharpness values."""
        sharpness = np.array([0.1, 5.0, 12.0] * 33 + [5.0], dtype=np.float32)
        with pytest.warns(UserWarning, match="Using extreme sharpness values"):
            validate_sharpness_for_writing(sharpness, 100)


class TestContextParameter:
    """Test that context parameter provides better error messages."""

    def test_positions_with_context(self) -> None:
        """Test position validation with custom context."""
        positions = np.array([1, 2, 3])
        with pytest.raises(ValidationError) as exc_info:
            validate_positions_for_writing(positions, context="batch positions")

        assert "batch positions:" in str(exc_info.value)

    def test_colors_with_context(self) -> None:
        """Test color validation with custom context."""
        colors = np.array([1.0, 0.5, 0.0])
        with pytest.raises(ValidationError) as exc_info:
            validate_colors_for_writing(colors, 100, context="batch colors")

        assert "batch colors:" in str(exc_info.value)

    def test_radii_with_context(self) -> None:
        """Test radii validation with custom context."""
        radii = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_radii_for_writing(radii, 100, context="point radii")

        assert "point radii:" in str(exc_info.value)

    def test_sharpness_with_context(self) -> None:
        """Test sharpness validation with custom context."""
        sharpness = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_sharpness_for_writing(sharpness, 100, context="point sharpness")

        assert "point sharpness:" in str(exc_info.value)
