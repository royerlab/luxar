"""Tests for `luxar.validation.base`."""

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
from luxar.validation.base import validate_zarr_attributes


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

    # [Python-R2/D-C3] Pin the documented contract: validate_positions_
    # for_writing does NOT check dtype. Float64 / int32 / etc. all pass
    # shape + finiteness validation; the validator returns only
    # (n_points, n_dims) and explicitly leaves dtype conversion to the
    # caller (downstream writers use ensure_float32 separately).
    # A regression that added dtype-strict validation here would silently
    # reject legitimate float64 inputs from upstream callers and would
    # break this regression-lock test.
    def test_validate_positions_dtype_not_checked(self) -> None:
        """The validator accepts non-float32 dtypes — dtype conversion
        is the caller's responsibility (see docstring + ensure_float32).
        """
        for dtype in [np.float64, np.float32, np.int32, np.int64]:
            positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
            n_points, n_dims = validate_positions_for_writing(positions)
            assert n_points == 2
            assert n_dims == 3


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

    # [Python-R4 / D-W3] Pin the HDR-warning BOUNDARY explicitly. The
    # validator warns when `max_val > 10.0` (strict). A mutation that
    # flipped to `>= 10.0` would trigger spurious warnings on legitimate
    # max-HDR data; a mutation that loosened to `> 100.0` would silence
    # the warning on real overflow. Test both sides of the boundary.
    def test_hdr_warning_silent_at_exactly_10(self) -> None:
        """max=10.0 must NOT warn (the threshold is strict `> 10.0`)."""
        colors = np.full((10, 3), 10.0, dtype=np.float32)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning would fail
            validate_colors_for_writing(colors, 10)

    def test_hdr_warning_fires_just_above_10(self) -> None:
        """max just-above 10.0 (10.001) MUST warn."""
        colors = np.full((10, 3), 10.0, dtype=np.float32)
        colors[0, 0] = 10.001
        with pytest.warns(UserWarning, match="HDR colors.*maximum value"):
            validate_colors_for_writing(colors, 10)

    def test_hdr_warning_does_not_fire_for_integer_dtypes(self) -> None:
        """Integer color arrays are SDR storage in their native integer
        range; the HDR warning is float-only by contract."""
        colors = np.full((10, 3), 255, dtype=np.uint8)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_colors_for_writing(colors, 10)


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
        assert "Found zero or negative values" in str(exc_info.value)
        assert "minimum: 0.000" in str(exc_info.value)
        assert "np.clip(radii, 0.01, None)" in str(exc_info.value)

    def test_negative_radii(self) -> None:
        """Test error for negative radii."""
        radii = np.ones(100, dtype=np.float32)
        radii[0] = -0.5
        with pytest.raises(ValidationError) as exc_info:
            validate_radii_for_writing(radii, 100)

        assert "must be positive" in str(exc_info.value)
        assert "Found zero or negative values" in str(exc_info.value)
        assert "minimum: -0.500" in str(exc_info.value)
        assert "np.clip(radii, 0.01, None)" in str(exc_info.value)


class TestSharpnessValidation:
    """Test sharpness validation with helpful errors."""

    def test_valid_sharpness(self) -> None:
        """Test valid sharpness passes validation."""
        sharpness = np.random.uniform(0.0, 1.0, 100).astype(np.float32)
        validate_sharpness_for_writing(sharpness, 100)  # Should not raise

    def test_2d_sharpness_error(self) -> None:
        """Test error for 2D sharpness array."""
        sharpness = np.random.uniform(0.0, 1.0, (100, 1)).astype(np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_sharpness_for_writing(sharpness, 100)

        assert "Got 2D array" in str(exc_info.value)
        assert "sharpness.ravel()" in str(exc_info.value)

    def test_mismatched_sharpness_count(self) -> None:
        """Test error for mismatched sharpness count."""
        sharpness = np.random.uniform(0.0, 1.0, 50).astype(np.float32)
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

        assert "must be >= 0.0" in str(exc_info.value)
        assert "Found minimum value: -1.000" in str(exc_info.value)
        assert "between 0.0 and 1.0" in str(
            exc_info.value
        )  # Matches SHARPNESS_MIN/MAX constants

    def test_out_of_range_sharpness_rejected(self) -> None:
        """Sharpness above the [0, 1] range raises a ValidationError."""
        sharpness = np.array([0.1, 0.5, 12.0] * 33 + [0.5], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_sharpness_for_writing(sharpness, 100)

        assert "exceed maximum" in str(exc_info.value)
        assert "Found maximum: 12.000" in str(exc_info.value)


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
        sharpness = np.full(100, -1.0, dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            validate_sharpness_for_writing(sharpness, 100, context="point sharpness")

        assert "point sharpness:" in str(exc_info.value)


class TestValidationErrorExtended:
    """Additional ValidationError tests merged from test_base_validation.py."""

    def test_validation_error_can_be_caught_as_value_error(self) -> None:
        """Test that ValidationError is a ValueError subclass."""
        with pytest.raises(ValueError):
            raise ValidationError("test error")

    def test_validation_error_preserves_suggestion(self) -> None:
        """Test that suggestion is accessible in error message."""
        try:
            raise ValidationError("problem", "try this fix")
        except ValidationError as e:
            assert "problem" in str(e)
            assert "try this fix" in str(e)


class TestValidateZarrAttributes:
    """Test validate_zarr_attributes function (merged from test_base_validation.py)."""

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
            "opacity": 1.0,
            "gamma": 1.0,
            "blending_mode": "normal",
        }
        # Should not raise
        validate_zarr_attributes(attrs, is_root=False)

    def test_root_attributes_with_version(self) -> None:
        """Test root-specific attributes with version."""
        attrs = {
            "type": "scene",
            "luxar_version": "0.1",
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
        for node_type in ["scene", "group", "points", "lines", "gsplats"]:
            attrs = {"type": node_type}
            if node_type == "scene":
                attrs["luxar_version"] = "0.1"
                validate_zarr_attributes(attrs, is_root=True)
            else:
                validate_zarr_attributes(attrs, is_root=False)

    def test_unsupported_version(self) -> None:
        """Test unsupported version raises error."""
        from luxar.typing_utils.config import SUPPORTED_VERSIONS

        attrs = {
            "type": "scene",
            "luxar_version": "999.999.999",
        }

        if "999.999.999" not in SUPPORTED_VERSIONS:
            with pytest.raises(ValidationError, match="Unsupported"):
                validate_zarr_attributes(attrs, is_root=True)

    def test_validation_error_has_suggestions(self) -> None:
        """Test that ValidationErrors include helpful suggestions."""
        attrs_missing_type: dict[str, object] = {}

        with pytest.raises(ValidationError, match="type"):
            validate_zarr_attributes(attrs_missing_type, is_root=False)
