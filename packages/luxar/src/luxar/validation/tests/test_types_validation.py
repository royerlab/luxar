"""Test validation functions in types.py module."""

from typing import Any, cast

import numpy as np
import pytest

from luxar.validation import validate_categories
from luxar.validation.types import (
    is_color_array,
    is_position_array,
    is_transform_matrix,
    validate_absorption,
    validate_blending_mode,
    validate_category_indices,
    validate_colors,
    validate_gamma,
    validate_intensity,
    validate_layer,
    validate_node_type,
    validate_offset,
    validate_opacity,
    validate_physical_unit,
    validate_positions,
    validate_radii,
    validate_sharpness,
    validate_transform,
)


class TestTransformValidation:
    """Test validate_transform function."""

    def test_valid_transform(self) -> None:
        """Test that a valid 4x4 transform matrix is accepted."""
        transform = np.eye(4, dtype=np.float32)
        result = validate_transform(transform)
        assert result.shape == (4, 4)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, transform)

    def test_transform_wrong_type(self) -> None:
        """Test that non-numpy array input raises ValueError."""
        with pytest.raises(ValueError, match="Transform must be a numpy array"):
            validate_transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def test_transform_wrong_shape(self) -> None:
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

    def test_transform_nan_rejected(self) -> None:
        """Test that transform with NaN values raises ValueError."""
        transform = np.eye(4, dtype=np.float32)
        transform[1, 2] = np.nan
        with pytest.raises(ValueError, match="Transform contains NaN"):
            validate_transform(transform)

    def test_transform_inf_rejected(self) -> None:
        """Test that transform with Inf values raises ValueError."""
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = np.inf
        with pytest.raises(ValueError, match="Transform contains Inf"):
            validate_transform(transform)

        transform_neg_inf = np.eye(4, dtype=np.float32)
        transform_neg_inf[2, 2] = -np.inf
        with pytest.raises(ValueError, match="Transform contains Inf"):
            validate_transform(transform_neg_inf)

    def test_transform_dtype_conversion(self) -> None:
        """Test that transform is converted to float32."""
        # Test float64 conversion
        transform_f64 = np.eye(4, dtype=np.float64)
        result = validate_transform(transform_f64)
        assert result.dtype == np.float32

        # Test int conversion
        transform_int = np.eye(4, dtype=np.int32)
        result = validate_transform(transform_int)
        assert result.dtype == np.float32

    @pytest.mark.parametrize(
        "bottom_row,test_id",
        [
            ([1, 0, 0, 2], "row_has_nonzero_xyz_with_nonunit_w"),
            ([0, 0, 0, 0], "all_zero_bottom_row"),
            ([0, 0, 0, 2], "scaled_w"),
            ([0, 0, 1, 1], "z_translation_in_bottom_row"),
            ([0.5, 0, 0, 1], "small_nonzero_x_in_bottom_row"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_transform_non_affine_bottom_row_rejected(
        self, bottom_row, test_id
    ) -> None:
        """Audit G7: validate_transform checks `bottom == [0,0,0,1]` but
        there was no test exercising the rejection path. Pin every
        non-affine variant so a mutation that drops the check (or
        loosens the atol) surfaces immediately.
        """
        transform = np.eye(4, dtype=np.float32)
        transform[3, :] = bottom_row
        with pytest.raises(ValueError, match="Invalid affine transform"):
            validate_transform(transform)


class TestNodeTypeValidation:
    """Test validate_node_type function."""

    def test_valid_node_types(self) -> None:
        """Test that valid node types are accepted."""
        assert validate_node_type("points") == "points"
        assert validate_node_type("group") == "group"
        assert validate_node_type("scene") == "scene"

    def test_invalid_node_type(self) -> None:
        """Test that invalid node types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid node type 'invalid'"):
            validate_node_type("invalid")

        with pytest.raises(ValueError, match="Invalid node type 'mesh'"):
            validate_node_type("mesh")

        with pytest.raises(ValueError, match="Invalid node type ''"):
            validate_node_type("")


class TestPhysicalUnitValidation:
    """Test validate_physical_unit function."""

    def test_valid_units(self) -> None:
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

    def test_invalid_unit(self) -> None:
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

    def test_valid_opacity(self) -> None:
        """Test that valid opacity values are accepted."""
        assert validate_opacity(0.0) == 0.0
        assert validate_opacity(0.5) == 0.5
        assert validate_opacity(1.0) == 1.0
        assert validate_opacity(0.25) == 0.25
        assert validate_opacity(0.99) == 0.99

    def test_opacity_type_conversion(self) -> None:
        """Test that opacity values are converted to float."""
        assert validate_opacity(1) == 1.0
        assert validate_opacity(0) == 0.0
        assert validate_opacity("0.5") == 0.5
        assert validate_opacity(np.float32(0.7)) == pytest.approx(0.7)

    def test_invalid_opacity(self) -> None:
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

    def test_valid_gamma(self) -> None:
        """Test that valid gamma values are accepted."""
        assert validate_gamma(0.2) == 0.2
        assert validate_gamma(1.0) == 1.0
        assert validate_gamma(2.0) == 2.0
        assert validate_gamma(0.5) == 0.5
        assert validate_gamma(1.5) == 1.5

    def test_gamma_type_conversion(self) -> None:
        """Test that gamma values are converted to float."""
        assert validate_gamma(1) == 1.0
        assert validate_gamma(2) == 2.0
        assert validate_gamma("1.5") == 1.5
        assert validate_gamma(np.float32(1.8)) == pytest.approx(1.8)

    def test_invalid_gamma(self) -> None:
        """Test that invalid gamma values raise ValueError.

        Note: GAMMA range is now [0.1, 10.0] per spec (symmetric: gamma and 1/gamma have equal range).
        """
        with pytest.raises(ValueError, match="Gamma must be between 0.1 and 10.0"):
            validate_gamma(0.0)

        with pytest.raises(ValueError, match="Gamma must be between 0.1 and 10.0"):
            validate_gamma(0.05)

        with pytest.raises(ValueError, match="Gamma must be between 0.1 and 10.0"):
            validate_gamma(15.0)

        with pytest.raises(ValueError, match="Gamma must be between 0.1 and 10.0"):
            validate_gamma(100.0)

        with pytest.raises(TypeError, match="Gamma must be convertible to float"):
            validate_gamma("not_a_number")


class TestAbsorptionValidation:
    """Test validate_absorption function (volumetric kappa)."""

    def test_valid_absorption(self) -> None:
        """Valid absorption values: 0 (additive limit), identity, large."""
        assert validate_absorption(0.0) == 0.0
        assert validate_absorption(1.0) == 1.0
        assert validate_absorption(10) == 10.0
        assert validate_absorption("2.5") == 2.5
        assert validate_absorption(1e6) == 1e6  # no upper bound
        assert validate_absorption(np.float32(0.5)) == pytest.approx(0.5)

    def test_invalid_absorption(self) -> None:
        """Negative, NaN, and infinite absorption are rejected."""
        with pytest.raises(ValueError, match="Absorption must be >= 0"):
            validate_absorption(-0.1)

        with pytest.raises(ValueError, match="Absorption must be finite"):
            validate_absorption(float("nan"))

        with pytest.raises(ValueError, match="Absorption must be finite"):
            validate_absorption(float("inf"))

        with pytest.raises(TypeError, match="Absorption must be convertible to float"):
            validate_absorption("not_a_number")

        with pytest.raises(TypeError, match="Absorption must be convertible to float"):
            validate_absorption(None)


class TestBlendingModeValidation:
    """Test validate_blending_mode function."""

    def test_valid_blending_modes(self) -> None:
        """Test that valid blending modes are accepted."""
        assert validate_blending_mode("normal") == "normal"
        assert validate_blending_mode("additive") == "additive"
        assert validate_blending_mode("max") == "max"
        assert validate_blending_mode("opaque") == "opaque"
        assert validate_blending_mode("luminous") == "luminous"
        assert validate_blending_mode("volumetric") == "volumetric"

    def test_invalid_blending_mode(self) -> None:
        """Test that invalid blending modes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid blending mode 'overlay'"):
            validate_blending_mode("overlay")

        with pytest.raises(ValueError, match="Invalid blending mode 'screen'"):
            validate_blending_mode("screen")

        with pytest.raises(ValueError, match="Invalid blending mode ''"):
            validate_blending_mode("")

        with pytest.raises(ValueError, match="Invalid blending mode 'NORMAL'"):
            validate_blending_mode("NORMAL")  # Case sensitive

    def test_blending_mode_type_error(self) -> None:
        """Test that non-string blending mode raises TypeError."""
        with pytest.raises(TypeError, match="Blending mode must be a string"):
            validate_blending_mode(123)

    def test_error_message_lists_all_enum_modes(self) -> None:
        """The error message enumerates every BlendingMode value.

        Pins the enum derivation: adding a 6th mode to the enum must make
        the validator accept it AND advertise it, with no second list to
        keep in sync.
        """
        from luxar.typing_utils.enums import BlendingMode

        with pytest.raises(ValueError) as exc_info:
            validate_blending_mode("bogus")
        message = str(exc_info.value)
        for mode in BlendingMode:
            assert mode.value in message

    def test_validator_accepts_every_enum_mode(self) -> None:
        """Every BlendingMode enum value passes validation (enum is the SSOT)."""
        from luxar.typing_utils.enums import BlendingMode

        for mode in BlendingMode:
            assert validate_blending_mode(mode.value) == mode.value

        with pytest.raises(TypeError, match="Blending mode must be a string"):
            validate_blending_mode(None)

        with pytest.raises(TypeError, match="Blending mode must be a string"):
            validate_blending_mode(["normal"])


class TestPositionsValidation:
    """Test validate_positions function."""

    def test_valid_positions_2d(self) -> None:
        """Test that valid 2D positions are accepted."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        result = validate_positions(positions)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, positions)

    def test_valid_positions_3d(self) -> None:
        """Test that valid 3D positions are accepted."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        result = validate_positions(positions)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    def test_valid_positions_high_dimensional(self) -> None:
        """Test that high-dimensional positions are accepted."""
        # 5D positions
        positions = np.random.rand(10, 5).astype(np.float32)
        result = validate_positions(positions)
        assert result.shape == (10, 5)
        assert result.dtype == np.float32

    def test_positions_dtype_conversion(self) -> None:
        """Test that positions are converted to float32."""
        # float64 to float32
        positions_f64 = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        result = validate_positions(positions_f64)
        assert result.dtype == np.float32

        # int to float32
        positions_int = np.array([[0, 1], [2, 3]], dtype=np.int32)
        result = validate_positions(positions_int)
        assert result.dtype == np.float32

    def test_positions_with_ndim_parameter(self) -> None:
        """Test ndim parameter validation."""
        positions_2d = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        positions_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        # Correct ndim
        result = validate_positions(positions_2d, ndim=2)
        assert result.shape == (2, 2)

        result = validate_positions(positions_3d, ndim=3)
        assert result.shape == (2, 3)

        # Incorrect ndim
        with pytest.raises(ValueError, match="Expected 3 dimensions, got 2"):
            validate_positions(positions_2d, ndim=3)

        with pytest.raises(ValueError, match="Expected 2 dimensions, got 3"):
            validate_positions(positions_3d, ndim=2)

    def test_positions_not_numpy_array(self) -> None:
        """Test that non-numpy array raises ValueError."""
        with pytest.raises(ValueError, match="Positions must be a numpy array"):
            validate_positions([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="Positions must be a numpy array"):
            validate_positions([0.0, 1.0, 2.0])

    def test_positions_wrong_ndim(self) -> None:
        """Test that wrong number of dimensions raises ValueError."""
        # 1D array
        positions_1d = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="Positions must have shape \\(N, D\\)"):
            validate_positions(positions_1d)

        # 3D array
        positions_3d = np.random.rand(5, 3, 2)
        with pytest.raises(ValueError, match="Positions must have shape \\(N, D\\)"):
            validate_positions(positions_3d)

    def test_positions_zero_dimensions(self) -> None:
        """Test that positions with 0 dimensions raises ValueError."""
        # This is an edge case - positions with shape (N, 0)
        positions_empty = np.empty((5, 0), dtype=np.float32)
        with pytest.raises(
            ValueError, match="Positions must have at least 1 dimension"
        ):
            validate_positions(positions_empty)


class TestColorsValidation:
    """Test validate_colors function."""

    def test_valid_colors(self) -> None:
        """Test that valid colors are accepted."""
        n_points = 10
        colors = np.random.rand(n_points, 3).astype(np.float32)
        result = validate_colors(colors, n_points)
        assert result.shape == (n_points, 3)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, colors)

    def test_colors_dtype_conversion(self) -> None:
        """Test that colors are converted to float32."""
        n_points = 5
        # float64 to float32
        colors_f64 = np.random.rand(n_points, 3)
        result = validate_colors(colors_f64, n_points)
        assert result.dtype == np.float32

        # int to float32 (e.g., 0-255 RGB)
        colors_int = np.array([[255, 128, 0], [0, 255, 128]], dtype=np.uint8)
        result = validate_colors(colors_int, 2)
        assert result.dtype == np.float32

    def test_colors_hdr_values(self) -> None:
        """Test that HDR color values (>1.0) are accepted."""
        n_points = 3
        # HDR colors with values > 1.0
        colors_hdr = np.array(
            [[1.0, 2.0, 3.0], [5.0, 10.0, 0.5], [0.1, 0.2, 20.0]], dtype=np.float32
        )
        result = validate_colors(colors_hdr, n_points)
        assert result.shape == (n_points, 3)
        np.testing.assert_array_equal(result, colors_hdr)

    def test_colors_not_numpy_array(self) -> None:
        """Test that non-numpy array raises ValueError."""
        with pytest.raises(ValueError, match="Colors must be a numpy array"):
            validate_colors([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 2)

        with pytest.raises(ValueError, match="Colors must be a numpy array"):
            validate_colors([1.0, 0.0, 0.0], 1)

    def test_colors_wrong_shape(self) -> None:
        """Test that wrong shape raises ValueError."""
        n_points = 5
        # Wrong number of points
        colors_wrong_n = np.random.rand(10, 3).astype(np.float32)
        with pytest.raises(
            ValueError, match="Colors must have shape \\(5, 3\\) or \\(5, 4\\)"
        ):
            validate_colors(colors_wrong_n, n_points)

        # RGBA is accepted by default (alpha = per-element opacity)
        colors_rgba = np.random.rand(n_points, 4).astype(np.float32)
        assert validate_colors(colors_rgba, n_points).shape == (n_points, 4)

        # …but rejected when the caller restricts channels (points/lines
        # until their volumetric phases land)
        with pytest.raises(ValueError, match="Colors must have shape \\(5, 3\\)"):
            validate_colors(colors_rgba, n_points, channels=(3,))

        # Wrong number of channels (5 is never valid)
        colors_wide = np.random.rand(n_points, 5).astype(np.float32)
        with pytest.raises(ValueError, match="Colors must have shape"):
            validate_colors(colors_wide, n_points)

        # Alpha bounds: opacity must be finite and within [0, 1]
        bad_alpha = np.random.rand(n_points, 4).astype(np.float32)
        bad_alpha[2, 3] = 1.5
        with pytest.raises(ValueError, match="alpha channel"):
            validate_colors(bad_alpha, n_points)
        nan_alpha = np.random.rand(n_points, 4).astype(np.float32)
        nan_alpha[1, 3] = np.nan
        with pytest.raises(ValueError, match="alpha channel"):
            validate_colors(nan_alpha, n_points)

        # 1D array
        colors_1d = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Colors must have shape \\(1, 3\\)"):
            validate_colors(colors_1d, 1)


class TestRadiiValidation:
    """Test validate_radii function."""

    def test_valid_radii(self) -> None:
        """Test that valid radii are accepted."""
        n_points = 10
        radii = np.ones(n_points, dtype=np.float32) * 0.5
        result = validate_radii(radii, n_points)
        assert result.shape == (n_points,)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, radii)

    def test_radii_dtype_conversion(self) -> None:
        """Test that radii are converted to float32."""
        n_points = 5
        # float64 to float32
        radii_f64 = np.ones(n_points, dtype=np.float64)
        result = validate_radii(radii_f64, n_points)
        assert result.dtype == np.float32

        # int to float32
        radii_int = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = validate_radii(radii_int, n_points)
        assert result.dtype == np.float32

    def test_radii_not_numpy_array(self) -> None:
        """Test that non-numpy array raises ValueError."""
        with pytest.raises(ValueError, match="Radii must be a numpy array"):
            validate_radii([1.0, 2.0, 3.0], 3)

        with pytest.raises(ValueError, match="Radii must be a numpy array"):
            validate_radii(1.0, 1)

    def test_radii_wrong_ndim(self) -> None:
        """Test that wrong number of dimensions raises ValueError."""
        n_points = 5
        # 2D array
        radii_2d = np.ones((n_points, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Radii must have shape \\(N,\\)"):
            validate_radii(radii_2d, n_points)

        # 0D (scalar)
        radii_scalar = np.array(1.0)
        with pytest.raises(ValueError, match="Radii must have shape \\(N,\\)"):
            validate_radii(radii_scalar, 1)

    def test_radii_wrong_length(self) -> None:
        """Test that wrong length raises ValueError."""
        n_points = 5
        radii_wrong_length = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="Radii shape .* doesn't match positions"):
            validate_radii(radii_wrong_length, n_points)

    def test_radii_negative_values(self) -> None:
        """Test that negative radii raise ValueError."""
        n_points = 5
        radii_negative = np.array([1.0, 2.0, -0.5, 1.0, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="All radii must be positive values"):
            validate_radii(radii_negative, n_points)

    def test_radii_zero_values(self) -> None:
        """Test that zero radii raise ValueError."""
        n_points = 3
        radii_with_zero = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="All radii must be positive values"):
            validate_radii(radii_with_zero, n_points)

    def test_radii_all_negative(self) -> None:
        """Test that all negative radii raise ValueError."""
        n_points = 5
        radii_all_negative = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
        with pytest.raises(ValueError, match="All radii must be positive values"):
            validate_radii(radii_all_negative, n_points)

    def test_radii_broadcast_single_value(self) -> None:
        """Test that single-element radii array is accepted (broadcast support)."""
        radii = np.array([1.5], dtype=np.float32)
        n_points = 100
        result = validate_radii(radii, n_points)
        assert result.shape == (1,)
        assert result[0] == 1.5
        assert result.dtype == np.float32


class TestSharpnessValidation:
    """Test validate_sharpness function."""

    def test_valid_sharpness(self) -> None:
        """Test that valid sharpness values are accepted."""
        n_points = 10
        sharpness = np.ones(n_points, dtype=np.float32) * 0.5
        result = validate_sharpness(sharpness, n_points)
        assert result.shape == (n_points,)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, sharpness)

    def test_sharpness_dtype_conversion(self) -> None:
        """Test that sharpness values are converted to float32."""
        n_points = 5
        # float64 to float32
        sharpness_f64 = np.ones(n_points, dtype=np.float64) * 0.6
        result = validate_sharpness(sharpness_f64, n_points)
        assert result.dtype == np.float32

        # int to float32 (all in [0, 1])
        sharpness_int = np.array([0, 1, 1, 0, 1], dtype=np.int32)
        result = validate_sharpness(sharpness_int, n_points)
        assert result.dtype == np.float32

    def test_sharpness_typical_range(self) -> None:
        """Test sharpness across the full normalized [0, 1] range is accepted."""
        n_points = 5
        sharpness = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        result = validate_sharpness(sharpness, n_points)
        assert result.shape == (n_points,)

    def test_sharpness_out_of_range_rejected(self) -> None:
        """Sharpness outside the normalized [0, 1] range raises ValueError."""
        n_points = 3

        # Below the minimum -> rejected
        sharpness_low = np.array([-0.1, 0.2, 0.3], dtype=np.float32)
        with pytest.raises(ValueError, match="must be in"):
            validate_sharpness(sharpness_low, n_points)

        # Above the maximum -> rejected
        sharpness_high = np.array([0.5, 2.0, 30.0], dtype=np.float32)
        with pytest.raises(ValueError, match="must be in"):
            validate_sharpness(sharpness_high, n_points)

    def test_sharpness_not_numpy_array(self) -> None:
        """Test that non-numpy array raises ValueError."""
        with pytest.raises(ValueError, match="Sharpness must be a numpy array"):
            validate_sharpness([1.0, 2.0, 3.0], 3)

        with pytest.raises(ValueError, match="Sharpness must be a numpy array"):
            validate_sharpness(1.0, 1)

    def test_sharpness_wrong_ndim(self) -> None:
        """Test that wrong number of dimensions raises ValueError."""
        n_points = 5
        # 2D array
        sharpness_2d = np.ones((n_points, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Sharpness must have shape \\(N,\\)"):
            validate_sharpness(sharpness_2d, n_points)

        # 0D (scalar)
        sharpness_scalar = np.array(1.0)
        with pytest.raises(ValueError, match="Sharpness must have shape \\(N,\\)"):
            validate_sharpness(sharpness_scalar, 1)

    def test_sharpness_wrong_length(self) -> None:
        """Test that wrong length raises ValueError."""
        n_points = 5
        sharpness_wrong_length = np.ones(10, dtype=np.float32)
        with pytest.raises(
            ValueError, match="Sharpness shape .* doesn't match positions"
        ):
            validate_sharpness(sharpness_wrong_length, n_points)

    def test_sharpness_broadcast_single_value(self) -> None:
        """Test that single-element sharpness array is accepted (broadcast support)."""
        sharpness = np.array([0.6], dtype=np.float32)
        n_points = 100
        result = validate_sharpness(sharpness, n_points)
        assert result.shape == (1,)
        assert result[0] == 0.6
        assert result.dtype == np.float32

    def test_sharpness_negative_values(self) -> None:
        """Test that negative sharpness values raise ValueError."""
        n_points = 5
        sharpness_negative = np.array([0.5, 0.8, -0.5, 0.5, 0.5], dtype=np.float32)
        with pytest.raises(ValueError, match="must be in"):
            validate_sharpness(sharpness_negative, n_points)

    def test_sharpness_zero_values(self) -> None:
        """Test that zero sharpness (the lower bound) is accepted."""
        n_points = 3
        sharpness_with_zero = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        result = validate_sharpness(sharpness_with_zero, n_points)
        assert result.shape == (n_points,)


class TestTypeGuards:
    """Test type guard functions (is_* functions)."""

    def test_is_position_array_valid(self) -> None:
        """Test that valid position arrays return True."""
        positions_2d = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        assert is_position_array(positions_2d) is True

        positions_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        assert is_position_array(positions_3d) is True

        positions_5d = np.random.rand(10, 5).astype(np.float32)
        assert is_position_array(positions_5d) is True

    def test_is_position_array_invalid(self) -> None:
        """Test that invalid position arrays return False."""
        # Not a numpy array
        assert is_position_array([[0.0, 0.0], [1.0, 1.0]]) is False

        # Wrong ndim (1D)
        assert is_position_array(np.array([0.0, 1.0, 2.0])) is False

        # Wrong ndim (3D)
        assert is_position_array(np.random.rand(5, 3, 2)) is False

        # Empty dimensions
        assert is_position_array(np.empty((5, 0))) is False

    def test_is_color_array_valid(self) -> None:
        """Test that valid color arrays return True."""
        n_points = 10
        colors = np.random.rand(n_points, 3).astype(np.float32)
        assert is_color_array(colors, n_points) is True

        # HDR colors
        colors_hdr = np.array([[5.0, 10.0, 2.0]], dtype=np.float32)
        assert is_color_array(colors_hdr, 1) is True

    def test_is_color_array_invalid(self) -> None:
        """Test that invalid color arrays return False."""
        n_points = 5
        # Not a numpy array
        assert is_color_array([[1.0, 0.0, 0.0]], n_points) is False

        # Wrong shape
        colors_wrong_shape = np.random.rand(10, 3).astype(np.float32)
        assert is_color_array(colors_wrong_shape, n_points) is False

        # RGBA is a valid color array (alpha = per-element opacity)
        colors_rgba = np.random.rand(n_points, 4).astype(np.float32)
        assert is_color_array(colors_rgba, n_points) is True

        # Wrong number of channels
        colors_wide = np.random.rand(n_points, 5).astype(np.float32)
        assert is_color_array(colors_wide, n_points) is False

    def test_is_transform_matrix_valid(self) -> None:
        """Test that valid transform matrices return True."""
        transform = np.eye(4, dtype=np.float32)
        assert is_transform_matrix(transform) is True

        transform_custom = np.random.rand(4, 4).astype(np.float32)
        transform_custom[3, :] = [0, 0, 0, 1]  # Valid affine bottom row
        assert is_transform_matrix(transform_custom) is True

    def test_is_transform_matrix_invalid(self) -> None:
        """Test that invalid transform matrices return False."""
        # Not a numpy array
        assert (
            is_transform_matrix(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            is False
        )

        # Wrong shape (3x3)
        assert is_transform_matrix(np.eye(3)) is False

        # Wrong shape (4x3)
        assert is_transform_matrix(np.ones((4, 3))) is False

        # Wrong ndim (1D)
        assert is_transform_matrix(np.array([1, 2, 3, 4])) is False


class TestCategoriesValidation:
    """Test validate_categories function."""

    def test_valid_categories(self) -> None:
        """Test that valid category lists are accepted."""
        # Simple categories
        result = validate_categories(["DAPI", "GFP", "mCherry"])
        assert result == ["DAPI", "GFP", "mCherry"]

        # Single category
        result = validate_categories(["Channel1"])
        assert result == ["Channel1"]

        # Many categories
        categories = [f"cat_{i}" for i in range(100)]
        result = validate_categories(categories)
        assert result is not None
        assert len(result) == 100

    def test_none_categories(self) -> None:
        """Test that None is accepted (non-categorical dimension)."""
        result = validate_categories(None)
        assert result is None

    def test_empty_categories_rejected(self) -> None:
        """Test that empty list is rejected."""
        with pytest.raises(ValueError, match="must have at least 1 element"):
            validate_categories([])

    def test_non_list_rejected(self) -> None:
        """Test that non-list types are rejected."""
        with pytest.raises(TypeError, match="must be a list or None"):
            validate_categories(("a", "b", "c"))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a list or None"):
            validate_categories({"a", "b", "c"})  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a list or None"):
            validate_categories("abc")  # type: ignore[arg-type]

    def test_empty_string_category_rejected(self) -> None:
        """Test that empty string categories are rejected."""
        with pytest.raises(ValueError, match="index 1 is empty string"):
            validate_categories(["DAPI", "", "GFP"])

    def test_non_string_category_rejected(self) -> None:
        """Test that non-string categories are rejected."""
        with pytest.raises(TypeError, match="must be a string"):
            validate_categories(cast(Any, ["DAPI", 123, "GFP"]))

        with pytest.raises(TypeError, match="must be a string"):
            validate_categories(cast(Any, [None, "GFP"]))

    def test_duplicate_categories_rejected(self) -> None:
        """Test that duplicate category names are rejected."""
        with pytest.raises(
            ValueError, match="duplicate category name.*DAPI.*indices 0 and 2"
        ):
            validate_categories(["DAPI", "GFP", "DAPI"])

    def test_long_category_rejected(self) -> None:
        """Test that overly long category names are rejected."""
        long_name = "x" * 2000  # Exceeds 1024 char limit
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_categories(["Short", long_name])


class TestCategoryIndicesValidation:
    """Test validate_category_indices function."""

    def test_valid_indices(self) -> None:
        """Test that valid category indices are accepted."""
        categories = ["DAPI", "GFP", "mCherry"]
        values = np.array([0, 1, 2, 0, 1])
        # Should not raise
        validate_category_indices(values, categories)

    def test_valid_indices_boundary(self) -> None:
        """Test boundary indices are accepted."""
        categories = ["A", "B", "C"]
        values = np.array([0, 2])  # min and max valid
        validate_category_indices(values, categories)

    def test_negative_index_rejected(self) -> None:
        """Test that negative indices are rejected."""
        categories = ["A", "B", "C"]
        values = np.array([0, -1, 2])
        with pytest.raises(ValueError, match="negative category index"):
            validate_category_indices(values, categories)

    def test_out_of_range_index_rejected(self) -> None:
        """Test that out-of-range indices are rejected."""
        categories = ["A", "B", "C"]  # valid: 0, 1, 2
        values = np.array([0, 1, 3])  # 3 is out of range
        with pytest.raises(ValueError, match="out of range.*Valid categories"):
            validate_category_indices(values, categories)

    def test_non_integer_index_rejected(self) -> None:
        """Test that non-integer indices are rejected."""
        categories = ["A", "B", "C"]
        values = np.array([0.0, 1.5, 2.0])  # 1.5 is not integer
        with pytest.raises(ValueError, match="non-integer category index"):
            validate_category_indices(values, categories)

    def test_integer_floats_accepted(self) -> None:
        """Test that float values that are integers are accepted."""
        categories = ["A", "B", "C"]
        values = np.array([0.0, 1.0, 2.0])  # All are integer-valued floats
        validate_category_indices(values, categories)

    def test_empty_categories_rejected(self) -> None:
        """Test that empty categories list is rejected."""
        values = np.array([0, 1])
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_category_indices(values, [])


class TestIntensityValidation:
    """Test validate_intensity function."""

    def test_valid_intensity(self) -> None:
        """Test that valid intensity values are accepted."""
        assert validate_intensity(0.0) == 0.0
        assert validate_intensity(1.0) == 1.0
        assert validate_intensity(50.0) == 50.0
        assert validate_intensity(100.0) == 100.0

    def test_intensity_boundary_values(self) -> None:
        """Test intensity at exact boundary values."""
        assert validate_intensity(0.0) == 0.0
        assert validate_intensity(100.0) == 100.0

    def test_intensity_type_conversion(self) -> None:
        """Test that intensity values are converted to float."""
        assert validate_intensity(1) == 1.0
        assert validate_intensity(0) == 0.0
        assert validate_intensity("50.0") == 50.0
        assert validate_intensity(np.float32(2.5)) == pytest.approx(2.5)

    def test_invalid_intensity_out_of_range(self) -> None:
        """Test that out-of-range intensity values raise ValueError."""
        with pytest.raises(ValueError, match="Intensity must be between 0.0 and 100.0"):
            validate_intensity(-0.1)

        with pytest.raises(ValueError, match="Intensity must be between 0.0 and 100.0"):
            validate_intensity(100.1)

        with pytest.raises(ValueError, match="Intensity must be between 0.0 and 100.0"):
            validate_intensity(200.0)

    def test_invalid_intensity_type(self) -> None:
        """Test that non-convertible types raise TypeError."""
        with pytest.raises(TypeError, match="Intensity must be convertible to float"):
            validate_intensity("not_a_number")

        with pytest.raises(TypeError, match="Intensity must be convertible to float"):
            validate_intensity(None)

        with pytest.raises(TypeError, match="Intensity must be convertible to float"):
            validate_intensity([1.0])


class TestOffsetValidation:
    """Test validate_offset function."""

    def test_valid_offset(self) -> None:
        """Test that valid offset values are accepted."""
        assert validate_offset(0.0) == 0.0
        assert validate_offset(5.0) == 5.0
        assert validate_offset(-5.0) == -5.0
        assert validate_offset(1.5) == 1.5

    def test_offset_boundary_values(self) -> None:
        """Test offset at exact boundary values."""
        assert validate_offset(-10.0) == -10.0
        assert validate_offset(10.0) == 10.0

    def test_offset_type_conversion(self) -> None:
        """Test that offset values are converted to float."""
        assert validate_offset(0) == 0.0
        assert validate_offset(5) == 5.0
        assert validate_offset("-3.5") == -3.5
        assert validate_offset(np.float32(1.2)) == pytest.approx(1.2)

    def test_invalid_offset_out_of_range(self) -> None:
        """Test that out-of-range offset values raise ValueError."""
        with pytest.raises(ValueError, match="Offset must be between -10.0 and 10.0"):
            validate_offset(-10.1)

        with pytest.raises(ValueError, match="Offset must be between -10.0 and 10.0"):
            validate_offset(10.1)

        with pytest.raises(ValueError, match="Offset must be between -10.0 and 10.0"):
            validate_offset(50.0)

    def test_invalid_offset_type(self) -> None:
        """Test that non-convertible types raise TypeError."""
        with pytest.raises(TypeError, match="Offset must be convertible to float"):
            validate_offset("not_a_number")

        with pytest.raises(TypeError, match="Offset must be convertible to float"):
            validate_offset(None)

        with pytest.raises(TypeError, match="Offset must be convertible to float"):
            validate_offset([0.0])


class TestLayerValidation:
    """Test validate_layer function."""

    def test_valid_bool_values(self) -> None:
        """Test that boolean values are accepted."""
        assert validate_layer(True) is True
        assert validate_layer(False) is False

    def test_int_coercion(self) -> None:
        """Test that integer values are coerced to bool."""
        assert validate_layer(1) is True
        assert validate_layer(0) is False
        assert validate_layer(42) is True

    def test_float_coercion(self) -> None:
        """Test that float values are coerced to bool."""
        assert validate_layer(1.0) is True
        assert validate_layer(0.0) is False
        assert validate_layer(0.5) is True

    def test_invalid_type_string(self) -> None:
        """Test that string values raise TypeError."""
        with pytest.raises(TypeError, match="Layer must be a boolean, got str"):
            validate_layer("true")

    def test_invalid_type_none(self) -> None:
        """Test that None raises TypeError."""
        with pytest.raises(TypeError, match="Layer must be a boolean, got NoneType"):
            validate_layer(None)

    def test_invalid_type_list(self) -> None:
        """Test that list values raise TypeError."""
        with pytest.raises(TypeError, match="Layer must be a boolean, got list"):
            validate_layer([True])
