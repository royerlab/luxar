"""Tests for enum types and their validation methods.

Tests cover:
- BlendingMode: validate method
- NodeType: validate method
- PhysicalUnit: validate method with normalized aliases
- RenderingLimits: class attributes
- Defaults: class attributes
"""

import pytest

from luxar.typing_utils.enums import (
    BlendingMode,
    Defaults,
    NodeType,
    PhysicalUnit,
    RenderingLimits,
)


class TestBlendingMode:
    """Tests for BlendingMode enum."""

    def test_values(self) -> None:
        """Test that BlendingMode has expected values."""
        assert BlendingMode.NORMAL.value == "normal"
        assert BlendingMode.ADDITIVE.value == "additive"
        assert BlendingMode.MAX.value == "max"

    def test_validate_normal(self) -> None:
        """Test validate with 'normal' string."""
        result = BlendingMode.validate("normal")
        assert result == BlendingMode.NORMAL

    def test_validate_additive(self) -> None:
        """Test validate with 'additive' string."""
        result = BlendingMode.validate("additive")
        assert result == BlendingMode.ADDITIVE

    def test_validate_max(self) -> None:
        """Test validate with 'max' string."""
        result = BlendingMode.validate("max")
        assert result == BlendingMode.MAX

    def test_validate_invalid(self) -> None:
        """Test validate raises for invalid value."""
        with pytest.raises(ValueError, match="Invalid blending mode"):
            BlendingMode.validate("invalid")

    def test_validate_error_lists_valid_options(self) -> None:
        """Test that error message lists valid options."""
        with pytest.raises(ValueError) as exc_info:
            BlendingMode.validate("bad")
        error_msg = str(exc_info.value)
        assert "'normal'" in error_msg
        assert "'additive'" in error_msg
        assert "'max'" in error_msg


class TestNodeType:
    """Tests for NodeType enum."""

    def test_values(self) -> None:
        """Test that NodeType has expected values."""
        assert NodeType.SCENE.value == "scene"
        assert NodeType.GROUP.value == "group"
        assert NodeType.POINTS.value == "points"
        assert NodeType.LINES.value == "lines"
        assert NodeType.GSPLATS.value == "gsplats"

    def test_validate_scene(self) -> None:
        """Test validate with 'scene' string."""
        result = NodeType.validate("scene")
        assert result == NodeType.SCENE

    def test_validate_group(self) -> None:
        """Test validate with 'group' string."""
        result = NodeType.validate("group")
        assert result == NodeType.GROUP

    def test_validate_points(self) -> None:
        """Test validate with 'points' string."""
        result = NodeType.validate("points")
        assert result == NodeType.POINTS

    def test_validate_lines(self) -> None:
        """Test validate with 'lines' string."""
        result = NodeType.validate("lines")
        assert result == NodeType.LINES

    def test_validate_gsplats(self) -> None:
        """Test validate with 'gsplats' string."""
        result = NodeType.validate("gsplats")
        assert result == NodeType.GSPLATS

    def test_validate_invalid(self) -> None:
        """Test validate raises for invalid value."""
        with pytest.raises(ValueError, match="Invalid node type"):
            NodeType.validate("invalid")

    def test_validate_error_lists_valid_options(self) -> None:
        """Test that error message lists valid options."""
        with pytest.raises(ValueError) as exc_info:
            NodeType.validate("bad")
        error_msg = str(exc_info.value)
        assert "'scene'" in error_msg
        assert "'group'" in error_msg
        assert "'points'" in error_msg


class TestPhysicalUnit:
    """Tests for PhysicalUnit enum."""

    def test_metric_values(self) -> None:
        """Test metric unit values."""
        assert PhysicalUnit.NANOMETER.value == "nm"
        assert PhysicalUnit.MICROMETER.value == "um"
        assert PhysicalUnit.MILLIMETER.value == "mm"
        assert PhysicalUnit.CENTIMETER.value == "cm"
        assert PhysicalUnit.METER.value == "m"
        assert PhysicalUnit.METRE.value == "metre"
        assert PhysicalUnit.KILOMETER.value == "km"

    def test_imperial_values(self) -> None:
        """Test imperial unit values."""
        assert PhysicalUnit.INCH.value == "inch"
        assert PhysicalUnit.FOOT.value == "foot"

    def test_other_values(self) -> None:
        """Test other unit values."""
        assert PhysicalUnit.PIXEL.value == "px"
        assert PhysicalUnit.ASTRONOMICAL_UNIT.value == "au"

    def test_validate_direct_values(self) -> None:
        """Test validate with direct enum values."""
        assert PhysicalUnit.validate("nm") == PhysicalUnit.NANOMETER
        assert PhysicalUnit.validate("um") == PhysicalUnit.MICROMETER
        assert PhysicalUnit.validate("mm") == PhysicalUnit.MILLIMETER
        assert PhysicalUnit.validate("cm") == PhysicalUnit.CENTIMETER
        assert PhysicalUnit.validate("m") == PhysicalUnit.METER
        assert PhysicalUnit.validate("metre") == PhysicalUnit.METRE
        assert PhysicalUnit.validate("km") == PhysicalUnit.KILOMETER
        assert PhysicalUnit.validate("inch") == PhysicalUnit.INCH
        assert PhysicalUnit.validate("foot") == PhysicalUnit.FOOT
        assert PhysicalUnit.validate("px") == PhysicalUnit.PIXEL
        assert PhysicalUnit.validate("au") == PhysicalUnit.ASTRONOMICAL_UNIT

    def test_validate_normalized_aliases_nanometer(self) -> None:
        """Test validate with normalized aliases for nanometer."""
        assert PhysicalUnit.validate("nanometer") == PhysicalUnit.NANOMETER
        assert PhysicalUnit.validate("nanometre") == PhysicalUnit.NANOMETER
        assert PhysicalUnit.validate("Nanometer") == PhysicalUnit.NANOMETER

    def test_validate_normalized_aliases_micrometer(self) -> None:
        """Test validate with normalized aliases for micrometer."""
        assert PhysicalUnit.validate("micrometer") == PhysicalUnit.MICROMETER
        assert PhysicalUnit.validate("micrometre") == PhysicalUnit.MICROMETER
        assert PhysicalUnit.validate("micron") == PhysicalUnit.MICROMETER
        # Unicode micro symbol
        assert PhysicalUnit.validate("μm") == PhysicalUnit.MICROMETER

    def test_validate_normalized_aliases_millimeter(self) -> None:
        """Test validate with normalized aliases for millimeter."""
        assert PhysicalUnit.validate("millimeter") == PhysicalUnit.MILLIMETER
        assert PhysicalUnit.validate("millimetre") == PhysicalUnit.MILLIMETER

    def test_validate_normalized_aliases_centimeter(self) -> None:
        """Test validate with normalized aliases for centimeter."""
        assert PhysicalUnit.validate("centimeter") == PhysicalUnit.CENTIMETER
        assert PhysicalUnit.validate("centimetre") == PhysicalUnit.CENTIMETER

    def test_validate_normalized_aliases_meter(self) -> None:
        """Test validate with normalized aliases for meter."""
        assert PhysicalUnit.validate("meter") == PhysicalUnit.METER
        assert PhysicalUnit.validate("metre") == PhysicalUnit.METRE

    def test_validate_normalized_aliases_kilometer(self) -> None:
        """Test validate with normalized aliases for kilometer."""
        assert PhysicalUnit.validate("kilometer") == PhysicalUnit.KILOMETER
        assert PhysicalUnit.validate("kilometre") == PhysicalUnit.KILOMETER

    def test_validate_normalized_aliases_pixel(self) -> None:
        """Test validate with normalized aliases for pixel."""
        assert PhysicalUnit.validate("pixel") == PhysicalUnit.PIXEL
        assert PhysicalUnit.validate("pixels") == PhysicalUnit.PIXEL

    def test_validate_invalid(self) -> None:
        """Test validate raises for invalid value."""
        with pytest.raises(ValueError, match="Invalid physical unit"):
            PhysicalUnit.validate("invalid_unit")

    def test_validate_error_lists_valid_options(self) -> None:
        """Test that error message lists valid options."""
        with pytest.raises(ValueError) as exc_info:
            PhysicalUnit.validate("bad")
        error_msg = str(exc_info.value)
        assert "'nm'" in error_msg
        assert "'um'" in error_msg
        assert "'mm'" in error_msg


class TestRenderingLimits:
    """Tests for RenderingLimits class."""

    def test_opacity_limits(self) -> None:
        """Test opacity limits."""
        assert RenderingLimits.OPACITY_MIN == 0.0
        assert RenderingLimits.OPACITY_MAX == 1.0

    def test_gamma_limits(self) -> None:
        """Test gamma limits."""
        assert RenderingLimits.GAMMA_MIN == 0.1
        assert RenderingLimits.GAMMA_MAX == 10.0

    def test_sharpness_limits(self) -> None:
        """Test sharpness limits."""
        assert RenderingLimits.SHARPNESS_MIN == 0.001
        assert RenderingLimits.SHARPNESS_MAX == 31.0

    def test_color_limits(self) -> None:
        """Test color limits."""
        assert RenderingLimits.COLOR_SDR_MIN == 0.0
        assert RenderingLimits.COLOR_SDR_MAX == 1.0
        assert RenderingLimits.COLOR_HDR_MAX == 10.0

    # [Python-R4/D-W2] Class-attribute values alone don't enforce the
    # actual invariants: GAMMA_MIN < GAMMA_MAX, OPACITY_MIN < OPACITY_MAX,
    # SHARPNESS_MIN < SHARPNESS_MAX, SHARPNESS_MIN > 0, COLOR_SDR_MIN <
    # COLOR_SDR_MAX, COLOR_SDR_MAX <= COLOR_HDR_MAX. A future edit that
    # swapped a MIN/MAX pair would silently pass every existing test
    # (each one anchors to a specific numeric value, so a swap reads
    # the same constants but means the wrong thing). Pin the relational
    # invariants so a swap surfaces immediately.
    def test_rendering_limits_relational_invariants(self) -> None:
        assert RenderingLimits.OPACITY_MIN < RenderingLimits.OPACITY_MAX
        assert RenderingLimits.GAMMA_MIN < RenderingLimits.GAMMA_MAX
        assert RenderingLimits.GAMMA_MIN > 0  # gamma=0 is undefined (∞ exponent)
        assert RenderingLimits.SHARPNESS_MIN < RenderingLimits.SHARPNESS_MAX
        assert RenderingLimits.SHARPNESS_MIN > 0  # sharpness must be positive
        assert RenderingLimits.COLOR_SDR_MIN < RenderingLimits.COLOR_SDR_MAX
        # HDR includes SDR: HDR's max must be at least SDR's max.
        assert RenderingLimits.COLOR_HDR_MAX >= RenderingLimits.COLOR_SDR_MAX

    def test_defaults_lie_within_rendering_limits(self) -> None:
        # OPACITY default must be in [OPACITY_MIN, OPACITY_MAX]
        assert (
            RenderingLimits.OPACITY_MIN
            <= Defaults.OPACITY
            <= RenderingLimits.OPACITY_MAX
        )
        # GAMMA default in [GAMMA_MIN, GAMMA_MAX]
        assert RenderingLimits.GAMMA_MIN <= Defaults.GAMMA <= RenderingLimits.GAMMA_MAX
        # SHARPNESS default in [SHARPNESS_MIN, SHARPNESS_MAX]
        assert (
            RenderingLimits.SHARPNESS_MIN
            <= Defaults.SHARPNESS
            <= RenderingLimits.SHARPNESS_MAX
        )


class TestDefaults:
    """Tests for Defaults class."""

    def test_rendering_defaults(self) -> None:
        """Test rendering property defaults."""
        assert Defaults.OPACITY == 1.0
        assert Defaults.GAMMA == 1.0
        assert Defaults.SHARPNESS == 2.0  # Quadratic polynomial falloff
        assert Defaults.BLENDING_MODE == BlendingMode.ADDITIVE

    def test_storage_defaults(self) -> None:
        """Test storage-related defaults."""
        assert Defaults.CHUNK_SIZE == 32768
        assert Defaults.RADIUS == 0.1

    def test_color_defaults(self) -> None:
        """Test color defaults."""
        assert Defaults.COLOR_WHITE == (1.0, 1.0, 1.0)
        assert Defaults.COLOR_BLACK == (0.0, 0.0, 0.0)


class TestEnumStringBehavior:
    """Test that enums behave as strings."""

    def test_blending_mode_is_str(self) -> None:
        """Test BlendingMode inherits from str."""
        assert isinstance(BlendingMode.NORMAL, str)
        assert BlendingMode.NORMAL.value == "normal"

    def test_node_type_is_str(self) -> None:
        """Test NodeType inherits from str."""
        assert isinstance(NodeType.POINTS, str)
        assert NodeType.POINTS.value == "points"

    def test_physical_unit_is_str(self) -> None:
        """Test PhysicalUnit inherits from str."""
        assert isinstance(PhysicalUnit.MICROMETER, str)
        assert PhysicalUnit.MICROMETER.value == "um"
