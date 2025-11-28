"""Comprehensive tests for typing_utils/datatypes.py module."""

import numpy as np
import pytest

from luxar.typing_utils.datatypes import (
    DEFAULT_CONFIG,
    MEMORY_CONFIG,
    PRECISION_CONFIG,
    DataTypeConfig,
    DataTypeMode,
    convert_array_dtype,
    get_dtype_info,
    infer_optimal_dtype,
    validate_dtype_string,
)


class TestDataTypeMode:
    """Test DataTypeMode enum."""

    def test_data_type_modes(self) -> None:
        """Test that all data type modes are defined."""
        assert DataTypeMode.AUTO == "auto"
        assert DataTypeMode.PRECISION == "precision"
        assert DataTypeMode.MEMORY == "memory"
        assert DataTypeMode.CUSTOM == "custom"


class TestDataTypeConfigInit:
    """Test DataTypeConfig initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        config = DataTypeConfig()
        assert config.mode == DataTypeMode.AUTO
        assert config._position_dtype is None
        assert config._color_dtype is None
        assert config._radius_dtype is None
        assert config._sharpness_dtype is None

    def test_init_with_mode(self) -> None:
        """Test initialization with specific mode."""
        config_precision = DataTypeConfig(mode=DataTypeMode.PRECISION)
        assert config_precision.mode == DataTypeMode.PRECISION

        config_memory = DataTypeConfig(mode=DataTypeMode.MEMORY)
        assert config_memory.mode == DataTypeMode.MEMORY

    def test_custom_mode_requires_all_dtypes(self) -> None:
        """Test that CUSTOM mode requires all dtypes to be specified."""
        # Missing all dtypes
        with pytest.raises(ValueError, match="CUSTOM mode requires all dtypes"):
            DataTypeConfig(mode=DataTypeMode.CUSTOM)

        # Missing some dtypes
        with pytest.raises(ValueError, match="CUSTOM mode requires all dtypes"):
            DataTypeConfig(
                mode=DataTypeMode.CUSTOM,
                position_dtype="float32",
                color_dtype="uint8",
            )

        # All dtypes provided - should work
        config = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float32",
            color_dtype="uint8",
            radius_dtype="float32",
            sharpness_dtype="uint8",
        )
        assert config.mode == DataTypeMode.CUSTOM

    def test_custom_mode_with_all_dtypes(self) -> None:
        """Test CUSTOM mode with all dtypes specified."""
        config = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float16",
            color_dtype="float32",
            radius_dtype="uint8",
            sharpness_dtype="float16",
        )
        assert config._position_dtype == "float16"
        assert config._color_dtype == "float32"
        assert config._radius_dtype == "uint8"
        assert config._sharpness_dtype == "float16"


class TestDataTypeConfigPositionDtype:
    """Test DataTypeConfig.get_position_dtype()."""

    def test_custom_mode(self) -> None:
        """Test position dtype in CUSTOM mode."""
        config = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float32",
            color_dtype="uint8",
            radius_dtype="float32",
            sharpness_dtype="uint8",
        )
        assert config.get_position_dtype() == np.float32

        config_f16 = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float16",
            color_dtype="uint8",
            radius_dtype="float32",
            sharpness_dtype="uint8",
        )
        assert config_f16.get_position_dtype() == np.float16

    def test_precision_mode(self) -> None:
        """Test position dtype in PRECISION mode."""
        config = DataTypeConfig(mode=DataTypeMode.PRECISION)
        assert config.get_position_dtype() == np.float32

    def test_memory_mode(self) -> None:
        """Test position dtype in MEMORY mode."""
        config = DataTypeConfig(mode=DataTypeMode.MEMORY)
        assert config.get_position_dtype() == np.float16

    def test_auto_mode(self) -> None:
        """Test position dtype in AUTO mode."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)
        # AUTO mode defaults to float32 for positions
        assert config.get_position_dtype() == np.float32

        # With data (currently ignored, but test interface)
        data = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        assert config.get_position_dtype(data) == np.float32


class TestDataTypeConfigColorDtype:
    """Test DataTypeConfig.get_color_dtype()."""

    def test_custom_mode(self) -> None:
        """Test color dtype in CUSTOM mode."""
        config_f32 = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float32",
            color_dtype="float32",
            radius_dtype="float32",
            sharpness_dtype="uint8",
        )
        assert config_f32.get_color_dtype() == np.float32

        config_u8 = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float32",
            color_dtype="uint8",
            radius_dtype="float32",
            sharpness_dtype="uint8",
        )
        assert config_u8.get_color_dtype() == np.uint8

    def test_precision_mode(self) -> None:
        """Test color dtype in PRECISION mode."""
        config = DataTypeConfig(mode=DataTypeMode.PRECISION)
        assert config.get_color_dtype() == np.float32

    def test_memory_mode(self) -> None:
        """Test color dtype in MEMORY mode."""
        config = DataTypeConfig(mode=DataTypeMode.MEMORY)
        assert config.get_color_dtype() == np.uint8

    def test_auto_mode_hdr_colors(self) -> None:
        """Test AUTO mode with HDR colors (values > 1.0)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # HDR colors
        hdr_colors = np.array([[1.0, 2.0, 3.0], [5.0, 10.0, 0.5]])
        assert config.get_color_dtype(hdr_colors) == np.float32

    def test_auto_mode_standard_colors(self) -> None:
        """Test AUTO mode with standard colors (0.0-1.0)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Standard colors
        std_colors = np.array([[0.0, 0.5, 1.0], [0.2, 0.8, 0.4]])
        assert config.get_color_dtype(std_colors) == np.uint8

    def test_auto_mode_no_data(self) -> None:
        """Test AUTO mode without data."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)
        # Default to uint8 for memory efficiency
        assert config.get_color_dtype() == np.uint8

    def test_auto_mode_negative_colors(self) -> None:
        """Test AUTO mode with negative colors (fallback to float32)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Negative colors (unusual, fallback to float32)
        neg_colors = np.array([[-0.1, 0.5, 1.0], [0.2, -0.3, 0.4]])
        assert config.get_color_dtype(neg_colors) == np.float32


class TestDataTypeConfigRadiusDtype:
    """Test DataTypeConfig.get_radius_dtype()."""

    def test_custom_mode(self) -> None:
        """Test radius dtype in CUSTOM mode."""
        config = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float32",
            color_dtype="uint8",
            radius_dtype="float16",
            sharpness_dtype="uint8",
        )
        assert config.get_radius_dtype() == np.float16

    def test_precision_mode(self) -> None:
        """Test radius dtype in PRECISION mode."""
        config = DataTypeConfig(mode=DataTypeMode.PRECISION)
        assert config.get_radius_dtype() == np.float32

    def test_memory_mode(self) -> None:
        """Test radius dtype in MEMORY mode."""
        config = DataTypeConfig(mode=DataTypeMode.MEMORY)
        assert config.get_radius_dtype() == np.uint8

    def test_auto_mode_small_range(self) -> None:
        """Test AUTO mode with small range radii (<=1.0)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Small range radii
        small_radii = np.array([0.1, 0.5, 0.9, 1.0])
        assert config.get_radius_dtype(small_radii) == np.uint8

    def test_auto_mode_medium_range(self) -> None:
        """Test AUTO mode with medium range radii (<1000)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Medium range radii
        medium_radii = np.array([1.0, 10.0, 100.0, 500.0])
        assert config.get_radius_dtype(medium_radii) == np.float16

    def test_auto_mode_large_range(self) -> None:
        """Test AUTO mode with large range radii (>=1000)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Large range radii
        large_radii = np.array([100.0, 1000.0, 5000.0])
        assert config.get_radius_dtype(large_radii) == np.float32

    def test_auto_mode_no_data(self) -> None:
        """Test AUTO mode without data."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)
        # Default to float32 for safety
        assert config.get_radius_dtype() == np.float32


class TestDataTypeConfigSharpnessDtype:
    """Test DataTypeConfig.get_sharpness_dtype()."""

    def test_custom_mode(self) -> None:
        """Test sharpness dtype in CUSTOM mode."""
        config = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float32",
            color_dtype="uint8",
            radius_dtype="float32",
            sharpness_dtype="float16",
        )
        assert config.get_sharpness_dtype() == np.float16

    def test_precision_mode(self) -> None:
        """Test sharpness dtype in PRECISION mode."""
        config = DataTypeConfig(mode=DataTypeMode.PRECISION)
        assert config.get_sharpness_dtype() == np.float32

    def test_memory_mode(self) -> None:
        """Test sharpness dtype in MEMORY mode."""
        config = DataTypeConfig(mode=DataTypeMode.MEMORY)
        assert config.get_sharpness_dtype() == np.uint8

    def test_auto_mode_typical_range(self) -> None:
        """Test AUTO mode with typical sharpness range (<=15)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Typical sharpness values
        typical_sharpness = np.array([0.5, 1.0, 5.0, 10.0, 15.0])
        assert config.get_sharpness_dtype(typical_sharpness) == np.uint8

    def test_auto_mode_medium_range(self) -> None:
        """Test AUTO mode with medium range sharpness (<1000)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Medium range sharpness
        medium_sharpness = np.array([20.0, 100.0, 500.0])
        assert config.get_sharpness_dtype(medium_sharpness) == np.float16

    def test_auto_mode_large_range(self) -> None:
        """Test AUTO mode with large range sharpness (>=1000)."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Large range sharpness
        large_sharpness = np.array([100.0, 1000.0, 5000.0])
        assert config.get_sharpness_dtype(large_sharpness) == np.float32

    def test_auto_mode_no_data(self) -> None:
        """Test AUTO mode without data."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)
        # Default to float32 for safety
        assert config.get_sharpness_dtype() == np.float32


class TestConvertArrayDtype:
    """Test convert_array_dtype function."""

    def test_no_conversion_needed(self) -> None:
        """Test that arrays with matching dtype are returned unchanged."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = convert_array_dtype(array, np.float32)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, array)

    def test_float_to_uint8_with_normalize(self) -> None:
        """Test float to uint8 conversion with normalization."""
        array = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = convert_array_dtype(array, np.uint8, normalize=True)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 127, 255])

    def test_float_to_uint8_with_custom_range(self) -> None:
        """Test float to uint8 with custom input range."""
        array = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        result = convert_array_dtype(
            array, np.uint8, normalize=True, input_range=(0.0, 100.0)
        )
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 127, 255])

    def test_float_to_uint8_degenerate_range(self) -> None:
        """Test float to uint8 with degenerate range (all same value)."""
        array = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        result = convert_array_dtype(
            array, np.uint8, normalize=True, input_range=(5.0, 5.0)
        )
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_float_to_uint8_without_normalize(self) -> None:
        """Test float to uint8 without normalization (direct cast with clip)."""
        array = np.array([0.0, 128.0, 255.0, 300.0], dtype=np.float32)
        result = convert_array_dtype(array, np.uint8, normalize=False)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 128, 255, 255])  # 300 clipped to 255

    def test_float_to_uint16_with_normalize(self) -> None:
        """Test float to uint16 conversion with normalization."""
        array = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = convert_array_dtype(array, np.uint16, normalize=True)
        assert result.dtype == np.uint16
        expected = [0, 32767, 65535]
        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_float_to_uint16_without_normalize(self) -> None:
        """Test float to uint16 without normalization."""
        array = np.array([0.0, 1000.0, 65535.0, 70000.0], dtype=np.float32)
        result = convert_array_dtype(array, np.uint16, normalize=False)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, [0, 1000, 65535, 65535])  # 70000 clipped

    def test_uint8_to_float_with_normalize(self) -> None:
        """Test uint8 to float conversion with normalization."""
        array = np.array([0, 127, 255], dtype=np.uint8)
        result = convert_array_dtype(array, np.float32, normalize=True)
        assert result.dtype == np.float32
        expected = [0.0, 127 / 255, 1.0]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_uint8_to_float_with_custom_range(self) -> None:
        """Test uint8 to float with custom output range."""
        array = np.array([0, 127, 255], dtype=np.uint8)
        result = convert_array_dtype(
            array, np.float32, normalize=True, input_range=(0.0, 100.0)
        )
        assert result.dtype == np.float32
        expected = [0.0, 127 / 255 * 100, 100.0]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_uint8_to_float_without_normalize(self) -> None:
        """Test uint8 to float without normalization."""
        array = np.array([0, 128, 255], dtype=np.uint8)
        result = convert_array_dtype(array, np.float32, normalize=False)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [0.0, 128.0, 255.0])

    def test_uint16_to_float_with_normalize(self) -> None:
        """Test uint16 to float conversion with normalization."""
        array = np.array([0, 32767, 65535], dtype=np.uint16)
        result = convert_array_dtype(array, np.float32, normalize=True)
        assert result.dtype == np.float32
        expected = [0.0, 32767 / 65535, 1.0]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_uint16_to_float_without_normalize(self) -> None:
        """Test uint16 to float without normalization."""
        array = np.array([0, 1000, 65535], dtype=np.uint16)
        result = convert_array_dtype(array, np.float32, normalize=False)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [0.0, 1000.0, 65535.0])

    def test_float16_to_float32(self) -> None:
        """Test float16 to float32 conversion."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result = convert_array_dtype(array, np.float32)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0], decimal=2)

    def test_float32_to_float16(self) -> None:
        """Test float32 to float16 conversion."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = convert_array_dtype(array, np.float16)
        assert result.dtype == np.float16
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0], decimal=2)


class TestInferOptimalDtype:
    """Test infer_optimal_dtype function."""

    def test_position_inference(self) -> None:
        """Test dtype inference for positions."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        dtype = infer_optimal_dtype(positions, "position")
        assert dtype == np.float32  # Positions always use float32

    def test_color_hdr_inference(self) -> None:
        """Test dtype inference for HDR colors."""
        hdr_colors = np.array([[1.0, 2.0, 3.0], [5.0, 10.0, 0.5]], dtype=np.float32)
        dtype = infer_optimal_dtype(hdr_colors, "color")
        assert dtype == np.float32

    def test_color_standard_inference(self) -> None:
        """Test dtype inference for standard colors."""
        std_colors = np.array([[0.0, 0.5, 1.0], [0.2, 0.8, 0.4]], dtype=np.float32)
        dtype = infer_optimal_dtype(std_colors, "color")
        assert dtype == np.uint8

    def test_color_unusual_range(self) -> None:
        """Test dtype inference for colors with unusual range."""
        unusual_colors = np.array([[-0.1, 0.5, 1.0], [0.2, 1.2, 0.4]], dtype=np.float32)
        dtype = infer_optimal_dtype(unusual_colors, "color")
        assert dtype == np.float32  # Safe default

    def test_radius_small_range(self) -> None:
        """Test dtype inference for small range radii."""
        small_radii = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        dtype = infer_optimal_dtype(small_radii, "radius")
        assert dtype == np.uint8

    def test_radius_medium_range(self) -> None:
        """Test dtype inference for medium range radii."""
        medium_radii = np.array([1.0, 10.0, 100.0], dtype=np.float32)
        dtype = infer_optimal_dtype(medium_radii, "radius")
        assert dtype == np.float16

    def test_radius_large_range(self) -> None:
        """Test dtype inference for large range radii."""
        large_radii = np.array([100.0, 1000.0, 100000.0], dtype=np.float32)
        dtype = infer_optimal_dtype(large_radii, "radius")
        assert dtype == np.float32

    def test_sharpness_inference(self) -> None:
        """Test dtype inference for sharpness (same logic as radius)."""
        typical_sharpness = np.array([0.5, 1.0, 5.0], dtype=np.float32)
        dtype = infer_optimal_dtype(typical_sharpness, "sharpness")
        # Max value is 5.0, which is > 1.0 and < 65536, so returns float16
        assert dtype == np.float16


class TestGetDtypeInfo:
    """Test get_dtype_info function."""

    def test_float32_info(self) -> None:
        """Test dtype info for float32."""
        info = get_dtype_info(np.float32)
        assert info["name"] == "float32"
        assert info["bytes"] == 4
        assert info["kind"] == "f"
        assert info["range"] == (-3.4e38, 3.4e38)
        assert info["normalized"] is False

    def test_float16_info(self) -> None:
        """Test dtype info for float16."""
        info = get_dtype_info(np.float16)
        assert info["name"] == "float16"
        assert info["bytes"] == 2
        assert info["kind"] == "f"
        assert info["range"] == (-65504.0, 65504.0)
        assert info["normalized"] is False

    def test_uint8_info(self) -> None:
        """Test dtype info for uint8."""
        info = get_dtype_info(np.uint8)
        assert info["name"] == "uint8"
        assert info["bytes"] == 1
        assert info["kind"] == "u"
        assert info["range"] == (0, 255)
        assert info["normalized"] is True  # WebGL can normalize

    def test_uint16_info(self) -> None:
        """Test dtype info for uint16."""
        info = get_dtype_info(np.uint16)
        assert info["name"] == "uint16"
        assert info["bytes"] == 2
        assert info["kind"] == "u"
        assert info["range"] == (0, 65535)
        assert info["normalized"] is True  # WebGL can normalize

    def test_int8_info(self) -> None:
        """Test dtype info for int8."""
        info = get_dtype_info(np.int8)
        assert info["name"] == "int8"
        assert info["bytes"] == 1
        assert info["kind"] == "i"
        assert info["range"] == (-128, 127)
        assert info["normalized"] is False

    def test_int16_info(self) -> None:
        """Test dtype info for int16."""
        info = get_dtype_info(np.int16)
        assert info["name"] == "int16"
        assert info["bytes"] == 2
        assert info["kind"] == "i"
        assert info["range"] == (-32768, 32767)
        assert info["normalized"] is False

    def test_dtype_string_input(self) -> None:
        """Test that get_dtype_info works with string dtype."""
        info = get_dtype_info("float32")
        assert info["name"] == "float32"
        assert info["bytes"] == 4


class TestValidateDtypeString:
    """Test validate_dtype_string function."""

    def test_valid_position_dtypes(self) -> None:
        """Test valid position dtypes."""
        assert validate_dtype_string("float32", "position") is True
        assert validate_dtype_string("float16", "position") is True

    def test_invalid_position_dtypes(self) -> None:
        """Test invalid position dtypes."""
        with pytest.raises(ValueError, match="Invalid dtype 'uint8' for position"):
            validate_dtype_string("uint8", "position")

        with pytest.raises(ValueError, match="Invalid dtype 'float64' for position"):
            validate_dtype_string("float64", "position")

    def test_valid_color_dtypes(self) -> None:
        """Test valid color dtypes."""
        assert validate_dtype_string("float32", "color") is True
        assert validate_dtype_string("uint8", "color") is True
        assert validate_dtype_string("uint16", "color") is True

    def test_invalid_color_dtypes(self) -> None:
        """Test invalid color dtypes."""
        with pytest.raises(ValueError, match="Invalid dtype 'float16' for color"):
            validate_dtype_string("float16", "color")

        with pytest.raises(ValueError, match="Invalid dtype 'int8' for color"):
            validate_dtype_string("int8", "color")

    def test_valid_radius_dtypes(self) -> None:
        """Test valid radius dtypes."""
        assert validate_dtype_string("float32", "radius") is True
        assert validate_dtype_string("float16", "radius") is True
        assert validate_dtype_string("uint8", "radius") is True

    def test_invalid_radius_dtypes(self) -> None:
        """Test invalid radius dtypes."""
        with pytest.raises(ValueError, match="Invalid dtype 'uint16' for radius"):
            validate_dtype_string("uint16", "radius")

    def test_valid_sharpness_dtypes(self) -> None:
        """Test valid sharpness dtypes."""
        assert validate_dtype_string("float32", "sharpness") is True
        assert validate_dtype_string("float16", "sharpness") is True
        assert validate_dtype_string("uint8", "sharpness") is True

    def test_unknown_attribute_type(self) -> None:
        """Test unknown attribute type."""
        with pytest.raises(ValueError, match="Unknown attribute type: unknown"):
            validate_dtype_string("float32", "unknown")


class TestDefaultConfigs:
    """Test predefined configuration constants."""

    def test_default_config(self) -> None:
        """Test DEFAULT_CONFIG."""
        assert DEFAULT_CONFIG.mode == DataTypeMode.AUTO

    def test_precision_config(self) -> None:
        """Test PRECISION_CONFIG."""
        assert PRECISION_CONFIG.mode == DataTypeMode.PRECISION

    def test_memory_config(self) -> None:
        """Test MEMORY_CONFIG."""
        assert MEMORY_CONFIG.mode == DataTypeMode.MEMORY
