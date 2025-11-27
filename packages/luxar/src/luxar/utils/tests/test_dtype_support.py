"""Tests for data type support and optimization."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import LuxarZarrCompiler
from luxar.typing_utils.datatypes import (
    DataTypeConfig,
    DataTypeMode,
    convert_array_dtype,
    get_dtype_info,
    infer_optimal_dtype,
    validate_dtype_string,
)


class TestDataTypeConfig:
    """Test DataTypeConfig class."""

    def test_auto_mode(self) -> None:
        """Test auto dtype selection mode."""
        config = DataTypeConfig(mode=DataTypeMode.AUTO)

        # Positions default to float32 for accuracy
        assert config.get_position_dtype() == np.float32

        # Colors detect HDR
        hdr_colors = np.array([[1.5, 2.0, 1.2]])
        assert config.get_color_dtype(hdr_colors) == np.float32

        sdr_colors = np.array([[0.5, 0.8, 0.2]])
        assert config.get_color_dtype(sdr_colors) == np.uint8

        # Radii/sharpness based on range
        small_radii = np.array([0.1, 0.5, 0.8])
        assert config.get_radius_dtype(small_radii) == np.uint8

        medium_radii = np.array([100.0, 200.0, 300.0])
        assert config.get_radius_dtype(medium_radii) == np.float16

        large_radii = np.array([10000.0, 20000.0, 30000.0])
        assert config.get_radius_dtype(large_radii) == np.float32

    def test_precision_mode(self) -> None:
        """Test precision mode always uses float32."""
        config = DataTypeConfig(mode=DataTypeMode.PRECISION)

        assert config.get_position_dtype() == np.float32
        assert config.get_color_dtype() == np.float32
        assert config.get_radius_dtype() == np.float32
        assert config.get_sharpness_dtype() == np.float32

    def test_memory_mode(self) -> None:
        """Test memory mode uses smallest types."""
        config = DataTypeConfig(mode=DataTypeMode.MEMORY)

        assert config.get_position_dtype() == np.float16
        assert config.get_color_dtype() == np.uint8
        assert config.get_radius_dtype() == np.uint8
        assert config.get_sharpness_dtype() == np.uint8

    def test_custom_mode(self) -> None:
        """Test custom mode with explicit dtypes."""
        config = DataTypeConfig(
            mode=DataTypeMode.CUSTOM,
            position_dtype="float16",
            color_dtype="uint16",
            radius_dtype="float16",
            sharpness_dtype="uint8",
        )

        assert config.get_position_dtype() == np.float16
        assert config.get_color_dtype() == np.uint16
        assert config.get_radius_dtype() == np.float16
        assert config.get_sharpness_dtype() == np.uint8

    def test_custom_mode_validation(self) -> None:
        """Test that custom mode requires all dtypes."""
        with pytest.raises(ValueError, match="CUSTOM mode requires all dtypes"):
            DataTypeConfig(
                mode=DataTypeMode.CUSTOM,
                position_dtype="float32",
                # Missing other dtypes
            )


class TestArrayConversion:
    """Test array dtype conversion utilities."""

    def test_float_to_uint8_normalized(self) -> None:
        """Test float to uint8 conversion with normalization."""
        float_arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        uint8_arr = convert_array_dtype(float_arr, np.uint8, normalize=True)

        assert uint8_arr.dtype == np.uint8
        np.testing.assert_array_equal(uint8_arr, [0, 127, 255])

    def test_uint8_to_float_normalized(self) -> None:
        """Test uint8 to float conversion with normalization."""
        uint8_arr = np.array([0, 127, 255], dtype=np.uint8)
        float_arr = convert_array_dtype(uint8_arr, np.float32, normalize=True)

        assert float_arr.dtype == np.float32
        np.testing.assert_array_almost_equal(float_arr, [0.0, 127 / 255, 1.0])

    def test_float16_float32_conversion(self) -> None:
        """Test conversion between float16 and float32."""
        float32_arr = np.array([1.5, -2.0, 3.14159], dtype=np.float32)
        float16_arr = convert_array_dtype(float32_arr, np.float16)

        assert float16_arr.dtype == np.float16
        # Float16 has less precision
        np.testing.assert_array_almost_equal(float16_arr, float32_arr, decimal=3)

        # Convert back
        back_to_float32 = convert_array_dtype(float16_arr, np.float32)
        assert back_to_float32.dtype == np.float32

    def test_no_conversion_needed(self) -> None:
        """Test that no conversion happens when types match."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        result = convert_array_dtype(arr, np.float32)
        assert result is arr  # Should be same object


class TestOptimalDtypeInference:
    """Test optimal dtype inference."""

    def test_position_dtype_inference(self) -> None:
        """Test dtype inference for positions."""
        positions = np.random.randn(100, 3)
        dtype = infer_optimal_dtype(positions, "position")
        assert dtype == np.float32  # Positions need good precision

    def test_color_dtype_inference(self) -> None:
        """Test dtype inference for colors."""
        # SDR colors
        sdr_colors = np.random.rand(100, 3)
        dtype = infer_optimal_dtype(sdr_colors, "color")
        assert dtype == np.uint8

        # HDR colors
        hdr_colors = np.random.rand(100, 3) * 2.0
        dtype = infer_optimal_dtype(hdr_colors, "color")
        assert dtype == np.float32

    def test_scalar_dtype_inference(self) -> None:
        """Test dtype inference for scalar values."""
        # Small range - use uint8
        small_values = np.random.rand(100) * 0.5
        dtype = infer_optimal_dtype(small_values, "radius")
        assert dtype == np.uint8

        # Medium range - use float16
        medium_values = np.random.randn(100) * 100
        dtype = infer_optimal_dtype(medium_values, "radius")
        assert dtype == np.float16

        # Large range - use float32
        large_values = np.random.randn(100) * 100000
        dtype = infer_optimal_dtype(large_values, "sharpness")
        assert dtype == np.float32


class TestDtypeInfo:
    """Test dtype information utilities."""

    def test_float32_info(self) -> None:
        """Test float32 dtype info."""
        info = get_dtype_info(np.float32)
        assert info["name"] == "float32"
        assert info["bytes"] == 4
        assert info["kind"] == "f"
        assert info["normalized"] is False

    def test_uint8_info(self) -> None:
        """Test uint8 dtype info."""
        info = get_dtype_info(np.uint8)
        assert info["name"] == "uint8"
        assert info["bytes"] == 1
        assert info["kind"] == "u"
        assert info["normalized"] is True
        assert info["range"] == (0, 255)


class TestDtypeValidation:
    """Test dtype string validation."""

    def test_valid_position_dtypes(self) -> None:
        """Test valid position dtypes."""
        assert validate_dtype_string("float32", "position")
        assert validate_dtype_string("float16", "position")

        with pytest.raises(ValueError, match="Invalid dtype"):
            validate_dtype_string("uint8", "position")

    def test_valid_color_dtypes(self) -> None:
        """Test valid color dtypes."""
        assert validate_dtype_string("float32", "color")
        assert validate_dtype_string("uint8", "color")
        assert validate_dtype_string("uint16", "color")

        with pytest.raises(ValueError, match="Invalid dtype"):
            validate_dtype_string("float16", "color")

    def test_valid_scalar_dtypes(self) -> None:
        """Test valid scalar dtypes."""
        assert validate_dtype_string("float32", "radius")
        assert validate_dtype_string("float16", "radius")
        assert validate_dtype_string("uint8", "radius")

        with pytest.raises(ValueError, match="Invalid dtype"):
            validate_dtype_string("uint16", "radius")


class TestCompilerWithDtypes:
    """Test LuxarZarrCompiler with different dtype configurations."""

    def test_compiler_with_memory_config(self) -> None:
        """Test compiler with memory-efficient configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create memory-efficient config
            config = DataTypeConfig(mode=DataTypeMode.MEMORY)

            # Create test data
            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32)
            radii = np.random.rand(100).astype(np.float32) * 0.5

            # Write with memory config
            with LuxarZarrCompiler(zarr_path, dtype_config=config) as compiler:
                scene = compiler.create_scene()
                scene.add_points("test", positions, colors=colors, radii=radii)

            # Check dtypes in zarr
            store = zarr.open_group(zarr_path, mode="r")
            points = store["test"]

            # Check dtype metadata
            assert points.attrs.get("position_dtype") == "float16"
            assert points.attrs.get("color_dtype") == "uint8"
            assert points.attrs.get("radius_dtype") == "uint8"

            # Check actual array dtypes
            assert points["positions"].dtype == np.float16
            assert points["colors"].dtype == np.uint8
            assert points["radii"].dtype == np.uint8

    def test_compiler_with_custom_config(self) -> None:
        """Test compiler with custom dtype configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create custom config
            config = DataTypeConfig(
                mode=DataTypeMode.CUSTOM,
                position_dtype="float32",
                color_dtype="uint16",
                radius_dtype="float16",
                sharpness_dtype="uint8",
            )

            # Create test data
            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32)
            radii = np.random.rand(100).astype(np.float32)
            sharpness = np.random.rand(100).astype(np.float32) * 2

            # Write with custom config
            with LuxarZarrCompiler(zarr_path, dtype_config=config) as compiler:
                scene = compiler.create_scene()
                scene.add_points(
                    "test", positions, colors=colors, radii=radii, sharpness=sharpness
                )

            # Check dtypes in zarr
            store = zarr.open_group(zarr_path, mode="r")
            points = store["test"]

            # Check actual array dtypes
            assert points["positions"].dtype == np.float32
            assert points["colors"].dtype == np.uint16
            assert points["radii"].dtype == np.float16
            assert points["sharpness"].dtype == np.uint8

    def test_hdr_color_detection(self) -> None:
        """Test that HDR colors are automatically detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Auto config
            config = DataTypeConfig(mode=DataTypeMode.AUTO)

            # Create HDR colors
            positions = np.random.randn(100, 3).astype(np.float32)
            hdr_colors = np.random.rand(100, 3).astype(np.float32) * 2.0  # HDR values

            # Write with auto config
            with LuxarZarrCompiler(zarr_path, dtype_config=config) as compiler:
                scene = compiler.create_scene()
                scene.add_points("test", positions, colors=hdr_colors)

            # Check that HDR was detected
            store = zarr.open_group(zarr_path, mode="r")
            points = store["test"]

            assert points.attrs.get("color_dtype") == "float32"
            assert points["colors"].dtype == np.float32

    def test_backward_compatibility(self) -> None:
        """Test that default behavior (no dtype_config) still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create test data
            positions = np.random.randn(100, 3).astype(np.float32)

            # Write without dtype_config (should use defaults)
            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene()
                scene.add_points("test", positions)

            # Check that it worked
            store = zarr.open_group(zarr_path, mode="r")
            assert "test" in store
            assert store["test"]["positions"].shape == (100, 3)
