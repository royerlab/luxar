"""Tests for validate_colormap in validation/types.py."""

import numpy as np
import pytest

from luxar.validation.types import validate_colormap


class TestValidateColormap:
    """Tests for the validate_colormap validator."""

    def test_accepts_string(self) -> None:
        assert validate_colormap("viridis") == "viridis"

    def test_accepts_any_string(self) -> None:
        """Any non-empty string is accepted (resolution happens later)."""
        assert validate_colormap("some_custom_name") == "some_custom_name"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_colormap("")

    def test_accepts_uint8_array(self) -> None:
        arr = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        result = validate_colormap(arr)
        np.testing.assert_array_equal(result, arr)

    def test_accepts_float32_array(self) -> None:
        arr = np.random.rand(256, 3).astype(np.float32)
        result = validate_colormap(arr)
        np.testing.assert_array_equal(result, arr)

    def test_accepts_float64_array(self) -> None:
        arr = np.random.rand(10, 3).astype(np.float64)
        result = validate_colormap(arr)
        np.testing.assert_array_equal(result, arr)

    def test_rejects_int32_array(self) -> None:
        """int32 arrays are not valid colormap data."""
        arr = np.zeros((256, 3), dtype=np.int32)
        with pytest.raises(TypeError, match="uint8"):
            validate_colormap(arr)

    def test_rejects_int16_array(self) -> None:
        arr = np.zeros((256, 3), dtype=np.int16)
        with pytest.raises(TypeError, match="uint8"):
            validate_colormap(arr)

    def test_rejects_float_out_of_range_high(self) -> None:
        arr = np.array([[0.0, 0.0, 1.5], [1.0, 1.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            validate_colormap(arr)

    def test_rejects_float_negative(self) -> None:
        arr = np.array([[-0.1, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            validate_colormap(arr)

    def test_rejects_wrong_shape_4_channels(self) -> None:
        arr = np.zeros((256, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            validate_colormap(arr)

    def test_rejects_1d_array(self) -> None:
        arr = np.zeros(256, dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            validate_colormap(arr)

    def test_rejects_single_entry(self) -> None:
        arr = np.zeros((1, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="at least 2"):
            validate_colormap(arr)

    def test_rejects_non_array_non_string(self) -> None:
        with pytest.raises(TypeError, match="string or numpy"):
            validate_colormap(42)

    def test_rejects_list(self) -> None:
        with pytest.raises(TypeError, match="string or numpy"):
            validate_colormap([[0, 0, 0], [255, 255, 255]])

    def test_rejects_none(self) -> None:
        with pytest.raises(TypeError, match="string or numpy"):
            validate_colormap(None)
