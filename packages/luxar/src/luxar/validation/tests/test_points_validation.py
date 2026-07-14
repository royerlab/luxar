"""Tests for point data validation.

This module tests the validation of positions, colors, radii, and sharpness
parameters when adding points to a scene.
"""

import warnings

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import ArrayDecoder, EncodingMode
from luxar.validation import ValidationError, validate_colors_for_writing

# =============================================================================
# Position Validation Tests
# =============================================================================


def test_bad_positions_shape(tmp_path) -> None:
    """Positions must be 2D array with shape (N, ndim)."""
    store = tmp_path / "bad.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="Positions must"):
            scene.add_points("Broken", np.ones((3,)), parent=scene)


def test_mismatched_colors(tmp_path) -> None:
    """Colors array length must match positions."""
    store = tmp_path / "bad2.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        pos = np.ones((10, 3), np.float32)
        col = np.ones((5, 3), np.uint8)
        with pytest.raises(ValueError, match="colors.*doesn't match"):
            scene.add_points("Nope", pos, col, parent=scene)


@pytest.mark.parametrize(
    "field,kwargs,error_pattern",
    [
        (
            "positions_nan",
            {"positions": np.array([[0.0, np.nan, 1.0]], dtype=np.float32)},
            "positions: Contains 1 NaN or Inf",
        ),
        (
            "positions_inf",
            {"positions": np.array([[0.0, np.inf, 1.0]], dtype=np.float32)},
            "positions: Contains 1 NaN or Inf",
        ),
        (
            "colors_nan",
            {"colors": np.array([[1.0, np.nan, 0.0]], dtype=np.float32)},
            "colors: Contains 1 NaN or Inf",
        ),
        (
            "colors_inf",
            {"colors": np.array([[1.0, np.inf, 0.0]], dtype=np.float32)},
            "colors: Contains 1 NaN or Inf",
        ),
        (
            "radii_nan",
            {"radii": np.array([np.nan], dtype=np.float32)},
            "radii: Contains 1 NaN or Inf",
        ),
        (
            "radii_inf",
            {"radii": np.array([np.inf], dtype=np.float32)},
            "radii: Contains 1 NaN or Inf",
        ),
        (
            "sharpness_nan",
            {"sharpness": np.array([np.nan], dtype=np.float32)},
            "sharpness: Contains 1 NaN or Inf",
        ),
    ],
)
def test_non_finite_point_attributes_rejected(
    tmp_path, field: str, kwargs: dict[str, np.ndarray], error_pattern: str
) -> None:
    """NaN/Inf values should fail before corrupting stored Zarr arrays."""
    store = tmp_path / f"bad_{field}.luxar.zarr"
    positions = kwargs.pop("positions", np.array([[0.0, 1.0, 2.0]], dtype=np.float32))

    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValidationError, match=error_pattern):
            compiler.write_points("bad", positions, **kwargs)


def test_empty_colors_are_valid_only_for_zero_points() -> None:
    """Empty colors should not reach min/max reductions."""
    validate_colors_for_writing(np.empty((0, 3), dtype=np.float32), n_points=0)

    with pytest.raises(ValidationError, match="Number of colors"):
        validate_colors_for_writing(np.empty((0, 3), dtype=np.float32), n_points=5)


def test_integer_color_ranges_do_not_emit_hdr_warning() -> None:
    """Integer SDR colors use native integer ranges, not HDR float ranges."""
    colors = np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_colors_for_writing(colors, n_points=2)

    assert caught == []


# =============================================================================
# Radii Validation Tests (Parametrized)
# =============================================================================


@pytest.mark.parametrize(
    "radii_factory,error_pattern,test_id",
    [
        # Negative values
        (
            lambda: np.random.uniform(-1.0, 1.0, 100).astype(np.float32),
            "Radii must be positive",
            "negative_values",
        ),
        # Wrong count (50 instead of 100)
        (
            lambda: np.random.uniform(0.1, 2.0, 50).astype(np.float32),
            "doesn't match number of points",
            "count_mismatch",
        ),
        # Wrong shape (2D instead of 1D)
        (
            lambda: np.random.uniform(0.1, 2.0, (100, 2)).astype(np.float32),
            "Expected 1D array",
            "wrong_shape",
        ),
        # All zeros (edge case)
        (
            lambda: np.zeros(100, dtype=np.float32),
            "Radii must be positive",
            "all_zeros",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_invalid_radii(tmp_path, radii_factory, error_pattern, test_id) -> None:
    """Test that various invalid radii configurations raise ValidationError."""
    store = tmp_path / f"invalid_radii_{test_id}.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        positions = np.random.randn(100, 3).astype(np.float32)

        with pytest.raises(ValidationError, match=error_pattern):
            compiler.write_points("test", positions, radii=radii_factory())


def test_valid_radii(tmp_path) -> None:
    """Test that valid radii are accepted and stored correctly."""
    store = tmp_path / "radii_test.luxar.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=EncodingMode.PRECISION, enable_spatial_index=False
    ) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        positions = np.random.randn(100, 3).astype(np.float32)
        radii = np.random.uniform(0.1, 2.0, 100).astype(np.float32)
        compiler.write_points("test", positions, radii=radii)

    root = zarr.open_group(store, mode="r")
    assert "test/radii" in root

    decoder = ArrayDecoder()
    stored_radii = decoder.decode(root["test/radii"], root)
    np.testing.assert_array_equal(stored_radii, radii)


# =============================================================================
# Sharpness Validation Tests (Parametrized)
# =============================================================================


@pytest.mark.parametrize(
    "sharpness_factory,error_pattern,test_id",
    [
        # Negative values (below the normalized [0, 1] range)
        (
            lambda: np.random.uniform(-1.0, 0.0, 100).astype(np.float32),
            "must be >= 0.0",
            "negative_values",
        ),
        # Above the maximum (> 1.0)
        (
            lambda: np.random.uniform(1.5, 5.0, 100).astype(np.float32),
            "exceed maximum",
            "above_max",
        ),
        # Wrong count (50 instead of 100)
        (
            lambda: np.random.uniform(0.0, 1.0, 50).astype(np.float32),
            "doesn't match",
            "count_mismatch",
        ),
        # Wrong shape (2D instead of 1D)
        (
            lambda: np.random.uniform(0.0, 1.0, (100, 2)).astype(np.float32),
            "Expected 1D array",
            "wrong_shape",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_invalid_sharpness(tmp_path, sharpness_factory, error_pattern, test_id) -> None:
    """Test that various invalid sharpness configurations raise ValidationError."""
    store = tmp_path / f"invalid_sharpness_{test_id}.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        positions = np.random.randn(100, 3).astype(np.float32)

        with pytest.raises(ValidationError, match=error_pattern):
            compiler.write_points("test", positions, sharpness=sharpness_factory())


def test_valid_sharpness(tmp_path) -> None:
    """Test that valid sharpness values are accepted and stored correctly."""
    store = tmp_path / "sharpness_test.luxar.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=EncodingMode.PRECISION, enable_spatial_index=False
    ) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        positions = np.random.randn(100, 3).astype(np.float32)
        sharpness = np.random.uniform(0.0, 1.0, 100).astype(np.float32)
        compiler.write_points("test", positions, sharpness=sharpness)

    root = zarr.open_group(store, mode="r")
    assert "test/sharpnesses" in root

    decoder = ArrayDecoder()
    stored_sharpness = decoder.decode(root["test/sharpnesses"], root)
    np.testing.assert_array_equal(stored_sharpness, sharpness)


def test_sharpness_out_of_range_rejected(tmp_path) -> None:
    """Out-of-range sharpness (outside [0, 1]) raises a ValidationError."""
    store = tmp_path / "sharpness_rejected.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        positions = np.random.randn(100, 3).astype(np.float32)
        # Mix of values including out-of-range (sharpness is a normalized [0, 1] knob)
        sharpness = np.array([0.3, 0.5, 15.0] * 33 + [0.5]).astype(np.float32)

        with pytest.raises(ValidationError, match="exceed maximum"):
            compiler.write_points("test", positions, sharpness=sharpness)
