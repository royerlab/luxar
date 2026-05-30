"""Tests for line data validation.

This module mirrors `test_points_validation.py` for Lines geometry (audit
G2 in `delme/test-audit-luxar-codebase/findings-global-pattern-sweep.md`
and G1 in `python-core-validation`). Points/Lines/GSplats symmetry rule
requires that every Points validation test has a parallel Lines variant.

For Lines the per-vertex scalar attributes are:
- ``widths`` (mandatory, must be positive — parallel to Points ``radii``)
- ``colors`` (optional, per-vertex)
- ``sharpness`` (optional, per-vertex — same constraints as Points)
"""

import warnings

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import ArrayDecoder, EncodingMode
from luxar.validation import ValidationError, validate_colors_for_writing


def _polyline_vertices(n: int) -> np.ndarray:
    """Build a deterministic polyline with ``n`` vertices in 3D."""
    return np.random.randn(n, 3).astype(np.float32)


# =============================================================================
# Vertex Validation Tests
# =============================================================================


def test_bad_vertices_shape(tmp_path) -> None:
    """Vertices must be 2D array with shape (N, ndim) — mirrors test_bad_positions_shape."""
    store = tmp_path / "bad.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises((ValueError, ValidationError)):
            compiler.write_lines("Broken", np.ones((3,), dtype=np.float32), widths=0.1)


def test_mismatched_colors(tmp_path) -> None:
    """Colors array length must match vertices — mirrors points equivalent."""
    store = tmp_path / "bad_colors.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        vertices = _polyline_vertices(10)
        widths = np.ones(10, dtype=np.float32) * 0.1
        bad_colors = np.ones((5, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            compiler.write_lines("Nope", vertices, widths=widths, colors=bad_colors)


@pytest.mark.parametrize(
    "field,kwargs,error_pattern",
    [
        (
            "vertices_nan",
            {"vertices": np.array([[0.0, np.nan, 1.0]], dtype=np.float32)},
            "Contains 1 NaN or Inf",
        ),
        (
            "vertices_inf",
            {"vertices": np.array([[0.0, np.inf, 1.0]], dtype=np.float32)},
            "Contains 1 NaN or Inf",
        ),
        (
            "colors_nan",
            {"colors": np.array([[1.0, np.nan, 0.0]], dtype=np.float32)},
            "colors: Contains 1 NaN or Inf",
        ),
        (
            "sharpness_nan",
            {"sharpness": np.array([np.nan, 1.0], dtype=np.float32)},
            "sharpness: Contains 1 NaN or Inf",
        ),
    ],
)
def test_non_finite_line_attributes_rejected(
    tmp_path, field: str, kwargs: dict, error_pattern: str
) -> None:
    """NaN/Inf values must fail before corrupting stored Zarr arrays.

    Mirror of test_non_finite_point_attributes_rejected — both rely on
    the shared validation in `luxar.validation.base`.
    """
    store = tmp_path / f"bad_{field}.zarr"
    vertices = kwargs.pop(
        "vertices", np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    )
    n_v = vertices.shape[0]
    kwargs.setdefault("widths", np.ones(n_v, dtype=np.float32) * 0.1)

    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises((ValueError, ValidationError), match=error_pattern):
            compiler.write_lines("bad", vertices, **kwargs)


def test_integer_color_ranges_do_not_emit_hdr_warning() -> None:
    """Integer SDR colors should not trigger HDR warning — symmetry with Points."""
    colors = np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_colors_for_writing(colors, n_points=2)
    assert caught == []


# =============================================================================
# Widths Validation Tests (Parametrized) — symmetric to Radii
# =============================================================================


@pytest.mark.parametrize(
    "widths_factory,error_pattern,test_id",
    [
        # Negative values
        (
            lambda: np.random.uniform(-1.0, 1.0, 100).astype(np.float32),
            "Widths must be positive",
            "negative_values",
        ),
        # Wrong count (50 instead of 100)
        (
            lambda: np.random.uniform(0.1, 2.0, 50).astype(np.float32),
            "doesn't match",
            "count_mismatch",
        ),
        # All zeros (edge case)
        (
            lambda: np.zeros(100, dtype=np.float32),
            "Widths must be positive",
            "all_zeros",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_invalid_widths(tmp_path, widths_factory, error_pattern, test_id) -> None:
    """Various invalid widths configurations raise ValueError."""
    store = tmp_path / f"invalid_widths_{test_id}.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        vertices = _polyline_vertices(100)
        with pytest.raises((ValueError, ValidationError), match=error_pattern):
            compiler.write_lines("test", vertices, widths=widths_factory())


def test_scalar_width_zero_rejected(tmp_path) -> None:
    """Scalar width must be > 0 — pin the boundary."""
    store = tmp_path / "scalar_zero_width.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        vertices = _polyline_vertices(50)
        with pytest.raises(ValueError, match="Width must be positive"):
            compiler.write_lines("test", vertices, widths=0.0)


def test_scalar_width_negative_rejected(tmp_path) -> None:
    """Scalar negative width rejected."""
    store = tmp_path / "scalar_neg_width.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        vertices = _polyline_vertices(50)
        with pytest.raises(ValueError, match="Width must be positive"):
            compiler.write_lines("test", vertices, widths=-0.5)


def test_valid_widths(tmp_path) -> None:
    """Valid widths are accepted and stored correctly."""
    store = tmp_path / "widths_test.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=EncodingMode.PRECISION, enable_spatial_index=False
    ) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        vertices = _polyline_vertices(100)
        widths = np.random.uniform(0.1, 2.0, 100).astype(np.float32)
        compiler.write_lines("test", vertices, widths=widths)

    root = zarr.open_group(store, mode="r")
    assert "test/widths" in root

    decoder = ArrayDecoder()
    stored = decoder.decode(root["test/widths"], root)
    np.testing.assert_array_equal(stored, widths)


# =============================================================================
# Sharpness Validation Tests (Parametrized) — same constraints as Points
# =============================================================================


@pytest.mark.parametrize(
    "sharpness_factory,error_pattern,test_id",
    [
        (
            lambda: np.random.uniform(-1.0, 1.0, 100).astype(np.float32),
            "Sharpness must be positive",
            "negative_values",
        ),
        (
            lambda: np.random.uniform(0.5, 10.0, 50).astype(np.float32),
            "doesn't match",
            "count_mismatch",
        ),
        (
            lambda: np.zeros(100, dtype=np.float32),
            "Sharpness must be positive",
            "all_zeros",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_invalid_sharpness(tmp_path, sharpness_factory, error_pattern, test_id) -> None:
    """Lines sharpness rejection mirrors Points."""
    store = tmp_path / f"invalid_sharpness_{test_id}.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        vertices = _polyline_vertices(100)
        widths = np.ones(100, dtype=np.float32) * 0.1
        with pytest.raises((ValueError, ValidationError), match=error_pattern):
            compiler.write_lines(
                "test", vertices, widths=widths, sharpness=sharpness_factory()
            )


def test_valid_sharpness(tmp_path) -> None:
    """Valid sharpness values stored correctly for lines."""
    store = tmp_path / "lines_sharp_test.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=EncodingMode.PRECISION, enable_spatial_index=False
    ) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        vertices = _polyline_vertices(100)
        widths = np.ones(100, dtype=np.float32) * 0.1
        sharpness = np.random.uniform(0.5, 10.0, 100).astype(np.float32)
        compiler.write_lines("test", vertices, widths=widths, sharpness=sharpness)

    root = zarr.open_group(store, mode="r")
    assert "test/sharpnesses" in root

    decoder = ArrayDecoder()
    stored = decoder.decode(root["test/sharpnesses"], root)
    np.testing.assert_array_equal(stored, sharpness)


def test_sharpness_warning(tmp_path) -> None:
    """Out-of-range sharpness for lines triggers the same warning as Points."""
    store = tmp_path / "lines_sharp_warn.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        vertices = _polyline_vertices(100)
        widths = np.ones(100, dtype=np.float32) * 0.1
        sharpness = np.array([0.3, 2.0, 15.0] * 33 + [5.0]).astype(np.float32)

        with pytest.warns(UserWarning, match="Using extreme sharpness values"):
            compiler.write_lines("test", vertices, widths=widths, sharpness=sharpness)
