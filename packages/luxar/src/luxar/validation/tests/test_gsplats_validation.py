"""Tests for Gaussian-splat data validation.

This module mirrors `test_points_validation.py` for GSplats geometry. The
Points/Lines/GSplats symmetry rule requires that every Points validation
test has a parallel GSplats variant.

For GSplats the per-splat scalar attributes are:
- ``amplitudes`` (mandatory, must be non-negative — parallel to Points ``radii``
  but with a strict ``>= 0`` rule rather than ``> 0``; an amplitude of zero
  is a valid, fully transparent splat).
- ``cholesky_factors`` (mandatory, packed lower-triangular per splat).
- ``colors`` (optional, per-splat).

GSplats do NOT have a per-splat ``sharpness`` attribute — sharpness is
implicit in the Cholesky covariance.
"""

import warnings

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.validation import ValidationError, validate_colors_for_writing


def _gsplat_centers(n: int, d: int = 3) -> np.ndarray:
    return np.random.randn(n, d).astype(np.float32)


def _packed_chol(n: int, d: int = 3) -> np.ndarray:
    """Build a deterministic packed lower-triangular Cholesky per splat."""
    k = d * (d + 1) // 2
    arr = np.zeros((n, k), dtype=np.float32)
    # Identity diagonal — column indices for the diagonal of packed (row-major)
    # lower-triangular are 0, 2, 5, 9, ... = sum(1..(i+1)) - 1 for i in 0..d-1.
    diag_cols = [(i * (i + 1)) // 2 + i for i in range(d)]
    for col in diag_cols:
        arr[:, col] = 1.0
    return arr


# =============================================================================
# Center Validation Tests
# =============================================================================


def test_bad_centers_shape(tmp_path) -> None:
    """Centers must be 2D array with shape (N, ndim) — mirrors test_bad_positions_shape."""
    store = tmp_path / "bad.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        bad_centers = np.ones((3,), dtype=np.float32)  # 1D
        with pytest.raises((ValueError, ValidationError)):
            compiler.write_gsplats(
                "Broken",
                bad_centers,
                amplitudes=1.0,
                cholesky_factors=_packed_chol(1),
            )


def test_mismatched_colors(tmp_path) -> None:
    """Colors array length must match centers — mirrors points equivalent."""
    store = tmp_path / "bad_colors.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        bad_colors = np.ones((5, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            compiler.write_gsplats(
                "Nope",
                centers,
                amplitudes=1.0,
                cholesky_factors=_packed_chol(10),
                colors=bad_colors,
            )


@pytest.mark.parametrize(
    "field,kwargs,error_pattern",
    [
        (
            "centers_nan",
            {"centers": np.array([[0.0, np.nan, 1.0]], dtype=np.float32)},
            "Contains 1 NaN or Inf",
        ),
        (
            "centers_inf",
            {"centers": np.array([[0.0, np.inf, 1.0]], dtype=np.float32)},
            "Contains 1 NaN or Inf",
        ),
        (
            "colors_nan",
            {"colors": np.array([[1.0, np.nan, 0.0]], dtype=np.float32)},
            "colors: Contains 1 NaN or Inf",
        ),
    ],
)
def test_non_finite_gsplat_attributes_rejected(
    tmp_path, field: str, kwargs: dict, error_pattern: str
) -> None:
    """NaN/Inf values must fail before corrupting stored Zarr arrays."""
    store = tmp_path / f"bad_{field}.zarr"
    centers = kwargs.pop("centers", np.array([[0.0, 1.0, 2.0]], dtype=np.float32))
    n = centers.shape[0]
    kwargs.setdefault("amplitudes", np.ones(n, dtype=np.float32))
    kwargs.setdefault("cholesky_factors", _packed_chol(n))

    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises((ValueError, ValidationError), match=error_pattern):
            compiler.write_gsplats("bad", centers, **kwargs)


def test_integer_color_ranges_do_not_emit_hdr_warning() -> None:
    """Integer SDR colors should not trigger HDR warning — symmetry with Points."""
    colors = np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_colors_for_writing(colors, n_points=2)
    assert caught == []


# =============================================================================
# Amplitude Validation Tests (Parametrized) — analogue of Points radii
# =============================================================================


@pytest.mark.parametrize(
    "amplitudes_factory,error_pattern,test_id",
    [
        # Negative values — but zero is allowed for amplitudes!
        (
            lambda: np.random.uniform(-1.0, -0.1, 100).astype(np.float32),
            "Amplitudes must be non-negative",
            "negative_values",
        ),
        # Wrong count
        (
            lambda: np.random.uniform(0.1, 2.0, 50).astype(np.float32),
            "doesn't match",
            "count_mismatch",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_invalid_amplitudes(
    tmp_path, amplitudes_factory, error_pattern, test_id
) -> None:
    """Negative/mismatched amplitudes rejected. Zero IS allowed (transparent splat)."""
    store = tmp_path / f"invalid_amps_{test_id}.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(100)
        with pytest.raises((ValueError, ValidationError), match=error_pattern):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=amplitudes_factory(),
                cholesky_factors=_packed_chol(100),
            )


def test_zero_amplitudes_allowed(tmp_path) -> None:
    """All-zero amplitudes ARE valid for GSplats (unlike radii for Points).

    This is the documented difference between the two geometries — a
    zero-amplitude splat is fully transparent but still a legitimate
    placeholder in the array.
    """
    store = tmp_path / "zero_amps.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        compiler.write_gsplats(
            "test",
            centers,
            amplitudes=np.zeros(10, dtype=np.float32),
            cholesky_factors=_packed_chol(10),
        )


def test_scalar_amplitude_negative_rejected(tmp_path) -> None:
    """Scalar negative amplitude rejected."""
    store = tmp_path / "neg_scalar_amp.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        with pytest.raises(ValueError, match="Amplitude must be non-negative"):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=-0.5,
                cholesky_factors=_packed_chol(10),
            )


def test_scalar_amplitude_zero_allowed(tmp_path) -> None:
    """Scalar amplitude = 0 is the boundary — must NOT raise."""
    store = tmp_path / "zero_scalar_amp.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        compiler.write_gsplats(
            "test",
            centers,
            amplitudes=0.0,
            cholesky_factors=_packed_chol(10),
        )


# =============================================================================
# Cholesky Factor Validation Tests
# =============================================================================


def test_cholesky_count_mismatch_rejected(tmp_path) -> None:
    """Cholesky factor count must match splat count (or be 1D for uniform)."""
    store = tmp_path / "chol_count.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        bad_chol = _packed_chol(5)  # wrong count
        with pytest.raises(ValueError, match="Cholesky factors shape mismatch"):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=1.0,
                cholesky_factors=bad_chol,
            )


def test_cholesky_packed_size_mismatch_rejected(tmp_path) -> None:
    """Cholesky packed-length must equal d*(d+1)/2 for given ndim."""
    store = tmp_path / "chol_packed.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)  # 3D → packed k = 6
        bad_chol = np.zeros((10, 7), dtype=np.float32)  # wrong packed size
        with pytest.raises(ValueError, match="Cholesky factors shape mismatch"):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=1.0,
                cholesky_factors=bad_chol,
            )


def test_uniform_cholesky_accepted(tmp_path) -> None:
    """1D Cholesky (broadcast / uniform across all splats) is allowed."""
    store = tmp_path / "uniform_chol.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=EncodingMode.PRECISION, enable_spatial_index=False
    ) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(50)
        uniform_chol = _packed_chol(1).reshape(-1)  # 1D
        compiler.write_gsplats(
            "test",
            centers,
            amplitudes=1.0,
            cholesky_factors=uniform_chol,
        )


# =============================================================================
# Valid happy-path
# =============================================================================


def test_valid_gsplats(tmp_path) -> None:
    """Valid GSplats round-trip end to end with no errors."""
    store = tmp_path / "valid_gsplats.luxar.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=EncodingMode.PRECISION, enable_spatial_index=False
    ) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(100)
        amplitudes = np.random.uniform(0.1, 1.0, 100).astype(np.float32)
        cholesky_factors = _packed_chol(100)
        compiler.write_gsplats(
            "test",
            centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky_factors,
        )
