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
        (
            "colors_inf",
            {"colors": np.array([[1.0, np.inf, 0.0]], dtype=np.float32)},
            "colors: Contains 1 NaN or Inf",
        ),
        # Amplitudes NaN/Inf: a NaN would silently pass the `>= 0` check
        # (nan < 0 is False) and corrupt the store — the bug class the
        # Points/Lines size-scalar finiteness checks already catch.
        (
            "amplitudes_nan",
            {"amplitudes": np.array([np.nan], dtype=np.float32)},
            "amplitudes: Contains 1 NaN or Inf",
        ),
        (
            "amplitudes_inf",
            {"amplitudes": np.array([np.inf], dtype=np.float32)},
            "amplitudes: Contains 1 NaN or Inf",
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


def test_cholesky_nan_rejected(tmp_path) -> None:
    """A NaN anywhere in the Cholesky factors must fail before any write.

    A single NaN used to pass the shape-only gate and die deep in the encoder
    AFTER centers were already on disk (a half-written node) — the exact
    failure the amplitudes/radii/widths finiteness checks already prevent.
    """
    # Use the canonical suffix so the store path is NOT rewritten
    # (LuxarZarrCompiler normalizes ``foo.zarr`` → ``foo.luxar.zarr``); this
    # makes the no-partial-node assertion below meaningful.
    store = tmp_path / "chol_nan.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        bad_chol = _packed_chol(10)
        bad_chol[3, 1] = np.nan  # off-diagonal NaN
        # Match the validator's own context-prefixed message so this proves the
        # early gate fired — not the deep encoder error (which also says
        # "Contains 1 NaN or Inf" but without the "cholesky_factors:" prefix).
        with pytest.raises(
            (ValueError, ValidationError), match="cholesky_factors: Contains"
        ):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=1.0,
                cholesky_factors=bad_chol,
            )
    # Rejected BEFORE any write: neither the node group nor its centers array
    # may exist on disk (the transactional fail-fast property of the fix).
    assert not (store / "test").exists()
    assert not (store / "test" / "centers").exists()


def test_cholesky_zero_diagonal_rejected(tmp_path) -> None:
    """A zero on the Cholesky diagonal is a singular covariance — rejected."""
    store = tmp_path / "chol_zero_diag.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        bad_chol = _packed_chol(10)
        bad_chol[4, 2] = 0.0  # diagonal slot for d=3 is column 2
        with pytest.raises(
            (ValueError, ValidationError),
            match="Cholesky diagonal must be positive",
        ):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=1.0,
                cholesky_factors=bad_chol,
            )
    # Rejected before write — no partial node on disk.
    assert not (store / "test").exists()
    assert not (store / "test" / "centers").exists()


def test_cholesky_negative_diagonal_rejected(tmp_path) -> None:
    """A negative Cholesky diagonal (silently clamped by the uint8 encoder)."""
    store = tmp_path / "chol_neg_diag.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        bad_chol = _packed_chol(10)
        bad_chol[7, 5] = -2.0  # diagonal slot for d=3 includes column 5
        with pytest.raises(
            (ValueError, ValidationError),
            match="Cholesky diagonal must be positive",
        ):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=1.0,
                cholesky_factors=bad_chol,
            )
    assert not (store / "test").exists()


def test_uniform_cholesky_bad_diagonal_rejected(tmp_path) -> None:
    """A 1D uniform/broadcast Cholesky with a non-positive diagonal is rejected.

    Exercises the shape-normalization path (``(k,)`` → ``(1, k)``) before the
    diagonal check runs.
    """
    store = tmp_path / "uniform_bad_diag.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(50)
        uniform_chol = _packed_chol(1).reshape(-1)  # 1D, k=6
        uniform_chol[0] = 0.0  # first diagonal slot
        with pytest.raises(
            (ValueError, ValidationError),
            match="Cholesky diagonal must be positive",
        ):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=1.0,
                cholesky_factors=uniform_chol,
            )


def test_cholesky_negative_offdiagonal_accepted(tmp_path) -> None:
    """Negative OFF-diagonals are valid — the gate constrains only the diagonal.

    A real lower-triangular Cholesky factor has strictly positive DIAGONAL
    entries but arbitrarily-signed off-diagonals. This locks in that the
    validator rejects non-positive diagonals ONLY and never touches the
    off-diagonal (signed) slots.
    """
    store = tmp_path / "neg_offdiag.luxar.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=EncodingMode.PRECISION, enable_spatial_index=False
    ) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(4)
        # Packed lower-tri (d=3): diagonal slots [0, 2, 5] positive; the
        # off-diagonal slots [1, 3, 4] carry negative values.
        row = np.array([1.0, -0.5, 1.0, 0.3, -0.7, 1.0], dtype=np.float32)
        cholesky = np.tile(row, (4, 1))
        compiler.write_gsplats(
            "test",
            centers,
            amplitudes=1.0,
            cholesky_factors=cholesky,
        )
    # Wrote successfully: the node and its centers array are present.
    assert (store / "test").exists()
    assert (store / "test" / "centers").exists()


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


def test_multi_additive_bad_later_level_leaves_no_partial_node(tmp_path) -> None:
    """A bad LATER additive level must leave NO partial node on disk.

    Reproduces the reviewer's scenario through the PUBLIC scene API: a
    two-level additive-ladder ``GSplatData`` whose level 0 is valid and whose
    level 1 has a zero on the Cholesky diagonal. The writer's per-level pass
    validates only the level it is about to write, so without an all-or-nothing
    pre-flight gate the invalid level 1 would be caught only AFTER the parent
    node group and a complete ``additive_0/`` were committed — a half-written
    node. This locks in the pre-flight gate: no group is created when any
    sub-LOD is invalid.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

    n = 10
    good = AdditiveSubLOD(
        centers=_gsplat_centers(n),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=_packed_chol(n),
    )
    bad_chol = _packed_chol(n)
    bad_chol[2, 2] = 0.0  # diagonal slot for d=3 is column 2 — singular covariance
    bad = AdditiveSubLOD(
        centers=_gsplat_centers(n),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=bad_chol,
    )
    data = GSplatData(additive_sublods=[good, bad])

    store = tmp_path / "bad.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(
            (ValueError, ValidationError),
            match="Cholesky diagonal must be positive",
        ):
            scene.add_gsplats_from_data("bad", data)
    # Transactional: the invalid LATER level was caught before any group was
    # created, so neither the parent node nor a committed additive_0/ exists.
    assert not (store / "bad").exists()
    assert not (store / "bad" / "additive_0").exists()


def test_multi_additive_bad_later_level_colors_leaves_no_partial_node(
    tmp_path,
) -> None:
    """A bad LATER additive level's COLORS must leave NO partial node on disk.

    Locks the COLORS half of the all-or-nothing gate (the sibling of
    ``test_multi_additive_bad_later_level_leaves_no_partial_node``, which covers
    the arrays half). Colors used to be validated only deep inside
    ``write_gsplat_arrays``, AFTER that level's centers/amplitudes/Cholesky were
    already on disk — a valid level 0 followed by a level 1 with bad colors
    would commit the parent node and a complete ``additive_0/`` before failing
    (a half-written node). Colors are now part of ``validate_gsplat_inputs``,
    so the pre-flight gate catches them before any group is created; this
    asserts that gate fires up front.

    A 5-channel colors array can't reach the writer (``GSplatData`` rejects it at
    construction when it concatenates the ladder's colors), so the invalid case
    here is NaN colors on level 1, which survives construction and reaches the
    writer.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

    n = 10
    good = AdditiveSubLOD(
        centers=_gsplat_centers(n),
        amplitudes=np.ones(n, np.float32),
        cholesky_factors=_packed_chol(n),
        colors=np.ones((n, 3), np.float32),
    )
    bad = AdditiveSubLOD(
        centers=_gsplat_centers(n),
        amplitudes=np.ones(n, np.float32),
        cholesky_factors=_packed_chol(n),
        colors=np.full((n, 3), np.nan, np.float32),  # invalid: non-finite colors
    )
    data = GSplatData(additive_sublods=[good, bad])

    store = tmp_path / "bad.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(
            (ValueError, ValidationError),
            match="colors: Contains",
        ):
            scene.add_gsplats_from_data("bad", data)
    # Transactional: the bad later-level colors were caught before any group was
    # created, so neither the parent node nor a committed additive_0/ exists.
    assert not (store / "bad").exists()
    assert not (store / "bad" / "additive_0").exists()


def test_flat_write_bad_colors_leaves_no_partial_node(tmp_path) -> None:
    """Invalid colors on the FLAT path fail before any group is created.

    Colors used to be validated only inside ``write_gsplat_arrays``, AFTER
    centers/amplitudes/Cholesky were on disk — the last input whose failure
    could leave a partial node. ``validate_gsplat_inputs`` now covers colors,
    so the flat ``write_gsplats`` pre-group gate catches them too.
    """
    store = tmp_path / "bad_colors_flat.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        centers = _gsplat_centers(10)
        bad_colors = np.ones((10, 3), dtype=np.float32)
        bad_colors[4, 1] = np.nan
        with pytest.raises((ValueError, ValidationError), match="colors: Contains"):
            compiler.write_gsplats(
                "test",
                centers,
                amplitudes=1.0,
                cholesky_factors=_packed_chol(10),
                colors=bad_colors,
            )
    assert not (store / "test").exists()


def test_flat_write_bad_broadcast_color_leaves_no_partial_node(tmp_path) -> None:
    """A NaN in a BROADCAST (tuple) color fails before any group is created.

    Broadcast list/tuple colors bypass ``validate_colors_for_writing`` (an
    ndarray-only check); they get the shared ``validate_broadcast_color`` gate
    instead (the same one Points/Lines run). Without it, a non-finite tuple
    component was discovered only in ``write_colors`` — after centers,
    amplitudes, and Cholesky factors were already on disk.
    """
    store = tmp_path / "bad_bcast_color.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(
            (ValueError, ValidationError), match="Uniform color component"
        ):
            compiler.write_gsplats(
                "test",
                _gsplat_centers(10),
                amplitudes=1.0,
                cholesky_factors=_packed_chol(10),
                colors=(np.nan, 0.0, 0.0),
            )
    assert not (store / "test").exists()


def test_flat_write_wrong_length_broadcast_color_rejected(tmp_path) -> None:
    """A 2-component broadcast color is rejected pre-write (RGB(A) only)."""
    store = tmp_path / "short_bcast_color.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(
            (ValueError, ValidationError),
            match="Uniform color must have 3 .RGB. or 4 .RGBA.",
        ):
            compiler.write_gsplats(
                "test",
                _gsplat_centers(10),
                amplitudes=1.0,
                cholesky_factors=_packed_chol(10),
                colors=[0.5, 0.5],
            )
    assert not (store / "test").exists()


def test_flat_write_valid_broadcast_color_accepted(tmp_path) -> None:
    """A valid uniform RGB tuple still writes (the gate is not over-strict)."""
    store = tmp_path / "good_bcast_color.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        compiler.write_gsplats(
            "test",
            _gsplat_centers(10),
            amplitudes=1.0,
            cholesky_factors=_packed_chol(10),
            colors=(1.0, 0.5, 0.0),
        )
    assert (store / "test" / "centers").exists()


def test_leaf_value_scanned_exactly_once(tmp_path, monkeypatch) -> None:
    """The additive-ladder write value-scans each sub-LOD exactly once.

    The pre-flight gate is the single O(N) value scan; the per-level writes
    re-run only shape normalization (``check_values=False``). Counting calls
    to the Cholesky validator pins that a two-level ladder is scanned twice
    (once per sub-LOD in preflight) — not four or six times (double preflight
    plus per-level re-validation).
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData
    from luxar.validation import base as validation_base

    calls = {"n": 0}
    real = validation_base.validate_cholesky_for_writing

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(validation_base, "validate_cholesky_for_writing", counting)

    n = 10
    data = GSplatData(
        additive_sublods=[
            AdditiveSubLOD(
                centers=_gsplat_centers(n),
                amplitudes=np.ones(n, dtype=np.float32),
                cholesky_factors=_packed_chol(n),
            )
            for _ in range(2)
        ]
    )
    store = tmp_path / "once.luxar.zarr"
    with LuxarZarrCompiler(store, enable_spatial_index=False) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_data("ladder", data)
    assert calls["n"] == 2


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
