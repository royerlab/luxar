"""Property-based tests for the spatial-ordering primitives.

Complement the example-based ordering tests with two invariants the per-geometry
suites only pin at hand-picked values:

1. The Morton/Hilbert encoders are pure, permutation-*equivariant* functions of
   the coordinates: reordering the input rows reorders the output codes
   identically, and the multiset of codes is invariant. This is what lets the
   ordering be a stable spatial sort regardless of the order splats/points
   arrive in.
2. ``_ordering/README.md`` Key Invariant 6 — a stored ``chunk_bounds`` interval
   contains the chunk's geometric footprint at ANY coordinate magnitude — swept
   across magnitudes and pads for all four bound builders.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from luxar.io._ordering.bounds import _store_outward_f32, _store_outward_f32_array
from luxar.io.ordering import (
    _BARRIER_BOUND_EPS,
    compute_chunk_bounds_gsplats,
    compute_chunk_bounds_points,
    compute_segment_chunk_bounds,
    compute_vertex_chunk_bounds,
    hilbert_encode_nd,
    morton_encode_nd,
    sort_splats_spatial,
)


@given(
    n=st.integers(min_value=1, max_value=64),
    d=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_morton_is_permutation_equivariant(n: int, d: int, data: st.DataObject) -> None:
    """morton(coords[perm]) == morton(coords)[perm] — a pure function of the
    coordinates, so a spatial sort is independent of input row order."""
    coords = data.draw(hnp.arrays(np.int64, (n, d), elements=st.integers(0, 2**16 - 1)))
    perm = data.draw(st.permutations(list(range(n))))
    perm_arr = np.asarray(perm, dtype=np.intp)
    base = morton_encode_nd(coords, bits_per_dim=16)
    permuted = morton_encode_nd(coords[perm_arr], bits_per_dim=16)
    np.testing.assert_array_equal(permuted, base[perm_arr])


@given(
    n=st.integers(min_value=1, max_value=64),
    d=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_hilbert_is_permutation_equivariant(
    n: int, d: int, data: st.DataObject
) -> None:
    """Same equivariance for the Hilbert encoder."""
    coords = data.draw(hnp.arrays(np.int64, (n, d), elements=st.integers(0, 2**16 - 1)))
    perm = data.draw(st.permutations(list(range(n))))
    perm_arr = np.asarray(perm, dtype=np.intp)
    base = hilbert_encode_nd(coords, bits_per_dim=16)
    permuted = hilbert_encode_nd(coords[perm_arr], bits_per_dim=16)
    np.testing.assert_array_equal(permuted, base[perm_arr])


# --- numba path vs numpy-fallback parity -------------------------------------
# The encoders have a Numba-JIT fast path and a pure-NumPy fallback (used when
# Numba is unavailable). With Numba installed the property tests above only
# exercise the JIT path; these deterministic tests force the fallback and assert
# it produces byte-identical codes — so the fallback is verified AND kept in
# parity with the JIT kernel.

# The lazy-compiled numba kernels are cached as module globals inside the
# curve encoders (io/_ordering/curves/{morton,hilbert}.py); patch them there.
import luxar.io._ordering.curves.hilbert as _hilbert_mod  # noqa: E402
import luxar.io._ordering.curves.morton as _morton_mod  # noqa: E402


def _encode_forcing_numpy_fallback(fn, coords: np.ndarray) -> np.ndarray:
    """Run ``fn`` with both Numba kernels forced OFF (pure-NumPy path)."""
    saved = (_morton_mod._morton_numba_kernel, _hilbert_mod._hilbert_numba_kernel)
    try:
        _morton_mod._morton_numba_kernel = False
        _hilbert_mod._hilbert_numba_kernel = False
        return fn(coords, bits_per_dim=16)
    finally:
        _morton_mod._morton_numba_kernel = saved[0]
        _hilbert_mod._hilbert_numba_kernel = saved[1]


def _encode_forcing_numba(fn, coords: np.ndarray) -> np.ndarray:
    """Run ``fn`` with the kernels reset to None so the JIT path lazy-loads
    (falls back to NumPy only if Numba is genuinely unavailable)."""
    saved = (_morton_mod._morton_numba_kernel, _hilbert_mod._hilbert_numba_kernel)
    try:
        _morton_mod._morton_numba_kernel = None
        _hilbert_mod._hilbert_numba_kernel = None
        return fn(coords, bits_per_dim=16)
    finally:
        _morton_mod._morton_numba_kernel = saved[0]
        _hilbert_mod._hilbert_numba_kernel = saved[1]


def test_morton_numba_numpy_parity() -> None:
    """morton: JIT path and NumPy fallback yield byte-identical codes."""
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 2**16, size=(256, 3), dtype=np.int64)
    np.testing.assert_array_equal(
        _encode_forcing_numpy_fallback(morton_encode_nd, coords),
        _encode_forcing_numba(morton_encode_nd, coords),
    )


def test_hilbert_numba_numpy_parity() -> None:
    """hilbert: JIT path and NumPy fallback yield byte-identical codes."""
    rng = np.random.default_rng(1)
    coords = rng.integers(0, 2**16, size=(256, 3), dtype=np.int64)
    np.testing.assert_array_equal(
        _encode_forcing_numpy_fallback(hilbert_encode_nd, coords),
        _encode_forcing_numba(hilbert_encode_nd, coords),
    )


# --- Key Invariant 6: never tighter than the footprint, at ANY magnitude ------
# The per-geometry suites pin this with hand-picked examples at |x| = 2e7 only,
# which is one point on a curve whose whole difficulty is scale. Sweep the
# float32 regimes instead: near-zero coordinates (only 0.0 is below the float32
# ULP of 1.0, 1.19e-7; 1e-6 is just above it), the 2**23/2**24 boundaries
# where a float32 ULP crosses 1.0 and then 2.0, and on out to the float32 max.
# A handful of elements per case keeps the whole sweep well under a second.

_MAGNITUDES = [
    0.0,
    1e-6,
    1.0,
    100.0,
    2.0**23,  # ULP becomes 1.0 — where an absolute pad starts vanishing
    2.0**24,  # ULP becomes 2.0; integers stop being exactly representable
    2.0**24 + 1,
    2.0**25,
    1e9,
    # From 1e15 up the pad no longer survives FLOAT64 either, so these cells
    # assert less than they look like they do. At 1e15 the float64 spacing is
    # 0.125, so ``m - 1e-3 == m`` and every small pad is already gone from the
    # builder's float64 accumulation AND from this test's expected footprint,
    # identically — the assertion degrades to plain coordinate containment (the
    # 100.0 pad still resolves there). By 1e20 the spacing is 1.638e4 and even
    # that one collapses. Not a bug — a float64 accumulator cannot preserve what
    # float64 cannot represent — but do not read the 3e38 cell as evidence that
    # a pad is preserved. The regime this fix is actually about is 2**23..1e9,
    # where float64 holds the pad and float32 does not.
    1e15,
    1e20,
    1e30,
    3e38,  # just under the float32 max
]

# Every pad the builders add is a small ABSOLUTE quantity; span the range from
# "far below a float32 ULP everywhere" to "comfortably above one".
_PADS = [1e-6, 0.05, 0.1, 0.5, 1.0, 100.0]

_N_ELEMENTS = 4
_NDIM = 3


def _cluster(magnitude: float, sign: float) -> np.ndarray:
    """A few float32 coordinates clustered around ``sign * magnitude``."""
    base = sign * magnitude
    offsets = np.arange(_N_ELEMENTS, dtype=np.float64)
    return np.tile((base + offsets)[:, None], (1, _NDIM)).astype(np.float32)


def _assert_contains(
    bounds: np.ndarray, coords: np.ndarray, pad: float, label: str
) -> None:
    """Stored float32 interval must contain the float64 footprint on every dim."""
    for d in range(coords.shape[1]):
        col = coords[:, d].astype(np.float64)
        lo = np.float64(bounds[0, d, 0])
        hi = np.float64(bounds[0, d, 1])
        assert lo <= col.min() - pad, (label, d, lo, col.min(), pad)
        assert hi >= col.max() + pad, (label, d, hi, col.max(), pad)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["pos", "neg"])
@pytest.mark.parametrize("magnitude", _MAGNITUDES, ids=[repr(m) for m in _MAGNITUDES])
def test_chunk_bounds_contain_the_footprint_at_every_magnitude(
    magnitude: float, sign: float
) -> None:
    """Key Invariant 6 for all FOUR bound builders, swept over magnitude × pad.

    Each builder adds a small absolute pad to a coordinate that may be large:
    a point radius, a gsplat's ``coverage_sigma·σ``, a line's full endpoint
    width, and ``_BARRIER_BOUND_EPS`` on a categorical axis. Past ``|x| ~ 2**23``
    such a pad is under half a float32 ULP, so a float32 accumulate plus a
    round-to-nearest store used to leave the stored bound TIGHTER than the
    footprint the renderer draws — dropping elements from queries at their own
    edge. Float64 accumulation plus the outward store must make containment hold
    everywhere on this grid.
    """
    coords = _cluster(magnitude, sign)
    segments = np.column_stack(
        [
            np.arange(_N_ELEMENTS - 1, dtype=np.uint32),
            np.arange(1, _N_ELEMENTS, dtype=np.uint32),
        ]
    )

    for pad in _PADS:
        # Points: spatial axes padded by the radius (scalar and per-point).
        pad32 = float(np.float32(pad))
        per_point = np.full(_N_ELEMENTS, pad32, dtype=np.float32)
        for radii in (pad32, per_point):
            _assert_contains(
                compute_chunk_bounds_points(
                    coords, radii=radii, chunk_size=_N_ELEMENTS
                ),
                coords,
                pad32,
                f"points pad={pad}",
            )

        # GSplats: extent = sqrt(covariance[d,d]) * coverage_sigma. Recompute it
        # the way the builder does so the expectation carries no extra rounding.
        sigma = np.float32(pad)
        chol = np.array([sigma, 0, sigma, 0, 0, sigma], dtype=np.float32)  # diagonal L
        s64 = np.float64(sigma)
        extent = float(np.sqrt(s64 * s64))  # coverage_sigma = 1.0 below
        for chol_arg in (chol.reshape(1, 6), np.tile(chol, (_N_ELEMENTS, 1))):
            _assert_contains(
                compute_chunk_bounds_gsplats(
                    coords, chol_arg, chunk_size=_N_ELEMENTS, coverage_sigma=1.0
                ),
                coords,
                extent,
                f"gsplats pad={pad}",
            )

        # Lines segments: the FULL per-segment max endpoint width (no /2).
        widths = np.full(_N_ELEMENTS, pad32, dtype=np.float32)
        seg_bounds = compute_segment_chunk_bounds(
            coords, segments, widths, chunk_size=len(segments)
        )
        # The chunk covers vertices 0..N-1 (every vertex appears in a segment).
        _assert_contains(seg_bounds, coords, pad32, f"segments pad={pad}")

    # Barrier arm (all four builders pad a categorical axis by _BARRIER_BOUND_EPS
    # and nothing else). Axis 0 is the barrier; axes 1-2 keep their extent.
    barrier_pad = _BARRIER_BOUND_EPS
    barrier_coords = coords[:, :1]
    widths = np.full(_N_ELEMENTS, 0.5, dtype=np.float32)
    chol = np.tile(
        np.array([0.5, 0, 0.5, 0, 0, 0.5], dtype=np.float32), (_N_ELEMENTS, 1)
    )
    for label, bounds in (
        (
            "points",
            compute_chunk_bounds_points(
                coords, radii=0.5, chunk_size=_N_ELEMENTS, slice_dims=[0]
            ),
        ),
        (
            "gsplats",
            compute_chunk_bounds_gsplats(
                coords, chol, chunk_size=_N_ELEMENTS, coverage_sigma=1.0, slice_dims=[0]
            ),
        ),
        (
            "vertices",
            compute_vertex_chunk_bounds(coords, _N_ELEMENTS, slice_dims=[0]),
        ),
        (
            "segments",
            compute_segment_chunk_bounds(
                coords, segments, widths, chunk_size=len(segments), slice_dims=[0]
            ),
        ),
    ):
        _assert_contains(bounds[:, :1], barrier_coords, barrier_pad, f"{label} barrier")


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["pos", "neg"])
@pytest.mark.parametrize("magnitude", _MAGNITUDES, ids=[repr(m) for m in _MAGNITUDES])
def test_padless_vertex_bounds_are_exact_at_every_magnitude(
    magnitude: float, sign: float
) -> None:
    """A vertex carries no spatial pad, so its stored bound must be EXACT.

    The other half of the outward store's contract: it steps a bound by one ULP
    only when the float32 cast moved it the wrong way, never unconditionally.
    ``compute_vertex_chunk_bounds`` on purely spatial dims has nothing to lose
    (the inputs are already float32), so equality — not containment — is the
    right assertion, and it is what would fail if the store widened everything.
    """
    coords = _cluster(magnitude, sign)
    bounds = compute_vertex_chunk_bounds(coords, _N_ELEMENTS)
    for d in range(_NDIM):
        assert bounds[0, d, 0] == coords[:, d].min()
        assert bounds[0, d, 1] == coords[:, d].max()


def test_out_of_range_slice_dims_raise_in_every_builder() -> None:
    """An unresolvable barrier index fails loudly instead of being ignored.

    The four builders used to sanitise ``slice_dims`` two different ways even
    though they now sit behind one ``bounds.py``: gsplats coerced with ``int()``
    and indexed, so a positive index raised ``IndexError`` deep in the loop while
    a negative one silently resolved to the LAST dim; points and both lines
    builders used a membership test that ignored out-of-range and negative
    entries outright. An out-of-range index means a categorical axis silently
    loses its barrier treatment and takes the geometric-extent expansion
    instead — a chunk bleeding into its neighbour's category, with nothing to
    notice — so all four reject it with the same ``ValueError``.
    """
    coords = np.zeros((4, 3), dtype=np.float32)
    segments = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    widths = np.ones(4, dtype=np.float32)
    chol = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (4, 1))

    for bad in ([7], [-1]):
        with pytest.raises(ValueError, match="out of range"):
            compute_chunk_bounds_points(coords, radii=0.5, chunk_size=4, slice_dims=bad)
        with pytest.raises(ValueError, match="out of range"):
            compute_chunk_bounds_gsplats(coords, chol, chunk_size=4, slice_dims=bad)
        with pytest.raises(ValueError, match="out of range"):
            compute_vertex_chunk_bounds(coords, 4, slice_dims=bad)
        with pytest.raises(ValueError, match="out of range"):
            compute_segment_chunk_bounds(
                coords, segments, widths, chunk_size=2, slice_dims=bad
            )


def test_out_of_range_slice_dims_raise_in_the_gsplat_sort_too() -> None:
    """The gsplat SORT rejects a bad barrier index at the first door.

    ``apply_gsplat_spatial_ordering`` hands one ``slice_dims`` list to both
    ``sort_splats_spatial`` and ``compute_chunk_bounds_gsplats``. A positive
    out-of-range index already died inside ``_compound_sort``
    (``coords[:, slice_dims]`` → ``IndexError``), but a NEGATIVE one used to sort
    silently: ``ordering_dims`` complements over ``range(ndim)``, so the last
    column was lexsorted as a barrier AND spatially curve-coded, and the raw
    ``-1`` was then persisted into the store's ``slice_dims`` attr (which the
    viewer rejects wholesale). See :func:`_normalise_slice_dims` for the full
    argument. Both now fail on the same ``ValueError``, before any splat moves.
    """
    centers = np.zeros((4, 3), dtype=np.float32)
    for bad in ([7], [-1]):
        with pytest.raises(ValueError, match="out of range"):
            sort_splats_spatial(centers, slice_dims=bad)


def test_outward_store_array_rejects_float32_input() -> None:
    """The array store refuses anything but float64, loudly.

    A pad already rounded away by float32 arithmetic upstream cannot be
    recovered by rounding outward, so accepting a float32 interval here would
    quietly hand back an under-tight bound that only shows up at large
    coordinate magnitudes. Refuse it instead of casting it up.
    """
    lo = np.array([1.0], dtype=np.float32)
    hi = np.array([2.0], dtype=np.float32)
    with pytest.raises(TypeError, match="requires float64 bounds"):
        _store_outward_f32_array(lo, hi)
    with pytest.raises(TypeError, match="requires float64 bounds"):
        _store_outward_f32_array(lo.astype(np.float64), hi)


def test_outward_store_scalar_contains_and_matches_the_array_form() -> None:
    """The scalar store (used by Points) obeys the same contract as the array one.

    Only the array form's dtype guard was pinned; the scalar form — the one every
    Points chunk goes through — had no direct test at all. Both halves of the
    contract are asserted here: the stored float32 interval CONTAINS the float64
    one at every magnitude, and the two implementations agree bit for bit (they
    are the same rule written twice, so they can drift).
    """
    intervals = [
        (0.0, 0.0),
        (-1e-6, 1e-6),
        (0.9999999, 1.0000001),
        (-0.5, 100.5),
        (2.0**23 - 0.5, 2.0**23 + 0.5),
        (2.0**24 - 1e-3, 2.0**24 + 1e-3),
        (1e9 - 0.1, 1e9 + 0.1),
        (-3.0e38, 3.0e38),
    ]
    for lo, hi in intervals:
        lo32, hi32 = _store_outward_f32(lo, hi)
        assert float(lo32) <= lo, (lo, hi, lo32)
        assert float(hi32) >= hi, (lo, hi, hi32)

    lo_arr = np.array([lo for lo, _ in intervals], dtype=np.float64)
    hi_arr = np.array([hi for _, hi in intervals], dtype=np.float64)
    lo_vec, hi_vec = _store_outward_f32_array(lo_arr, hi_arr)
    for i, (lo, hi) in enumerate(intervals):
        lo32, hi32 = _store_outward_f32(lo, hi)
        assert lo_vec[i] == lo32, (lo, hi)
        assert hi_vec[i] == hi32, (lo, hi)


def test_outward_store_is_silent_past_the_float32_ceiling() -> None:
    """An out-of-float32-range bound must not warn, in either form.

    ``np.float32(1e39)`` overflows to ``inf`` with a ``RuntimeWarning``. ``inf``
    is the RIGHT answer for a bound (it over-fetches; it can never drop a chunk),
    but several io tests wrap a whole compile in ``-W error``, where that warning
    would abort the save. Both helpers suppress it locally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lo32, hi32 = _store_outward_f32(-1e39, 1e39)
        lo_vec, hi_vec = _store_outward_f32_array(
            np.array([-1e39, 1e300], dtype=np.float64),
            np.array([1e39, 1e300], dtype=np.float64),
        )

    # Containment still holds: the low end saturates to -inf, the high to +inf.
    assert float(lo32) == -np.inf
    assert float(hi32) == np.inf
    assert lo_vec[0] == -np.float32(np.inf)
    assert hi_vec[0] == np.float32(np.inf)
    # A finite float64 interval far above the float32 max: the low end must still
    # land at or below it (the largest finite float32), the high end at +inf.
    assert float(lo_vec[1]) <= 1e300
    assert float(hi_vec[1]) == np.inf
