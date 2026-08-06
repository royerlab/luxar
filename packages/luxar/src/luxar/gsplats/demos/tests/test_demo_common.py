"""Unit tests for the shared gsplats-demo helpers in ``_demo_common``.

The helpers themselves are pure numpy — no napari, no GPU, no data download —
so each test runs in microseconds. Importing them is the slow part: it
initializes ``luxar.gsplats``, which eagerly pulls in torch and scipy — over a
second, once per session (the exact figure swings ~2x with machine load).
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.demos._demo_common import (
    ellipse_polygon_from_L,
    ellipsoid_wireframe_from_L,
    psnr,
)

# The eigenvalue floor both geometry helpers clip to, and the ``t`` they default
# to. A degenerate axis therefore collapses to a half-extent of t*sqrt(EPS).
_EIGENVALUE_FLOOR = 1e-12
_DEFAULT_T = 2.0


def _level_set_values(pts: np.ndarray, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Evaluate (x-mu)^T Sigma^-1 (x-mu) at every row of ``pts``, Sigma = L L^T."""
    Sigma = L @ L.T
    delta = pts.astype(np.float64) - mu[None, :]
    return np.einsum("ij,jk,ik->i", delta, np.linalg.inv(Sigma), delta)


def _smallest_axis_extent(pts: np.ndarray, L: np.ndarray) -> float:
    """Half-extent of ``pts`` along Sigma's minor axis, where a clipped
    degenerate axis shows up.

    ``pts`` MUST be centred on the origin: this projects onto the smallest
    eigenvalue's eigenvector directly, without subtracting a centre, so a nonzero
    centre would itself be measured as extent.
    """
    _, evecs = np.linalg.eigh(L @ L.T)
    return float(np.abs(pts.astype(np.float64) @ evecs[:, 0]).max())


# ---------------------------------------------------------------------------
# ellipse_polygon_from_L
# ---------------------------------------------------------------------------


def test_ellipse_shape_and_dtype() -> None:
    """The polygon is (n_pts, 2) float32, ready for a napari Shapes layer."""
    mu = np.array([3.0, -7.0])
    L = np.diag([2.0, 5.0])
    pts = ellipse_polygon_from_L(mu, L, t=2.0, n_pts=17)
    assert pts.shape == (17, 2)
    assert pts.dtype == np.float32


def test_ellipse_defaults() -> None:
    """Pin the signature defaults (t=2.0, n_pts=64).

    ``demo_boundary_containment.make_ellipse_polygons`` omits ``n_pts``, so that
    default is live. No caller relies on the ``t`` default: the six other overlay
    sites pass ``t=2.0`` explicitly and that one passes ``t=3.0``. Pin it anyway —
    it is published API, and leaving it unpinned is how it drifts.
    """
    mu = np.array([0.0, 0.0])
    sigma = np.array([2.0, 5.0])
    pts = ellipse_polygon_from_L(mu, np.diag(sigma))  # no t=, no n_pts=
    assert pts.shape == (64, 2)
    half_extent = (pts.max(axis=0) - pts.min(axis=0)) / 2.0
    np.testing.assert_allclose(half_extent, 2.0 * sigma, rtol=1e-6)


def test_ellipse_axis_aligned_extents_and_centroid() -> None:
    """A diagonal L gives an axis-aligned ellipse of half-extent t*sigma."""
    mu = np.array([10.0, -4.0])
    sigma = np.array([2.0, 5.0])
    L = np.diag(sigma)
    t = 2.0
    # n_pts divisible by 4 puts vertices exactly on the principal axes, so the
    # sampled extents reach the true ellipse extents (no polygonal shortfall).
    pts = ellipse_polygon_from_L(mu, L, t=t, n_pts=64).astype(np.float64)

    half_extent = (pts.max(axis=0) - pts.min(axis=0)) / 2.0
    np.testing.assert_allclose(half_extent, t * sigma, rtol=1e-6)
    # Uniform angular sampling of a centered ellipse averages back to the center.
    np.testing.assert_allclose(pts.mean(axis=0), mu, atol=1e-5)


def test_ellipse_points_lie_on_level_set_diagonal() -> None:
    """Every vertex satisfies (x-mu)^T Sigma^-1 (x-mu) == t^2."""
    mu = np.array([1.5, 2.5])
    L = np.diag([3.0, 0.75])
    t = 1.7
    pts = ellipse_polygon_from_L(mu, L, t=t, n_pts=64)
    np.testing.assert_allclose(
        _level_set_values(pts, mu, L), np.full(64, t**2), rtol=1e-5
    )


def test_ellipse_points_lie_on_level_set_rotated() -> None:
    """A non-diagonal L still lands on the level set (catches eigvec/eigval mixups)."""
    mu = np.array([-8.0, 4.0])
    # Lower-triangular Cholesky factor of a genuinely correlated covariance.
    L = np.array([[2.0, 0.0], [1.3, 0.9]])
    assert not np.allclose(L @ L.T, np.diag(np.diag(L @ L.T)))  # really rotated
    t = 2.0
    pts = ellipse_polygon_from_L(mu, L, t=t, n_pts=48)
    np.testing.assert_allclose(
        _level_set_values(pts, mu, L), np.full(48, t**2), rtol=1e-4
    )


def test_ellipse_vertex_order_walks_the_contour_once() -> None:
    """Consecutive vertices advance by a constant +2*pi/n_pts step.

    ``demo_boundary_containment.py`` feeds this straight to
    ``viewer.add_shapes(..., shape_type="polygon")``, which draws an edge between
    consecutive rows -- so vertex ADJACENCY is part of the contract, not just the
    point set. (The wireframe has no such constraint: ``demo_3d_synthetic_phantom``
    renders it via ``add_points``, and ``demo_3d_dapi_microscopy`` builds it into a
    list it never reads.) Permuting the sampling angles leaves the point set,
    level-set values, extents, centroid, shape and dtype all identical while
    turning the outline into a self-intersecting star: nothing else here notices.
    """
    mu = np.array([2.0, -3.0])
    L = np.array([[2.0, 0.0], [1.3, 0.9]])  # rotated, well-conditioned
    t = 2.0
    n_pts = 64
    pts = ellipse_polygon_from_L(mu, L, t=t, n_pts=n_pts)

    # Recover each vertex's parametric angle in the eigenbasis, where the ellipse
    # is axis-aligned and dividing out the radii turns it back into a unit circle.
    evals, evecs = np.linalg.eigh(L @ L.T)
    radii = t * np.sqrt(evals)
    local = (pts.astype(np.float64) - mu) @ evecs
    angles = np.arctan2(local[:, 1] / radii[1], local[:, 0] / radii[0])

    # Wrap each step into (-pi, pi] so the arctan2 branch cut is not mistaken
    # for a jump; a single forward loop then has every step equal to +2*pi/n.
    steps = np.mod(np.diff(angles) + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(steps, 2 * np.pi / n_pts, rtol=1e-4)


def test_ellipse_near_singular_L_is_clipped_not_nan() -> None:
    """A near-singular L round-trips to a sub-floor eigenvalue; the clip saves it.

    ``L`` below has a 1e-10 last pivot, so ``eigh(L @ L.T)`` returns a smallest
    eigenvalue of about -2.2e-16 on this platform: literally negative, so
    ``sqrt`` without ``np.clip(evals, 1e-12, None)`` yields NaN and the whole
    polygon is destroyed. The floor assertion is the sign-independent half of
    the guard — even where LAPACK rounds to a tiny *positive* value instead,
    dropping the clip moves the minor half-extent by orders of magnitude.
    """
    # Origin-centred for two reasons: _smallest_axis_extent projects without
    # subtracting a centre, so a nonzero mu would be measured as extent (7.55
    # instead of 2e-6 at mu=(3,-7)); and it also keeps the tiny minor axis
    # exactly representable in float32.
    mu = np.zeros(2)
    L = np.array([[3.923, 0.0], [1.1, 1e-10]])
    evals, _ = np.linalg.eigh(L @ L.T)  # eigh: the call the helper itself makes
    assert evals.min() < _EIGENVALUE_FLOOR, (
        "premise broke: this L no longer produces a sub-floor eigenvalue, so the "
        "test no longer exercises the clip -- pick a more ill-conditioned L"
    )

    pts = ellipse_polygon_from_L(mu, L, t=_DEFAULT_T, n_pts=64)
    assert np.all(np.isfinite(pts)), "clip removed -> sqrt of a negative eigenvalue"
    np.testing.assert_allclose(
        _smallest_axis_extent(pts, L),
        _DEFAULT_T * np.sqrt(_EIGENVALUE_FLOOR),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# ellipsoid_wireframe_from_L
# ---------------------------------------------------------------------------


def test_ellipsoid_shape_and_dtype() -> None:
    """Three principal-plane circles are stacked into one (3*n_pts, 3) array."""
    mu = np.array([1.0, 2.0, 3.0])
    L = np.diag([1.0, 2.0, 3.0])
    pts = ellipsoid_wireframe_from_L(mu, L, t=2.0, n_pts=11)
    assert pts.shape == (33, 3)
    assert pts.dtype == np.float32


def test_ellipsoid_defaults() -> None:
    """Pin the signature defaults (t=2.0, n_pts=32).

    Purely an API-contract pin: unlike the ellipse, neither wireframe call site
    uses a default (``demo_3d_dapi_microscopy.py`` passes ``n_pts=32``,
    ``demo_3d_synthetic_phantom.py`` passes ``n_pts=24``).
    """
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([1.0, 2.0, 3.0])
    pts = ellipsoid_wireframe_from_L(mu, np.diag(sigma))  # no t=, no n_pts=
    assert pts.shape == (96, 3)  # 3 * 32
    half_extent = (pts.max(axis=0) - pts.min(axis=0)) / 2.0
    np.testing.assert_allclose(half_extent, 2.0 * sigma, rtol=1e-6)


def test_ellipsoid_points_lie_on_level_set_rotated() -> None:
    """Every wireframe point satisfies the level set for a rotated 3x3 Sigma."""
    mu = np.array([5.0, -2.0, 0.5])
    L = np.array([[2.0, 0.0, 0.0], [0.7, 1.5, 0.0], [-0.4, 0.9, 1.1]])
    Sigma = L @ L.T
    assert not np.allclose(Sigma, np.diag(np.diag(Sigma)))  # really rotated
    t = 2.0
    n_pts = 24
    pts = ellipsoid_wireframe_from_L(mu, L, t=t, n_pts=n_pts)
    np.testing.assert_allclose(
        _level_set_values(pts, mu, L), np.full(3 * n_pts, t**2), rtol=1e-4
    )


def test_ellipsoid_axis_aligned_extents() -> None:
    """A diagonal L gives half-extents of t*sigma along each axis."""
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([1.0, 4.0, 9.0])
    t = 3.0
    pts = ellipsoid_wireframe_from_L(mu, np.diag(sigma), t=t, n_pts=64).astype(
        np.float64
    )
    half_extent = (pts.max(axis=0) - pts.min(axis=0)) / 2.0
    np.testing.assert_allclose(half_extent, t * sigma, rtol=1e-5)


def test_ellipsoid_near_singular_L_is_clipped_not_nan() -> None:
    """3D counterpart of the 2D clip test -- the higher-risk of the two.

    A near-singular 3x3 reaches a negative smallest eigenvalue more readily than
    a 2x2. Only ``radii[0]`` goes NaN, which poisons two of the three circles --
    ``circle_xy`` and ``circle_xz`` -- while ``circle_yz`` (radii 1 and 2 only)
    stays clean, so a bare "is anything finite?" check would miss it.
    ``eigh(L @ L.T)`` returns -1.3541833365605588e-16 for this L on this platform.
    """
    mu = np.zeros(3)  # see the 2D sibling: _smallest_axis_extent needs it centred
    L = np.array([[3.923, 0.0, 0.0], [0.447, 2.14, 0.0], [1.31, 0.77, 1e-10]])
    evals, _ = np.linalg.eigh(L @ L.T)  # eigh: the call the helper itself makes
    assert evals.min() < _EIGENVALUE_FLOOR, (
        "premise broke: this L no longer produces a sub-floor eigenvalue, so the "
        "test no longer exercises the clip -- pick a more ill-conditioned L"
    )

    pts = ellipsoid_wireframe_from_L(mu, L, t=_DEFAULT_T, n_pts=32)
    assert np.all(np.isfinite(pts)), "clip removed -> sqrt of a negative eigenvalue"
    np.testing.assert_allclose(
        _smallest_axis_extent(pts, L),
        _DEFAULT_T * np.sqrt(_EIGENVALUE_FLOOR),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# psnr
# ---------------------------------------------------------------------------


def test_psnr_identical_is_inf() -> None:
    """A perfect reconstruction has zero MSE -> infinite PSNR."""
    target = np.linspace(0.0, 1.0, 64).reshape(8, 8).astype(np.float32)
    assert psnr(target.copy(), target) == float("inf")


def test_psnr_constant_target_is_inf() -> None:
    """A flat target has zero dynamic range, so PSNR is undefined -> inf."""
    target = np.full((5, 5), 0.25)
    rendered = target + 0.1
    assert psnr(rendered, target) == float("inf")


def test_psnr_peak_is_range_not_max() -> None:
    """A fixed additive error matches 10*log10(range^2 / mse), with range != max.

    The floor is nonzero (3.0..7.0 -> range 4.0, max 7.0) precisely so this pins
    the peak convention. Every inline PSNR still computed in the sibling demos
    uses ``max**2`` -- except ``demo_performance_metrics``, which uses a fixed
    peak of 1.0 -- and each would give a different number here.
    """
    target = np.linspace(3.0, 7.0, 100).reshape(10, 10)
    error = 0.25
    rendered = target + error
    expected = 10.0 * np.log10(4.0**2 / error**2)
    assert psnr(rendered, target) == pytest.approx(expected, rel=1e-12)
    # And it is genuinely not the max-based convention.
    assert psnr(rendered, target) != pytest.approx(
        10.0 * np.log10(7.0**2 / error**2), rel=1e-12
    )


def test_psnr_promotes_integer_input() -> None:
    """The rendered-vs-target DIFFERENCE is taken in float64, so it never wraps.

    (The peak is promoted separately -- see
    ``test_psnr_signed_integer_peak_does_not_overflow``.)
    """
    target = np.full((4, 4), 50, dtype=np.uint8)
    target[0, 0] = 200  # range == 150, max == 200
    rendered = np.full((4, 4), 51, dtype=np.uint8)
    # Signed errors are +1 on fifteen pixels and -149 on the bright one. Drop the
    # promotion and both steps wrap in uint8: the -149 difference becomes 107, and
    # squaring 107 wraps again to 185, so the MSE collapses from 1388.5 to 12.5 and
    # the reported PSNR *rises* from 12.10 dB to 32.55 dB -- silently, with no
    # overflow warning. A quietly optimistic score is the real hazard here, not an
    # obviously broken one.
    mse = (149.0**2 + 15 * 1.0**2) / 16.0
    expected = 10.0 * np.log10(150.0**2 / mse)
    assert psnr(rendered, target) == pytest.approx(expected, rel=1e-12)


def test_psnr_signed_integer_peak_does_not_overflow() -> None:
    """A signed-int target spanning >half its dtype range still scores finitely.

    The per-demo copies all wrote ``float(target.max() - target.min())``, doing
    the subtraction in the input dtype. For int8 spanning -128..127 that wraps to
    -1, the ``rng <= 0`` guard fires, and a reconstruction that misses the bright
    pixel entirely is handed ``inf`` -- a perfect score. The shared helper
    promotes both ends first. int16/int32 wrap identically.
    """
    for dtype, lo, hi in (
        (np.int8, -128, 127),
        (np.int16, -32768, 32767),
        (np.int32, -(2**31), 2**31 - 1),
    ):
        target = np.full((4, 4), lo, dtype=dtype)
        target[0, 0] = hi
        rendered = np.full((4, 4), lo, dtype=dtype)  # misses the bright pixel

        peak = float(hi) - float(lo)
        # Fifteen pixels exact, one off by the full peak.
        expected = 10.0 * np.log10(peak**2 / (peak**2 / 16.0))
        got = psnr(rendered, target)
        assert np.isfinite(got), f"{dtype.__name__}: peak overflowed -> {got}"
        assert got == pytest.approx(expected, rel=1e-12), dtype.__name__
