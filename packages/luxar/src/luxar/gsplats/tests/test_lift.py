"""Tests for lifting Points into isotropic Gaussian splats (``gsplats/lift.py``)."""

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lift import (
    compute_ray_integral_factor,
    lift_lines_to_gsplats,
    lift_points_to_gsplats,
    render_light,
)
from luxar.gsplats.utils.trils import unpack_tril


def test_ray_integral_factor_matches_ts_reference():
    # Matches materials/gsplat/math.ts: ~2.433 at T=3, ~sqrt(2 pi)=2.507 untruncated.
    assert compute_ray_integral_factor(3.0) == pytest.approx(2.433, abs=2e-3)
    assert compute_ray_integral_factor(2.0) == pytest.approx(1.851, abs=2e-3)
    # Closed form: sqrt(2 pi) erf(T/sqrt2) - 2 T exp(-T^2/2).
    for T in (2.0, 3.0, 4.0):
        from math import erf, exp, pi, sqrt

        want = sqrt(2 * pi) * erf(T / sqrt(2)) - 2 * T * exp(-0.5 * T * T)
        assert compute_ray_integral_factor(T) == pytest.approx(want, abs=2e-6)


def test_lift_shapes_and_dtypes():
    rng = np.random.default_rng(0)
    n, d = 100, 3
    pos = rng.uniform(-10, 10, (n, d)).astype(np.float32)
    radii = rng.uniform(0.5, 2.0, n)
    colors = rng.uniform(0, 1, (n, 3))
    data = lift_points_to_gsplats(pos, radii, colors)
    assert isinstance(data, GSplatData)
    assert data.n_splats == n
    assert data.ndim == d
    flat = data.flattened()
    assert np.asarray(flat.centers).dtype == np.float32
    assert np.asarray(flat.amplitudes).dtype == np.float32
    assert np.asarray(flat.cholesky_factors).dtype == np.float32
    assert np.asarray(flat.cholesky_factors).shape == (n, d * (d + 1) // 2)
    assert np.asarray(flat.colors).shape == (n, 3)


def test_isotropic_covariance_sigma_is_2R_over_T():
    # sigma = 2 R / T; covariance Sigma = sigma^2 I (diagonal, equal).
    T = 3.0
    R = np.array([1.0, 2.0, 0.5])
    data = lift_points_to_gsplats(
        np.zeros((3, 3), np.float32), R, None, truncation_radius=T
    )
    L = unpack_tril(np.asarray(data.flattened().cholesky_factors), 3)
    sigma_expected = 2.0 * R / T
    for i in range(3):
        np.testing.assert_allclose(L[i], np.eye(3) * sigma_expected[i], atol=1e-5)


def test_peak_match_amplitude():
    # Single-gsplat peak screen intensity = a * sigma * uRIF must equal opacity.
    T, opacity = 3.0, 0.7
    R = 1.5
    data = lift_points_to_gsplats(
        np.zeros((1, 3), np.float32), R, None, opacity=opacity, truncation_radius=T
    )
    flat = data.flattened()
    a = float(np.asarray(flat.amplitudes)[0])
    sigma = 2.0 * R / T
    uRIF = compute_ray_integral_factor(T)
    assert a * sigma * uRIF == pytest.approx(opacity, rel=1e-5)


def test_amplitude_inversely_proportional_to_sigma():
    # a ∝ 1/sigma ∝ 1/R at fixed opacity/T.
    data = lift_points_to_gsplats(
        np.zeros((2, 3), np.float32), np.array([1.0, 2.0]), None
    )
    a = np.asarray(data.flattened().amplitudes)
    assert a[0] / a[1] == pytest.approx(2.0, rel=1e-5)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_nd_support(d):
    rng = np.random.default_rng(d)
    pos = rng.uniform(-5, 5, (20, d)).astype(np.float32)
    data = lift_points_to_gsplats(pos, 1.0, None)
    assert data.ndim == d
    assert data.n_splats == 20
    assert np.asarray(data.flattened().cholesky_factors).shape == (20, d * (d + 1) // 2)


def test_scalar_radius_broadcasts():
    data = lift_points_to_gsplats(np.zeros((5, 3), np.float32), 1.0, None)
    assert data.n_splats == 5
    a = np.asarray(data.flattened().amplitudes)
    assert np.allclose(a, a[0])  # uniform radius -> uniform amplitude


def test_colors_none_leaves_unset():
    data = lift_points_to_gsplats(np.zeros((4, 3), np.float32), 1.0, None)
    colors = data.flattened().colors
    assert colors is None or np.asarray(colors).shape[0] == 4


def test_radius_scale_applied():
    # radius_scale mirrors the shader dtype normalisation.
    d_unscaled = lift_points_to_gsplats(np.zeros((1, 3), np.float32), 255.0, None)
    d_scaled = lift_points_to_gsplats(
        np.zeros((1, 3), np.float32), 255.0, None, radius_scale=1.0 / 255.0
    )
    L_u = unpack_tril(np.asarray(d_unscaled.flattened().cholesky_factors), 3)
    L_s = unpack_tril(np.asarray(d_scaled.flattened().cholesky_factors), 3)
    assert L_u[0, 0, 0] == pytest.approx(255.0 * L_s[0, 0, 0], rel=1e-5)


def test_zero_radius_points_dropped():
    data = lift_points_to_gsplats(
        np.zeros((4, 3), np.float32), np.array([1.0, 0.0, 2.0, 0.0]), None
    )
    assert data.n_splats == 2  # the two zero-radius points are dropped


def test_uint8_colors_normalized_to_unit_range():
    # CRITICAL regression: uint8 RGB (0..255) must be scaled to float [0,1] so the
    # gsplat colour writer treats coarse LOD levels as SDR (not HDR -> ~255x too
    # bright). Pre-fix the lift cast uint8 -> float32 preserving 0..255.
    pos = np.zeros((4, 3), np.float32)
    colors_u8 = np.array(
        [[255, 128, 0], [0, 255, 255], [10, 20, 30], [200, 200, 200]], dtype=np.uint8
    )
    data = lift_points_to_gsplats(pos, 1.0, colors=colors_u8)
    c = np.asarray(data.flattened().colors)
    assert c.dtype == np.float32
    assert c.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(c[0], [1.0, 128 / 255, 0.0], atol=1e-6)


def test_float_colors_passed_through_unchanged():
    # Float colours (already 0..1 or HDR) must NOT be divided — only integers are.
    pos = np.zeros((2, 3), np.float32)
    colors_f = np.array([[0.5, 0.25, 1.0], [2.0, 0.1, 0.0]], dtype=np.float32)  # HDR ok
    c = np.asarray(lift_points_to_gsplats(pos, 1.0, colors=colors_f).flattened().colors)
    np.testing.assert_allclose(c, colors_f, atol=1e-6)


def test_render_light_anisotropic_uses_det():
    # render_light must use |det(L)| (= product of the triangular diagonal), not a
    # sum or an isotropic shortcut. Construct an anisotropic gaussian and pin it.
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.utils.trils import pack_tril

    s = np.array([2.0, 3.0, 0.5])
    L = np.diag(s).astype(np.float32)[None]  # (1,3,3) lower-tri (diagonal)
    chol = pack_tril(L).astype(np.float32)
    a = np.array([1.5], np.float32)
    data = GSplatData(
        centers=np.zeros((1, 3), np.float32), amplitudes=a, cholesky_factors=chol
    )
    assert render_light(data) == pytest.approx(1.5 * (2.0 * 3.0 * 0.5))  # a * det(L)


def test_compute_ray_integral_factor_rejects_nonpositive_T():
    with pytest.raises(ValueError):
        compute_ray_integral_factor(0.0)
    with pytest.raises(ValueError):
        compute_ray_integral_factor(-1.0)


def test_render_light_isotropic_formula():
    # render_light = sum a * sigma^3 for isotropic d=3.
    T = 3.0
    R = np.array([1.0, 2.0])
    data = lift_points_to_gsplats(np.zeros((2, 3), np.float32), R, None, truncation_radius=T)
    a = np.asarray(data.flattened().amplitudes, dtype=np.float64)
    sigma = 2.0 * R / T
    assert render_light(data) == pytest.approx(float(np.sum(a * sigma**3)), rel=1e-5)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        lift_points_to_gsplats(np.zeros(3, np.float32), 1.0, None)  # 1-D positions
    with pytest.raises(ValueError):
        lift_points_to_gsplats(
            np.zeros((2, 3), np.float32), 1.0, None, truncation_radius=0.0
        )


# ── lift_lines_to_gsplats (segment → isotropic beads) ──────────────────────


def _bead_sigma(data, d=3):
    from luxar.gsplats.utils.trils import unpack_tril

    L = unpack_tril(np.asarray(data.flattened().cholesky_factors), d)
    return np.abs(np.linalg.det(L)) ** (1.0 / d)


def test_lines_segment_bead_count():
    # L=20, width=1, T=3 -> sigma_perp=2/3, spacing=sigma -> ~30 beads.
    verts = np.array([[0, 0, 0], [20, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(verts, 1.0, line_type="segments", truncation_radius=3.0)
    assert data.n_splats == 30


def test_lines_beads_are_isotropic_with_sigma_2w_over_T():
    verts = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(verts, 1.5, line_type="segments", truncation_radius=3.0)
    sig = _bead_sigma(data)
    # uniform width -> all beads share sigma = 2*1.5/3 = 1.0
    np.testing.assert_allclose(sig, 1.0, rtol=1e-4)


def test_lines_tube_centerline_matches_opacity():
    # Overlapping beads sum to ~opacity at the centreline (the √(2π) calibration).
    verts = np.array([[0, 0, 0], [20, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(verts, 1.0, line_type="segments", opacity=1.0,
                                 truncation_radius=3.0)
    flat = data.flattened()
    c = np.asarray(flat.centers, np.float64)
    a = np.asarray(flat.amplitudes, np.float64)
    sig = _bead_sigma(data)
    uRIF = compute_ray_integral_factor(3.0)
    mid = np.array([10.0, 0.0, 0.0])
    # sum of per-bead screen peaks (points peak formula a*sigma*uRIF) at the centre
    peaks = a * sig * uRIF * np.exp(-0.5 * (np.linalg.norm(c - mid, axis=1) / sig) ** 2)
    assert float(peaks.sum()) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("line_type", ["segments", "polyline", "loop"])
def test_lines_line_types_produce_beads(line_type):
    verts = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [10, 5, 0]], np.float32)
    data = lift_lines_to_gsplats(verts, 0.5, line_type=line_type, truncation_radius=3.0)
    assert data.n_splats > 0
    assert data.ndim == 3


def test_lines_indexed_uses_edge_list():
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], np.float32)
    idx = np.array([[0, 1], [0, 2]], np.intp)  # two edges from vertex 0
    data = lift_lines_to_gsplats(verts, 1.0, line_type="indexed", indices=idx)
    assert data.n_splats > 0
    with pytest.raises(ValueError):
        lift_lines_to_gsplats(verts, 1.0, line_type="indexed", indices=None)


def test_lines_colors_interpolated_and_normalized():
    verts = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    colors = np.array([[255, 0, 0], [0, 0, 255]], np.uint8)  # red -> blue
    data = lift_lines_to_gsplats(verts, 1.0, line_type="segments", colors=colors)
    c = np.asarray(data.flattened().colors)
    assert c.max() <= 1.0 + 1e-6  # uint8 normalized to [0,1]
    # first bead near red, last near blue
    assert c[0, 0] > c[0, 2]
    assert c[-1, 2] > c[-1, 0]


def test_lines_render_light_positive():
    verts = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(verts, 1.0, line_type="segments")
    assert render_light(data) > 0.0


def test_lines_single_vertex_polyline_is_empty():
    data = lift_lines_to_gsplats(np.array([[1.0, 2.0, 3.0]], np.float32), 1.0,
                                 line_type="polyline")
    assert data.n_splats == 0


def test_lines_nd_support():
    verts = np.random.default_rng(0).uniform(0, 10, (6, 4)).astype(np.float32)
    data = lift_lines_to_gsplats(verts, 1.0, line_type="segments")
    assert data.ndim == 4
