"""Tests for lifting Points into isotropic Gaussian splats (``gsplats/lift.py``)."""

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lift import (
    _cap_aspect,
    coarse_substitutive_levels,
    compute_ray_integral_factor,
    lift_lines_to_gsplats,
    lift_points_to_gsplats,
    render_light,
)
from luxar.gsplats.utils.trils import pack_tril, unpack_tril


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
    data = lift_points_to_gsplats(
        np.zeros((2, 3), np.float32), R, None, truncation_radius=T
    )
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
    data = lift_lines_to_gsplats(
        verts, 1.0, line_type="segments", truncation_radius=3.0
    )
    assert data.n_splats == 30


def test_lines_beads_are_isotropic_with_sigma_2w_over_T():
    verts = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(
        verts, 1.5, line_type="segments", truncation_radius=3.0
    )
    sig = _bead_sigma(data)
    # uniform width -> all beads share sigma = 2*1.5/3 = 1.0
    np.testing.assert_allclose(sig, 1.0, rtol=1e-4)


def test_lines_tube_centerline_matches_opacity():
    # Overlapping beads sum to ~opacity at the centreline (the √(2π) calibration).
    verts = np.array([[0, 0, 0], [20, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(
        verts, 1.0, line_type="segments", opacity=1.0, truncation_radius=3.0
    )
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


def test_lines_indexed_rejects_float_indices():
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], np.float32)
    with pytest.raises(ValueError, match="integer array"):
        lift_lines_to_gsplats(
            verts,
            1.0,
            line_type="indexed",
            indices=np.array([[0.9, 1.9]], np.float64),
        )


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (np.array([[-1, 0]], np.int64), r"Index -1 < 0"),
        (np.array([[0, 3]], np.int64), r"Index 3 >= n_vertices 3"),
    ],
)
def test_lines_indexed_rejects_out_of_bounds_indices(indices, message):
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], np.float32)
    with pytest.raises(ValueError, match=message):
        lift_lines_to_gsplats(
            verts,
            1.0,
            line_type="indexed",
            indices=indices,
        )


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
    data = lift_lines_to_gsplats(
        np.array([[1.0, 2.0, 3.0]], np.float32), 1.0, line_type="polyline"
    )
    assert data.n_splats == 0


def test_lines_nd_support():
    verts = np.random.default_rng(0).uniform(0, 10, (6, 4)).astype(np.float32)
    data = lift_lines_to_gsplats(verts, 1.0, line_type="segments")
    assert data.ndim == 4


def test_lines_zero_width_does_not_explode():
    # Regression: width=0 -> sigma=0 must NOT blow ceil(L/sigma) into an OOM.
    data = lift_lines_to_gsplats(
        np.array([[0, 0, 0], [10, 0, 0]], np.float32), 0.0, line_type="segments"
    )
    assert data.n_splats == 0  # degenerate-width segment dropped, no allocation


def test_lines_tiny_width_capped():
    # A tiny-but-positive width must be capped, not explode to billions of beads.
    data = lift_lines_to_gsplats(
        np.array([[0, 0, 0], [10, 0, 0]], np.float32), 1e-6, line_type="segments"
    )
    assert 0 < data.n_splats <= 4096


@pytest.mark.parametrize("seg_len", [20.0, 2.0, 0.5])
def test_lines_tube_centerline_equals_opacity_for_any_length(seg_len):
    # Per-segment comb amplitude: long tubes AND short (single-bead) segments must
    # both peak at opacity (the asymptotic sqrt(2pi) alone under-renders short ones).
    verts = np.array([[0, 0, 0], [seg_len, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(
        verts, 1.0, line_type="segments", opacity=1.0, truncation_radius=3.0
    )
    flat = data.flattened()
    c = np.asarray(flat.centers, np.float64)
    a = np.asarray(flat.amplitudes, np.float64)
    sig = _bead_sigma(data)
    uRIF = compute_ray_integral_factor(3.0)
    mid = np.array([seg_len / 2, 0.0, 0.0])
    peaks = a * sig * uRIF * np.exp(-0.5 * (np.linalg.norm(c - mid, axis=1) / sig) ** 2)
    assert float(peaks.sum()) == pytest.approx(1.0, abs=0.05)


def test_lines_loop_has_more_beads_than_polyline():
    # The loop closing edge (N-1,0) must contribute beads — dropping it would
    # make loop == polyline.
    verts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], np.float32)
    poly = lift_lines_to_gsplats(verts, 1.0, line_type="polyline").n_splats
    loop = lift_lines_to_gsplats(verts, 1.0, line_type="loop").n_splats
    assert loop > poly


def test_lines_indexed_lifts_all_edges():
    # Both edges of the indexed edge list must be lifted (not just one).
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], np.float32)
    idx = np.array([[0, 1], [0, 2]], np.intp)
    data = lift_lines_to_gsplats(verts, 1.0, line_type="indexed", indices=idx)
    c = np.asarray(data.flattened().centers, np.float64)
    # beads near both edge midpoints [5,0,0] and [0,5,0]
    assert np.any(np.linalg.norm(c - [5, 0, 0], axis=1) < 1.0)
    assert np.any(np.linalg.norm(c - [0, 5, 0], axis=1) < 1.0)


def test_lines_scalars_interpolate_then_lut():
    # scalars+colormap: scalar interpolated per bead THEN LUT (not RGB-interpolated).
    verts = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(
        verts,
        1.0,
        line_type="segments",
        scalars=np.array([0.0, 1.0], np.float32),
        colormap="viridis",
    )
    from luxar.colormaps import resolve_colormap

    lut = resolve_colormap("viridis").astype(np.float64) / 255.0
    c = np.asarray(data.flattened().colors, np.float64)
    # first bead (t≈small) near LUT low end, last near LUT high end
    assert np.linalg.norm(c[0] - lut[0]) < np.linalg.norm(c[0] - lut[255])
    assert np.linalg.norm(c[-1] - lut[255]) < np.linalg.norm(c[-1] - lut[0])


def test_lines_scalars_nonfinite_does_not_poison_the_lut_range():
    # The colormap range comes from the FINITE vertex scalars only. Pre-fix a
    # single non-finite vertex set vmin/vmax to NaN/Inf, and every bead index
    # came out garbage: -Inf/NaN raised `IndexError: index -92233...` from the
    # LUT gather, +Inf silently collapsed the whole tube to LUT[0]. The scene
    # API refuses non-finite scalars upstream; a direct caller does not.
    verts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], np.float32)
    for bad in (np.nan, np.inf, -np.inf):
        data = lift_lines_to_gsplats(
            verts,
            1.0,
            line_type="polyline",
            scalars=np.array([0.0, 1.0, bad], np.float32),
            colormap="viridis",
        )
        colors = np.asarray(data.flattened().colors, np.float64)
        # The two finite vertices still span the map, so the beads between them
        # must NOT all share one colour (the +Inf collapse) — and nothing raised.
        assert len(np.unique(colors, axis=0)) > 1, f"scalars with {bad} lost the ramp"


# ── lift hardening (colours, degenerate inputs, bead budget) ────────────────


def test_rgba_colors_rejected():
    # gsplats DO carry per-splat alpha; what the lift refuses is a VARYING one —
    # the substitutive merge is untested on it, and shape alone cannot tell it
    # from a constant. Only the uniform form (below) may carry an alpha column.
    pos = np.zeros((2, 3), np.float32)
    rgba = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5]], np.float32)
    with pytest.raises(ValueError, match="RGB"):
        lift_points_to_gsplats(pos, 1.0, colors=rgba)
    # 1-D grayscale is equally invalid (no channel axis).
    with pytest.raises(ValueError, match="RGB"):
        lift_points_to_gsplats(pos, 1.0, colors=np.array([0.5, 0.5], np.float32))


# ── uniform (broadcast) colours (#1444) ─────────────────────────────────────

#: The RGB every uniform form below carries; the RGBA forms append this alpha,
#: which the lift KEEPS (gsplats carry per-splat alpha, and every shader scales
#: intensity by it — dropping it would make a coarse level 1/alpha too bright).
_UNIFORM_RGB = (0.25, 0.5, 1.0)
_UNIFORM_ALPHA = 0.5

_UNIFORM_FORMS = [
    (_UNIFORM_RGB, _UNIFORM_RGB),  # RGB tuple
    (list(_UNIFORM_RGB), _UNIFORM_RGB),  # RGB list
    (np.array([_UNIFORM_RGB], np.float32), _UNIFORM_RGB),  # (1, 3) row
    ((*_UNIFORM_RGB, _UNIFORM_ALPHA), (*_UNIFORM_RGB, _UNIFORM_ALPHA)),  # RGBA tuple
    (  # (1, 4) row
        np.array([(*_UNIFORM_RGB, _UNIFORM_ALPHA)], np.float32),
        (*_UNIFORM_RGB, _UNIFORM_ALPHA),
    ),
]


@pytest.mark.parametrize("colors,want", _UNIFORM_FORMS)
def test_points_uniform_color_broadcast_to_every_splat(colors, want):
    # A uniform colour is a legal, documented form on the flat path, and the one
    # case a coarse LOD level can honour trivially — the lift must broadcast it
    # rather than refuse it (#1444). A uniform ALPHA rides along: gsplats carry
    # it, and losing it would brighten every coarse level by 1/alpha.
    pos = np.zeros((40, 3), np.float32)
    c = np.asarray(lift_points_to_gsplats(pos, 1.0, colors=colors).flattened().colors)
    assert c.shape == (40, len(want))
    np.testing.assert_allclose(c, np.broadcast_to(want, c.shape), atol=1e-6)


@pytest.mark.parametrize("colors,want", _UNIFORM_FORMS)
def test_points_uniform_color_survives_the_zero_radius_drop(colors, want):
    # The zero-radius mask indexes `colors` per point, so the expansion MUST
    # happen before it — masking a bare 3/4-component row would corrupt it.
    pos = np.arange(12, dtype=np.float32).reshape(4, 3)
    radii = np.array([1.0, 0.0, 2.0, 0.0], np.float32)  # keep rows 0 and 2
    data = lift_points_to_gsplats(pos, radii, colors=colors)
    assert data.n_splats == 2
    c = np.asarray(data.flattened().colors)
    assert c.shape == (2, len(want))
    np.testing.assert_allclose(c, np.broadcast_to(want, c.shape), atol=1e-6)


def test_per_element_rgba_still_refused_even_when_every_row_is_equal():
    # The 4th column is admitted for the UNIFORM form only — the substitutive
    # merge is untested on a varying alpha, and shape alone cannot tell a
    # constant (N, 4) from a varying one, so per-element RGBA stays refused.
    pos = np.zeros((5, 3), np.float32)
    rgba = np.tile([*_UNIFORM_RGB, _UNIFORM_ALPHA], (5, 1)).astype(np.float32)
    with pytest.raises(ValueError, match="per-element"):
        lift_points_to_gsplats(pos, 1.0, colors=rgba)
    verts = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    with pytest.raises(ValueError, match="per-element"):
        lift_lines_to_gsplats(
            verts,
            1.0,
            line_type="segments",
            colors=np.tile([*_UNIFORM_RGB, _UNIFORM_ALPHA], (2, 1)).astype(np.float32),
        )


@pytest.mark.parametrize("dtype", [np.int64, np.uint32, np.int8, np.bool_])
@pytest.mark.parametrize("per_element", [False, True])
def test_colors_of_a_dtype_the_leaf_refuses_raise_before_anything_is_written(
    dtype, per_element
):
    # Normalising these would divide by the wrong iinfo max (a near-black coarse
    # level) and the ENCODER would only refuse them at the finest child — after
    # the coarse levels are already on disk. Refuse at lift time instead, for
    # the uniform ROW and the per-element ARRAY alike.
    n = 4
    rows = n if per_element else 1
    colors = np.ones((rows, 3), dtype=dtype)
    pos = np.zeros((n, 3), np.float32)
    with pytest.raises(ValueError, match="dtype"):
        lift_points_to_gsplats(pos, 1.0, colors=colors)
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], np.float32)
    with pytest.raises(ValueError, match="dtype"):
        lift_lines_to_gsplats(verts, 1.0, line_type="segments", colors=colors)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_colors_of_a_dtype_the_leaf_accepts_are_unaffected(dtype):
    # The control for the guard above: every dtype the leaf writes must still
    # lift, per-element and uniform, and integer widths still normalise to [0, 1].
    n = 4
    top = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    per_element = np.full((n, 3), top, dtype=dtype)
    c = np.asarray(
        lift_points_to_gsplats(np.zeros((n, 3), np.float32), 1.0, per_element)
        .flattened()
        .colors
    )
    np.testing.assert_allclose(c, np.ones((n, 3)), atol=1e-6)
    row = np.full((1, 3), top, dtype=dtype)
    c_row = np.asarray(
        lift_points_to_gsplats(np.zeros((n, 3), np.float32), 1.0, row)
        .flattened()
        .colors
    )
    np.testing.assert_allclose(c_row, np.ones((n, 3)), atol=1e-6)


def test_broadcast_colors_stay_row_major():
    # Layout hygiene, not a reproduced bug: a 0-stride broadcast view under the
    # default astype(order="K") comes out channel-major (Fortran) for this one
    # input class, where every other lift input is C-contiguous. Nothing
    # downstream was observed to mind; pinning it costs nothing.
    pos = np.zeros((40, 3), np.float32)
    c = np.asarray(
        lift_points_to_gsplats(pos, 1.0, colors=(0.25, 0.5, 1.0, 0.5))
        .flattened()
        .colors
    )
    assert c.flags["C_CONTIGUOUS"]


def test_uniform_int_tuple_is_face_value_but_uint8_row_is_normalized():
    # Value semantics mirror the leaf writer: a list/tuple's components are taken
    # at FACE value (write_colors classifies HDR by value, it never divides by
    # 255), while an ARRAY row keeps the dtype rule (uint8 means 0..255).
    pos = np.zeros((3, 3), np.float32)
    c_tuple = np.asarray(
        lift_points_to_gsplats(pos, 1.0, colors=(255, 0, 0)).flattened().colors
    )
    np.testing.assert_allclose(c_tuple, np.broadcast_to([255.0, 0.0, 0.0], (3, 3)))
    c_row = np.asarray(
        lift_points_to_gsplats(pos, 1.0, colors=np.array([[255, 0, 0]], np.uint8))
        .flattened()
        .colors
    )
    np.testing.assert_allclose(c_row, np.broadcast_to([1.0, 0.0, 0.0], (3, 3)))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_uniform_integer_rgba_row_normalizes_the_alpha_too(dtype):
    # The one place the alpha column meets the integer dtype normalisation: it
    # must be scaled like the RGB, not left at 128 (an "HDR" opacity).
    top = float(np.iinfo(dtype).max)
    row = np.array([[top, 0, 0, top / 2]], dtype=dtype)
    c = np.asarray(
        lift_points_to_gsplats(np.zeros((4, 3), np.float32), 1.0, row)
        .flattened()
        .colors
    )
    assert c.shape == (4, 4)
    np.testing.assert_allclose(
        c, np.broadcast_to([1.0, 0.0, 0.0, 0.5], (4, 4)), atol=2e-3
    )


@pytest.mark.parametrize("line_type", ["segments", "polyline", "loop", "indexed"])
def test_lines_uniform_rgba_on_every_line_type(line_type):
    # The rule is documented for all line types, not just `segments` (which is
    # the only one the other colour tests exercise).
    verts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], np.float32)
    indices = np.array([[0, 1], [2, 3]], np.intp) if line_type == "indexed" else None
    want = (*_UNIFORM_RGB, _UNIFORM_ALPHA)
    data = lift_lines_to_gsplats(
        verts, 1.0, line_type=line_type, indices=indices, colors=want
    )
    c = np.asarray(data.flattened().colors)
    assert c.shape == (data.n_splats, 4)
    np.testing.assert_allclose(c, np.broadcast_to(want, c.shape), atol=1e-6)


def test_one_bead_line_does_not_relaunder_per_element_rgba_as_uniform():
    # A segment far shorter than its width yields exactly ONE bead, so the
    # per-bead colour array is (1, 4) — which a second uniformity classification
    # would read as the uniform form and admit, averaging two DIFFERENT alphas
    # into a value neither vertex has. Uniformity is decided once, per vertex.
    verts = np.array([[0, 0, 0], [0.001, 0, 0]], np.float32)
    varying = np.array([[0.2, 0.4, 0.6, 0.3], [0.2, 0.4, 0.6, 0.9]], np.float32)
    with pytest.raises(ValueError, match="per-element"):
        lift_lines_to_gsplats(verts, 10.0, line_type="segments", colors=varying)
    # ... while a genuinely uniform RGBA on the same one-bead line still lifts.
    want = (*_UNIFORM_RGB, _UNIFORM_ALPHA)
    data = lift_lines_to_gsplats(verts, 10.0, line_type="segments", colors=want)
    assert data.n_splats == 1
    np.testing.assert_allclose(np.asarray(data.flattened().colors), [want], atol=1e-6)


def test_uniform_predicate_mirrors_is_broadcast_color():
    # The helper's list/tuple admission test is a hand-copied mirror of
    # core.group.compositing.is_broadcast_color (importing it would invert the
    # core -> gsplats layering). This is the drift guard for that copy.
    from luxar.core.group.compositing import is_broadcast_color
    from luxar.gsplats.lift import _expand_uniform_colors

    cases = [
        (1.0, 0.0, 0.0),  # RGB tuple
        [1.0, 0.0, 0.0, 0.5],  # RGBA list
        [np.float32(1), np.float32(0), np.float32(0)],  # numpy scalars
        (1.0, 0.0),  # too few components
        (1.0, 0.0, 0.0, 0.5, 0.5),  # too many
        [[1.0, 0.0, 0.0]] * 3,  # list of triples (entries are not numbers)
        ["a", "b", "c"],  # non-numeric
    ]
    for colors in cases:
        _, expanded = _expand_uniform_colors(colors, 4)
        assert expanded is is_broadcast_color(colors), f"drift on {colors!r}"


@pytest.mark.parametrize("colors,want", _UNIFORM_FORMS)
def test_lines_uniform_color_broadcast_to_every_bead(colors, want):
    # Pre-fix this gathered the colour's COMPONENTS as if they were vertex rows
    # (a bare IndexError at scale, silent garbage on a 2-vertex segment). The
    # alpha survives the per-bead interpolation too (a constant interpolates to
    # itself), which is what lets the inner point lift accept 4 columns.
    verts = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    data = lift_lines_to_gsplats(verts, 1.0, line_type="segments", colors=colors)
    c = np.asarray(data.flattened().colors)
    assert data.n_splats > 1  # a real bead string, not a single bead
    assert c.shape == (data.n_splats, len(want))
    np.testing.assert_allclose(c, np.broadcast_to(want, c.shape), atol=1e-6)


def test_lift_points_degenerate_drops_matching_colours():
    # The zero-radius drop must carry colours along with it: the surviving splats'
    # colours must be exactly the colours of the surviving (non-zero-radius) points.
    pos = np.arange(12, dtype=np.float32).reshape(4, 3)
    radii = np.array([1.0, 0.0, 2.0, 0.0], np.float32)  # keep rows 0 and 2
    colors = np.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]], np.uint8
    )
    data = lift_points_to_gsplats(pos, radii, colors=colors)
    assert data.n_splats == 2
    c = np.asarray(data.flattened().colors, np.float64) * 255.0
    np.testing.assert_allclose(c[0], [10, 20, 30], atol=1e-3)
    np.testing.assert_allclose(c[1], [70, 80, 90], atol=1e-3)


def test_lift_points_nonfinite_amplitude_raises():
    # Absurdly tiny radii underflow sigma -> amplitude overflows float32 to inf;
    # the lift must surface this rather than write inf into the gsplat.
    with pytest.raises(ValueError, match="non-finite amplitudes"):
        lift_points_to_gsplats(np.zeros((1, 3), np.float32), 1e-40, colors=None)


def test_lines_radius_scale_applied():
    # radius_scale mirrors the point lift: a uint8-style width with radius_scale
    # = 1/255 gives the same bead sigma as the equivalent float width at scale 1.
    v = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    d_scaled = lift_lines_to_gsplats(
        v, 255.0, line_type="segments", radius_scale=1.0 / 255.0
    )
    d_plain = lift_lines_to_gsplats(v, 1.0, line_type="segments")
    assert _bead_sigma(d_scaled) == pytest.approx(_bead_sigma(d_plain), rel=1e-4)


def test_lines_input_validation_raises():
    v2 = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    with pytest.raises(ValueError, match=r"\(N, d\)"):  # 1-D vertices
        lift_lines_to_gsplats(np.zeros(3, np.float32), 1.0, line_type="segments")
    with pytest.raises(ValueError, match="truncation_radius"):
        lift_lines_to_gsplats(v2, 1.0, line_type="segments", truncation_radius=0.0)
    with pytest.raises(ValueError, match="line_type"):
        lift_lines_to_gsplats(v2, 1.0, line_type="bogus")


def test_bead_spacing_factor_scales_bead_count():
    # Wider spacing -> fewer beads (spacing = factor * sigma); halving it doubles.
    v = np.array([[0, 0, 0], [20, 0, 0]], np.float32)
    base = lift_lines_to_gsplats(v, 1.0, line_type="segments").n_splats
    sparse = lift_lines_to_gsplats(
        v, 1.0, line_type="segments", bead_spacing_factor=2.0
    ).n_splats
    dense = lift_lines_to_gsplats(
        v, 1.0, line_type="segments", bead_spacing_factor=0.5
    ).n_splats
    assert sparse == pytest.approx(base / 2, abs=1)
    assert dense == pytest.approx(base * 2, abs=1)


def test_lines_per_segment_clamp_warns():
    # A tiny-but-positive width over a long segment exceeds MAX_BEADS_PER_SEGMENT;
    # the clamp must fire AND warn (the doc promises it), not silently truncate.
    v = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    with pytest.warns(UserWarning, match="MAX_BEADS_PER_SEGMENT"):
        data = lift_lines_to_gsplats(v, 1e-6, line_type="segments")
    assert 0 < data.n_splats <= 4096


def test_lines_total_bead_cap(monkeypatch):
    # Many segments, each individually under the per-segment cap, can still SUM to
    # an OOM. With a lowered total budget the lift must widen spacing to fit AND
    # warn — never allocate beyond the budget.
    from luxar.gsplats import lift as lift_mod

    monkeypatch.setattr(lift_mod, "MAX_TOTAL_BEADS", 200)
    # 100 segments of length 20, width 1 -> ~30 beads each = ~3000 raw beads >> 200.
    n_seg = 100
    verts = np.zeros((n_seg * 2, 3), np.float32)
    verts[1::2, 0] = 20.0
    verts[0::2, 1] = np.arange(n_seg)  # offset each segment so they're distinct
    verts[1::2, 1] = np.arange(n_seg)
    with pytest.warns(UserWarning, match="MAX_TOTAL_BEADS"):
        data = lift_lines_to_gsplats(verts, 1.0, line_type="segments")
    # >=1 bead per segment is the floor, so the total can't drop below n_seg, but
    # it must be near the budget — and crucially far below the ~3000 raw count.
    assert n_seg <= data.n_splats <= 400


# ---------------------------------------------------------------------------
# _cap_aspect — anisotropy cap on lifted coarse levels
# ---------------------------------------------------------------------------


def _diag_gsplats(sigmas, amps):
    """Flat GSplatData with diagonal covariances (one row of sigmas per splat)."""
    sig = np.asarray(sigmas, np.float64)
    n, d = sig.shape
    L = np.zeros((n, d, d))
    for i in range(d):
        L[:, i, i] = sig[:, i]
    return GSplatData(
        centers=np.zeros((n, d), np.float32),
        amplitudes=np.asarray(amps, np.float32),
        cholesky_factors=pack_tril(L).astype(np.float32),
        truncation_radius=3.0,
    )


def test_cap_aspect_fattens_and_preserves_mass():
    # sigma (10, 1, 1) at tau=3 -> thin axes fattened to 10/3; mass a*|det L|
    # exactly unchanged (the cap must never change a splat's X-ray integral).
    data = _diag_gsplats([[10.0, 1.0, 1.0]], [2.0])
    capped = _cap_aspect(data, None, 3.0)
    L = unpack_tril(np.asarray(capped.cholesky_factors, np.float64), 3)
    ev = np.linalg.eigvalsh(L @ np.swapaxes(L, 1, 2))
    s = np.sqrt(ev[0])
    assert s.max() / s.min() == pytest.approx(3.0, rel=1e-6)
    assert s.max() == pytest.approx(10.0, rel=1e-6)  # long axis untouched
    mass_before = 2.0 * 10.0 * 1.0 * 1.0
    mass_after = float(capped.amplitudes[0]) * float(np.prod(s))
    assert mass_after == pytest.approx(mass_before, rel=1e-5)


def test_cap_aspect_isotropic_noop():
    # Isotropic input is already within any cap: bitwise-unchanged output.
    data = _diag_gsplats([[2.0, 2.0, 2.0], [0.5, 0.5, 0.5]], [1.0, 3.0])
    capped = _cap_aspect(data, None, 3.0)
    np.testing.assert_array_equal(
        np.asarray(capped.cholesky_factors), np.asarray(data.cholesky_factors)
    )
    np.testing.assert_array_equal(
        np.asarray(capped.amplitudes), np.asarray(data.amplitudes)
    )


def test_cap_aspect_excludes_barrier_dims():
    # 4D splat with a near-delta time axis (dim 0) and an elongated spatial
    # block; coarsen_dims=(1,2,3). The cap must (a) bound the SPATIAL aspect,
    # (b) leave the time row/col bitwise untouched — fattening a sliced axis
    # would bleed geometry across slices.
    sig = [[1e-9, 8.0, 1.0, 1.0]]
    data = _diag_gsplats(sig, [1.0])
    capped = _cap_aspect(data, (1, 2, 3), 2.0)
    L = unpack_tril(np.asarray(capped.cholesky_factors, np.float64), 4)[0]
    Sig = L @ L.T
    assert Sig[0, 0] == pytest.approx(1e-18, rel=1e-6)  # sigma_t^2 untouched
    np.testing.assert_allclose(Sig[0, 1:], 0.0, atol=1e-30)
    s_spatial = np.sqrt(np.linalg.eigvalsh(Sig[1:, 1:]))
    assert s_spatial.max() / s_spatial.min() == pytest.approx(2.0, rel=1e-5)
    # Mass over ALL dims still preserved (submatrix det ratio == full ratio).
    mass_before = 1.0 * 1e-9 * 8.0
    mass_after = float(capped.amplitudes[0]) * float(
        np.prod(np.sqrt(np.linalg.eigvalsh(Sig)))
    )
    assert mass_after == pytest.approx(mass_before, rel=1e-4)


def test_coarse_substitutive_levels_max_aspect_none_disables():
    # The knob must be live: None keeps the raw (elongated) merge output.
    v = np.zeros((400, 3), np.float32)
    v[:, 2] = np.repeat(np.arange(200) * 2.0, 2)
    v[1::2, 2] += 2.0  # 200 collinear segments -> a long bead string
    lifted = lift_lines_to_gsplats(v, 0.5, line_type="segments")

    def max_aspect_of(levels_list):
        worst = 1.0
        for lvl in levels_list:
            L = unpack_tril(np.asarray(lvl.cholesky_factors, np.float64), 3)
            ev = np.linalg.eigvalsh(L @ np.swapaxes(L, 1, 2))
            worst = max(worst, float(np.sqrt(ev[:, -1] / ev[:, 0]).max()))
        return worst

    capped = coarse_substitutive_levels(
        lifted, compression_factor=4, levels=2, device="cpu", seed=0
    )
    uncapped = coarse_substitutive_levels(
        lifted, compression_factor=4, levels=2, device="cpu", seed=0, max_aspect=None
    )
    assert max_aspect_of(capped) <= 3.0 * (1 + 1e-4)
    assert max_aspect_of(uncapped) > 3.0


def test_coarse_substitutive_levels_stamp_final_written_quality():
    rng = np.random.RandomState(12)
    points = rng.normal(0, 4, (96, 3)).astype(np.float32)
    lifted = lift_points_to_gsplats(points, 0.8)

    coarse = coarse_substitutive_levels(
        lifted, compression_factor=4, levels=1, device="cpu", seed=0
    )

    assert len(coarse) == 1
    assert 0.0 <= coarse[0].stats["quality"] < 0.99
