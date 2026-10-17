"""Projected-sum prediction and the auto-exposure plan (``gsplats/exposure.py``)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pytest

from luxar.gsplats.exposure import (
    LINEAR_TARGET_BRIGHTNESS,
    SIDE_CAP,
    HiddenAxis,
    default_target_brightness,
    emission_weights,
    plan_projected_exposure,
    predict_projected_sum,
    stamped_amplitude_window,
)
from luxar.gsplats.lift import compute_ray_integral_factor


def _random_set(
    n: int, *, spread: float, smax: float, seed: int = 0, ndim: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, spread, (n, ndim))
    n_packed = ndim * (ndim + 1) // 2
    chol = rng.normal(0.0, 0.3, (n, n_packed))
    sigmas = rng.uniform(0.5, smax, (n, ndim))
    for i in range(ndim):
        chol[:, i * (i + 1) // 2 + i] = sigmas[:, i]
    return centers, chol, rng.uniform(0.1, 1.0, n)


def _brute_force(
    centers: np.ndarray,
    chol: np.ndarray,
    weights: np.ndarray,
    view_axis: int,
    dims: Sequence[int],
    grid: int = 384,
    trunc: float = 2.75,
) -> tuple[float, float]:
    """Per-splat reference rasterizer: no radius cap, no levels, no sampling."""
    ndim = centers.shape[1]
    lower = np.zeros((len(centers), ndim, ndim))
    k = 0
    for i in range(ndim):
        for j in range(i + 1):
            lower[:, i, j] = chol[:, k]
            k += 1
    cov = (lower @ np.transpose(lower, (0, 2, 1)))[:, dims][:, :, dims]
    a = list(dims).index(view_axis)
    keep = [i for i in range(3) if i != a]
    sigma_ray = 1.0 / np.sqrt(np.linalg.inv(cov)[:, a, a])
    peak = weights * compute_ray_integral_factor(trunc) * sigma_ray
    plane = cov[:, keep][:, :, keep]
    inv = np.linalg.inv(plane)
    uv = centers[:, [dims[i] for i in keep]]
    sd = np.sqrt(plane[:, [0, 1], [0, 1]].max(axis=1))
    pad = trunc * float(np.median(sd))
    lo, hi = uv.min(0) - pad, uv.max(0) + pad
    px = float((hi - lo).max() / grid)
    nx, ny = (np.ceil((hi - lo) / px).astype(int) + 1).tolist()
    xs = (np.arange(nx) + 0.5) * px + lo[0]
    ys = (np.arange(ny) + 0.5) * px + lo[1]
    img = np.zeros((nx, ny))
    for s in range(len(uv)):
        dx = xs[:, None] - uv[s, 0]
        dy = ys[None, :] - uv[s, 1]
        m = inv[s, 0, 0] * dx * dx + 2 * inv[s, 0, 1] * dx * dy + inv[s, 1, 1] * dy * dy
        img += peak[s] * np.exp(-0.5 * m) * (m <= trunc * trunc)
    occupied = img[img > 2e-3]
    p99, p999 = np.percentile(occupied, [99.0, 99.9])
    return float(p99), float(p999)


@pytest.mark.parametrize(
    ("n", "spread", "smax", "axis"),
    [(400, 20.0, 3.0, 2), (300, 25.0, 14.0, 0), (500, 8.0, 2.0, 1)],
)
def test_predictor_matches_brute_force(
    n: int, spread: float, smax: float, axis: int
) -> None:
    centers, chol, weights = _random_set(n, spread=spread, smax=smax)
    ref_p99, ref_p999 = _brute_force(centers, chol, weights, axis, [0, 1, 2], grid=96)
    got = predict_projected_sum(centers, chol, weights, axis, grid=96, max_radius=6)
    assert got.p999 == pytest.approx(ref_p999, rel=0.1)
    assert got.p99 == pytest.approx(ref_p99, rel=0.1)


def test_predictor_uses_the_displayed_columns_of_an_nd_set() -> None:
    centers, chol, weights = _random_set(300, spread=15.0, smax=3.0, ndim=4)
    _, ref = _brute_force(centers, chol, weights, 3, [1, 2, 3], grid=96)
    got = predict_projected_sum(
        centers, chol, weights, 3, display_dims=[1, 2, 3], grid=96
    )
    assert got.p999 == pytest.approx(ref, rel=0.1)


def test_subsampling_preserves_the_statistic() -> None:
    centers, chol, weights = _random_set(20_000, spread=30.0, smax=3.0)
    full = predict_projected_sum(centers, chol, weights, 2, grid=128)
    sampled = predict_projected_sum(
        centers, chol, weights, 2, grid=128, max_splats=10_000
    )
    assert sampled.n_splats == 10_000
    # Sampling noise can only push the percentile up (toward dimming).
    assert full.p999 * 0.97 <= sampled.p999 <= full.p999 * 1.1


def test_predictor_is_linear_in_the_weights() -> None:
    centers, chol, weights = _random_set(200, spread=10.0, smax=2.0)
    one = predict_projected_sum(centers, chol, weights, 2, occupancy_floor=0.0)
    three = predict_projected_sum(centers, chol, 3 * weights, 2, occupancy_floor=0.0)
    assert three.p999 == pytest.approx(3 * one.p999, rel=1e-9)


def test_planar_set_needs_no_view_axis() -> None:
    centers, chol, weights = _random_set(100, spread=10.0, smax=2.0, ndim=2)
    stats = predict_projected_sum(centers, chol, weights, None)
    assert stats.p999 > 0.0


def test_view_axis_must_be_displayed() -> None:
    centers, chol, weights = _random_set(10, spread=1.0, smax=1.0, ndim=4)
    with pytest.raises(ValueError, match="not displayed"):
        predict_projected_sum(centers, chol, weights, 0, display_dims=[1, 2, 3])


@pytest.mark.parametrize("colormap", [None, "viridis"])
def test_lut_weights_scale_linearly_with_amplitudes(colormap: Optional[str]) -> None:
    amps = np.random.default_rng(1).uniform(0.0, 2.0, 500)
    base = emission_weights(amps, colormap=colormap)
    np.testing.assert_allclose(
        emission_weights(0.25 * amps, colormap=colormap), 0.25 * base
    )


def test_colour_weights_use_the_brightest_channel_and_alpha() -> None:
    amps = np.array([1.0, 2.0])
    colors = np.array([[255, 0, 0, 255], [0, 51, 0, 128]], dtype=np.uint8)
    np.testing.assert_allclose(
        emission_weights(amps, colors, opacity=0.5),
        [0.5, 2.0 * 0.2 * (128 / 255) * 0.5],
    )


def test_plan_dims_to_target_and_never_brightens() -> None:
    centers, chol, amps = _random_set(2000, spread=5.0, smax=3.0)
    bright = plan_projected_exposure(
        centers, chol, amps, view_axis=0, display_dims=[0, 1, 2], target=2.0
    )
    assert bright.predicted_before > 2.0
    assert bright.factor < 1.0
    assert bright.predicted_after == pytest.approx(2.0)

    dim = plan_projected_exposure(
        centers, chol, amps, view_axis=0, display_dims=[0, 1, 2], target=1e6
    )
    assert dim.factor == 1.0
    assert dim.predicted_after == dim.predicted_before


def _stack(
    centers: np.ndarray, chol: np.ndarray, values: Sequence[float], sigma: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Copies of a 3-D set at each hidden-coordinate value (a 4th column)."""
    stacked = np.concatenate(
        [np.column_stack([centers, np.full(len(centers), t)]) for t in values]
    )
    chol4 = np.zeros((len(stacked), 10))
    chol4[:, :6] = np.tile(chol, (len(values), 1))
    chol4[:, 9] = sigma
    return stacked, chol4


def test_plan_predicts_the_opening_slice_of_a_stack() -> None:
    """A hidden time column draws ONE timepoint, so stacking more must not dim."""
    centers, chol, amps = _random_set(500, spread=5.0, smax=2.0, ndim=3)
    one = plan_projected_exposure(
        centers, chol, amps, view_axis=0, display_dims=[0, 1, 2]
    )
    stacked, chol4 = _stack(centers, chol, (0.0, 1.0, 2.0))
    time = HiddenAxis(3, 0.0, True, 1.0, (0.0, 2.0))
    plan = plan_projected_exposure(
        stacked,
        chol4,
        np.tile(amps, 3),
        view_axis=0,
        display_dims=[0, 1, 2],
        hidden=[time],
        data_range=(float(amps.min()), float(np.percentile(amps, 99.9))),
    )
    # The neighbouring timepoint leaks at exp(-8) of its weight, no more.
    assert plan.predicted_before == pytest.approx(one.predicted_before, rel=2e-3)


def test_a_jittered_discrete_axis_keeps_its_timepoint() -> None:
    """Splats jittered off the grid still count (the old half-min-gap mask kept 1)."""
    centers, chol, amps = _random_set(500, spread=5.0, smax=2.0, ndim=3)
    stacked, chol4 = _stack(centers, chol, (0.0, 1.0))
    rng = np.random.default_rng(0)
    stacked[:, 3] += rng.uniform(-0.05, 0.05, len(stacked))
    window = (float(amps.min()), float(np.percentile(amps, 99.9)))
    exact = plan_projected_exposure(
        centers, chol, amps, view_axis=0, display_dims=[0, 1, 2], data_range=window
    )
    jittered = plan_projected_exposure(
        stacked,
        chol4,
        np.tile(amps, 2),
        view_axis=0,
        display_dims=[0, 1, 2],
        hidden=[HiddenAxis(3, 0.0, True, 1.0, (0.0, 1.0))],
        data_range=window,
    )
    assert jittered.predicted_before == pytest.approx(exact.predicted_before, rel=0.1)


def test_a_continuous_hidden_axis_is_weighted_at_its_centre() -> None:
    """A continuous 4th axis opens at its centre and weights by the marginal."""
    rng = np.random.default_rng(1)
    centers, chol, amps = _random_set(800, spread=5.0, smax=2.0, ndim=3)
    w = rng.uniform(0.0, 10.0, len(centers))
    stacked = np.column_stack([centers, w])
    chol4 = np.zeros((len(centers), 10))
    chol4[:, :6] = chol
    chol4[:, 9] = 0.5
    plan = plan_projected_exposure(
        stacked,
        chol4,
        amps,
        view_axis=0,
        display_dims=[0, 1, 2],
        hidden=[HiddenAxis(3, 5.0, False, 0.0, (0.0, 10.0))],
    )
    near = np.abs(w - 5.0) < 1.5  # what the marginal mostly keeps
    reference = plan_projected_exposure(
        centers[near],
        chol[near],
        amps[near],
        view_axis=0,
        display_dims=[0, 1, 2],
        data_range=(float(amps.min()), float(np.percentile(amps, 99.9))),
    )
    assert 0.3 * reference.predicted_before < plan.predicted_before
    assert plan.predicted_before < 1.5 * reference.predicted_before


def test_other_timepoints_cap_the_factor() -> None:
    """A dim opening timepoint and a bright later one: the later one caps."""
    centers, chol, amps = _random_set(1500, spread=4.0, smax=2.0, ndim=3)
    dim_c, dim_l, _ = _random_set(30, spread=4.0, smax=2.0, ndim=3, seed=5)
    early, early_l = _stack(dim_c, dim_l, (0.0,))
    late, late_l = _stack(centers, chol, (4.0,))
    plan = plan_projected_exposure(
        np.concatenate([early, late]),
        np.concatenate([early_l, late_l]),
        np.concatenate([np.full(30, 0.8), amps]),
        view_axis=0,
        display_dims=[0, 1, 2],
        hidden=[HiddenAxis(3, 0.0, True, 1.0, (0.0, 4.0))],
    )
    assert plan.side_capped
    assert plan.cap_source == "other timepoints"
    assert plan.side_p999 is not None
    assert plan.side_p999 * plan.factor == pytest.approx(SIDE_CAP * 2.0)


def test_constant_amplitudes_use_the_lut_midpoint() -> None:
    """A degenerate window maps to the LUT midpoint (as the viewer does), not 0."""
    weights = emission_weights(np.full(10, 3.0))
    np.testing.assert_allclose(weights, 3.0 * 128 / 255)


def test_stamped_window_pools_partition_tops_by_count() -> None:
    from luxar.gsplats.gsplat_data import GSplatData

    def leaf(amps: np.ndarray) -> GSplatData:
        n = len(amps)
        return GSplatData(
            centers=np.zeros((n, 3), np.float32),
            amplitudes=amps.astype(np.float32),
            cholesky_factors=np.tile(np.float32([1, 0, 1, 0, 0, 1]), (n, 1)),
        )

    from luxar.gsplats.tree import GSplatPartition

    a = np.linspace(0.0, 1.0, 1000)
    b = np.linspace(0.5, 9.0, 3000)
    part = GSplatPartition(children=[leaf(a).tree, leaf(b).tree])
    lo, hi = stamped_amplitude_window(part) or (None, None)
    assert lo == pytest.approx(0.0)
    top_a, top_b = np.percentile(a, 99.9), np.percentile(b, 99.9)
    assert hi == pytest.approx((1000 * top_a + 3000 * top_b) / 4000, rel=1e-5)


@pytest.mark.parametrize(
    ("tone", "target"),
    [(None, 2.0), ("ACES", 2.0), ("AgX", 2.0), ("None", 0.9), ("Linear", 0.9)],
)
def test_default_target_follows_the_tone_mapping(
    tone: Optional[str], target: float
) -> None:
    assert default_target_brightness(tone) == target
    assert LINEAR_TARGET_BRIGHTNESS == 0.9


def _slab(n: int = 3000, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin in column 0, wide in 1 and 2: dim face-on, bright edge-on."""
    rng = np.random.default_rng(seed)
    centers = np.column_stack(
        [rng.normal(0.0, 0.5, n), rng.uniform(0, 60, n), rng.uniform(0, 60, n)]
    )
    chol = np.tile([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], (n, 1))
    return centers, chol, rng.uniform(0.5, 1.0, n)


def test_side_views_cap_the_factor() -> None:
    centers, chol, amps = _slab()
    plan = plan_projected_exposure(
        centers, chol, amps, view_axis=0, display_dims=[0, 1, 2], target=2.0
    )
    assert plan.side_capped
    assert plan.side_p999 is not None
    assert plan.side_p999 * plan.factor == pytest.approx(SIDE_CAP * 2.0)
    assert plan.predicted_after < 2.0


def test_side_cap_is_idle_when_the_opening_view_is_the_brightest() -> None:
    centers, chol, amps = _slab()
    plan = plan_projected_exposure(
        centers, chol, amps, view_axis=1, display_dims=[0, 1, 2], target=2.0
    )
    assert not plan.side_capped
    assert plan.predicted_after == pytest.approx(2.0)
