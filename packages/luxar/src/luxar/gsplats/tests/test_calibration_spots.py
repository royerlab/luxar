"""Spot (punctum) fidelity check that ``cal`` reports next to K*."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.calibration.spots import (
    SPOT_MIN_COUNT,
    SpotReference,
    detect_spots,
    spot_fidelity_warning,
    spot_peak_ratio,
)


def _puncta_volume(n: int = 40, seed: int = 0) -> np.ndarray:
    """Sparse diffraction-limited spots on a dark, slightly noisy background."""
    rng = np.random.default_rng(seed)
    shape = (24, 64, 64)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    V = rng.normal(0.0, 1.0, shape).astype(np.float32)
    # A spaced lattice keeps the spots apart (one maximum per 5-voxel window).
    sites = [
        (z, y, x) for z in (6, 12, 18) for y in range(6, 60, 8) for x in range(6, 60, 8)
    ]
    for z, y, x in rng.permutation(np.array(sites))[:n]:
        V += 100.0 * np.exp(-((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) / 2.0)
    return np.clip(V, 0.0, None)


def test_detect_spots_finds_isolated_puncta() -> None:
    spots = detect_spots(_puncta_volume(n=40))
    assert spots is not None
    assert spots.n == 40


def test_detect_spots_survives_a_mostly_zero_volume() -> None:
    # After a floor clip most voxels are exactly 0, so the MAD is 0; the
    # spread is then measured over the non-zero voxels.
    V = np.zeros((24, 64, 64), dtype=np.float32)
    V[12, 32, 32] = 50.0
    V[12, 20, 20] = 60.0
    V[6, 40, 12] = 1.0
    spots = detect_spots(V)
    assert spots is not None and spots.n >= 2
    assert detect_spots(np.zeros((8, 8, 8), dtype=np.float32)) is None


def test_spot_peak_ratio_tracks_the_rendered_peak() -> None:
    V = _puncta_volume()
    spots = detect_spots(V)
    assert spots is not None
    assert spot_peak_ratio(V, spots) == pytest.approx(1.0, abs=1e-6)
    assert spot_peak_ratio(0.5 * V, spots) == pytest.approx(0.5, abs=1e-6)
    assert spot_peak_ratio(np.zeros_like(V), spots) == 0.0


def test_warning_fires_when_k_star_under_renders_puncta() -> None:
    ks = [1000, 4000, 16000, 32000]
    message = spot_fidelity_warning(ks, [0.68, 0.83, 0.98, 0.99], 1000, 178)
    assert message is not None
    # 0.68 / 0.99: the share of the best ratio, not the raw ratio.
    assert "K*=1,000" in message and "69% of the best" in message
    assert "--seeds 16,000" in message  # first K within 98% of the best
    assert "for the whole volume" not in message


def test_warning_fires_when_k_star_renders_no_spot_at_all() -> None:
    ks = [1000, 4000, 16000]
    message = spot_fidelity_warning(ks, [0.0, 0.7, 0.97], 1000, 178)
    assert message is not None and "0% of the best" in message
    assert spot_fidelity_warning(ks, [0.0, 0.0, 0.0], 1000, 178) is None


def test_crop_scoped_advice_uses_the_planner_transfer() -> None:
    """Under --region / --auto-region the grid K are crop budgets.

    `fit --seeds` takes a whole-volume budget, so the suggestion is restated
    with the same transfer as K*: seeds * eff(K_spot) / k_star_crop_seeds,
    with the EFFECTIVE (post-cull) count at K_spot. Without a whole-volume
    figure it names the content-tiling density flags for that count.
    """
    ks = [1000, 4000, 16000]
    eff = [900, 3600, 14000]
    ratios = [0.68, 0.83, 0.97]
    whole = {"seeds": 26_000, "k_star_crop_seeds": 900}
    scaled = spot_fidelity_warning(
        ks, ratios, 1000, 178, k_effective=eff, crop_scoped=True, whole_volume=whole
    )
    assert scaled is not None
    assert "--seeds 404,444 for the whole volume" in scaled  # 26,000 x 14,000 / 900
    assert "crop K=16,000" in scaled
    density = {"n_features_reference": 89, "feature_threshold": 60.475}
    unscaled = spot_fidelity_warning(
        ks, ratios, 1000, 178, k_effective=eff, crop_scoped=True, density=density
    )
    assert unscaled is not None
    assert (
        "fit --tiling content --k-star-ref 14000 --n-features-ref 89 "
        "--feature-threshold 60.475" in unscaled
    )


def test_warning_says_when_the_spot_curve_is_still_rising() -> None:
    message = spot_fidelity_warning([500, 1000], [0.5, 0.7], 500, 300)
    assert message is not None
    assert "still rising" in message and "extend --k-max" in message
    stopped = spot_fidelity_warning([500, 1000], [0.5, 0.7], 500, 300, [2000, 4000])
    assert stopped is not None
    assert "skipped K up to 4,000" in stopped and "--no-early-stop" in stopped


@pytest.mark.parametrize(
    ("ratios", "n_spots"),
    [
        ([0.97, 0.99], 300),  # K* already within 95% of the best
        ([0.70, 0.60], 300),  # no larger K did better
        ([0.5, 0.7], SPOT_MIN_COUNT - 1),  # too few spots to call it punctate
        ([float("nan"), float("nan")], 300),  # check not run
        ([], 300),  # cal.json that predates the check
    ],
)
def test_warning_stays_silent(ratios: list, n_spots: int) -> None:
    assert spot_fidelity_warning([500, 1000], ratios, 500, n_spots) is None


def test_patch_ratio_matches_a_whole_volume_blur() -> None:
    """The per-spot patches reproduce the whole-volume blur, borders included."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(3)
    rendered = rng.random((12, 20, 17)).astype(np.float32)
    indices = (
        np.array([2, 6, 9, 3]),
        np.array([2, 10, 17, 15]),
        np.array([2, 8, 14, 3]),
    )
    spots = SpotReference(indices=indices, peaks=np.ones(4, dtype=np.float32))
    full = ndi.maximum_filter(ndi.gaussian_filter(rendered, 1.0), size=3)[indices]
    assert spot_peak_ratio(rendered, spots) == pytest.approx(float(np.median(full)))
    torch = pytest.importorskip("torch")
    assert spot_peak_ratio(torch.from_numpy(rendered), spots) == pytest.approx(
        float(np.median(full))
    )


@pytest.mark.parametrize("peak", [2.0, 3.0])
def test_dim_spots_on_a_perfect_render_do_not_warn(peak: float) -> None:
    """A noisy reference peak is biased up, so dim spots cannot score 1.

    Spots peaking 2-3 counts above a Poisson background of 2, floor-subtracted
    as `cal` does: a perfect render scores well under 0.9 against the noisy
    reference. Every render shares that bias, so a sweep whose renders are all
    perfect must not warn, however many spots were detected.
    """
    from scipy import ndimage as ndi

    rng = np.random.default_rng(0)
    shape = (32, 96, 96)
    truth = np.zeros(shape, dtype=np.float32)
    truth[tuple(rng.integers(3, n - 3, 150) for n in shape)] = 1.0
    truth = ndi.gaussian_filter(truth, 1.2) * (2 * np.pi * 1.2**2) ** 1.5 * peak
    noisy = np.clip(rng.poisson(truth + 2.0).astype(np.float32) - 2.0, 0, None)

    spots = detect_spots(noisy)
    assert spots is not None and spots.n > 0
    perfect = spot_peak_ratio(truth, spots)
    assert perfect < 0.9  # the raw ratio would have warned
    ratios = [perfect, perfect, perfect]
    for n_spots in (spots.n, 1000):  # independent of the detected count
        assert spot_fidelity_warning([500, 1000, 2000], ratios, 500, n_spots) is None
    # The same reference still flags a K* render that loses half the peak.
    half = spot_peak_ratio(0.5 * truth, spots)
    assert spot_fidelity_warning([500, 1000], [half, perfect], 500, 1000) is not None
