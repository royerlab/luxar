"""Every tile of one volume must share an intensity scale.

A tile normalized by its OWN min/max is stretched to fill [0, 1] by a different
factor than its neighbour. Output amplitudes are rescaled by that same factor,
so the physical amplitude of a linear fit largely cancels — but the optimiser's
absolute criteria (convergence tolerance, seeding and culling thresholds) do
not, so the same structure is resolved to a different accuracy in each tile.

Two things the shared range must NOT do, both covered below: clip a voxel
brighter than the bounded sample that produced it, and lift ``image_min`` off
zero (which would break the Hann partition of unity across tile overlaps).
"""

import numpy as np
import pytest

from luxar.gsplats.fit_tiled_gsplats import _tile_norm_range
from luxar.gsplats.fitting.preprocessing import (
    _normalize_data,
    resolve_volume_norm_range,
)


@pytest.fixture
def ramp_volume() -> np.ndarray:
    """A Y-ramp both tiles span, plus a bright spot only the LEFT tile contains.

    The spot is what makes the two tiles' local maxima disagree — the real case
    being one tile holding a bright neurite while its neighbour holds only
    background.
    """
    vol = np.zeros((16, 64, 64), dtype=np.float32)
    vol += np.linspace(10.0, 200.0, 64)[None, :, None]
    vol[8, 40, 10] = 500.0
    return vol


def test_tiles_agree_on_brightness_with_global_range(ramp_volume):
    left = ramp_volume[:, :, :32].copy()
    right = ramp_volume[:, :, 32:].copy()
    # precondition: the tiles really do disagree without a shared range
    assert left.max() != right.max()

    rng = resolve_volume_norm_range(ramp_volume, 0.0)
    norm_l, *_ = _normalize_data(left, 0.0, False, None, rng)
    norm_r, *_ = _normalize_data(right, 0.0, False, None, rng)

    # same row => same physical brightness in both tiles
    assert norm_l[0, 20, 0] == pytest.approx(norm_r[0, 20, 0], abs=1e-6)
    assert norm_l[0, 55, 5] == pytest.approx(norm_r[0, 55, 5], abs=1e-6)


def test_per_tile_normalization_is_what_disagrees(ramp_volume):
    """Pin the disagreement this guards against, so the guard is not vacuous."""
    left = ramp_volume[:, :, :32].copy()
    right = ramp_volume[:, :, 32:].copy()
    norm_l, *_ = _normalize_data(left, 0.0, False, None, None)
    norm_r, *_ = _normalize_data(right, 0.0, False, None, None)
    # >50% disagreement on identical brightness when each tile self-normalizes
    assert abs(norm_l[0, 20, 0] - norm_r[0, 20, 0]) > 0.5 * norm_r[0, 20, 0]


def test_norm_range_none_preserves_whole_volume_behaviour(ramp_volume):
    """The knob must be inert for a non-tiled fit."""
    base, *_ = _normalize_data(ramp_volume.copy(), 0.0, False, None, None)
    same, *_ = _normalize_data(
        ramp_volume.copy(),
        0.0,
        False,
        None,
        resolve_volume_norm_range(ramp_volume, 0.0),
    )
    np.testing.assert_allclose(base, same)


def test_subtract_shifts_the_range_into_post_floor_terms(ramp_volume):
    """Tiles are floor-subtracted before fitting, so the range must match."""
    plain = resolve_volume_norm_range(ramp_volume, 0.0)
    shifted = resolve_volume_norm_range(ramp_volume, 0.0, subtract=10.0)
    assert shifted[0] == pytest.approx(0.0)
    assert shifted[1] == pytest.approx(plain[1] - 10.0)


def test_subtract_clamps_at_zero(ramp_volume):
    """A floor above the volume minimum must not produce a negative image_min."""
    lo, hi = resolve_volume_norm_range(ramp_volume, 0.0, subtract=1e6)
    assert lo == pytest.approx(0.0)
    assert hi > lo


def test_percentile_range_is_resolved_globally(ramp_volume):
    """norm_percentile must be honoured on the WHOLE volume, not per tile.

    Asserted against numpy on the whole array rather than against the min/max:
    in this fixture each ramp value occupies ~1.6% of the voxels, so the 1st
    percentile legitimately IS the minimum. The contract is "the same number
    numpy would give for the whole volume", not "strictly inside the extremes".
    """
    lo, hi = resolve_volume_norm_range(ramp_volume, 1.0)
    assert lo == pytest.approx(float(np.percentile(ramp_volume, 1.0)))
    assert hi == pytest.approx(float(np.percentile(ramp_volume, 99.0)))
    # the single bright outlier must be clipped away by the 99th percentile
    assert hi < float(ramp_volume.max())


def test_supplied_full_range_does_not_clip_a_brighter_voxel(ramp_volume):
    """The range is sampled, so a tile may hold a voxel above it.

    Clipping there would flatten the brightest structure — precisely what
    per-array normalization never does, since that array's own max is its
    ceiling by construction.
    """
    lo, hi = 0.0, 200.0  # deliberately below the fixture's 500.0 spot
    norm, *_ = _normalize_data(ramp_volume.copy(), 0.0, False, None, (lo, hi))
    assert norm[8, 40, 10] == pytest.approx(500.0 / 200.0)
    assert norm.min() >= 0.0


def test_supplied_percentile_range_still_clips_outliers(ramp_volume):
    """A percentile range asked for bright-outlier clipping; it keeps it."""
    norm, *_ = _normalize_data(ramp_volume.copy(), 1.0, False, None, (0.0, 200.0))
    assert norm.max() == pytest.approx(1.0)


def test_tile_range_pins_image_min_at_zero_without_a_floor():
    """Apodized, floor-subtracted tile data starts at zero — so must the range.

    With no floor to subtract (``--floor none``, or a floor the guard refused)
    the volume minimum is a pedestal the tiles never see: the Hann window
    tapers every overlapped face to 0. A positive ``image_min`` would subtract
    a constant from BOTH sides of an overlap and clip the taper away.
    """
    vol = np.full((8, 32, 32), 100.0, dtype=np.float32)
    vol[4, 16, 16] = 900.0

    lo, hi = _tile_norm_range(vol, {}, None)
    assert lo == 0.0
    assert hi == pytest.approx(900.0)

    # the volume minimum really is well above zero — the guard is not vacuous
    assert resolve_volume_norm_range(vol, 0.0)[0] == pytest.approx(100.0)


def test_tile_range_stays_in_post_floor_terms():
    """With a floor applied the top is shifted, and the bottom is still zero."""
    vol = np.full((8, 32, 32), 100.0, dtype=np.float32)
    vol[4, 16, 16] = 900.0
    lo, hi = _tile_norm_range(vol, {}, 100.0)
    assert lo == 0.0
    assert hi == pytest.approx(800.0)


def test_progressive_residual_passes_drop_the_shared_range(monkeypatch):
    """A residual is a fraction of the volume range; it renormalizes itself.

    Pass 0 shares the whole-volume scale (that is the point of supplying it);
    a residual pass normalized against it would sit under the absolute
    convergence tolerance and end at its first evaluation.
    """
    from luxar.gsplats import fit_progressive_gsplats as fpg
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.utils.trils import tril_size

    seen: list = []

    def _fake_fit(V, **kwargs):
        seen.append(kwargs.get("norm_range", "absent"))
        ndim = np.asarray(V).ndim
        return GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros((0,), dtype=np.float32),
            cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
            stats={},
        )

    # the progressive fitter imports it lazily inside the function body
    monkeypatch.setattr(
        "luxar.gsplats.fit_gsplats.fit_gaussian_splats", _fake_fit, raising=True
    )

    vol = np.zeros((8, 16, 16), dtype=np.float32)
    vol[4, 8, 8] = 1.0
    fpg.fit_progressive_gaussian_splats(
        vol,
        max_splats=10,
        max_splats_per_pass=5,
        iters_per_pass=1,
        max_passes=2,
        device="cpu",
        verbose=False,
        norm_range=(0.0, 4.0),
    )

    assert seen == [(0.0, 4.0), None]
