"""Every tile of one volume must share an intensity scale.

Regression for box-shaped amplitude steps in tiled fits: a tile normalized by
its OWN min/max maps a given physical brightness to a different normalized
value than its neighbour, so identical structure fits to different amplitudes.
Bright localized structure hides it (it sets its own tile's maximum); anything
varying smoothly across a tile boundary shows it as a box edge.

Measured on a 2573x2707x463 confocal mosaic before the fix: amplitude per unit
source intensity varied 3.0x across the volume, with 78% jumps between
neighbouring regions (p95).
"""

import numpy as np
import pytest

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
    """Pin the bug this guards against, so the test cannot pass vacuously."""
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
