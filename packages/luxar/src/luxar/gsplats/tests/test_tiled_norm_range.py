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
    NORM_RANGE_MIN_SPAN,
    _normalize_data,
    resolve_volume_floor_denoised,
    resolve_volume_norm_range,
    resolve_volume_norm_range_denoised,
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


def test_tiled_range_tracks_the_denoised_basis(monkeypatch):
    """The shared top must describe the array tiles fit, not the raw input."""
    volume = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)

    def _compress_extremes(block, h, **kwargs):
        del h, kwargs
        return np.asarray(block, dtype=np.float32) * 0.5

    monkeypatch.setattr(
        "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
        _compress_extremes,
    )

    resolved = _tile_norm_range(
        volume,
        {"_denoise_h": 0.04, "_denoise_params": {}},
        None,
    )

    assert resolved == pytest.approx((0.0, float(volume.max()) * 0.5))


def test_denoised_range_endpoint_shift_is_bit_exact(monkeypatch):
    from luxar.gsplats.fitting import preprocessing

    raw_endpoint = 1_000_000.5
    denoised_endpoint = 123_456.78
    monkeypatch.setattr(
        preprocessing,
        "resolve_volume_norm_range",
        lambda *_args, **_kwargs: (raw_endpoint, raw_endpoint),
    )
    monkeypatch.setattr(
        preprocessing,
        "_denoise_probe_arrays",
        lambda *_args, **_kwargs: (
            np.asarray([raw_endpoint]),
            np.asarray([denoised_endpoint]),
        ),
    )

    resolved = resolve_volume_norm_range_denoised(
        np.asarray([raw_endpoint]),
        0.0,
        denoise_h=0.04,
        denoise_params={},
    )

    assert resolved == (denoised_endpoint, denoised_endpoint)


def test_floor_and_norm_range_reuse_one_denoise_probe(monkeypatch):
    volume = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
    calls = 0

    def _denoise(block, **_kwargs):
        nonlocal calls
        calls += 1
        return np.asarray(block, dtype=np.float32) * 0.5

    monkeypatch.setattr(
        "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
        _denoise,
    )
    cache: dict[str, object] = {}

    floor = resolve_volume_floor_denoised(
        volume, "p10", denoise_h=0.04, denoise_params={}, probe_cache=cache
    )
    resolved = resolve_volume_norm_range_denoised(
        volume,
        0.0,
        denoise_h=0.04,
        denoise_params={},
        probe_cache=cache,
    )

    assert floor is not None
    assert resolved == pytest.approx((0.0, float(volume.max()) * 0.5))
    assert calls == 1


def test_oversized_norm_range_skips_denoise_probe(monkeypatch):
    volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    monkeypatch.setattr(
        "luxar.gsplats.fitting.preprocessing._volume_fits_probe_budget",
        lambda _volume, _budget: False,
    )
    monkeypatch.setattr(
        "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
        lambda *_args, **_kwargs: pytest.fail("oversized range must not denoise"),
    )

    assert resolve_volume_norm_range_denoised(
        volume, 0.0, denoise_h=0.04, denoise_params={}
    ) == pytest.approx((0.0, 63.0))


def test_norm_range_probe_read_failure_keeps_raw_range(monkeypatch):
    volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    monkeypatch.setattr(
        "luxar.gsplats.fitting.preprocessing._denoise_probe_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read failed")),
    )

    assert resolve_volume_norm_range_denoised(
        volume, 0.0, denoise_h=0.04, denoise_params={}
    ) == pytest.approx((0.0, 63.0))


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


def test_tile_range_declines_when_the_floor_swallows_the_volume():
    """A floor at or above the sampled top leaves no usable shared scale.

    Reachable: the single-tile CLI worker applies a numeric ``--floor``
    unguarded (so one parent-resolved level survives onto a dim timepoint), and
    the sampled max under-reports a peak the bounded sample missed. The shifted
    range then collapses to the epsilon floor, and normalizing a tile by it
    would divide real signal by ~1e-12. Declining the shared scale sends the
    tile back to its own — worse comparability, but the measurement that would
    have provided it is meaningless anyway.
    """
    vol = np.full((8, 32, 32), 100.0, dtype=np.float32)
    vol[4, 16, 16] = 900.0

    assert _tile_norm_range(vol, {}, 900.0) is None
    assert _tile_norm_range(vol, {}, 5000.0) is None
    # just below the top is still a usable scale — the guard is not over-broad
    assert _tile_norm_range(vol, {}, 899.0) == pytest.approx((0.0, 1.0))


def test_tile_range_declines_a_non_finite_top():
    """NaN compares False against everything, so it must be tested for.

    The raw-tile NaN check inside the fitter already ran on a clean tile; a NaN
    arriving through the SHARED scale would turn that tile's normalized array
    into NaN with nothing left to catch it.
    """
    vol = np.full((8, 32, 32), 100.0, dtype=np.float32)
    vol[0, 0, 0] = np.nan
    assert _tile_norm_range(vol, {}, None) is None

    vol_inf = np.full((8, 32, 32), 100.0, dtype=np.float32)
    vol_inf[0, 0, 0] = np.inf
    assert _tile_norm_range(vol_inf, {}, None) is None


def test_tile_range_declines_a_top_with_no_positive_extent():
    """A sample that saw no signal is not a scale, floor or no floor.

    Reachable without any floor at all: the bounded sample lands entirely in an
    empty (or masked, or padded) region of a large volume. Sharing ``hi == 0``
    would give every tile ``intensity_range == 0``, and ``_normalize_data``
    answers that by filling the tile with a uniform 0.5 — a fabricated flat
    field fitted as if it were data. A negative top (background-subtracted data
    whose positive structure the sample missed) inverts the sign instead.
    """
    assert _tile_norm_range(np.zeros((8, 32, 32), dtype=np.float32), {}, None) is None
    assert (
        _tile_norm_range(np.full((8, 32, 32), -5.0, dtype=np.float32), {}, None) is None
    )


def test_zero_range_fabricates_a_flat_field():
    """Pin what the decline above avoids, so that guard is not vacuous."""
    tile = np.zeros((4, 8, 8), dtype=np.float32)
    tile[2, 4, 4] = 5.0
    shared_zero, *_ = _normalize_data(tile.copy(), 0.0, False, None, (0.0, 0.0))
    assert np.allclose(shared_zero, 0.5)


def test_tile_range_keeps_a_genuinely_tiny_scale():
    """Small is not the same as collapsed — a tiny float stack keeps its scale.

    The decline exists for a range MANUFACTURED by the floor subtraction, not
    for data that is honestly this dim: normalizing such a volume by its own
    extent is exactly right, and withdrawing the shared scale would silently
    restore the per-tile disagreement this module exists to remove.
    """
    vol = np.zeros((8, 32, 32), dtype=np.float64)
    vol[4, 16, 16] = 5e-13
    assert _tile_norm_range(vol, {}, None) == pytest.approx((0.0, 5e-13))


def test_a_declined_range_is_not_resolved_again_per_tile(monkeypatch):
    """A decline must not cost one bounded volume read per tile.

    ``None`` is both "no shared scale" and the un-resolved default, so the
    orchestrator marks that it has already looked; without that marker every
    tile would re-sample the whole volume to be told the same thing again (and
    re-print the note), which on a lazy zarr is the per-tile read the shared
    resolution exists to avoid.
    """
    from luxar.gsplats import fit_tiled_gsplats as ftg
    from luxar.gsplats.tiling import compute_tile_specs

    calls: list = []
    real = ftg._tile_norm_range

    def _counting(volume, fit_kwargs, applied_floor, verbose=False):
        calls.append(applied_floor)
        return real(volume, fit_kwargs, applied_floor, verbose=verbose)

    monkeypatch.setattr(ftg, "_tile_norm_range", _counting)

    vol = np.zeros((8, 32, 32), dtype=np.float32)
    vol[4, 16, 16] = 900.0
    spec = compute_tile_specs(vol.shape, 32, 0)[0]

    ftg.fit_tile(
        vol,
        spec,
        floor="none",
        norm_range=None,
        _norm_range_resolved=True,
        seeds=10,
        n_iters=1,
        verbose=False,
    )
    assert calls == []

    # and without the marker the same call DOES resolve — the guard is not vacuous
    ftg.fit_tile(
        vol, spec, floor="none", norm_range=None, seeds=10, n_iters=1, verbose=False
    )
    assert len(calls) == 1


def test_fit_tiled_resolves_a_declined_range_once_for_the_grid(monkeypatch):
    """The orchestrator's half of the same contract, over a real tile grid.

    An empty volume is the reachable decline that needs no floor at all: every
    tile is skipped as signal-free, so what this measures is purely how many
    times the grid asked for a shared scale.
    """
    from luxar.gsplats import fit_tiled_gsplats as ftg

    calls: list = []
    real = ftg._tile_norm_range

    def _counting(volume, fit_kwargs, applied_floor, verbose=False):
        calls.append(applied_floor)
        return real(volume, fit_kwargs, applied_floor, verbose=verbose)

    monkeypatch.setattr(ftg, "_tile_norm_range", _counting)

    vol = np.zeros((8, 64, 64), dtype=np.float32)
    ftg.fit_tiled(
        vol, tile_size=32, overlap=0, floor="none", seeds=10, n_iters=1, verbose=False
    )
    assert len(calls) == 1


def test_fit_tiled_honours_an_upstream_declined_range(monkeypatch):
    """A range the CALLER already declined must not be measured again (#2838).

    The occupancy scan (``_weighted_uniform_seed_counts``) resolves the shared
    scale in order to weigh the tiles and marks it ``_norm_range_resolved``; when
    ``_tile_norm_range`` declines, the answer it records is ``None``. ``fit_tiled``
    tested ``norm_range is None`` BEFORE consulting that marker, so it re-read the
    volume and printed the decline a second time — the one case the marker exists
    to distinguish from "nobody has looked yet".
    """
    from luxar.gsplats import fit_tiled_gsplats as ftg

    calls: list = []

    def _counting(volume, fit_kwargs, applied_floor, verbose=False):
        calls.append(applied_floor)
        return None

    monkeypatch.setattr(ftg, "_tile_norm_range", _counting)

    vol = np.zeros((8, 64, 64), dtype=np.float32)
    ftg.fit_tiled(
        vol,
        tile_size=32,
        overlap=0,
        floor="none",
        seeds=10,
        n_iters=1,
        verbose=False,
        norm_range=None,
        _norm_range_resolved=True,
    )
    assert calls == []


def test_supplied_raw_range_is_shifted_to_the_floor_subtracted_tile_basis():
    """Batch workers receive raw units; tile fitting consumes post-floor units."""
    from luxar.gsplats import fit_tiled_gsplats as ftg

    fit_kwargs = {"norm_range": (10.0, 110.0)}
    ftg._ensure_tile_norm_range(
        np.zeros((4, 4, 4), dtype=np.float32), fit_kwargs, applied_floor=30.0
    )
    assert fit_kwargs["norm_range"] == pytest.approx((0.0, 80.0))

    collapsed = {"norm_range": (10.0, 30.0)}
    ftg._ensure_tile_norm_range(
        np.zeros((4, 4, 4), dtype=np.float32), collapsed, applied_floor=30.0
    )
    assert collapsed["norm_range"] is None


def test_fit_tiled_marks_a_supplied_range_after_shifting_it_once(monkeypatch):
    """The orchestrator's converted range must not be shifted again by fit_tile."""
    from luxar.gsplats import fit_tiled_gsplats as ftg

    seen = {}

    class _Stop(Exception):
        pass

    def _capture(volume, spec, **fit_kwargs):
        seen.update(fit_kwargs)
        raise _Stop

    monkeypatch.setattr(ftg, "fit_tile", _capture)
    volume = np.full((8, 8, 8), 50.0, dtype=np.float32)
    volume[4, 4, 4] = 100.0

    with pytest.raises(_Stop):
        ftg.fit_tiled(
            volume,
            tile_size=8,
            overlap=0,
            floor=30.0,
            norm_range=(10.0, 110.0),
            verbose=False,
        )

    assert seen["norm_range"] == pytest.approx((0.0, 80.0))
    assert seen["_norm_range_resolved"] is True


def test_collapsed_range_would_amplify_signal_by_1e12():
    """Pin the hazard the decline above avoids, so that guard is not vacuous."""
    tile = np.zeros((4, 8, 8), dtype=np.float32)
    tile[2, 4, 4] = 5.0
    collapsed, *_ = _normalize_data(
        tile.copy(), 0.0, False, None, (0.0, NORM_RANGE_MIN_SPAN)
    )
    assert collapsed.max() > 1e12
    # and the fallback the decline produces keeps it in [0, 1]
    own, *_ = _normalize_data(tile.copy(), 0.0, False, None, None)
    assert own.max() == pytest.approx(1.0)


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
        floor="none",
        norm_range=(0.0, 4.0),
    )

    assert seen == [(0.0, 4.0), None]


@pytest.mark.parametrize(
    "bad",
    [
        (5.0, 5.0),  # empty range -> the "nearly uniform" fill-with-0.5 branch
        (2.0, 1.0),  # reversed -> every voxel normalizes negative, clips to 0
        (0.0, float("nan")),
        (0.0, float("inf")),
        (1.0,),  # not a pair
    ],
)
def test_a_degenerate_norm_range_is_refused(bad):
    """A bad supplied range must fail loudly, not fit successfully on nonsense."""
    from luxar.gsplats import fit_gaussian_splats

    vol = np.zeros((4, 8, 8), dtype=np.float32)
    vol[2, 4, 4] = 1.0
    with pytest.raises(ValueError):
        fit_gaussian_splats(
            vol, seeds=4, n_iters=1, device="cpu", verbose=False, norm_range=bad
        )


def test_a_resolved_tile_range_passes_that_validation(ramp_volume):
    """Control: what the tiled fitter actually produces IS a valid range."""
    from luxar.gsplats import fit_gaussian_splats

    resolved = _tile_norm_range(ramp_volume, {}, None)
    assert resolved is not None and resolved[1] > resolved[0]
    result = fit_gaussian_splats(
        ramp_volume,
        seeds=4,
        n_iters=1,
        device="cpu",
        verbose=False,
        norm_range=resolved,
    )
    assert result is not None
