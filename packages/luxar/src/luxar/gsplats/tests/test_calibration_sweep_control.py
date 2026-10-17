"""Sweep control for ``gsplat cal``: early stop, ETA, low-edge extension, auto grid."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from luxar.gsplats.calibration import (
    CalibrationResult,
    SweepControl,
    auto_k_bounds,
    calibrate,
)
from luxar.gsplats.calibration import driver as driver_module
from luxar.gsplats.calibration.grid import (
    FG_VOXELS_PER_SPLAT_AT_K_MAX,
    K_MAX_CEILING,
    K_MAX_FLOOR,
)
from luxar.gsplats.calibration.sweep_control import (
    estimate_remaining_seconds,
    extension_below,
    format_duration,
    plateau_reached,
)

# -----------------------------------------------------------------------------
# plateau_reached
# -----------------------------------------------------------------------------


class TestPlateauReached:
    def test_rising_curve_never_stops(self) -> None:
        assert not plateau_reached([20.0, 21.0, 22.0, 23.0, 24.0])

    def test_two_flat_points_stop(self) -> None:
        assert plateau_reached([20.0, 22.0, 22.05, 22.08])

    def test_peak_then_decline_stops(self) -> None:
        assert plateau_reached([20.0, 23.0, 22.5, 22.0])

    def test_one_flat_point_is_not_enough(self) -> None:
        assert not plateau_reached([20.0, 22.0, 22.05])

    def test_a_late_gain_resets(self) -> None:
        # flat, then a real gain on the last point: keep going
        assert not plateau_reached([20.0, 22.0, 22.05, 22.5])

    def test_slow_creep_counts_as_plateau(self) -> None:
        # each step gains less than the 0.1 dB bar
        assert plateau_reached([22.0, 22.08, 22.16])

    def test_nan_counts_as_non_improving(self) -> None:
        assert plateau_reached([20.0, 22.0, float("nan"), 21.9])

    def test_a_clear_decline_stops_at_once(self) -> None:
        # The dense 37.7 Mvoxel synthetic: foreground peak at 64000, -0.25 dB at
        # 128000 (a 63-minute fit) -> stop before the 2-hour 256000 point.
        assert plateau_reached([26.60, 29.38, 33.90, 35.33, 35.72, 35.47])
        # ... but a dip shallower than the decline bar needs the full patience
        assert not plateau_reached([29.61, 29.98, 29.94])
        assert not plateau_reached([29.61, 29.98, 29.85], decline_db=None)

    def test_needs_a_finite_head(self) -> None:
        assert not plateau_reached([float("nan"), 1.0, 1.0])

    def test_patience_validated(self) -> None:
        with pytest.raises(ValueError):
            plateau_reached([1.0, 2.0], patience=0)


# -----------------------------------------------------------------------------
# ETA
# -----------------------------------------------------------------------------


class TestEta:
    def test_nothing_remaining(self) -> None:
        assert estimate_remaining_seconds([1000], [10.0], []) == 0.0

    def test_nothing_measured(self) -> None:
        assert math.isnan(estimate_remaining_seconds([], [], [2000]))

    def test_linear_cost_extrapolates(self) -> None:
        eta = estimate_remaining_seconds(
            [500, 1000, 2000, 4000], [99.0, 10.0, 20.0, 40.0], [8000]
        )
        assert eta == pytest.approx(80.0, rel=1e-6)

    def test_falling_times_clip_to_flat(self) -> None:
        # Early points dominated by overhead can get FASTER with K: predict the
        # flat mean (first, warm-up point dropped), never a negative cost.
        eta = estimate_remaining_seconds(
            [125, 250, 500, 1000], [29.0, 15.0, 10.0, 11.0], [2000]
        )
        assert eta == pytest.approx(12.0, rel=1e-6)

    def test_flat_jitter_does_not_explode(self) -> None:
        # The A549/GB10 times at 250..2000: a two-point slope predicted hours.
        eta = estimate_remaining_seconds(
            [125, 250, 500], [14.0, 8.0, 10.0], [1000 * 2**i for i in range(10)]
        )
        assert eta < 600

    def test_format_duration(self) -> None:
        assert format_duration(42.4) == "42s"
        assert format_duration(185) == "3m05s"
        assert format_duration(4320) == "1h12m"
        assert format_duration(float("nan")) == "?"


# -----------------------------------------------------------------------------
# extension_below
# -----------------------------------------------------------------------------


class TestExtensionBelow:
    def test_continues_the_bottom_ratio(self) -> None:
        assert extension_below([1000, 2000, 4000]) == [250, 500]

    def test_ratio_is_clamped(self) -> None:
        # a 10x first step is clamped to 4x
        assert extension_below([1000, 10000]) == [62, 250]

    def test_floor_drops_tiny_points(self) -> None:
        assert extension_below([40, 80], floor=16) == [20]


# -----------------------------------------------------------------------------
# auto_k_bounds
# -----------------------------------------------------------------------------


def _blob_volume(
    shape: tuple[int, int, int], n_blobs: int, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    V = np.full(shape, 100.0, dtype=np.float32)
    zz, yy, xx = np.indices(shape)
    for _ in range(n_blobs):
        c = [rng.uniform(0, s) for s in shape]
        r2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        V += 400.0 * np.exp(-r2 / (2 * 2.0**2)).astype(np.float32)
    return V + rng.normal(0, 5, shape).astype(np.float32)


class TestAutoKBounds:
    def test_sparse_volume_caps_far_below_the_old_default(self) -> None:
        V = _blob_volume((32, 64, 64), 20)
        b = auto_k_bounds(V)
        assert b.k_max < 512_000
        assert b.k_max >= K_MAX_FLOOR
        assert b.k_min < b.k_max
        assert 0.0 < b.foreground_fraction < 0.5

    def test_cap_tracks_foreground_count(self) -> None:
        V = _blob_volume((48, 128, 128), 400)
        b = auto_k_bounds(V)
        expected = b.n_foreground_voxels / FG_VOXELS_PER_SPLAT_AT_K_MAX
        if K_MAX_FLOOR < expected < K_MAX_CEILING:
            assert b.k_max == pytest.approx(expected, rel=0.05)

    def test_ceiling_is_the_historical_default(self) -> None:
        V = np.zeros((8, 8, 8), np.float32)
        V[:4] = 1.0  # half the volume is "foreground"
        b = auto_k_bounds(V)
        assert b.k_max <= K_MAX_CEILING
        assert b.to_dict()["rule"]
        assert "Auto K grid" in b.summary()


# -----------------------------------------------------------------------------
# calibrate(): early stop + extension, with a stubbed per-K fit
# -----------------------------------------------------------------------------


def _stub_run(curve: Dict[int, float], calls: List[int]):
    def run(self: Any, K: int) -> Dict[str, Any]:
        calls.append(int(K))
        v = curve[int(K)]
        return {
            "k": int(K),
            "eff": int(K),
            "n_fitted": int(K),
            "n_stored": int(K) // 2,
            "held_psnr_robust": v - 3.0,
            "held_psnr_fg_robust": v - 13.0,
            "fit_s": 0.01,
            "path": None,
            "held_mse": 10 ** (-v / 10),
            "held_psnr": v,
            "held_psnr_fg": v - 10.0,
            "held_gain": v - 20.0,
            "held_psnr_fg_weighted": v - 5.0,
            "train_psnr": v + 1.0,
            "full_psnr": v,
            "full_ssim": 0.9,
        }

    return run


def _tiny_volume() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(0.5, 0.1, (8, 12, 12)), 0, 1).astype(np.float32)


_FIT = {"device": "cpu", "use_cuda": False, "use_metal": False}


class TestCalibrateSweepControl:
    def test_no_control_runs_the_whole_grid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curve = {1000: 30.0, 2000: 31.0, 4000: 31.0, 8000: 31.0, 16000: 31.0}
        calls: List[int] = []
        monkeypatch.setattr(driver_module.PointContext, "run", _stub_run(curve, calls))
        r = calibrate(
            _tiny_volume(), list(curve), fit_kwargs=_FIT, compute_rd_model=False
        )
        assert calls == list(curve)
        assert r.sweep is None

    def test_early_stop_skips_the_tail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        curve = {1000: 30.0, 2000: 31.0, 4000: 31.05, 8000: 30.9, 16000: 35.0}
        calls: List[int] = []
        monkeypatch.setattr(driver_module.PointContext, "run", _stub_run(curve, calls))
        control = SweepControl(early_stop=True, extend_below=False)
        lines: List[str] = []
        r = calibrate(
            _tiny_volume(),
            list(curve),
            fit_kwargs=_FIT,
            compute_rd_model=False,
            sweep_control=control,
            progress_callback=lambda i, n, m: lines.append(m),
        )
        assert calls == [1000, 2000, 4000, 8000]
        assert r.k_values_requested == [1000, 2000, 4000, 8000]
        assert r.sweep is not None
        assert r.sweep["stopped_early"] is True
        # the stop watched psnr_minmax itself: its K* is not "truncated"
        assert r.sweep["minmax_curve_truncated"] is False
        assert r.sweep["k_values_skipped"] == [16000]
        assert any("stopping" in m and "16000" in m for m in lines)
        assert any("eta" in m for m in lines)
        # cal.json round trip keeps the record
        r.to_json(tmp_path / "cal.json")
        assert CalibrationResult.from_json(tmp_path / "cal.json").sweep == r.sweep

    def test_early_stop_follows_the_selected_metric(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curve = {1000: 30.0, 2000: 31.0, 4000: 31.0, 8000: 31.0}
        calls: List[int] = []
        run = _stub_run(curve, calls)

        def run_fg_rising(self: Any, K: int) -> Dict[str, Any]:
            rec = run(self, K)
            rec["held_psnr_fg"] = float(np.log2(K))  # +1 dB per doubling
            return rec

        monkeypatch.setattr(driver_module.PointContext, "run", run_fg_rising)
        calibrate(
            _tiny_volume(),
            list(curve),
            fit_kwargs=_FIT,
            compute_rd_model=False,
            k_star_metric="psnr_foreground",
            sweep_control=SweepControl(early_stop=True, extend_below=False),
        )
        assert calls == list(curve)

    def test_bottom_edge_extends_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curve = {250: 32.0, 500: 33.5, 1000: 33.0, 2000: 32.0, 4000: 31.0}
        calls: List[int] = []
        monkeypatch.setattr(driver_module.PointContext, "run", _stub_run(curve, calls))
        control = SweepControl(early_stop=False, extend_below=True)
        r = calibrate(
            _tiny_volume(),
            [1000, 2000, 4000],
            fit_kwargs=_FIT,
            compute_rd_model=False,
            sweep_control=control,
        )
        assert calls == [1000, 2000, 4000, 250, 500]
        assert r.k_values_requested == [250, 500, 1000, 2000, 4000]
        assert r.held_out_peak.k_star == 500
        assert r.sweep is not None
        assert r.sweep["k_values_extended_below"] == [250, 500]

    def test_interior_k_star_does_not_extend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curve = {1000: 30.0, 2000: 33.0, 4000: 31.0}
        calls: List[int] = []
        monkeypatch.setattr(driver_module.PointContext, "run", _stub_run(curve, calls))
        r = calibrate(
            _tiny_volume(),
            list(curve),
            fit_kwargs=_FIT,
            compute_rd_model=False,
            sweep_control=SweepControl(early_stop=False, extend_below=True),
        )
        assert calls == list(curve)
        assert r.sweep is not None
        assert r.sweep["k_values_extended_below"] == []


# -----------------------------------------------------------------------------
# Robust vs full-range peak: a constant offset, so K* does not move
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("offset_db", [-6.0, -1.3, 2.5, 9.0])
def test_k_star_is_invariant_to_the_psnr_peak(offset_db: float) -> None:
    """Every K* rule compares dB DIFFERENCES; the peak only adds a constant."""
    from luxar.gsplats.calibration import find_k_star

    rng = np.random.default_rng(3)
    ks = [125 * 2**i for i in range(12)]
    for _ in range(50):
        curve = list(np.cumsum(rng.normal(0.1, 0.4, len(ks))) + 30.0)
        a = find_k_star(ks, curve)
        b = find_k_star(ks, [v + offset_db for v in curve])
        assert (a.k_star, a.type, a.k_knee) == (b.k_star, b.type, b.k_knee)
        for n in range(3, len(ks)):
            assert plateau_reached(curve[:n]) == plateau_reached(
                [v + offset_db for v in curve[:n]]
            )


def test_real_sweep_reports_robust_peak_and_counts() -> None:
    """A real (tiny, CPU) sweep: robust = full-range + constant; counts ordered."""
    V = _blob_volume((12, 40, 40), 10)  # 19,200 voxels: p99.99 is the 2nd max
    V[0, 0, 0] = V.max() * 5  # one hot pixel sets the full range
    r = calibrate(
        V,
        [20, 60],
        fit_kwargs={**_FIT, "n_iters": 40, "early_stop_patience": 40},
        compute_rd_model=False,
    )
    off = [
        a - b
        for a, b in zip(r.held_out_psnr_robust_db, r.held_out_psnr_db, strict=True)
    ]
    expected = 20 * math.log10(r.robust_data_range / r.data_range)
    assert off == pytest.approx([expected] * 2, abs=1e-6)
    assert expected < -10  # the hot pixel inflated the full range ~5x
    for fitted, kept, stored in zip(
        r.k_values_fitted, r.k_values_effective, r.k_values_stored, strict=True
    ):
        assert stored <= kept <= fitted
    assert r.k_star_stored in r.k_values_stored


def test_whole_volume_seeds_sums_the_density_transfer_per_box() -> None:
    """A volume of 4 copies of the crop gets ~4x the crop K*; an empty one 0."""
    from luxar.gsplats.calibration import (
        count_features,
        feature_threshold,
        whole_volume_seeds,
    )

    crop = _blob_volume((8, 32, 32), 12) - 100.0
    thr = feature_threshold(crop, "peaks")
    n_crop = count_features(crop, method="peaks")
    density = {
        "feature_method": "peaks",
        "n_features_reference": int(n_crop),
        "k_star_reference": 1000,
        "saturation_exponent": 0.44,
        "saturation_cap": 4000,
        "splats_per_feature": 1000 / max(n_crop, 1),
        "feature_threshold": thr,
    }
    whole = np.tile(crop, (1, 2, 2))
    out = whole_volume_seeds(
        whole,
        crop_shape=crop.shape,
        k_star_seeds=2000,
        splat_density=density,
        floor_subtracted=None,
    )
    assert out is not None and out["n_boxes"] == 4
    assert out["seeds"] == pytest.approx(4 * 2000, rel=0.15)
    empty = whole_volume_seeds(
        np.zeros_like(whole),
        crop_shape=crop.shape,
        k_star_seeds=2000,
        splat_density=density,
        floor_subtracted=None,
    )
    assert empty is not None and empty["seeds"] == 0


# -----------------------------------------------------------------------------
# should_stop: the early stop must agree with find_k_star (reviewer's creep.py)
# -----------------------------------------------------------------------------

_CREEP_KS = [1000 * 2**i for i in range(10)]
_CREEP_CURVES = {
    "creep 0.09 dB/doubling": [20 + 0.09 * i for i in range(10)],
    "steep then creep": [18, 19, 19.6, 19.69, 19.78, 19.87, 19.96, 20.05, 20.14, 20.23],
    "creep then rise": [20, 20.05, 20.1, 20.6, 21.2, 21.5, 21.6, 21.6, 21.5, 21.4],
    "falls from the start": [23.37, 23.27, 23.14, 23.01, 22.92, 22.93, 22.9, 22.95],
}


def _replay(curve: List[float]) -> List[Dict[str, Any]]:
    from luxar.gsplats.calibration.sweep import should_stop

    control = SweepControl()
    recs: List[Dict[str, Any]] = []
    for i, (k, v) in enumerate(zip(_CREEP_KS, curve, strict=False)):
        recs.append({"k": k, "m": float(v)})
        if i + 1 < len(curve) and should_stop(recs, "m", control):
            break
    return recs


@pytest.mark.parametrize("name", sorted(_CREEP_CURVES))
def test_early_stop_never_leaves_an_undecided_curve(name: str) -> None:
    from luxar.gsplats.calibration import find_k_star

    recs = _replay(_CREEP_CURVES[name])
    vals = [r["m"] for r in recs]
    ks = [r["k"] for r in recs]
    peak = find_k_star(ks, vals)
    if len(recs) < len(_CREEP_CURVES[name]):  # it stopped early
        assert max(vals) - min(vals) >= peak.knee_margin_db
        assert peak.type != "signal_limited"
        assert peak.k_star != ks[-1]


def test_slow_creep_is_not_cut_at_three_points_and_does_not_extend() -> None:
    from luxar.gsplats.calibration import find_k_star
    from luxar.gsplats.calibration.sweep import _best_at_bottom

    recs = _replay(_CREEP_CURVES["creep 0.09 dB/doubling"])
    assert len(recs) >= 5  # the 0.3 dB knee margin has to be crossed first
    vals = [r["m"] for r in recs]
    assert find_k_star([r["k"] for r in recs], vals).k_star != _CREEP_KS[0]
    assert not _best_at_bottom(vals)


def test_steep_then_creep_is_not_cut_while_signal_limited() -> None:
    from luxar.gsplats.calibration import find_k_star

    # After 5 points the curve still reads signal_limited (tail mean 0.26 dB).
    first5 = _CREEP_CURVES["steep then creep"][:5]
    assert find_k_star(_CREEP_KS[:5], first5).type == "signal_limited"
    assert len(_replay(_CREEP_CURVES["steep then creep"])) > 5


def test_a_falling_curve_stops_and_extends() -> None:
    from luxar.gsplats.calibration.sweep import _best_at_bottom

    recs = _replay(_CREEP_CURVES["falls from the start"])
    assert len(recs) == 4  # span reaches 0.36 dB at the 4th point
    assert _best_at_bottom([r["m"] for r in recs])


def test_auto_bounds_never_exceed_the_voxel_count() -> None:
    V = np.zeros((4, 8, 8), np.float32)
    V[:2] = 1.0
    b = auto_k_bounds(V)
    assert b.k_max <= V.size
    assert 1 <= b.k_min <= b.k_max // 2


_A549_KS = [98, 196, 392, 783, 1566, 3130, 6257, 12510]


@pytest.mark.parametrize(
    ("values", "points", "k_star"),
    [
        # A fresh MPS cal of A549 (re-verification probe): the max landed at 783
        # and 3130 fell 0.13 dB. Stopping there read a plateau with K* = 392;
        # one more point resolves the peak.
        ([21.029, 22.7, 23.126, 23.298, 23.293, 23.167, 23.05, 22.95], 7, 783),
        # The same curve jittered so the max lands at 1566: a clear peak at once.
        ([21.029, 22.7, 23.126, 23.28, 23.30, 23.167, 23.05, 22.95], 6, 1566),
    ],
)
def test_a_falling_plateau_is_resolved_before_stopping(
    values: List[float], points: int, k_star: int
) -> None:
    from luxar.gsplats.calibration import find_k_star
    from luxar.gsplats.calibration.sweep import should_stop

    control = SweepControl()
    for i in range(2, len(values) + 1):
        recs = [{"k": k, "m": v} for k, v in zip(_A549_KS[:i], values[:i], strict=True)]
        if should_stop(recs, "m", control):
            break
    assert i == points
    peak = find_k_star(_A549_KS[:i], values[:i])
    assert (peak.k_star, peak.type) == (k_star, "peak")
    # the old rule would have stopped at 3130 on a plateau with K* = 392
    assert find_k_star(_A549_KS[:6], values[:6]).k_star in (392, 1566)
