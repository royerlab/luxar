"""Tests for the multi-page PDF calibration report (``--pdf``)."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from luxar.gsplats.calibration import (
    CalibrationResult,
    NoiseFloor,
    find_k_star,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from luxar.gsplats.calibration_report import (  # noqa: E402
    _knee_display_idx,
    render_calibration_report,
)


def _make_result(ks: List[int], held: List[float]) -> CalibrationResult:
    peak = find_k_star(ks, held)
    return CalibrationResult(
        k_values_requested=ks,
        k_values_effective=ks,
        held_out_psnr_db=held,
        train_psnr_db=[h + 1.0 for h in held],
        held_out_mse=[1e-4] * len(ks),
        full_psnr_db=held,
        full_ssim=[0.8 + 0.02 * i for i in range(len(ks))],
        held_out_peak=peak,
        noise_floor=NoiseFloor(
            sigma_hat=0.01,
            sigma_laplacian=0.01,
            sigma_haar=0.01,
            sigma_background=0.01,
            psnr_max_db=45.0,
        ),
        fit_times_seconds=[float(i + 1) for i in range(len(ks))],
        splat_paths=None,
        mask_seed=0,
        mask_fraction=0.05,
        donut_radius=2,
        fit_config={},
        volume_shape=[32, 32, 32],
        volume_dtype="float32",
        timestamp="2026-01-01T00:00:00",
    )


# Signal-limited curve whose knee (256000) decouples from K* (512000)
SIGNAL_LIMITED_KS = [16000, 64000, 128000, 256000, 512000]
SIGNAL_LIMITED_PSNR = [29.2, 35.2, 37.0, 37.59, 37.84]


class TestKneeDisplayIdx:
    def test_signal_limited_knee_is_shown(self):
        res = _make_result(SIGNAL_LIMITED_KS, SIGNAL_LIMITED_PSNR)
        assert res.held_out_peak.k_knee == 256000
        assert res.held_out_peak.k_star == 512000
        assert _knee_display_idx(res) == 3

    def test_peak_knee_is_suppressed(self):
        # peak: k_knee == k_star -> the report shows only the K* marker
        res = _make_result(
            [1, 2, 4, 8, 16, 32, 64], [22.0, 26.0, 30.0, 34.0, 33.5, 32.0, 30.0]
        )
        assert res.held_out_peak.k_knee == res.held_out_peak.k_star
        assert _knee_display_idx(res) is None

    def test_legacy_zero_knee_is_suppressed(self):
        # A hand-built HeldOutPeak with the default k_knee=0 (pre-field object)
        res = _make_result(SIGNAL_LIMITED_KS, SIGNAL_LIMITED_PSNR)
        res.held_out_peak.k_knee = 0
        assert _knee_display_idx(res) is None


class TestRenderReport:
    def test_render_with_decoupled_knee(self, tmp_path):
        res = _make_result(SIGNAL_LIMITED_KS, SIGNAL_LIMITED_PSNR)
        out = tmp_path / "report.pdf"
        render_calibration_report(res, np.zeros((32, 32, 32), dtype=np.float32), out)
        assert out.exists() and out.stat().st_size > 0

    def test_render_peak_curve(self, tmp_path):
        res = _make_result(
            [1, 2, 4, 8, 16, 32, 64], [22.0, 26.0, 30.0, 34.0, 33.5, 32.0, 30.0]
        )
        out = tmp_path / "report.pdf"
        render_calibration_report(res, np.zeros((32, 32, 32), dtype=np.float32), out)
        assert out.exists() and out.stat().st_size > 0
