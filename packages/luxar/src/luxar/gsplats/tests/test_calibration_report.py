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

import luxar.gsplats.calibration_report as calibration_report  # noqa: E402
from luxar.gsplats.calibration_report import (  # noqa: E402
    _knee_display_idx,
    _plot_blind_spot,
    _selected_metric,
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

    def test_selected_metric_controls_curve_peak_and_knee(self):
        res = _make_result([1, 2, 4], [20.0, 21.0, 22.0])
        selected_curve = [18.0, 25.0, 19.0]
        res.k_star_metric = "psnr_fg_weighted"
        res.held_out_psnr_fg_weighted_db = selected_curve
        res.held_out_peak_selected = find_k_star(res.k_values_requested, selected_curve)

        curve, label, peak, resolved = _selected_metric(res)

        np.testing.assert_array_equal(curve, selected_curve)
        assert label == "foreground-weighted held-out PSNR"
        assert peak is res.held_out_peak_selected
        assert resolved == "psnr_fg_weighted"
        assert _knee_display_idx(res) is None

    def test_missing_selected_peak_falls_back_to_minmax_curve(self):
        res = _make_result([1, 2, 4], [20.0, 21.0, 22.0])
        res.k_star_metric = "psnr_fg_weighted"
        res.held_out_psnr_fg_weighted_db = [float("nan")] * 3

        curve, label, peak, resolved = _selected_metric(res)

        np.testing.assert_array_equal(curve, res.held_out_psnr_db)
        assert label == "held-out PSNR"
        assert peak is res.held_out_peak
        assert resolved == "psnr_minmax"

    def test_fallback_draws_only_matching_minmax_overlays(self):
        import matplotlib.pyplot as plt

        res = _make_result([1, 2, 4], [20.0, 21.0, 22.0])
        res.k_star_metric = "psnr_fg_weighted"
        res.held_out_psnr_fg_weighted_db = [float("nan")] * 3
        fig, ax = plt.subplots()

        _plot_blind_spot(fig, ax, res)

        labels = [line.get_label() for line in ax.lines]
        assert "train" in labels
        assert len(ax.collections) == 1
        assert any("noise floor" in text.get_text() for text in ax.texts)
        plt.close(fig)

    def test_weighted_curve_omits_minmax_overlays(self):
        import matplotlib.pyplot as plt

        res = _make_result([1, 2, 4], [20.0, 21.0, 22.0])
        selected_curve = [18.0, 25.0, 19.0]
        res.k_star_metric = "psnr_fg_weighted"
        res.held_out_psnr_fg_weighted_db = selected_curve
        res.held_out_peak_selected = find_k_star(res.k_values_requested, selected_curve)
        fig, ax = plt.subplots()

        _plot_blind_spot(fig, ax, res)

        labels = [line.get_label() for line in ax.lines]
        assert "train" not in labels
        assert len(ax.collections) == 0
        assert not any("noise floor" in text.get_text() for text in ax.texts)
        plt.close(fig)

    def test_legacy_zero_knee_is_suppressed(self):
        # A hand-built HeldOutPeak with the default k_knee=0 (pre-field object)
        res = _make_result(SIGNAL_LIMITED_KS, SIGNAL_LIMITED_PSNR)
        res.held_out_peak.k_knee = 0
        assert _knee_display_idx(res) is None


class TestRenderReport:
    def test_selected_peak_controls_report_text_montage_and_metadata(
        self, tmp_path, monkeypatch
    ):
        res = _make_result([10, 20, 40], [20.0, 25.0, 24.0])
        selected_curve = [20.0, 24.0, 30.0]
        res.k_star_metric = "psnr_fg_weighted"
        res.held_out_psnr_fg_weighted_db = selected_curve
        res.held_out_peak_selected = find_k_star(res.k_values_requested, selected_curve)
        captured_pages = []
        metadata = {}
        rendered_paths = []

        class CapturePdfPages:
            def __init__(self, _path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def savefig(self, fig):
                captured_pages.append(
                    (
                        fig._suptitle.get_text() if fig._suptitle else "",
                        [text.get_text() for text in fig.texts],
                    )
                )

            def infodict(self):
                return metadata

        def render_splat(path, shape):
            rendered_paths.append(path)
            return np.zeros(shape, dtype=np.float32)

        monkeypatch.setattr("matplotlib.backends.backend_pdf.PdfPages", CapturePdfPages)
        monkeypatch.setattr(calibration_report, "_render_splat_path", render_splat)

        render_calibration_report(
            res,
            np.zeros((32, 32, 32), dtype=np.float32),
            tmp_path / "report.pdf",
            splat_paths=["k10", "k20", "k40"],
        )

        assert "K* = 40" in captured_pages[0][0]
        assert "metric: psnr_fg_weighted" in captured_pages[0][0]
        assert "psnr_minmax K* = 20" in captured_pages[0][0]
        assert any("K* = 40" in text for text in captured_pages[1][1])
        assert "K* = 40" in captured_pages[2][0]
        assert rendered_paths == ["k10", "k40", "k40"]
        assert metadata["Subject"] == (
            f"Recommended K = 40 (psnr_fg_weighted, {res.held_out_peak_selected.type})"
        )

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
