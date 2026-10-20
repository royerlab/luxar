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

    def test_selected_metric_controls_decoupled_knee(self):
        res = _make_result([10, 20, 40, 80], [10.0, 15.0, 19.9, 20.0])
        selected_curve = [10.0, 19.85, 19.9, 20.0]
        res.k_star_metric = "psnr_fg_weighted"
        res.held_out_psnr_fg_weighted_db = selected_curve
        res.held_out_peak_selected = find_k_star(res.k_values_requested, selected_curve)

        assert res.held_out_peak.k_knee == 40
        assert res.held_out_peak_selected.k_knee == 20
        assert _knee_display_idx(res) == 1

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
        assert any("noise ceiling" in text.get_text() for text in ax.texts)
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
        assert not any("noise ceiling" in text.get_text() for text in ax.texts)
        plt.close(fig)

    def test_legacy_zero_knee_is_suppressed(self):
        # A hand-built HeldOutPeak with the default k_knee=0 (pre-field object)
        res = _make_result(SIGNAL_LIMITED_KS, SIGNAL_LIMITED_PSNR)
        res.held_out_peak.k_knee = 0
        assert _knee_display_idx(res) is None


class TestRenderReport:
    def test_fallback_reason_reaches_report_text_and_metadata(
        self, tmp_path, monkeypatch
    ):
        res = _make_result([10, 20, 40], [20.0, 25.0, 24.0])
        res.k_star_metric = "psnr_fg_weighted"
        res.held_out_psnr_fg_weighted_db = [float("nan")] * 3
        captured_pages = []
        metadata = {}

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

        monkeypatch.setattr("matplotlib.backends.backend_pdf.PdfPages", CapturePdfPages)

        render_calibration_report(
            res,
            np.zeros((32, 32, 32), dtype=np.float32),
            tmp_path / "report.pdf",
        )

        fallback = "psnr_minmax (fallback from psnr_fg_weighted — curve undefined)"
        assert f"metric: {fallback}" in captured_pages[0][0]
        assert any(f"metric: {fallback}" in text for text in captured_pages[1][1])
        assert metadata["Subject"] == f"Recommended K = 20 ({fallback}, peak)"

    def test_selected_peak_controls_report_text_montage_and_metadata(
        self, tmp_path, monkeypatch
    ):
        res = _make_result([10, 20, 40, 80], [20.0, 25.0, 24.0, 23.0])
        selected_curve = [20.0, 24.0, 30.0, 29.0]
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
            splat_paths=["k10", "k20", "k40", "k80"],
        )

        assert "K* = 40" in captured_pages[0][0]
        assert "metric: psnr_fg_weighted" in captured_pages[0][0]
        assert "psnr_minmax K* = 20" in captured_pages[0][0]
        assert any("K* = 40" in text for text in captured_pages[1][1])
        assert "K* = 40" in captured_pages[2][0]
        assert rendered_paths == ["k10", "k40", "k80"]
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


class TestRobustSSIMField:
    """``full_ssim_robust``: persisted, and old files fall back to ``full_ssim``."""

    def test_round_trips_through_json_and_old_files_load(self, tmp_path) -> None:
        import json

        res = _make_result([1, 2, 4], [20.0, 21.0, 22.0])
        res.full_ssim_robust = [0.4, 0.5, 0.6]
        path = tmp_path / "cal.json"
        res.to_json(path)
        assert CalibrationResult.from_json(path).full_ssim_robust == [0.4, 0.5, 0.6]
        raw = json.loads(path.read_text())
        del raw["full_ssim_robust"]
        path.write_text(json.dumps(raw))
        assert CalibrationResult.from_json(path).full_ssim_robust == []

    @pytest.mark.parametrize("robust", [True, False])
    def test_cal_table_uses_robust_ssim_or_says_it_fell_back(
        self, robust: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from luxar.cli.gsplat_ops.fitting.calibrate import _print_cal_table

        res = _make_result([1, 2, 4], [20.0, 21.0, 22.0])
        res.full_ssim = [0.981, 0.982, 0.983]
        if robust:
            res.full_ssim_robust = [0.311, 0.312, 0.313]
        _print_cal_table(res)
        out = capsys.readouterr().out
        if robust:
            assert "0.311" in out and "0.981" not in out
            assert "PSNR and SSIM on the robust peak" in out
        else:
            assert "0.981" in out
            assert "PSNR (SSIM on the full range) on the robust peak" in out

    @pytest.mark.parametrize(
        ("robust", "label"), [(True, "SSIM (robust peak)"), (False, "SSIM")]
    )
    def test_pdf_ssim_panel_label(self, robust: bool, label: str) -> None:
        import matplotlib.pyplot as plt

        from luxar.gsplats.calibration_report import _plot_rate_distortion

        res = _make_result([1, 2, 4], [20.0, 21.0, 22.0])
        if robust:
            res.full_ssim_robust = [0.4, 0.5, 0.6]
        fig, axes = plt.subplots(2, 2)
        try:
            _plot_rate_distortion(fig, *axes.ravel(), res)
            assert axes.ravel()[1].get_ylabel() == label
        finally:
            plt.close(fig)


class TestPsnrBasis:
    """The report plots PSNR on the robust peak the terminal table quotes."""

    def _robust_result(self) -> CalibrationResult:
        res = _make_result([10, 20, 40], [40.0, 42.0, 41.0])
        res.data_range, res.robust_data_range = 100.0, 50.0  # -6.02 dB
        res.noise_sigma_rms = 0.5
        res.psnr_ceiling_robust_db = 40.0
        return res

    def test_curves_shift_to_the_robust_peak_and_the_axis_says_so(self):
        import matplotlib.pyplot as plt

        res = self._robust_result()
        fig, axes = plt.subplots(2, 2)
        calibration_report._plot_rate_distortion(
            fig, axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], res
        )
        ax = axes[0, 0]
        assert "robust peak" in ax.get_ylabel()
        held = ax.get_lines()[0].get_ydata()
        assert held == pytest.approx([40.0 - 6.0206, 42.0 - 6.0206, 41.0 - 6.0206])
        # The ceiling line is the RMS-noise ceiling, already on the robust basis.
        assert any("noise ceiling ≈ 40.0 dB" in t.get_text() for t in ax.texts)
        # The train/held-out gap is a difference: unshifted.
        gap = axes[1, 1].get_lines()[0].get_ydata()
        assert gap == pytest.approx([1.0, 1.0, 1.0])
        plt.close(fig)

    def test_blind_spot_page_uses_the_same_basis(self):
        import matplotlib.pyplot as plt

        res = self._robust_result()
        fig, ax = plt.subplots()
        _plot_blind_spot(fig, ax, res)
        assert "robust peak" in ax.get_ylabel()
        held = [line for line in ax.get_lines() if line.get_marker() == "o"][0]
        assert held.get_ydata()[1] == pytest.approx(42.0 - 6.0206)
        plt.close(fig)

    def test_legacy_result_stays_full_range_and_labelled(self):
        res = _make_result([10, 20, 40], [40.0, 42.0, 41.0])
        off, label = calibration_report._psnr_basis(res)
        assert off == 0.0 and "full range" in label
        # Legacy ceiling falls back to the median-voxel ensemble value.
        assert calibration_report._plot_ceiling_db(res) == pytest.approx(45.0)
        assert "median-voxel" in calibration_report._noise_annotation(res)

    def test_montage_target_is_floor_subtracted(self):
        res = _make_result([10, 20, 40], [40.0, 42.0, 41.0])
        res.fit_config = {"floor_subtracted": 100.0}
        V = np.full((32, 32, 32), 90.0, np.float32)
        V[0, 0, 0] = 400.0
        target = calibration_report._montage_target(V, res)
        assert target.min() == 0.0 and target.max() == pytest.approx(300.0)
        res.fit_config = {}
        assert calibration_report._montage_target(V, res).max() == 400.0
