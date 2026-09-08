"""Focused tests for ``luxar gsplat cal`` output helpers."""

from types import SimpleNamespace

from luxar.cli.gsplat_ops.fitting.calibrate import _selected_summary


def test_selected_summary_fallback_displays_minmax_curve() -> None:
    result = SimpleNamespace(
        held_out_peak=object(),
        held_out_peak_selected=None,
        held_out_psnr_db=[20.0, 21.0],
        held_out_psnr_fg_db=[],
        held_out_psnr_fg_weighted_db=[float("nan"), float("nan")],
        held_out_gain_db=[],
        k_star_metric="psnr_fg_weighted",
        k_values_requested=[20, 60],
    )

    peak, label, curve = _selected_summary(result)

    assert peak is result.held_out_peak
    assert label == "psnr_minmax (fallback: selected curve is undefined)"
    assert curve is result.held_out_psnr_db
