"""Focused tests for ``luxar gsplat cal`` output helpers."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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


def _fake_result(k_grid: list[int], psnr: list[float]) -> Any:
    from luxar.gsplats.calibration import CalibrationResult, NoiseFloor, find_k_star

    n = len(k_grid)
    return CalibrationResult(
        k_values_requested=list(k_grid),
        k_values_effective=list(k_grid),
        held_out_psnr_db=list(psnr),
        train_psnr_db=list(psnr),
        held_out_mse=[1e-4] * n,
        full_psnr_db=list(psnr),
        full_ssim=[0.9] * n,
        held_out_peak=find_k_star(k_grid, psnr),
        noise_floor=NoiseFloor(
            sigma_hat=0.01,
            sigma_laplacian=0.01,
            sigma_haar=0.01,
            sigma_background=0.01,
            psnr_max_db=40.0,
        ),
        fit_times_seconds=[1.0] * n,
        splat_paths=None,
        mask_seed=42,
        mask_fraction=0.05,
        donut_radius=1,
        fit_config={},
        volume_shape=[8, 16, 16],
        volume_dtype="float32",
        timestamp="2026-10-14T00:00:00+00:00",
    )


def test_spot_warning_reaches_cal_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from luxar.cli.gsplat_ops.fitting import calibrate as cal_cli
    from luxar.gsplats.calibration import CalibrationResult

    result = _fake_result([1000, 4000], [38.0, 37.0])
    result.k_values_effective = [800, 3000]
    result.spot_count = 40
    result.spot_peak_ratio = [0.5, 0.9]
    path = tmp_path / "cal.json"
    result.to_json(path)

    lines: list[str] = []
    monkeypatch.setattr(cal_cli, "aprint", lambda message: lines.append(str(message)))
    cal_cli._print_regime_warnings(CalibrationResult.from_json(path))
    output = "\n".join(lines)
    assert "small bright structures: K*=1,000" in output
    assert "--seeds 4,000" in output


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quiet: bool,
    psnr: tuple[float, ...] = (38.0, 37.0, 36.0),  # falling: best at smallest K
    not_converged: bool = False,
) -> dict[str, Any]:
    import json

    import numpy as np

    import luxar.gsplats.calibration as cal_mod
    from luxar.cli.gsplat_ops.fitting.calibrate import run_calibrate_command

    ks = [1000, 2000, 4000]
    fake = _fake_result(ks, list(psnr))
    fake.not_converged = not_converged
    monkeypatch.setattr(cal_mod, "calibrate", lambda *a, **k: fake)
    vol = tmp_path / "vol.npy"
    np.save(vol, np.random.default_rng(0).random((8, 16, 16), dtype=np.float32))
    out = tmp_path / "cal.json"
    run_calibrate_command(
        input_path=vol,
        output_json=out,
        k_grid="1000,2000,4000",
        n_grid=3,
        k_min=1000,
        k_max=4000,
        progression="exp",
        power=2,
        mask_seed=42,
        mask_fraction=0.05,
        preset="draft",
        config=None,
        floor=None,
        device="cpu",
        channel=None,
        timepoint=None,
        array_key=None,
        axes=None,
        k_star_metric="psnr_minmax",
        fg_bg_ratio=1.0,
        auto_region=False,
        region_size=256,
        region_strategy="densest",
        feature_metric="peaks",
        saturation_exponent=0.44,
        rd_model=False,
        fit_exponent=False,
        exponent_scales=None,
        pdf_report=None,
        keep_fits=None,
        quiet=quiet,
    )
    data: dict[str, Any] = json.loads(out.read_text())
    return data


def test_grid_edge_warning_printed_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _run(tmp_path, monkeypatch, quiet=False)
    printed = capsys.readouterr().out
    assert data["held_out_peak"]["type"] == "low_k_optimal"
    assert any("smallest grid point" in w for w in data["held_out_peak"]["warnings"])
    assert "type: low_k_optimal" in printed
    assert "smallest grid point" in printed and "--k-min" in printed


def test_grid_edge_warning_survives_quiet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(tmp_path, monkeypatch, quiet=True)
    printed = capsys.readouterr().out
    assert "Recommended K*" not in printed  # the table is suppressed
    assert "smallest grid point" in printed  # the warning is not


def test_signal_limited_prints_one_high_edge_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With the not-converged line printed, the "largest grid point" warning
    would say the same thing again: the table keeps only the former, while
    cal.json still records the grid warning."""
    data = _run(
        tmp_path, monkeypatch, quiet=False, psnr=(30.0, 34.0, 38.0), not_converged=True
    )
    printed = capsys.readouterr().out
    assert data["held_out_peak"]["type"] == "signal_limited"
    assert any("largest grid point" in w for w in data["held_out_peak"]["warnings"])
    assert "not converged" in printed
    assert "largest grid point" not in printed
