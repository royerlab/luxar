"""`gsplat cal` sweep UX: auto grid, early stop, checkpoint, Ctrl-C, stored count."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.tests._testing import normalized_cli_output
from luxar.gsplats.calibration import driver as driver_module


def _volume(path: Path) -> Path:
    rng = np.random.default_rng(0)
    V = np.full((12, 24, 24), 100.0, np.float32)
    zz, yy, xx = np.indices(V.shape)
    for _ in range(6):
        c = [rng.uniform(0, s) for s in V.shape]
        V += 300.0 * np.exp(
            -((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / 8.0
        ).astype(np.float32)
    np.save(path, V + rng.normal(0, 3, V.shape).astype(np.float32))
    return path


def _stub(curve: Dict[int, float], calls: List[int], stop_at: int | None = None):
    def run(self: Any, K: int) -> Dict[str, Any]:
        if stop_at is not None and K == stop_at:
            raise KeyboardInterrupt
        calls.append(int(K))
        v = curve.get(int(K), 30.0)
        return {
            "k": int(K),
            "eff": int(K) - 1,
            "n_fitted": int(K),
            "n_stored": int(K) // 3,
            "fit_s": 0.01,
            "path": None,
            "held_mse": 10 ** (-v / 10),
            "held_psnr": v,
            "held_psnr_robust": v + 2.0,
            "held_psnr_fg": v - 10.0,
            "held_psnr_fg_robust": v - 8.0,
            "held_gain": v - 20.0,
            "held_psnr_fg_weighted": v - 5.0,
            "train_psnr": v + 1.0,
            "full_psnr": v,
            "full_ssim": 0.9,
        }

    return run


def _cal(tmp_path: Path, *extra: str) -> tuple[int, str, Path]:
    src = _volume(tmp_path / "v.npy")
    out = tmp_path / "cal.json"
    result = CliRunner().invoke(
        app, ["gsplat", "cal", str(src), str(out), "--device", "cpu", *extra]
    )
    return result.exit_code, normalized_cli_output(result), out


def test_default_grid_is_derived_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: List[int] = []
    monkeypatch.setattr(driver_module.PointContext, "run", _stub({}, calls))
    code, output, out = _cal(tmp_path, "--n-grid", "4", "--no-early-stop")
    assert code == 0, output
    cal = json.loads(out.read_text())
    assert cal["sweep"]["grid_source"] == "auto"
    auto = cal["sweep"]["auto_grid"]
    assert auto["k_max"] < 512_000 and auto["n_foreground_voxels"] > 0
    assert calls[0] == auto["k_min"] and max(calls) == auto["k_max"]
    assert "Auto K grid" in output
    assert cal["status"] == "complete"


def test_kstar_line_reports_the_stored_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curve = {1000: 30.0, 2000: 32.0, 4000: 31.0}
    monkeypatch.setattr(driver_module.PointContext, "run", _stub(curve, []))
    code, output, out = _cal(tmp_path, "--k-grid", "1000,2000,4000")
    assert code == 0, output
    cal = json.loads(out.read_text())
    assert cal["k_values_stored"] == [333, 666, 1333]
    assert cal["k_star_stored"] == 666
    assert "fit --seeds 2,000 will store about 666 splats" in output
    assert "robust peak" in output


def test_ctrl_c_keeps_the_measured_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curve = {1000: 30.0, 2000: 32.0, 4000: 31.0, 8000: 30.0}
    calls: List[int] = []
    monkeypatch.setattr(driver_module.PointContext, "run", _stub(curve, calls, 8000))
    code, output, out = _cal(
        tmp_path, "--k-grid", "1000,2000,4000,8000", "--no-early-stop"
    )
    assert code == 130, output
    cal = json.loads(out.read_text())
    assert cal["status"] == "interrupted"
    assert cal["k_values_requested"] == [1000, 2000, 4000]
    assert cal["sweep"]["interrupted"] is True
    assert "best so far is K*=2,000" in output


def test_checkpoint_is_written_after_every_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: List[str] = []
    out = tmp_path / "cal.json"
    stub = _stub({}, [])

    def run(self: Any, K: int) -> Dict[str, Any]:
        if out.exists():
            seen.append(json.loads(out.read_text())["status"])
        return stub(self, K)

    monkeypatch.setattr(driver_module.PointContext, "run", run)
    code, output, _ = _cal(
        tmp_path, "--k-grid", "1000,2000,4000", "--no-early-stop", "--no-extend-below"
    )
    assert code == 0, output
    assert seen == ["partial", "partial"]
    assert json.loads(out.read_text())["status"] == "complete"


def test_minmax_k_star_is_labelled_when_the_stop_cut_its_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curve = {1000: 30.0, 2000: 32.0, 4000: 31.0, 8000: 30.0}
    calls: List[int] = []
    monkeypatch.setattr(driver_module.PointContext, "run", _stub(curve, calls))
    code, output, out = _cal(tmp_path, "--k-grid", "1000,2000,4000,8000")
    assert code == 0, output
    assert calls == [1000, 2000, 4000]
    cal = json.loads(out.read_text())
    assert cal["sweep"]["minmax_curve_truncated"] is True
    assert "on the early-stopped curve" in output
    # the saturation cap was not measured at the skipped grid top: estimated
    density = cal["splat_density"]
    assert density["saturation_cap_measured"] is False
    assert density["saturation_cap"] == round(8000 * 3999 / 4000)


def test_region_skips_the_whole_input_read_above_the_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from luxar.gsplats.calibration import whole_volume as wv

    curve = {1000: 30.0, 2000: 32.0, 4000: 31.0}
    monkeypatch.setattr(driver_module.PointContext, "run", _stub(curve, []))
    monkeypatch.setattr(wv, "MAX_WHOLE_INPUT_VOXELS", 10)
    code, output, out = _cal(
        tmp_path, "--k-grid", "1000,2000,4000", "--region", ":6,:,:"
    )
    assert code == 0, output
    assert "suggestion skipped" in output
    cal = json.loads(out.read_text())
    assert cal["whole_volume_seeds"] is None and cal["status"] == "complete"


def test_ctrl_c_after_the_sweep_keeps_a_complete_cal_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import luxar.gsplats.calibration as cal_pkg

    def boom(*a: Any, **k: Any) -> None:
        raise KeyboardInterrupt

    curve = {1000: 30.0, 2000: 32.0, 4000: 31.0}
    monkeypatch.setattr(driver_module.PointContext, "run", _stub(curve, []))
    monkeypatch.setattr(cal_pkg, "whole_volume_seeds", boom)
    code, output, out = _cal(
        tmp_path, "--k-grid", "1000,2000,4000", "--region", ":6,:,:"
    )
    assert code == 130, output
    cal = json.loads(out.read_text())
    assert cal["status"] == "complete"
    assert cal["sweep"]["post_sweep_interrupted"] is True


def test_content_density_from_an_unfinished_cal_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from luxar.cli.gsplat_ops.planner import _resolve_density

    curve = {1000: 30.0, 2000: 32.0, 4000: 31.0, 8000: 30.0}
    monkeypatch.setattr(
        driver_module.PointContext, "run", _stub(curve, [], stop_at=8000)
    )
    code, _, out = _cal(tmp_path, "--k-grid", "1000,2000,4000,8000", "--no-early-stop")
    assert code == 130
    capsys.readouterr()
    density = _resolve_density(out, None, None, 0.44, None, None)
    assert density.k_star_reference > 0
    assert "status is 'interrupted'" in capsys.readouterr().out
