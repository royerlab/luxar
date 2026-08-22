"""Tests for the demo calibration driver's fit preprocessing contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "calibrate_gsplat_demos.py"
_spec = importlib.util.spec_from_file_location("calibrate_gsplat_demos", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


@pytest.mark.parametrize(
    ("floor", "expected"),
    [(675.0, "675.0"), (None, None)],
)
def test_cli_calibration_passes_only_an_explicit_floor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    floor: float | None,
    expected: str | None,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: Any) -> None:
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cal.subprocess, "run", fake_run)

    cal._run_cli_cal(
        tmp_path / "volume.npy",
        tmp_path / "cal.json",
        k_min=2_000,
        k_max=1_000_000,
        floor=floor,
    )

    command = captured["command"]
    assert isinstance(command, list)
    if expected is None:
        assert "--floor" not in command
    else:
        floor_index = command.index("--floor")
        assert command[floor_index + 1] == expected
    assert captured["kwargs"] == {"check": True, "cwd": str(cal.REPO_ROOT)}


def test_tribolium_registry_uses_the_demo_floor_constant() -> None:
    tribolium = next(demo for demo in cal.DEMOS if demo["name"] == "tribolium_embryo")
    assert tribolium["floor"]() == 675.0


def test_demo_floor_reaches_each_sample_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_calibrate(
        label: str,
        volume: np.ndarray,
        out_dir: Path,
        k_min: int,
        k_max: int,
        floor: float | None = None,
    ) -> dict[str, Any]:
        captured.update(
            label=label,
            volume=volume,
            out_dir=out_dir,
            k_min=k_min,
            k_max=k_max,
            floor=floor,
        )
        return {"label": label, "k_star": 12_000}

    volume = cal.np.ones((2, 2, 2), dtype=cal.np.float32)
    demo = {
        "name": "floor_test",
        "loader": lambda: [("sample", volume)],
        "k_min": 2_000,
        "k_max": 32_000,
        "floor": lambda: 675.0,
        "comment": "test",
    }
    monkeypatch.setattr(cal, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cal, "_calibrate_one_sample", fake_calibrate)

    summary = cal._calibrate_demo(demo, skip_existing=False)

    assert captured["floor"] == 675.0
    assert captured["volume"] is volume
    assert summary["recommended_k_star"] == 12_000
