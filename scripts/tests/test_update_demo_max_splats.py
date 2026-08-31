"""Tests for applying calibrated splat budgets to demos."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "update_demo_max_splats.py"
_spec = importlib.util.spec_from_file_location("update_demo_max_splats", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
update = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update)


def test_cells3d_per_channel_budget_is_not_calibration_managed() -> None:
    assert "cells3d_multichannel" not in update.DEMO_FILE_MAP


def test_main_reports_missing_constant_as_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summaries = tmp_path / "summaries.json"
    summaries.write_text(
        json.dumps(
            [
                {
                    "name": "renamed_constant_demo",
                    "status": "ok",
                    "recommended_k_star": 12_345,
                }
            ]
        )
    )
    demos_dir = tmp_path / "demos"
    demos_dir.mkdir()
    (demos_dir / "demo.py").write_text("PER_CHANNEL_SPLATS = 12000\n")

    monkeypatch.setattr(update, "SUMMARIES", summaries)
    monkeypatch.setattr(update, "DEMOS_DIR", demos_dir)
    monkeypatch.setattr(update, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        update,
        "DEMO_FILE_MAP",
        {"renamed_constant_demo": ("demo.py", "MAX_SPLATS")},
    )

    assert update.main() == 1

    output = capsys.readouterr().out
    assert "ERROR: MAX_SPLATS not found in demo.py" in output
    assert "demo.py → MAX_SPLATS=" not in output
    assert "Updated 0 demo files." in output
