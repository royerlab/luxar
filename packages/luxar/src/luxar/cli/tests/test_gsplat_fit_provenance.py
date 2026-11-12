"""CLI provenance spans loading, fitting, and the completed store write."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar.cli import app
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import _stamp_content_hash
from luxar.gsplats.planner.spec import FitPlan, PlanBox


def test_cli_fit_records_command_and_save_timing(tmp_path, monkeypatch) -> None:
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[3:5, 3:5, 3:5] = 1.0
    source = tmp_path / "source.npy"
    output = tmp_path / "output.gsplats.zarr"
    np.save(source, volume)
    argv = [
        "luxar",
        "gsplat",
        "fit",
        str(source),
        str(output),
        "--tiling",
        "none",
        "--preset",
        "draft",
        "--seeds",
        "8",
        "--iters",
        "6",
        "--device",
        "cpu",
        "--floor",
        "none",
        "--quiet",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    result = CliRunner().invoke(app, argv[1:])
    assert result.exit_code == 0, result.output

    root = zarr.open_group(str(output), mode="r")
    consolidated = json.loads((output / "zarr.json").read_text())[
        "consolidated_metadata"
    ]
    assert (
        consolidated["metadata"]["fitting"]["attributes"]["fit_provenance"]["timings"][
            "saving"
        ]
        > 0
    )
    record = root["fitting"].attrs["fit_provenance"]
    assert record["command"] == argv
    assert record["config"]["preset"] == "draft"
    assert record["config"]["seed_count"] > 0
    assert record["timings"]["loading"] >= 0
    assert record["timings"]["cli_preprocessing"] >= 0
    assert record["timings"]["saving"] > 0
    resaved = tmp_path / "resaved.gsplats.zarr"
    loaded = GSplatData.load(output, include_stats=True)
    loaded.save(resaved)
    assert (
        zarr.open_group(str(resaved), mode="r")["fitting"].attrs["fit_provenance"][
            "timings"
        ]["saving"]
        == record["timings"]["saving"]
    )
    assert (
        loaded.stats["fit_provenance"]["timings"]["saving"]
        == record["timings"]["saving"]
    )
    unstamped = GSplatData.load(output, include_stats=True)
    unstamped.stats["fit_provenance"]["timings"].pop("saving")
    unstamped_path = tmp_path / "first_save.gsplats.zarr"
    unstamped.save(unstamped_path)
    assert "saving" not in unstamped.stats["fit_provenance"]["timings"]
    assert (
        zarr.open_group(str(unstamped_path), mode="r")["fitting"].attrs[
            "fit_provenance"
        ]["timings"]["saving"]
        >= 0
    )
    before = root.attrs["content_hash"]
    mutable = zarr.open_group(str(output), mode="r+")
    assert _stamp_content_hash(mutable) == before

    summary = CliRunner().invoke(
        app, ["gsplat", "info", str(output), "--no-histograms", "--provenance"]
    )
    assert summary.exit_code == 0, summary.output
    assert "Fit provenance: 1 fit(s)" in summary.output
    assert "saving:" in summary.output

    legacy = GSplatData.load(output, include_stats=True)
    legacy.stats.pop("early_stop_reason", None)
    legacy.stats["fit_provenance"]["outcome"].pop("early_stop_reason", None)
    legacy_path = tmp_path / "without_stop_reason.gsplats.zarr"
    legacy.save(legacy_path)
    legacy_summary = CliRunner().invoke(
        app, ["gsplat", "info", str(legacy_path), "--no-histograms", "--provenance"]
    )
    assert legacy_summary.exit_code == 0, legacy_summary.output
    assert "Stop reasons: unrecorded 1" in legacy_summary.output

    mixed = zarr.open_group(str(legacy_path), mode="r+")
    mixed["fitting"].attrs["part_provenance"] = [
        {
            "slot_id": "tile_0",
            "fitting": {
                "fit_provenance": {"timings": {"loading": 100.0, "optimization": 1.0}}
            },
        },
        {
            "slot_id": "tile_1",
            "fitting": {"fit_provenance": {"timings": {"optimization": 2.0}}},
        },
    ]
    mixed_summary = CliRunner().invoke(
        app, ["gsplat", "info", str(legacy_path), "--no-histograms", "--provenance"]
    )
    assert mixed_summary.exit_code == 0, mixed_summary.output
    assert "Fit stages (sum): 3.00 s" in mixed_summary.output
    assert "Slowest: fit/tile_1 2.00s, fit/tile_0 1.00s" in mixed_summary.output


@pytest.mark.parametrize("jobs", [1, 2])
def test_content_fit_keeps_box_provenance(tmp_path, monkeypatch, jobs: int) -> None:
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[6:10, 6:10, 3:5] = 1.0
    volume[6:10, 6:10, 11:13] = 1.0
    source = tmp_path / "content.npy"
    output = tmp_path / "content.gsplats.zarr"
    plan_path = tmp_path / "plan.json"
    np.save(source, volume)
    FitPlan(
        volume_shape=[16, 16, 16],
        boxes=[
            PlanBox([0, 16, 0, 16, 0, 8], 16, 1),
            PlanBox([0, 16, 0, 16, 8, 16], 16, 1),
        ],
        overlap=2,
        feature_method="intensity",
        min_leaf=8,
        max_leaf=16,
    ).to_json(plan_path)
    argv = [
        "luxar",
        "gsplat",
        "fit",
        str(source),
        str(output),
        "--tiling",
        "content",
        "--plan",
        str(plan_path),
        "--k-star-ref",
        "16",
        "--n-features-ref",
        "2",
        "--preset",
        "draft",
        "--iters",
        "3",
        "--device",
        "cpu",
        "--floor",
        "none",
        "--quiet",
        "--jobs",
        str(jobs),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    result = CliRunner().invoke(app, argv[1:])
    assert result.exit_code == 0, result.output
    parts = zarr.open_group(str(output), mode="r")["fitting"].attrs["part_provenance"]
    assert [part["slot_id"] for part in parts] == ["box_0", "box_1"]
    assert [part["region"] for part in parts] == [
        [[0, 16], [0, 16], [0, 10]],
        [[0, 16], [0, 16], [6, 16]],
    ]
    commands = [part["fitting"]["fit_provenance"]["command"] for part in parts]
    if jobs == 1:
        assert commands == [argv, argv]
    else:
        assert all("--plan-box" in command for command in commands)
