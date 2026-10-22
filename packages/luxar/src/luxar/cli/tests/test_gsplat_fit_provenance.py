"""CLI provenance spans loading, fitting, and the completed store write."""

from __future__ import annotations

import sys

import numpy as np
import zarr
from typer.testing import CliRunner

from luxar.cli import app
from luxar.gsplats.io.save_gsplats import _stamp_content_hash


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
    record = root["fitting"].attrs["fit_provenance"]
    assert record["command"] == argv
    assert record["config"]["preset"] == "draft"
    assert record["config"]["seed_count"] > 0
    assert record["timings"]["loading"] >= 0
    assert record["timings"]["cli_preprocessing"] >= 0
    assert record["timings"]["saving"] > 0
    before = root.attrs["content_hash"]
    mutable = zarr.open_group(str(output), mode="r+")
    assert _stamp_content_hash(mutable) == before

    summary = CliRunner().invoke(
        app, ["gsplat", "info", str(output), "--no-histograms", "--provenance"]
    )
    assert summary.exit_code == 0, summary.output
    assert "Fit provenance: 1 fit(s)" in summary.output
    assert "saving:" in summary.output
