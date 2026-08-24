"""Tests for shared CLI test helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pytest import MonkeyPatch
from typer.testing import CliRunner

from luxar.cli.gsplat_commands import app_gsplat
from luxar.cli.tests._testing import CliResult, normalized_cli_output


@dataclass
class _SplitResult:
    stdout: str
    stderr: str


class _StdoutOnlyResult:
    stdout = "ordinary output"

    @property
    def stderr(self) -> str:
        raise ValueError("stderr was not separately captured")


class _MissingStderrResult:
    stdout = "ordinary output"


def _write_manifest(output_dir: Path) -> None:
    from luxar.gsplats.batch.manifest import BatchManifest, save_manifest

    manifest = BatchManifest(
        input_path="/data/test.zarr",
        output_dir=str(output_dir),
        n_timepoints=1,
        n_channels=1,
        spatial_shape=(64, 64, 64),
        tile_size=64,
        n_tiles=1,
        total_tasks=1,
    )
    save_manifest(manifest, output_dir)


def test_normalized_cli_output_combines_and_flattens_rich_streams() -> None:
    result = _SplitResult(
        stdout="\x1b[38:2:1:2:3m--compression\x1b[0m-factor",
        stderr="╭─ Error ─╮\n│ --refine\n│ volume cannot be combined │\n╰─────────╯",
    )

    output = normalized_cli_output(result)

    assert "--compression-factor" in output
    assert "--refine volume cannot be combined" in output
    assert "\x1b" not in output
    assert "│" not in output


def test_normalized_cli_output_tolerates_unavailable_stderr() -> None:
    assert normalized_cli_output(_StdoutOnlyResult()) == "ordinary output"
    missing = cast(CliResult, _MissingStderrResult())
    assert normalized_cli_output(missing) == "ordinary output"


def test_normalized_cli_output_handles_real_rich_usage_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _write_manifest(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = CliRunner().invoke(
        app_gsplat, ["batch-fit", "merge", str(tmp_path), "--n-lods", "6"]
    )

    assert result.exit_code == 2
    assert "recipe-specific but no recipe is in effect" in normalized_cli_output(result)
