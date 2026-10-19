"""``fit --denoise`` says its quality summary was scored against the denoised input.

The fit scores itself against the volume it was handed, which under ``--denoise``
is the filtered one: a fit can read several dB better there while being worse
against the raw data, so the summary must not pass for a raw-volume score.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from luxar.cli.gsplat_ops.fitting.fit_utils import announce_denoised_quality_basis


def _ctx(h: float | None) -> Any:
    return cast(Any, SimpleNamespace(denoise_effective_h=h))


def test_silent_without_denoise(capsys: pytest.CaptureFixture[str]) -> None:
    announce_denoised_quality_basis(_ctx(None), is_tiled=False)
    assert "denoised" not in capsys.readouterr().out


def test_whole_volume_fit_names_the_denoised_reference(
    capsys: pytest.CaptureFixture[str],
) -> None:
    announce_denoised_quality_basis(_ctx(0.01), is_tiled=False)
    out = capsys.readouterr().out
    assert "vs the denoised input (NLM h=0.01)" in out
    assert "luxar gsplat compare" in out


def test_tiled_fit_separates_tile_and_merged_scores(
    capsys: pytest.CaptureFixture[str],
) -> None:
    announce_denoised_quality_basis(_ctx(0.02), is_tiled=True)
    out = capsys.readouterr().out
    assert "ran per tile" in out
    assert "vs its denoised tile" in out and "vs the raw input" in out


def test_parallel_tiled_fit_prints_the_note(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-j N`` exits from inside the dispatch; the note must come first."""
    import numpy as np
    from typer.testing import CliRunner

    import luxar.cli.gsplat_ops.fitting.fit as fit_mod
    from luxar.cli.main import app

    monkeypatch.setattr(fit_mod, "dispatch_parallel_tiled", lambda *a, **k: True)
    vol = tmp_path / "vol.npy"
    np.save(vol, np.random.default_rng(0).random((16, 16, 16)).astype(np.float32))
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(tmp_path / "out.gsplats.zarr"),
            "--tiling",
            "uniform",
            "-j",
            "2",
            "--device",
            "cpu",
            "--denoise",
            "--denoise-h",
            "0.02",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ran per tile" in result.output
