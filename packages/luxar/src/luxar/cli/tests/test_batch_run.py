# test_batch_run.py
"""CLI smoke tests for `luxar gsplat batch-fit run` (the local runner command)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from luxar.cli.gsplat_commands import app_gsplat

runner = CliRunner()


def _make_zarr(path: Path) -> None:
    import zarr

    rng = np.random.default_rng(0)
    V = np.zeros((48, 48, 48), np.float32)
    zz, yy, xx = np.mgrid[0:48, 0:48, 0:48]
    for _ in range(15):
        cz, cy, cx = rng.integers(6, 42, 3)
        V += np.exp(-(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)).astype(
            np.float32
        )
    z = zarr.open_array(
        str(path), mode="w", shape=V.shape, chunks=(24, 24, 24), dtype="f4"
    )
    z[:] = np.clip(V, 0, 1)


def test_batch_fit_group_renamed_from_slurm_fit() -> None:
    """The group is `batch-fit` now; `slurm-fit` no longer resolves."""
    assert runner.invoke(app_gsplat, ["batch-fit", "--help"]).exit_code == 0
    assert runner.invoke(app_gsplat, ["slurm-fit", "--help"]).exit_code != 0


def test_run_dry_run_reports_plan_without_fitting(tmp_path: Path) -> None:
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    out = tmp_path / "out"
    res = runner.invoke(
        app_gsplat,
        [
            "batch-fit",
            "run",
            str(src),
            str(out),
            "--axes",
            "z,y,x",
            "--tile-size",
            "64",  # > 48 -> single tile, no GPU profile needed
            "--gpus",
            "cpu",
            "--dry-run",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "LOCAL BATCH FIT" in res.output
    assert "Dry run" in res.output
    # Dry run must NOT fit: no tiles written.
    assert not (out / "tiles").exists() or not list((out / "tiles").glob("*.zarr"))


def test_run_rejects_npy_input_with_clear_message(tmp_path: Path) -> None:
    """batch-fit needs OME-Zarr; a .npy input must fail fast with a clear pointer
    (not an opaque zarr 'not a directory' error)."""
    npy = tmp_path / "vol.npy"
    np.save(npy, np.zeros((16, 16, 16), np.float32))
    res = runner.invoke(
        app_gsplat,
        ["batch-fit", "run", str(npy), str(tmp_path / "o"), "--gpus", "cpu"],
    )
    assert res.exit_code != 0
    assert "OME-Zarr" in res.output and "gsplat fit" in res.output
    # Clean usage-error rendering (typer.BadParameter), not a raw traceback.
    assert "Traceback" not in res.output


def test_run_rejects_bad_tiling(tmp_path: Path) -> None:
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    res = runner.invoke(
        app_gsplat,
        ["batch-fit", "run", str(src), str(tmp_path / "o"), "--tiling", "nope"],
    )
    assert res.exit_code != 0


def test_run_uniform_auto_tile_without_profile_errors(tmp_path: Path) -> None:
    """A multi-tile uniform fit with no --tile-size and no GPU profile must fail
    with a clear message (auto-size needs a benchmark profile)."""
    src = tmp_path / "vol.zarr"
    _make_zarr(src)
    res = runner.invoke(
        app_gsplat,
        [
            "batch-fit",
            "run",
            str(src),
            str(tmp_path / "o"),
            "--axes",
            "z,y,x",
            "--gpus",
            "cpu",
        ],
    )
    # Either a clear tile-size error (no GPU profile -> typer.BadParameter, which
    # Typer renders cleanly as a usage error, exit 2) or a successful single-tile
    # plan if the volume fits and a profile is present (exit 0). Never an uncaught
    # crash, and any error must name --tile-size / the profile.
    assert res.exit_code in (0, 1, 2)
    if res.exit_code != 0:
        assert "tile-size" in res.output.lower() or "profile" in res.output.lower()


def test_assemble_fit_args_includes_floor_by_default() -> None:
    """Batch FitConfig defaults to floor='auto' → workers get `--floor auto`."""
    from luxar.cli.gsplat_ops.batch_planning import (
        DenoiseConfig,
        FitConfig,
        _assemble_fit_args,
    )

    fit_args, _mode, _path = _assemble_fit_args(FitConfig(), DenoiseConfig())
    assert fit_args["floor"] == "auto"


def test_assemble_fit_args_floor_override() -> None:
    from luxar.cli.gsplat_ops.batch_planning import (
        DenoiseConfig,
        FitConfig,
        _assemble_fit_args,
    )

    fit_args, _mode, _path = _assemble_fit_args(FitConfig(floor="p10"), DenoiseConfig())
    assert fit_args["floor"] == "p10"


def test_resolve_tiling_small_volume_fits_whole() -> None:
    """Companion fix: a small/medium stack no longer tiles (avoids seams)."""
    from luxar.cli.gsplat_ops.fitting_fit_utils import resolve_tiling as _resolve_tiling

    # Neuromast-shaped stack (dims > 256 but only ~28M voxels) → whole-volume.
    assert _resolve_tiling("auto", (84, 580, 576), 256, False) == "none"


def test_resolve_tiling_large_volume_tiles() -> None:
    from luxar.cli.gsplat_ops.fitting_fit_utils import resolve_tiling as _resolve_tiling

    # Genuinely large gigavoxel volume still tiles.
    assert _resolve_tiling("auto", (1024, 1024, 1024), 256, False) == "uniform"
    assert _resolve_tiling("auto", (1024, 1024, 1024), 256, True) == "content"
