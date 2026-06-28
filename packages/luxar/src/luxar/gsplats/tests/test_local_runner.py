# test_local_runner.py
"""Tests for the local (non-Slurm) multi-GPU batch runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.batch.local_runner import (
    _finalize_output,
    _task_voxels,
    build_device_assignment,
)
from luxar.gsplats.batch.manifest import BatchManifest


# ---------------------------------------------------------------------------
# Device assignment (weighted round-robin)
# ---------------------------------------------------------------------------


def test_assignment_cpu_sentinel() -> None:
    assert build_device_assignment([0, 1, 2], {-1: 4}) == {0: -1, 1: -1, 2: -1}
    assert build_device_assignment([0, 1], {}) == {0: -1, 1: -1}


def test_assignment_weighted_by_worker_count() -> None:
    # gpu0 has 4 workers, gpu1 has 1 -> first 5 tasks fill exactly each capacity.
    assign = build_device_assignment(list(range(10)), {0: 4, 1: 1})
    first5 = [assign[i] for i in range(5)]
    assert first5.count(0) == 4 and first5.count(1) == 1
    # Over 10 tasks the ratio holds ~4:1.
    allv = [assign[i] for i in range(10)]
    assert allv.count(0) == 8 and allv.count(1) == 2


def test_assignment_single_gpu() -> None:
    assert build_device_assignment([0, 1, 2], {3: 2}) == {0: 3, 1: 3, 2: 3}


# ---------------------------------------------------------------------------
# Task-voxel proxy
# ---------------------------------------------------------------------------


def test_task_voxels_uniform_caps_at_total() -> None:
    m = BatchManifest(mode="uniform", tile_size=64, spatial_shape=(16, 16, 16))
    assert _task_voxels(m) == 16**3  # tile (64^3) capped at total (16^3)
    m2 = BatchManifest(mode="uniform", tile_size=8, spatial_shape=(64, 64, 64))
    assert _task_voxels(m2) == 8**3


def test_task_voxels_fallback_total_when_no_tile() -> None:
    m = BatchManifest(mode="uniform", tile_size=0, spatial_shape=(10, 10, 10))
    assert _task_voxels(m) == 1000


# ---------------------------------------------------------------------------
# Output finalization (atomic rename + empty marker)
# ---------------------------------------------------------------------------


def test_finalize_renames_tmp(tmp_path: Path) -> None:
    out = tmp_path / "t00_c00_tile000.gsplats.zarr"
    tmp = Path(str(out) + ".tmp")
    tmp.mkdir()
    (tmp / "data").write_text("x")
    ok, empty = _finalize_output(out)
    assert ok and not empty
    assert out.exists() and not tmp.exists()
    assert (out / "data").read_text() == "x"


def test_finalize_empty_marker(tmp_path: Path) -> None:
    out = tmp_path / "t00_c00_box003.gsplats.zarr"
    Path(str(out) + ".tmp").mkdir()
    Path(str(out) + ".tmp.empty").touch()
    ok, empty = _finalize_output(out)
    assert ok and empty
    assert Path(str(out) + ".empty").exists()
    assert not out.exists()
    assert not Path(str(out) + ".tmp").exists()


def test_finalize_missing_is_not_ok(tmp_path: Path) -> None:
    out = tmp_path / "missing.gsplats.zarr"
    ok, empty = _finalize_output(out)
    assert not ok and not empty


# ---------------------------------------------------------------------------
# CPU end-to-end (real `luxar gsplat fit` subprocesses)
# ---------------------------------------------------------------------------


def _make_4d_zarr(path: Path, n_t: int = 2) -> None:
    import zarr

    rng = np.random.default_rng(0)
    V = np.zeros((n_t, 16, 16, 16), np.float32)
    zz, yy, xx = np.mgrid[0:16, 0:16, 0:16]
    for t in range(n_t):
        for _ in range(6):
            cz, cy, cx = rng.integers(3, 13, 3)
            V[t] += np.exp(
                -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)
            ).astype(np.float32)
    V = np.clip(V, 0, 1)
    z = zarr.open_array(
        str(path), mode="w", shape=V.shape, chunks=(1, 16, 16, 16), dtype="f4"
    )
    z[:] = V


@pytest.mark.slow
def test_run_batch_local_cpu_end_to_end(tmp_path: Path) -> None:
    """Plan a tiny 2-timepoint uniform batch and run it locally on CPU."""
    from luxar.cli.gsplat_ops.batch_planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )
    from luxar.gsplats.batch.local_runner import run_batch_local
    from luxar.gsplats.gsplat_data import GSplatData

    src = tmp_path / "vol.zarr"
    _make_4d_zarr(src, n_t=2)
    out = tmp_path / "out"

    plan = plan_batch(
        input_path=src,
        output_dir=out,
        tiling="uniform",
        tile_size=64,  # > 16 -> single tile, no profile needed
        tile_overlap=0,
        axes_list=["t", "z", "y", "x"],
        array_key=None,
        timepoints_slice=None,
        channels_slice=None,
        fit=FitConfig(preset="draft", seeds="150", iters=25),
        denoise=DenoiseConfig(),
        content=ContentKnobs(),
        merge=MergeConfig(),
        merge_recipe_args={},
    )
    assert plan.manifest.total_tasks == 2  # 2 timepoints × 1 channel × 1 tile

    final = run_batch_local(
        plan.manifest, out, gpus="cpu", jobs_per_gpu=2, verbose=False
    )
    assert final.exists()
    g = GSplatData.load(final)
    assert g.n_splats > 0
    # Both per-timepoint tile outputs were written.
    tiles = sorted((out / "tiles").glob("*.gsplats.zarr"))
    assert len(tiles) == 2


@pytest.mark.slow
def test_run_batch_local_resume_skips_existing(tmp_path: Path) -> None:
    """A second run with one output deleted re-fits only the missing task."""
    from luxar.cli.gsplat_ops.batch_planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )
    from luxar.gsplats.batch.local_runner import run_batch_local

    src = tmp_path / "vol.zarr"
    _make_4d_zarr(src, n_t=2)
    out = tmp_path / "out"
    common = dict(
        input_path=src,
        output_dir=out,
        tiling="uniform",
        tile_size=64,
        tile_overlap=0,
        axes_list=["t", "z", "y", "x"],
        array_key=None,
        timepoints_slice=None,
        channels_slice=None,
        fit=FitConfig(preset="draft", seeds="150", iters=25),
        denoise=DenoiseConfig(),
        content=ContentKnobs(),
        merge=MergeConfig(),
        merge_recipe_args={},
    )
    plan = plan_batch(**common)
    run_batch_local(plan.manifest, out, gpus="cpu", jobs_per_gpu=2, verbose=False)

    tiles = sorted((out / "tiles").glob("*.gsplats.zarr"))
    assert len(tiles) == 2
    # Delete one tile + the merged result; re-run should regenerate just that one.
    import shutil

    shutil.rmtree(tiles[0])
    mtime_kept = tiles[1].stat().st_mtime
    plan2 = plan_batch(**common)
    run_batch_local(plan2.manifest, out, gpus="cpu", jobs_per_gpu=2, verbose=False)
    assert tiles[0].exists()  # regenerated
    assert tiles[1].stat().st_mtime == mtime_kept  # untouched (skipped)
