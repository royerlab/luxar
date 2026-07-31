# test_local_runner.py
"""Tests for the local (non-Slurm) multi-GPU batch runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.batch.local_runner import (
    _finalize_output,
    _staging_path,
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
    staging = _staging_path(out, 12345)
    staging.mkdir()
    (staging / "data").write_text("x")
    ok, empty = _finalize_output(out, staging)
    assert ok and not empty
    assert out.exists() and not staging.exists()
    assert (out / "data").read_text() == "x"


def test_finalize_empty_marker(tmp_path: Path) -> None:
    out = tmp_path / "t00_c00_box003.gsplats.zarr"
    staging = _staging_path(out, 12345)
    staging.mkdir()
    Path(str(staging) + ".empty").touch()
    ok, empty = _finalize_output(out, staging)
    assert ok and empty
    assert Path(str(out) + ".empty").exists()
    assert not out.exists()
    assert not staging.exists()


def test_finalize_missing_is_not_ok(tmp_path: Path) -> None:
    out = tmp_path / "missing.gsplats.zarr"
    ok, empty = _finalize_output(out, _staging_path(out, 12345))
    assert not ok and not empty


# ---------------------------------------------------------------------------
# Per-attempt staging isolation (issue #679)
# ---------------------------------------------------------------------------


def test_staging_path_is_per_token(tmp_path: Path) -> None:
    """Two invocations (distinct tokens) get distinct staging dirs for one out."""
    out = tmp_path / "t00_c00_tile000.gsplats.zarr"
    s_a = _staging_path(out, 111)
    s_b = _staging_path(out, 222)
    assert s_a != s_b
    # Both derive from the final output path and carry their token.
    assert str(s_a).startswith(str(out) + ".tmp.")
    assert str(s_a).endswith("111") and str(s_b).endswith("222")


def test_finalize_does_not_clobber_when_out_already_claimed(tmp_path: Path) -> None:
    """A losing attempt drops its own staging without touching the winner's out.

    Simulates two concurrent invocations on the same ``out``: attempt A already
    promoted its staging to the final ``out``; attempt B finalizes afterwards and
    must NOT overwrite A's output — it cleans up its own (independent) staging.
    """
    out = tmp_path / "t00_c00_tile000.gsplats.zarr"

    # Attempt A won: final output exists with A's content.
    out.mkdir()
    (out / "data").write_text("winner")

    # Attempt B has its own populated, independently-named staging.
    staging_b = _staging_path(out, 999)
    staging_b.mkdir()
    (staging_b / "data").write_text("loser")

    ok, empty = _finalize_output(out, staging_b)
    assert ok and not empty
    assert not staging_b.exists()  # B's duplicate cleaned up
    assert (out / "data").read_text() == "winner"  # A's output untouched


def test_finalize_toctou_replace_race_takes_loser_path(
    tmp_path: Path, monkeypatch
) -> None:
    """A rename that races another attempt takes the loser path, not a crash.

    A concurrent attempt claims ``out`` between finalize's ``out.exists()`` check
    and the rename, so ``os.replace`` raises OSError (refuses to overwrite a
    non-empty dir) while ``out`` now exists. Finalize must swallow it, drop the
    duplicate staging, and report completed-by-other — never propagate.
    """
    import luxar.gsplats.batch.local_runner as lr

    out = tmp_path / "t00_c00_tile000.gsplats.zarr"
    staging = _staging_path(out, "host-1")
    staging.mkdir()
    (staging / "data").write_text("x")

    def _boom(_src, _dst):  # type: ignore[no-untyped-def]
        # Simulate the race: the winner materialized `out` just before rename.
        out.mkdir()
        (out / "data").write_text("winner")
        raise OSError("Directory not empty")

    monkeypatch.setattr(lr.os, "replace", _boom)
    ok, empty = _finalize_output(out, staging)
    assert ok and not empty
    assert not staging.exists()  # duplicate cleaned up, no exception propagated
    assert (out / "data").read_text() == "winner"  # winner untouched


def test_finalize_genuine_oserror_keeps_staging_and_reports_not_ok(
    tmp_path: Path, monkeypatch
) -> None:
    """A non-race ``os.replace`` OSError must NOT destroy the fit.

    If the rename fails while ``out`` is still absent (staging vanished under a
    concurrent ``validate --fix``, or EACCES/EIO), finalize must report not-ok
    and LEAVE staging on disk for inspection — mirroring the Slurm "mv failed and
    output missing" → rc 1 branch, not silently deleting the fitted result.
    """
    import luxar.gsplats.batch.local_runner as lr

    out = tmp_path / "t00_c00_tile000.gsplats.zarr"
    staging = _staging_path(out, "host-1")
    staging.mkdir()
    (staging / "data").write_text("x")

    def _boom(_src, _dst):  # type: ignore[no-untyped-def]
        raise OSError("EIO")  # out stays absent

    monkeypatch.setattr(lr.os, "replace", _boom)
    ok, empty = _finalize_output(out, staging)
    assert not ok and not empty
    assert staging.exists()  # fitted result preserved for inspection
    assert not out.exists()


class _StopRun(Exception):
    """Abort a driven ``run_batch_local`` once argv has been captured."""


def test_argv_staging_is_per_invocation(tmp_path: Path, monkeypatch) -> None:
    """The staging path threaded into the worker argv is pid-scoped.

    Two ``run_batch_local`` processes on the same ``output_dir`` must build
    distinct staging paths (keyed by host + ``os.getpid()``), so neither can
    delete or interleave the other's in-progress store. We capture the output
    path passed to ``build_task_fit_argv`` under two different faked pids by
    driving the real runner and stubbing only the task pool.
    """
    import luxar.gsplats.batch.local_runner as lr
    from luxar.cli.gsplat_ops.batch_planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    src = tmp_path / "vol.zarr"
    _make_4d_zarr(src, n_t=1)
    out_dir = tmp_path / "out"

    manifest = plan_batch(
        input_path=src,
        output_dir=out_dir,
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
    ).manifest
    task0 = manifest.jobs[0].task_id

    captured: list[Path] = []
    real_build = lr.build_task_fit_argv

    def _spy(manifest_, job_, out_path, **kw):  # type: ignore[no-untyped-def]
        captured.append(Path(out_path))
        return real_build(manifest_, job_, out_path, **kw)

    # Stub the pool to invoke the runner's real `_argv` for task 0 (so the spy
    # records the actual staging path), then abort before any fit/merge runs.
    def _fake_pool(task_ids, *, argv_builder, **kw):  # type: ignore[no-untyped-def]
        argv_builder(task0)
        raise _StopRun()

    monkeypatch.setattr(lr, "build_task_fit_argv", _spy)
    monkeypatch.setattr(lr, "run_task_pool", _fake_pool)
    # Deterministic host prefix so we can assert the exact host+pid token.
    monkeypatch.setattr(lr.socket, "gethostname", lambda: "testhost")

    def _capture_for_pid(pid: int) -> Path:
        captured.clear()
        monkeypatch.setattr(lr.os, "getpid", lambda: pid)
        with pytest.raises(_StopRun):
            lr.run_batch_local(manifest, out_dir, gpus="cpu", verbose=False)
        return captured[0]

    s1 = _capture_for_pid(4001)
    s2 = _capture_for_pid(4002)
    expected_out = out_dir / "tiles" / manifest.jobs[0].output_filename
    # Token is host+pid, so the staging path carries both; the pid remains a
    # substring of the path (invocation-scoped) and two pids never collide.
    assert s1 == _staging_path(expected_out, "testhost-4001")
    assert s2 == _staging_path(expected_out, "testhost-4002")
    assert "4001" in str(s1) and "4002" in str(s2)
    assert s1 != s2  # distinct invocations never share a staging dir


def test_no_resume_removes_stale_final_output_before_refit(
    tmp_path: Path, monkeypatch
) -> None:
    """resume=False must delete a prior run's final output before refitting.

    Otherwise the fresh refit lands in staging and the finalize loser-guard
    (``out`` already exists) silently discards it, keeping the STALE tile — the
    regression introduced by the per-attempt loser-guard. ``_argv`` must remove
    the stale final output up front so overwrite-on-refit is restored.
    """
    import luxar.gsplats.batch.local_runner as lr
    from luxar.cli.gsplat_ops.batch_planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    src = tmp_path / "vol.zarr"
    _make_4d_zarr(src, n_t=1)
    out_dir = tmp_path / "out"

    manifest = plan_batch(
        input_path=src,
        output_dir=out_dir,
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
    ).manifest
    task0 = manifest.jobs[0].task_id

    # A prior completed run left a STALE final output on disk.
    stale_out = out_dir / "tiles" / manifest.jobs[0].output_filename
    stale_out.mkdir(parents=True)
    (stale_out / "STALE").write_text("old")

    # Drive the real `_argv` for task 0 (which, under resume=False, must nuke the
    # stale final output), then abort before any fit/merge runs.
    def _fake_pool(task_ids, *, argv_builder, **kw):  # type: ignore[no-untyped-def]
        argv_builder(task0)
        raise _StopRun()

    monkeypatch.setattr(lr, "run_task_pool", _fake_pool)

    with pytest.raises(_StopRun):
        lr.run_batch_local(manifest, out_dir, gpus="cpu", resume=False, verbose=False)

    # The stale output was removed up front, so a subsequent refit+promote lands
    # the fresh result instead of being discarded by the loser-guard.
    assert not stale_out.exists()


def test_no_resume_forces_merge_rebuild(tmp_path: Path, monkeypatch) -> None:
    """resume=False must force the merge, not serve a stale merged/final.

    The merge short-circuits on a pre-existing ``merged/final.gsplats.zarr``. A
    ``--no-resume`` rerun refits every tile, so the merge MUST be rebuilt over
    the fresh tiles (``force=True``) instead of returning the old artifact.
    """
    import luxar.gsplats.batch.local_runner as lr
    from luxar.cli.gsplat_ops.batch_planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
        plan_batch,
    )

    src = tmp_path / "vol.zarr"
    _make_4d_zarr(src, n_t=1)
    out_dir = tmp_path / "out"

    manifest = plan_batch(
        input_path=src,
        output_dir=out_dir,
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
    ).manifest

    captured: dict[str, object] = {}

    def _fake_merge(**kw):  # type: ignore[no-untyped-def]
        captured["force"] = kw["force"]
        return out_dir / "merged" / "final.gsplats.zarr"

    # No tiles fitted (empty results) — we only assert how the merge is invoked.
    monkeypatch.setattr(lr, "run_task_pool", lambda *a, **k: [])
    monkeypatch.setattr(lr, "merge_batch_results", _fake_merge)

    lr.run_batch_local(manifest, out_dir, gpus="cpu", resume=False, verbose=False)
    assert captured["force"] is True

    # Sanity: with resume=True and no explicit force, the merge is NOT forced.
    captured.clear()
    lr.run_batch_local(manifest, out_dir, gpus="cpu", resume=True, verbose=False)
    assert captured["force"] is False


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
    # The manifest is persisted so `batch-fit status`/`validate` work locally.
    assert (out / "manifest.json").exists()
    from luxar.gsplats.batch.status import check_batch_status

    status = check_batch_status(out)
    assert status.completed == 2


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
