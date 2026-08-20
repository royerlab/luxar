"""Regression tests for deferred batch floor resolution on denoised data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from luxar._zarr_compat import create_array, open_group
from luxar.cli.gsplat_ops.batch.denoise_workers import resolve_deferred_batch_floor
from luxar.cli.gsplat_ops.batch.submit_slurm import submit_batch_jobs
from luxar.gsplats.batch.manifest import (
    BatchJob,
    BatchManifest,
    load_manifest,
    save_manifest,
)
from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch, generate_floor_sbatch
from luxar.gsplats.batch.status import check_batch_status


def _write_canonical_store(path: Path, data: np.ndarray) -> None:
    group = open_group(path, mode="w")
    array = create_array(group, "data", shape=data.shape, dtype="float32")
    array[...] = data


def test_preprocess_floor_is_resolved_from_denoised_store(tmp_path: Path) -> None:
    source = tmp_path / "raw.zarr"
    source.mkdir()
    denoised_path = tmp_path / "denoised.zarr"
    data = np.stack(
        [
            np.full((1, 4, 4, 4), 10.0, dtype=np.float32),
            np.full((1, 4, 4, 4), 20.0, dtype=np.float32),
        ]
    )
    data[0, 0, 2, 2, 2] = 100.0
    data[1, 0, 2, 2, 2] = 200.0
    _write_canonical_store(denoised_path, data)

    manifest = BatchManifest(
        input_path=str(source),
        output_dir=str(tmp_path),
        n_timepoints=2,
        n_channels=1,
        spatial_shape=[4, 4, 4],
        denoise=True,
        denoise_mode="preprocess",
        denoised_zarr_path=str(denoised_path),
        floor_spec="p10",
        floor_deferred=True,
    )
    save_manifest(manifest, tmp_path)

    resolve_deferred_batch_floor(tmp_path)

    resolved = load_manifest(tmp_path)
    payload = json.loads((tmp_path / "floor_level.json").read_text())
    assert resolved.floor_deferred is False
    assert resolved.floor_level == pytest.approx(10.0)
    assert resolved.fit_args["floor"] == pytest.approx(10.0)
    assert payload == {"level": pytest.approx(10.0), "forward": pytest.approx(10.0)}


def test_on_the_fly_percentile_uses_each_channels_calibrated_h(monkeypatch) -> None:
    from luxar.cli.gsplat_ops.batch import planning
    from luxar.gsplats.fitting import preprocessing

    volumes = {
        0: np.full((4, 4, 4), 10.0, dtype=np.float32),
        1: np.full((4, 4, 4), 20.0, dtype=np.float32),
    }
    seen: list[tuple[float, tuple[float, float], object]] = []

    monkeypatch.setattr(
        planning,
        "_pinned_slice_volume",
        lambda _path, *, channel, **_kwargs: volumes[channel],
    )
    monkeypatch.setattr(
        preprocessing,
        "_sample_volume_for_floor",
        lambda volume, _budget: np.asarray([float(volume.min())]),
    )

    def _resolved(_volume, _floor, *, denoise_h, denoise_params, **_kwargs):
        seen.append((denoise_h, denoise_params["norm_range"], denoise_params["device"]))
        return 9.0 if denoise_h == 0.03 else 18.0

    monkeypatch.setattr(preprocessing, "resolve_volume_floor_denoised", _resolved)

    level, forward = planning.resolve_batch_floor(
        Path("unused.zarr"),
        "p10",
        n_channels=2,
        denoise_h_values={0: 0.03, 1: 0.05},
        denoise_params={"backend": "auto", "device": "cuda"},
    )

    assert level == pytest.approx(9.0)
    assert forward == pytest.approx(9.0)
    assert seen == [
        (0.03, (10.0, 10.0), "cuda"),
        (0.05, (20.0, 20.0), "cuda"),
    ]


def test_deferred_on_the_fly_floor_keeps_discovered_time_axis(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "movie.zarr"
    data = np.stack(
        [
            np.full((4, 4, 4), 20.0, dtype=np.float32),
            np.full((4, 4, 4), 10.0, dtype=np.float32),
        ]
    )
    data[:, 2, 2, 2] = 100.0
    _write_canonical_store(source, data)
    open_group(source, mode="a").attrs["axes"] = ["t", "z", "y", "x"]
    monkeypatch.setattr(
        "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
        lambda volume, **_kwargs: np.asarray(volume, dtype=np.float32),
    )
    manifest = BatchManifest(
        input_path=str(source),
        output_dir=str(tmp_path),
        array_key="data",
        n_timepoints=2,
        n_channels=1,
        channel_shape=[],
        spatial_shape=[4, 4, 4],
        denoise=True,
        denoise_mode="on-the-fly",
        denoise_h=0.04,
        floor_spec="p10",
        floor_deferred=True,
    )
    save_manifest(manifest, tmp_path)

    resolve_deferred_batch_floor(tmp_path)

    resolved = load_manifest(tmp_path)
    assert resolved.floor_level == pytest.approx(10.0)


def test_deferred_on_the_fly_floor_spans_full_store_despite_selection(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "movie.zarr"
    data = np.stack(
        [np.full((4, 4, 4), value, dtype=np.float32) for value in (30, 25, 20, 5)]
    )
    data[:, 2, 2, 2] = 100.0
    _write_canonical_store(source, data)
    open_group(source, mode="a").attrs["axes"] = ["t", "z", "y", "x"]
    monkeypatch.setattr(
        "luxar.gsplats.preprocessing.denoise_pipeline.denoise_volume_array",
        lambda volume, **_kwargs: np.asarray(volume, dtype=np.float32),
    )
    manifest = BatchManifest(
        input_path=str(source),
        output_dir=str(tmp_path),
        array_key="data",
        n_timepoints=2,
        n_channels=1,
        timepoint_indices=[0, 1],
        spatial_shape=[4, 4, 4],
        denoise=True,
        denoise_mode="on-the-fly",
        denoise_h=0.04,
        floor_spec="p10",
        floor_deferred=True,
    )
    save_manifest(manifest, tmp_path)

    resolve_deferred_batch_floor(tmp_path)

    assert load_manifest(tmp_path).floor_level == pytest.approx(5.0)


def test_submit_preserves_floor_job_manifest_updates(
    tmp_path: Path, monkeypatch
) -> None:
    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, job_id: int) -> None:
            self.stdout = f"Submitted batch job {job_id}"

    calls = 0

    def _run(argv, **_kwargs):
        nonlocal calls
        calls += 1
        if argv[-1].endswith("resolve_floor.sbatch"):
            written = load_manifest(tmp_path)
            written.floor_deferred = False
            written.floor_level = 7.0
            written.fit_args["floor"] = 7.0
            save_manifest(written, tmp_path)
        return Result(100 + calls)

    monkeypatch.setattr("subprocess.run", _run)
    manifest = BatchManifest(
        output_dir=str(tmp_path), total_tasks=1, floor_deferred=True
    )

    submit_batch_jobs(
        output_dir=tmp_path,
        manifest=manifest,
        fit_script="fit",
        merge_script="merge",
        preamble="env",
        calibrate_script=None,
        denoise_script=None,
        floor_script="floor",
        preempt_fit_script=None,
        total_tasks=1,
        preempt_partition=None,
    )

    written = load_manifest(tmp_path)
    assert written.floor_deferred is False
    assert written.floor_level == pytest.approx(7.0)
    assert written.fit_args["floor"] == pytest.approx(7.0)
    assert written.floor_job_id == 101


def test_deferred_floor_scripts_feed_one_level_to_every_worker(tmp_path: Path) -> None:
    manifest = BatchManifest(
        input_path="in.zarr",
        output_dir=str(tmp_path),
        mode="uniform",
        n_tiles=2,
        tile_size=32,
        tile_overlap=4,
        floor_spec="p10",
        floor_deferred=True,
        denoise=True,
        denoise_mode="on-the-fly",
        slurm_partition="gpu",
    )

    fit_script = generate_fit_sbatch(manifest, "")
    floor_script = generate_floor_sbatch(manifest, "")

    assert "floor_level.json" in fit_script
    assert "--floor $FLOOR_LEVEL" in fit_script
    assert "--allow-empty-tile" in fit_script
    assert "batch-fit resolve-floor" in floor_script


def test_slurm_fit_waits_for_denoised_floor_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, job_id: int) -> None:
            self.stdout = f"Submitted batch job {job_id}"

    responses = iter(Result(job_id) for job_id in range(101, 106))

    def _run(argv, **_kwargs):
        calls.append(argv)
        return next(responses)

    monkeypatch.setattr("subprocess.run", _run)
    manifest = BatchManifest(
        output_dir=str(tmp_path),
        n_timepoints=1,
        n_channels=1,
        total_tasks=1,
    )

    submit_batch_jobs(
        output_dir=tmp_path,
        manifest=manifest,
        fit_script="fit",
        merge_script="merge",
        preamble="env",
        calibrate_script="calibrate",
        denoise_script="denoise",
        floor_script="floor",
        preempt_fit_script=None,
        total_tasks=1,
        preempt_partition=None,
    )

    assert calls[1][1] == "--dependency=afterok:101"
    assert calls[2][1] == "--dependency=afterok:102"
    assert calls[3][1] == "--dependency=afterok:103"
    assert calls[4][1] == "--dependency=afterok:104"
    assert manifest.floor_job_id == 103
    assert manifest.array_job_id == 104
    assert manifest.merge_job_id == 105


def test_all_empty_floor_slice_is_failed_by_status_and_merge(tmp_path: Path) -> None:
    jobs = [
        BatchJob(
            task_id=index,
            timepoint=0,
            channel=0,
            tile_index=index,
            output_filename=f"tile-{index}.gsplats.zarr",
            estimated_wall_seconds=0.0,
        )
        for index in range(2)
    ]
    manifest = BatchManifest(
        output_dir=str(tmp_path),
        mode="uniform",
        n_tiles=2,
        total_tasks=2,
        fit_args={"floor": 12.5},
        jobs=jobs,
    )
    save_manifest(manifest, tmp_path)
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    for job in jobs:
        (tiles / f"{job.output_filename}.empty").touch()

    status = check_batch_status(tmp_path)

    assert status.completed == 0
    assert status.failed == 2
    with pytest.raises(RuntimeError, match="erased every spatial tile"):
        merge_batch_results(manifest, tmp_path)


def test_all_empty_floor_free_slice_remains_legitimate(tmp_path: Path) -> None:
    job = BatchJob(
        task_id=0,
        timepoint=0,
        channel=0,
        tile_index=0,
        output_filename="tile.gsplats.zarr",
        estimated_wall_seconds=0.0,
    )
    manifest = BatchManifest(
        output_dir=str(tmp_path),
        total_tasks=1,
        fit_args={"floor": "none"},
        jobs=[job],
    )
    save_manifest(manifest, tmp_path)
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / f"{job.output_filename}.empty").touch()

    status = check_batch_status(tmp_path)

    assert status.completed == 1
    assert status.failed == 0
