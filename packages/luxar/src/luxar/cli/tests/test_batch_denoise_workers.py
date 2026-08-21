"""Regression tests for batch preprocess-denoise volume selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from luxar._zarr_compat import create_array, open_group
from luxar.gsplats.batch.manifest import BatchManifest, save_manifest


def _write_noncanonical_source(path: Path) -> np.ndarray:
    data = np.zeros((2, 2, 2, 3, 4), dtype=np.float32)
    for channel in range(2):
        for timepoint in range(2):
            data[channel, timepoint] = 10 * channel + timepoint
            data[channel, timepoint, timepoint, channel, channel] += 1
    root = open_group(path, mode="w")
    create_array(root, "data", data=data)
    return data


def test_preprocess_reads_source_axes_then_fits_canonical_store(
    tmp_path: Path, monkeypatch
) -> None:
    from luxar.cli.gsplat_ops.batch.denoise_workers import (
        run_batch_denoise_preprocess_cmd,
    )
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch
    from luxar.gsplats.preprocessing import denoise_pipeline

    source = tmp_path / "source.zarr"
    data = _write_noncanonical_source(source)
    output = tmp_path / "batch"
    manifest = BatchManifest(
        input_path=str(source),
        output_dir=str(output),
        array_key="data",
        axes="channel,time,z,y,x",
        n_timepoints=2,
        n_channels=2,
        spatial_shape=(2, 3, 4),
        n_tiles=1,
        total_tasks=4,
        denoise=True,
        denoise_mode="preprocess",
        denoised_zarr_path=str(output / "denoised.zarr"),
    )
    save_manifest(manifest, output)
    (output / "denoise_h_values.json").write_text(json.dumps({"1": 0.04}))
    monkeypatch.setattr(
        denoise_pipeline, "denoise_volume_array", lambda volume, **_: volume
    )

    run_batch_denoise_preprocess_cmd(output, task_id=1)

    denoised = open_group(output / "denoised.zarr", mode="r")["data"]
    assert denoised.shape == (2, 2, 2, 3, 4)
    np.testing.assert_array_equal(denoised[0, 1], data[1, 0])

    script = generate_fit_sbatch(manifest, "")
    assert f"luxar gsplat fit {output / 'denoised.zarr'}" in script
    assert "--array-key data" in script
    assert "--timepoint $T_IDX" in script
    assert "--channel $C_IDX" in script
    assert "--axes t,c,z,y,x" in script


def test_preprocess_reads_explicit_singleton_timepoint_and_channel(
    tmp_path: Path, monkeypatch
) -> None:
    from luxar.cli.gsplat_ops.batch.denoise_workers import (
        run_batch_denoise_preprocess_cmd,
    )
    from luxar.gsplats.preprocessing import denoise_pipeline

    source = tmp_path / "source.zarr"
    data = np.zeros((3, 4, 2, 3, 4), dtype=np.float32)
    for channel in range(3):
        for timepoint in range(4):
            data[channel, timepoint] = 10 * channel + timepoint
    root = open_group(source, mode="w")
    create_array(root, "data", data=data)

    output = tmp_path / "batch"
    manifest = BatchManifest(
        input_path=str(source),
        output_dir=str(output),
        array_key="data",
        axes="channel,time,z,y,x",
        n_timepoints=1,
        n_channels=1,
        timepoint_indices=[3],
        channel_indices=[2],
        spatial_shape=(2, 3, 4),
        n_tiles=1,
        total_tasks=1,
        denoise=True,
        denoise_mode="preprocess",
        denoised_zarr_path=str(output / "denoised.zarr"),
    )
    save_manifest(manifest, output)
    (output / "denoise_h_values.json").write_text(json.dumps({"2": 0.04}))
    monkeypatch.setattr(
        denoise_pipeline, "denoise_volume_array", lambda volume, **_: volume
    )

    run_batch_denoise_preprocess_cmd(output, task_id=0)

    denoised = open_group(output / "denoised.zarr", mode="r")["data"]
    np.testing.assert_array_equal(denoised[0, 0], data[2, 3])


def test_preprocess_fit_uses_explicit_axes_for_2d_store(tmp_path: Path) -> None:
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

    manifest = BatchManifest(
        input_path="source.zarr",
        output_dir=str(tmp_path),
        axes="channel,time,x,y",
        n_timepoints=3,
        n_channels=2,
        spatial_shape=(4, 5),
        n_tiles=1,
        total_tasks=6,
        denoise=True,
        denoise_mode="preprocess",
        denoised_zarr_path=str(tmp_path / "denoised.zarr"),
    )

    script = generate_fit_sbatch(manifest, "")

    assert "--axes t,c,x,y" in script
    assert "--array-key data" in script

    manifest.axes = None
    script = generate_fit_sbatch(manifest, "")

    assert "--axes t,c,y,x" in script


def test_preprocess_fit_without_source_axes_reads_assigned_2d_slice(
    tmp_path: Path, monkeypatch
) -> None:
    from luxar.cli.gsplat_ops.batch.denoise_workers import (
        run_batch_denoise_preprocess_cmd,
    )
    from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch
    from luxar.gsplats.preprocessing import denoise_pipeline
    from luxar.io.volume import load_volume

    source = tmp_path / "source.zarr"
    data = np.zeros((3, 2, 1, 4, 5), dtype=np.float32)
    for timepoint in range(3):
        for channel in range(2):
            data[timepoint, channel] = 10 * timepoint + channel
    root = open_group(source, mode="w")
    create_array(root, "data", data=data)

    output = tmp_path / "batch"
    manifest = BatchManifest(
        input_path=str(source),
        output_dir=str(output),
        array_key="data",
        n_timepoints=3,
        n_channels=2,
        spatial_shape=(4, 5),
        n_tiles=1,
        total_tasks=6,
        denoise=True,
        denoise_mode="preprocess",
        denoised_zarr_path=str(output / "denoised.zarr"),
    )
    save_manifest(manifest, output)
    (output / "denoise_h_values.json").write_text(json.dumps({"1": 0.04}))
    monkeypatch.setattr(
        denoise_pipeline, "denoise_volume_array", lambda volume, **_: volume
    )

    run_batch_denoise_preprocess_cmd(output, task_id=5)

    script = generate_fit_sbatch(manifest, "")
    assert "--axes t,c,y,x" in script
    loaded = load_volume(
        output / "denoised.zarr",
        channel=1,
        timepoint=2,
        array_key="data",
        axes="t,c,y,x",
    )
    np.testing.assert_array_equal(loaded, data[2, 1, 0])


def test_denoise_calibration_command_reads_source_axes(
    tmp_path: Path, monkeypatch
) -> None:
    from luxar.cli.gsplat_ops.batch.denoise_workers import (
        run_batch_denoise_calibrate_cmd,
    )
    from luxar.gsplats.preprocessing import calibration

    source = tmp_path / "source.zarr"
    data = _write_noncanonical_source(source)
    output = tmp_path / "batch"
    manifest = BatchManifest(
        input_path=str(source),
        output_dir=str(output),
        array_key="data",
        axes="channel,time,z,y,x",
        n_timepoints=2,
        n_channels=2,
        channel_indices=[1],
        timepoint_indices=[0],
        denoise=True,
        calibration_samples=1,
    )
    save_manifest(manifest, output)
    seen: list[np.ndarray] = []

    def _capture(volume, **_):
        seen.append(volume.cpu().numpy())
        return 0.125

    monkeypatch.setattr(calibration, "calibrate_nlm_h", _capture)

    run_batch_denoise_calibrate_cmd(output)

    assert json.loads((output / "denoise_h_values.json").read_text()) == {"1": 0.125}
    expected = data[1, 0]
    expected = (expected - expected.min()) / (expected.max() - expected.min())
    np.testing.assert_array_equal(seen, [expected])
