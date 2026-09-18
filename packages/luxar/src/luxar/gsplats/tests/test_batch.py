"""Tests for batch fitting infrastructure (manifest, env, slurm, time, merge)."""

from __future__ import annotations

import builtins
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from luxar._zarr_compat import create_array


class TestSelectPlanTimepoints:
    """The shared content-plan timepoint sampler (`--plan-timepoint`/`--plan-samples`)."""

    def test_pinned_timepoint_scans_only_that_one(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import _select_plan_timepoints

        assert _select_plan_timepoints(
            [0, 5, 10, 15], plan_timepoint=10, plan_samples=16
        ) == [10]

    def test_pinned_timepoint_out_of_range_raises(self) -> None:
        import typer

        from luxar.cli.gsplat_ops.batch.planning import _select_plan_timepoints

        with pytest.raises(typer.BadParameter):
            _select_plan_timepoints([0, 5, 10], plan_timepoint=7, plan_samples=16)

    def test_fewer_timepoints_than_samples_returns_all(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import _select_plan_timepoints

        assert _select_plan_timepoints(
            [0, 1, 2, 3], plan_timepoint=None, plan_samples=16
        ) == [
            0,
            1,
            2,
            3,
        ]

    def test_caps_to_evenly_spaced_sample_with_endpoints(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import _select_plan_timepoints

        out = _select_plan_timepoints(
            list(range(100)), plan_timepoint=None, plan_samples=5
        )
        assert len(out) == 5
        assert out[0] == 0 and out[-1] == 99  # endpoints always included
        assert out == sorted(set(out))  # strictly increasing, deduplicated

    def test_real_indices_preserved_in_sample(self) -> None:
        from luxar.cli.gsplat_ops.batch.planning import _select_plan_timepoints

        # Sliced selection (e.g. --timepoints '::10') keeps REAL dataset indices.
        real = [0, 10, 20, 30, 40, 50, 60, 70]
        out = _select_plan_timepoints(real, plan_timepoint=None, plan_samples=3)
        assert len(out) == 3
        assert set(out).issubset(set(real))
        assert out[0] == 0 and out[-1] == 70

    def test_zero_samples_raises(self) -> None:
        import typer

        from luxar.cli.gsplat_ops.batch.planning import _select_plan_timepoints

        with pytest.raises(typer.BadParameter):
            _select_plan_timepoints([0, 1, 2], plan_timepoint=None, plan_samples=0)


# ====================================================================
# Manifest tests
# ====================================================================


class TestManifest:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from luxar.gsplats.batch.manifest import (
            BatchJob,
            BatchManifest,
            load_manifest,
            save_manifest,
        )

        manifest = BatchManifest(
            input_path="/data/test.zarr",
            output_dir=str(tmp_path),
            n_timepoints=5,
            n_channels=2,
            channel_axes=["camera", "channel"],
            channel_shape=(1, 2),
            spatial_shape=(128, 256, 256),
            tile_size=128,
            n_tiles=4,
            total_tasks=40,
            preset="standard",
            slurm_partition="gpu",
            merge_recipe="substitutive",
            merge_recipe_args={"compression-factor": "4", "levels": "3"},
            grid_scale=[4.0, 1.0, 1.0],
        )
        manifest.jobs = [
            BatchJob(
                task_id=0,
                timepoint=0,
                channel=0,
                tile_index=0,
                output_filename="t00_c00_tile000.gsplats.zarr",
                estimated_wall_seconds=300.0,
                channel_coords=(0, 0),
            )
        ]

        save_manifest(manifest, tmp_path)
        loaded = load_manifest(tmp_path)

        assert loaded.n_timepoints == 5
        assert loaded.n_channels == 2
        assert loaded.channel_axes == ["camera", "channel"]
        assert loaded.channel_shape == (1, 2)
        assert loaded.spatial_shape == (128, 256, 256)
        assert len(loaded.jobs) == 1
        assert loaded.jobs[0].task_id == 0
        assert loaded.jobs[0].channel_coords == (0, 0)
        assert loaded.merge_recipe == "substitutive"
        assert loaded.merge_recipe_args == {"compression-factor": "4", "levels": "3"}
        # The frame the fit tasks emit in (#1587) survives the JSON round-trip as
        # floats — the merge's split planes are built from it.
        assert loaded.grid_scale == [4.0, 1.0, 1.0]

    def test_decode_task_id(self) -> None:
        from luxar.gsplats.batch.manifest import BatchManifest, decode_task_id

        manifest = BatchManifest(n_channels=3, n_tiles=4)

        # task_id = t * (3 * 4) + c * 4 + k
        assert decode_task_id(0, manifest) == (0, 0, 0)
        assert decode_task_id(1, manifest) == (0, 0, 1)
        assert decode_task_id(4, manifest) == (0, 1, 0)
        assert decode_task_id(12, manifest) == (1, 0, 0)
        assert decode_task_id(17, manifest) == (1, 1, 1)

    def test_load_with_unknown_fields(self, tmp_path: Path) -> None:
        """Manifest should load even if JSON has extra fields from a newer version."""
        import json

        from luxar.gsplats.batch.manifest import load_manifest

        manifest_data = {
            "version": 1,
            "input_path": "/data/test.zarr",
            "output_dir": str(tmp_path),
            "n_timepoints": 1,
            "n_channels": 1,
            "spatial_shape": [64, 64, 64],
            "tile_size": 64,
            "n_tiles": 1,
            "total_tasks": 1,
            "slurm_partition": "gpu",
            "unknown_future_field": "should be ignored",
            "another_new_field": 42,
            "jobs": [
                {
                    "task_id": 0,
                    "timepoint": 0,
                    "channel": 0,
                    "tile_index": 0,
                    "output_filename": "t00_c00_tile000.gsplats.zarr",
                    "estimated_wall_seconds": 60.0,
                    "new_job_field": "ignored",
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_data))

        loaded = load_manifest(tmp_path)
        assert loaded.n_timepoints == 1
        assert len(loaded.jobs) == 1
        assert loaded.jobs[0].task_id == 0

    def test_output_filename(self) -> None:
        from luxar.gsplats.batch.manifest import output_filename

        assert output_filename(0, 0, 0) == "t00_c00_tile000.gsplats.zarr"
        assert output_filename(5, 2, 15) == "t05_c02_tile015.gsplats.zarr"

    def test_sliced_batch_jobs_use_real_indices_in_filenames(self) -> None:
        from luxar.gsplats.batch.manifest import BatchJob, output_filename
        from luxar.io.volume import decode_flat_channel_index

        t_indices = [0, 72]
        c_indices = [1, 5]
        n_t = len(t_indices)
        n_c = len(c_indices)
        n_tiles = 2
        jobs = []
        for task_id in range(n_t * n_c * n_tiles):
            t_seq = task_id // (n_c * n_tiles)
            r = task_id % (n_c * n_tiles)
            c_seq = r // n_tiles
            k = r % n_tiles
            t_real = t_indices[t_seq]
            c_real = c_indices[c_seq]
            jobs.append(
                BatchJob(
                    task_id=task_id,
                    timepoint=t_real,
                    channel=c_real,
                    tile_index=k,
                    output_filename=output_filename(
                        t_real,
                        c_real,
                        k,
                        max(t_indices) + 1,
                        max(c_indices) + 1,
                        n_tiles,
                    ),
                    estimated_wall_seconds=1.0,
                    channel_coords=decode_flat_channel_index(c_real, (2, 3)),
                )
            )

        assert jobs[0].output_filename == "t00_c01_tile000.gsplats.zarr"
        assert jobs[-1].output_filename == "t72_c05_tile001.gsplats.zarr"
        assert jobs[-1].channel_coords == (1, 2)


# ====================================================================
# Environment capture tests
# ====================================================================


class TestEnvCapture:
    def test_partition_node_resources_parse_scheduler_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="16+ 128000+\n64 512000\n", stderr=""
        )
        calls: list[list[str]] = []

        def fake_run(args, **_kwargs):
            calls.append(args)
            return result

        monkeypatch.setattr(env_capture.subprocess, "run", fake_run)

        assert env_capture.get_partition_node_resources("gpu") == [
            (16, 128000),
            (64, 512000),
        ]
        assert "--exact" in calls[0]

    def test_capture_conda(self) -> None:
        from luxar.gsplats.batch.env_capture import capture_environment

        with patch.dict(os.environ, {"CONDA_PREFIX": "/opt/conda/envs/myenv"}):
            env = capture_environment()
            assert env.conda_prefix == "/opt/conda/envs/myenv"

    def test_capture_env_vars(self) -> None:
        from luxar.gsplats.batch.env_capture import capture_environment

        with patch.dict(
            os.environ,
            {"CUDA_HOME": "/usr/local/cuda", "TORCH_HOME": "/tmp/torch"},
            clear=False,
        ):
            env = capture_environment()
            assert env.env_vars.get("CUDA_HOME") == "/usr/local/cuda"
            assert env.env_vars.get("TORCH_HOME") == "/tmp/torch"

    @pytest.mark.parametrize(
        ("probe_name", "args", "expected"),
        [
            ("get_slurm_scheduler_info", (), "unknown"),
            ("is_slurm_mps_available", (), False),
            ("_partition_has_gpu", ("gpu",), False),
            ("detect_preemptible_gpu_partition", (), None),
            ("validate_partition_access", ("gpu",), False),
            ("get_partition_node_resources", ("gpu",), []),
        ],
    )
    def test_broken_slurm_probe_warns_once_and_keeps_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_name: str,
        args: tuple[str, ...],
        expected: object,
    ) -> None:
        from luxar.gsplats.batch import env_capture

        monkeypatch.setattr(
            env_capture.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())
        probe = getattr(env_capture, probe_name)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = probe(*args)
            second = probe(*args)

        if probe_name == "get_slurm_scheduler_info":
            assert first["scheduler_type"] == expected
            assert second["scheduler_type"] == expected
        else:
            assert first == expected
            assert second == expected
        assert len(caught) == 1
        assert probe_name in str(caught[0].message)
        assert "RuntimeError: boom" in str(caught[0].message)

    @pytest.mark.parametrize(
        ("probe_name", "args", "expected"),
        [
            ("get_slurm_scheduler_info", (), "unknown"),
            ("is_slurm_mps_available", (), False),
            ("_partition_has_gpu", ("gpu",), False),
            ("detect_preemptible_gpu_partition", (), None),
            ("validate_partition_access", ("gpu",), False),
            ("get_partition_node_resources", ("gpu",), []),
        ],
    )
    @pytest.mark.parametrize(
        "probe_error",
        [
            FileNotFoundError("missing Slurm command"),
            subprocess.TimeoutExpired("Slurm command", 5),
        ],
        ids=["missing", "timeout"],
    )
    def test_unavailable_slurm_probe_stays_quiet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        probe_name: str,
        args: tuple[str, ...],
        expected: object,
        probe_error: Exception,
    ) -> None:
        from luxar.gsplats.batch import env_capture

        monkeypatch.setattr(
            env_capture.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(probe_error),
        )
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())
        probe = getattr(env_capture, probe_name)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = probe(*args)

        if probe_name == "get_slurm_scheduler_info":
            assert result["scheduler_type"] == expected
        else:
            assert result == expected
        assert not caught

    def test_preemptible_partition_warns_when_sinfo_probe_breaks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        partition_listing = SimpleNamespace(
            returncode=0,
            stdout="PartitionName=preempt-gpu State=UP\n",
        )

        def run(command: list[str], **_kwargs: object) -> object:
            if command[0] == "scontrol":
                return partition_listing
            raise RuntimeError("sinfo broke")

        monkeypatch.setattr(env_capture.subprocess, "run", run)
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())

        with pytest.warns(
            RuntimeWarning, match="_partition_has_gpu.*RuntimeError: sinfo broke"
        ) as caught:
            assert env_capture.detect_preemptible_gpu_partition() is None
        assert Path(caught[0].filename) == Path(__file__)

    def test_malformed_cuda_build_info_warns_and_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        info_path = tmp_path / "cuda_build_info.json"
        info_path.write_text("{not json")
        monkeypatch.setattr(env_capture, "_cuda_build_info_path", lambda: info_path)
        monkeypatch.setattr(env_capture, "_detect_loaded_modules", lambda: [])
        monkeypatch.setattr(env_capture, "version", lambda _name: "test")
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())

        with pytest.warns(
            RuntimeWarning, match="read_cuda_build_info.*JSONDecodeError"
        ) as caught:
            env = env_capture.capture_environment()

        assert env.loaded_modules == []
        assert Path(caught[0].filename) == Path(__file__)

    def test_missing_cuda_package_stays_quiet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        monkeypatch.setattr(env_capture, "find_spec", lambda _name: None)
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert env_capture._cuda_build_info_path() is None

        assert not caught

    def test_cuda_spec_import_failure_stays_quiet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        monkeypatch.setattr(
            env_capture,
            "find_spec",
            lambda _name: (_ for _ in ()).throw(ImportError("optional dependency")),
        )
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert env_capture._cuda_build_info_path() is None

        assert not caught

    def test_cuda_metadata_lookup_does_not_import_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        real_import = builtins.__import__

        def import_module(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "luxar.gsplats.models.gsplats.cuda":
                raise AssertionError("metadata lookup imported the CUDA backend")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", import_module)
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            info_path = env_capture._cuda_build_info_path()

        assert info_path is not None
        assert info_path.name == "cuda_build_info.json"
        assert not caught

    def test_broken_cuda_package_warns_at_external_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        monkeypatch.setattr(
            env_capture,
            "find_spec",
            lambda _name: (_ for _ in ()).throw(RuntimeError("CUDA package broke")),
        )
        monkeypatch.setattr(env_capture, "_detect_loaded_modules", lambda: [])
        monkeypatch.setattr(env_capture, "version", lambda _name: "test")
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())

        with pytest.warns(
            RuntimeWarning,
            match="_cuda_build_info_path.*RuntimeError: CUDA package broke",
        ) as caught:
            env = env_capture.capture_environment()

        assert env.loaded_modules == []
        assert Path(caught[0].filename) == Path(__file__)

    def test_version_probe_failure_warns_and_uses_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        monkeypatch.setattr(env_capture, "_detect_loaded_modules", lambda: [])
        monkeypatch.setattr(env_capture, "read_cuda_build_info", lambda: {})
        monkeypatch.setattr(
            env_capture,
            "version",
            lambda _name: (_ for _ in ()).throw(RuntimeError("metadata broke")),
        )
        monkeypatch.setattr(env_capture, "_warned_probe_failures", set())

        with pytest.warns(
            RuntimeWarning, match="_luxar_version.*RuntimeError"
        ) as caught:
            env = env_capture.capture_environment()

        assert env.luxar_version == "unknown"
        assert Path(caught[0].filename) == Path(__file__)

    def test_missing_version_metadata_stays_quiet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luxar.gsplats.batch import env_capture

        monkeypatch.setattr(
            env_capture,
            "version",
            lambda _name: (_ for _ in ()).throw(
                env_capture.PackageNotFoundError("luxar")
            ),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert env_capture._luxar_version() == "unknown"

        assert not caught

    def test_generate_preamble_conda(self) -> None:
        from luxar.gsplats.batch.env_capture import CapturedEnv, generate_env_preamble

        env = CapturedEnv(
            conda_prefix="/opt/conda/envs/myenv",
            loaded_modules=["cuda/12.1"],
            env_vars={"CUDA_HOME": "/usr/local/cuda"},
        )
        preamble = generate_env_preamble(env)
        assert "conda activate myenv" in preamble
        assert "module load cuda/12.1" in preamble
        assert "export CUDA_HOME=/usr/local/cuda" in preamble

    def test_generate_preamble_venv(self) -> None:
        from luxar.gsplats.batch.env_capture import CapturedEnv, generate_env_preamble

        env = CapturedEnv(virtual_env="/home/user/venv")
        preamble = generate_env_preamble(env)
        assert "source /home/user/venv/bin/activate" in preamble

    def test_shell_injection_prevention(self) -> None:
        """Verify that shell metacharacters are properly escaped via shlex.quote."""
        from luxar.gsplats.batch.env_capture import CapturedEnv, generate_env_preamble

        env = CapturedEnv(
            conda_prefix="/opt/envs/$(whoami)",
            loaded_modules=["cuda/12.1; rm -rf /"],
            env_vars={"LD_LIBRARY_PATH": '/usr/lib"; echo PWNED; echo "'},
        )
        preamble = generate_env_preamble(env)
        # shlex.quote wraps dangerous strings in single quotes
        # so semicolons should only appear inside single-quoted strings
        for line in preamble.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            if ";" in line:
                # The semicolon must be inside single quotes (shlex.quote output)
                assert "'" in line, f"Unquoted semicolon in: {line}"


# ====================================================================
# Time estimation tests
# ====================================================================


class TestTimeEstimate:
    @pytest.fixture
    def throughput_table(self) -> list:
        return [
            {
                "voxels": 16777216,
                "amp_train_ms": 3.5,
                "oom": False,
                "label": "256",
                "shape": [256, 256, 256],
                "splats": 5000,
            },
            {
                "voxels": 134217728,
                "amp_train_ms": 8.5,
                "oom": False,
                "label": "512",
                "shape": [512, 512, 512],
                "splats": 20000,
            },
            {
                "voxels": 452984832,
                "amp_train_ms": 23.0,
                "oom": False,
                "label": "768",
                "shape": [768, 768, 768],
                "splats": 50000,
            },
        ]

    def test_estimate_within_range(self, throughput_table: list) -> None:
        from luxar.gsplats.batch.time_estimate import estimate_tile_wall_seconds

        # 384^3 = ~56M voxels, between 256^3 and 512^3
        seconds = estimate_tile_wall_seconds(56623104, 3000, throughput_table)
        # Should be between 3.5*3000/1000 and 8.5*3000/1000 (with overheads)
        assert seconds > 0
        assert seconds < 3600  # Should be well under 1 hour

    def test_estimate_below_range(self, throughput_table: list) -> None:
        from luxar.gsplats.batch.time_estimate import estimate_tile_wall_seconds

        seconds = estimate_tile_wall_seconds(1000, 100, throughput_table)
        assert seconds > 0

    def test_estimate_above_range(self, throughput_table: list) -> None:
        from luxar.gsplats.batch.time_estimate import estimate_tile_wall_seconds

        # Above max — should extrapolate with safety factor
        seconds = estimate_tile_wall_seconds(1000000000, 3000, throughput_table)
        assert seconds > 0

    def test_slurm_time_limit_format(self) -> None:
        from luxar.gsplats.batch.time_estimate import estimate_slurm_time_limit

        assert estimate_slurm_time_limit(60) == "00:15:00"  # 1 min -> 15 min
        assert estimate_slurm_time_limit(900) == "00:15:00"  # 15 min exactly -> 15 min
        assert (
            estimate_slurm_time_limit(901) == "00:30:00"
        )  # Just over 15 min -> 30 min
        assert estimate_slurm_time_limit(3600) == "01:00:00"  # 60 min exactly


# ====================================================================
# Slurm script generation tests
# ====================================================================


class TestSlurmGen:
    def test_fit_script_has_task_decoding(self) -> None:
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

        manifest = BatchManifest(
            input_path="/data/test.zarr",
            output_dir="/output",
            n_channels=2,
            n_tiles=4,
            total_tasks=8,
            tile_size=256,
            tile_overlap=32,
            preset="standard",
            slurm_partition="gpu",
            slurm_time_limit="01:00:00",
        )

        script = generate_fit_sbatch(manifest, "# env preamble\n")
        assert "#!/bin/bash" in script
        assert "#SBATCH --array=0-7" in script
        assert "#SBATCH --partition=gpu" in script
        assert "N_CHANNELS=2" in script
        assert "N_TILES=4" in script
        assert "luxar gsplat fit" in script
        assert "--tile $K/4" in script
        assert "--tile-size 256" in script

    def test_fit_script_content_mode(self) -> None:
        """A content-mode manifest fans the SHARED plan across the array: each
        task fits one box via `fit --tiling content --plan … --plan-box $K`, and
        the uniform `--tile/--tile-size` flags are absent."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

        manifest = BatchManifest(
            input_path="/data/test.zarr",
            output_dir="/output",
            mode="content",
            plan_path="/output/plan.json",
            n_channels=2,
            n_tiles=5,  # box count in content mode
            total_tasks=10,
            preset="standard",
            slurm_partition="gpu",
            slurm_time_limit="01:00:00",
        )

        script = generate_fit_sbatch(manifest, "")
        assert "--tiling content" in script
        assert "--plan /output/plan.json" in script
        assert "--plan-box $K" in script
        assert "N_TILES=5" in script
        # the uniform-only flags must NOT appear
        assert "--tile $K/" not in script
        assert "--tile-size" not in script
        # outputs use the box label, not tile
        assert "_box$(printf" in script
        # a legitimately-empty box (per-attempt ${STAGING}.empty marker) is a
        # clean exit-0, not a failed task — the script records a ${OUTPUT}.empty
        # marker for the merge.
        assert "${STAGING}.empty" in script
        assert 'touch "${OUTPUT}.empty"' in script

    def test_merge_script(self) -> None:
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.slurm_gen import generate_merge_sbatch

        manifest = BatchManifest(
            output_dir="/output",
            slurm_partition="gpu",
        )

        script = generate_merge_sbatch(manifest, "# env\n")
        assert "luxar gsplat batch-fit merge" in script
        assert "#SBATCH --job-name=luxar-merge" in script
        # No recipe planned → plain partition merge (no --recipe flag).
        assert "--recipe" not in script

    def test_merge_script_emits_per_part_recipe(self) -> None:
        """A planned per-part merge recipe + knobs are threaded onto the merge
        command so the Slurm merge job streams a partition of LOD'd parts."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.slurm_gen import generate_merge_sbatch

        manifest = BatchManifest(
            output_dir="/output",
            slurm_partition="gpu",
            merge_recipe="substitutive",
            merge_recipe_args={"compression-factor": "4", "levels": "2"},
        )

        script = generate_merge_sbatch(manifest, "# env\n")
        assert "luxar gsplat batch-fit merge" in script
        # Legacy manifest spelling is canonicalized on emission (the merge
        # CLI rejects old names).
        assert "--recipe levels" in script
        assert "--compression-factor 4" in script
        assert "--levels 2" in script

    def test_fit_script_with_account(self) -> None:
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

        manifest = BatchManifest(
            input_path="/data/test.zarr",
            output_dir="/output",
            total_tasks=1,
            n_tiles=1,
            n_channels=1,
            tile_size=256,
            slurm_partition="gpu",
            slurm_account="myproject",
            slurm_time_limit="00:30:00",
        )

        script = generate_fit_sbatch(manifest, "")
        assert "#SBATCH --account=myproject" in script

    def test_task_id_decoding_arithmetic(self) -> None:
        """Verify the bash arithmetic matches Python decode_task_id."""
        from luxar.gsplats.batch.manifest import BatchManifest, decode_task_id

        manifest = BatchManifest(n_channels=3, n_tiles=5)

        for task_id in range(15):
            t, c, k = decode_task_id(task_id, manifest)
            # Replicate bash arithmetic
            n_c = 3
            n_k = 5
            bash_t = task_id // (n_c * n_k)
            bash_r = task_id % (n_c * n_k)
            bash_c = bash_r // n_k
            bash_k = bash_r % n_k
            assert (bash_t, bash_c, bash_k) == (t, c, k), (
                f"Mismatch at task_id={task_id}"
            )

    def test_fit_args_shell_injection_prevention(self) -> None:
        """Verify fit_args values are shell-escaped in generated scripts."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

        manifest = BatchManifest(
            input_path="/data/test.zarr",
            output_dir="/output",
            total_tasks=1,
            n_tiles=1,
            n_channels=1,
            tile_size=256,
            slurm_partition="gpu",
            slurm_time_limit="00:30:00",
            fit_args={"seed_method": 'edges"; rm -rf /'},
        )
        script = generate_fit_sbatch(manifest, "")
        # The dangerous value should be single-quoted by shlex.quote
        assert "rm -rf" not in script or "'" in script
        # Verify the value appears quoted
        assert "'edges" in script


# ====================================================================
# Tiling integration tests
# ====================================================================


class TestTilingIntegration:
    def test_tile_count_matches_compute_tile_specs(self) -> None:
        """Verify that batch plan always uses compute_tile_specs for tile count,
        even when volume_shape == tile_size (overlap creates extra tiles)."""
        from luxar.gsplats.tiling import compute_tile_specs

        # Volume exactly equal to tile size — overlap still creates a 2x2x2 grid
        specs = compute_tile_specs((128, 128, 128), 128, 32)
        assert len(specs) > 1, f"Expected >1 tiles due to overlap, got {len(specs)}"

        # Volume smaller than tile size — always 1 tile
        specs = compute_tile_specs((64, 64, 64), 128, 32)
        assert len(specs) == 1

        # Volume much larger — many tiles
        specs = compute_tile_specs((512, 512, 512), 128, 32)
        assert len(specs) > 1


# ====================================================================
# OME-Zarr discovery tests
# ====================================================================


class TestOMEZarrDiscovery:
    def test_discover_3d_zarr(self, tmp_path: Path) -> None:
        import zarr

        from luxar.io.ome_zarr import discover_ome_zarr_shape

        store_path = tmp_path / "test.zarr"
        root = zarr.open(str(store_path), mode="w")
        create_array(root, "0", data=np.zeros((64, 128, 128), dtype=np.float32))

        info = discover_ome_zarr_shape(store_path)
        assert info.n_timepoints == 1
        assert info.n_channels == 1
        assert info.spatial_shape == (64, 128, 128)

    def test_discover_5d_zarr_heuristic(self, tmp_path: Path) -> None:
        import zarr

        from luxar.io.ome_zarr import discover_ome_zarr_shape

        store_path = tmp_path / "test.zarr"
        root = zarr.open(str(store_path), mode="w")
        create_array(root, "0", data=np.zeros((10, 3, 64, 128, 128), dtype=np.float32))

        info = discover_ome_zarr_shape(store_path)
        assert info.n_timepoints == 10
        assert info.n_channels == 3
        assert info.spatial_shape == (64, 128, 128)

    def test_discover_ome_ngff_metadata(self, tmp_path: Path) -> None:
        import zarr

        from luxar.io.ome_zarr import discover_ome_zarr_shape

        store_path = tmp_path / "ome.zarr"
        root = zarr.open(str(store_path), mode="w")
        create_array(root, "0", data=np.zeros((5, 2, 32, 64, 64), dtype=np.float32))
        root.attrs["multiscales"] = [
            {
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.3, 0.3]}
                        ],
                    }
                ],
            }
        ]

        info = discover_ome_zarr_shape(store_path)
        assert info.n_timepoints == 5
        assert info.n_channels == 2
        assert info.channel_axes == ["c"]
        assert info.channel_shape == (2,)
        assert info.spatial_shape == (32, 64, 64)
        assert info.spatial_axes == ["z", "y", "x"]
        assert info.voxel_size == (0.5, 0.3, 0.3)
        assert info.unit == "micrometer"
        assert info.resolution_levels == 1


# ====================================================================
# Batch plan regression tests (from scripts/test_batch_plan_fixes.py)
# ====================================================================


class TestBatchPlanRegression:
    """Regression tests for batch plan HPC fixes."""

    def test_zarr_zip_suffix_detection(self) -> None:
        """Path('.zarr.zip') should be routed to zarr loading."""
        p = Path("foo.zarr.zip")
        assert p.suffix.lower() == ".zip"
        assert p.stem.endswith(".zarr")

        p2 = Path("bar.zarr")
        assert p2.suffix.lower() == ".zarr"

    def test_custom_axes_parsing(self) -> None:
        """_parse_custom_axes_attr correctly identifies T, C, and spatial dims."""
        from luxar.io.ome_zarr import _parse_custom_axes_attr

        # 6D Keller-style
        info = _parse_custom_axes_attr(
            ["time", "camera", "channel", "z", "y", "x"],
            (10, 2, 4, 97, 627, 1383),
            Path("test.zarr"),
        )
        assert info.n_timepoints == 10
        assert info.n_channels == 8  # 2*4
        assert info.channel_axes == ["camera", "channel"]
        assert info.channel_shape == (2, 4)
        assert info.spatial_shape == (97, 627, 1383)

        # 4D TimeFused-style
        info2 = _parse_custom_axes_attr(
            ["time", "z", "y", "x"],
            (1434, 108, 1352, 532),
            Path("test.zarr"),
        )
        assert info2.n_timepoints == 1434
        assert info2.n_channels == 1
        assert info2.channel_axes == []
        assert info2.channel_shape == ()
        assert info2.spatial_shape == (108, 1352, 532)

        # All spatial (no time, no channel)
        info3 = _parse_custom_axes_attr(
            ["z", "y", "x"],
            (100, 200, 300),
            Path("test.zarr"),
        )
        assert info3.n_timepoints == 1
        assert info3.n_channels == 1
        assert info3.spatial_shape == (100, 200, 300)

    def test_axes_override_validation(self, tmp_path: Path) -> None:
        """axes_override must match array ndim."""
        import zarr

        from luxar.io.ome_zarr import discover_ome_zarr_shape

        path = tmp_path / "test.zarr"
        z = zarr.open(str(path), mode="w")
        create_array(z, "data", data=np.zeros((5, 10, 20), dtype=np.float32))
        z.attrs["axes"] = ["z", "y", "x"]

        info = discover_ome_zarr_shape(path, axes_override=["time", "y", "x"])
        assert info.n_timepoints == 5

        with pytest.raises(ValueError):
            discover_ome_zarr_shape(path, axes_override=["t", "c", "z", "y", "x"])

    def test_array_selection_consistency(self, tmp_path: Path) -> None:
        """Both discover_ome_zarr_shape and _load_zarr_volume pick the largest array."""
        import zarr

        from luxar.io.ome_zarr import discover_ome_zarr_shape
        from luxar.io.volume import _load_zarr_volume

        path = tmp_path / "test.zarr"
        z = zarr.open(str(path), mode="w")
        create_array(z, "session1", data=np.ones((3, 10, 10), dtype=np.float32))
        create_array(z, "session2", data=np.ones((100, 20, 20), dtype=np.float32) * 2)

        info = discover_ome_zarr_shape(path)
        assert info.shape == (100, 20, 20)

        vol = _load_zarr_volume(path, channel=None, timepoint=None, array_key=None)
        assert vol.shape == (100, 20, 20)

    def test_auto_tile_small_volume(self) -> None:
        """Volumes that fit in GPU capacity should produce 1 tile."""
        from luxar.gsplats.tiling import compute_tile_specs

        spatial = (108, 1352, 532)
        tile_size = max(spatial) + 32
        specs = compute_tile_specs(spatial, tile_size, 32)
        assert len(specs) == 1

    def test_cull_retention_defaults(self) -> None:
        """fit_gaussian_splats should default to cull_retention=0.95."""
        import inspect

        from luxar.gsplats.fit_gsplats import fit_gaussian_splats

        sig = inspect.signature(fit_gaussian_splats)
        default = sig.parameters["cull_retention"].default
        assert default == 0.95

    def test_6d_channel_decoding(self) -> None:
        """Flat channel index should decode to channel-like axis coordinates."""
        from luxar.io.volume import decode_flat_channel_index

        assert decode_flat_channel_index(5, (2, 4)) == (1, 1)
        assert decode_flat_channel_index(0, (2, 4)) == (0, 0)
        assert decode_flat_channel_index(7, (2, 4)) == (1, 3)
        with pytest.raises(ValueError, match="out of range"):
            decode_flat_channel_index(8, (2, 4))

    def test_6d_zarr_volume_load_uses_flat_channel_index(self, tmp_path: Path) -> None:
        import zarr

        from luxar.io.volume import _load_zarr_volume

        path = tmp_path / "sixd.zarr"
        data = np.arange(3 * 2 * 4 * 2 * 3 * 5, dtype=np.float32).reshape(
            3, 2, 4, 2, 3, 5
        )
        root = zarr.open(str(path), mode="w")
        create_array(root, "0", data=data)

        loaded = _load_zarr_volume(path, channel=5, timepoint=2, array_key=None)
        np.testing.assert_array_equal(loaded, data[2, 1, 1])

    def test_labeled_multi_channel_plan_jobs_load_named_voxels(
        self, tmp_path: Path
    ) -> None:
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )
        from luxar.gsplats.batch.fit_command import build_task_fit_argv
        from luxar.io.volume import load_volume

        path = tmp_path / "multi-channel.zarr"
        data = np.arange(2 * 3 * 2 * 4 * 5 * 6, dtype=np.float32).reshape(
            2, 3, 2, 4, 5, 6
        )
        root = zarr.open(str(path), mode="w")
        create_array(root, "0", data=data)
        axes = ["camera", "time", "channel", "z", "y", "x"]

        manifest = plan_batch(
            input_path=path,
            output_dir=tmp_path / "out",
            tiling="uniform",
            tile_size=8,
            tile_overlap=0,
            axes_list=axes,
            array_key=None,
            timepoints_slice=None,
            channels_slice=None,
            fit=FitConfig(floor="none"),
            denoise=DenoiseConfig(),
            content=ContentKnobs(),
            merge=MergeConfig(),
        ).manifest

        assert manifest.n_timepoints == 3
        assert manifest.n_channels == 4
        for job in manifest.jobs:
            argv = build_task_fit_argv(manifest, job, "unused", argv0=["luxar"])
            loaded = load_volume(
                path,
                channel=int(argv[argv.index("--channel") + 1]),
                timepoint=int(argv[argv.index("--timepoint") + 1]),
                axes=argv[argv.index("--axes") + 1],
            )
            camera, channel = job.channel_coords
            np.testing.assert_array_equal(loaded, data[camera, job.timepoint, channel])

    @pytest.mark.parametrize(
        ("axes", "error"),
        [
            (["time", "t", "z", "y", "x"], "more than one time axis"),
            (["time", "view", "z", "y", "x"], "label 'view' not recognised"),
        ],
    )
    def test_plan_rejects_axes_workers_cannot_load(
        self, tmp_path: Path, axes: list[str], error: str
    ) -> None:
        import typer
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )

        path = tmp_path / "unsupported-axes.zarr"
        root = zarr.open(str(path), mode="w")
        create_array(root, "0", data=np.zeros((3, 2, 2, 3, 4), dtype=np.uint16))

        with pytest.raises(typer.BadParameter, match=error):
            plan_batch(
                input_path=path,
                output_dir=tmp_path / "out",
                tiling="uniform",
                tile_size=8,
                tile_overlap=0,
                axes_list=axes,
                array_key=None,
                timepoints_slice=None,
                channels_slice=None,
                fit=FitConfig(floor="none"),
                denoise=DenoiseConfig(),
                content=ContentKnobs(),
                merge=MergeConfig(),
            )

    def test_plan_squeezes_singleton_spatial_axes_for_positional_workers(
        self, tmp_path: Path
    ) -> None:
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )
        from luxar.gsplats.batch.fit_command import build_task_fit_argv
        from luxar.io.volume import load_volume

        path = tmp_path / "thin-movie.zarr"
        data = np.arange(3 * 1 * 1 * 8 * 8, dtype=np.float32).reshape(3, 1, 1, 8, 8)
        root = zarr.open(str(path), mode="w")
        create_array(root, "0", data=data)

        manifest = plan_batch(
            input_path=path,
            output_dir=tmp_path / "out",
            tiling="uniform",
            tile_size=8,
            tile_overlap=0,
            axes_list=None,
            array_key=None,
            timepoints_slice=None,
            channels_slice=None,
            fit=FitConfig(floor="none"),
            denoise=DenoiseConfig(),
            content=ContentKnobs(),
            merge=MergeConfig(),
        ).manifest

        assert manifest.spatial_shape == (8, 8)
        assert manifest.n_tiles == 1
        for job in manifest.jobs:
            argv = build_task_fit_argv(manifest, job, "unused", argv0=["luxar"])
            assert "--axes" not in argv
            loaded = load_volume(
                path,
                timepoint=int(argv[argv.index("--timepoint") + 1]),
            )
            assert loaded.shape == manifest.spatial_shape
            np.testing.assert_array_equal(loaded, data[job.timepoint, 0, 0])

    def test_content_plan_rejects_spatial_shape_squeezed_below_3d(
        self, tmp_path: Path
    ) -> None:
        import typer
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )

        path = tmp_path / "thin-content.zarr"
        root = zarr.open(str(path), mode="w")
        create_array(root, "0", data=np.ones((3, 2, 1, 64, 64), dtype=np.float32))

        with pytest.raises(typer.BadParameter) as excinfo:
            plan_batch(
                input_path=path,
                output_dir=tmp_path / "out",
                tiling="content",
                tile_size=None,
                tile_overlap=8,
                axes_list=None,
                array_key=None,
                timepoints_slice=None,
                channels_slice=None,
                fit=FitConfig(floor="none"),
                denoise=DenoiseConfig(),
                content=ContentKnobs(k_star_ref=4000, n_features_ref=200),
                merge=MergeConfig(),
            )

        message = str(excinfo.value)
        assert "spatial axes squeeze to (64, 64)" in message
        assert "--tiling uniform" in message
        assert "--axes" in message
        assert not (tmp_path / "out").exists()

    def test_content_plan_rejects_already_2d_spatial_shape(
        self, tmp_path: Path
    ) -> None:
        import typer
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )

        path = tmp_path / "plain-2d-content.zarr"
        root = zarr.open(str(path), mode="w")
        create_array(root, "0", data=np.ones((64, 64), dtype=np.float32))

        with pytest.raises(typer.BadParameter) as excinfo:
            plan_batch(
                input_path=path,
                output_dir=tmp_path / "out",
                tiling="content",
                tile_size=None,
                tile_overlap=8,
                axes_list=None,
                array_key=None,
                timepoints_slice=None,
                channels_slice=None,
                fit=FitConfig(floor="none"),
                denoise=DenoiseConfig(),
                content=ContentKnobs(k_star_ref=4000, n_features_ref=200),
                merge=MergeConfig(),
            )

        message = str(excinfo.value)
        assert "spatial shape (64, 64) is already 2-D" in message
        assert "squeeze" not in message
        assert "--tiling uniform" in message
        assert "--axes" not in message
        assert not (tmp_path / "out").exists()

    def test_planning_scan_squeezes_singleton_spatial_axes_lazily(
        self, tmp_path: Path
    ) -> None:
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import _pinned_slice_volume

        path = tmp_path / "thin-scan.zarr"
        data = np.arange(3 * 1 * 1 * 8 * 8, dtype=np.float32).reshape(3, 1, 1, 8, 8)
        root = zarr.open(str(path), mode="w")
        create_array(root, "0", data=data)

        view = _pinned_slice_volume(
            path,
            channel=0,
            timepoint=2,
            array_key=None,
            axes=None,
            axes_labels=["t", "c", "z", "y", "x"],
            channel_shape=(1,),
            spatial_shape=(8, 8),
        )

        assert view.shape == (8, 8)
        assert not isinstance(view, np.ndarray)
        np.testing.assert_array_equal(view[:], data[2, 0, 0])

    def test_planning_scan_eager_fallback_squeezes_singleton_spatial_axes(
        self, tmp_path: Path
    ) -> None:
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import _pinned_slice_volume

        path = tmp_path / "thin-fallback.zarr"
        data = np.arange(3 * 1 * 1 * 8 * 8, dtype=np.float32).reshape(3, 1, 1, 8, 8)
        root = zarr.open(str(path), mode="w")
        create_array(root, "0", data=data)

        view = _pinned_slice_volume(
            path,
            channel=0,
            timepoint=2,
            array_key=None,
            axes=None,
            axes_labels=["t", "c", "z", "y", "x"],
            channel_shape=(1, 1),
            spatial_shape=(8, 8),
        )

        assert view.shape == (8, 8)
        np.testing.assert_array_equal(view, data[2, 0, 0])

    def test_tasks_per_job_manifest(self) -> None:
        """Manifest fields are serializable and have correct defaults."""
        import json
        from dataclasses import asdict

        from luxar.gsplats.batch.manifest import BatchManifest

        m = BatchManifest()
        assert m.tasks_per_job == 1
        assert m.parallel_tasks_per_job is False

        d = asdict(m)
        json.dumps(d)  # should not raise

    def test_dimension_metadata_round_trips_through_manifest_json(
        self, tmp_path: Path
    ) -> None:
        from luxar.gsplats.batch.manifest import (
            BatchManifest,
            load_manifest,
            save_manifest,
        )

        metadata = [
            {"name": "z", "unit": "micrometer", "scale": 2.0},
            {"name": "time", "unit": "second", "scale": 0.5},
        ]
        save_manifest(BatchManifest(dimension_metadata=metadata), tmp_path)

        assert load_manifest(tmp_path).dimension_metadata == metadata

    def test_env_capture_ld_library_path_prepend(self) -> None:
        """LD_LIBRARY_PATH should be prepended, not replaced, in preamble."""
        from luxar.gsplats.batch.env_capture import CapturedEnv, generate_env_preamble

        env = CapturedEnv(
            env_vars={"LD_LIBRARY_PATH": "/some/path"},
            loaded_modules=["cuda/12.8"],
        )
        preamble = generate_env_preamble(env)
        assert "${LD_LIBRARY_PATH:-}" in preamble
        assert "export LD_LIBRARY_PATH='/some/path'\n" not in preamble

        ld_line_idx = preamble.find("export LD_LIBRARY_PATH=")
        module_line_idx = preamble.find("module load")
        assert ld_line_idx < module_line_idx, (
            "LD_LIBRARY_PATH export must appear before module load in preamble"
        )

    def test_sbatch_omits_channel_when_single(self) -> None:
        """When n_channels=1, sbatch script must NOT pass --channel."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.slurm_gen import generate_fit_sbatch

        m = BatchManifest(
            input_path="/data/test.zarr.zip",
            output_dir="/output/test",
            n_timepoints=10,
            n_channels=1,
            tile_size=1384,
            tile_overlap=32,
            n_tiles=1,
            total_tasks=10,
            preset="draft",
            slurm_partition="gpu",
            slurm_time_limit="00:15:00",
        )
        script = generate_fit_sbatch(m, "# preamble\n")
        assert "--channel" not in script
        assert "--timepoint" in script

        # Multi-channel, single timepoint
        m.n_channels = 4
        m.n_timepoints = 1
        m.total_tasks = 4
        script2 = generate_fit_sbatch(m, "# preamble\n")
        assert "--channel" in script2
        assert "--timepoint" not in script2

        # Both multi
        m.n_timepoints = 5
        m.total_tasks = 20
        script3 = generate_fit_sbatch(m, "# preamble\n")
        assert "--channel" in script3
        assert "--timepoint" in script3


# ====================================================================
# Merge orchestrator round-trip (post-v3.0-cutover)
# ====================================================================


class TestMergeOrchestrator:
    """End-to-end `merge_batch_results` fan-in over v3.0 .gsplats.zarr tiles.

    Exercises every level the orchestrator drives — Level 1 ``concatenate``,
    Level 2 ``combine_as_new_dimension``, Level 3 ``merge_with_channel_colors``
    — proving they all round-trip through the unified v3.0 node-tree writer.
    """

    def _tile(self, n: int, seed: int):
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        return GSplatData(
            centers=rng.uniform(0, 50, (n, 3)).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        )

    def _write_tiles(self, tiles_dir: Path, n_t: int, n_c: int, n_k: int):
        from luxar.gsplats.batch.manifest import output_filename

        tiles_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        seed = 0
        for t in range(n_t):
            for c in range(n_c):
                for k in range(n_k):
                    fname = output_filename(t, c, k, n_t, n_c, n_k)
                    tile = self._tile(4, seed)
                    tile.save(tiles_dir / fname)
                    total += tile.n_splats
                    seed += 1
        return total

    @staticmethod
    def _read_tree(path: Path):
        """Read a written .gsplats.zarr (any node kind) as a tree + root attrs."""
        import zarr

        from luxar.io._compiler.gsplat_tree import read_gsplat_node

        root = zarr.open_group(str(path), mode="r")
        node = read_gsplat_node(root, root)
        return node, dict(root.attrs)

    def test_merge_partition_barriers_time_from_spatial_shape(
        self, tmp_path: Path
    ) -> None:
        """REGRESSION (deep-double-check +deeper): _merge_partition derives the
        ordering barrier authoritatively from manifest.spatial_shape (the stacked
        -time axis = index len(spatial_shape)) and passes it to the streaming
        writer — so per-timepoint chunk locality holds even on SPARSE tiles where
        the value-based auto-detect (n_unique*4<=n guard) would miss the barrier.

        Fails on the pre-hardening code: without the explicit barrier the merge
        relied on auto-detect, which returns [] for this sparse part, leaving the
        time axis smeared across chunks (slice_dims=[])."""
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.io.ordering import detect_barrier_dims

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        # 14 timepoints × 2 tiles, only 3 splats per (t,k) → each stacked part has
        # 42 splats over 14 timepoints (42 < 14*4=56 → auto-detect MISSES it).
        n_t, n_k = 14, 2
        for t in range(n_t):
            for k in range(n_k):
                self._tile(3, seed=t * n_k + k).save(
                    tiles_dir / output_filename(t, 0, k, n_t, 1, n_k)
                )
        # spatial_shape has 3 dims → the stacked-time axis lands at index 3.
        manifest = BatchManifest(
            n_timepoints=n_t, n_channels=1, n_tiles=n_k, spatial_shape=(8, 8, 8)
        )
        final = merge_batch_results(manifest, out_dir, verbose=False, recipe=None)

        root = zarr.open_group(str(final), mode="r")
        assert root.attrs["kind"] == "partition"
        checked = 0
        for part_name in [k for k in root.group_keys() if k.startswith("part_")]:
            part = root[part_name]
            # Sanity: this part IS the sparse regime auto-detect would miss.
            centers = np.asarray(part["centers"])
            assert centers.shape[1] == 4  # 3 spatial + stacked time
            assert detect_barrier_dims(centers) == []  # auto-detect misses it
            # But the merge threaded the authoritative barrier → time is barriered.
            assert list(part.attrs["slice_dims"]) == [3]
            checked += 1
        assert checked >= 1

    def test_merge_single_tile_no_recipe_barriers_time(self, tmp_path: Path) -> None:
        """REGRESSION (deep-double-check +deeper): the K==1 / no-recipe merge
        branch (part.save bare leaf) must ALSO pass the authoritative stacked-time
        barrier — not silently fall back to value-based auto-detect. A single
        spatial tile stacked over sparse timepoints would otherwise smear time
        across chunks."""
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.io.ordering import detect_barrier_dims

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        n_t = 14  # single tile (n_k=1), 3 splats/tp → 42 splats over 14 tps (sparse)
        for t in range(n_t):
            self._tile(3, seed=t).save(tiles_dir / output_filename(t, 0, 0, n_t, 1, 1))
        manifest = BatchManifest(
            n_timepoints=n_t, n_channels=1, n_tiles=1, spatial_shape=(8, 8, 8)
        )
        final = merge_batch_results(manifest, out_dir, verbose=False, recipe=None)

        root = zarr.open_group(str(final), mode="r")
        # K==1 no-recipe → a bare leaf at the root (not a partition).
        assert root.attrs.get("kind") != "partition"
        centers = np.asarray(root["centers"])
        assert centers.shape[1] == 4
        assert detect_barrier_dims(centers) == []  # auto-detect misses the sparse axis
        assert list(root.attrs["slice_dims"]) == [3]  # authoritative barrier applied

    @pytest.mark.parametrize("flat", [False, True])
    def test_merge_preserves_dimension_metadata(
        self, tmp_path: Path, flat: bool
    ) -> None:
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for t in range(2):
            for k in range(2):
                self._tile(3, seed=t * 2 + k).save(
                    tiles_dir / output_filename(t, 0, k, 2, 1, 2)
                )
        metadata = [
            {"name": "z", "unit": "micrometer", "scale": 2.0},
            {"name": "y", "unit": "micrometer", "scale": 0.75},
            {"name": "x", "unit": "micrometer", "scale": 0.75},
            {"name": "time", "unit": "second", "scale": 0.5},
        ]
        manifest = BatchManifest(
            n_timepoints=2,
            n_channels=1,
            n_tiles=2,
            spatial_shape=(8, 8, 8),
            dimension_metadata=metadata,
        )

        final = merge_batch_results(
            manifest, out_dir, verbose=False, recipe=None, flat=flat
        )

        root = zarr.open_group(str(final), mode="r")
        assert root.attrs["dimension_metadata"] == metadata

        from luxar.gsplats.io.load_gsplats import load_gsplat_node

        node, _ = load_gsplat_node(final)
        assert node.meta["dimension_metadata"] == metadata

    def test_single_flat_tile_preserves_metadata_and_pipeline(
        self, tmp_path: Path
    ) -> None:
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        tile = self._tile(1, seed=0)
        tile.stats.update(
            {
                "final_loss": 0.25,
                "n_iters": 999,
                "image_min": 7.0,
                "floor": 3.25,
            }
        )
        tile.save(tiles_dir / output_filename(0, 0, 0, 1, 1, 1))
        metadata = [
            {"name": "z", "scale": 1.0},
            {"name": "y", "scale": 1.0},
            {"name": "x", "scale": 1.0},
        ]
        manifest = BatchManifest(
            n_timepoints=1,
            n_channels=1,
            n_tiles=1,
            spatial_shape=(8, 8, 8),
            dimension_metadata=metadata,
        )

        final = merge_batch_results(
            manifest, out_dir, verbose=False, recipe=None, flat=True
        )

        root = zarr.open_group(str(final), mode="r")
        assert root.attrs["dimension_metadata"] == metadata
        assert root["fitting"].attrs["final_loss"] == pytest.approx(0.25)
        assert root["pipeline"].attrs["n_iters"] == 999
        assert root["pipeline"].attrs["image_min"] == pytest.approx(7.0)
        assert root["pipeline"].attrs["floor"] == pytest.approx(3.25)
        stored_hash = root.attrs["content_hash"]

        from luxar._zarr_compat import open_group as zc_open_group
        from luxar.gsplats.io.save_gsplats import _stamp_content_hash

        assert _stamp_content_hash(zc_open_group(final, mode="r+")) == stored_hash

    @pytest.mark.parametrize("flat", [False, True])
    def test_merge_omits_dimension_metadata_with_wrong_width(
        self, tmp_path: Path, flat: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for k in range(2):
            self._tile(2, seed=k).save(tiles_dir / output_filename(0, 0, k, 1, 1, 2))
        manifest = BatchManifest(
            n_timepoints=1,
            n_channels=1,
            n_tiles=2,
            spatial_shape=(8, 8, 8),
            dimension_metadata=[{"name": "z"}, {"name": "y"}],
        )

        final = merge_batch_results(
            manifest, out_dir, verbose=False, recipe=None, flat=flat
        )

        root = zarr.open_group(str(final), mode="r")
        assert "dimension_metadata" not in root.attrs
        assert "omitting it" in capsys.readouterr().out

    def test_merge_channels_with_colors_round_trips(self, tmp_path: Path) -> None:
        """T=1, C=2, K=2 (--flat): Level-1 concat then Level-3 color merge."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData

        out_dir = tmp_path / "batch"
        total = self._write_tiles(out_dir / "tiles", n_t=1, n_c=2, n_k=2)

        manifest = BatchManifest(n_timepoints=1, n_channels=2, n_tiles=2)
        final = merge_batch_results(
            manifest,
            out_dir,
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            verbose=False,
            flat=True,
        )
        assert final.exists()
        merged = GSplatData.load(final)
        assert merged.n_splats == total  # 1*2*2 tiles * 4 splats = 16
        # channel colors were applied
        assert merged.colors is not None

    def test_stack_timepoints_round_trips(self, tmp_path: Path) -> None:
        """T=2, C=1, K=1 (--flat): Level-2 combine_as_new_dimension lifts 3D->4D."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData

        out_dir = tmp_path / "batch"
        total = self._write_tiles(out_dir / "tiles", n_t=2, n_c=1, n_k=1)

        manifest = BatchManifest(n_timepoints=2, n_channels=1, n_tiles=1)
        final = merge_batch_results(manifest, out_dir, verbose=False, flat=True)
        assert final.exists()
        merged = GSplatData.load(final)
        assert merged.n_splats == total
        assert merged.ndim == 4  # promoted by the timepoint stack

    # ── Default partition path (tile-outer, streaming) ──────────────

    def test_default_merge_is_partition_with_one_part_per_tile(
        self, tmp_path: Path
    ) -> None:
        """K>1 default → kind=partition, K parts, total splats conserved.

        The ``kind == "partition"`` assertion FAILS on the pre-change flat-leaf
        output (which had no ``kind`` attr) — pinning the new default.
        """
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.tree import GSplatPartition, total_splats

        out_dir = tmp_path / "batch"
        n_k = 3
        total = self._write_tiles(out_dir / "tiles", n_t=1, n_c=1, n_k=n_k)

        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=n_k)
        final = merge_batch_results(manifest, out_dir, verbose=False)
        assert final.exists()

        node, attrs = self._read_tree(final)
        # Would FAIL on the old flat-leaf output (a leaf has no kind=partition).
        assert attrs["kind"] == "partition"
        assert attrs["type"] == "group"
        assert attrs["display_type"] == "gsplats"
        assert isinstance(node, GSplatPartition)
        assert node.n_children == n_k
        # Total splat count conserved vs the flat-concat path.
        assert total_splats(node) == total

    @pytest.mark.parametrize("n_tiles", [1, 3])
    def test_merge_preserves_source_matched_amplitudes(
        self, tmp_path: Path, n_tiles: int
    ) -> None:
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.tree import iter_leaves

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        self._write_tiles(tiles_dir, n_t=1, n_c=1, n_k=n_tiles)
        expected_parts = []
        for tile_index in range(n_tiles):
            path = tiles_dir / output_filename(0, 0, tile_index, 1, 1, n_tiles)
            tile = GSplatData.load(path)
            tile.amplitudes[:] = np.geomspace(
                1e-3, 1.0, tile.n_splats, dtype=np.float32
            )
            tile.stats["source_dtype"] = "uint8"
            tile.save(path, amplitude_bits="auto")
            expected_parts.append(tile.amplitudes)

        final = merge_batch_results(
            BatchManifest(n_timepoints=1, n_channels=1, n_tiles=n_tiles),
            out_dir,
            verbose=False,
        )

        root = zarr.open_group(str(final), mode="r")
        amplitude_arrays = []

        def collect(group) -> None:
            if "amplitudes" in group.array_keys():
                amplitude_arrays.append(group["amplitudes"])
            for name in group.group_keys():
                collect(group[name])

        collect(root)
        assert amplitude_arrays
        for array in amplitude_arrays:
            encoding = array.attrs["encoding"]
            if encoding["name"] == "array_ref":
                assert np.dtype(root[encoding["target"]].dtype) == np.dtype(np.uint8)
            else:
                assert np.dtype(array.dtype) == np.dtype(np.uint8)
                assert encoding["name"] == "geolog_scalar_uint8"
        node, _ = self._read_tree(final)
        actual = np.concatenate(
            [
                sublod.amplitudes
                for leaf in iter_leaves(node)
                for sublod in leaf.additive_sublods
            ]
        )
        expected = np.concatenate(expected_parts)
        np.testing.assert_allclose(
            np.sort(actual), np.sort(expected), rtol=0.04, atol=1e-7
        )

    def test_partition_matches_flat_total_and_has_tight_bounds(
        self, tmp_path: Path
    ) -> None:
        """Partition total == flat total; per-part + union bounds are tight."""
        import numpy as np
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.tree import total_splats

        # Flat reference total.
        flat_dir = tmp_path / "flat"
        total = self._write_tiles(flat_dir / "tiles", n_t=1, n_c=1, n_k=3)
        m1 = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=3)
        flat_final = merge_batch_results(m1, flat_dir, verbose=False, flat=True)
        flat_total = GSplatData.load(flat_final).n_splats
        assert flat_total == total

        # Partition over the SAME tiles.
        part_dir = tmp_path / "part"
        self._write_tiles(part_dir / "tiles", n_t=1, n_c=1, n_k=3)
        m2 = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=3)
        part_final = merge_batch_results(m2, part_dir, verbose=False)

        node, attrs = self._read_tree(part_final)
        assert total_splats(node) == flat_total

        # Each part's position_bounds is the TIGHT actual-splat extent.
        root = zarr.open_group(str(part_final), mode="r")
        from luxar.gsplats.tree import center_bounds

        union_min = None
        union_max = None
        for i, child in enumerate(node.children):
            pb = dict(root[f"part_{i}"].attrs["position_bounds"])
            cb = center_bounds(child)
            assert cb is not None
            lo, hi = cb
            np.testing.assert_allclose(pb["min"], lo, rtol=1e-5, atol=1e-4)
            np.testing.assert_allclose(pb["max"], hi, rtol=1e-5, atol=1e-4)
            union_min = lo if union_min is None else np.minimum(union_min, lo)
            union_max = hi if union_max is None else np.maximum(union_max, hi)
        # Root union bounds correct.
        np.testing.assert_allclose(
            attrs["position_bounds"]["min"], union_min, rtol=1e-5, atol=1e-4
        )
        np.testing.assert_allclose(
            attrs["position_bounds"]["max"], union_max, rtol=1e-5, atol=1e-4
        )

    def test_single_tile_emits_bare_leaf_not_partition(self, tmp_path: Path) -> None:
        """K=1 → bare leaf, no partition wrapper."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.tree import GSplatLeaf

        out_dir = tmp_path / "batch"
        total = self._write_tiles(out_dir / "tiles", n_t=1, n_c=1, n_k=1)

        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=1)
        final = merge_batch_results(manifest, out_dir, verbose=False)
        node, attrs = self._read_tree(final)
        assert attrs.get("kind") != "partition"
        assert isinstance(node, GSplatLeaf)
        # A bare leaf round-trips through GSplatData.load (matrix-shaped).
        merged = GSplatData.load(final)
        assert merged.n_splats == total

    def test_single_tile_moves_quality_under_part_provenance(
        self, tmp_path: Path
    ) -> None:
        """K=1 keeps the tile score attributable instead of claiming it at root."""
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        self._write_tiles(tiles_dir, n_t=1, n_c=1, n_k=1)
        path = tiles_dir / output_filename(0, 0, 0, 1, 1, 1)
        tile = GSplatData.load(path)
        tile.stats.update(psnr_db=40.0, foreground_psnr_db=20.0)
        tile.save(path)

        final = merge_batch_results(
            BatchManifest(n_timepoints=1, n_channels=1, n_tiles=1),
            out_dir,
            verbose=False,
        )

        fitting = dict(zarr.open_group(str(final), mode="r")["fitting"].attrs)
        assert "psnr_db" not in fitting
        assert "foreground_psnr_db" not in fitting
        provenance = fitting["part_provenance"]
        assert [record["coordinate"] for record in provenance] == [0.0]
        assert provenance[0]["fitting"]["psnr_db"] == 40.0
        assert provenance[0]["fitting"]["foreground_psnr_db"] == 20.0

    def test_flat_flag_emits_single_leaf(self, tmp_path: Path) -> None:
        """--flat → single flat leaf (old behavior), loadable as GSplatData."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.tree import GSplatLeaf

        out_dir = tmp_path / "batch"
        total = self._write_tiles(out_dir / "tiles", n_t=1, n_c=1, n_k=3)

        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=3)
        final = merge_batch_results(manifest, out_dir, verbose=False, flat=True)
        node, attrs = self._read_tree(final)
        assert attrs.get("kind") != "partition"
        assert isinstance(node, GSplatLeaf)
        assert GSplatData.load(final).n_splats == total

    def test_partition_4d_parts_carry_stacked_timepoints(self, tmp_path: Path) -> None:
        """T=2, C=1, K=2 → partition whose parts are 4D (timepoints stacked)."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.tree import GSplatPartition, total_splats

        out_dir = tmp_path / "batch"
        total = self._write_tiles(out_dir / "tiles", n_t=2, n_c=1, n_k=2)

        manifest = BatchManifest(n_timepoints=2, n_channels=1, n_tiles=2)
        final = merge_batch_results(manifest, out_dir, verbose=False)
        node, attrs = self._read_tree(final)
        assert attrs["kind"] == "partition"
        assert isinstance(node, GSplatPartition)
        assert node.n_children == 2
        # Each part stacked its 2 timepoints → 4D, and holds both tps' splats.
        for child in node.children:
            assert child.ndim == 4
            assert total_splats(child) == 8  # 2 timepoints * 4 splats
        assert total_splats(node) == total  # = 2*1*2 tiles * 4 = 16

    def test_partition_merge_collects_nested_tile_fit_provenance(
        self, tmp_path: Path
    ) -> None:
        """Streaming merge retains every tile stamp without inventing a root score."""
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        self._write_tiles(tiles_dir, n_t=2, n_c=2, n_k=3)

        for t in range(2):
            for c in range(2):
                for k in range(3):
                    path = tiles_dir / output_filename(t, c, k, 2, 2, 3)
                    if k == 1:
                        shutil.rmtree(path)
                        Path(f"{path}.empty").touch()
                        continue
                    tile = GSplatData.load(path)
                    if (t, c, k) != (1, 1, 2):
                        tile.stats.update(
                            psnr_db=40.0 + 4 * k + 2 * c + t,
                            foreground_psnr_db=20.0 + 4 * k + 2 * c + t,
                            source_shape=[4, 5, 6],
                            source_dtype="uint16",
                        )
                    tile.save(path)

        final = merge_batch_results(
            BatchManifest(n_timepoints=2, n_channels=2, n_tiles=3),
            out_dir,
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            verbose=False,
        )

        root = zarr.open_group(str(final), mode="r")
        fitting = dict(root["fitting"].attrs)
        assert "psnr_db" not in fitting
        parts = fitting["part_provenance"]
        assert [part["coordinate"] for part in parts] == [0.0, 2.0]
        assert all("fit_reference" not in part for part in parts)
        assert "part_provenance" not in dict(root["part_0"].attrs["lod_stats"])
        assert "part_provenance" not in dict(root["part_1"].attrs["lod_stats"])

        channels = parts[1]["fitting"]["part_provenance"]
        assert [channel["coordinate"] for channel in channels] == [0.0, 1.0]
        timepoints = channels[1]["fitting"]["part_provenance"]
        assert [timepoint["coordinate"] for timepoint in timepoints] == [0.0, 1.0]
        assert timepoints[0]["fitting"]["psnr_db"] == 50.0
        assert "psnr_db" not in timepoints[1]["fitting"]
        assert "foreground_psnr_db" not in timepoints[1]["fitting"]
        assert "source_shape" not in timepoints[1]["fitting"]

    def test_partition_merge_embeds_lone_surviving_timepoint_coordinate(
        self, tmp_path: Path
    ) -> None:
        """A tile present only at t=2 remains a one-frame 4D stack at t=2."""
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        self._write_tiles(tiles_dir, n_t=3, n_c=1, n_k=2)
        for t in (0, 1):
            path = tiles_dir / output_filename(t, 0, 1, 3, 1, 2)
            shutil.rmtree(path)
            Path(f"{path}.empty").touch()
        path = tiles_dir / output_filename(2, 0, 1, 3, 1, 2)
        tile = GSplatData.load(path)
        tile.stats.update(source_shape=[4, 5, 6], source_dtype="uint16")
        tile.save(path)

        final = merge_batch_results(
            BatchManifest(n_timepoints=3, n_channels=1, n_tiles=2),
            out_dir,
            verbose=False,
        )

        root = zarr.open_group(str(final), mode="r")
        parts = dict(root["fitting"].attrs)["part_provenance"]
        timepoints = parts[1]["fitting"]["part_provenance"]
        assert [record["coordinate"] for record in timepoints] == [2.0]
        assert parts[1]["fitting"]["source_shape"] == [1, 4, 5, 6]

    def test_partition_multichannel_parts_carry_colors(self, tmp_path: Path) -> None:
        """T=1, C=2, K=2 with colors → partition; each part is color-merged."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.tree import GSplatPartition, total_splats

        out_dir = tmp_path / "batch"
        total = self._write_tiles(out_dir / "tiles", n_t=1, n_c=2, n_k=2)

        manifest = BatchManifest(n_timepoints=1, n_channels=2, n_tiles=2)
        final = merge_batch_results(
            manifest,
            out_dir,
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            verbose=False,
        )
        node, attrs = self._read_tree(final)
        assert attrs["kind"] == "partition"
        assert isinstance(node, GSplatPartition)
        assert node.n_children == 2
        for child in node.children:
            # Each part merged its 2 channels' splats and carries colors.
            assert total_splats(child) == 8  # 2 channels * 4 splats
            for leaf in self._leaves(child):
                assert leaf.additive_sublods[0].colors is not None
        assert total_splats(node) == total

    @staticmethod
    def _leaves(node):
        from luxar.gsplats.tree import iter_leaves

        return list(iter_leaves(node))

    def test_partition_path_never_concatenates_all_tiles(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """OOM-fix guard: the partition path must NOT concatenate across tiles.

        ``concatenate`` is legitimately used WITHIN a part only for multi-channel
        (no-color) merges; for the single-channel case here it must never run
        over the full tile set. We spy on it and assert it is not called.
        """
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData

        out_dir = tmp_path / "batch"
        self._write_tiles(out_dir / "tiles", n_t=1, n_c=1, n_k=4)

        calls = {"n": 0}
        real = GSplatData.concatenate.__func__  # type: ignore[attr-defined]

        def _spy(cls, datasets):
            calls["n"] += 1
            return real(cls, datasets)

        monkeypatch.setattr(GSplatData, "concatenate", classmethod(_spy))

        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=4)
        merge_batch_results(manifest, out_dir, verbose=False)
        assert calls["n"] == 0

    def test_concatenate_preserves_pyramid_per_cell(self, tmp_path: Path) -> None:
        """Multi-substitutive concatenate keeps every (substitutive, additive)
        cell — the orchestrator never silently flattens a pyramid."""
        from luxar.gsplats.gsplat_data import (
            AdditiveSubLOD,
            GSplatData,
            SubstitutiveLevel,
        )

        def _pyr(seed: int) -> GSplatData:
            def _sub(n: int, s: int) -> AdditiveSubLOD:
                rng = np.random.default_rng(s)
                chol = np.zeros((n, 6), dtype=np.float32)
                chol[:, [0, 2, 5]] = 1.0
                return AdditiveSubLOD(
                    centers=rng.uniform(0, 50, (n, 3)).astype(np.float32),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=chol,
                )

            return GSplatData.from_substitutive_levels(
                [
                    SubstitutiveLevel(additive_sublods=[_sub(20, seed)], level_index=0),
                    SubstitutiveLevel(
                        additive_sublods=[_sub(5, seed + 100)],
                        compression_factor=4,
                        level_index=1,
                    ),
                ]
            )

        merged = GSplatData.concatenate([_pyr(0), _pyr(1)])
        assert merged.n_substitutive == 2
        assert merged.at_substitutive(0).n_splats == 40  # 20 + 20
        assert merged.at_substitutive(1).n_splats == 10  # 5 + 5

        # round-trips through v3.0
        out = tmp_path / "pyr.gsplats.zarr"
        merged.save(out)
        reloaded = GSplatData.load(out)
        assert reloaded.n_substitutive == 2
        assert reloaded.at_substitutive(0).n_splats == 40
        assert reloaded.at_substitutive(1).n_splats == 10

    # ----------------------------------------------------------------
    # --timepoints / --channels slicing: tile filenames carry the REAL
    # dataset indices (e.g. t0072, not t01), so the merge must resolve
    # them through manifest.timepoint_indices / channel_indices via the
    # shared _tile_indices / _tile_path helpers. Both the partition and
    # the flat path go through those helpers; cover both.
    # ----------------------------------------------------------------
    def _write_sliced_tiles(
        self,
        tiles_dir: Path,
        t_indices: list[int],
        c_indices: list[int],
        n_k: int,
    ) -> int:
        """Write tiles named by REAL dataset indices, exactly as the sbatch fit
        job does (``slurm_gen.generate_fit_sbatch``): printf widths derived from
        the MAX real index, not the selection count. Reproduced independently of
        ``output_filename`` so the test pins the fit-side ↔ merge-side naming
        contract rather than tautologically reusing the merge's own helper.
        """
        tiles_dir.mkdir(parents=True, exist_ok=True)
        # Mirror slurm_gen.py:106-108 (t_width/c_width/k_width).
        t_w = max(2, len(str(max(t_indices))))
        c_w = max(2, len(str(max(c_indices))))
        k_w = max(3, len(str(max(0, n_k - 1))))
        total = 0
        seed = 0
        for t in t_indices:
            for c in c_indices:
                for k in range(n_k):
                    fname = f"t{t:0{t_w}d}_c{c:0{c_w}d}_tile{k:0{k_w}d}.gsplats.zarr"
                    tile = self._tile(4, seed)
                    tile.save(tiles_dir / fname)
                    total += tile.n_splats
                    seed += 1
        return total

    def test_partition_merge_resolves_sliced_real_indices(self, tmp_path: Path) -> None:
        """--timepoints/--channels slicing → partition merge resolves the real
        filename indices AND places splats at the real timepoint coordinates.

        Regression guard: a merge that used sequential indices (0,1) instead of
        the manifest's real indices (5,72) would either raise FileNotFoundError
        (wrong filename) or stack splats at the wrong time coordinate.
        """
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.tree import GSplatPartition, total_splats

        out_dir = tmp_path / "batch"
        t_indices = [5, 72]  # real, non-sequential, two-digit (width 2)
        c_indices = [1, 3]
        n_k = 2
        total = self._write_sliced_tiles(out_dir / "tiles", t_indices, c_indices, n_k)

        manifest = BatchManifest(
            n_timepoints=len(t_indices),
            n_channels=len(c_indices),
            n_tiles=n_k,
            timepoint_indices=t_indices,
            channel_indices=c_indices,
        )
        final = merge_batch_results(manifest, out_dir, verbose=False)

        node, attrs = self._read_tree(final)
        assert attrs["kind"] == "partition"
        assert isinstance(node, GSplatPartition)
        assert node.n_children == n_k
        assert total_splats(node) == total

        # No recipe → bare-leaf parts → no pipeline/ group (matches a plain fit).
        import zarr

        root = zarr.open_group(str(final), mode="r")
        assert "pipeline" not in root

        # Each part stacked its 2 timepoints → 4D, and the stacked time
        # coordinate (last appended axis) is the REAL index set {5, 72},
        # not the sequential {0, 1}.
        for child in node.children:
            for leaf in self._leaves(child):
                centers = leaf.additive_sublods[0].centers
                assert centers.shape[1] == 4  # promoted to 4D
                time_coords = np.unique(np.round(centers[:, -1]).astype(int))
                np.testing.assert_array_equal(time_coords, np.array([5, 72]))

    def test_flat_merge_resolves_sliced_real_indices(self, tmp_path: Path) -> None:
        """The legacy --flat path shares _tile_indices/_tile_path, so it must
        resolve the same real (5,72)/(1,3) filenames and conserve all splats."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData

        out_dir = tmp_path / "batch"
        t_indices = [5, 72]
        c_indices = [1, 3]
        n_k = 2
        total = self._write_sliced_tiles(out_dir / "tiles", t_indices, c_indices, n_k)

        manifest = BatchManifest(
            n_timepoints=len(t_indices),
            n_channels=len(c_indices),
            n_tiles=n_k,
            timepoint_indices=t_indices,
            channel_indices=c_indices,
        )
        final = merge_batch_results(manifest, out_dir, verbose=False, flat=True)

        merged = GSplatData.load(final)
        assert merged.n_splats == total
        assert merged.ndim == 4
        time_coords = np.unique(np.round(merged.centers[:, -1]).astype(int))
        np.testing.assert_array_equal(time_coords, np.array([5, 72]))

    # ----------------------------------------------------------------
    # Per-part LOD at merge (--recipe): each spatial tile-part gets its own
    # LOD ladder as it streams, closing the tiled-data LOD gap (the `lod`
    # CLI rejects a partition, so the canonical fit → lod chain can't add
    # LODs to tiled output otherwise). additive → partitioned topology;
    # substitutive → mosaic. Conservation is checked via node.n_splats (a
    # partition sums each child's rendered/finest count).
    # ----------------------------------------------------------------
    def test_merge_recipe_additive_makes_partition_of_ladders(
        self, tmp_path: Path
    ) -> None:
        """--recipe stream → kind=partition where each part is a leaf carrying
        an additive ladder (n_additive_sublods > 1); finest total conserved."""
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.tree import GSplatLeaf, GSplatPartition

        out_dir = tmp_path / "batch"
        # 6 splats/tile so a 3-bin additive ladder isn't clamped flat.
        n_k = 3
        total = 0
        seed = 0
        from luxar.gsplats.batch.manifest import output_filename

        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n_k):
            tile = self._tile(6, seed)
            tile.save(tiles_dir / output_filename(0, 0, k, 1, 1, n_k))
            total += tile.n_splats
            seed += 1

        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=n_k)
        final = merge_batch_results(manifest, out_dir, verbose=False, recipe="stream")

        node, attrs = self._read_tree(final)
        assert attrs["kind"] == "partition"
        assert isinstance(node, GSplatPartition)
        assert node.n_children == n_k
        for child in node.children:
            assert isinstance(child, GSplatLeaf)
            assert child.n_additive_sublods > 1  # a real ladder, not a flat leaf
        assert node.n_splats == total  # finest count conserved across parts

        # The recipe's reduction provenance is persisted in the pipeline/ group
        # (pre-fix, write_partition_streaming's pipeline_info was never passed).
        import zarr

        root = zarr.open_group(str(final), mode="r")
        assert "pipeline" in root, "merged partition is missing the pipeline/ group"
        pipe = dict(root["pipeline"].attrs)
        assert pipe["recipe"] == "stream"  # the recipe (build instruction)
        assert pipe["lod_kind"] == "additive"  # the reduction MECHANISM, not the recipe
        assert pipe["per_part"] is True
        assert pipe["n_lods"] == 4  # RecipeParams default

    def test_merge_recipe_levels_makes_adaptive(self, tmp_path: Path) -> None:
        """--recipe levels → kind=partition where each part is its own
        substitutive lod group (>= 2 levels); finest total conserved."""
        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

        out_dir = tmp_path / "batch"
        n_k = 2
        total = 0
        seed = 0
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n_k):
            tile = self._tile(64, seed)  # enough for K=4, L>=2 coarsening
            tile.save(tiles_dir / output_filename(0, 0, k, 1, 1, n_k))
            total += tile.n_splats
            seed += 1

        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=n_k)
        final = merge_batch_results(manifest, out_dir, verbose=False, recipe="levels")

        node, attrs = self._read_tree(final)
        assert attrs["kind"] == "partition"
        assert isinstance(node, GSplatPartition)
        assert node.n_children == n_k
        for child in node.children:
            assert isinstance(child, GSplatLodGroup)
            assert child.n_children >= 2  # coarse↔fine levels
        assert node.n_splats == total  # finest level per part summed

        # Substitutive reduction provenance (mirrors what `gsplat lod` persists)
        # lands in the pipeline/ group of the merged store.
        import zarr

        root = zarr.open_group(str(final), mode="r")
        assert "pipeline" in root, "merged partition is missing the pipeline/ group"
        pipe = dict(root["pipeline"].attrs)
        assert pipe["recipe"] == "levels"  # the recipe (build instruction)
        assert pipe["lod_kind"] == "substitutive"  # the reduction MECHANISM
        assert pipe["per_part"] is True
        assert pipe["compression_factor"] == 4  # RecipeParams defaults
        assert pipe["levels"] == 3
        assert pipe["conserve_mass"] is True
        assert pipe["refine"] == "none"
        # This manifest records no `spatial_shape`, so the merge cannot
        # establish the part width and refuses to invent one: the stamp stays
        # `None` (no provenance) rather than naming columns that may not exist.
        # A manifest that DOES record it publishes the explicit list — see
        # `test_merge_recipe_levels_stamps_the_part_coarsen_dims`.
        assert pipe["coarsen_dims"] is None

    def _gridded_tile(self, n: int, seed: int):
        """A tile whose LAST spatial axis sits on a coarse integer grid.

        The continuous `_tile` cannot show that the stamp is load-bearing: the
        writer's fallback finds no barrier on it either, so an unstamped store
        gets the same layout by accident. Three integral z planes over ``n``
        splats clear ``detect_barrier_dims``' ``n_unique * 4 <= n`` guard, so
        the fallback DOES flag axis 2 on the unreduced finest level while the
        merged levels (whose z the reduction averaged away) come out bare —
        the per-level mixture the stamp exists to replace.
        """
        from luxar.gsplats.gsplat_data import GSplatData

        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        centers = np.empty((n, 3), dtype=np.float32)
        centers[:, :2] = rng.uniform(0, 50, (n, 2))
        centers[:, 2] = rng.integers(0, 3, size=n) * 10.0
        return GSplatData(
            centers=centers,
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        )

    @pytest.mark.parametrize(
        ("n_t", "stamp", "barrier"),
        [
            # Single timepoint: the parts are 3D and the reduction coarsens all
            # three, so the complement is empty — a real "no barrier", and the
            # stamp is the ONLY thing that puts it there (the fallback would
            # barrier the gridded z axis on the finest level).
            (1, [0, 1, 2], []),
            # Stacked timepoints: the merge appends the time axis LAST and
            # `_finalize_part_node` keeps it out of the coarsening, so the stamp
            # names the spatial dims only and the complement is [3]. Here the
            # layout is nailed down twice over — `_merge_partition` also passes
            # the stacked axis as an authoritative `barrier_dims`, which wins in
            # `write_partition_streaming` — so this row pins that the two AGREE.
            # A stamp naming dim 3 would publish a provenance the store's own
            # layout contradicts.
            (2, [0, 1, 2], [3]),
        ],
        ids=["single-timepoint", "stacked-timepoints"],
    )
    def test_merge_recipe_levels_stamps_the_part_coarsen_dims(
        self, tmp_path: Path, n_t: int, stamp: "list[int]", barrier: "list[int]"
    ) -> None:
        """The per-part default is PUBLISHED, not left to the writer to guess.

        ``coarsen_dims: null`` is indistinguishable from an absent key to
        ``_barrier_from_coarsen_dims``, so the merge's own choice used to be
        withheld from the store and re-derived by auto-detection (#1600). The
        width comes from ``manifest.spatial_shape`` — the same field the
        explicit stacked-time ``barrier_dims`` is already taken from — so the
        record is available before the first tile is assembled.

        Asserted on the LAYOUT as well as the key: this is the one path where
        the width is INFERRED from the manifest rather than observed on a part,
        so an attr-only assertion would pass on a stamp the writer never acted
        on (#1600 review).
        """
        import zarr

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.io.ordering import detect_barrier_dims

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        n_k = 2
        for t in range(n_t):
            for k in range(n_k):
                self._gridded_tile(40, seed=t * n_k + k).save(
                    tiles_dir / output_filename(t, 0, k, n_t, 1, n_k)
                )
        # The control that makes the single-timepoint row mean something: on the
        # unreduced finest level the fallback flags the gridded axis, so the
        # expected `[]` cannot be reached by the guess the stamp displaces.
        assert detect_barrier_dims(
            np.asarray(self._gridded_tile(40, seed=0).centers)
        ) == [2]

        manifest = BatchManifest(
            n_timepoints=n_t, n_channels=1, n_tiles=n_k, spatial_shape=(50, 50, 50)
        )
        final = merge_batch_results(manifest, out_dir, verbose=False, recipe="levels")

        root = zarr.open_group(str(final), mode="r")
        assert list(dict(root["pipeline"].attrs)["coarsen_dims"]) == stamp

        found: "list[list[int]]" = []

        def walk(group) -> None:
            if "slice_dims" in group.attrs:
                found.append([int(d) for d in group.attrs["slice_dims"]])
            for name in group.group_keys():
                walk(group[name])

        walk(root)
        assert found, "no ordering attrs written at all"
        assert all(b == barrier for b in found), (
            f"the written ordering barrier {found} does not follow the stamped {stamp}"
        )

    def test_merge_recipe_levels_uses_manifest_floor_basis(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Reloaded tiles have no stats, so the manifest must supply the basis."""
        import luxar.gsplats.lod.recipes as recipes
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        out_dir = tmp_path / "batch"
        self._write_tiles(out_dir / "tiles", n_t=1, n_c=1, n_k=2)
        captured = []

        def fake_build(part, recipe, params, *, cell=None):
            captured.append(params.image_min)
            return part

        monkeypatch.setattr(recipes, "build_part_lod", fake_build)
        manifest = BatchManifest(
            n_timepoints=1,
            n_channels=1,
            n_tiles=2,
            floor_level=500.0,
            fit_args={"floor": 500.0},
        )
        merge_batch_results(manifest, out_dir, verbose=False, recipe="levels")

        assert captured == [500.0, 500.0]

    def test_merge_recipe_single_tile_emits_lod_not_partition(
        self, tmp_path: Path
    ) -> None:
        """K=1 + --recipe → a bare lod group / leaf-with-ladder, NOT a partition."""
        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.tree import GSplatLeaf

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        tile = self._tile(6, 0)
        tile.save(tiles_dir / output_filename(0, 0, 0, 1, 1, 1))

        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=1)
        final = merge_batch_results(manifest, out_dir, verbose=False, recipe="stream")
        node, attrs = self._read_tree(final)
        assert attrs.get("kind") != "partition"
        assert isinstance(node, GSplatLeaf)
        assert node.n_additive_sublods > 1  # the ladder was applied
        assert node.n_splats == tile.n_splats

        # The K=1 (write_gsplats_tree) branch carries the same pipeline/ group
        # as the streaming K>1 branch.
        import zarr

        root = zarr.open_group(str(final), mode="r")
        assert "pipeline" in root
        assert root["pipeline"].attrs["recipe"] == "stream"
        assert root["pipeline"].attrs["lod_kind"] == "additive"

    def test_merge_recipe_4d_substitutive_barriers_on_timepoint(
        self, tmp_path: Path
    ) -> None:
        """A 4D part (timepoints stacked) coarsened with --recipe levels
        must NOT blend across time: the stacked-timepoint axis is a barrier, so
        the coarsest level of each part still spans BOTH timepoints {0, 1}."""
        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

        out_dir = tmp_path / "batch"
        n_t, n_k = 2, 2
        total = 0
        seed = 0
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for t in range(n_t):
            for k in range(n_k):
                tile = self._tile(40, seed)
                tile.save(tiles_dir / output_filename(t, 0, k, n_t, 1, n_k))
                total += tile.n_splats
                seed += 1

        manifest = BatchManifest(n_timepoints=n_t, n_channels=1, n_tiles=n_k)
        final = merge_batch_results(manifest, out_dir, verbose=False, recipe="levels")
        node, _ = self._read_tree(final)
        assert isinstance(node, GSplatPartition)
        assert node.n_splats == total
        for child in node.children:
            assert isinstance(child, GSplatLodGroup)
            assert child.ndim == 4
            # Coarsest level (index 0, coarsest→finest) still spans both
            # timepoints — coarsening did not merge across the time barrier.
            # The level carries a default additive ladder, so check the UNION
            # of its sub-LOD prefix chunks (a single chunk holds only the
            # first splats in additive order, not the whole level).
            coarse = child.children[0]
            tvals = np.concatenate([s.centers[:, -1] for s in coarse.additive_sublods])
            tcoords = np.unique(np.round(tvals).astype(int))
            np.testing.assert_array_equal(tcoords, np.array([0, 1]))

    def test_merge_recipe_rejects_flat_and_composed(self, tmp_path: Path) -> None:
        """--recipe is mutually exclusive with --flat, and composed recipes
        (which re-partition) are rejected per-part."""
        import pytest

        from luxar.gsplats.batch.manifest import BatchManifest, output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for k in range(2):
            self._tile(6, k).save(tiles_dir / output_filename(0, 0, k, 1, 1, 2))
        manifest = BatchManifest(n_timepoints=1, n_channels=1, n_tiles=2)

        with pytest.raises(ValueError, match="mutually exclusive"):
            merge_batch_results(
                manifest, out_dir, verbose=False, flat=True, recipe="additive"
            )
        with pytest.raises(ValueError, match="not supported"):
            merge_batch_results(manifest, out_dir, verbose=False, recipe="tiles")

    def test_merge_recipe_params_normalises_hyphenated_method(self) -> None:
        """`--substitutive-method kmeans-lloyd` (the documented spelling) must map
        to the canonical `kmeans_lloyd`, like the `gsplat lod` command — else it
        reaches make_substitutive_lod invalid and raises. Fails pre-fix (the raw
        hyphenated string was passed through verbatim)."""
        from luxar.cli.gsplat_ops.batch.recipe_args import (
            build_merge_recipe_params as _build_merge_recipe_params,
        )

        # CLI-supplied value.
        p = _build_merge_recipe_params(
            {},
            n_lods=None,
            compression_factor=None,
            levels=None,
            substitutive_method="kmeans-lloyd",
            coarsen_dims=None,
        )
        assert p.substitutive_method == "kmeans_lloyd"

        # Plan-recorded value (manifest.merge_recipe_args, string-valued).
        p2 = _build_merge_recipe_params(
            {"substitutive-method": "greedy-lloyd"},
            n_lods=None,
            compression_factor=None,
            levels=None,
            substitutive_method=None,
            coarsen_dims=None,
        )
        assert p2.substitutive_method == "greedy_lloyd"

    def test_batch_merge_cli_parses_slurm_emitted_recipe_flags(
        self, tmp_path: Path
    ) -> None:
        """Round-trip the Slurm path: the flags `generate_merge_sbatch` emits for
        a planned per-part recipe must parse back through the real `batch merge`
        Typer command and produce a kind=partition of LOD'd parts.

        Guards the write↔read seam (slurm emit ↔ CLI parse) AND the hyphenated
        `--substitutive-method` normalisation end-to-end.
        """
        import shlex

        from typer.testing import CliRunner

        from luxar.cli.gsplat_ops.batch.commands import app_batch
        from luxar.gsplats.batch.manifest import (
            BatchManifest,
            output_filename,
            save_manifest,
        )
        from luxar.gsplats.batch.slurm_gen import generate_merge_sbatch
        from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        n_k = 2
        total = 0
        for k in range(n_k):
            tile = self._tile(64, k)
            tile.save(tiles_dir / output_filename(0, 0, k, 1, 1, n_k))
            total += tile.n_splats

        # Plan records a substitutive per-part recipe with a HYPHENATED method.
        manifest = BatchManifest(
            output_dir=str(out_dir),
            n_timepoints=1,
            n_channels=1,
            n_tiles=n_k,
            merge_recipe="substitutive",
            merge_recipe_args={
                "compression-factor": "4",
                "levels": "2",
                "substitutive-method": "kmeans-lloyd",
                "coarsen-dims": "0,1,2",
            },
        )
        save_manifest(manifest, out_dir)

        # Extract the exact `luxar gsplat batch-fit merge ...` line the Slurm job runs.
        script = generate_merge_sbatch(manifest, "# env\n")
        merge_line = next(ln for ln in script.splitlines() if "batch-fit merge" in ln)
        tokens = shlex.split(merge_line)
        merge_idx = tokens.index("merge")
        # app_batch is the `batch-fit` sub-app, so keep `merge` as its subcommand;
        # drop only the `luxar gsplat batch-fit` prefix.
        cli_args = tokens[merge_idx:]
        # Point the (absolute) output_dir arg at the tmp dir (already is).
        # The manifest stores the legacy spellings — of BOTH the recipe name and
        # the method flag — and the sbatch generator must emit the canonical form
        # of each, because the merge CLI rejects legacy spellings. The method half
        # guards a runtime break: `slurm_gen` builds the flag as f"--{stored_key}",
        # so without emit-time translation a pre-rename manifest emitted
        # `--substitutive-method` and every Slurm merge job died on an unknown
        # option — invisible to every unit test that did not read the sbatch line.
        assert "--recipe" in cli_args and "levels" in cli_args
        assert "substitutive" not in cli_args
        assert "--subst-method" in cli_args and "kmeans-lloyd" in cli_args
        assert "--coarsen-dims" in cli_args and "0,1,2" in cli_args

        result = CliRunner().invoke(app_batch, cli_args)
        assert result.exit_code == 0, result.output

        node, attrs = self._read_tree(out_dir / "merged" / "final.gsplats.zarr")
        assert attrs["kind"] == "partition"
        assert isinstance(node, GSplatPartition)
        assert node.n_children == n_k
        for child in node.children:
            assert isinstance(child, GSplatLodGroup)  # substitutive → mosaic
        assert node.n_splats == total

    def test_merge_recipe_params_rejects_invalid_substitutive_method(self) -> None:
        """An unknown --substitutive-method is rejected UP FRONT with a clean
        typer.BadParameter (mirroring `gsplat lod`), not deferred to a deep
        make_substitutive_lod ValueError. Fails pre-fix (no validation → the bad
        string was wrapped into RecipeParams unchecked)."""
        import typer

        from luxar.cli.gsplat_ops.batch.recipe_args import (
            build_merge_recipe_params as _build_merge_recipe_params,
        )

        with pytest.raises(typer.BadParameter, match="subst-method"):
            _build_merge_recipe_params(
                {},
                n_lods=None,
                compression_factor=None,
                levels=None,
                substitutive_method="kmeans-typo",
                coarsen_dims=None,
            )

    def test_merge_recipe_params_rejects_non_numeric_coarsen_dims(self) -> None:
        """Bad --coarsen-dims tokens raise a clean typer.BadParameter, not a raw
        ValueError surfaced as a traceback. Fails pre-fix (unwrapped int())."""
        import typer

        from luxar.cli.gsplat_ops.batch.recipe_args import (
            build_merge_recipe_params as _build_merge_recipe_params,
        )

        with pytest.raises(typer.BadParameter, match="coarsen-dims"):
            _build_merge_recipe_params(
                {},
                n_lods=None,
                compression_factor=None,
                levels=None,
                substitutive_method=None,
                coarsen_dims="0,x,2",
            )

    def test_batch_merge_invalid_method_writes_no_output(self, tmp_path: Path) -> None:
        """An invalid --subst-method must fail BEFORE the streaming writer
        overwrites final.gsplats.zarr — otherwise a corrected re-run (without
        --force) would silently skip the broken stub. Asserts non-zero exit AND
        that no output file was created. Fails pre-fix (deep raise left a stub).

        The flag SPELLING is load-bearing here even though the assertion is only
        `exit_code != 0`: left at the pre-2026-08 `--substitutive-method`, typer
        would reject the unknown option and the test would pass without ever
        reaching the validation it exists to pin — green for the wrong reason.
        """
        from typer.testing import CliRunner

        from luxar.cli.gsplat_ops.batch.commands import app_batch
        from luxar.gsplats.batch.manifest import (
            BatchManifest,
            output_filename,
            save_manifest,
        )

        out_dir = tmp_path / "batch"
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for k in range(2):
            self._tile(32, k).save(tiles_dir / output_filename(0, 0, k, 1, 1, 2))
        save_manifest(
            BatchManifest(
                output_dir=str(out_dir), n_timepoints=1, n_channels=1, n_tiles=2
            ),
            out_dir,
        )

        result = CliRunner().invoke(
            app_batch,
            [
                "merge",
                str(out_dir),
                "--recipe",
                "levels",
                "--subst-method",
                "kmeans-typo",
            ],
        )
        assert result.exit_code != 0
        assert not (out_dir / "merged" / "final.gsplats.zarr").exists()


class TestContentSubmitDryRun:
    """End-to-end wiring of `batch-fit submit --tiling content` (no Slurm needed):
    it builds the shared box plan from a representative (t,c) and would fan it
    across the array. --dry-run writes plan.json during planning, then exits."""

    def test_content_dry_run_writes_plan_and_reports_boxes(
        self, tmp_path: Path
    ) -> None:
        import zarr
        from typer.testing import CliRunner

        from luxar.cli.gsplat_commands import app_gsplat
        from luxar.gsplats.planner import FitPlan

        # Synthetic 3D (ZYX) volume with blobs so the planner finds content.
        rng = np.random.default_rng(0)
        V = np.zeros((64, 64, 64), np.float32)
        zz, yy, xx = np.mgrid[0:64, 0:64, 0:64]
        for _ in range(20):
            cz, cy, cx = rng.integers(6, 58, 3)
            V += np.exp(
                -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
            ).astype(np.float32)
        V = np.clip(V, 0, 1)
        src = tmp_path / "vol.zarr"
        z = zarr.open_array(
            str(src), mode="w", shape=V.shape, chunks=(32, 32, 32), dtype="f4"
        )
        z[:] = V

        out = tmp_path / "batch_out"
        runner = CliRunner()
        res = runner.invoke(
            app_gsplat,
            [
                "batch-fit",
                "submit",
                str(src),
                str(out),
                "-p",
                "gpu",
                "--tiling",
                "content",
                "--axes",
                "z,y,x",
                "--k-star-ref",
                "4000",
                "--n-features-ref",
                "200",
                "--feature-threshold",
                "0.1",
                "--feature-metric",
                "peaks",
                "--cell",
                "8",
                "--min-leaf",
                "16",
                "--max-leaf",
                "32",
                "--dry-run",
            ],
        )
        assert res.exit_code == 0, res.output
        # plan.json is written during planning (before the dry-run exit) and is
        # a usable FitPlan — the shared plan every array task would consume.
        plan_json = out / "plan.json"
        assert plan_json.exists()
        plan = FitPlan.from_json(plan_json)
        assert plan.n_boxes >= 1 and plan.total_budget > 0
        # the summary reflects content mode (boxes, not tiles)
        assert "content plan" in res.output.lower()
        assert "boxes" in res.output.lower()

    def test_content_plan_threads_axes_into_the_plan_read(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The content-plan scan must read with the user's --axes spec, not the
        positional heuristic — else an --axes dataset is scanned with a wrong-
        shaped/ordered volume and the box plan is wrong. Every plan read goes
        through `planning._pinned_slice_volume`, which pins the slice by LABEL,
        so the spec has to arrive there: as the `axes` string AND as the
        `axes_labels` the pin is actually built from."""
        import zarr
        from typer.testing import CliRunner

        import luxar.cli.gsplat_ops.batch.planning as planning
        from luxar.cli.gsplat_commands import app_gsplat

        rng = np.random.default_rng(1)
        V = np.zeros((64, 64, 64), np.float32)
        zz, yy, xx = np.mgrid[0:64, 0:64, 0:64]
        for _ in range(20):
            cz, cy, cx = rng.integers(6, 58, 3)
            V += np.exp(
                -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
            ).astype(np.float32)
        V = np.clip(V, 0, 1)
        src = tmp_path / "vol.zarr"
        z = zarr.open_array(
            str(src), mode="w", shape=V.shape, chunks=(32, 32, 32), dtype="f4"
        )
        z[:] = V

        seen_reads: list = []
        real_read = planning._pinned_slice_volume

        def _spy(*a, **kw):
            seen_reads.append((kw.get("axes"), kw.get("axes_labels")))
            return real_read(*a, **kw)

        monkeypatch.setattr(planning, "_pinned_slice_volume", _spy)

        out = tmp_path / "batch_out"
        res = CliRunner().invoke(
            app_gsplat,
            [
                "batch-fit",
                "submit",
                str(src),
                str(out),
                "-p",
                "gpu",
                "--tiling",
                "content",
                "--axes",
                "z,y,x",
                "--k-star-ref",
                "4000",
                "--n-features-ref",
                "200",
                "--feature-threshold",
                "0.1",
                "--feature-metric",
                "peaks",
                "--cell",
                "8",
                "--min-leaf",
                "16",
                "--max-leaf",
                "32",
                "--dry-run",
            ],
        )
        assert res.exit_code == 0, res.output
        assert seen_reads, (
            "the plan never read a slice through _pinned_slice_volume (the read "
            "seam moved, so this test can no longer see what the plan asked for)"
        )
        assert all(axes == "z,y,x" for axes, _ in seen_reads), (
            f"a content-plan read did not carry axes='z,y,x'; saw {seen_reads} "
            "(the --axes spec was not threaded into the plan scan)"
        )
        assert all(labels == ["z", "y", "x"] for _, labels in seen_reads), (
            f"a content-plan read did not carry axes_labels=['z', 'y', 'x']; saw "
            f"{seen_reads} (the pin is built from those labels, so the plan would "
            "slice by the positional heuristic instead of the user's --axes)"
        )


class TestContentMerge:
    """The content fan-out writes `_box{k}` outputs; the merge must look for that
    label (not the default `_tile{k}`) and skip legitimately-empty boxes."""

    @staticmethod
    def _write_box(path: Path, n: int, offset: float) -> None:
        rng = np.random.default_rng(int(offset))
        centers = (offset + rng.uniform(0, 8, size=(n, 3))).astype(np.float32)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        from luxar.gsplats.io.save_gsplats import save_gsplats

        save_gsplats(
            str(path),
            centers=centers,
            amplitudes=rng.uniform(0.2, 1.0, size=(n,)).astype(np.float32),
            cholesky_factors=chol,
        )

    def _content_manifest(self, out: Path, n_boxes: int, **kwargs: Any) -> "object":
        from luxar.gsplats.batch.manifest import (
            BatchJob,
            BatchManifest,
            output_filename,
        )

        jobs = [
            BatchJob(
                task_id=k,
                timepoint=0,
                channel=0,
                tile_index=k,
                output_filename=output_filename(0, 0, k, 1, 1, n_boxes, label="box"),
                estimated_wall_seconds=1.0,
            )
            for k in range(n_boxes)
        ]
        m = BatchManifest(
            input_path="/data/x.zarr",
            output_dir=str(out),
            n_timepoints=1,
            n_channels=1,
            spatial_shape=(64, 64, 64),
            mode="content",
            n_tiles=n_boxes,
            plan_path=str(out / "plan.json"),
            total_tasks=n_boxes,
            slurm_partition="gpu",
            **kwargs,
        )
        m.jobs = jobs
        return m

    def test_content_split_planes_follow_physical_grid_scale(
        self, tmp_path: Path
    ) -> None:
        """The content plan stays voxel-authored while its merge tree is physical."""
        from luxar.gsplats.batch.merge_orchestrator import _slot_bsp_tree
        from luxar.gsplats.planner.spec import FitPlan, PlanBox

        out = tmp_path / "batch"
        out.mkdir()
        FitPlan(
            volume_shape=[20, 20, 20],
            boxes=[
                PlanBox(box=[0, 10, 0, 20, 0, 20], n_features=1, budget=1),
                PlanBox(box=[10, 20, 0, 20, 0, 20], n_features=1, budget=1),
            ],
            overlap=0,
            feature_method="peaks",
            min_leaf=4,
            max_leaf=16,
            bsp_tree={
                "axis": 0,
                "split": 10.0,
                "left": {"part": 0},
                "right": {"part": 1},
            },
        ).to_json(out / "plan.json")

        tree = _slot_bsp_tree(
            self._content_manifest(out, 2, grid_scale=[5.0, 2.0, 1.0]),
            out,
            False,
        )

        assert tree == {
            "axis": 0,
            "split": 50.0,
            "left": {"part": 0},
            "right": {"part": 1},
        }

    def test_content_merge_finds_box_outputs_and_skips_empty(
        self, tmp_path: Path
    ) -> None:
        """Merge assembles a kind=partition from `_box{k}` outputs (the label bug
        would FileNotFound on `_tile{k}`), and an empty box (`.empty` marker, no
        store) is skipped rather than crashing the merge."""
        from luxar.gsplats.batch.manifest import output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import GSplatPartition, total_splats

        out = tmp_path / "batch"
        tiles = out / "tiles"
        tiles.mkdir(parents=True)

        n_boxes = 3
        # boxes 0 and 2 have splats; box 1 is legitimately empty (.empty marker).
        self._write_box(
            tiles / output_filename(0, 0, 0, 1, 1, n_boxes, label="box"), 30, 0.0
        )
        (
            tiles / (output_filename(0, 0, 1, 1, 1, n_boxes, label="box") + ".empty")
        ).write_text("")
        self._write_box(
            tiles / output_filename(0, 0, 2, 1, 1, n_boxes, label="box"), 40, 100.0
        )

        manifest = self._content_manifest(out, n_boxes)
        final = merge_batch_results(manifest, out, verbose=False)

        node, _ = load_gsplat_node(final)
        assert isinstance(node, GSplatPartition)
        assert node.n_children == 2  # the empty box was skipped
        assert total_splats(node) == 70

    def test_content_merge_carries_plan_split_planes_renumbered(
        self, tmp_path: Path
    ) -> None:
        """The merge stamps the plan's split planes, RENUMBERED past skipped boxes.

        Slot indices and written part indices diverge the moment a box is empty, so
        a tree carried over verbatim would name parts that do not exist (or, worse,
        the wrong ones — still a valid permutation, so it fails silently). Here box
        1 is empty: the 3-leaf plan tree must arrive as a 2-leaf tree labelled
        ``0``/``1``, and its plane must still separate the two surviving parts.
        """
        from luxar.gsplats.batch.manifest import output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.planner.spec import FitPlan, PlanBox
        from luxar.gsplats.tree import GSplatPartition

        out = tmp_path / "batch"
        tiles = out / "tiles"
        tiles.mkdir(parents=True)

        n_boxes = 3
        # Boxes 0 and 2 hold splats (near 0 and near 100); box 1 is empty.
        self._write_box(
            tiles / output_filename(0, 0, 0, 1, 1, n_boxes, label="box"), 30, 0.0
        )
        (
            tiles / (output_filename(0, 0, 1, 1, 1, n_boxes, label="box") + ".empty")
        ).write_text("")
        self._write_box(
            tiles / output_filename(0, 0, 2, 1, 1, n_boxes, label="box"), 40, 100.0
        )

        # A plan whose tree splits box 0 off below 50, then 1 from 2 above it.
        plan = FitPlan(
            volume_shape=[128, 128, 128],
            boxes=[
                PlanBox(box=[0, 50, 0, 128, 0, 128], n_features=1, budget=1),
                PlanBox(box=[50, 75, 0, 128, 0, 128], n_features=1, budget=1),
                PlanBox(box=[75, 128, 0, 128, 0, 128], n_features=1, budget=1),
            ],
            overlap=0,
            feature_method="peaks",
            min_leaf=16,
            max_leaf=64,
            bsp_tree={
                "axis": 0,
                "split": 50.0,
                "left": {"part": 0},
                "right": {
                    "axis": 0,
                    "split": 75.0,
                    "left": {"part": 1},
                    "right": {"part": 2},
                },
            },
        )
        out.mkdir(parents=True, exist_ok=True)
        plan.to_json(out / "plan.json")

        manifest = self._content_manifest(out, n_boxes)
        final = merge_batch_results(manifest, out, verbose=False)

        node, _ = load_gsplat_node(final)
        assert isinstance(node, GSplatPartition)
        assert node.n_children == 2
        tree = node.bsp_tree
        assert tree is not None, "merged partition lost the plan's split planes"
        # The subtree that held boxes 1 and 2 collapsed to the one survivor, so the
        # 75.0 plane is gone and part 2 has been renumbered to child_index 1.
        assert tree == {
            "axis": 0,
            "split": 50.0,
            "left": {"part": 0},
            "right": {"part": 1},
        }
        # ...and that plane really does separate the two written parts.
        lo = np.concatenate(
            [np.asarray(s.centers) for s in node.children[0].additive_sublods]
        )
        hi = np.concatenate(
            [np.asarray(s.centers) for s in node.children[1].additive_sublods]
        )
        assert lo[:, 0].max() < 50.0 <= hi[:, 0].min()

    def test_content_merge_skips_split_planes_when_plan_disagrees(
        self, tmp_path: Path
    ) -> None:
        """A plan whose box count contradicts the manifest yields NO tree.

        Mislabelled planes are worse than absent ones: the traversal still returns
        a valid permutation, so the ordering silently points at the wrong parts
        instead of falling back to the centroid heuristic.
        """
        from luxar.gsplats.batch.manifest import output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.planner.spec import FitPlan, PlanBox

        out = tmp_path / "batch"
        tiles = out / "tiles"
        tiles.mkdir(parents=True)
        n_boxes = 2
        for k, off in enumerate((0.0, 100.0)):
            self._write_box(
                tiles / output_filename(0, 0, k, 1, 1, n_boxes, label="box"), 20, off
            )
        # One box too many vs the manifest's n_tiles=2.
        FitPlan(
            volume_shape=[128, 128, 128],
            boxes=[
                PlanBox(box=[0, 43, 0, 128, 0, 128], n_features=1, budget=1),
                PlanBox(box=[43, 86, 0, 128, 0, 128], n_features=1, budget=1),
                PlanBox(box=[86, 128, 0, 128, 0, 128], n_features=1, budget=1),
            ],
            overlap=0,
            feature_method="peaks",
            min_leaf=16,
            max_leaf=64,
            bsp_tree={
                "axis": 0,
                "split": 43.0,
                "left": {"part": 0},
                "right": {
                    "axis": 0,
                    "split": 86.0,
                    "left": {"part": 1},
                    "right": {"part": 2},
                },
            },
        ).to_json(out / "plan.json")

        manifest = self._content_manifest(out, n_boxes)
        node, _ = load_gsplat_node(merge_batch_results(manifest, out, verbose=False))
        assert node.bsp_tree is None

    def test_content_merge_missing_box_raises(self, tmp_path: Path) -> None:
        """A box with neither a store nor an `.empty` marker (the task never ran)
        still raises — distinct from a legitimately-empty box."""
        from luxar.gsplats.batch.manifest import output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        out = tmp_path / "batch"
        tiles = out / "tiles"
        tiles.mkdir(parents=True)
        n_boxes = 2
        self._write_box(
            tiles / output_filename(0, 0, 0, 1, 1, n_boxes, label="box"), 30, 0.0
        )
        # box 1 absent entirely
        manifest = self._content_manifest(out, n_boxes)
        with pytest.raises(FileNotFoundError):
            merge_batch_results(manifest, out, verbose=False)

    def _mc_manifest(self, out: Path, n_boxes: int, n_channels: int) -> "object":
        from luxar.gsplats.batch.manifest import (
            BatchJob,
            BatchManifest,
            output_filename,
        )

        jobs = []
        tid = 0
        for c in range(n_channels):
            for k in range(n_boxes):
                jobs.append(
                    BatchJob(
                        task_id=tid,
                        timepoint=0,
                        channel=c,
                        tile_index=k,
                        output_filename=output_filename(
                            0, c, k, 1, n_channels, n_boxes, label="box"
                        ),
                        estimated_wall_seconds=1.0,
                    )
                )
                tid += 1
        m = BatchManifest(
            input_path="/data/x.zarr",
            output_dir=str(out),
            n_timepoints=1,
            n_channels=n_channels,
            spatial_shape=(64, 64, 64),
            mode="content",
            n_tiles=n_boxes,
            plan_path=str(out / "plan.json"),
            total_tasks=n_boxes * n_channels,
            slurm_partition="gpu",
        )
        m.jobs = jobs
        return m

    def test_content_merge_channel_colors_skips_empty_channel(
        self, tmp_path: Path
    ) -> None:
        """A box empty in ONLY ONE channel must not crash --channel-colors merge:
        the colors are subset to the surviving channels (PR-1 empty-slot skip vs
        the fixed-length color list)."""
        from luxar.gsplats.batch.manifest import output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import GSplatPartition

        out = tmp_path / "batch"
        tiles = out / "tiles"
        tiles.mkdir(parents=True)
        n_boxes, n_c = 2, 2

        def fn(c: int, k: int) -> str:
            return output_filename(0, c, k, 1, n_c, n_boxes, label="box")

        # box 0: present in both channels; box 1: empty in c0, present in c1.
        self._write_box(tiles / fn(0, 0), 20, 0.0)
        self._write_box(tiles / fn(1, 0), 25, 50.0)
        (tiles / (fn(0, 1) + ".empty")).write_text("")
        self._write_box(tiles / fn(1, 1), 30, 100.0)

        manifest = self._mc_manifest(out, n_boxes, n_c)
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        # Pre-fix this raised ValueError (per_channel len 1 != 2 colors).
        final = merge_batch_results(manifest, out, channel_colors=colors, verbose=False)
        node, _ = load_gsplat_node(final)
        assert isinstance(node, GSplatPartition)
        assert node.n_children == 2  # both boxes present (box 1 via its c1 splats)

    def test_single_channel_colors_keep_fitted_not_tinted(self, tmp_path: Path) -> None:
        """n_c=1 + --channel-colors must NOT tint — fitted colors are preserved,
        matching the --flat path's `n_c > 1` gate. Pre-fix the partition path
        applied `merge_with_channel_colors` for a single channel too, overwriting
        every splat with the lone channel color (a flat-vs-partition divergence)."""
        from luxar.gsplats.batch.manifest import output_filename
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import iter_leaves

        out = tmp_path / "batch"
        tiles = out / "tiles"
        tiles.mkdir(parents=True)
        n_boxes = 2
        for k in range(n_boxes):
            self._write_box(
                tiles / output_filename(0, 0, k, 1, 1, n_boxes, label="box"),
                20,
                float(k * 50),
            )
        manifest = self._content_manifest(out, n_boxes)  # single channel
        final = merge_batch_results(
            manifest, out, channel_colors=[(1.0, 0.0, 0.0)], verbose=False
        )
        node, _ = load_gsplat_node(final)
        # _write_box leaves colors=None; a single channel must keep that (no tint).
        # Pre-fix, merge_with_channel_colors ran for n_c=1 and wrote a red color
        # array onto every part.
        for leaf in iter_leaves(node):
            gd = GSplatData.from_tree(leaf)
            assert gd.colors is None, (
                "single-channel merge tinted splats with the channel color — it "
                "should preserve the fitted colors (matching the flat n_c>1 gate)"
            )

    def test_flat_rejected_for_content_mode(self, tmp_path: Path) -> None:
        """`--flat` on a content batch is rejected: content places boxes once from a
        representative timepoint, so a (t,c) slot can be empty across every box and
        the flat 3-level fan-in (dense-grid assumption) would crash. Pre-fix this
        reached _merge_flat and failed with an unrelated error."""
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        out = tmp_path / "batch"
        (out / "tiles").mkdir(parents=True)
        manifest = self._content_manifest(out, n_boxes=2)
        with pytest.raises(ValueError, match="not supported for content"):
            merge_batch_results(manifest, out, flat=True, verbose=False)

    def test_status_counts_empty_box_as_completed(self, tmp_path: Path) -> None:
        """An empty content box (`.empty` marker) is COMPLETED, not failed/unknown."""
        from luxar.gsplats.batch.manifest import output_filename, save_manifest
        from luxar.gsplats.batch.status import (
            check_batch_status,
            format_status_report,
        )

        out = tmp_path / "batch"
        tiles = out / "tiles"
        tiles.mkdir(parents=True)
        n_boxes = 2
        self._write_box(
            tiles / output_filename(0, 0, 0, 1, 1, n_boxes, label="box"), 30, 0.0
        )
        (
            tiles / (output_filename(0, 0, 1, 1, 1, n_boxes, label="box") + ".empty")
        ).write_text("")
        manifest = self._content_manifest(out, n_boxes)
        save_manifest(manifest, out)

        st = check_batch_status(out)
        assert st.completed == 2  # store + empty-marker both count as completed
        assert st.failed == 0
        assert st.unknown == 0

        # format_status_report takes (status, manifest) positionally — no
        # verbose kwarg (the dead `-v` flag was removed).
        report = format_status_report(st, manifest)
        assert "2/2 completed" in report

    def test_validate_reports_empty_box_separately(self, tmp_path: Path) -> None:
        """`batch-fit validate` counts an empty box as EMPTY, not MISSING."""
        from typer.testing import CliRunner

        from luxar.cli.gsplat_commands import app_gsplat
        from luxar.gsplats.batch.manifest import output_filename, save_manifest

        out = tmp_path / "batch"
        tiles = out / "tiles"
        tiles.mkdir(parents=True)
        n_boxes = 2
        self._write_box(
            tiles / output_filename(0, 0, 0, 1, 1, n_boxes, label="box"), 30, 0.0
        )
        (
            tiles / (output_filename(0, 0, 1, 1, 1, n_boxes, label="box") + ".empty")
        ).write_text("")
        save_manifest(self._content_manifest(out, n_boxes), out)

        res = CliRunner().invoke(app_gsplat, ["batch-fit", "validate", str(out)])
        assert res.exit_code == 0, res.output
        assert "EMPTY:      1" in res.output
        assert "MISSING:    0" in res.output

    def test_invalid_tiling_value_rejected(self, tmp_path: Path) -> None:
        """An unknown --tiling value fails loudly (not a silent fall-through to
        uniform that would submit a large array in the wrong mode)."""
        import zarr
        from typer.testing import CliRunner

        from luxar.cli.gsplat_commands import app_gsplat

        src = tmp_path / "vol.zarr"
        z = zarr.open_array(str(src), mode="w", shape=(8, 8, 8), dtype="f4")
        z[:] = 0.0
        res = CliRunner().invoke(
            app_gsplat,
            [
                "batch-fit",
                "submit",
                str(src),
                str(tmp_path / "o"),
                "-p",
                "gpu",
                "--tiling",
                "bogus",
                "--axes",
                "z,y,x",
                "--dry-run",
            ],
        )
        assert res.exit_code != 0
        assert "uniform|content" in res.output


class TestSubmitRecipeValidation:
    """batch-fit submit --merge-recipe must reject cross-recipe knobs (fail-fast,
    matching `fit --recipe` and `gsplat lod`) instead of silently dropping them."""

    def test_merge_recipe_rejects_cross_recipe_knob(self, tmp_path: Path) -> None:
        import zarr
        from typer.testing import CliRunner

        from luxar.cli.gsplat_commands import app_gsplat

        src = tmp_path / "vol.zarr"
        z = zarr.open_array(str(src), mode="w", shape=(32, 32, 32), dtype="f4")
        z[:] = 0.0
        # stream merge recipe + a levels-only knob -> rejected before submit
        res = CliRunner().invoke(
            app_gsplat,
            [
                "batch-fit",
                "submit",
                str(src),
                str(tmp_path / "o"),
                "-p",
                "gpu",
                "--tile-size",
                "24",
                "--overlap",
                "4",
                "--axes",
                "z,y,x",
                "--dry-run",
                "--merge-recipe",
                "stream",
                "--merge-compression-factor",
                "4",
            ],
        )
        assert res.exit_code != 0
        # Strip ANSI: on CI (and any color-capable terminal) typer/Rich colorizes
        # the error panel, inserting escape codes *inside* the option name
        # (`\x1b[…m--merge\x1b[…m-compression-factor`), which breaks a naive
        # contiguous-substring check. Local runs without color pass either way.
        import re

        clean = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
        assert "--merge-compression-factor" in clean
        assert "not used" in clean.lower()


class TestMergeRecipeAdditiveKnobs:
    """batch-fit merge/submit reach parity with fit --recipe / gsplat lod: the
    additive ladder method/breakpoints and the lod-method are tunable."""

    def test_build_merge_recipe_params_additive_knobs(self) -> None:
        from luxar.cli.gsplat_ops.batch.recipe_args import (
            build_merge_recipe_params as _build_merge_recipe_params,
        )

        # stored (plan-time) values; CLI overrides None → use stored.
        params = _build_merge_recipe_params(
            {
                "add-method": "self_energy",
                "breakpoints": "counts:100,500",
                "n-lods": "5",
            },
            n_lods=None,
            additive_method=None,
            breakpoints=None,
            compression_factor=None,
            levels=None,
            substitutive_method=None,
            coarsen_dims=None,
        )
        assert params.additive_method == "self_energy"
        assert params.breakpoints == [100, 500]
        assert params.n_lods == 5

    def test_pre_rename_manifest_tokens_still_resolve(self) -> None:
        """A manifest from before the 2026-08 method-flag rename must still resolve.

        The READ side of the same hazard the sbatch-emit test guards. Stored plan
        tokens are dash-less (`substitutive-method`), and the reader now looks up
        the new spelling — so without normalisation an in-flight batch run resumed
        after the rename would MISS its planned methods and silently fall back to
        the recipe defaults (`auto`), quietly doing different work than planned
        rather than failing. Both legacy tokens are checked because the additive and
        substitutive halves are normalised by one pass and a per-key fix would be
        easy to apply to only one.
        """
        from luxar.cli.gsplat_ops.batch.recipe_args import (
            build_merge_recipe_params as _build_merge_recipe_params,
        )

        params = _build_merge_recipe_params(
            {
                "additive-method": "self_energy",
                "substitutive-method": "kmeans_lloyd",
            },
            n_lods=None,
            additive_method=None,
            breakpoints=None,
            compression_factor=None,
            levels=None,
            substitutive_method=None,
            coarsen_dims=None,
        )

        assert params.additive_method == "self_energy"
        assert params.substitutive_method == "kmeans_lloyd"

    def test_build_merge_recipe_params_validation(self) -> None:
        # a bad additive method fails fast
        import typer

        from luxar.cli.gsplat_ops.batch.recipe_args import (
            build_merge_recipe_params as _build_merge_recipe_params,
        )

        with pytest.raises(typer.BadParameter):
            _build_merge_recipe_params(
                {},
                n_lods=None,
                additive_method="nope",
                breakpoints=None,
                compression_factor=None,
                levels=None,
                substitutive_method=None,
                coarsen_dims=None,
            )

    def test_submit_threads_additive_knobs_into_sbatch(self, tmp_path: Path) -> None:
        """`batch-fit submit --merge-recipe stream --merge-add-method ...`
        records the knobs in the manifest and the merge sbatch invokes them."""
        import zarr
        from typer.testing import CliRunner

        from luxar.cli.gsplat_commands import app_gsplat
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.batch.slurm_gen import generate_merge_sbatch

        src = tmp_path / "vol.zarr"
        z = zarr.open_array(str(src), mode="w", shape=(32, 32, 32), dtype="f4")
        z[:] = 0.0
        out = tmp_path / "o"
        # Mock sbatch so the (non-dry-run) submit writes the manifest without Slurm.
        from unittest.mock import patch

        class _R:
            returncode = 0
            stdout = "123"
            stderr = ""

        with patch("subprocess.run", return_value=_R()):
            res = CliRunner().invoke(
                app_gsplat,
                [
                    "batch-fit",
                    "submit",
                    str(src),
                    str(out),
                    "-p",
                    "gpu",
                    "--tile-size",
                    "24",
                    "--overlap",
                    "4",
                    "--axes",
                    "z,y,x",
                    "--merge-recipe",
                    "stream",
                    "--merge-add-method",
                    "self_energy",
                    "--merge-breakpoints",
                    "energy:0.5,1.0",
                ],
            )
        assert res.exit_code == 0, res.output
        manifest = load_manifest(out)
        assert manifest.merge_recipe_args.get("add-method") == "self_energy"
        assert manifest.merge_recipe_args.get("breakpoints") == "energy:0.5,1.0"
        script = generate_merge_sbatch(manifest, "")
        assert "--add-method self_energy" in script
        assert "--breakpoints energy:0.5,1.0" in script

    def test_submit_rejects_malformed_merge_breakpoints(self, tmp_path: Path) -> None:
        """A malformed --merge-breakpoints fails fast at submit (parity with the
        other merge knobs), not hours later in the merge job."""
        import zarr
        from typer.testing import CliRunner

        from luxar.cli.gsplat_commands import app_gsplat

        src = tmp_path / "vol.zarr"
        z = zarr.open_array(str(src), mode="w", shape=(32, 32, 32), dtype="f4")
        z[:] = 0.0
        res = CliRunner().invoke(
            app_gsplat,
            [
                "batch-fit",
                "submit",
                str(src),
                str(tmp_path / "o"),
                "-p",
                "gpu",
                "--tile-size",
                "24",
                "--overlap",
                "4",
                "--axes",
                "z,y,x",
                "--dry-run",
                "--merge-recipe",
                "stream",
                "--merge-breakpoints",
                "counts:not_a_number",
            ],
        )
        assert res.exit_code != 0
        assert "breakpoints" in res.output.lower()


class TestStatusPartitionMergeDetection:
    def test_single_channel_partition_merge_detected_completed(
        self, tmp_path: Path
    ) -> None:
        """The partition-default merge writes merged/final.gsplats.zarr for ALL
        shapes; status must report a single-channel batch's merge as completed
        (pre-fix it looked only for t00_c00.gsplats.zarr)."""
        from luxar.gsplats.batch.manifest import (
            BatchJob,
            BatchManifest,
            output_filename,
            save_manifest,
        )
        from luxar.gsplats.batch.status import check_batch_status

        out = tmp_path / "batch"
        (out / "tiles").mkdir(parents=True)
        (out / "merged").mkdir(parents=True)
        m = BatchManifest(
            input_path="/d/x.zarr",
            output_dir=str(out),
            n_timepoints=1,
            n_channels=1,
            spatial_shape=(64, 64, 64),
            n_tiles=1,
            total_tasks=1,
            slurm_partition="gpu",
        )
        m.jobs = [BatchJob(0, 0, 0, 0, output_filename(0, 0, 0, 1, 1, 1), 1.0)]
        save_manifest(m, out)
        # the partition-default merge output (NOT the per-shape t00_c00 name)
        (out / "merged" / "final.gsplats.zarr").mkdir()

        st = check_batch_status(out)
        assert st.merge_status == "completed"


class TestStatusPackedSacctMapping:
    def test_packed_task_states_map_through_array_element(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With tasks_per_job > 1, sacct states are keyed by array ELEMENT index,
        so each flat task id must map through task_id // tasks_per_job. Pre-fix,
        task ids beyond the element range came back unknown and the rest read
        the wrong element's state."""
        from luxar.gsplats.batch import status as status_mod
        from luxar.gsplats.batch.manifest import BatchManifest, save_manifest
        from luxar.gsplats.batch.status import check_batch_status

        out = tmp_path / "batch"
        (out / "tiles").mkdir(parents=True)
        m = BatchManifest(
            input_path="/d/x.zarr",
            output_dir=str(out),
            n_timepoints=1,
            n_channels=1,
            spatial_shape=(64, 64, 64),
            n_tiles=4,
            total_tasks=4,
            tasks_per_job=3,
            slurm_partition="gpu",
        )
        m.array_job_id = 12345
        save_manifest(m, out)

        # Element 0 (tasks 0-2) failed; element 1 (task 3) is running.
        monkeypatch.setattr(
            status_mod, "_query_sacct", lambda job_id: {0: "FAILED", 1: "RUNNING"}
        )

        st = check_batch_status(out)
        assert st.failed == 3
        assert st.running == 1
        assert st.unknown == 0


class TestMergeRefineReachesEveryConsumer:
    """`--merge-refine` must survive plan time and reach BOTH merge consumers.

    There are three ways to merge a batch, and they read the recipe knobs from
    different places: the explicit `batch-fit merge` command takes CLI options,
    the local auto-merge (`batch-fit run`) reads ONLY the stored
    `manifest.merge_recipe_args`, and the Slurm merge job gets those same stored
    args re-emitted as CLI flags. A knob wired into just one of them is silently
    dropped by the others — which is exactly what happened first time round, with
    only the explicit command wired.
    """

    @staticmethod
    def _args(**kw):
        from luxar.cli.gsplat_ops.batch.planning import (
            MergeConfig,
            resolve_merge_recipe_args,
        )

        return resolve_merge_recipe_args(MergeConfig(**kw))

    def test_recorded_at_plan_time(self) -> None:
        args = self._args(recipe="levels", levels=1, refine="volume", refine_iters=7)
        assert args["refine"] == "volume"
        assert args["refine-iters"] == "7"

    def test_reaches_the_local_auto_merge(self) -> None:
        """`batch-fit run` builds its params from the stored dict alone."""
        from luxar.cli.gsplat_ops.batch.recipe_args import build_merge_recipe_params

        params = build_merge_recipe_params(
            self._args(recipe="levels", refine="volume", refine_iters=7)
        )
        assert params.refine == "volume"
        assert params.refine_iters == 7

    def test_reaches_the_slurm_merge_job(self) -> None:
        """The Slurm emitter turns each stored arg into ``--<key> <value>`` for the
        merge command, so the stored KEY has to match that command's option name."""
        args = self._args(recipe="levels", refine="volume", refine_iters=7)
        emitted = " ".join(f"--{k} {v}" for k, v in args.items())
        assert "--refine volume" in emitted
        assert "--refine-iters 7" in emitted

    def test_plan_time_validation_bites(self) -> None:
        """A typo must cost nothing: caught before any tile is fitted, not after."""
        import typer

        with pytest.raises(typer.BadParameter, match="must be one of"):
            self._args(recipe="levels", refine="bogus")
        with pytest.raises(typer.BadParameter, match="only applies with"):
            self._args(recipe="levels", refine_iters=5)
        # A knob with no recipe used to be silently dropped.
        with pytest.raises(typer.BadParameter, match="require a --merge-recipe"):
            self._args(refine="volume")
        # `stream` has no coarse levels to refine.
        with pytest.raises(typer.BadParameter, match="not used by --merge-recipe"):
            self._args(recipe="stream", refine="volume")

    def test_absent_when_not_requested(self) -> None:
        """Unset must stay unset, so a plan made before these knobs existed merges
        byte-identically."""
        args = self._args(recipe="levels", levels=1)
        assert "refine" not in args and "refine-iters" not in args

    def test_orphan_iters_still_bites_at_merge_time(self) -> None:
        """`batch-fit merge --refine-iters N` with no --refine is an orphan too.

        Plan time rejects it; the merge-time builder must agree rather than
        recording an iteration count against a `refine` of "none" that never reads
        it. Its `refine` may also come from the stored plan, so the resolved PAIR
        is what gets validated.
        """
        import typer

        from luxar.cli.gsplat_ops.batch.recipe_args import build_merge_recipe_params

        with pytest.raises(typer.BadParameter, match="only applies with"):
            build_merge_recipe_params({}, refine_iters=5)
        # Resolved against a stored refine, the same pair is legal.
        params = build_merge_recipe_params({"refine": "volume"}, refine_iters=5)
        assert params.refine == "volume" and params.refine_iters == 5


class TestMergeRefineSourceValidatedAtPlanTime:
    """`--merge-refine volume` re-opens THIS input at merge time, which is AFTER
    every tile has been fitted. Anything that makes the source unmappable has to
    be caught while planning, or a mistake costs the whole fit."""

    @staticmethod
    def _error(axes, n_t=1, n_c=1):
        from luxar.gsplats.batch.merge_orchestrator import volume_refit_source_error

        return volume_refit_source_error(axes, n_t, n_c)

    def test_the_mappable_shapes_pass(self) -> None:
        # Stacked timelapse: the time axis is the one the re-fit walks.
        assert self._error("t,z,y,x", n_t=3) is None
        # Canonical OME-Zarr: the channel axis is pinned to the fitted channel.
        assert self._error("t,c,z,y,x", n_t=3, n_c=1) is None
        # Single timepoint: nothing is stacked, so the time axis is pinned too.
        assert self._error("t,c,z,y,x", n_t=1, n_c=1) is None
        assert self._error("z,y,x") is None

    def test_the_unmappable_shapes_are_named(self) -> None:
        assert "axis labels" in (self._error(None) or "")
        assert "not recognised" in (self._error("q,y,x") or "")
        assert "no spatial axis" in (self._error("t,c") or "")
        assert "several channels" in (self._error("t,c,z,y,x", n_t=2, n_c=3) or "")
        # A flat channel index folded over two axes has no single axis to pin.
        assert "more than one axis" in (self._error("c,cam,z,y,x") or "")
        # Timepoints stacked but no time axis to map them onto.
        assert "no time axis" in (self._error("z,y,x", n_t=4) or "")

    def test_the_planner_refuses_before_submitting(self, tmp_path) -> None:
        """End-to-end: the guard is wired into plan_batch, not just available."""
        import typer
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )

        source = tmp_path / "movie.zarr"
        z = zarr.open(str(source), mode="w", shape=(3, 8, 8, 8), dtype="u2")
        z[:] = np.zeros((3, 8, 8, 8), np.uint16)

        def _plan(axes_list, refine="volume"):
            return plan_batch(
                input_path=source,
                output_dir=tmp_path / "out",
                tiling="uniform",
                tile_size=8,
                tile_overlap=0,
                axes_list=axes_list,
                array_key=None,
                timepoints_slice=None,
                channels_slice=None,
                fit=FitConfig(),
                denoise=DenoiseConfig(),
                content=ContentKnobs(),
                merge=MergeConfig(recipe="levels", levels=1, refine=refine),
            )

        with pytest.raises(typer.BadParameter, match="axis labels"):
            _plan(None)
        # The mode is NORMALISED before it is recorded, so the source check has to
        # read it the same way — otherwise a padded value is stored as "volume"
        # while skipping validation, and the failure resurfaces at merge time,
        # after every tile has been fitted.
        with pytest.raises(typer.BadParameter, match="axis labels"):
            _plan(None, refine=" volume ")
        # With labels the plan goes through, and records the knob for the merge.
        result = _plan(["t", "z", "y", "x"])
        assert result.manifest.merge_recipe_args["refine"] == "volume"


class TestUniformSlotBspTreeFrame:
    """The batch uniform merge's split planes must be in the SPLATS' frame.

    ``batch-fit run/submit --config <yaml>`` forwards that YAML verbatim to
    every ``fit --tile k/M`` worker, so a ``voxel_size:`` with the default
    ``output_space: real`` makes every worker emit PHYSICAL centers while the
    manifest's ``spatial_shape`` — and the tile grid rebuilt from it — stays in
    voxels. The batch merge is the fourth producer of these planes and needs the
    same reconciliation as ``fit`` itself (#1587).

    That reconciliation is resolved ONCE at PLAN time — where the merged fit
    config is in hand — and recorded on the manifest as ``grid_scale``; the
    merge only reads it. So the tests come in two halves: what the planner
    records, and what the merge does with what it finds recorded.

    ``voxel_size`` is the ONLY term this factor has on the batch path. A config
    ``downscale:`` is not one: every task rescales its splats back to the
    full-resolution frame this planner tiled, so it never moves them off the
    grid (#1624). Where a decimating value cannot complete at all — a multi-tile
    uniform plan, which is exactly this class's 64-tile setup — it is refused at
    plan time instead; ``test_batch_run.py`` covers the single-tile and content
    plans that DO complete with one and record no frame.
    """

    SPATIAL_SHAPE = (32, 32, 32)
    TILE_SIZE = 12
    OVERLAP = 2

    @classmethod
    def _manifest(cls, **kwargs: Any) -> Any:
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.tiling import compute_tile_specs

        n_tiles = len(compute_tile_specs(cls.SPATIAL_SHAPE, cls.TILE_SIZE, cls.OVERLAP))
        return BatchManifest(
            mode="uniform",
            spatial_shape=cls.SPATIAL_SHAPE,
            tile_size=cls.TILE_SIZE,
            tile_overlap=cls.OVERLAP,
            n_tiles=n_tiles,
            **kwargs,
        )

    @staticmethod
    def _splits(tree: dict) -> list:
        """Every internal node's ``(axis, split)``, in DFS order."""
        if "part" in tree:
            return []
        return (
            [(int(tree["axis"]), float(tree["split"]))]
            + TestUniformSlotBspTreeFrame._splits(tree["left"])
            + TestUniformSlotBspTreeFrame._splits(tree["right"])
        )

    @classmethod
    def _tree(cls, **kwargs: Any) -> Any:
        from luxar.gsplats.batch.merge_orchestrator import _slot_bsp_tree

        return _slot_bsp_tree(cls._manifest(**kwargs), Path("."), False)

    @classmethod
    def _plan(
        cls,
        tmp_path: Path,
        *,
        config: "Path | None" = None,
        preset: str = "standard",
        refine: "str | None" = None,
    ) -> Any:
        """Plan a real uniform batch over ``SPATIAL_SHAPE``; return its manifest."""
        import zarr

        from luxar.cli.gsplat_ops.batch.planning import (
            ContentKnobs,
            DenoiseConfig,
            FitConfig,
            MergeConfig,
            plan_batch,
        )

        source = tmp_path / "vol.zarr"
        if not source.exists():
            z = zarr.open(str(source), mode="w", shape=cls.SPATIAL_SHAPE, dtype="u2")
            z[:] = np.zeros(cls.SPATIAL_SHAPE, np.uint16)
        merge = (
            MergeConfig(recipe="levels", levels=1, refine=refine)
            if refine
            else MergeConfig()
        )
        return plan_batch(
            input_path=source,
            output_dir=tmp_path / "out",
            tiling="uniform",
            tile_size=cls.TILE_SIZE,
            tile_overlap=cls.OVERLAP,
            axes_list=["z", "y", "x"],
            array_key=None,
            timepoints_slice=None,
            channels_slice=None,
            fit=FitConfig(preset=preset, config=config),
            denoise=DenoiseConfig(),
            content=ContentKnobs(),
            merge=merge,
        ).manifest

    # -- what the planner records -------------------------------------------

    def test_the_planner_records_a_voxel_size_as_the_frame(
        self, tmp_path: Path
    ) -> None:
        """The regression: anisotropic on purpose, so an axis swap anywhere in
        the recovery would show up."""
        config = tmp_path / "fit.yaml"
        config.write_text("voxel_size: [4.0, 1.0, 1.0]\n")
        assert self._plan(tmp_path, config=config).grid_scale == [4.0, 1.0, 1.0]

    def test_output_space_voxel_records_no_frame(self, tmp_path: Path) -> None:
        """Non-vacuity control: the spacing term is value-scoped, not blanket —
        with ``output_space: voxel`` the centers stay in voxels, so there is
        nothing to reconcile and nothing is recorded."""
        config = tmp_path / "voxel.yaml"
        config.write_text("voxel_size: [4.0, 1.0, 1.0]\noutput_space: voxel\n")
        assert self._plan(tmp_path, config=config).grid_scale is None

    def test_a_config_without_a_spacing_records_no_frame(self, tmp_path: Path) -> None:
        """The overwhelmingly common case must record nothing, so its merge is
        bit-identical to a pre-#1587 one."""
        config = tmp_path / "plain.yaml"
        config.write_text("n_iters: 50\nfloor: none\n")
        assert self._plan(tmp_path, config=config).grid_scale is None
        # A `--preset` alone likewise carries no spacing.
        assert self._plan(tmp_path, preset="draft").grid_scale is None

    def test_a_decimating_downscale_is_refused_for_this_multi_tile_plan(
        self, tmp_path: Path
    ) -> None:
        """``downscale:`` is a documented YAML key, but not one THIS plan can use.

        There is nothing for this tree to reconcile, because the run never gets
        as far as a merge: the planner builds its tasks on the FULL-resolution
        shape while each ``fit --tile k/M`` worker recomputes the grid on its own
        decimated one, so the tile counts disagree. Measured on this 32^3 store
        with ``--tile-size 12 --overlap 2``: the plan builds 64 tiles per slot
        and a ``downscale: 2`` worker (volume ``(16, 16, 16)``) sees 8, so tiles
        8..63 exit 1 with "tile index out of range". Refused at plan time
        instead (#1624) — before a task is submitted or a tile is written —
        rather than recorded as a frame the merge would never be reached to
        apply. The message must carry both counts: that difference IS the
        diagnosis, and neither number is visible from the config.

        FAILS pre-fix: the plan succeeded and recorded ``[2.0, 2.0, 2.0]``.
        """
        import typer

        config = tmp_path / "ds.yaml"
        config.write_text("downscale: 2\n")
        with pytest.raises(typer.BadParameter) as excinfo:
            self._plan(tmp_path, config=config)
        message = str(excinfo.value)
        assert "downscale" in message
        assert "64 tiles" in message  # what this plan built
        assert "only 8" in message  # what a worker would see
        assert "(16, 16, 16)" in message  # its decimated volume

    def test_a_no_op_downscale_still_plans(self, tmp_path: Path) -> None:
        """Non-vacuity control: the refusal is value-scoped, not "any downscale key".

        ``downscale: 1`` (and its per-axis spelling) decimates nothing — the
        grids already agree — so it must plan silently and record no frame,
        exactly as a config without the key at all, in the very multi-tile setup
        a decimating value is refused for.
        """
        for spelling in ("downscale: 1\n", "downscale: [1, 1, 1]\n"):
            config = tmp_path / "noop.yaml"
            config.write_text(spelling)
            assert self._plan(tmp_path, config=config).grid_scale is None

    def test_the_recorded_config_path_is_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--config fit.yaml`` is handed to workers that run from a Slurm job's
        own working directory, so it is stored absolute like every other path on
        the manifest — never as the relative spelling the user typed."""
        config = tmp_path / "fit.yaml"
        config.write_text("n_iters: 50\n")
        monkeypatch.chdir(tmp_path)
        manifest = self._plan(tmp_path, config=Path("fit.yaml"))
        assert manifest.fit_args["config"] == str(config.resolve())
        assert Path(manifest.fit_args["config"]).is_absolute()

    # -- what the merge does with it ----------------------------------------

    def test_the_merge_scales_every_plane_by_the_recorded_frame(self) -> None:
        """A recorded frame moves the planes onto the splats, per axis."""
        plain = self._splits(self._tree())
        physical = self._splits(self._tree(grid_scale=[4.0, 1.0, 1.0]))
        assert len(plain) == len(physical) > 0
        for (ax_p, s_p), (ax_s, s_s) in zip(plain, physical):
            assert ax_p == ax_s
            assert s_s == pytest.approx((4.0, 1.0, 1.0)[ax_s] * s_p)
        # The measured symptom of the bug: without the factor every axis-0 plane
        # sits in the first quarter of the object.
        assert sorted({s for ax, s in plain if ax == 0}) == [11.0, 21.0, 31.0]
        assert sorted({s for ax, s in physical if ax == 0}) == [44.0, 84.0, 124.0]

    def test_a_manifest_without_a_frame_keeps_the_voxel_grid(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Backward compatibility: an ABSENT ``grid_scale`` — every manifest
        written before this field existed, and every run whose frames agree —
        means no scale, which is exactly the pre-#1587 behaviour. Silently, with
        no warning and no dropped planes."""
        assert self._tree(grid_scale=None) == self._tree()
        assert self._tree(grid_scale=[1.0, 1.0, 1.0]) == self._tree()
        assert self._tree() is not None
        assert "WARNING" not in capsys.readouterr().out

    def test_the_merge_reads_the_manifest_not_the_config_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """The frame travels ON the manifest, so the merge never touches the YAML.

        Re-resolving it at merge time had two measured failure modes, and this
        pins both shut: a config that has since been DELETED (a Slurm run merged
        days later) must not cost the planes, and an unrelated same-named file
        sitting where the merge runs must not be able to restate the frame.
        """
        config = tmp_path / "fit.yaml"
        config.write_text("voxel_size: [4.0, 1.0, 1.0]\n")
        manifest = self._plan(tmp_path, config=config)
        recorded = manifest.grid_scale
        assert recorded == [4.0, 1.0, 1.0]

        config.unlink()  # the YAML is gone by merge time
        decoy = tmp_path / "decoy"
        decoy.mkdir()
        (decoy / "fit.yaml").write_text("voxel_size: [9.0, 9.0, 9.0]\n")

        tree = self._tree(grid_scale=recorded, fit_args=dict(manifest.fit_args))
        splits = self._splits(tree)
        # Absolute values, not a comparison against another recovered tree: the
        # planes must be where the PHYSICAL splats are (44/84/124 on axis 0),
        # neither back in voxels (11/21/31, the deleted-YAML failure) nor at the
        # decoy's 9x (99/189/279), nor dropped for want of a config.
        assert sorted({s for ax, s in splits if ax == 0}) == [44.0, 84.0, 124.0]
        assert "WARNING" not in capsys.readouterr().out

    def test_a_corrupt_recorded_frame_degrades_loudly(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Only reachable from a hand-edited manifest, but a merge that runs
        after every tile has been fitted must not abort over it: drop the planes
        (the documented centroid fallback) and say so, loudly."""
        assert self._tree(grid_scale=[4.0, 1.0]) is None
        out = capsys.readouterr().out
        assert "WARNING" in out and "grid_scale" in out
        assert self._tree(grid_scale=[4.0, 0.0, 1.0]) is None
        assert "WARNING" in capsys.readouterr().out
        # A NaN and a non-number entry are the same class of hand-edit and must
        # degrade the same way — the second raises TypeError rather than
        # ValueError, so it has to be caught too or the merge dies here, after
        # every tile was fitted.
        assert self._tree(grid_scale=[4.0, float("nan"), 1.0]) is None
        assert "WARNING" in capsys.readouterr().out
        assert self._tree(grid_scale=[4.0, None, 1.0]) is None
        assert "WARNING" in capsys.readouterr().out

    # -- the frame also gates --merge-refine volume --------------------------

    def test_refine_volume_is_refused_when_the_frames_disagree(
        self, tmp_path: Path
    ) -> None:
        """A per-part re-fit crops the source in VOXELS from the tile grid, so a
        recorded frame makes every crop a factor off — refused at PLAN time, so
        the mistake costs nothing rather than surfacing after every tile has been
        fitted (`gsplat fit` refuses the same combination)."""
        import typer

        config = tmp_path / "fit.yaml"
        config.write_text("voxel_size: [4.0, 1.0, 1.0]\n")
        with pytest.raises(typer.BadParameter, match="frame scaled by"):
            self._plan(tmp_path, config=config, refine="volume")
        # Whitespace must not smuggle it past: the mode is normalised first.
        with pytest.raises(typer.BadParameter, match="frame scaled by"):
            self._plan(tmp_path, config=config, refine=" volume ")
        # Non-vacuity: the same plan with the frames in agreement goes through
        # and records the knob for the merge.
        ok = self._plan(tmp_path, refine="volume")
        assert ok.grid_scale is None
        assert ok.merge_recipe_args["refine"] == "volume"

    def test_a_frame_the_resolver_refuses_fails_at_plan_time(
        self, tmp_path: Path
    ) -> None:
        """Readable but unusable is caught while the user is still here.

        ``resolve_grid_scale`` refuses an out-of-vocabulary ``output_space``
        rather than reading it as "voxel", which would silently drop the spacing
        term — the very #1587 mismatch. Raised at PLAN time, not from a merge
        days later that would have to choose between a wrong tree and none.
        """
        import typer

        config = tmp_path / "typo.yaml"
        config.write_text("voxel_size: [4.0, 1.0, 1.0]\noutput_space: physical\n")
        with pytest.raises(typer.BadParameter, match="output_space"):
            self._plan(tmp_path, config=config)

        # Likewise a spacing the tree cannot be stated in. `.nan` is legal YAML
        # and the fitter's own `<= 0` check does not reject it, so without the
        # resolver's finiteness check it would be recorded on the manifest and
        # become a NaN split plane at merge time.
        nan_config = tmp_path / "nan.yaml"
        nan_config.write_text("voxel_size: [.nan, 1.0, 1.0]\n")
        with pytest.raises(typer.BadParameter, match="finite and strictly positive"):
            self._plan(tmp_path, config=nan_config)

    def test_an_unreadable_config_fails_at_plan_time(self, tmp_path: Path) -> None:
        """Same door, other side: a ``--config`` that cannot be read at all is a
        planning error, not something the merge should have to guess around."""
        import typer

        with pytest.raises(typer.BadParameter, match="could not read the fit config"):
            self._plan(tmp_path, config=tmp_path / "gone.yaml")
