"""Tests for batch fitting infrastructure (manifest, env, slurm, time, merge)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

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
            spatial_shape=(128, 256, 256),
            tile_size=128,
            n_tiles=4,
            total_tasks=40,
            preset="standard",
            slurm_partition="gpu",
        )
        manifest.jobs = [
            BatchJob(
                task_id=0,
                timepoint=0,
                channel=0,
                tile_index=0,
                output_filename="t00_c00_tile000.gsplats.zarr",
                estimated_wall_seconds=300.0,
            )
        ]

        save_manifest(manifest, tmp_path)
        loaded = load_manifest(tmp_path)

        assert loaded.n_timepoints == 5
        assert loaded.n_channels == 2
        assert loaded.spatial_shape == (128, 256, 256)
        assert len(loaded.jobs) == 1
        assert loaded.jobs[0].task_id == 0

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


# ====================================================================
# Environment capture tests
# ====================================================================


class TestEnvCapture:
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

    def test_merge_script(self) -> None:
        from luxar.gsplats.batch.manifest import BatchManifest
        from luxar.gsplats.batch.slurm_gen import generate_merge_sbatch

        manifest = BatchManifest(
            output_dir="/output",
            slurm_partition="gpu",
        )

        script = generate_merge_sbatch(manifest, "# env\n")
        assert "luxar gsplat batch merge" in script
        assert "#SBATCH --job-name=luxar-merge" in script

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

        from luxar.cli.gsplat_config import discover_ome_zarr_shape

        store_path = tmp_path / "test.zarr"
        root = zarr.open(str(store_path), mode="w")
        root.create_dataset("0", data=np.zeros((64, 128, 128), dtype=np.float32))

        info = discover_ome_zarr_shape(store_path)
        assert info.n_timepoints == 1
        assert info.n_channels == 1
        assert info.spatial_shape == (64, 128, 128)

    def test_discover_5d_zarr_heuristic(self, tmp_path: Path) -> None:
        import zarr

        from luxar.cli.gsplat_config import discover_ome_zarr_shape

        store_path = tmp_path / "test.zarr"
        root = zarr.open(str(store_path), mode="w")
        root.create_dataset("0", data=np.zeros((10, 3, 64, 128, 128), dtype=np.float32))

        info = discover_ome_zarr_shape(store_path)
        assert info.n_timepoints == 10
        assert info.n_channels == 3
        assert info.spatial_shape == (64, 128, 128)

    def test_discover_ome_ngff_metadata(self, tmp_path: Path) -> None:
        import zarr

        from luxar.cli.gsplat_config import discover_ome_zarr_shape

        store_path = tmp_path / "ome.zarr"
        root = zarr.open(str(store_path), mode="w")
        root.create_dataset("0", data=np.zeros((5, 2, 32, 64, 64), dtype=np.float32))
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
        assert info.spatial_shape == (32, 64, 64)
        assert info.spatial_axes == ["z", "y", "x"]
        assert info.voxel_size == (0.5, 0.3, 0.3)
        assert info.unit == "micrometer"
        assert info.resolution_levels == 1
