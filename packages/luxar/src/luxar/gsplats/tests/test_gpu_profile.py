"""Tests for GPU profile management (multi-GPU registry)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def profile_path(tmp_path: Path) -> Path:
    return tmp_path / "gpu_profiles.yaml"


@pytest.fixture
def sample_run() -> dict:
    """A minimal benchmark run data dict."""
    return {
        "timestamp": "2026-03-11T14:00:00+00:00",
        "cuda_version": "12.1",
        "pytorch_version": "2.5.1",
        "free_memory_gb": 24.0,
        "throughput": {
            "3d": [
                {
                    "label": "3D 256 5K",
                    "shape": [256, 256, 256],
                    "voxels": 16777216,
                    "splats": 5000,
                    "oom": False,
                    "fp32_ms": 1.1,
                    "fp32_gvoxel_per_s": 15.0,
                    "amp_ms": 1.2,
                    "amp_gvoxel_per_s": 14.0,
                    "fp32_train_ms": 3.5,
                    "amp_train_ms": 3.6,
                },
                {
                    "label": "3D 512 20K",
                    "shape": [512, 512, 512],
                    "voxels": 134217728,
                    "splats": 20000,
                    "oom": False,
                    "fp32_ms": 3.0,
                    "fp32_gvoxel_per_s": 44.0,
                    "amp_train_ms": 8.5,
                },
                {
                    "label": "3D 1024 50K",
                    "shape": [1024, 1024, 1024],
                    "voxels": 1073741824,
                    "splats": 50000,
                    "oom": True,
                },
            ],
        },
        "oom_boundaries": {
            "3d": {
                "max_successful_shape": [512, 512, 512],
                "max_successful_voxels": 134217728,
                "min_oom_shape": [1024, 1024, 1024],
                "min_oom_voxels": 1073741824,
            }
        },
        "recommendations": {
            "peak_throughput_3d": {
                "shape": [512, 512, 512],
                "gvoxel_per_s": 44.0,
                "splats": 20000,
            },
            "memory_safe_max_voxels_fp32": 3000000000,
            "memory_safe_max_cube_side_3d": 1442,
        },
    }


@pytest.fixture
def sample_gpu_info() -> dict:
    return {
        "total_memory_gb": 24.0,
        "compute_capability": "8.6",
        "sm_count": 84,
    }


class TestLoadSaveProfiles:
    def test_load_nonexistent_returns_empty(self, profile_path: Path) -> None:
        from luxar.gsplats.gpu_profile import load_profiles

        profiles = load_profiles(profile_path)
        assert profiles["schema_version"] == 2
        assert profiles["gpus"] == {}

    def test_save_and_load_roundtrip(self, profile_path: Path) -> None:
        from luxar.gsplats.gpu_profile import load_profiles, save_profiles

        data = {
            "schema_version": 2,
            "gpus": {
                "TestGPU": {
                    "info": {"total_memory_gb": 16.0},
                    "runs": [],
                    "summary": {},
                }
            },
        }
        save_profiles(data, profile_path)
        loaded = load_profiles(profile_path)
        assert loaded["gpus"]["TestGPU"]["info"]["total_memory_gb"] == 16.0


class TestAppendRun:
    def test_append_creates_profile(
        self, profile_path: Path, sample_run: dict, sample_gpu_info: dict
    ) -> None:
        from luxar.gsplats.gpu_profile import append_run, load_profiles

        append_run("TestGPU", sample_run, sample_gpu_info, path=profile_path)

        profiles = load_profiles(profile_path)
        assert "TestGPU" in profiles["gpus"]
        assert len(profiles["gpus"]["TestGPU"]["runs"]) == 1
        assert "summary" in profiles["gpus"]["TestGPU"]

    def test_append_multiple_runs(
        self, profile_path: Path, sample_run: dict, sample_gpu_info: dict
    ) -> None:
        from luxar.gsplats.gpu_profile import append_run, load_profiles

        append_run("TestGPU", sample_run, sample_gpu_info, path=profile_path)
        append_run("TestGPU", sample_run, sample_gpu_info, path=profile_path)

        profiles = load_profiles(profile_path)
        assert len(profiles["gpus"]["TestGPU"]["runs"]) == 2

    def test_multi_gpu(
        self, profile_path: Path, sample_run: dict, sample_gpu_info: dict
    ) -> None:
        from luxar.gsplats.gpu_profile import append_run, load_profiles

        append_run("GPU_A", sample_run, sample_gpu_info, path=profile_path)
        append_run("GPU_B", sample_run, {"total_memory_gb": 80.0}, path=profile_path)

        profiles = load_profiles(profile_path)
        assert len(profiles["gpus"]) == 2
        assert "GPU_A" in profiles["gpus"]
        assert "GPU_B" in profiles["gpus"]


class TestRecomputeSummary:
    def test_summary_has_recommendations(self, sample_run: dict) -> None:
        from luxar.gsplats.gpu_profile import recompute_summary

        summary = recompute_summary([sample_run])
        assert "recommendations" in summary
        assert "peak_throughput_3d" in summary["recommendations"]

    def test_conservative_oom(self, sample_run: dict) -> None:
        # Create a second run with a smaller OOM boundary
        import copy

        from luxar.gsplats.gpu_profile import recompute_summary

        run2 = copy.deepcopy(sample_run)
        run2["oom_boundaries"]["3d"]["max_successful_voxels"] = 100000000
        run2["oom_boundaries"]["3d"]["max_successful_shape"] = [464, 464, 464]

        summary = recompute_summary([sample_run, run2])
        oom = summary["oom_boundaries"]["3d"]
        # Should take the min (most conservative)
        assert oom["max_successful_voxels"] == 100000000

    def test_average_throughput(self, sample_run: dict) -> None:
        import copy

        from luxar.gsplats.gpu_profile import recompute_summary

        run2 = copy.deepcopy(sample_run)
        # Modify throughput slightly
        run2["throughput"]["3d"][0]["fp32_gvoxel_per_s"] = 17.0

        summary = recompute_summary([sample_run, run2])
        avg_3d = summary["throughput_avg"]["3d"]
        first = [e for e in avg_3d if e["label"] == "3D 256 5K"][0]
        # Average of 15.0 and 17.0 = 16.0
        assert first["fp32_gvoxel_per_s"] == 16.0


class TestGetGpuSummary:
    def test_single_gpu_auto_select(
        self, profile_path: Path, sample_run: dict, sample_gpu_info: dict
    ) -> None:
        from luxar.gsplats.gpu_profile import append_run, get_gpu_summary

        append_run("OnlyGPU", sample_run, sample_gpu_info, path=profile_path)
        summary = get_gpu_summary(path=profile_path)
        assert summary is not None
        assert "recommendations" in summary

    def test_by_name(
        self, profile_path: Path, sample_run: dict, sample_gpu_info: dict
    ) -> None:
        from luxar.gsplats.gpu_profile import append_run, get_gpu_summary

        append_run("MyGPU", sample_run, sample_gpu_info, path=profile_path)
        assert get_gpu_summary("MyGPU", path=profile_path) is not None
        assert get_gpu_summary("OtherGPU", path=profile_path) is None

    def test_by_gpu_mem(self, profile_path: Path, sample_run: dict) -> None:
        from luxar.gsplats.gpu_profile import append_run, get_gpu_summary

        append_run("Small", sample_run, {"total_memory_gb": 8.0}, path=profile_path)
        append_run("Big", sample_run, {"total_memory_gb": 80.0}, path=profile_path)

        # Closest to 40GB
        summary = get_gpu_summary(gpu_mem=40.0, path=profile_path)
        assert summary is not None


class TestMigrateV1:
    def test_migrate_v1_profile(self, tmp_path: Path) -> None:
        from luxar.gsplats.gpu_profile import load_profiles, migrate_v1_profile

        old_path = tmp_path / "gpu_benchmark_profile.yaml"
        new_path = tmp_path / "gpu_profiles.yaml"

        # Write a v1 profile
        v1_data = {
            "gpu": {
                "name": "NVIDIA RTX 3090",
                "total_memory_gb": 24.0,
                "free_memory_gb": 22.0,
                "compute_capability": "8.6",
                "sm_count": 82,
                "cuda_version": "12.1",
                "pytorch_version": "2.5.0",
            },
            "timestamp": "2026-01-01T00:00:00",
            "throughput": {"3d": []},
            "oom_boundaries": {},
            "recommendations": {},
        }
        with open(old_path, "w") as f:
            yaml.dump(v1_data, f)

        result = migrate_v1_profile(old_path, new_path)
        assert result is True
        assert not old_path.exists()  # Renamed to .bak
        assert (tmp_path / "gpu_benchmark_profile.yaml.bak").exists()

        profiles = load_profiles(new_path)
        assert "NVIDIA RTX 3090" in profiles["gpus"]
        assert len(profiles["gpus"]["NVIDIA RTX 3090"]["runs"]) == 1

    def test_migrate_idempotent_with_existing_bak(self, tmp_path: Path) -> None:
        """Migration should not crash if .bak already exists."""
        from luxar.gsplats.gpu_profile import migrate_v1_profile

        old_path = tmp_path / "gpu_benchmark_profile.yaml"
        new_path = tmp_path / "gpu_profiles.yaml"
        bak_path = old_path.with_suffix(".yaml.bak")

        v1_data = {
            "gpu": {"name": "TestGPU", "total_memory_gb": 16.0},
            "timestamp": "2026-01-01",
            "throughput": {},
            "oom_boundaries": {},
            "recommendations": {},
        }
        with open(old_path, "w") as f:
            yaml.dump(v1_data, f)
        # Pre-create .bak to simulate prior migration
        bak_path.write_text("old backup")

        result = migrate_v1_profile(old_path, new_path)
        assert result is True
        assert not old_path.exists()


class TestCorruptedProfile:
    def test_load_corrupted_yaml(self, tmp_path: Path) -> None:
        """Corrupted YAML should return empty profile, not crash."""
        from luxar.gsplats.gpu_profile import load_profiles

        path = tmp_path / "gpu_profiles.yaml"
        path.write_text("schema_version: 2\ngpus:\n  broken: [\n")  # Invalid YAML

        profiles = load_profiles(path)
        assert profiles["schema_version"] == 2
        assert profiles["gpus"] == {}

    def test_load_truncated_yaml(self, tmp_path: Path) -> None:
        """Truncated YAML should return empty profile."""
        from luxar.gsplats.gpu_profile import load_profiles

        path = tmp_path / "gpu_profiles.yaml"
        path.write_bytes(b"\xff\xfe")  # Invalid unicode

        profiles = load_profiles(path)
        assert profiles["gpus"] == {}
