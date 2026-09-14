"""Tests for Gaussian-splat PyTorch device-selection helpers."""

from __future__ import annotations

import pytest
import torch

import luxar.gsplats.metrics as metrics
import luxar.gsplats.utils.device as device_mod
from luxar.gsplats.utils.device import (
    GpuInfo,
    enumerate_gpus,
    is_mps_available,
    resolve_gpu_selection,
    resolve_jobs_per_gpu,
    resolve_jobs_per_gpu_with_limit,
    resolve_torch_device,
)

_GB = 1024**3


class _FakeMPSBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def test_resolve_explicit_device_overrides_accelerator_flags(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(False), raising=False)

    device = resolve_torch_device("cuda", use_cuda=False, use_metal=False)

    assert device.type == "cuda"


def test_resolve_auto_honors_accelerator_opt_outs(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(True), raising=False)

    device = resolve_torch_device("auto", use_cuda=False, use_metal=False)

    assert device.type == "cpu"


def test_resolve_auto_is_case_sensitive() -> None:
    with pytest.raises(RuntimeError, match="device type"):
        resolve_torch_device("AUTO")


def test_resolve_honors_accelerator_opt_outs(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(True), raising=False)

    device = resolve_torch_device(None, use_cuda=False, use_metal=False)

    assert device.type == "cpu"


def test_resolve_prefers_cuda_over_mps_when_both_enabled(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(True), raising=False)

    device = resolve_torch_device(None, use_cuda=True, use_metal=True)

    assert device.type == "cuda"


def test_resolve_uses_mps_when_cuda_disabled(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _FakeMPSBackend(True), raising=False)

    device = resolve_torch_device(None, use_cuda=False, use_metal=True)

    assert device.type == "mps"


def test_is_mps_available_handles_missing_backend(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends, "mps", None, raising=False)

    assert is_mps_available() is False


# ---------------------------------------------------------------------------
# Multi-GPU enumeration & selection
# ---------------------------------------------------------------------------


class _FakeProps:
    def __init__(self, name: str, total_memory: int) -> None:
        self.name = name
        self.total_memory = total_memory


def _fake_cuda(monkeypatch, mems: dict[int, tuple[str, int, int | None]]) -> None:
    """Patch torch.cuda + metrics._gpu_free_memory to model a fixed GPU set.

    ``mems`` maps device index -> (name, total_bytes, free_bytes_or_None).
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(mems))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(mems))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: _FakeProps(mems[i][0], mems[i][1]),
    )
    monkeypatch.setattr(
        metrics,
        "_gpu_free_memory",
        lambda dev: mems[dev.index][2],
    )


def test_enumerate_gpus_empty_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert enumerate_gpus() == []


def test_enumerate_gpus_reports_name_and_memory(monkeypatch) -> None:
    _fake_cuda(
        monkeypatch,
        {0: ("Big", 96 * _GB, 90 * _GB), 1: ("Small", 8 * _GB, 7 * _GB)},
    )
    gpus = enumerate_gpus()
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0] == GpuInfo(0, "Big", 96 * _GB, 90 * _GB)
    assert gpus[1].name == "Small" and gpus[1].free_mem == 7 * _GB


def test_resolve_gpu_selection_cpu_sentinel() -> None:
    assert resolve_gpu_selection("cpu") == []
    assert resolve_gpu_selection("CPU") == []


def test_resolve_gpu_selection_auto_skips_small_card(monkeypatch) -> None:
    # Big (90 GB free) clears the 12 GB floor; the 8 GB card does not.
    monkeypatch.setattr(
        device_mod,
        "enumerate_gpus",
        lambda: [
            GpuInfo(0, "Big", 96 * _GB, 90 * _GB),
            GpuInfo(1, "Small", 8 * _GB, 7 * _GB),
        ],
    )
    assert resolve_gpu_selection("auto") == [0]
    assert resolve_gpu_selection("all") == [0, 1]


def test_resolve_gpu_selection_auto_falls_back_to_largest(monkeypatch) -> None:
    # No card clears the floor -> pick the single largest rather than nothing.
    monkeypatch.setattr(
        device_mod,
        "enumerate_gpus",
        lambda: [GpuInfo(0, "A", 6 * _GB, 5 * _GB), GpuInfo(1, "B", 8 * _GB, 7 * _GB)],
    )
    assert resolve_gpu_selection("auto") == [1]


def test_resolve_gpu_selection_explicit_and_dedup(monkeypatch) -> None:
    monkeypatch.setattr(
        device_mod,
        "enumerate_gpus",
        lambda: [GpuInfo(i, f"g{i}", 16 * _GB, 16 * _GB) for i in range(4)],
    )
    assert resolve_gpu_selection("0,2") == [0, 2]
    assert resolve_gpu_selection("1,1,3") == [1, 3]  # de-duplicated, order kept


def test_resolve_gpu_selection_out_of_range_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        device_mod,
        "enumerate_gpus",
        lambda: [GpuInfo(0, "g0", 16 * _GB, 16 * _GB)],
    )
    with pytest.raises(ValueError, match="out of range"):
        resolve_gpu_selection("5")


def test_resolve_gpu_selection_unparseable_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        device_mod,
        "enumerate_gpus",
        lambda: [GpuInfo(0, "g0", 16 * _GB, 16 * _GB)],
    )
    with pytest.raises(ValueError, match="Could not parse"):
        resolve_gpu_selection("fast")


def test_resolve_gpu_selection_no_cuda_raises(monkeypatch) -> None:
    monkeypatch.setattr(device_mod, "enumerate_gpus", lambda: [])
    with pytest.raises(ValueError, match="no CUDA device"):
        resolve_gpu_selection("auto")
    with pytest.raises(ValueError, match="no CUDA device"):
        resolve_gpu_selection("0")


def test_resolve_jobs_per_gpu_cpu(monkeypatch) -> None:
    assert resolve_jobs_per_gpu([], task_voxels=4096, jobs_per_gpu=3) == {-1: 3}
    auto = resolve_jobs_per_gpu([], task_voxels=4096, jobs_per_gpu="auto")
    assert set(auto) == {-1} and auto[-1] >= 1


def test_resolve_jobs_per_gpu_auto_caps_reported_small_task(monkeypatch) -> None:
    free = {0: int(7.5 * _GB)}
    monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: free[dev.index])
    monkeypatch.setattr(device_mod, "available_host_memory_bytes", lambda: 125 * _GB)
    monkeypatch.setattr(device_mod.os, "cpu_count", lambda: 32)

    plan = resolve_jobs_per_gpu_with_limit(
        [0], task_voxels=41 * 512 * 512, jobs_per_gpu="auto"
    )

    assert plan.workers == {0: 8}
    assert plan.limit == "hard cap"


def test_resolve_jobs_per_gpu_auto_scales_with_vram(monkeypatch) -> None:
    # The host-wide cap is weighted toward the card with more free VRAM.
    free = {0: 40 * _GB, 1: 8 * _GB}
    monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: free[dev.index])
    monkeypatch.setattr(device_mod, "available_host_memory_bytes", lambda: 125 * _GB)
    monkeypatch.setattr(device_mod.os, "cpu_count", lambda: 32)
    workers = resolve_jobs_per_gpu([0, 1], task_voxels=256**3, jobs_per_gpu="auto")
    assert sum(workers.values()) == 8
    assert workers[0] > workers[1] >= 1


def test_resolve_jobs_per_gpu_applies_host_ram_once_across_gpus(monkeypatch) -> None:
    monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: 40 * _GB)
    monkeypatch.setattr(device_mod, "available_host_memory_bytes", lambda: 4 * _GB)
    monkeypatch.setattr(device_mod.os, "cpu_count", lambda: 32)

    plan = resolve_jobs_per_gpu_with_limit(
        [0, 1], task_voxels=1000, jobs_per_gpu="auto"
    )

    assert sum(plan.workers.values()) == 3
    assert set(plan.workers) == {0, 1}
    assert plan.limit == "host RAM"


def test_resolve_jobs_per_gpu_applies_cpu_count_across_gpus(monkeypatch) -> None:
    monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: 40 * _GB)
    monkeypatch.setattr(device_mod, "available_host_memory_bytes", lambda: 125 * _GB)
    monkeypatch.setattr(device_mod.os, "cpu_count", lambda: 4)

    plan = resolve_jobs_per_gpu_with_limit(
        [0, 1], task_voxels=1000, jobs_per_gpu="auto"
    )

    assert sum(plan.workers.values()) == 4
    assert plan.limit == "CPU count"


def test_resolve_jobs_per_gpu_unknown_gpu_memory_is_conservative(monkeypatch) -> None:
    monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: None)
    monkeypatch.setattr(device_mod, "available_host_memory_bytes", lambda: 125 * _GB)
    monkeypatch.setattr(device_mod.os, "cpu_count", lambda: 32)

    plan = resolve_jobs_per_gpu_with_limit(
        [0, 1], task_voxels=1000, jobs_per_gpu="auto"
    )

    assert plan.workers == {0: 1, 1: 1}
    assert plan.limit == "GPU memory unavailable"


def test_resolve_jobs_per_gpu_unknown_host_memory_still_hard_caps(monkeypatch) -> None:
    monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: 40 * _GB)
    monkeypatch.setattr(device_mod, "available_host_memory_bytes", lambda: None)
    monkeypatch.setattr(device_mod.os, "cpu_count", lambda: 32)

    plan = resolve_jobs_per_gpu_with_limit([0], task_voxels=1000, jobs_per_gpu="auto")

    assert plan.workers == {0: 8}
    assert plan.limit == "hard cap"


def test_resolve_jobs_per_gpu_explicit_uniform() -> None:
    workers = resolve_jobs_per_gpu([0, 1, 2], task_voxels=1, jobs_per_gpu=2)
    assert workers == {0: 2, 1: 2, 2: 2}
