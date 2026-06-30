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
    assert resolve_jobs_per_gpu([], task_voxels=4096, n_tasks=10, jobs_per_gpu=3) == {
        -1: 3
    }
    auto = resolve_jobs_per_gpu([], task_voxels=4096, n_tasks=10, jobs_per_gpu="auto")
    assert set(auto) == {-1} and auto[-1] >= 1


def test_resolve_jobs_per_gpu_auto_scales_with_vram(monkeypatch) -> None:
    # 40 GB vs 8 GB free -> the big card gets ~5x the workers.
    free = {0: 40 * _GB, 1: 8 * _GB}
    monkeypatch.setattr(metrics, "_gpu_free_memory", lambda dev: free[dev.index])
    # ~256^3 voxels working set.
    workers = resolve_jobs_per_gpu(
        [0, 1], task_voxels=256**3, n_tasks=1000, jobs_per_gpu="auto"
    )
    assert workers[0] > workers[1] >= 1
    assert 4 <= workers[0] / workers[1] <= 6


def test_resolve_jobs_per_gpu_explicit_uniform() -> None:
    workers = resolve_jobs_per_gpu(
        [0, 1, 2], task_voxels=1, n_tasks=100, jobs_per_gpu=2
    )
    assert workers == {0: 2, 1: 2, 2: 2}
