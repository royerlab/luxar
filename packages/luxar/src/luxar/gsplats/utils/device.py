"""PyTorch device-selection helpers for Gaussian splat code.

Use :func:`resolve_torch_device` for default-device auto-selection (CUDA > MPS
> CPU) and :func:`is_mps_available` for guarded availability checks. Memory
operations such as ``torch.cuda.empty_cache()`` should keep using ``torch.cuda``
directly since they are not about device selection.

Here, ``device="auto"`` selects one PyTorch device; ``--gpus auto`` separately
selects every CUDA device that clears the free-memory floor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

# Free-VRAM floor (bytes) for ``--gpus auto``: a CUDA device is selected only if
# it has at least this much free memory, so a small co-resident card (e.g. an
# 8 GB display GPU) is skipped rather than bottlenecking or OOMing a fit.
# Override with the ``LUXAR_GPU_VRAM_FLOOR_GB`` environment variable.
_AUTO_VRAM_FLOOR_BYTES = 12 * 1024**3

# ``auto`` concurrency must account for the fixed cost of a worker process, not
# only the bytes in its tile.  A CUDA context plus library handles is hundreds
# of MiB even for a tiny task; host RSS has a similar floor once Python, torch,
# and the fit stack are imported.
_AUTO_CUDA_WORKER_OVERHEAD_BYTES = 512 * 1024**2
_AUTO_HOST_WORKER_OVERHEAD_BYTES = 1024**3
_AUTO_WORKER_HARD_CAP = 8

WorkerLimitReason = Literal[
    "CPU count",
    "GPU memory",
    "GPU memory unavailable",
    "hard cap",
    "host RAM",
    "requested count",
    "selected GPU count",
    "task count",
]


def is_mps_available() -> bool:
    """Return True when PyTorch's MPS backend is available.

    Some older or CPU-only PyTorch builds do not expose ``torch.backends.mps``.
    Centralizing the check keeps device auto-detection robust across builds.
    """
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def resolve_torch_device(
    device: Optional[str | torch.device] = None,
    *,
    use_cuda: bool = True,
    use_metal: bool = True,
) -> torch.device:
    """Resolve an explicit or auto-selected PyTorch device.

    Explicit device names always win except ``"auto"``, which is equivalent to
    ``None``. Auto-selection prefers CUDA over MPS/Metal, and both accelerator
    classes honor their corresponding opt-in flags before falling back to CPU.
    """
    if device is not None and device != "auto":
        return torch.device(device)

    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if use_metal and is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Multi-GPU enumeration & selection (for the local batch-fit runner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    """A visible CUDA device and its memory.

    ``free_mem`` is ``None`` when ``torch.cuda.mem_get_info`` is unavailable for
    the device (rare). GPU selection falls back to ``total_mem``; auto worker
    sizing conservatively assigns one worker to an unqueryable device.
    """

    index: int
    name: str
    total_mem: int  # bytes
    free_mem: Optional[int]  # bytes, or None if unqueryable


@dataclass(frozen=True)
class WorkerLimit:
    """Resolved worker count and the resource that limited it."""

    count: int
    limit: WorkerLimitReason


@dataclass(frozen=True)
class WorkerAllocation:
    """Per-device worker allocation and its limiting resource."""

    workers: dict[int, int]
    limit: WorkerLimitReason


def format_worker_limit(limit: WorkerLimitReason) -> str:
    """Format a worker-limiting reason consistently for user-facing output."""
    return f"limited by {limit}"


def _linux_available_memory_bytes() -> Optional[int]:
    """Read Linux ``MemAvailable`` in bytes, or return ``None``."""
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):  # pragma: no cover - non-Linux/platform
        return None
    for line in meminfo.splitlines():
        key, separator, value = line.partition(":")
        if key != "MemAvailable" or not separator:
            continue
        fields = value.split()
        if not fields:
            return None
        try:
            available_kib = int(fields[0])
        except ValueError:
            return None
        return available_kib * 1024
    return None


def available_host_memory_bytes() -> Optional[int]:
    """Return allocatable host memory in bytes, or ``None`` if unavailable."""
    linux_available = _linux_available_memory_bytes()
    if linux_available is not None:
        return linux_available
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):  # pragma: no cover - platform
        return None
    if pages <= 0 or page_size <= 0:  # pragma: no cover - platform
        return None
    return int(pages * page_size)


def _effective_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity/cgroups."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count is not None:
            return max(1, int(count))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - platform
        return max(1, os.cpu_count() or 1)


def _intraop_thread_count() -> int:
    """Return the largest configured per-worker BLAS/OpenMP thread count."""
    counts = [1]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            counts.append(max(1, int(os.environ.get(name, "1"))))
        except ValueError:
            continue
    return max(counts)


def _auto_worker_hard_cap(minimum: int = 1) -> int:
    """Return the configurable auto-worker cap, never below ``minimum``."""
    override = os.environ.get("LUXAR_AUTO_WORKER_HARD_CAP")
    if override:
        try:
            return max(minimum, int(override))
        except ValueError:
            pass
    return max(minimum, _AUTO_WORKER_HARD_CAP)


def _task_working_set_bytes(
    task_voxels: int, dtype_bytes: int, safety_factor: float
) -> int:
    return max(1, int(safety_factor * task_voxels * dtype_bytes))


def _host_worker_limit(task_bytes: int) -> WorkerLimit:
    cpu_workers = max(1, _effective_cpu_count() // max(2, _intraop_thread_count()))
    candidates: list[tuple[int, WorkerLimitReason]] = [
        (cpu_workers, "CPU count"),
        (_auto_worker_hard_cap(), "hard cap"),
    ]
    available = available_host_memory_bytes()
    if available is not None:
        per_worker = task_bytes + _AUTO_HOST_WORKER_OVERHEAD_BYTES
        candidates.append((max(1, available // per_worker), "host RAM"))
    count, limit = min(candidates, key=lambda item: item[0])
    return WorkerLimit(int(count), limit)


def _gpu_worker_capacity(free_memory: Optional[int], task_bytes: int) -> WorkerLimit:
    if free_memory is None:
        return WorkerLimit(1, "GPU memory unavailable")
    per_worker = task_bytes + _AUTO_CUDA_WORKER_OVERHEAD_BYTES
    return WorkerLimit(max(1, int(free_memory // per_worker)), "GPU memory")


def resolve_auto_worker_limit(
    *,
    task_voxels: int,
    free_device_memory: Optional[int] = None,
    require_device_memory: bool = False,
    dtype_bytes: int = 4,
    safety_factor: float = 2.0,
) -> WorkerLimit:
    """Resolve safe host concurrency for one device and report its limiter."""
    task_bytes = _task_working_set_bytes(task_voxels, dtype_bytes, safety_factor)
    host_limit = _host_worker_limit(task_bytes)
    if not require_device_memory:
        return host_limit
    if free_device_memory is None:
        return WorkerLimit(host_limit.count, "GPU memory unavailable")
    return min(
        (host_limit, _gpu_worker_capacity(free_device_memory, task_bytes)),
        key=lambda item: item.count,
    )


def _allocate_workers(capacities: dict[int, int], total: int) -> dict[int, int]:
    """Distribute a host-wide worker budget proportionally across devices."""
    if total >= len(capacities):
        allocation = {gpu: 1 for gpu in capacities}
        remaining = total - len(capacities)
    else:
        allocation = {gpu: 0 for gpu in capacities}
        remaining = max(1, total)
    for _ in range(remaining):
        eligible = [gpu for gpu, cap in capacities.items() if allocation[gpu] < cap]
        if not eligible:
            break
        gpu = max(
            eligible,
            key=lambda idx: (capacities[idx] / (allocation[idx] + 1), -idx),
        )
        allocation[gpu] += 1
    return {gpu: count for gpu, count in allocation.items() if count > 0}


def enumerate_gpus() -> list[GpuInfo]:
    """List visible CUDA devices with name + total/free memory.

    Returns an empty list when CUDA is unavailable.  Reuses
    :func:`luxar.gsplats.metrics._gpu_free_memory` for the per-device free-byte
    query (it already accepts an indexed ``torch.device``).
    """
    if not torch.cuda.is_available():
        return []
    from luxar.gsplats.metrics import _gpu_free_memory

    infos: list[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        infos.append(
            GpuInfo(
                index=i,
                name=props.name,
                total_mem=int(props.total_memory),
                free_mem=_gpu_free_memory(torch.device(f"cuda:{i}")),
            )
        )
    return infos


def _vram_floor_bytes() -> int:
    """The ``--gpus auto`` free-VRAM floor, honoring the env override."""
    override = os.environ.get("LUXAR_GPU_VRAM_FLOOR_GB")
    if override:
        try:
            return int(float(override) * 1024**3)
        except ValueError:
            pass
    return _AUTO_VRAM_FLOOR_BYTES


def resolve_gpu_selection(spec: str) -> list[int]:
    """Resolve a ``--gpus`` spec to concrete CUDA device indices.

    ``'cpu'``  -> ``[]`` (sentinel: run on CPU, no device pinning).
    ``'all'``  -> every visible CUDA index.
    ``'auto'`` -> every visible index whose memory clears the VRAM floor
                  (see :data:`_AUTO_VRAM_FLOOR_BYTES`); if none qualify but GPUs
                  exist, the single largest card (so a GPU box always uses a GPU).
    ``'0,1,3'``-> those explicit indices, validated against ``device_count()``.

    Raises ``ValueError`` on an unparseable spec, an out-of-range index, or when
    GPU indices are requested but no CUDA device is visible.
    """
    s = (spec or "").strip().lower()
    if s in ("cpu", "none"):
        return []

    gpus = enumerate_gpus()
    if s == "all":
        if not gpus:
            raise ValueError("--gpus all requested but no CUDA device is visible.")
        return [g.index for g in gpus]

    if s == "auto":
        if not gpus:
            raise ValueError(
                "--gpus auto requested but no CUDA device is visible "
                "(use --gpus cpu to fit on CPU)."
            )
        floor = _vram_floor_bytes()

        def _mem(g: GpuInfo) -> int:
            return g.free_mem if g.free_mem is not None else g.total_mem

        qualifying = [g.index for g in gpus if _mem(g) >= floor]
        if qualifying:
            return qualifying
        # Nothing clears the floor — fall back to the single largest card.
        return [max(gpus, key=_mem).index]

    # Explicit comma-separated indices.
    try:
        indices = [int(tok) for tok in s.split(",") if tok != ""]
    except ValueError as exc:
        raise ValueError(
            f"Could not parse --gpus {spec!r}: expected 'auto', 'all', 'cpu', "
            f"or a comma-separated list of device indices (e.g. '0,1')."
        ) from exc
    if not indices:
        raise ValueError(f"--gpus {spec!r} selected no devices.")
    if not gpus:
        raise ValueError(
            f"--gpus {spec!r} requested GPU indices but no CUDA device is visible "
            f"(use --gpus cpu to fit on CPU)."
        )
    n = len(gpus)
    for idx in indices:
        if not (0 <= idx < n):
            raise ValueError(
                f"--gpus index {idx} is out of range (only {n} CUDA "
                f"device(s) visible: 0..{n - 1})."
            )
    # De-duplicate while preserving order.
    seen: set[int] = set()
    unique: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def resolve_jobs_per_gpu(
    gpu_indices: list[int],
    *,
    task_voxels: int,
    jobs_per_gpu: str | int = "auto",
    dtype_bytes: int = 4,
    safety_factor: float = 2.0,
) -> WorkerAllocation:
    """Per-device concurrent-worker count for the local runner.

    Mirrors :func:`luxar.gsplats.fit_tiled_parallel.resolve_jobs` but sizes each
    selected GPU independently from *its own* free VRAM (so a small card gets
    fewer workers than a large one), then applies host RAM, CPU-count, and hard
    caps to their summed concurrency. An explicit int applies uniformly.

    ``gpu_indices == []`` means CPU: returns ``{-1: n}`` where ``n`` is the
    explicit count or the same host-level auto limit.
    """
    """Resolve per-device workers and the limiting resource."""
    explicit: Optional[int] = None
    if not (isinstance(jobs_per_gpu, str) and jobs_per_gpu.strip().lower() == "auto"):
        explicit = max(1, int(jobs_per_gpu))

    if not gpu_indices:  # CPU sentinel
        if explicit is not None:
            return WorkerAllocation({-1: explicit}, "requested count")
        task_bytes = _task_working_set_bytes(task_voxels, dtype_bytes, safety_factor)
        host_limit = _host_worker_limit(task_bytes)
        return WorkerAllocation({-1: host_limit.count}, host_limit.limit)

    from luxar.gsplats.metrics import _gpu_free_memory

    if explicit is not None:
        return WorkerAllocation(
            {idx: explicit for idx in gpu_indices}, "requested count"
        )

    task_bytes = _task_working_set_bytes(task_voxels, dtype_bytes, safety_factor)
    capacities: dict[int, int] = {}
    unknown_device_memory = 0
    for idx in gpu_indices:
        free = _gpu_free_memory(torch.device(f"cuda:{idx}"))
        capacity = _gpu_worker_capacity(free, task_bytes)
        capacities[idx] = capacity.count
        if free is None:
            unknown_device_memory += 1

    host_limit = _host_worker_limit(task_bytes)
    device_total = sum(capacities.values())
    unconstrained_total = min(device_total, host_limit.count)
    total = max(len(capacities), unconstrained_total)
    if unconstrained_total < len(capacities):
        limiting_resource: WorkerLimitReason = "selected GPU count"
    elif device_total <= host_limit.count:
        limiting_resource = (
            "GPU memory unavailable" if unknown_device_memory > 0 else "GPU memory"
        )
    else:
        limiting_resource = host_limit.limit
    return WorkerAllocation(_allocate_workers(capacities, total), limiting_resource)
