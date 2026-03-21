"""Batch job manifest: tracks all tasks, parameters, and submission state."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BatchJob:
    """A single fitting task within a batch."""

    task_id: int
    """Flat ``SLURM_ARRAY_TASK_ID``."""

    timepoint: int
    """Timepoint index (T)."""

    channel: int
    """Channel index (C)."""

    tile_index: int
    """Tile index within the spatial grid."""

    output_filename: str
    """Relative path from ``output_dir/tiles/``."""

    estimated_wall_seconds: float
    """Estimated wall time in seconds."""


@dataclass
class BatchManifest:
    """Complete batch job manifest persisted as JSON."""

    # Metadata
    version: int = 1
    created: str = ""

    # Input
    input_path: str = ""
    output_dir: str = ""

    # Dataset shape
    n_timepoints: int = 1
    n_channels: int = 1
    spatial_shape: Tuple[int, ...] = ()

    # Tiling
    tile_size: int = 256
    tile_overlap: int = 32
    n_tiles: int = 1
    total_tasks: int = 1

    # Fit config
    preset: Optional[str] = None
    fit_args: Dict[str, Any] = field(default_factory=dict)

    # GPU + time
    gpu_name: str = ""
    estimated_seconds_per_task: float = 0.0
    slurm_time_limit: str = "02:00:00"

    # Slurm params
    slurm_partition: str = ""
    slurm_account: Optional[str] = None
    slurm_qos: Optional[str] = None
    slurm_gpus: int = 1
    slurm_cpus: int = 4
    slurm_mem_gb: int = 32
    slurm_extra_args: List[str] = field(default_factory=list)

    # Packing
    tasks_per_job: int = 1
    """Number of fitting tasks to run within each Slurm job."""

    parallel_tasks_per_job: bool = False
    """If True, tasks within a job run concurrently (background processes).
    If False (default), they run sequentially."""

    # Jobs
    jobs: List[BatchJob] = field(default_factory=list)

    # Merge config
    channel_colors: Optional[List[str]] = None

    # Index arrays (when --timepoints/--channels slicing is used)
    timepoint_indices: Optional[List[int]] = None
    """Actual timepoint indices into the dataset, or None for contiguous 0..n_t-1."""

    channel_indices: Optional[List[int]] = None
    """Actual channel indices into the dataset, or None for contiguous 0..n_c-1."""

    # Post-submit state
    array_job_id: Optional[int] = None
    merge_job_id: Optional[int] = None


def save_manifest(manifest: BatchManifest, output_dir: Path) -> Path:
    """Write ``manifest.json`` to *output_dir*.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"

    data = asdict(manifest)
    # Convert tuples to lists for JSON (asdict converts to lists already,
    # but be explicit for clarity)
    data["spatial_shape"] = list(manifest.spatial_shape)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return path


def load_manifest(output_dir: Path) -> BatchManifest:
    """Load ``manifest.json`` from *output_dir*."""
    path = output_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest.json found in {output_dir}")

    with open(path) as f:
        data = json.load(f)

    # Reconstruct BatchJob objects
    job_fields = {f.name for f in dataclasses.fields(BatchJob)}
    jobs = [
        BatchJob(**{k: v for k, v in j.items() if k in job_fields})
        for j in data.pop("jobs", [])
    ]
    # Convert spatial_shape back to tuple
    data["spatial_shape"] = tuple(data.get("spatial_shape", ()))

    # Filter to known fields (forward-compatible with newer manifests)
    known_fields = {f.name for f in dataclasses.fields(BatchManifest)}
    manifest = BatchManifest(**{k: v for k, v in data.items() if k in known_fields})
    manifest.jobs = jobs
    return manifest


def decode_task_id(task_id: int, manifest: BatchManifest) -> Tuple[int, int, int]:
    """Decode a flat task ID to ``(timepoint, channel, tile_index)``.

    Encoding: ``task_id = t * (C * K) + c * K + k``
    where ``C = n_channels``, ``K = n_tiles``.
    """
    n_c = manifest.n_channels
    n_k = manifest.n_tiles
    t = task_id // (n_c * n_k)
    remainder = task_id % (n_c * n_k)
    c = remainder // n_k
    k = remainder % n_k
    return (t, c, k)


def output_filename(
    t: int,
    c: int,
    k: int,
    n_timepoints: int = 100,
    n_channels: int = 100,
    n_tiles: int = 1000,
) -> str:
    """Generate canonical output filename for a task.

    Widths are computed from the max index so filenames sort lexicographically.
    """
    tw = max(2, len(str(max(0, n_timepoints - 1))))
    cw = max(2, len(str(max(0, n_channels - 1))))
    kw = max(3, len(str(max(0, n_tiles - 1))))
    return f"t{t:0{tw}d}_c{c:0{cw}d}_tile{k:0{kw}d}.gsplats.zarr"
