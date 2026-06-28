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
    """Channel index (C). For sliced batches, this is the real flat dataset index."""

    tile_index: int
    """Tile index within the spatial grid."""

    output_filename: str
    """Relative path from ``output_dir/tiles/``."""

    estimated_wall_seconds: float
    """Estimated wall time in seconds."""

    channel_coords: Tuple[int, ...] = ()
    """Coordinates in folded channel-like axes, if known."""


@dataclass
class BatchManifest:
    """Complete batch job manifest persisted as JSON."""

    # Metadata
    version: int = 1
    created: str = ""

    # Input
    input_path: str = ""
    output_dir: str = ""

    # Array selection
    array_key: Optional[str] = None
    """Key path to a specific array within the zarr store (e.g. 'h2afva/fused')."""

    axes: Optional[str] = None
    """Explicit axis-order override (e.g. ``'t,c,z,y,x'``) used both to discover
    the dataset shape AND forwarded to every fit task (``fit --axes``). Without
    it, tasks fall back to the positional ndim heuristic — which must then agree
    with the shape the planner used, or the tile grid / merge corrupts."""

    # Dataset shape
    n_timepoints: int = 1
    n_channels: int = 1
    spatial_shape: Tuple[int, ...] = ()

    # Tiling
    mode: str = "uniform"
    """Decomposition: ``uniform`` (a regular tile grid) or ``content`` (a shared
    content-balanced ``FitPlan`` of boxes — see ``plan_path``)."""
    tile_size: int = 256
    tile_overlap: int = 32
    n_tiles: int = 1
    """Spatial slots per (t,c): tile count in ``uniform`` mode, box count in
    ``content`` mode (kept as one field so task-id decode / packing are shared)."""
    plan_path: Optional[str] = None
    """``content`` mode: path to the shared ``FitPlan`` JSON (relative to
    ``output_dir``) every array task reads via ``fit --plan … --plan-box``."""
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

    max_concurrent: Optional[int] = None
    """Maximum number of Slurm array tasks running simultaneously.
    Maps to ``--array=0-N%MAX``. None means no limit."""

    preemptible: bool = False
    """Whether a preemptible array job is submitted alongside the main one."""

    preemptible_partition: Optional[str] = None
    """Name of the preemptible partition (auto-detected or explicit)."""

    preemptible_max_concurrent: Optional[int] = None
    """Max concurrent tasks on preemptible partition."""

    # Jobs
    jobs: List[BatchJob] = field(default_factory=list)

    # Merge config
    channel_colors: Optional[List[str]] = None

    merge_recipe: Optional[str] = None
    """Per-part LOD recipe applied to each spatial tile-part at merge time
    (``additive`` / ``substitutive``). ``None`` = bare-leaf parts
    (no per-part LOD; the historical partition output). ``additive`` yields the
    ``partitioned`` topology (each part a prefix-sum ladder); ``substitutive``
    yields the ``mosaic`` topology (each part its own coarse↔fine lod group)."""

    merge_recipe_args: Dict[str, Any] = field(default_factory=dict)
    """Extra knobs for the per-part merge recipe, mirroring the ``lod`` CLI:
    ``n-lods`` / ``compression-factor`` / ``levels`` / ``substitutive-method`` /
    ``coarsen-dims``. Threaded verbatim into the merge sbatch command."""

    # Index arrays (when --timepoints/--channels slicing is used)
    timepoint_indices: Optional[List[int]] = None
    """Actual timepoint indices into the dataset, or None for contiguous 0..n_t-1."""

    channel_indices: Optional[List[int]] = None
    """Actual flat channel indices into the dataset, or None for contiguous 0..n_c-1."""

    channel_axes: List[str] = field(default_factory=list)
    """Names of axes folded into the flat channel index."""

    channel_shape: Tuple[int, ...] = ()
    """Shape of axes folded into the flat channel index."""

    # Denoise config
    denoise: bool = False
    """Whether NLM denoising is enabled."""

    denoise_2d: bool = False
    """Use 2D NLM (slice-by-slice) instead of 3D."""

    denoise_h: Optional[float] = None
    """Manual h override (None = auto-calibrate)."""

    denoise_patch_size: int = 3
    denoise_search_distance: int = 5
    denoise_backend: str = "auto"

    denoise_mode: Optional[str] = None
    """'preprocess' or 'on-the-fly'. Set by batch plan based on n_tiles."""

    denoise_h_values: Optional[Dict[str, float]] = None
    """Per-channel calibrated h values. Key is str(channel_index), value is h."""

    denoised_zarr_path: Optional[str] = None
    """Path to denoised.zarr when denoise_mode='preprocess'."""

    calibration_samples: int = 5
    """Number of timepoints to sample for h calibration."""

    # Post-submit state
    calibrate_job_id: Optional[int] = None
    denoise_job_id: Optional[int] = None
    array_job_id: Optional[int] = None
    preemptible_job_id: Optional[int] = None
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
    # Convert tuple-valued fields back to tuples
    data["spatial_shape"] = tuple(data.get("spatial_shape", ()))
    data["channel_shape"] = tuple(data.get("channel_shape", ()))

    # Filter to known fields (forward-compatible with newer manifests)
    known_fields = {f.name for f in dataclasses.fields(BatchManifest)}
    manifest = BatchManifest(**{k: v for k, v in data.items() if k in known_fields})
    for job in jobs:
        job.channel_coords = tuple(job.channel_coords)
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
    label: str = "tile",
) -> str:
    """Generate canonical output filename for a task.

    Widths are computed from the max index so filenames sort lexicographically.
    ``label`` is ``tile`` (uniform) or ``box`` (content) — the spatial-slot kind.
    """
    tw = max(2, len(str(max(0, n_timepoints - 1))))
    cw = max(2, len(str(max(0, n_channels - 1))))
    kw = max(3, len(str(max(0, n_tiles - 1))))
    return f"t{t:0{tw}d}_c{c:0{cw}d}_{label}{k:0{kw}d}.gsplats.zarr"
