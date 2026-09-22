"""Batch job manifest: tracks all tasks, parameters, and submission state."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def allocate_weighted_integer_seeds(
    total: int, weights: Sequence[float]
) -> Tuple[int, ...]:
    """Allocate proportional integer seeds with deterministic largest remainders."""
    positive = [index for index, weight in enumerate(weights) if weight > 0.0]
    allocation = [0] * len(weights)
    if not positive or total <= 0:
        return tuple(allocation)
    if total < len(positive):
        for index in positive:
            allocation[index] = 1
        return tuple(allocation)

    remaining_indices = positive
    remaining_total = total
    while remaining_indices:
        weight_sum = sum(weights[index] for index in remaining_indices)
        clamped = [
            index
            for index in remaining_indices
            if remaining_total * weights[index] / weight_sum < 1.0
        ]
        if not clamped:
            break
        for index in clamped:
            allocation[index] = 1
        remaining_total -= len(clamped)
        remaining_indices = [
            index for index in remaining_indices if index not in clamped
        ]

    if not remaining_indices:
        return tuple(allocation)
    weight_sum = sum(weights[index] for index in remaining_indices)
    shares = [
        remaining_total * weights[index] / weight_sum for index in remaining_indices
    ]
    floors = [math.floor(share) for share in shares]
    for index, count in zip(remaining_indices, floors, strict=True):
        allocation[index] = count
    remainder = remaining_total - sum(floors)
    ranked = sorted(
        range(len(remaining_indices)),
        key=lambda offset: (
            -(shares[offset] - floors[offset]),
            remaining_indices[offset],
        ),
    )
    for offset in ranked[:remainder]:
        allocation[remaining_indices[offset]] += 1
    return tuple(allocation)


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
    """Explicit axis-order override (e.g. ``'t,c,z,y,x'``) used to discover and
    reload the source dataset. Direct fit tasks receive it as ``fit --axes``;
    preprocess fits receive the corresponding canonical ``t,c,*spatial`` axes
    for ``denoised.zarr``. Without it, source readers retain the positional ndim
    heuristic while preprocess fits derive canonical axes from the recorded
    worker-visible spatial rank."""

    # Dataset shape
    n_timepoints: int = 1
    n_channels: int = 1
    spatial_shape: Tuple[int, ...] = ()
    """Spatial shape emitted by workers; positional loading removes singleton axes."""

    dimension_metadata: Optional[List[Dict[str, Any]]] = None
    """Merged center-axis names, units, and physical scales in output order."""

    # Tiling
    mode: str = "uniform"
    """Decomposition: ``uniform`` (a regular tile grid) or ``content`` (a shared
    content-balanced ``FitPlan`` of boxes — see ``plan_path``)."""
    tile_size: int = 256
    tile_overlap: int = 32
    fold_tile_slivers: bool = False
    """Whether this manifest uses overlap-dominated trailing-tile folding."""
    n_tiles: int = 1
    """Spatial slots per (t,c): tile count in ``uniform`` mode, box count in
    ``content`` mode (kept as one field so task-id decode / packing are shared)."""
    tile_local_reads: bool = False
    """Whether uniform workers may read only their manifest-derived tile region."""
    tile_nonempty_counts: Optional[List[int]] = None
    """One floor + Hann-window non-empty tile count per selected ``(t, c)``.

    Only positive integer whole-volume seed budgets need these counts. Tile-local
    reads remain enabled without them for auto seeds and compression ratios. New
    plans persist the corresponding per-tile weights alongside these counts.
    """
    tile_occupancy_weights: Optional[List[List[float]]] = None
    """One mass-weight row per selected ``(t, c)`` slice.

    Rows already carry the minimum-share floor
    (:func:`luxar.gsplats.fit_tiled_gsplats.floor_tile_seed_shares`). A
    proportional integer allocation of each row is the final per-tile budget.
    """
    plan_path: Optional[str] = None
    """``content`` mode: path to the shared ``FitPlan`` JSON (relative to
    ``output_dir``) every array task reads via ``fit --plan … --plan-box``."""
    total_tasks: int = 1

    # Fit config
    preset: Optional[str] = None
    fit_args: Dict[str, Any] = field(default_factory=dict)

    floor_level: Optional[float] = None
    """The ONE background floor level resolved before fitting, subtracted by every
    ``(t, c)`` task. Resolution normally happens at plan time; denoised-basis
    plans defer it to one dependent job after calibration/preprocessing.
    ``batch-fit`` deliberately uses a single global level for the
    whole timelapse rather than letting each task re-estimate on its own
    sub-volume, which would be a time-varying pedestal (brightness flicker) across
    the merged partition. It is the MINIMUM of the levels resolved on a bounded set
    of evenly spaced ``(t, c)`` slices spanning the store's full extent (see
    :func:`luxar.cli.gsplat_ops.batch.planning.resolve_batch_floor`): a minimum is
    a lower bound on every SAMPLED slice's pedestal, so it cannot clip a sampled
    sub-volume to zero — a
    dimmer non-sampled slice still can, bounded sampling being bounded — and it
    does not depend on the ``--timepoints`` selection for raw and on-the-fly
    denoise runs. Raw-basis resolution also ignores ``--channels``; on-the-fly
    denoised resolution follows the selected channels because NLM ``h`` is
    calibrated per channel. Preprocess runs necessarily resolve against the
    selected-only ``denoised.zarr`` store, so both axes are selection-scoped. The
    level that actually drives the fits is the concrete number in ``fit_args["floor"]``;
    this field records it for inspection and merge provenance. ``None`` means "no level is
    pinned": suppression is disabled, or the resolved level was negative so the
    SPEC was forwarded and each task resolves it itself, or the manifest predates
    this field — in the last two cases the run keeps its recorded ``fit_args``
    floor SPEC, so a resumed old batch behaves exactly as it did when planned."""

    floor_spec: Any = None
    """Original effective floor spec when resolution must happen after denoising."""

    floor_deferred: bool = False
    """Whether a dependent runtime stage resolves ``floor_spec`` on denoised data."""

    norm_range: Optional[Tuple[float, float]] = None
    """The ONE raw-input normalization range forwarded to every fit task.

    The range is measured from bounded samples of the same representative slices
    used for ``floor_level``. Denoising runs leave it unset unless the user supplied
    an explicit range, so each task resolves against the data it actually fits.
    ``None`` means the manifest predates shared normalization or no usable range
    could be resolved, so tasks retain their historical per-sub-volume behavior.
    """

    grid_scale: Optional[List[float]] = None
    """Per-axis factor mapping the VOXEL tile grid onto the frame the fit tasks'
    splats actually come back in (uniform tiles or content boxes; #1587).

    Uniform tasks can inherit ``voxel_size`` from their fit config; content tasks
    use physical coordinates only under the explicit batch ``--physical`` opt-in.
    Either can make workers emit PHYSICAL centers while ``spatial_shape`` and a
    content ``FitPlan`` remain in voxels. The planner records that factor ONCE so
    merge-time consumers never need to re-read a config that may have moved. A
    config ``downscale:`` is deliberately NOT a term: every task rescales its
    splats back to the full-resolution frame the planner tiled (#1624).

    ``None`` means NO scale — the tile grid and the splats share a frame. That
    is both the overwhelmingly common case and what a manifest written before
    this field existed says by omission, which is exactly the pre-#1587
    behaviour: build the planes straight off the voxel grid."""

    # GPU + time
    gpu_name: str = ""
    estimated_seconds_per_task: float = 0.0
    slurm_time_limit: str = "02:00:00"

    # Slurm params
    slurm_partition: str = ""
    slurm_account: Optional[str] = None
    slurm_qos: Optional[str] = None
    slurm_gpus: int = 1
    """GPUs per task, from ``batch-fit submit --gpus-per-task``.

    Emitted verbatim as ``#SBATCH --gpus-per-task``. Keeps the ``slurm_*`` field
    name (persisted manifests resume from it) though the flag is now spelled for
    the directive — the short ``--gpus`` collided with ``batch-fit run --gpus``,
    which SELECTS local devices rather than counting them.
    """

    slurm_cpus: int = 4
    slurm_mem_gb: int = 32
    slurm_cpus_total: int = 0
    """Total CPUs requested by the fit allocation; 0 falls back to ``slurm_cpus``."""

    slurm_mem_gb_total: int = 0
    """Total RAM requested by the fit allocation; 0 falls back to ``slurm_mem_gb``."""

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
    floor_job_id: Optional[int] = None
    array_job_id: Optional[int] = None
    preemptible_job_id: Optional[int] = None
    merge_job_id: Optional[int] = None


@dataclass(frozen=True)
class TileLocalReadPlan:
    """Validated manifest-derived worker regions and optional seed divisors."""

    regions: Tuple[str, ...]
    volume_shape: str
    nonempty_counts: Optional[Tuple[int, ...]]
    seed_counts: Optional[Tuple[Tuple[int, ...], ...]]


def _positive_manifest_seed_budget(manifest: BatchManifest) -> int:
    """Return the positive integer budget required by occupancy metadata."""
    raw_seeds = manifest.fit_args.get("seeds")
    if isinstance(raw_seeds, bool) or not isinstance(raw_seeds, (int, str)):
        raise ValueError("tile occupancy weights require positive integer seeds")
    try:
        total_seeds = int(raw_seeds)
    except ValueError as exc:
        raise ValueError(
            "tile occupancy weights require positive integer seeds"
        ) from exc
    if total_seeds <= 0:
        raise ValueError("tile occupancy weights require positive integer seeds")
    return total_seeds


def _validated_occupancy_seed_counts(
    manifest: BatchManifest, counts: Optional[List[int]]
) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """Validate persisted occupancy rows and derive exact worker counts."""
    weights = manifest.tile_occupancy_weights
    if weights is None:
        return None
    if counts is None:
        raise ValueError("tile occupancy weights require non-empty counts")
    expected_rows = manifest.n_timepoints * manifest.n_channels
    if len(weights) != expected_rows:
        raise ValueError(
            "tile occupancy weight mismatch: manifest records "
            f"{len(weights)} rows but T*C requires {expected_rows}"
        )
    normalized_rows: list[Tuple[float, ...]] = []
    for row_index, row in enumerate(weights):
        if len(row) != manifest.n_tiles:
            raise ValueError(
                f"tile occupancy row {row_index} has {len(row)} weights; "
                f"expected {manifest.n_tiles}"
            )
        normalized = tuple(float(weight) for weight in row)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in normalized):
            raise ValueError("tile occupancy weights must be finite and non-negative")
        if sum(weight > 0.0 for weight in normalized) != counts[row_index]:
            raise ValueError(
                f"tile occupancy row {row_index} disagrees with its non-empty count"
            )
        normalized_rows.append(normalized)

    total_seeds = _positive_manifest_seed_budget(manifest)
    # A zero allocation is safe only because zero-weight tiles are skipped before fit.
    return tuple(
        allocate_weighted_integer_seeds(total_seeds, row) for row in normalized_rows
    )


def tile_local_read_plan(manifest: BatchManifest) -> Optional[TileLocalReadPlan]:
    """Resolve the single local/Slurm tile-local worker handoff."""
    if not manifest.tile_local_reads:
        return None
    if manifest.mode != "uniform":
        raise ValueError("tile-local reads require a uniform batch manifest")
    if not manifest.spatial_shape:
        raise ValueError("tile-local reads require a non-empty spatial_shape")

    from luxar.gsplats.tiling import compute_tile_specs

    specs = compute_tile_specs(
        tuple(manifest.spatial_shape),
        manifest.tile_size,
        manifest.tile_overlap,
        fold_slivers=manifest.fold_tile_slivers,
    )
    if len(specs) != manifest.n_tiles:
        raise ValueError(
            "tile-local grid mismatch: manifest records "
            f"{manifest.n_tiles} tiles but spatial_shape/tile settings produce "
            f"{len(specs)}"
        )
    regions = tuple(
        ",".join(f"{span.start}:{span.stop}" for span in spec.slices) for spec in specs
    )

    counts = manifest.tile_nonempty_counts
    if counts is not None:
        expected_rows = manifest.n_timepoints * manifest.n_channels
        if len(counts) != expected_rows:
            raise ValueError(
                "tile-local count mismatch: manifest records "
                f"{len(counts)} rows but T*C requires {expected_rows}"
            )
        if any(count <= 0 or count > manifest.n_tiles for count in counts):
            raise ValueError(
                "tile-local non-empty counts must be between 1 and n_tiles"
            )

    seed_counts = _validated_occupancy_seed_counts(manifest, counts)

    return TileLocalReadPlan(
        regions=regions,
        volume_shape=",".join(str(size) for size in manifest.spatial_shape),
        nonempty_counts=tuple(counts) if counts is not None else None,
        seed_counts=seed_counts,
    )


def floor_suppression_applied(manifest: BatchManifest) -> bool:
    """Whether workers subtract a non-zero background level/spec."""
    value = manifest.fit_args.get("floor")
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("", "none"):
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return True


def floor_erased_slices(
    manifest: BatchManifest, tiles_dir: Path
) -> set[tuple[int, int]]:
    """Return uniform ``(t, c)`` pairs wholly empty under an applied floor.

    Content plans may legitimately place no box over a slice, so an all-empty
    content slot is ambiguous and deliberately excluded.
    """
    if manifest.mode != "uniform" or not floor_suppression_applied(manifest):
        return set()
    by_slice: dict[tuple[int, int], list[BatchJob]] = {}
    for job in manifest.jobs:
        by_slice.setdefault((job.timepoint, job.channel), []).append(job)
    erased = set()
    for pair, jobs in by_slice.items():
        if jobs and all(
            Path(f"{tiles_dir / job.output_filename}.empty").exists() for job in jobs
        ):
            erased.add(pair)
    return erased


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
    if data.get("norm_range") is not None:
        data["norm_range"] = tuple(data["norm_range"])

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
