"""Plan-summary output for ``batch-fit submit``."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from arbol import aprint


def print_batch_submit_plan(
    *,
    input_name: str,
    n_t: int,
    n_c: int,
    spatial: Sequence[int],
    resolved_gpu: str,
    peak_gvs: object,
    peak_shape: Sequence[int],
    mode: str,
    n_tiles: int,
    tile_overlap: int,
    needs_tiling: bool,
    tile_size: int,
    auto_tile: bool,
    total_tasks: int,
    tasks_per_job: int,
    packing_limit: str,
    parallel: bool,
    n_slurm_jobs: int,
    mps_available_fn: Callable[[], bool],
    uses_backfill: bool,
    no_job_limit: bool,
    est_seconds: float,
    preset: str,
    n_iters: int,
    est_seconds_per_job: float,
    total_gpu_hours: float,
    slurm_time: str,
    partition: str,
    gpus_per_task: int,
    cpus_per_task: int,
    mem_gb_per_task: int,
    cpus_total: int,
    mem_gb_total: int,
    output_dir: Path,
) -> None:
    """Print the human-facing batch-plan summary (stable CLI output)."""
    spatial_str = "x".join(str(s) for s in spatial)
    peak_shape_str = "x".join(str(s) for s in peak_shape)

    aprint("")
    aprint("=" * 60)
    aprint("BATCH PLAN")
    aprint("=" * 60)
    aprint(f"  Input: {input_name} (T={n_t}, C={n_c}, spatial={spatial_str})")
    aprint(f"  GPU: {resolved_gpu} (peak: {peak_gvs} GV/s at {peak_shape_str})")
    if mode == "content":
        aprint(
            f"  Decomposition: content plan, {n_tiles} boxes/volume "
            f"(overlap={tile_overlap}); shared across all (t,c)"
        )
    elif needs_tiling:
        aprint(
            f"  Tile: {tile_size}^{len(spatial)}"
            f" ({'auto' if auto_tile else 'manual'})"
            f", overlap={tile_overlap}, {n_tiles} tiles/volume"
        )
    else:
        aprint("  Tile: not needed (volume fits in GPU memory)")

    slot = "boxes" if mode == "content" else "tiles"
    aprint(f"  Jobs: {n_t} x {n_c} x {n_tiles} = {total_tasks} fitting tasks ({slot})")

    if parallel or tasks_per_job > 1:
        run_mode = "parallel" if parallel else "sequential"
        mps_note = ""
        if parallel:
            # Probed lazily: `scontrol show config` (subprocess, 5 s timeout)
            # only runs when the note is actually printed.
            if mps_available_fn():
                mps_note = " [MPS available]"
            else:
                mps_note = " [bash background processes]"
        aprint(
            f"  Packing: {tasks_per_job} tasks/job ({run_mode}) "
            f"→ {n_slurm_jobs} Slurm jobs{mps_note}; limited by {packing_limit}"
        )
        if parallel:
            aprint(
                f"  Fit allocation: {cpus_per_task} CPUs/task × {tasks_per_job} = "
                f"{cpus_total} CPUs, {mem_gb_per_task}G RAM/task × "
                f"{tasks_per_job} = {mem_gb_total}G RAM"
            )
    else:
        aprint(f"  Slurm array: {total_tasks} jobs (1 task each)")

    if uses_backfill:
        sched_note = "backfill scheduler — short jobs get scheduled fastest"
        if no_job_limit:
            sched_note += ", no job count limit"
        aprint(f"  Scheduler: {sched_note}")

    aprint(
        f"  Est. time/task: ~{est_seconds / 60:.0f} min (preset: {preset}, {n_iters} iters)"
    )
    if tasks_per_job > 1:
        if parallel:
            aprint(
                f"  Est. time/job: ~{est_seconds_per_job / 60:.0f} min "
                f"({tasks_per_job} tasks in parallel)"
            )
        else:
            aprint(
                f"  Est. time/job: ~{est_seconds_per_job / 60:.0f} min "
                f"({tasks_per_job} tasks × {est_seconds / 60:.0f} min)"
            )

    aprint(f"  Est. total GPU-hours: {total_gpu_hours:.0f} h")
    aprint(f"  Slurm --time: {slurm_time}")
    aprint(
        f"  Partition: {partition}, GPUs/task: {gpus_per_task}, "
        f"CPUs/task: {cpus_per_task}, Mem/task: {mem_gb_per_task}G"
    )
    aprint(f"  Output: {output_dir}")
    aprint("")
