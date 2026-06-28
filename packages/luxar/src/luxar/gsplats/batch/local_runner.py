# local_runner.py
"""Local (non-Slurm) execution of a batch manifest across multiple GPUs.

The local counterpart of the Slurm fit array + merge: given a planned
:class:`BatchManifest`, fit every ``(t, c, slot)`` task with a multi-GPU
subprocess pool, then run the existing streaming merge to a ``kind=partition``
``.gsplats.zarr``.  Workers are pinned to GPUs via ``CUDA_VISIBLE_DEVICES``
(:mod:`task_pool`'s env hook); per-GPU concurrency is sized from each card's free
VRAM.  Resumable: a task whose output already exists is skipped.

This engine consumes only the manifest + already-parsed merge options, so it has
no dependency on the CLI layer.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

from arbol import aprint, asection

from luxar.gsplats.batch.fit_command import build_task_fit_argv
from luxar.gsplats.batch.manifest import BatchJob, BatchManifest
from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
from luxar.gsplats.batch.task_pool import TaskResult, run_task_pool
from luxar.gsplats.utils.device import resolve_gpu_selection, resolve_jobs_per_gpu


def _task_voxels(manifest: BatchManifest) -> int:
    """Working-set proxy (voxels) for VRAM-based concurrency sizing.

    Content mode: the largest padded box of the shared plan. Uniform: the tile
    volume (capped at the whole volume). Falls back to total voxels.
    """
    total = math.prod(manifest.spatial_shape) if manifest.spatial_shape else 1
    if manifest.mode == "content" and manifest.plan_path:
        try:
            from luxar.gsplats.planner import FitPlan
            from luxar.gsplats.planner.fit_planned_parallel import (
                max_padded_box_voxels,
            )

            plan = FitPlan.from_json(Path(manifest.plan_path))
            return max(1, max_padded_box_voxels(plan))
        except Exception:
            return max(1, total)
    if manifest.tile_size and manifest.tile_size > 0 and manifest.spatial_shape:
        tv = int(manifest.tile_size ** len(manifest.spatial_shape))
        return max(1, min(tv, total))
    return max(1, total)


def build_device_assignment(
    task_ids: List[int], workers: dict[int, int]
) -> dict[int, int]:
    """Map each ``task_id`` to a GPU index (``-1`` = CPU).

    Round-robin weighted by per-GPU worker count: a card with 4 workers gets ~4x
    the tasks of a 1-worker card. Because the pool launches ``sum(workers)`` tasks
    at once and the slot list repeats each GPU ``count`` times, the first
    concurrent wave fills exactly each card's capacity.
    """
    if not workers or list(workers.keys()) == [-1]:
        return {tid: -1 for tid in task_ids}
    slots: list[int] = []
    for gpu, count in workers.items():
        slots.extend([gpu] * max(1, count))
    if not slots:
        slots = [next(iter(workers))]
    return {tid: slots[i % len(slots)] for i, tid in enumerate(task_ids)}


def _finalize_output(out: Path) -> Tuple[bool, bool]:
    """Promote a worker's ``{out}.tmp`` to its final path.

    Mirrors the Slurm ``run_task`` finalization: a sibling ``{out}.tmp.empty``
    marker (0-splat box) becomes ``{out}.empty``; otherwise the ``.tmp`` store is
    atomically renamed to ``out``. Returns ``(ok, empty)``. ``ok`` is False when
    the worker exited 0 but left nothing usable.
    """
    tmp = Path(str(out) + ".tmp")
    tmp_empty = Path(str(out) + ".tmp.empty")
    if tmp_empty.exists():
        tmp_empty.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        Path(str(out) + ".empty").touch()
        return True, True
    if tmp.exists():
        os.replace(tmp, out)  # atomic within the same filesystem
        return True, False
    return False, False


def run_batch_local(
    manifest: BatchManifest,
    output_dir: Path,
    *,
    gpus: str = "auto",
    jobs_per_gpu: str | int = "auto",
    resume: bool = True,
    force_merge: bool = False,
    channel_colors: Optional[List[Tuple[float, float, float]]] = None,
    recipe: Optional[str] = None,
    recipe_params: "Optional[Any]" = None,
    verbose: bool = True,
) -> Path:
    """Fit every task in ``manifest`` locally across GPUs, then merge.

    Parameters
    ----------
    gpus
        ``--gpus`` spec: ``auto`` (cards above a VRAM floor) / ``all`` / ``cpu`` /
        ``'0,1,3'``.
    jobs_per_gpu
        Concurrent fit workers per GPU (``auto`` sizes from each card's free VRAM).
    resume
        Skip tasks whose output (or ``.empty`` marker) already exists.
    channel_colors, recipe, recipe_params
        Forwarded to :func:`merge_batch_results` (parsed by the caller). ``recipe``
        defaults to ``manifest.merge_recipe``.

    Returns
    -------
    Path
        The merged ``kind=partition`` ``.gsplats.zarr``.
    """
    output_dir = Path(output_dir)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "merged").mkdir(parents=True, exist_ok=True)

    gpu_indices = resolve_gpu_selection(gpus)
    task_ids = [j.task_id for j in manifest.jobs]
    workers = resolve_jobs_per_gpu(
        gpu_indices,
        task_voxels=_task_voxels(manifest),
        n_tasks=len(task_ids),
        jobs_per_gpu=jobs_per_gpu,
    )
    assignment = build_device_assignment(task_ids, workers)
    global_workers = max(1, min(sum(workers.values()), len(task_ids)))

    job_by_id = {j.task_id: j for j in manifest.jobs}

    def _out_path(job: BatchJob) -> Path:
        return tiles_dir / job.output_filename

    def _denoise_h(job: BatchJob) -> Optional[float]:
        if (
            manifest.denoise
            and manifest.denoise_mode == "on-the-fly"
            and manifest.denoise_h is None
            and manifest.denoise_h_values
        ):
            return manifest.denoise_h_values.get(str(job.channel))
        return None

    def _skip(task_id: int) -> bool:
        out = _out_path(job_by_id[task_id])
        return resume and (out.exists() or Path(str(out) + ".empty").exists())

    def _argv(task_id: int) -> list[str]:
        job = job_by_id[task_id]
        out = _out_path(job)
        tmp = Path(str(out) + ".tmp")
        # Clear a stale .tmp from an interrupted run so fit writes cleanly.
        shutil.rmtree(tmp, ignore_errors=True)
        Path(str(out) + ".tmp.empty").unlink(missing_ok=True)
        return build_task_fit_argv(manifest, job, tmp, denoise_h=_denoise_h(job))

    def _env(task_id: int) -> dict[str, str]:
        gpu = assignment.get(task_id, -1)
        return {} if gpu < 0 else {"CUDA_VISIBLE_DEVICES": str(gpu)}

    n_run = sum(0 if _skip(t) else 1 for t in task_ids)
    dev_desc = "CPU" if not gpu_indices else f"GPU(s) {gpu_indices}"
    with asection(
        f"Local batch fit: {n_run}/{len(task_ids)} tasks on {dev_desc}, "
        f"{global_workers} concurrent worker(s)"
    ):
        if verbose and n_run < len(task_ids):
            aprint(f"Resuming: {len(task_ids) - n_run} task(s) already complete")

        def _on_done(res: TaskResult, done: int, total: int) -> None:
            if not verbose:
                return
            job = job_by_id[res.key]
            tag = f"(T={job.timepoint} C={job.channel} K={job.tile_index})"
            if res.skipped:
                aprint(f"[{done}/{total}] skip {tag} (exists)")
            elif res.returncode == 0:
                aprint(f"[{done}/{total}] done {tag}")
            else:
                aprint(f"[{done}/{total}] FAILED {tag} (exit {res.returncode})")

        results = run_task_pool(
            task_ids,
            max_workers=global_workers,
            argv_builder=_argv,
            env_builder=_env,
            skip_if=_skip,
            on_done=_on_done,
            verbose=verbose,
        )

    # Promote successful .tmp outputs; collect failures (curated like the
    # parallel tiled path — name failing (t,c,k) + stderr tails, keep .tmp).
    failures: list[tuple[int, str]] = []
    for res in results:
        if res.skipped:
            continue
        job = job_by_id[res.key]
        out = _out_path(job)
        if res.returncode != 0:
            tail = "\n".join(res.output.strip().splitlines()[-20:])
            failures.append((res.key, tail))
            continue
        ok, _empty = _finalize_output(out)
        if not ok:
            failures.append(
                (res.key, "worker exited 0 but wrote no output (.tmp missing)")
            )

    if failures:

        def _tag(tid: int) -> str:
            j = job_by_id[tid]
            return f"t{j.timepoint}/c{j.channel}/k{j.tile_index}"

        ids = ", ".join(_tag(t) for t, _ in failures)
        detail = "\n\n".join(
            f"--- task {_tag(t)} stderr (tail) ---\n{msg}" for t, msg in failures
        )
        raise RuntimeError(
            f"{len(failures)} of {len(task_ids)} fit tasks failed ({ids}). "
            f"Temp outputs kept under {tiles_dir} for inspection.\n{detail}"
        )

    with asection(f"Merging {len(task_ids)} results → partition"):
        return merge_batch_results(
            manifest=manifest,
            output_dir=output_dir,
            channel_colors=channel_colors,
            force=force_merge,
            recipe=recipe if recipe is not None else manifest.merge_recipe,
            recipe_params=recipe_params,
            verbose=verbose,
        )
