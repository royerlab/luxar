# local_runner.py
"""Local (non-Slurm) execution of a batch manifest across multiple GPUs.

The local counterpart of the Slurm fit array + merge: given a planned
:class:`BatchManifest`, fit every ``(t, c, slot)`` task with a multi-GPU
subprocess pool, then run the existing streaming merge to a ``kind=partition``
``.gsplats.zarr``.  The :mod:`task_pool` env hook pins GPU workers via
``CUDA_VISIBLE_DEVICES`` and exposes their host/device quality-memory shares;
auto concurrency is sized from per-card VRAM plus host RAM, CPU count, and a
hard cap. Resumable: a task whose output already exists is skipped.

This engine consumes only the manifest + already-parsed merge options, so it has
no dependency on the CLI layer.
"""

from __future__ import annotations

import math
import os
import shutil
import socket
from collections import Counter
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from arbol import aprint, asection

from luxar.gsplats.batch.fit_command import build_task_fit_argv
from luxar.gsplats.batch.manifest import BatchJob, BatchManifest, save_manifest
from luxar.gsplats.batch.merge_orchestrator import merge_batch_results
from luxar.gsplats.batch.task_pool import TaskResult, run_task_pool
from luxar.gsplats.merged_quality import (
    QUALITY_WORKERS_PER_DEVICE_ENV,
    QUALITY_WORKERS_PER_HOST_ENV,
)
from luxar.gsplats.utils.device import (
    WorkerLimitReason,
    format_worker_limit,
    resolve_gpu_selection,
    resolve_jobs_per_gpu,
)


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


def _denoise_h_for_job(manifest: BatchManifest, job: BatchJob) -> Optional[float]:
    """Return the pinned on-the-fly NLM strength for one local worker."""
    if (
        manifest.denoise
        and manifest.denoise_mode == "on-the-fly"
        and manifest.denoise_h is None
        and manifest.denoise_h_values
    ):
        return manifest.denoise_h_values.get(str(job.channel))
    return None


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


def _worker_env(gpu: int, workers: dict[int, int], host_workers: int) -> dict[str, str]:
    """Pin one worker and expose its fair share of quality-memory budgets."""
    quality_workers = str(max(1, workers.get(gpu, 1)))
    quality_host_workers = str(max(1, host_workers))
    if gpu < 0:
        return {
            "CUDA_VISIBLE_DEVICES": "",
            QUALITY_WORKERS_PER_DEVICE_ENV: quality_workers,
            QUALITY_WORKERS_PER_HOST_ENV: quality_host_workers,
        }
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_token = (
        parent_visible.split(",")[gpu].strip() if parent_visible else str(gpu)
    )
    return {
        "CUDA_VISIBLE_DEVICES": visible_token,
        QUALITY_WORKERS_PER_DEVICE_ENV: quality_workers,
        QUALITY_WORKERS_PER_HOST_ENV: quality_host_workers,
    }


def _gpu_device_name(gpu: int) -> str:
    """Return the parent-visible CUDA device name for startup diagnostics."""
    try:
        import torch

        return str(torch.cuda.get_device_properties(gpu).name)
    except (AttributeError, RuntimeError, AssertionError):
        return "unknown GPU"


def _report_gpu_mappings(
    gpu_indices: list[int], workers: dict[int, int], host_workers: int
) -> None:
    """Print the parent-visible index, child token, and device name per GPU."""
    if not gpu_indices:
        aprint("CUDA hidden from workers; fit device forced to CPU")
        return
    for gpu in gpu_indices:
        visible_token = _worker_env(gpu, workers, host_workers)["CUDA_VISIBLE_DEVICES"]
        aprint(
            f"visible index {gpu} -> CUDA_VISIBLE_DEVICES={visible_token} "
            f"({_gpu_device_name(gpu)})"
        )


def _active_worker_counts(
    task_ids: list[int],
    assignment: dict[int, int],
    workers: dict[int, int],
    skip_if: Callable[[int], bool],
) -> dict[int, int]:
    """Count active workers per device with one resume check per task."""
    active_tasks = Counter(
        assignment.get(task_id, -1) for task_id in task_ids if not skip_if(task_id)
    )
    return {
        gpu: max(1, min(count, active_tasks[gpu])) for gpu, count in workers.items()
    }


def _active_host_workers(active_workers: dict[int, int], n_run: int) -> int:
    """Count workers sharing host RAM, capped by runnable tasks."""
    return max(1, min(sum(active_workers.values()), n_run))


def _auto_limit_suffix(
    jobs_per_gpu: str | int,
    n_run: int,
    workers: dict[int, int],
    limiting_resource: WorkerLimitReason,
) -> str:
    """Describe the effective limiter for an automatic local worker count."""
    if not (isinstance(jobs_per_gpu, str) and jobs_per_gpu.strip().lower() == "auto"):
        return ""
    limit = "task count" if n_run < sum(workers.values()) else limiting_resource
    return f", {format_worker_limit(limit)}"


def _staging_path(out: Path, token: str | int) -> Path:
    """Per-attempt staging directory for a task's fit output.

    Isolated by ``token`` (host + pid of the runner) so two concurrent
    ``run_batch_local`` processes targeting the same ``output_dir`` never share a
    ``.tmp`` store: they can neither delete each other's in-progress staging nor
    interleave chunk/metadata writes. The host prefix keeps the token unique even
    when two invocations on different machines share an NFS ``output_dir`` and
    happen to hold equal pids. Only the atomic claim of the final ``out`` races
    between attempts.
    """
    return Path(str(out) + f".tmp.{token}")


def _finalize_output(
    out: Path, staging: Path, *, overwrite: bool = False
) -> Tuple[bool, bool]:
    """Promote a worker's per-attempt ``staging`` store to its final path.

    Mirrors the Slurm ``run_task`` finalization: a sibling ``{staging}.empty``
    marker (0-splat box) becomes ``{out}.empty``; otherwise the staging store is
    atomically renamed to ``out``. If ``out`` already exists (another attempt won
    the race), the duplicate staging is dropped instead of clobbering it (mirror
    of the Slurm loser path). With ``overwrite`` (a ``--no-resume`` refit) a
    pre-existing ``out`` is a stale prior result, not a winner: it is moved
    aside and replaced — but only here, after the refit fully succeeded, and
    restored if the promotion itself fails — so a failed refit never destroys
    the previous valid tile. Returns ``(ok, empty)``. ``ok`` is False when the
    worker exited 0 but left nothing usable.
    """
    out_empty = Path(str(out) + ".empty")
    staging_empty = Path(str(staging) + ".empty")
    if staging_empty.exists():
        staging_empty.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
        if overwrite:
            # The refit legitimately produced 0 splats; the stale real store
            # from the prior run goes with it.
            shutil.rmtree(out, ignore_errors=True)
            out_empty.touch()
            return True, True
        # Claim the marker FIRST, then recheck. A racing real attempt removes
        # the marker after its rename, so whichever way the two interleave, a
        # real store and the marker never both survive. (Checking before
        # touching leaves a window — the recheck passes, the real attempt
        # promotes and clears, then the touch lands — that would strand both
        # terminal representations behind: if the store were later removed, a
        # lingering marker would make resume/status/merge treat the slot as
        # legitimately empty.)
        out_empty.touch()
        if out.exists():
            # A concurrent attempt already promoted a real store — it wins.
            out_empty.unlink(missing_ok=True)
        return True, True
    if staging.exists():
        moved_aside: Path | None = None
        try:
            if overwrite:
                # Replace the stale prior output only now that the refit
                # succeeded — and move it ASIDE rather than deleting it, so no
                # moment exists where the old tile is gone and the new one is
                # not yet in place. A stale empty marker from the prior run
                # goes with it.
                out_empty.unlink(missing_ok=True)
                if out.exists():
                    moved_aside = Path(str(staging) + ".old")
                    shutil.rmtree(moved_aside, ignore_errors=True)
                    os.replace(out, moved_aside)
            elif out.exists():
                shutil.rmtree(staging, ignore_errors=True)
                return True, False
            os.replace(staging, out)  # atomic within the same filesystem
        except OSError:
            if not overwrite and out.exists():
                # TOCTOU: another attempt claimed `out` between the check above
                # and this rename (os.replace refuses to overwrite a non-empty
                # dir). Drop our duplicate, completed-by-other (Slurm mv -T loser).
                shutil.rmtree(staging, ignore_errors=True)
                return True, False
            if moved_aside is not None and not out.exists():
                # The promotion failed after the prior tile was set aside — put
                # it back, so a failed refit never leaves the slot with nothing.
                try:
                    os.replace(moved_aside, out)
                except OSError:
                    pass  # the aside copy stays on disk for manual recovery
            # Genuine failure (staging vanished — e.g. a concurrent
            # `validate --fix` glob-deleted it — or EACCES/EIO). Keep staging on
            # disk for inspection and report not-ok (mirrors the Slurm
            # "mv failed and output missing" → rc 1 branch).
            return False, False
        if moved_aside is not None:
            shutil.rmtree(moved_aside, ignore_errors=True)
        # A real store now stands at `out` — drop any stale empty marker (e.g.
        # from a lost-race empty attempt) so the two terminal representations
        # never coexist.
        out_empty.unlink(missing_ok=True)
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
        Concurrent fit workers per GPU. ``auto`` applies shared GPU-memory, host
        RAM, CPU-thread, and configurable hard-cap limits.
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

    # Persist the manifest so `batch-fit status` / `validate` (which load_manifest)
    # work against a local run's output, exactly as for a Slurm submit.
    save_manifest(manifest, output_dir)

    gpu_indices = resolve_gpu_selection(gpus)
    task_ids = [j.task_id for j in manifest.jobs]
    worker_plan = resolve_jobs_per_gpu(
        gpu_indices,
        task_voxels=_task_voxels(manifest),
        jobs_per_gpu=jobs_per_gpu,
    )
    workers = worker_plan.workers
    assignment = build_device_assignment(task_ids, workers)
    job_by_id = {j.task_id: j for j in manifest.jobs}

    # Per-invocation staging token: host + pid. Two concurrent run_batch_local()
    # processes never share a token (distinct pids on one host; the host prefix
    # disambiguates equal pids on different machines over a shared NFS
    # output_dir), so their staging dirs never collide. Resume/skip still keys
    # off the final out.
    staging_token = f"{socket.gethostname()}-{os.getpid()}"

    def _out_path(job: BatchJob) -> Path:
        return tiles_dir / job.output_filename

    def _skip(task_id: int) -> bool:
        out = _out_path(job_by_id[task_id])
        return resume and (out.exists() or Path(str(out) + ".empty").exists())

    active_workers = _active_worker_counts(task_ids, assignment, workers, _skip)
    n_run = sum(0 if _skip(task_id) else 1 for task_id in task_ids)
    active_host_workers = _active_host_workers(active_workers, n_run)
    global_workers = max(1, min(sum(workers.values()), n_run))

    def _argv(task_id: int) -> list[str]:
        job = job_by_id[task_id]
        out = _out_path(job)
        staging = _staging_path(out, staging_token)
        # Clear a stale staging from a crashed run of THIS invocation so fit
        # writes cleanly. Never touch another process's staging (different token).
        # With resume disabled a stale final output may also exist; it is kept
        # until finalize replaces it (overwrite=True) AFTER the refit succeeded —
        # deleting it here would destroy the previous valid tile minutes before
        # its replacement exists, and a failed refit would then leave nothing.
        shutil.rmtree(staging, ignore_errors=True)
        Path(str(staging) + ".empty").unlink(missing_ok=True)
        argv = build_task_fit_argv(
            manifest, job, staging, denoise_h=_denoise_h_for_job(manifest, job)
        )
        if assignment.get(task_id, -1) < 0:
            argv += ["--device", "cpu"]
        return argv

    def _env(task_id: int) -> dict[str, str]:
        gpu = assignment.get(task_id, -1)
        return _worker_env(gpu, active_workers, active_host_workers)

    active_gpu_indices = [gpu for gpu in workers if gpu >= 0]
    dev_desc = "CPU" if not active_gpu_indices else f"GPU(s) {active_gpu_indices}"
    auto_limit = _auto_limit_suffix(jobs_per_gpu, n_run, workers, worker_plan.limit)
    with asection(
        f"Local batch fit: {n_run}/{len(task_ids)} tasks on {dev_desc}, "
        f"{global_workers} concurrent worker(s){auto_limit}"
    ):
        _report_gpu_mappings(active_gpu_indices, active_workers, active_host_workers)
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

    # Promote each successful per-attempt staging store; collect failures
    # (curated like the parallel tiled path — name failing (t,c,k) + stderr
    # tails, keep the staging store on disk for inspection).
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
        ok, _empty = _finalize_output(
            out, _staging_path(out, staging_token), overwrite=not resume
        )
        if not ok:
            failures.append(
                (
                    res.key,
                    "worker exited 0 but its output could not be promoted "
                    "(staging missing, or the rename to the final path failed)",
                )
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
        # resume=False means the user asked to refit every tile, so the merge
        # must be rebuilt over the fresh tiles — otherwise it short-circuits on a
        # pre-existing merged/final.gsplats.zarr and serves the STALE artifact.
        return merge_batch_results(
            manifest=manifest,
            output_dir=output_dir,
            channel_colors=channel_colors,
            force=force_merge or not resume,
            recipe=recipe if recipe is not None else manifest.merge_recipe,
            recipe_params=recipe_params,
            verbose=verbose,
        )
