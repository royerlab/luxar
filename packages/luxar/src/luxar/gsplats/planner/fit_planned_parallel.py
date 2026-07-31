"""Parallel planned fitting: one subprocess per plan box, then merge.

The local, single-GPU counterpart of the sequential :func:`fit_planned`. Each
content-balanced box of a :class:`FitPlan` is fit by a separate
``luxar gsplat fit ... --tiling content --plan ... --plan-box i`` worker process (own CUDA context, shared
GPU memory pool); up to ``jobs`` run concurrently. After all succeed, the
per-box ``.gsplats.zarr`` outputs are reloaded and **concatenated** — boxes are
spatially disjoint by construction (the BSP partition tiles the volume and only
core-centred splats are kept), so the merge is a plain concatenation with no
blending, identical to what :func:`fit_planned` builds.

This is the planner-path sibling of :mod:`luxar.gsplats.fit_tiled_parallel`
(uniform tiles); it reuses that module's :func:`resolve_jobs` and
:func:`luxar_argv0`, and mirrors its battle-tested failure handling
(per-worker stderr tails, ``tmp_dir`` retained on failure, missing/corrupt/empty
detection). It does not touch the uniform-tiled path.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fit_tiled_parallel import luxar_argv0

from .fit_planned import _padded_bounds
from .spec import FitPlan

# Builds the argv for plan box ``i`` writing to a given output path.
WorkerCmdBuilder = Callable[[int, Path], "list[str]"]


def _default_worker_cmd_builder(
    input_path: str | Path,
    plan_json_path: str | Path,
    *,
    preset: str = "standard",
    device: Optional[str] = None,
    floor: Optional[str] = None,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
) -> WorkerCmdBuilder:
    """Build a ``luxar gsplat fit <in> <out> --tiling content --plan <plan>
    --plan-box i ...`` argv.

    The worker re-reads the existing ``plan_json`` (it does not re-scan/plan),
    rebuilds the fit config from ``--preset`` exactly as the parent did, fits the
    single box, and writes its global-coordinate splats to ``out`` (or a sibling
    ``.empty`` marker for a 0-splat box).
    """
    argv0 = luxar_argv0()

    def builder(box_idx: int, out_path: Path) -> list[str]:
        cmd = [
            *argv0,
            "gsplat",
            "fit",
            str(input_path),
            str(out_path),
            "--tiling",
            "content",
            "--plan",
            str(plan_json_path),
            "--plan-box",
            str(box_idx),
            "--preset",
            preset,
        ]
        if device:
            cmd += ["--device", device]
        if floor is not None:
            cmd += ["--floor", str(floor)]
        if channel is not None:
            cmd += ["--channel", str(channel)]
        if timepoint is not None:
            cmd += ["--timepoint", str(timepoint)]
        if array_key:
            cmd += ["--array-key", array_key]
        if axes:
            cmd += ["--axes", axes]
        return cmd

    return builder


def fit_planned_parallel(
    plan: FitPlan,
    *,
    jobs: int,
    tmp_dir: Path,
    worker_cmd_builder: WorkerCmdBuilder,
    keep_boxes: bool = False,
    partition: bool = False,
    recipe: Optional[str] = None,
    recipe_params: "Optional[Any]" = None,
    verbose: bool = True,
) -> "Any":
    """Fit every budgeted box via concurrent worker subprocesses, then merge.

    Parameters
    ----------
    plan : FitPlan
        The plan whose boxes to fit. Only boxes with ``budget > 0`` are spawned.
    jobs : int
        Maximum number of concurrent worker processes.
    tmp_dir : Path
        Directory for per-box outputs. Cleared first; removed on success unless
        ``keep_boxes``; **retained** on failure for inspection.
    worker_cmd_builder : callable
        ``(box_idx, out_path) -> argv`` returning the command to fit one box.
        The injection seam for testing (see :func:`_default_worker_cmd_builder`).
    keep_boxes : bool, default False
        Keep the per-box temp outputs after a successful merge.

    Returns
    -------
    GSplatData
        Merged result (concatenation of every box's core-kept splats).

    Raises
    ------
    RuntimeError
        If any worker exits non-zero, or exits cleanly but writes neither an
        output nor an ``.empty`` marker, or writes an unreadable store. The
        message names the offending boxes; ``tmp_dir`` is retained.
    ValueError
        If every box produced 0 splats.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    tmp_dir = Path(tmp_dir)
    # Start clean: a retained dir from a prior (failed / keep_boxes) run could
    # leave a stale box_{i} that this run would reload instead of its own output.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    budgeted = [i for i, b in enumerate(plan.boxes) if b.budget > 0]
    box_paths = {i: tmp_dir / f"box_{i}.gsplats.zarr" for i in budgeted}

    def _run(i: int) -> tuple[int, int, str]:
        try:
            cmd = [str(c) for c in worker_cmd_builder(i, box_paths[i])]
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as exc:  # bad argv / builder bug → funnel to failure path
            return i, -1, f"failed to build/launch worker: {exc!r}"
        return i, proc.returncode, proc.stderr or proc.stdout or ""

    failures: list[tuple[int, str]] = []
    n = len(budgeted)
    with asection(f"Parallel planned fitting: {n} boxes, {jobs} concurrent worker(s)"):
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futures = [ex.submit(_run, i) for i in budgeted]
            done = 0
            for fut in as_completed(futures):
                i, rc, stream = fut.result()
                done += 1
                if rc != 0:
                    tail = "\n".join(stream.strip().splitlines()[-20:])
                    failures.append((i, tail))
                    if verbose:
                        aprint(f"Box {i} FAILED (exit {rc}) [{done}/{n}]")
                elif verbose:
                    aprint(f"Box {i} done [{done}/{n}]")

    if failures:
        idxs = ", ".join(str(i) for i, _ in failures)
        detail = "\n\n".join(
            f"--- box {i} stderr (tail) ---\n{msg}" for i, msg in failures
        )
        raise RuntimeError(
            f"{len(failures)} of {n} box fits failed (boxes: {idxs}). "
            f"Temp outputs kept at {tmp_dir} for inspection.\n{detail}"
        )

    # Reload each box. A successful worker either wrote its .gsplats.zarr or — for
    # a 0-splat box — a sibling ".empty" marker (the gsplats writer rejects empty
    # stores). Neither present after a clean exit is a silent spatial hole; a
    # present-but-unreadable store (e.g. OOM mid-save) is corrupt — both fail.
    regions: list[GSplatData] = []  # one core-kept GSplatData per non-empty box
    n_boxes_fit = 0
    missing: list[int] = []
    corrupt: list[tuple[int, str]] = []
    for i in budgeted:
        p = box_paths[i]
        if p.exists():
            try:
                gd = GSplatData.load(p)
            except Exception as exc:  # present but unreadable/partial store
                corrupt.append((i, repr(exc)))
                continue
            if gd.n_splats > 0:
                regions.append(gd)
            n_boxes_fit += 1
        elif Path(str(p) + ".empty").exists():
            n_boxes_fit += 1  # ran, legitimately produced 0 splats
        else:
            missing.append(i)

    if missing or corrupt:
        parts = []
        if missing:
            parts.append(f"wrote no output (boxes: {', '.join(map(str, missing))})")
        if corrupt:
            ids = ", ".join(f"{i}: {e}" for i, e in corrupt)
            parts.append(f"wrote an unreadable store (boxes: {ids})")
        raise RuntimeError(
            f"{len(missing) + len(corrupt)} of {n} boxes exited cleanly but "
            f"{'; '.join(parts)}. Temp outputs kept at {tmp_dir} for inspection."
        )

    if not regions:
        raise ValueError("fit_planned_parallel produced no splats (all boxes empty?)")

    elapsed = time.perf_counter() - t0
    if partition:
        # One part per box — boxes are core-disjoint, an exact spatial partition.
        # ``recipe`` gives each part its own LOD ladder/group at assembly time
        # (the per-box workers only fit bare leaves).
        result: Any = GSplatData.partition_from_regions(
            regions, recipe=recipe, recipe_params=recipe_params
        )
    else:
        result = GSplatData(
            centers=np.concatenate([r.centers for r in regions]).astype(np.float32),
            amplitudes=np.concatenate([r.amplitudes for r in regions]).astype(
                np.float32
            ),
            cholesky_factors=np.concatenate(
                [r.cholesky_factors for r in regions]
            ).astype(np.float32),
            stats={
                "planned_fit": True,
                "n_boxes": len(plan.boxes),
                "n_boxes_fit": n_boxes_fit,
                "overlap": int(plan.overlap),
                "volume_shape": list(plan.volume_shape),
                "parallel_jobs": int(jobs),
                "elapsed_seconds": float(elapsed),
            },
        )

    if not keep_boxes:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        aprint(f"Kept boxes at {tmp_dir}")
    return result


def max_padded_box_voxels(plan: FitPlan) -> int:
    """Largest padded-box voxel count over budgeted boxes (for VRAM sizing).

    The per-worker working set is bounded by the biggest box, so ``-j auto``
    divides free VRAM by ~2x this (via ``resolve_jobs``) to size concurrency.
    """
    vs = [int(s) for s in plan.volume_shape]
    shape = (vs[0], vs[1], vs[2])
    best = 1
    for b in plan.boxes:
        if b.budget <= 0:
            continue
        pz0, pz1, py0, py1, px0, px1 = _padded_bounds(b, int(plan.overlap), shape)
        vox = (pz1 - pz0) * (py1 - py0) * (px1 - px0)
        best = max(best, int(vox))
    return best


__all__ = [
    "fit_planned_parallel",
    "max_padded_box_voxels",
    "_default_worker_cmd_builder",
]
