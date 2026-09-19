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

import contextlib
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from arbol import aprint, asection

from luxar.gsplats.batch.task_pool import cancel_pool_on_interrupt
from luxar.gsplats.fit_basis import fit_image_min
from luxar.gsplats.fit_tiled_parallel import luxar_argv0
from luxar.gsplats.gsplat_data import scrub_measured_stats, scrub_region_scoped_stats

from .fit_planned import (
    _padded_bounds,
    _planned_merge_stats,
    _score_planned_merge,
    _stamp_planned_normalization,
)
from .spec import FitPlan

# Builds the argv for plan box ``i`` writing to a given output path.
WorkerCmdBuilder = Callable[[int, Path], "list[str]"]

_SKIP_CONTENT_BOX_STAMP_ENV = "LUXAR_INTERNAL_SKIP_CONTENT_BOX_STAMP"


def _worker_env(keep_boxes: bool) -> "dict[str, str]":
    """Build the internal parent-to-child environment for box scoring.

    Disposable boxes skip stamps that the merge scrubs anyway; ``keep_boxes``
    opts retained box artifacts back into scoring.
    """
    env = os.environ.copy()
    if keep_boxes:
        env.pop(_SKIP_CONTENT_BOX_STAMP_ENV, None)
    else:
        env[_SKIP_CONTENT_BOX_STAMP_ENV] = "1"
    return env


def _default_worker_cmd_builder(
    input_path: str | Path,
    plan_json_path: str | Path,
    *,
    preset: Optional[str] = None,
    config: "Optional[str | Path]" = None,
    iters: Optional[int] = None,
    loss: Optional[str] = None,
    lr: Optional[float] = None,
    cull_retention: Optional[float] = None,
    device: Optional[str] = None,
    floor: "Optional[str | float]" = None,
    norm_range: "Optional[tuple[float, float]]" = None,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
) -> WorkerCmdBuilder:
    """Build a ``luxar gsplat fit <in> <out> --tiling content --plan <plan>
    --plan-box i ...`` argv.

    The worker re-reads the existing ``plan_json`` (it does not re-scan/plan),
    rebuilds the fit config from the forwarded flags exactly as the parent did,
    fits the single box, and writes its global-coordinate splats to ``out`` (or a
    sibling ``.empty`` marker for a 0-splat box).

    ``preset`` / ``config`` / ``iters`` / ``loss`` / ``lr`` / ``cull_retention``
    are the run's fit configuration, forwarded so ``-j N`` resolves the same fit
    config as ``-j 1`` (mirroring the uniform-tiled sibling
    :func:`luxar.gsplats.fit_tiled_parallel.build_worker_cmd`). ``truncate:`` is
    settable ONLY through a YAML ``--config`` (no preset sets it and there is no
    ``--truncate`` flag), so an unforwarded config left every box both fitted and
    stamped at the 2.75 default whatever the config said — #1637. ``--seeds`` is
    deliberately NOT
    forwarded: a content box's budget comes from the plan, and ``run_content_fit``
    pops it. Each of these is omitted from the argv when ``None`` (as
    ``device``/``floor`` already are), leaving the worker to resolve its own
    default — ``preset=None`` included, because a preset the user did not ask for
    would layer its own ``n_iters``/``cull_retention`` on top of the config and
    make the worker fit with different parameters than the sequential path.

    ``floor`` is expected to be the parent's already-RESOLVED background level (a
    number, or ``"none"`` when suppression is off) rather than a spec like
    ``auto``/``pNN``: abutting boxes that each re-estimate their own pedestal
    subtract different levels and show brightness steps at box boundaries
    (#1174). A spec is still accepted and forwarded verbatim — the worker then
    resolves it against its whole (t, c) volume, never the box crop.
    """
    argv0 = luxar_argv0()

    # The optional flags, resolved ONCE (nothing here depends on the box): a
    # (flag, value-or-None) table rather than a branch per flag, so forwarding one
    # more of the fit config is a row instead of another rung of complexity.
    # ``None`` means "omit" — the worker then resolves its own default.
    optional: list[tuple[str, Optional[str]]] = [
        ("--preset", preset or None),
        ("--config", str(config) if config else None),
        ("--iters", None if iters is None else str(iters)),
        # `is None`, not truthiness: an empty `--loss ""` is a usage error the
        # parent raises on, and swallowing it here would let `-j N` quietly fit
        # with the default loss where `-j 1` fails.
        ("--loss", None if loss is None else str(loss)),
        ("--lr", None if lr is None else str(lr)),
        ("--cull-retention", None if cull_retention is None else str(cull_retention)),
        ("--device", device or None),
        ("--floor", None if floor is None else str(floor)),
        (
            "--norm-range",
            None
            if norm_range is None
            else f"{float(norm_range[0]):.17g},{float(norm_range[1]):.17g}",
        ),
        ("--channel", None if channel is None else str(channel)),
        ("--timepoint", None if timepoint is None else str(timepoint)),
        ("--array-key", array_key or None),
        ("--axes", axes or None),
    ]
    extra = [
        part for flag, value in optional if value is not None for part in (flag, value)
    ]

    def builder(box_idx: int, out_path: Path) -> list[str]:
        return [
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
            *extra,
        ]

    return builder


def fit_planned_parallel(
    plan: FitPlan,
    *,
    jobs: int,
    tmp_dir: Path,
    worker_cmd_builder: WorkerCmdBuilder,
    volume: Any = None,
    device: Optional[str] = None,
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
    volume : array-like, optional
        The exact array the workers fit: same channel/timepoint selection and
        same resolution level, on ``plan.volume_shape``'s grid. Required to
        stamp merged quality metrics on the merged result; another array with the
        same shape would produce a plausible but invalid score. Direct callers
        may omit it, in which case the omission is announced.
    device : str, optional
        Device used to render the merged reconstruction for scoring. ``None``
        auto-detects, matching :func:`render_to_volume_tensor`.
    keep_boxes : bool, default False
        Keep the per-box temp outputs after a successful merge and let each
        retained worker score and stamp its own box output.
    verbose : bool, default True
        Emit the section header and per-box progress lines. ``False``
        (``fit --quiet``) suppresses progress output; failures still raise.

    Returns
    -------
    GSplatData or GSplatNode
        Flat concatenation, or a tree retaining every box as an additive part.
        Both carry whole-volume merged quality metrics when ``volume`` is
        available; a tree stores them in its root ``meta["fit_stats"]``.

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
    stop = threading.Event()
    worker_env = _worker_env(keep_boxes)

    def _run(i: int) -> tuple[int, int, str]:
        # A worker that dequeued this box after a Ctrl-C must not spawn a new
        # fit subprocess (issue #736): bail before launching anything.
        if stop.is_set():
            return i, -1, "cancelled before launch"
        try:
            cmd = [str(c) for c in worker_cmd_builder(i, box_paths[i])]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=worker_env)
        except Exception as exc:  # bad argv / builder bug → funnel to failure path
            return i, -1, f"failed to build/launch worker: {exc!r}"
        return i, proc.returncode, proc.stderr or proc.stdout or ""

    failures: list[tuple[int, str]] = []
    n = len(budgeted)
    section = (
        asection(f"Parallel planned fitting: {n} boxes, {jobs} concurrent worker(s)")
        if verbose
        else contextlib.nullcontext()
    )
    with section:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            done = 0
            try:
                futures = [ex.submit(_run, i) for i in budgeted]
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
            except KeyboardInterrupt:
                # Cancel queued boxes (and signal in-flight workers) BEFORE the
                # ``with`` block's __exit__ would otherwise drain them, then
                # re-raise so the CLI still exits on the interrupt.
                cancel_pool_on_interrupt(ex, stop)
                raise

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
    # Plan-box index of each region. `budgeted` is already a subset of the boxes
    # and empty ones drop out below, so position in `regions` is NOT the box index
    # the plan's split-plane tree is labelled by.
    region_boxes: list[int] = []
    n_boxes_fit = 0
    missing: list[int] = []
    corrupt: list[tuple[int, str]] = []
    for i in budgeted:
        p = box_paths[i]
        if p.exists():
            try:
                gd = GSplatData.load(p, include_stats=True)
            except Exception as exc:  # present but unreadable/partial store
                corrupt.append((i, repr(exc)))
                continue
            if gd.n_splats > 0:
                scrub_measured_stats(gd)
                scrub_region_scoped_stats(gd)
                regions.append(gd)
                region_boxes.append(i)
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
            regions,
            recipe=recipe,
            recipe_params=recipe_params,
            # The planner's own split planes — see the sequential twin in
            # ``fit_planned``; core-disjoint boxes order exactly.
            bsp_tree=plan.bsp_tree,
            region_labels=region_boxes,
        )
        from luxar.gsplats.tree import GSplatPartition, total_splats

        _stamp_planned_normalization(result.meta, regions)
        fit_stats = _planned_merge_stats(
            regions,
            n_boxes=len(plan.boxes),
            n_boxes_fit=n_boxes_fit,
            overlap=int(plan.overlap),
            volume_shape=tuple(int(s) for s in plan.volume_shape),
            elapsed=elapsed,
            delivered_splats=int(total_splats(result)),
            parallel_jobs=jobs,
            partition=True,
        )
        is_partition = isinstance(result, GSplatPartition)
        _score_planned_merge(
            regions if is_partition else regions[0],
            volume,
            plan_shape=tuple(int(s) for s in plan.volume_shape),
            device=device,
            verbose=verbose,
            image_min=fit_image_min(result.meta),
            stats=fit_stats,
        )
        result.meta["fit_stats"] = fit_stats
    else:
        # `concatenate` keeps the reloaded boxes' shared truncation_radius; a
        # manual re-`GSplatData(...)` of the three arrays reset it to the default
        # (#1637). It REPLACES stats with its own summary, so the planned-fit
        # keys are applied afterwards.
        result = GSplatData.concatenate(regions)
        result.stats.update(
            _planned_merge_stats(
                regions,
                n_boxes=len(plan.boxes),
                n_boxes_fit=n_boxes_fit,
                overlap=int(plan.overlap),
                volume_shape=tuple(int(s) for s in plan.volume_shape),
                elapsed=elapsed,
                delivered_splats=result.n_splats,
                parallel_jobs=jobs,
            )
        )
        _score_planned_merge(
            result,
            volume,
            plan_shape=tuple(int(s) for s in plan.volume_shape),
            device=device,
            verbose=verbose,
            image_min=fit_image_min(result.stats),
        )

    if not keep_boxes:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    elif verbose:
        aprint(f"Kept boxes at {tmp_dir}")
    return result


def max_padded_box_voxels(plan: FitPlan) -> int:
    """Largest padded-box voxel count over budgeted boxes (for worker sizing).

    The per-worker working set is bounded by the biggest box, so ``-j auto``
    uses this for its GPU-memory and host-RAM estimates via ``resolve_jobs``.
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
