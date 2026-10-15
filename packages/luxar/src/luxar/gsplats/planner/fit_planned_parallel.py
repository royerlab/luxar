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
from luxar.gsplats.fit_tiled_gsplats import _report_merged_min_extent
from luxar.gsplats.fit_tiled_parallel import luxar_argv0
from luxar.gsplats.gsplat_data import scrub_measured_stats, scrub_region_scoped_stats

from .box_checkpoint import BoxCheckpoint, plan_fingerprint
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
    min_extent: Optional[float] = None,
    reseed_sigma: Optional[float] = None,
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
    default — ``preset=None`` included: the worker's ``load_fit_config`` then
    resolves ``DEFAULT_FIT_PRESET`` exactly as the sequential path's does, so
    forwarding a call-site default instead could only let the two drift apart.

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
        ("--min-extent", None if min_extent is None else str(min_extent)),
        ("--reseed-sigma", None if reseed_sigma is None else str(reseed_sigma)),
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


def _boxes_to_fit(
    checkpoint: BoxCheckpoint, budgeted: "list[int]", verbose: bool
) -> "tuple[list[int], dict[int, Any]]":
    """Budgeted boxes still to fit, and the finished boxes already loaded.

    A finished box is loaded here (once -- the merge reuses it) so a box marked
    done whose store no longer reads is discarded and refitted in this run
    instead of failing the merge on every rerun. Stale leftovers of the boxes
    to fit are discarded before any worker runs.
    """
    preloaded: dict[int, Any] = {}
    for i in budgeted:
        if checkpoint.is_done(i):
            ok, region = checkpoint.load_or_discard(i)
            if ok:
                preloaded[i] = region
    to_fit = [i for i in budgeted if i not in preloaded]
    for i in to_fit:
        checkpoint.discard(i)
    if preloaded and verbose:
        aprint(f"Resuming: {len(preloaded)} of {len(budgeted)} boxes already done")
    return to_fit, preloaded


class _BoxWorkers:
    """Box worker subprocesses that an interrupt TERMINATES, not just abandons.

    ``cancel_pool_on_interrupt`` only drops queued boxes; a worker already
    running would keep fitting (and writing its ``box_{i}`` store) after the
    parent unwinds and releases the checkpoint lock, racing a rerun's worker
    for the same path. :meth:`stop` sets the stop flag (no new launches, no
    more done-markers) and terminates every live worker; the executor join that
    follows then returns promptly, still inside the lock.
    """

    _TERM_TIMEOUT = 5.0

    def __init__(self, checkpoint: BoxCheckpoint, env: "dict[str, str]") -> None:
        self.checkpoint = checkpoint
        self.env = env
        self.stopped = threading.Event()
        self._live: dict[int, subprocess.Popen[str]] = {}
        self._mutex = threading.Lock()

    def run(self, i: int, cmd: "list[str]") -> tuple[int, str]:
        """Run box ``i``'s worker; mark it done only on a clean, uninterrupted exit."""
        with self._mutex:
            if self.stopped.is_set():
                return -1, "cancelled before launch"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
            )
            self._live[i] = proc
        try:
            out, err = proc.communicate()
        finally:
            with self._mutex:
                self._live.pop(i, None)
        if self.stopped.is_set():
            return -1, "interrupted"
        self._mark_if_written(i, proc.returncode)
        return proc.returncode, err or out or ""

    def _mark_if_written(self, i: int, returncode: int) -> None:
        # A missing output still fails the merge and is never marked.
        ck = self.checkpoint
        written = ck.box_path(i).exists() or ck.empty_marker(i).exists()
        if returncode == 0 and written:
            ck.mark_done(i, -1)

    def stop(self) -> None:
        """Stop launching, then terminate (and finally kill) live workers.

        The kill runs in a ``finally``: a second SIGTERM/Ctrl-C arriving while
        this waits for the terminated workers still kills every survivor
        before the interrupt propagates.
        """
        with self._mutex:
            self.stopped.set()
            procs = list(self._live.values())
        try:
            for proc in procs:
                with contextlib.suppress(OSError):
                    proc.terminate()
            deadline = time.monotonic() + self._TERM_TIMEOUT
            for proc in procs:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        finally:
            for proc in procs:
                if proc.poll() is None:
                    with contextlib.suppress(OSError):
                        proc.kill()


def _reload_boxes(
    budgeted: "list[int]",
    box_paths: "dict[int, Path]",
    preloaded: "dict[int, Any]",
) -> "tuple[list[Any], list[int], int, list[int], list[tuple[int, str]]]":
    """Load every budgeted box's output for the merge.

    A successful worker either wrote its .gsplats.zarr or -- for a 0-splat box
    -- a sibling ".empty" marker (the gsplats writer rejects empty stores).
    Neither present after a clean exit is a silent spatial hole; a
    present-but-unreadable store (e.g. OOM mid-save) is corrupt -- both are
    returned for the caller to fail on. Returns ``(regions, region_boxes,
    n_boxes_fit, missing, corrupt)``; ``region_boxes`` holds each region's
    plan-box index (empty boxes drop out, so position is NOT the box index the
    plan's split-plane tree is labelled by).
    """
    from luxar.gsplats.gsplat_data import GSplatData

    regions: list[Any] = []
    region_boxes: list[int] = []
    n_boxes_fit = 0
    missing: list[int] = []
    corrupt: list[tuple[int, str]] = []
    for i in budgeted:
        p = box_paths[i]
        if p.exists():
            try:
                gd = (
                    preloaded[i]
                    if preloaded.get(i) is not None
                    else GSplatData.load(p, include_stats=True)
                )
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
    return regions, region_boxes, n_boxes_fit, missing, corrupt


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
    fingerprint: Optional[str] = None,
    remove_tmp_on_success: bool = True,
) -> "Any":
    """Fit every budgeted box via concurrent worker subprocesses, then merge.

    Parameters
    ----------
    plan : FitPlan
        The plan whose boxes to fit. Only boxes with ``budget > 0`` are spawned.
    jobs : int
        Maximum number of concurrent worker processes.
    tmp_dir : Path
        Directory for per-box outputs, doubling as the resume checkpoint
        (:class:`~luxar.gsplats.planner.box_checkpoint.BoxCheckpoint`): it is
        NOT cleared first. A box whose ``box_{i}.done.json`` marker matches
        ``fingerprint`` (and whose output is present) is reused without
        spawning a worker; any other leftover for that box is discarded and
        refitted. Removed on success unless ``keep_boxes`` (or
        ``remove_tmp_on_success=False``); **retained** on failure, so a rerun
        resumes.
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
    fingerprint : str, optional
        Identity of this run's boxes (see
        :func:`~luxar.gsplats.planner.box_checkpoint.plan_fingerprint`).
        ``None`` fingerprints the plan alone.
    remove_tmp_on_success : bool, default True
        ``False`` leaves ``tmp_dir`` in place after the merge so the caller
        can remove it only once the merged output is safely saved.

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
    # With a fingerprint, resume rather than clean-slate: a box finished under
    # THIS fingerprint is reused; a stale leftover (other plan/config, or a box
    # killed mid-save and so never marked done) is discarded before its worker
    # runs. Without one the caller has no way to say what the boxes represent,
    # so the directory is cleared as before -- a plan-only digest would happily
    # reuse boxes fitted from another volume or config.
    if fingerprint is None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    checkpoint = BoxCheckpoint(
        tmp_dir, fingerprint if fingerprint is not None else plan_fingerprint(plan)
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    budgeted = [i for i, b in enumerate(plan.boxes) if b.budget > 0]
    box_paths = {i: checkpoint.box_path(i) for i in budgeted}
    to_fit, preloaded = _boxes_to_fit(checkpoint, budgeted, verbose)
    workers = _BoxWorkers(checkpoint, _worker_env(keep_boxes))

    def _run(i: int) -> tuple[int, int, str]:
        # A worker that dequeued this box after a Ctrl-C must not spawn a new
        # fit subprocess (issue #736): `workers.run` bails before launching.
        try:
            cmd = [str(c) for c in worker_cmd_builder(i, box_paths[i])]
            rc, stream = workers.run(i, cmd)
        except Exception as exc:  # bad argv / builder bug → funnel to failure path
            return i, -1, f"failed to build/launch worker: {exc!r}"
        return i, rc, stream

    failures: list[tuple[int, str]] = []
    n = len(budgeted)
    section = (
        asection(f"Parallel planned fitting: {n} boxes, {jobs} concurrent worker(s)")
        if verbose
        else contextlib.nullcontext()
    )
    with section:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            done = len(budgeted) - len(to_fit)
            try:
                futures = [ex.submit(_run, i) for i in to_fit]
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
                # Cancel queued boxes BEFORE the ``with`` block's __exit__ would
                # otherwise drain them, terminate the running workers (so none
                # keeps writing after we unwind and drop the checkpoint lock),
                # then re-raise so the CLI still exits on the interrupt. The
                # executor join in __exit__ then returns promptly.
                cancel_pool_on_interrupt(ex, workers.stopped)
                workers.stop()
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

    regions, region_boxes, n_boxes_fit, missing, corrupt = _reload_boxes(
        budgeted, box_paths, preloaded
    )

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
        _report_merged_min_extent(fit_stats, verbose=verbose)
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
        _report_merged_min_extent(result.stats, verbose=verbose)

    if keep_boxes and verbose:
        aprint(f"Kept boxes at {tmp_dir}")
    elif not keep_boxes and remove_tmp_on_success:
        shutil.rmtree(tmp_dir, ignore_errors=True)
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
