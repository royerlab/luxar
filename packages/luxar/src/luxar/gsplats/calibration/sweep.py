"""The per-K loop of :func:`~.driver.calibrate`.

One grid point = one fit of the donut-filled volume, rendered and scored
against the original (:class:`PointContext`). :func:`run_sweep` walks a grid in
order with early stopping, an ETA line and a per-point checkpoint hook;
:func:`extend_below` adds the cheap points under the grid once when K* lands on
its smallest point. Everything here works on plain per-point RECORDS (dicts),
which :func:`curves` turns into the per-K lists ``CalibrationResult`` stores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .curve_analysis import HeldOutPeak, find_k_star
from .sweep_control import (
    SweepControl,
    estimate_remaining_seconds,
    extension_below,
    format_duration,
    plateau_reached,
)

ProgressCallback = Callable[[int, int, str], None]
"""``(index, total, message)`` callback emitted before/after each fit."""

PointHook = Callable[[List[Dict[str, Any]]], None]
"""Called with every record so far after each grid point (checkpointing)."""

#: The record key each ``k_star_metric`` reads.
METRIC_RECORD_KEY = {
    "psnr_minmax": "held_psnr",
    "psnr_foreground": "held_psnr_fg",
    "psnr_fg_weighted": "held_psnr_fg_weighted",
    "gain": "held_gain",
}

#: Per-point keys turned into per-K lists by :func:`curves`.
_CURVE_KEYS = (
    "k",
    "eff",
    "n_fitted",
    "n_stored",
    "fit_s",
    "held_psnr",
    "held_psnr_fg",
    "held_psnr_fg_weighted",
    "held_gain",
    "held_psnr_robust",
    "held_psnr_fg_robust",
    "train_psnr",
    "held_mse",
    "full_psnr",
    "full_ssim",
)


@dataclass
class PointContext:
    """Everything one grid point needs: the fit target, device copies, scoring.

    ``cull_retention`` is applied HERE rather than inside the fit, so the point
    can report three counts from one fit: ``n_fitted`` (before any closing
    reduction), ``eff`` (after the cumulative cull — the model that is scored,
    as before) and ``n_stored`` (after the default ``fit``'s closing reductions,
    speck pruning at ``stored_min_extent`` THEN the cull — what
    ``fit --seeds K`` would store for this fit).
    """

    V_filled: np.ndarray
    fit_kwargs: Dict[str, Any]
    render_device: str
    ref_dev: Any
    mask_dev: Any
    held_fg_sel: Any
    weighting: Tuple[Any, Any, int, int, float]
    data_range: float
    robust_range: float
    baseline_mse: float
    keep_fits: Optional[Path]
    cull_retention: Optional[float]
    stored_min_extent: Optional[float]
    stored_min_extent_max_mass_share: float

    def _cull(self, splats: Any) -> Any:
        from luxar.gsplats.fit_gsplats import _post_fit_cull

        r = self.cull_retention
        if r is None or not 0 < r < 1.0 or splats.n_splats == 0:
            return splats
        return _post_fit_cull(splats, r, verbose=False)[0]

    def _stored_count(self, splats: Any) -> int:
        """Splats the default ``fit`` would keep: speck prune, then cull."""
        from luxar.gsplats.fit_gsplats import prune_min_extent
        from luxar.gsplats.fitting.splat_hygiene import fitted_voxel_unit

        pruned = splats
        if self.stored_min_extent and splats.n_splats > 0:
            unit = fitted_voxel_unit(
                splats.ndim,
                voxel_size=self.fit_kwargs.get("voxel_size"),
                output_space=self.fit_kwargs.get("output_space", "real"),
                downscale=self.fit_kwargs.get("downscale"),
            )
            pruned, _ = prune_min_extent(
                splats,
                float(self.stored_min_extent),
                unit,
                verbose=False,
                max_mass_share=self.stored_min_extent_max_mass_share,
            )
        return int(self._cull(pruned).n_splats)

    def run(self, K: int) -> Dict[str, Any]:
        """Fit at ``K``, render, score; return the point's record."""
        import torch

        from luxar.gsplats.fit_gsplats import fit_gaussian_splats
        from luxar.gsplats.metrics import compute_ssim
        from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

        from .driver import _per_k_metrics

        t0 = time.perf_counter()
        fitted = fit_gaussian_splats(
            self.V_filled, seeds=int(K), **{**self.fit_kwargs, "cull_retention": None}
        )
        splats = self._cull(fitted)
        elapsed = time.perf_counter() - t0
        with torch.no_grad():
            rendered = render_to_volume_tensor(
                splats, shape=self.V_filled.shape, device=self.render_device
            )
            record: Dict[str, Any] = _per_k_metrics(
                rendered,
                self.ref_dev,
                self.mask_dev,
                self.held_fg_sel,
                self.weighting,
                data_range=self.data_range,
                baseline_mse=self.baseline_mse,
                compute_ssim=compute_ssim,
                robust_range=self.robust_range,
            )
            del rendered  # now the squared-error volume (reused in place)
        record.update(
            k=int(K),
            eff=int(splats.n_splats),
            n_fitted=int(fitted.n_splats),
            n_stored=self._stored_count(fitted),
            fit_s=elapsed,
            path=None,
        )
        if self.keep_fits is not None:
            out_path = self.keep_fits / f"k{int(K):08d}.gsplats.zarr"
            # The fitter's own metrics ride along via splats.stats;
            # calibration-specific fields would be dropped by GSplatData.save()'s
            # fitting-info whitelist anyway, so they live only in cal.json
            # (which references this file via splat_paths).
            splats.save(out_path, include_fitting_info=True)
            record["path"] = str(out_path)
        del splats, fitted
        if str(self.render_device).startswith("cuda"):
            torch.cuda.empty_cache()
        return record


def _point_line(record: Dict[str, Any], elapsed: float, eta: float, bound: bool) -> str:
    """One progress line per grid point: K, counts, scores, timing, ETA.

    PSNRs are on the ROBUST peak (p99.99 - floor), the convention ``fit`` and
    ``compare`` headline; ``cal.json`` keeps the full-range curves too.
    """
    # An estimate either way (it under-reads while fits still cost a flat
    # overhead); with early stopping on, the sweep may also end sooner.
    del bound
    eta_txt = "" if eta == 0.0 else f" eta~{format_duration(eta)}"
    return (
        f"K={record['k']} kept={record['eff']} stored={record['n_stored']} "
        f"held={record['held_psnr_robust']:.2f}dB "
        f"fg={record['held_psnr_fg_robust']:.2f}dB (robust peak; full range "
        f"{record['held_psnr']:.2f}dB) fit={format_duration(record['fit_s'])} "
        f"elapsed={format_duration(elapsed)}{eta_txt}"
    )


def _progress_line(
    records: Sequence[Dict[str, Any]],
    grid: Sequence[int],
    i: int,
    t0: float,
    early: bool,
    control: SweepControl,
    stopped: bool,
) -> str:
    """The after-metrics line for point ``i``, with ETA and any stop notice."""
    remaining = [] if stopped else list(grid[i + 1 :])
    eta = estimate_remaining_seconds(
        [r["k"] for r in records], [r["fit_s"] for r in records], remaining
    )
    line = _point_line(records[-1], time.perf_counter() - t0, eta, early)
    if stopped:
        line += (
            f" | stopping (no gain > {control.early_stop_min_gain_db:g} dB for "
            f"{control.early_stop_patience} points, or a fall of "
            f">= {control.early_stop_decline_db:g} dB past the best); skipping "
            f"{control.k_values_skipped}"
        )
    return line


def should_stop(
    records: Sequence[Dict[str, Any]], key: str, control: SweepControl
) -> bool:
    """Early-stop decision on the measured part of the grid (ascending K).

    :func:`plateau_reached` is necessary but not sufficient: the truncated
    curve must also be one :func:`find_k_star` can decide on its own. So the
    sweep keeps going while the measured curve spans less than its knee margin
    (0.3 dB: a slow creep of < 0.1 dB per doubling would otherwise stop after
    three points and put K* at the bottom while the full grid's knee lay four
    doublings higher), and while it still reads ``signal_limited`` or puts K* on
    its top point (a steep rise followed by a creep must not be cut while it is
    still climbing). A ``peak`` / ``low_k_optimal`` reading stops; a ``plateau``
    stops only while flat (its latest point within the 0.1 dB bar of the best).
    """
    values = [float(r[key]) for r in records]
    if not plateau_reached(
        values,
        patience=control.early_stop_patience,
        min_gain_db=control.early_stop_min_gain_db,
        decline_db=control.early_stop_decline_db,
    ):
        return False
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < 2:
        return False
    ks = [int(r["k"]) for r in records]
    peak = find_k_star(ks, values)
    if max(finite) - min(finite) < peak.knee_margin_db:
        return False
    if peak.type == "signal_limited" or peak.k_star == ks[-1]:
        return False
    if peak.type in ("peak", "low_k_optimal"):
        return True
    # A plateau may stop only while it is FLAT: the latest point within the
    # improvement bar of the best. A plateau whose latest point already fell
    # below that is a peak the truncation has not resolved yet (find_k_star
    # needs the mean of the points after the max >= 0.1 dB lower, and two
    # jittery points may not get there), and its 0.3 dB knee can sit far below
    # the real optimum: on A549 (MPS) 783 = 19.27, 1566 = 19.26, 3130 = 19.14
    # read as a plateau with K* = 392, where the full grid has a clear peak
    # near 1566. One more point resolves it.
    return values[-1] >= max(finite) - control.early_stop_min_gain_db


def run_sweep(
    ctx: PointContext,
    grid: Sequence[int],
    control: SweepControl,
    k_star_metric: str,
    progress_callback: Optional[ProgressCallback],
    *,
    records: Optional[List[Dict[str, Any]]] = None,
    on_point: Optional[PointHook] = None,
    allow_early_stop: bool = True,
) -> List[Dict[str, Any]]:
    """Fit every K of ``grid`` in order, stopping early on a plateau.

    Appends to ``records`` (in place, so a caller interrupted mid-sweep keeps
    every finished point) and calls ``on_point(records)`` after each one.
    Early stopping watches the curve K* is chosen on (``k_star_metric``).
    """
    key = METRIC_RECORD_KEY[k_star_metric]
    early = control.early_stop and allow_early_stop
    records = [] if records is None else records
    n_before = len(records)
    t0 = time.perf_counter()
    n = len(grid)
    for i, K in enumerate(grid):
        if progress_callback is not None:
            progress_callback(i, n, f"fit K={K}")
        records.append(ctx.run(int(K)))
        mine = records[n_before:]
        stopped = early and i + 1 < n and should_stop(mine, key, control)
        if stopped:
            control.stopped_early = True
            control.k_values_skipped = [int(k) for k in grid[i + 1 :]]
            control.minmax_curve_truncated = k_star_metric != "psnr_minmax"
        if progress_callback is not None:
            progress_callback(
                i, n, _progress_line(mine, grid, i, t0, early, control, stopped)
            )
        if on_point is not None:
            on_point(records)
        if stopped:
            break
    return records


def peaks(
    curve_lists: Dict[str, List[Any]], k_star_metric: str
) -> Tuple[HeldOutPeak, Optional[HeldOutPeak]]:
    """The min--max K* (always) and the selected-metric K* (when not min--max).

    A single measured point (an interrupted or checkpointed sweep) is reported
    as itself, with a warning, since no curve exists yet.
    """
    ks = [int(k) for k in curve_lists["k"]]
    if len(ks) == 1:
        single = HeldOutPeak(
            k_star=ks[0],
            type="plateau",
            confidence_db=0.0,
            k_knee=ks[0],
            knee_idx=0,
            warnings=["only one grid point measured: K* is not a measurement"],
        )
        return single, (None if k_star_metric == "psnr_minmax" else single)
    peak = find_k_star(ks, curve_lists["held_psnr"])
    if k_star_metric == "psnr_minmax":
        return peak, None
    curve = curve_lists[METRIC_RECORD_KEY[k_star_metric]]
    if not bool(np.isfinite(np.asarray(curve, dtype=float)).any()):
        return peak, None
    return peak, find_k_star(ks, curve)


def curves(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Per-key lists over the records, in ascending K."""
    ordered = sorted(records, key=lambda r: int(r["k"]))
    return {k: [r.get(k, float("nan")) for r in ordered] for k in _CURVE_KEYS}


def _best_at_bottom(values: Sequence[float]) -> bool:
    """Whether the curve's best value is its smallest-K point.

    Only then can the optimum lie below the grid. A plateau whose 0.3 dB knee
    happens to be the first point while the curve still creeps up later (the
    best value is elsewhere) gains nothing from cheaper points.
    """
    arr = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(arr))
    if finite.size == 0:
        return False
    return int(finite[np.argmax(arr[finite])]) == int(finite[0])


def extend_below(
    ctx: PointContext,
    records: List[Dict[str, Any]],
    control: SweepControl,
    k_star_metric: str,
    progress_callback: Optional[ProgressCallback],
    on_point: Optional[PointHook] = None,
) -> List[Dict[str, Any]]:
    """Fit a few cheaper points under the grid once, when K* sits on its bottom.

    K* on the smallest grid point is a bound, not a measurement; the points
    below cost the least of the whole sweep, so measure them instead of only
    advising a re-run — when the curve's best value is that bottom point (see
    :func:`_best_at_bottom`). Runs once: if K* is still on the (new) bottom
    edge, ``find_k_star``'s grid warning says so.
    """
    if not control.extend_below or len(records) < 2:
        return records
    curve_lists = curves(records)
    peak, selected = peaks(curve_lists, k_star_metric)
    k_star = (selected or peak).k_star
    ks = sorted(int(r["k"]) for r in records)
    if k_star != ks[0] or not _best_at_bottom(
        curve_lists[METRIC_RECORD_KEY[k_star_metric]]
    ):
        return records
    below = extension_below(ks)
    if not below:
        return records
    control.k_values_extended_below = below
    if progress_callback is not None:
        progress_callback(
            0,
            len(below),
            f"K*={k_star} is the smallest grid point; extending to {below}",
        )
    return run_sweep(
        ctx,
        below,
        control,
        k_star_metric,
        progress_callback,
        records=records,
        on_point=on_point,
        allow_early_stop=False,
    )
