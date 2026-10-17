"""Sweep control for :func:`calibrate`: early stopping, ETA, grid extension.

Three small, pure decisions the K sweep makes while it runs:

* :func:`plateau_reached` — stop once the K*-metric has failed to improve
  meaningfully for ``patience`` consecutive grid points, or has clearly fallen
  past its peak (the rest of a geometric grid is its most expensive part, and a
  flat or falling held-out curve does not start climbing again).
* :func:`estimate_remaining_seconds` — an ETA from the fit times measured so
  far, modelled as a fixed cost plus a per-splat cost (flat at small K, where
  convergence and whole-volume scoring dominate, then growing with K).
* :func:`extension_below` — the cheap points to add under the grid when K* lands
  on its smallest point, so the optimum is bracketed instead of only advised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: A grid point "improves" the curve only when it beats the best value so far
#: by more than this (dB). Matches the 0.1 dB per-step "worth it" bar that
#: :func:`~.curve_analysis.find_k_star` uses for its peak / still-climbing rules.
EARLY_STOP_MIN_GAIN_DB = 0.1

#: Consecutive non-improving points that end the sweep.
EARLY_STOP_PATIENCE = 2

#: A single point this far (dB) BELOW the best so far ends the sweep at once: a
#: clear decline past a peak, not jitter (twice the improvement bar). On a
#: doubling grid the points after the peak are the costliest of the sweep: on a
#: dense 37.7 Mvoxel synthetic stack (GB10) the foreground curve peaked at
#: K = 64000 (fit 10.5 min) and fell 0.25 dB at 128000 (fit 63 min), so waiting
#: for a second non-improving point would have spent ~2 h more on 256000.
EARLY_STOP_DECLINE_DB = 0.2

#: Points added below the grid when K* lands on its smallest point.
EXTEND_BELOW_POINTS = 2

#: Never extend below this many splats (a fit needs some splats to mean anything).
EXTEND_BELOW_FLOOR = 16


@dataclass
class SweepControl:
    """What the sweep did beyond the requested grid; recorded in ``cal.json``."""

    early_stop: bool = True
    """Whether early stopping was enabled."""
    early_stop_patience: int = EARLY_STOP_PATIENCE
    early_stop_min_gain_db: float = EARLY_STOP_MIN_GAIN_DB
    early_stop_decline_db: float = EARLY_STOP_DECLINE_DB
    stopped_early: bool = False
    """True when the plateau rule ended the sweep before the grid's last point."""
    k_values_skipped: List[int] = field(default_factory=list)
    """Grid points never fitted because the sweep stopped early."""
    minmax_curve_truncated: bool = False
    """True when the always-reported ``psnr_minmax`` K* (``held_out_peak``) was
    detected on a curve the early stop cut short while watching ANOTHER metric
    (the CLI default ``psnr_foreground``): it is the min--max optimum of the
    measured points only, not of the full grid (A549: 392 vs 1566)."""
    interrupted: bool = False
    """True when the sweep was cut short by Ctrl-C (the result is partial)."""
    extend_below: bool = True
    """Whether a K* on the smallest grid point triggers a downward extension."""
    k_values_extended_below: List[int] = field(default_factory=list)
    """Points added under the original grid (empty when no extension ran)."""
    grid_source: str = "explicit"
    """``explicit`` (``--k-grid``), ``bounds`` (``--k-min``/``--k-max``) or
    ``auto`` (bounds derived from the volume; see ``auto_grid``)."""
    auto_grid: Optional[Dict[str, Any]] = None
    """Provenance of a data-derived grid (see :func:`~.grid.auto_k_bounds`)."""

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data form for ``cal.json``."""
        from dataclasses import asdict

        return asdict(self)


def plateau_reached(
    values: Sequence[float],
    *,
    patience: int = EARLY_STOP_PATIENCE,
    min_gain_db: float = EARLY_STOP_MIN_GAIN_DB,
    decline_db: Optional[float] = EARLY_STOP_DECLINE_DB,
) -> bool:
    """True when the curve has stopped paying for more splats.

    Either the last ``patience`` points each failed to beat the best value
    before them by more than ``min_gain_db``, or the latest point fell at least
    ``decline_db`` below the best so far (a clear decline past a peak). A
    non-finite point counts as non-improving (a failed fit is no evidence the
    curve is still rising). Needs at least one finite point before the window,
    so the sweep never stops on its opening points.
    """
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    vals = [float(v) for v in values]
    prior = [v for v in vals[:-1] if math.isfinite(v)]
    if (
        decline_db is not None
        and prior
        and math.isfinite(vals[-1])
        and vals[-1] <= max(prior) - decline_db
    ):
        return True
    if len(vals) <= patience:
        return False
    head = [v for v in vals[:-patience] if math.isfinite(v)]
    if not head:
        return False
    best = max(head)
    for v in vals[-patience:]:
        if math.isfinite(v) and v > best + min_gain_db:
            return False
        if math.isfinite(v):
            best = max(best, v)
    return True


def _linear_cost(ks: Sequence[int], times: Sequence[float]) -> tuple[float, float]:
    """``(a, b)`` for ``t = a + b * K``, least squares with ``a, b >= 0``.

    A fit's wall-clock is a fixed cost (convergence, render and scoring of the
    whole volume) plus a per-splat cost that only shows at large K: on the A549
    stack (3 Mvoxel, GB10) the time hovered at 7-15 s from K = 250 to 16000,
    then grew ~linearly (42 s at 64000, 77 s at 128000, 161 s at 256000). A
    least-squares line over every point averages out the jitter of the flat
    regime (a two-point slope there predicted 1h42m for a 10-minute sweep). The
    first point is dropped once there are three, since it pays the one-off
    warm-up (kernel compilation), and a slope needs three points: two jittery
    flat-regime points would make one up.
    """
    k = [float(x) for x in ks]
    t = [max(float(x), 0.0) for x in times]
    if len(k) >= 3:
        k, t = k[1:], t[1:]
    n = len(k)
    k_mean, t_mean = sum(k) / n, sum(t) / n
    var = sum((x - k_mean) ** 2 for x in k)
    b = (
        sum((x - k_mean) * (y - t_mean) for x, y in zip(k, t, strict=True)) / var
        if var > 0 and n >= 3
        else 0.0
    )
    if b <= 0.0:
        return t_mean, 0.0
    a = t_mean - b * k_mean
    if a < 0.0:
        return 0.0, sum(x * y for x, y in zip(k, t, strict=True)) / sum(
            x * x for x in k
        )
    return a, b


def estimate_remaining_seconds(
    done_k: Sequence[int],
    done_seconds: Sequence[float],
    remaining_k: Sequence[int],
) -> float:
    """Estimated wall-clock for the remaining grid points (seconds).

    Extrapolates the measured fit times with :func:`_linear_cost`. Returns 0
    when nothing remains and NaN when nothing has been measured yet. Early in a
    sweep (fixed-cost regime) it UNDER-estimates the tail badly (0.1x of the
    actual remaining fit time over the first four points of the dense A549 and
    cells3d sweeps); past K = 32000 it read 0.4-0.6x (A549) and 0.4-1.8x
    (cells3d), since an `n2s` fit's time follows its convergence as much as K.
    A rough guide, labelled ``eta~``.
    """
    if not remaining_k:
        return 0.0
    if not done_k:
        return float("nan")
    a, b = _linear_cost(done_k, done_seconds)
    return float(sum(a + b * int(k) for k in remaining_k))


def format_duration(seconds: float) -> str:
    """``42s`` / ``3m05s`` / ``1h12m`` (``?`` when unknown)."""
    if not math.isfinite(seconds):
        return "?"
    s = int(round(max(seconds, 0.0)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def extension_below(
    k_grid: Sequence[int],
    *,
    n_points: int = EXTEND_BELOW_POINTS,
    floor: int = EXTEND_BELOW_FLOOR,
) -> List[int]:
    """Points to add under ``k_grid`` (ascending), continuing its bottom ratio.

    The ratio is the grid's own first step (``k[1] / k[0]``), clamped to
    ``[1.5, 4]`` so an irregular explicit grid neither crawls nor leaps. Points
    at or below ``floor`` are dropped, as are duplicates.
    """
    ks = sorted(int(k) for k in k_grid)
    if not ks:
        return []
    ratio = ks[1] / ks[0] if len(ks) >= 2 and ks[0] > 0 else 2.0
    ratio = min(max(ratio, 1.5), 4.0)
    out: List[int] = []
    k = float(ks[0])
    for _ in range(n_points):
        k /= ratio
        ki = int(round(k))
        if ki < floor or (out and ki >= out[-1]):
            break
        out.append(ki)
    return sorted(out)
