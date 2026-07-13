"""K-sweep curve analysis: grid, peak detection, density transfer, R-D model.

Everything that turns a sweep of ``(K, held-out PSNR / MSE)`` samples into a
decision: the K-grid constructor, the hybrid peak/plateau/signal-limited
detector, the transferable splat-density (cal→planner) model with its
multi-scale exponent fit, and a parametric rate-distortion model for cheap
extrapolation + convergence flagging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# K-grid construction
# =============================================================================


def build_k_grid(
    explicit: Optional[Sequence[int]] = None,
    n_points: int = 10,
    k_min: int = 1_000,
    k_max: int = 512_000,
    progression: str = "exp",
    power: int = 2,
) -> List[int]:
    """Construct a sweep grid of splat counts.

    When ``explicit`` is provided it takes precedence; otherwise
    ``n_points`` values are placed between ``k_min`` and ``k_max``
    according to ``progression``:

    * ``"exp"`` — log-spaced (geometric). Default. Matches the
      manuscript's ``{1K, 2K, ..., 512K}`` at ``n_points=10``,
      ``k_min=1000``, ``k_max=512000``.
    * ``"power"`` — polynomial: ``K_i = k_min + (k_max - k_min) *
      (i/(N-1))**power``. Denser at low K when ``power > 1``.

    Duplicates from rounding are removed but the sequence is kept
    monotonic. Endpoints are guaranteed to be exactly ``k_min`` and
    ``k_max``.
    """
    if explicit is not None:
        explicit_out = [int(k) for k in explicit]
        if len(explicit_out) < 2:
            raise ValueError(
                f"explicit grid needs at least 2 points, got {len(explicit_out)}"
            )
        if any(k <= 0 for k in explicit_out):
            raise ValueError(f"all K values must be positive, got {explicit_out}")
        return explicit_out

    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if k_min <= 0 or k_max <= 0:
        raise ValueError(f"k_min and k_max must be positive, got ({k_min}, {k_max})")
    if k_max <= k_min:
        raise ValueError(f"k_max must be > k_min, got ({k_min}, {k_max})")

    if progression == "exp":
        log_min = math.log(k_min)
        log_max = math.log(k_max)
        raw = [
            int(round(math.exp(log_min + (log_max - log_min) * i / (n_points - 1))))
            for i in range(n_points)
        ]
    elif progression == "power":
        if power < 1:
            raise ValueError(f"power must be >= 1, got {power}")
        raw = [
            int(round(k_min + (k_max - k_min) * (i / (n_points - 1)) ** power))
            for i in range(n_points)
        ]
    else:
        raise ValueError(f"unknown progression {progression!r}; use 'exp' or 'power'")

    # Pin endpoints exactly (rounding may drift)
    raw[0] = k_min
    raw[-1] = k_max

    # Deduplicate while preserving order; ensure strictly monotonic
    parametric_out: List[int] = []
    for k in raw:
        if not parametric_out or k > parametric_out[-1]:
            parametric_out.append(k)
        else:
            parametric_out.append(parametric_out[-1] + 1)
    return parametric_out


# =============================================================================
# Peak detection
# =============================================================================


@dataclass
class HeldOutPeak:
    """Detected K* and qualitative shape of the held-out PSNR curve."""

    k_star: int
    """The recommended splat count."""

    type: Literal["peak", "plateau", "signal_limited"]
    """``peak`` — clear interior maximum; ``plateau`` — flat top, smallest K
    within 0.3 dB returned; ``signal_limited`` — monotone-rising through
    the largest tested K (no peak in sampled range)."""

    confidence_db: float
    """For ``peak``: margin to the second-best K in dB.
    For ``plateau``: spread across the in-tolerance plateau.
    For ``signal_limited``: total dB rise across the sweep."""

    # --- Operating point (point of diminishing returns) + supporting metadata ---
    # These are additive and defaulted so older cal.json files hydrate cleanly.
    k_knee: int = 0
    """The point of diminishing returns, independent of the budget anchor
    ``k_star`` and of the regime label: the interior argmax for a clear peak,
    otherwise the smallest K within ``knee_margin_db`` of the maximum. For
    ``peak`` and ``plateau`` this equals ``k_star``; for ``signal_limited`` it
    is the (earlier) knee while ``k_star`` remains the last/max K used for the
    splat budget. Defaults to 0; callers that predate this field should fall
    back to ``k_star`` (``from_json`` does this)."""

    knee_idx: int = -1
    """Positional index of ``k_knee`` in the input ``k_values`` (−1 if unset)."""

    drop_after_peak_db: float = 0.0
    """Held-out PSNR at the last K minus the peak (≤ 0; its magnitude is the
    post-peak overfitting drop for a ``peak`` curve)."""

    tail_rise_db: float = 0.0
    """Mean per-step held-out rise over the trailing run of adjacent finite K
    (the still-climbing discriminator; ≥ 0.1 dB drives ``signal_limited``)."""

    plateau_spread_db: float = 0.0
    """Peak minus the smallest held-out value among K within
    ``knee_margin_db`` of the maximum (how flat the in-tolerance top is)."""

    total_rise_db: float = 0.0
    """Peak minus the first finite held-out value (total climb across the sweep)."""

    still_climbing: bool = False
    """True when the curve is signal-limited: argmax at the last K, total rise
    across the sweep ≥ ``knee_margin_db`` (0.3 dB), the trailing tail still
    rising ≥ 0.1 dB/step on average, and the final step ≥ 0.05 dB. ``k_star`` is
    then the last K (budget anchor); ``k_knee`` may still be an earlier K when
    one is already within ``knee_margin_db`` of the maximum, and equals the last
    K only when no earlier K is that close."""

    knee_margin_db: float = 0.3
    """The dB tolerance used to locate the knee / plateau onset."""


def find_k_star(
    k_values: Sequence[int],
    held_out_psnr_values: Sequence[float],
) -> HeldOutPeak:
    """Detect the held-out PSNR peak via the manuscript's hybrid rule.

    Hybrid rule (``splat_count_vs_quality`` §4.2):

    1. **Peak**: the argmax is strictly interior AND both ``mean(pre-argmax)``
       and ``mean(post-argmax)`` are at least 0.1 dB below the peak. Return
       the argmax.
    2. **Signal-limited**: the argmax is the last K, the curve rose by
       ≥ 0.3 dB across the sweep, AND it is *still climbing at the top*
       (mean rise over the trailing run of adjacent finite steps ≥ 0.1 dB
       AND the single final step ≥ 0.05 dB). Return the last K. The tail
       checks stop a flat-topped plateau (whose noisy max lands on the
       last K) from being misread as signal-limited.
    3. **Plateau**: otherwise. Return the smallest K within 0.3 dB of the
       maximum — the onset of diminishing returns.

    The returned :class:`HeldOutPeak` reports both ``k_star`` (the budget
    anchor above — max K for ``signal_limited``) and ``k_knee`` (the point of
    diminishing returns: the argmax for a clear peak, else the knee), plus
    supporting per-regime metadata. ``k_knee`` equals ``k_star`` for ``peak``
    and ``plateau`` and is the earlier knee for ``signal_limited``; it is the
    field to use when a single "reasonable operating point" is wanted
    regardless of regime.
    """
    knee_margin_db = 0.3
    k_arr = np.asarray(list(k_values), dtype=int)
    psnr_arr = np.asarray(list(held_out_psnr_values), dtype=float)
    if k_arr.size != psnr_arr.size:
        raise ValueError(
            f"k_values has {k_arr.size} entries but psnr has {psnr_arr.size}"
        )
    n = k_arr.size
    if n < 2:
        raise ValueError(f"need at least 2 K values, got {n}")

    finite = np.isfinite(psnr_arr)
    if not finite.any():
        raise ValueError("all held-out PSNR values are non-finite")

    argmax = int(np.argmax(np.where(finite, psnr_arr, -np.inf)))
    peak = float(psnr_arr[argmax])

    # --- Common curve metadata (computed once, attached to every regime) ---
    fin_idx = np.flatnonzero(finite)
    first_finite_psnr = float(psnr_arr[fin_idx[0]])
    last_finite_psnr = float(psnr_arr[fin_idx[-1]])
    total_rise = peak - first_finite_psnr
    drop_after_peak = last_finite_psnr - peak  # ≤ 0

    # Knee = smallest K within ``knee_margin_db`` of the max (always well-defined;
    # the argmax itself qualifies). This is the plateau/diminishing-returns onset.
    above_idx = np.where(np.where(finite, psnr_arr, -np.inf) >= peak - knee_margin_db)[
        0
    ]
    knee_idx = int(above_idx[0])
    plateau_spread = float(
        peak - psnr_arr[above_idx][np.isfinite(psnr_arr[above_idx])].min()
    )

    # "Still climbing at the top" = mean per-step rise over the trailing run of
    # *adjacent* finite K's (up to 3 steps). Adjacency guards against a NaN/inf gap
    # reading as a climb (M1); averaging guards against a single sub-0.1 dB final
    # step demoting a steadily-climbing curve to plateau (M2). ``last_step`` is the
    # single final inter-sample step: it prevents an earlier large step from
    # carrying the averaged ``tail_rise`` over threshold when the curve has in fact
    # gone flat at the very top (M3) — which would otherwise mislabel a narrow
    # flat-topped plateau as ``signal_limited`` and inflate the K*-derived budget.
    tail_rise = 0.0
    last_step = 0.0
    if fin_idx.size >= 2 and int(fin_idx[-1] - fin_idx[-2]) == 1:
        run = [int(fin_idx[-1])]
        for j in range(fin_idx.size - 2, -1, -1):
            if int(fin_idx[j + 1] - fin_idx[j]) == 1 and len(run) < 4:
                run.append(int(fin_idx[j]))
            else:
                break
        seg = psnr_arr[np.array(sorted(run))]
        if seg.size >= 2:
            tail_rise = float(np.mean(np.diff(seg)))
            last_step = float(seg[-1] - seg[-2])

    # --- Clear interior peak? (both flanks ≥ 0.1 dB below the peak) ---
    pre = psnr_arr[:argmax]
    post = psnr_arr[argmax + 1 :]
    pre_finite = pre[np.isfinite(pre)]
    post_finite = post[np.isfinite(post)]
    is_clear_peak = (
        pre_finite.size > 0
        and post_finite.size > 0
        and (peak - float(np.mean(pre_finite))) >= 0.1
        and (peak - float(np.mean(post_finite))) >= 0.1
    )
    # last-step floor (0.05 dB) is half the 0.1 dB per-step "worth it" bar: the
    # final doubling must itself still deliver a non-trivial gain, so a flat top
    # after an earlier steep climb reads as a plateau, not signal-limited.
    still_climbing = (
        argmax == n - 1
        and total_rise >= knee_margin_db
        and tail_rise >= 0.1
        and last_step >= 0.05
    )

    # ``k_knee`` = the point of diminishing returns (replicates the manuscript's
    # ``find_cv_optimal_idx``): the argmax for a clear peak, else the knee. It is
    # decoupled from both the regime label and the budget anchor ``k_star``.
    knee_operating_idx = argmax if is_clear_peak else knee_idx

    def _build(k_star_idx: int, kind: str, confidence: float) -> HeldOutPeak:
        return HeldOutPeak(
            k_star=int(k_arr[k_star_idx]),
            type=kind,  # type: ignore[arg-type]
            confidence_db=float(confidence),
            k_knee=int(k_arr[knee_operating_idx]),
            knee_idx=int(knee_operating_idx),
            drop_after_peak_db=float(drop_after_peak),
            tail_rise_db=float(tail_rise),
            plateau_spread_db=float(plateau_spread),
            total_rise_db=float(total_rise),
            still_climbing=bool(still_climbing),
            knee_margin_db=float(knee_margin_db),
        )

    # 1. Peak — clear interior maximum; k_star = k_knee = argmax.
    if is_clear_peak:
        sorted_psnr = np.sort(psnr_arr[finite])
        margin = (
            float(sorted_psnr[-1] - sorted_psnr[-2])
            if sorted_psnr.size >= 2
            else float(peak)
        )
        return _build(argmax, "peak", margin)

    # 2. Signal-limited — argmax at last K, rose meaningfully, still climbing.
    #    k_star = last/max K (the budget anchor a still-detail-limited tile needs),
    #    while k_knee is the (earlier) diminishing-returns knee.
    if still_climbing:
        return _build(n - 1, "signal_limited", total_rise)

    # 3. Plateau — flat top; k_star = k_knee = knee.
    return _build(knee_idx, "plateau", plateau_spread)


# =============================================================================
# Transferable splat density (cal -> planner interface)
# =============================================================================


@dataclass
class SplatDensity:
    """Transferable splat budget derived from one calibration.

    The investigation found splats-to-saturate scales **sub-linearly** with
    feature content (``K ~ features^alpha``, alpha≈0.44; n_peaks the best
    predictor). This packages K* + the reference feature count + the exponent
    so any tile can get a budget via :meth:`predict_k` without re-calibrating —
    the cal→planner interface. Assumes tiles of roughly the reference scale
    ("calibrate at the scale you fit at").
    """

    feature_method: str
    n_features_reference: int
    k_star_reference: int
    saturation_exponent: float
    """Sub-linear exponent ``alpha`` in ``K ~ features^alpha`` (default 0.44)."""
    saturation_cap: int
    """Effective K beyond which the reference region overfits / plateaus."""
    splats_per_feature: float
    """Linear reference density ``k_star / n_features`` (for reporting)."""
    feature_threshold: float = 0.0
    """Absolute intensity threshold used to count ``n_features_reference``. The
    planner must scan at this same absolute level so its per-box counts are on the
    same scale as the reference (threshold-relative-to-local-max does not compose
    across regions, especially with hot outliers)."""

    def predict_k(self, n_features: int) -> int:
        """Predict the splat budget for a region with ``n_features`` features."""
        if self.n_features_reference <= 0:
            k = float(self.k_star_reference)
        else:
            ratio = max(0.0, float(n_features)) / float(self.n_features_reference)
            k = float(self.k_star_reference) * (ratio**self.saturation_exponent)
        return int(min(max(round(k), 0), self.saturation_cap))


@dataclass
class ExponentFit:
    """Multi-scale fit of the saturation exponent ``alpha`` in ``K ~ features^alpha``.

    The single-scale calibration assumes the empirical default ``alpha=0.44``;
    this *measures* it by calibrating K* at several region scales (each a
    different feature count) and regressing ``log K*`` on ``log n_features``.
    The slope is ``alpha``; the per-scale ``(n_features, k_star)`` points and the
    fit ``r_squared`` are kept for reporting / provenance.
    """

    alpha: float
    """Fitted sub-linear exponent (the regression slope in log-log space)."""
    intercept: float
    """Log-space intercept ``log C`` (so ``K = exp(intercept)·features^alpha``)."""
    r_squared: float
    """Goodness-of-fit of the log-log regression (1 = perfect power law). **NaN**
    when it cannot be assessed: fewer than 3 *distinct* feature counts (a line
    through 2 points is trivially perfect), or zero K* variance (a flat/degenerate
    fit). A NaN here means "treat the exponent as provisional"."""
    n_points: int
    """Total scales measured (before collapsing duplicate feature counts)."""
    n_distinct: int
    """Distinct feature counts actually regressed (the meaningful sample size).
    ``< 3`` ⇒ ``r_squared`` is NaN (under-determined)."""
    scales: List[int]
    """Region edge lengths (voxels), one per distinct regressed point."""
    n_features: List[int]
    """Feature count of each distinct regressed point (shared-threshold count)."""
    k_star: List[int]
    """Effective K* at each distinct regressed point."""


def fit_saturation_exponent(
    points: "Sequence[Tuple[int, float, float]]",
) -> Optional[ExponentFit]:
    """Least-squares fit of ``alpha`` in ``K ~ features^alpha`` (log-log regression).

    ``points`` is a sequence of ``(scale, n_features, k_star)`` triples (one per
    calibrated region scale). Duplicate feature counts (scales that clamped to the
    same crop) are collapsed to one point. Returns ``None`` when fewer than two
    *distinct* feature counts remain (no spread to fit a slope).

    ``r_squared`` is set to **NaN** when it cannot be meaningfully assessed —
    fewer than three distinct points (a 2-point line is always perfect), or zero
    K* variance (a degenerate flat fit, ``alpha≈0``) — so a caller's
    goodness-of-fit gate is not fooled by a structural ``R²==1.0``.
    """
    clean = [
        (int(s), float(nf), float(k))
        for (s, nf, k) in points
        if np.isfinite(nf) and np.isfinite(k) and nf > 0 and k > 0
    ]
    if len(clean) < 2:
        return None

    # Collapse duplicate feature counts (clamped/identical crops → one region):
    # keep the smallest scale and the mean K* per distinct feature count, so the
    # regression — and its reported sample size — reflect distinct evidence only.
    by_feat: Dict[float, List[Tuple[int, float]]] = {}
    for s, nf, k in clean:
        by_feat.setdefault(nf, []).append((s, k))
    distinct_feats = sorted(by_feat)
    if len(distinct_feats) < 2:
        return None  # degenerate: every scale had the same feature count

    scales_out = [min(s for s, _ in by_feat[feat]) for feat in distinct_feats]
    nf_arr = np.asarray(distinct_feats, dtype=float)
    ks_arr = np.asarray(
        [float(np.mean([k for _, k in by_feat[f]])) for f in distinct_feats],
        dtype=float,
    )

    x = np.log(nf_arr)
    y = np.log(ks_arr)
    # Guard an ill-conditioned slope: distinct-but-near-identical feature counts
    # (e.g. 1e6 vs 1e6+1 on a huge volume) give a near-zero x-spread in log space,
    # so lstsq returns a noise-dominated alpha. ptp(x) < 1e-3 ≈ a <0.1% feature
    # ratio between the extremes — too little spread to regress. Treat as
    # un-fittable (caller keeps the default exponent) rather than emit garbage.
    if float(np.ptp(x)) < 1e-3:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, intercept = float(sol[0]), float(sol[1])
    pred = alpha * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n_distinct = len(distinct_feats)
    # A 2-point line is trivially perfect (ss_res==0); a flat fit (ss_tot==0) is
    # degenerate, not perfect. Report NaN rather than a misleading R²==1.0.
    if n_distinct < 3 or ss_tot == 0.0:
        r2 = float("nan")
    else:
        r2 = 1.0 - ss_res / ss_tot
    return ExponentFit(
        alpha=alpha,
        intercept=intercept,
        r_squared=float(r2),
        n_points=len(clean),
        n_distinct=n_distinct,
        scales=scales_out,
        n_features=[int(v) for v in nf_arr],
        k_star=[int(round(v)) for v in ks_arr],
    )


# =============================================================================
# Parametric rate-distortion model
# =============================================================================


@dataclass
class RDModel:
    """Parametric fit of held-out error vs K: ``error ≈ floor + a·K^-beta``.

    Lets us extrapolate the sweep cheaply and, crucially, flag when a curve is
    still climbing at K_max (``converged_fraction`` < 1) — the cheap detector
    that would have caught the original ``signal_limited`` false alarm without
    an expensive dense high-K sweep.
    """

    floor: float
    a: float
    beta: float
    rmse: float
    n_points: int
    converged_fraction: float
    """Fraction of the achievable error drop realised by K_max (1 = converged)."""

    def predict_error(self, k: float) -> float:
        return float(self.floor + self.a * (float(k) ** (-self.beta)))

    def k_for_error(self, target_error: float) -> float:
        """Invert the model: smallest K reaching ``target_error`` (inf if < floor)."""
        if target_error <= self.floor or self.a <= 0.0 or self.beta <= 0.0:
            return float("inf")
        return float((self.a / (target_error - self.floor)) ** (1.0 / self.beta))


def fit_rd_model(
    k_values: Sequence[float], error_values: Sequence[float]
) -> Optional[RDModel]:
    """Fit ``error ≈ floor + a·K^-beta`` (least-squares). ``None`` if <3 finite
    points or the fit fails. ``error_values`` should be held-out MSE."""
    k = np.asarray(list(k_values), dtype=float)
    e = np.asarray(list(error_values), dtype=float)
    finite = np.isfinite(k) & np.isfinite(e) & (k > 0) & (e > 0)
    k, e = k[finite], e[finite]
    if k.size < 3:
        return None
    try:
        from scipy.optimize import curve_fit
    except Exception:  # pragma: no cover - scipy is a dependency
        return None

    def _model(kk: np.ndarray, floor: float, a: float, beta: float) -> np.ndarray:
        return np.asarray(floor + a * np.power(kk, -beta), dtype=float)

    floor0 = max(0.0, float(e.min()) * 0.5)
    a0 = float(max(e.max() - floor0, 1e-12)) * (float(k.min()) ** 0.5)
    p0 = [floor0, a0, 0.5]
    bounds = ([0.0, 0.0, 1e-2], [float(e.max()) if e.max() > 0 else 1.0, np.inf, 5.0])
    try:
        popt, _ = curve_fit(_model, k, e, p0=p0, bounds=bounds, maxfev=10_000)
    except Exception:
        return None
    floor, a, beta = (float(x) for x in popt)
    pred = _model(k, floor, a, beta)
    rmse = float(np.sqrt(np.mean((pred - e) ** 2)))
    e_kmin = float(_model(np.array([k.min()]), floor, a, beta)[0])
    e_kmax = float(_model(np.array([k.max()]), floor, a, beta)[0])
    denom = e_kmin - floor
    converged = float((e_kmin - e_kmax) / denom) if denom > 1e-12 else 1.0
    return RDModel(
        floor=floor,
        a=a,
        beta=beta,
        rmse=rmse,
        n_points=int(k.size),
        converged_fraction=float(np.clip(converged, 0.0, 1.0)),
    )
