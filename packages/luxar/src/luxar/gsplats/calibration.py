"""Blind-spot cross-validation for Gaussian-splat model selection.

Implements the manuscript's calibration protocol (Supp. Doc. 2,
``splat_count_vs_quality``):

1. Mask 5% of voxels with a deterministic Bernoulli draw (seed=42).
2. Replace masked voxels with the median of their 3^D donut neighbourhood
   (centre excluded) — Noise2Self self-supervision.
3. Fit a Gaussian-splat model on the donut-filled volume at each ``K`` in
   a sweep; the optimiser never sees the original noisy values at masked
   positions.
4. Evaluate held-out PSNR against the *original* (pre-fill) values at the
   masked positions.
5. The ``K`` that maximises held-out PSNR is the principled splat budget
   — capacity beyond ``K*`` memorises noise rather than signal.

As a free byproduct, an ensemble noise-floor estimator (Laplacian +
Haar HH + background MAD) places each dataset in absolute terms.

The protocol is purely additive: ``fit_gaussian_splats`` is called
unchanged at each ``K``; this module owns mask generation, donut fill,
held-out evaluation, K-grid construction, peak detection, and noise-floor
estimation.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# CV mask
# =============================================================================


def cv_mask(
    shape: Tuple[int, ...],
    fraction: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic Bernoulli boolean mask for blind-spot cross-validation.

    Defaults match Batson & Royer (2019) and the Luxar manuscript: 5% of
    voxels are held out with seed 42.

    Parameters
    ----------
    shape : tuple of int
        Output array shape.
    fraction : float, default=0.05
        Probability of any voxel being marked True (held out).
    seed : int, default=42
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray of bool, shape ``shape``
        ``True`` at held-out positions, ``False`` elsewhere.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    rng = np.random.RandomState(seed)
    return rng.rand(*shape) < fraction


# =============================================================================
# Donut-median fill
# =============================================================================


def donut_median_fill(
    V: np.ndarray,
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Replace masked voxels with the median of their donut neighbourhood.

    The donut is the ``(2r+1)^D`` cube around each masked voxel with the
    centre excluded — 26 neighbours in 3D when ``r=1``. Operates on
    arrays of arbitrary dimension (works for 2D, 3D, 4D, ...). Edge
    voxels use ``mode='reflect'`` padding.

    Vectorised: gathers donut values for all masked positions at once
    via stacked shifted-index lookups against a single padded copy of
    ``V``. Memory cost: ``(2r+1)^D - 1`` floats per masked voxel.

    Parameters
    ----------
    V : np.ndarray
        Volume to fill.
    mask : np.ndarray of bool, same shape as ``V``
        ``True`` at positions to replace.
    radius : int, default=1
        Donut half-width. Default ``1`` → ``3^D`` neighbourhood, matching
        the manuscript.

    Returns
    -------
    np.ndarray, same shape and dtype as ``V``
        Copy of ``V`` with masked voxels replaced by donut medians.
        Unmasked voxels are unchanged.
    """
    if V.shape != mask.shape:
        raise ValueError(f"V shape {V.shape} != mask shape {mask.shape}")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    D = V.ndim
    # Donut footprint: all (2r+1)^D offsets except (0,...,0)
    offsets = [
        o
        for o in itertools.product(range(-radius, radius + 1), repeat=D)
        if any(c != 0 for c in o)
    ]

    masked_idx = np.nonzero(mask)
    n_masked = masked_idx[0].size if len(masked_idx) > 0 else 0

    if n_masked == 0:
        out_empty: np.ndarray = V.copy()
        return out_empty

    # Pad with reflect so edge voxels have full neighbourhoods
    V_pad = np.pad(V, radius, mode="reflect")

    n_donut = len(offsets)
    donut_values = np.empty((n_donut, n_masked), dtype=V.dtype)

    for k, offset in enumerate(offsets):
        shifted = tuple(masked_idx[d] + radius + offset[d] for d in range(D))
        donut_values[k] = V_pad[shifted]

    median_values = np.median(donut_values, axis=0)

    V_filled: np.ndarray = V.copy()
    V_filled[masked_idx] = median_values.astype(V.dtype, copy=False)
    return V_filled


# =============================================================================
# Held-out PSNR
# =============================================================================


def held_out_psnr(
    V_hat: np.ndarray,
    V_original: np.ndarray,
    mask: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """PSNR of reconstruction at masked voxels vs the original (pre-fill) values.

    Parameters
    ----------
    V_hat : np.ndarray
        Reconstructed volume from the splat fit.
    V_original : np.ndarray
        The unmodified original volume (NOT the donut-filled one).
    mask : np.ndarray of bool
        Held-out mask. Must broadcast to the volume shape.
    data_range : float, optional
        Dynamic range for PSNR. If ``None``, uses
        ``V_original.max() - V_original.min()`` over the whole volume.

    Returns
    -------
    float
        PSNR in dB. ``+inf`` when MSE is zero, ``nan`` when mask is empty.
    """
    if V_hat.shape != V_original.shape:
        raise ValueError(
            f"shape mismatch: V_hat {V_hat.shape} vs V_original {V_original.shape}"
        )
    if mask.shape != V_original.shape:
        raise ValueError(
            f"mask shape {mask.shape} != volume shape {V_original.shape}"
        )

    held_pred = V_hat[mask]
    held_true = V_original[mask]

    if held_pred.size == 0:
        return float("nan")

    mse = float(np.mean((held_pred - held_true) ** 2))
    if mse == 0.0:
        return float("inf")
    if data_range is None:
        data_range = float(V_original.max() - V_original.min())
    if data_range == 0.0:
        return float("inf")
    return float(10.0 * math.log10(data_range**2 / mse))


# =============================================================================
# Noise-floor estimation
# =============================================================================


@dataclass
class NoiseFloor:
    """Ensemble noise-floor estimate for a [0, 1]-normalised volume.

    All ``sigma_*`` fields are noise standard deviations in the volume's
    intensity units. ``psnr_max_db`` is the corresponding PSNR ceiling
    assuming a ``data_range = 1.0``.
    """

    sigma_hat: float
    """Ensemble estimate (median of available high-pass estimators)."""

    sigma_laplacian: float
    """Discrete Laplacian MAD (Immerkaer 1996, kernel-norm = ``2D(2D+1)``)."""

    sigma_haar: float
    """Haar HH-subband MAD over slice-pairs (Donoho & Johnstone 1994)."""

    sigma_background: float
    """MAD of voxels in the bottom 10% intensity percentile."""

    psnr_max_db: float
    """``-20 log10(sigma_hat)`` for [0,1] data; ``+inf`` when ``sigma_hat == 0``."""


def _laplacian_mad(V: np.ndarray) -> float:
    """Sigma estimate via the discrete-Laplacian MAD (Immerkaer 1996).

    The Laplacian kernel is the sum of axis-wise centred second
    differences. Its squared L2 norm is ``K = 2 D (2D + 1)`` where ``D``
    is the array dimensionality (``D = 3 → K = 42``).
    """
    D = V.ndim
    L = np.zeros_like(V, dtype=np.float64)
    for axis in range(D):
        forward = np.roll(V, -1, axis=axis)
        backward = np.roll(V, 1, axis=axis)
        L += forward - 2.0 * V + backward
    K = 2.0 * D * (2.0 * D + 1.0)
    mad = float(np.median(np.abs(L)))
    return mad / (0.6745 * math.sqrt(K))


def _haar_mad(V: np.ndarray) -> float:
    """Sigma estimate via the Haar HH-subband MAD over the last two axes.

    For a slice ``Y``, the HH coefficient is
    ``D[y,x] = Y[y,x] - Y[y,x+1] - Y[y+1,x] + Y[y+1,x+1]``;
    its MAD divided by ``0.6745 * 2`` is an unbiased ``sigma`` estimate
    under independent Gaussian noise. For >2D arrays, the statistic is
    pooled over all ``Y``-``X`` slices along the leading axes.
    """
    if V.ndim < 2:
        return float("nan")
    if V.ndim == 2:
        slices: List[np.ndarray] = [V]
    else:
        leading = V.shape[:-2]
        slices = [V[idx] for idx in itertools.product(*(range(s) for s in leading))]

    diffs: List[np.ndarray] = []
    for sl in slices:
        if sl.shape[0] < 2 or sl.shape[1] < 2:
            continue
        d = sl[:-1, :-1] - sl[:-1, 1:] - sl[1:, :-1] + sl[1:, 1:]
        diffs.append(d.reshape(-1))
    if not diffs:
        return float("nan")
    d_all = np.concatenate(diffs)
    mad = float(np.median(np.abs(d_all)))
    return mad / (0.6745 * 2.0)


def _background_mad(V: np.ndarray, percentile: float = 10.0) -> float:
    """Sigma estimate from voxels below the ``percentile``-th intensity.

    Sensitive to detector noise floors and pre-processing clamps; tends
    to *under*-estimate when the dark tail is quantised or clipped, so
    the ensemble takes the median across estimators rather than the mean.
    """
    threshold = float(np.percentile(V, percentile))
    bg = V[V <= threshold]
    if bg.size == 0:
        return float("nan")
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    return mad / 0.6745


def estimate_noise_floor(V: np.ndarray) -> NoiseFloor:
    """Three-estimator ensemble noise-floor estimate.

    Returns the median of the (Laplacian, Haar, background) estimators
    that finite-valued — robust to one outlier on the low side
    (typical when the dark tail is quantised, e.g. ``acto3d_heart_nuclei``
    in the manuscript).

    The PSNR ceiling assumes ``data_range = 1.0`` (the [0, 1]
    normalisation enforced by ``fit_gaussian_splats``). When ``sigma_hat``
    is exactly zero (saturation at float32 precision), the ceiling is
    ``+inf``; callers can clamp to a conservative finite value.
    """
    sl = _laplacian_mad(V)
    sh = _haar_mad(V)
    sb = _background_mad(V)
    candidates = [s for s in (sl, sh, sb) if not (math.isnan(s) or math.isinf(s))]
    if not candidates:
        sigma_hat = float("nan")
    else:
        sigma_hat = float(np.median(candidates))

    if sigma_hat == 0.0:
        psnr_max_db = float("inf")
    elif math.isnan(sigma_hat):
        psnr_max_db = float("nan")
    else:
        psnr_max_db = float(-20.0 * math.log10(sigma_hat))

    return NoiseFloor(
        sigma_hat=sigma_hat,
        sigma_laplacian=sl,
        sigma_haar=sh,
        sigma_background=sb,
        psnr_max_db=psnr_max_db,
    )


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


def find_k_star(
    k_values: Sequence[int],
    held_out_psnr_values: Sequence[float],
) -> HeldOutPeak:
    """Detect the held-out PSNR peak via the manuscript's hybrid rule.

    Hybrid rule (``splat_count_vs_quality`` §4.2):

    1. **Peak**: the argmax is strictly interior AND both ``mean(pre-argmax)``
       and ``mean(post-argmax)`` are at least 0.1 dB below the peak. Return
       the argmax.
    2. **Signal-limited**: the argmax is the last K and the curve rose by
       ≥ 0.3 dB across the sweep (monotone rise in range). Return the
       last K.
    3. **Plateau**: otherwise. Return the smallest K within 0.3 dB of the
       maximum — the onset of diminishing returns.
    """
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

    # 1. Peak
    pre = psnr_arr[:argmax]
    post = psnr_arr[argmax + 1 :]
    pre_finite = pre[np.isfinite(pre)]
    post_finite = post[np.isfinite(post)]
    if pre_finite.size > 0 and post_finite.size > 0:
        pre_gap = peak - float(np.mean(pre_finite))
        post_gap = peak - float(np.mean(post_finite))
        if pre_gap >= 0.1 and post_gap >= 0.1:
            sorted_psnr = np.sort(psnr_arr[finite])
            margin = float(sorted_psnr[-1] - sorted_psnr[-2]) if sorted_psnr.size >= 2 else float(peak)
            return HeldOutPeak(
                k_star=int(k_arr[argmax]),
                type="peak",
                confidence_db=margin,
            )

    # 2. Signal-limited
    first_finite_psnr = float(psnr_arr[finite][0])
    if argmax == n - 1 and (peak - first_finite_psnr) >= 0.3:
        return HeldOutPeak(
            k_star=int(k_arr[-1]),
            type="signal_limited",
            confidence_db=float(peak - first_finite_psnr),
        )

    # 3. Plateau
    threshold = peak - 0.3
    above_idx = np.where(np.where(finite, psnr_arr, -np.inf) >= threshold)[0]
    smallest_above = int(above_idx[0])
    plateau_spread = float(peak - psnr_arr[above_idx][np.isfinite(psnr_arr[above_idx])].min())
    return HeldOutPeak(
        k_star=int(k_arr[smallest_above]),
        type="plateau",
        confidence_db=plateau_spread,
    )


# =============================================================================
# Result container
# =============================================================================


@dataclass
class CalibrationResult:
    """Output of :func:`calibrate`. Serialisable to JSON."""

    k_values_requested: List[int]
    """K values passed to the sweep."""

    k_values_effective: List[int]
    """Post-cull splat counts actually realised at each K."""

    held_out_psnr_db: List[float]
    """PSNR at masked positions, against the original pre-fill values."""

    train_psnr_db: List[float]
    """PSNR at unmasked positions, against the original values."""

    held_out_mse: List[float]
    """MSE at masked positions, against the original values."""

    full_psnr_db: List[float]
    """PSNR over the whole volume against the original — for cross-run comparison."""

    full_ssim: List[float]
    """SSIM over the whole volume against the original."""

    held_out_peak: HeldOutPeak
    """The recommended K* and curve type."""

    noise_floor: NoiseFloor
    """Ensemble noise-floor estimate for the input volume."""

    fit_times_seconds: List[float]
    """Wall-clock time per fit, in seconds."""

    splat_paths: Optional[List[str]]
    """Per-K ``.gsplats.zarr`` paths when ``--keep-fits`` is set; else ``None``."""

    mask_seed: int
    mask_fraction: float
    donut_radius: int

    fit_config: Dict[str, Any]
    """Fit kwargs that were applied (sans the per-K ``seeds`` value)."""

    volume_shape: List[int]
    volume_dtype: str
    timestamp: str

    def to_json(self, path: Path) -> None:
        """Serialise to JSON. Non-finite floats become ``null``."""

        def _safe(v: Any) -> Any:
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
            if isinstance(v, np.generic):
                return _safe(v.item())
            if isinstance(v, (list, tuple)):
                return [_safe(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _safe(x) for k, x in v.items()}
            return v

        data = asdict(self)
        data = _safe(data)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "CalibrationResult":
        """Load from JSON. ``null`` floats become ``nan``."""
        raw = json.loads(Path(path).read_text())

        def _hydrate_float_list(xs: List[Any]) -> List[float]:
            return [float("nan") if x is None else float(x) for x in xs]

        peak_raw = raw["held_out_peak"]
        peak = HeldOutPeak(
            k_star=int(peak_raw["k_star"]),
            type=peak_raw["type"],
            confidence_db=float(peak_raw["confidence_db"]),
        )
        nf_raw = raw["noise_floor"]
        nf = NoiseFloor(
            sigma_hat=float(nf_raw["sigma_hat"]) if nf_raw["sigma_hat"] is not None else float("nan"),
            sigma_laplacian=float(nf_raw["sigma_laplacian"]) if nf_raw["sigma_laplacian"] is not None else float("nan"),
            sigma_haar=float(nf_raw["sigma_haar"]) if nf_raw["sigma_haar"] is not None else float("nan"),
            sigma_background=float(nf_raw["sigma_background"]) if nf_raw["sigma_background"] is not None else float("nan"),
            psnr_max_db=float(nf_raw["psnr_max_db"]) if nf_raw["psnr_max_db"] is not None else float("inf"),
        )
        return cls(
            k_values_requested=[int(x) for x in raw["k_values_requested"]],
            k_values_effective=[int(x) for x in raw["k_values_effective"]],
            held_out_psnr_db=_hydrate_float_list(raw["held_out_psnr_db"]),
            train_psnr_db=_hydrate_float_list(raw["train_psnr_db"]),
            held_out_mse=_hydrate_float_list(raw["held_out_mse"]),
            full_psnr_db=_hydrate_float_list(raw["full_psnr_db"]),
            full_ssim=_hydrate_float_list(raw["full_ssim"]),
            held_out_peak=peak,
            noise_floor=nf,
            fit_times_seconds=_hydrate_float_list(raw["fit_times_seconds"]),
            splat_paths=raw.get("splat_paths"),
            mask_seed=int(raw["mask_seed"]),
            mask_fraction=float(raw["mask_fraction"]),
            donut_radius=int(raw["donut_radius"]),
            fit_config=dict(raw.get("fit_config", {})),
            volume_shape=[int(x) for x in raw["volume_shape"]],
            volume_dtype=str(raw["volume_dtype"]),
            timestamp=str(raw["timestamp"]),
        )


# =============================================================================
# Top-level driver
# =============================================================================


ProgressCallback = Callable[[int, int, str], None]
"""``(index, total, message)`` callback emitted before/after each fit."""


def calibrate(
    V: np.ndarray,
    k_grid: Sequence[int],
    *,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    mask_seed: int = 42,
    mask_fraction: float = 0.05,
    donut_radius: int = 1,
    keep_fits: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> CalibrationResult:
    """Run a blind-spot CV sweep over ``k_grid`` on volume ``V``.

    Pipeline (one execution per call):

    1. Generate a deterministic CV mask, donut-fill ``V`` to make
       ``V_filled``.
    2. Fit a Gaussian-splat model with ``fit_gaussian_splats`` at each
       ``K`` in ``k_grid``, using ``V_filled`` as the target. ``fit_kwargs``
       are forwarded verbatim except ``seeds`` (overridden per-K).
    3. Render each fit back to volume; compute held-out / train / full
       PSNR (and full SSIM) against the original ``V``.
    4. Estimate the noise floor on ``V``.
    5. Detect K* via :func:`find_k_star`.

    Parameters
    ----------
    V : np.ndarray
        Input volume (``ndim >= 2``). Will be passed verbatim to the
        fitter, which handles its own normalisation.
    k_grid : sequence of int
        Splat counts to evaluate.
    fit_kwargs : dict, optional
        Forwarded to ``fit_gaussian_splats`` (preset / config / device /
        cull_retention / verbose / ...). The ``seeds`` key is always
        overridden per-K.
    mask_seed, mask_fraction, donut_radius
        Mask construction parameters; the defaults match the manuscript.
    keep_fits : Path, optional
        Directory to persist the per-K ``.gsplats.zarr`` outputs. If
        ``None``, the fitted splats are not saved (memory only during
        the run).
    progress_callback : callable, optional
        Invoked as ``(i, n, msg)`` before each fit and after metrics.

    Returns
    -------
    CalibrationResult
    """
    # Lazy imports — keep module-load cheap for tests that only touch utilities
    import torch

    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.metrics import compute_psnr, compute_ssim
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    if V.ndim < 2:
        raise ValueError(f"V must be at least 2D, got shape {V.shape}")
    if len(k_grid) < 1:
        raise ValueError("k_grid must contain at least one K value")

    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.pop("seeds", None)
    # cv runs many fits — keep their internal logging quiet by default
    fit_kwargs.setdefault("verbose", False)

    if keep_fits is not None:
        keep_fits = Path(keep_fits)
        keep_fits.mkdir(parents=True, exist_ok=True)

    # 1. Mask + donut fill
    mask = cv_mask(V.shape, fraction=mask_fraction, seed=mask_seed)
    V_filled = donut_median_fill(V, mask, radius=donut_radius)

    # Pre-compute originals on torch for in-loop PSNR/SSIM (kept on CPU to
    # avoid VRAM contention with the fitter; the metric calls are O(volume)
    # and dominated by the rendered tensor's location, not this one).
    V_orig_t = torch.from_numpy(V.astype(np.float32, copy=False))
    data_range = float(V.max() - V.min())
    if data_range == 0.0:
        data_range = 1.0
    mask_t = torch.from_numpy(mask)

    k_values_eff: List[int] = []
    held_psnr: List[float] = []
    train_psnr: List[float] = []
    held_mse: List[float] = []
    full_psnr: List[float] = []
    full_ssim: List[float] = []
    fit_times: List[float] = []
    splat_paths: List[str] = []

    # 2-3. Fit at each K
    for i, K in enumerate(k_grid):
        if progress_callback is not None:
            progress_callback(i, len(k_grid), f"fit K={K}")
        t0 = time.perf_counter()
        splats = fit_gaussian_splats(V_filled, seeds=int(K), **fit_kwargs)
        elapsed = time.perf_counter() - t0
        fit_times.append(elapsed)

        # Render on the same device the splats live on (or CPU as a fallback)
        device_pref = fit_kwargs.get("device")
        with torch.no_grad():
            rendered = render_to_volume_tensor(
                splats, shape=V.shape, device=device_pref
            )
            ref_t = V_orig_t.to(rendered.device)
            mask_dev = mask_t.to(rendered.device)

            held_pred = rendered[mask_dev]
            held_true = ref_t[mask_dev]
            train_pred = rendered[~mask_dev]
            train_true = ref_t[~mask_dev]

            held_mse_val = float(torch.mean((held_pred - held_true) ** 2).item())
            held_mse.append(held_mse_val)
            held_psnr_val = (
                float("inf")
                if held_mse_val == 0.0
                else float(10.0 * math.log10(data_range**2 / held_mse_val))
            )
            held_psnr.append(held_psnr_val)

            train_mse = float(torch.mean((train_pred - train_true) ** 2).item())
            train_psnr_val = (
                float("inf")
                if train_mse == 0.0
                else float(10.0 * math.log10(data_range**2 / train_mse))
            )
            train_psnr.append(train_psnr_val)

            full_psnr_val = compute_psnr(rendered, ref_t, data_range=data_range)
            full_psnr.append(float(full_psnr_val))
            full_ssim_val = compute_ssim(rendered, ref_t, data_range=data_range)
            full_ssim.append(float(full_ssim_val))

            del rendered, ref_t, mask_dev

        eff_k = int(splats.n_splats)
        k_values_eff.append(eff_k)

        if keep_fits is not None:
            out_path = keep_fits / f"k{int(K):08d}.gsplats.zarr"
            # The fitter's own metrics (psnr_db, ssim, time_seconds, ...) ride
            # along via splats.stats; calibration-specific fields would be
            # dropped by GSplatData.save()'s fitting-info whitelist anyway, so
            # we keep them only in the global cal.json (which references this
            # file via splat_paths).
            splats.save(out_path, include_fitting_info=True)
            splat_paths.append(str(out_path))

        if progress_callback is not None:
            progress_callback(
                i,
                len(k_grid),
                f"K={K} eff={eff_k} train={train_psnr_val:.2f}dB held={held_psnr_val:.2f}dB t={elapsed:.1f}s",
            )

        del splats
        if device_pref and device_pref.startswith("cuda"):
            torch.cuda.empty_cache()

    # 4. Noise floor (on the original volume; needs [0, 1] for the PSNR ceiling
    # to be meaningful — the manuscript and fit_gaussian_splats both work in
    # that range, so we pre-normalise)
    Vn = V.astype(np.float32, copy=False)
    vmin = float(Vn.min())
    vmax = float(Vn.max())
    if vmax > vmin:
        Vn_norm = (Vn - vmin) / (vmax - vmin)
    else:
        Vn_norm = Vn - vmin
    noise_floor = estimate_noise_floor(Vn_norm)

    # 5. Peak detection
    peak = find_k_star(list(k_grid), held_psnr)

    return CalibrationResult(
        k_values_requested=[int(k) for k in k_grid],
        k_values_effective=k_values_eff,
        held_out_psnr_db=held_psnr,
        train_psnr_db=train_psnr,
        held_out_mse=held_mse,
        full_psnr_db=full_psnr,
        full_ssim=full_ssim,
        held_out_peak=peak,
        noise_floor=noise_floor,
        fit_times_seconds=fit_times,
        splat_paths=splat_paths if keep_fits is not None else None,
        mask_seed=mask_seed,
        mask_fraction=mask_fraction,
        donut_radius=donut_radius,
        fit_config={k: v for k, v in fit_kwargs.items() if _json_safe(v)},
        volume_shape=list(V.shape),
        volume_dtype=str(V.dtype),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _json_safe(v: Any) -> bool:
    """Return True iff ``v`` survives ``json.dumps`` (used to filter fit_config)."""
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    "CalibrationResult",
    "HeldOutPeak",
    "NoiseFloor",
    "ProgressCallback",
    "build_k_grid",
    "calibrate",
    "cv_mask",
    "donut_median_fill",
    "estimate_noise_floor",
    "find_k_star",
    "held_out_psnr",
]
