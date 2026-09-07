"""Content / feature estimation — the shared metric for density + planner.

Feature content (local-maxima count by default) is the empirical predictor of
how many splats a region needs. These helpers also pick a content-rich crop to
calibrate at the fitting scale, and expose the exact absolute threshold the
detectors count at so the cal→planner contract stays on one scale.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


def _otsu_threshold(V: np.ndarray) -> float:
    """Dependency-free Otsu threshold, subsampled for bounded histogram cost."""
    v = np.asarray(V)
    flat = v.reshape(-1)
    if flat.size > 5_000_000:  # cap histogram sample on gigavoxel volumes
        flat = flat[:: max(1, flat.size // 5_000_000)]
    import torch

    from luxar.gsplats.metrics import otsu_threshold

    return otsu_threshold(torch.from_numpy(flat.astype(np.float32, copy=False)))


def foreground_mask_otsu(V: np.ndarray) -> np.ndarray:
    """Boolean foreground mask via Otsu's threshold (``V > thr``)."""
    v = np.asarray(V)
    return v > _otsu_threshold(v)


def foreground_mask_otsu_smoothed(V: np.ndarray) -> Tuple[np.ndarray, float]:
    """Foreground mask for weighted calibration scoring.

    Applies one light separable tent blur, then computes Otsu on the blurred
    field. The input is expected floor-subtracted; the returned threshold is
    on the smoothed scale and is recorded with the metric for reproducibility.
    """
    from luxar.gsplats.seeds.utils import soft_blur_nd

    smoothed = soft_blur_nd(np.asarray(V, dtype=np.float32))
    threshold = _otsu_threshold(smoothed)
    return smoothed > threshold, threshold


def count_features(
    V: np.ndarray,
    method: str = "peaks",
    *,
    threshold_abs: Optional[float] = None,
    **kwargs: Any,
) -> int:
    """Estimate the feature content of a volume — the predictor of splat need.

    The empirical investigation found local-maxima count (``peaks``) the best
    predictor of how many splats a region needs (better than intensity-sum or
    foreground-count), so it is the default. ``edges`` (summed Sobel gradient
    magnitude, thresholded) suits non-punctate structure (filaments,
    membranes); ``intensity`` is a robust foreground-voxel count.

    Parameters
    ----------
    V : np.ndarray
        Input volume.
    method : {"peaks", "edges", "intensity"}, default="peaks"
        Feature estimator. Pluggable so non-nuclear data can choose ``edges``.
    threshold_abs : float, optional
        Absolute detection level (the value :func:`feature_threshold` returns).
        When given, every method counts at this *shared* level instead of a
        per-volume relative one — so counts on different crops compose (required
        when ranking sliding windows; a per-crop relative threshold lets a
        faint-noise window out-score a real one). ``peaks``/``edges`` threshold
        the blurred / gradient field at it; ``intensity`` counts ``V > thr``
        (strict ``>``, matching :func:`foreground_mask_otsu`).
    **kwargs
        Forwarded to the underlying estimator (e.g. ``radius``,
        ``threshold_rel`` for ``peaks``).

    Returns
    -------
    int
        A non-negative feature count.
    """
    v = np.asarray(V, dtype=np.float32)
    if method == "peaks":
        from luxar.gsplats.seeds.utils import count_local_maxima

        return int(count_local_maxima(v, threshold_abs=threshold_abs, **kwargs))
    if method == "edges":
        from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

        mag = np.asarray(_compute_nd_sobel_magnitude(v))
        if threshold_abs is not None:
            thr = float(threshold_abs)
        else:
            m = float(mag.max())
            if m <= 0.0:
                return 0
            thr = float(kwargs.get("threshold_rel", 0.1)) * m
        return int(np.count_nonzero(mag >= thr))
    if method == "intensity":
        if threshold_abs is not None:
            return int(np.count_nonzero(v > float(threshold_abs)))
        return int(np.count_nonzero(foreground_mask_otsu(v)))
    raise ValueError(
        f"unknown feature method {method!r}; use 'peaks', 'edges', or 'intensity'"
    )


def feature_threshold(
    V: np.ndarray, method: str = "peaks", threshold_rel: float = 0.1
) -> float:
    """The *exact absolute intensity level* :func:`count_features` thresholds at.

    Single source of truth for the cal→planner contract: the calibration records
    this so the planner's ``scan_content`` counts on the identical scale (the
    detectors threshold relative to a *blurred* / gradient / Otsu level, NOT the
    raw max, so a naive ``0.1*max`` drifts — badly with hot outliers).

    * ``peaks``     → ``threshold_rel * max(soft_blur(V))`` (matches count_local_maxima)
    * ``edges``     → ``threshold_rel * max(|∇V|)``
    * ``intensity`` → the Otsu cut
    """
    v = np.asarray(V, dtype=np.float32)
    if v.size == 0:
        return 0.0
    if method == "peaks":
        from luxar.gsplats.seeds.utils import soft_blur_nd

        return float(threshold_rel * float(np.asarray(soft_blur_nd(v)).max()))
    if method == "edges":
        from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

        return float(
            threshold_rel * float(np.asarray(_compute_nd_sobel_magnitude(v)).max())
        )
    if method == "intensity":
        return _otsu_threshold(v)
    raise ValueError(
        f"unknown feature method {method!r}; use 'peaks', 'edges', or 'intensity'"
    )


def _robust_feature_level(
    V: np.ndarray,
    method: str = "peaks",
    threshold_rel: float = 0.1,
    q: float = 99.9,
    **_: Any,
) -> float:
    """Outlier-robust analogue of :func:`feature_threshold` for window ranking.

    Identical in spirit to :func:`feature_threshold` but bases the level on the
    ``q``-th percentile of the (method-transformed) field instead of its **max**,
    so a handful of hot voxels (dead/stuck pixels, cosmic-ray hits) cannot set the
    global signal level above genuine content — which would otherwise gate every
    real window to zero and let the lone-outlier window win
    (:func:`select_calibration_region`). Used ONLY for ranking; the cal→planner
    contract still records the max-based :func:`feature_threshold` on the chosen
    crop, so this does not perturb the transferred density.
    """
    v = np.asarray(V, dtype=np.float32)
    if v.size == 0:
        return 0.0

    def _subsample(field: np.ndarray) -> np.ndarray:
        flat = field.reshape(-1)
        if flat.size > 5_000_000:  # bound the percentile cost on gigavoxel volumes
            flat = flat[:: max(1, flat.size // 5_000_000)]
        return flat

    if method == "peaks":
        from luxar.gsplats.seeds.utils import soft_blur_nd

        field = _subsample(np.asarray(soft_blur_nd(v)))
        return float(threshold_rel * float(np.percentile(field, q)))
    if method == "edges":
        from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

        field = _subsample(np.asarray(_compute_nd_sobel_magnitude(v)))
        return float(threshold_rel * float(np.percentile(field, q)))
    if method == "intensity":
        flat = _subsample(v)
        hi = float(np.percentile(flat, q))
        return _otsu_threshold(np.minimum(flat, hi))  # Otsu on winsorised values
    raise ValueError(
        f"unknown feature method {method!r}; use 'peaks', 'edges', or 'intensity'"
    )


@dataclass
class RegionSelection:
    """Provenance of an auto-selected calibration sub-region."""

    origin: List[int]
    """Top-left corner of the crop in the original volume's coordinates."""
    size: List[int]
    """Crop shape actually used (clamped per-axis to the volume)."""
    strategy: str
    """``densest`` | ``median`` | ``whole``."""
    n_features: int
    """Feature count inside the chosen crop."""
    score: float
    """Feature *density* (features / voxel) used to rank candidate windows."""


def select_calibration_region(
    V: np.ndarray,
    region_size: int = 256,
    strategy: str = "densest",
    feature: str = "peaks",
    **feature_kwargs: Any,
) -> Tuple[np.ndarray, RegionSelection]:
    """Pick a content-rich sub-region to calibrate at the *fitting* scale.

    The manuscript calibrates on crops ≤~20 M voxels; on a large sparse volume
    the held-out metric is background-dominated and the absolute K wrong-scale.
    This slides non-overlapping ``region_size`` windows, scores each by feature
    density (:func:`count_features`), and returns the chosen crop + provenance.

    ``strategy="densest"`` picks the highest-density window (worst case for
    splat budget); ``"median"`` picks the median-density window (representative,
    avoids the single brightest outlier). For volumes no larger than
    ``region_size`` on every axis the whole volume is returned
    (``strategy="whole"``).
    """
    v = np.asarray(V)
    if v.ndim != 3:
        raise ValueError(
            f"select_calibration_region expects a 3-D volume, got shape {v.shape}"
        )
    shape = v.shape
    size = [int(min(region_size, s)) for s in shape]

    def _origins(n: int, t: int) -> List[int]:
        os_ = list(range(0, n - t + 1, t))
        if not os_ or os_[-1] != n - t:
            os_.append(max(0, n - t))
        return os_

    # whole-volume short-circuit (no cross-window ranking needed)
    if all(size[d] == shape[d] for d in range(v.ndim)):
        n = count_features(v, method=feature, **feature_kwargs)
        return v, RegionSelection(
            origin=[0] * v.ndim,
            size=list(size),
            strategy="whole",
            n_features=int(n),
            score=float(n) / max(1, v.size),
        )

    # One *shared, outlier-robust* absolute level for all windows. Counting each
    # window at this fixed level (not its own relative max) makes counts compose
    # across windows — and using a high percentile rather than the raw max means a
    # lone hot voxel can't raise the bar above genuine signal and gate every real
    # window to zero (which previously let the single-outlier window win).
    global_thr = _robust_feature_level(v, feature, **feature_kwargs)

    axis_origins = [_origins(shape[d], size[d]) for d in range(v.ndim)]
    # Each candidate ranked primarily by feature count, ties broken by total
    # intensity (so equal-count windows prefer the one with more signal, not a
    # faint blob tail). `score` reported = feature density (features / voxel).
    candidates: List[Tuple[int, float, List[int], int]] = []
    for origin in itertools.product(*axis_origins):
        sl = tuple(slice(origin[d], origin[d] + size[d]) for d in range(v.ndim))
        crop = v[sl]
        # Count at the shared absolute level: a background window scores 0
        # naturally (no voxel reaches the level), so no separate gate is needed —
        # and the previous gate compared a *raw* crop max against a *blurred*-field
        # threshold (a units mismatch that mis-gated sharp single-voxel features).
        n = count_features(
            crop, method=feature, threshold_abs=global_thr, **feature_kwargs
        )
        candidates.append((int(n), float(crop.sum()), list(origin), int(n)))

    if not candidates:  # pragma: no cover - defensive
        n = count_features(v, method=feature, **feature_kwargs)
        return v, RegionSelection([0] * v.ndim, list(size), "whole", int(n), 0.0)

    candidates.sort(key=lambda c: (c[0], c[1]))
    if strategy == "densest":
        _, _, best_origin, n = candidates[-1]
    elif strategy == "median":
        _, _, best_origin, n = candidates[len(candidates) // 2]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use 'densest' or 'median'")

    sl = tuple(slice(best_origin[d], best_origin[d] + size[d]) for d in range(v.ndim))
    crop = v[sl]
    return crop, RegionSelection(
        origin=list(best_origin),
        size=list(size),
        strategy=strategy,
        n_features=int(n),
        score=float(n) / max(1, crop.size),
    )
