"""Content-balanced, orientation-aware BSP over a content-density field.

Recursively splits the volume so every leaf holds ~equal feature content (≤
``target_features``), bounding leaf size on both ends:

* ``min_leaf`` — never finer than the baseline tile (caps the overlap fraction
  and avoids tile-tile seams),
* ``max_leaf`` — force-split larger leaves (keeps each fit effective; very large
  sparse leaves spread their few features too thin).

Orientation at each node = the axis of greatest *signal-weighted* spread (×
physical extent), cut at the weighted median — balances content AND keeps leaves
compact. Per-leaf budgets come from the calibration's transferable
:class:`~luxar.gsplats.calibration.SplatDensity` (sub-linear ``K ∝ features^α``,
capped at saturation). This is the validated planner from the investigation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from luxar.gsplats.calibration import SplatDensity

from .content_scan import ContentField, scan_content
from .spec import FitPlan, PlanBox


def _density_from(obj: Union[SplatDensity, dict]) -> SplatDensity:
    if isinstance(obj, SplatDensity):
        return obj
    return SplatDensity(
        feature_method=str(obj.get("feature_method", "peaks")),
        n_features_reference=int(obj["n_features_reference"]),
        k_star_reference=int(obj["k_star_reference"]),
        saturation_exponent=float(obj.get("saturation_exponent", 0.44)),
        saturation_cap=int(obj["saturation_cap"]),
        splats_per_feature=float(obj.get("splats_per_feature", float("nan"))),
    )


def plan_partition(
    field: ContentField,
    density: Union[SplatDensity, dict],
    *,
    target_features: Optional[int] = None,
    min_leaf: int = 256,
    max_leaf: int = 512,
    overlap: int = 32,
) -> FitPlan:
    """Build a content-balanced, size-bounded BSP + per-leaf budgets.

    ``target_features`` defaults to the density's reference feature count, so each
    leaf holds ~the calibrated reference content and therefore gets ~the
    calibrated ``k_star`` budget — the cleanest "calibrate at the scale you fit
    at" coupling.
    """
    dens = _density_from(density)
    if target_features is None:
        target_features = max(1, int(dens.n_features_reference))
    Z, Y, X = field.shape
    c = field.cell

    leaves: List[Tuple[int, int, int, int, int, int]] = []

    def _split(box: Tuple[int, int, int, int, int, int], depth: int) -> None:
        z0, z1, y0, y1, x0, x1 = box
        w = field.box_weight(*box)
        dims = [z1 - z0, y1 - y0, x1 - x0]
        must = max(dims) > max_leaf
        can = max(dims) >= 2 * min_leaf
        if (not must and w <= target_features) or not can or depth > 24:
            leaves.append(box)
            return
        cand = [a for a in range(3) if dims[a] >= 2 * min_leaf]
        if must:
            over = [a for a in cand if dims[a] > max_leaf]
            if over:
                cand = over
        best: Optional[Tuple[float, int, np.ndarray]] = None
        for ax in cand:
            m = field.marginal(box, ax).astype(float)
            pos = np.arange(m.size)
            if m.sum() > 0:
                mean = (pos * m).sum() / m.sum()
                spread = float(np.sqrt(((pos - mean) ** 2 * m).sum() / m.sum()))
            else:  # near-empty: fall back to geometric extent
                spread = dims[ax] / c / 3.46
            score = spread * dims[ax]
            if best is None or score > best[0]:
                best = (score, ax, m)
        if best is None:
            leaves.append(box)
            return
        _, ax, m = best
        if m.sum() <= 0:  # uniform fallback -> geometric midpoint
            m = np.ones_like(m)
        cdf = np.cumsum(m)
        cell_idx = int(np.searchsorted(cdf, cdf[-1] / 2.0))
        origin = [z0, y0, x0][ax]
        cut = origin + int(
            np.clip((cell_idx + 1) * c, min_leaf, dims[ax] - min_leaf)
        )
        lo, hi = list(box), list(box)
        lo[2 * ax + 1] = cut
        hi[2 * ax] = cut
        _split(tuple(lo), depth + 1)  # type: ignore[arg-type]
        _split(tuple(hi), depth + 1)  # type: ignore[arg-type]

    _split((0, Z, 0, Y, 0, X), 0)

    boxes = [
        PlanBox(
            box=[int(v) for v in bx],
            n_features=int(round(field.box_weight(*bx))),
            budget=int(dens.predict_k(int(round(field.box_weight(*bx))))),
        )
        for bx in leaves
    ]
    return FitPlan(
        volume_shape=[Z, Y, X],
        boxes=boxes,
        overlap=int(overlap),
        feature_method=field.method,
        min_leaf=int(min_leaf),
        max_leaf=int(max_leaf),
        density=_as_dict(dens),
        meta={"target_features": int(target_features), "cell": int(c)},
    )


def plan_volume(
    volume: np.ndarray,
    density: Union[SplatDensity, dict],
    *,
    feature_method: str = "peaks",
    cell: int = 16,
    target_features: Optional[int] = None,
    min_leaf: int = 256,
    max_leaf: int = 512,
    overlap: int = 32,
    device: Optional[str] = None,
    threshold_abs: Optional[float] = None,
) -> FitPlan:
    """Convenience: scan ``volume`` then plan. Returns a :class:`FitPlan`.

    ``threshold_abs`` defaults to the density's recorded ``feature_threshold`` so
    the scan counts features on the same absolute scale as the calibration's
    reference — the only way the per-box budgets are correctly scaled.
    """
    dens = _density_from(density)
    if threshold_abs is None and getattr(dens, "feature_threshold", 0.0) > 0:
        threshold_abs = dens.feature_threshold
    field = scan_content(
        volume,
        cell=cell,
        method=feature_method,
        threshold_abs=threshold_abs,
        device=device,
    )
    return plan_partition(
        field,
        dens,
        target_features=target_features,
        min_leaf=min_leaf,
        max_leaf=max_leaf,
        overlap=overlap,
    )


def _as_dict(d: SplatDensity) -> dict:
    from dataclasses import asdict

    return asdict(d)


__all__ = ["plan_partition", "plan_volume"]
