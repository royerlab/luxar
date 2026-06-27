"""Fit a volume according to a :class:`FitPlan` and merge the result.

Each box is fit on a halo-padded crop (so the optimiser has neighbour context),
then only the splats whose centre lands in the *core* box are kept (no
double-counting, no gaps). This is the validated, seam-free approach from the
investigation — far simpler than apodized blending and it held up visually.

This is a sibling of ``fit_tiled`` that consumes the planner's content-balanced,
budgeted boxes instead of a uniform grid. It does not touch ``fit_tiled`` /
``fit_tiled_parallel`` (it only fits each box via ``fit_gaussian_splats``).

The per-box body lives in :func:`_fit_one_box` so the **sequential** driver here
and the **parallel** subprocess worker (``fit_planned_parallel`` /
``gsplat plan --fit-box``) share one source of truth for budget scaling, the
padded crop, and the keep-core mask — they can never drift.
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .spec import FitPlan, PlanBox

ProgressCallback = Callable[[int, int, str], None]


def _padded_bounds(
    box: PlanBox, overlap: int, shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int, int, int]:
    """Halo-padded crop bounds for ``box`` clamped to the volume ``shape``."""
    Z, Y, X = shape
    z0, z1, y0, y1, x0, x1 = box.box
    pad = int(overlap)
    return (
        max(0, z0 - pad),
        min(Z, z1 + pad),
        max(0, y0 - pad),
        min(Y, y1 + pad),
        max(0, x0 - pad),
        min(X, x1 + pad),
    )


def _scaled_budget(
    box: PlanBox, overlap: int, shape: Tuple[int, int, int], cap: int
) -> int:
    """Per-box seed budget, scaled up for the padded crop and capped.

    The plan budget sizes the *core*; the optimiser fits the padded crop, so we
    scale by the padded/core voxel ratio (capped at 4x to bound cost on thin
    slabs) and at the calibrated saturation ``cap`` so a box never fits past the
    over-saturated regime.
    """
    pz0, pz1, py0, py1, px0, px1 = _padded_bounds(box, overlap, shape)
    sub_size = (pz1 - pz0) * (py1 - py0) * (px1 - px0)
    core_vox = max(1, box.voxels)
    budget = int(min(box.budget * sub_size / core_vox, box.budget * 4))
    if cap > 0:
        budget = min(budget, cap)
    return budget


def _fit_one_box(
    volume: np.ndarray,
    box: PlanBox,
    overlap: int,
    cap: int,
    **fit_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a single plan box and return its core-kept splats in GLOBAL coords.

    Crops a halo-padded region, fits it with the (padded-scaled, capped) budget,
    translates the fitted centres back to global coordinates, and keeps only the
    splats whose centre lands in the half-open core box ``[z0:z1, y0:y1, x0:x1]``
    (so adjacent boxes tile the volume with no gaps and no double-counting).

    Returns ``(centers, amplitudes, cholesky_factors)`` — possibly empty (a box
    with non-positive scaled budget yields 0 splats). Shared by the sequential
    :func:`fit_planned` loop and the parallel per-box worker.
    """
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.utils.trils import tril_size

    V = np.asarray(volume, dtype=np.float32)
    ndim = V.ndim
    pz0, pz1, py0, py1, px0, px1 = _padded_bounds(box, overlap, V.shape)
    z0, z1, y0, y1, x0, x1 = box.box

    empty = (
        np.zeros((0, ndim), np.float32),
        np.zeros((0,), np.float32),
        np.zeros((0, tril_size(ndim)), np.float32),
    )
    budget = _scaled_budget(box, overlap, V.shape, cap)
    if budget <= 0:
        return empty

    sub = V[pz0:pz1, py0:py1, px0:px1]
    gd = fit_gaussian_splats(sub, seeds=budget, **fit_kwargs)
    c = np.asarray(gd.centers, dtype=np.float32) + np.array([pz0, py0, px0], np.float32)
    a = np.asarray(gd.amplitudes, dtype=np.float32)
    k = np.asarray(gd.cholesky_factors, dtype=np.float32)
    keep = (
        (c[:, 0] >= z0)
        & (c[:, 0] < z1)
        & (c[:, 1] >= y0)
        & (c[:, 1] < y1)
        & (c[:, 2] >= x0)
        & (c[:, 2] < x1)
    )
    out = (c[keep], a[keep], k[keep])

    # free the per-box working set before the next box / before the worker exits
    del gd
    gc.collect()
    device = fit_kwargs.get("device")
    if device and str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    return out


def fit_planned(
    volume: np.ndarray,
    plan: FitPlan,
    *,
    device: Optional[str] = None,
    verbose: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    **fit_kwargs: Any,
) -> "Any":
    """Fit every box in ``plan`` and return the merged :class:`GSplatData`.

    ``fit_kwargs`` are forwarded to ``fit_gaussian_splats`` per box (preset /
    n_iters / loss / cull_retention / ...). Each box's seed budget comes from the
    plan; near-empty boxes (budget 0) are skipped.

    Boxes are fit **sequentially**. For concurrent fitting on one GPU use
    :func:`luxar.gsplats.planner.fit_planned_parallel.fit_planned_parallel`
    (``gsplat plan --fit -j N``), which fits each box in its own subprocess and
    merges the same way.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    V = np.asarray(volume, dtype=np.float32)
    if V.ndim != 3:
        raise ValueError(f"fit_planned expects a 3-D volume, got shape {V.shape}")
    pad = int(plan.overlap)
    fit_kwargs.setdefault("cull_retention", 0.999)
    fit_kwargs.setdefault("verbose", False)
    fit_kwargs["device"] = device

    # Per-tile saturation cap (from the calibration): the halo inflation must not
    # push the fit past the K the calibration measured as over-saturated.
    cap = int(plan.density.get("saturation_cap", 0)) if plan.density else 0

    cs: list[np.ndarray] = []
    amps: list[np.ndarray] = []
    chols: list[np.ndarray] = []
    n = len(plan.boxes)
    for i, b in enumerate(plan.boxes):
        if b.budget <= 0:
            continue
        budget = _scaled_budget(b, pad, V.shape, cap)
        if budget <= 0:
            continue
        if progress_callback is not None:
            progress_callback(i, n, f"box {i + 1}/{n} budget={budget}")
        c, a, k = _fit_one_box(V, b, pad, cap, **fit_kwargs)
        cs.append(c)
        amps.append(a)
        chols.append(k)
        if verbose:
            from arbol import aprint

            aprint(f"  box {i + 1}/{n}: kept {int(c.shape[0]):,} splats")

    if sum(int(c.shape[0]) for c in cs) == 0:
        raise ValueError("fit_planned produced no splats (all boxes empty?)")
    merged = GSplatData(
        centers=np.concatenate(cs).astype(np.float32),
        amplitudes=np.concatenate(amps).astype(np.float32),
        cholesky_factors=np.concatenate(chols).astype(np.float32),
        stats={
            "planned_fit": True,
            "n_boxes": n,
            "n_boxes_fit": len(cs),
            "overlap": pad,
            "volume_shape": list(V.shape),
        },
    )
    return merged


__all__ = ["fit_planned"]
