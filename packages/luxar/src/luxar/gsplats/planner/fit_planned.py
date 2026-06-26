"""Fit a volume according to a :class:`FitPlan` and merge the result.

Each box is fit on a halo-padded crop (so the optimiser has neighbour context),
then only the splats whose centre lands in the *core* box are kept (no
double-counting, no gaps). This is the validated, seam-free approach from the
investigation — far simpler than apodized blending and it held up visually.

This is a sibling of ``fit_tiled`` that consumes the planner's content-balanced,
budgeted boxes instead of a uniform grid. It does not touch ``fit_tiled`` /
``fit_tiled_parallel`` (it only fits each box via ``fit_gaussian_splats``).
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Optional

import numpy as np

from .spec import FitPlan

ProgressCallback = Callable[[int, int, str], None]


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
    """
    import torch

    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.gsplat_data import GSplatData

    V = np.asarray(volume, dtype=np.float32)
    if V.ndim != 3:
        raise ValueError(f"fit_planned expects a 3-D volume, got shape {V.shape}")
    Z, Y, X = V.shape
    pad = int(plan.overlap)
    fit_kwargs.setdefault("cull_retention", 0.999)
    fit_kwargs.setdefault("verbose", False)
    fit_kwargs["device"] = device

    cs: list[np.ndarray] = []
    amps: list[np.ndarray] = []
    chols: list[np.ndarray] = []
    n = len(plan.boxes)
    for i, b in enumerate(plan.boxes):
        if b.budget <= 0:
            continue
        z0, z1, y0, y1, x0, x1 = b.box
        pz0, pz1 = max(0, z0 - pad), min(Z, z1 + pad)
        py0, py1 = max(0, y0 - pad), min(Y, y1 + pad)
        px0, px1 = max(0, x0 - pad), min(X, x1 + pad)
        sub = V[pz0:pz1, py0:py1, px0:px1]
        # scale the budget up for the padded volume so the core keeps ~b.budget
        # splats (capped at 4x to bound cost on thin slabs)
        core_vox = max(1, b.voxels)
        budget = int(min(b.budget * sub.size / core_vox, b.budget * 4))
        if budget <= 0:
            continue
        if progress_callback is not None:
            progress_callback(i, n, f"box {i + 1}/{n} budget={budget}")
        gd = fit_gaussian_splats(sub, seeds=budget, **fit_kwargs)
        c = np.asarray(gd.centers, dtype=np.float32) + np.array(
            [pz0, py0, px0], np.float32
        )
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
        cs.append(c[keep])
        amps.append(a[keep])
        chols.append(k[keep])
        if verbose:
            from arbol import aprint

            aprint(f"  box {i + 1}/{n}: kept {int(keep.sum()):,} splats")
        del gd
        gc.collect()
        if device and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not cs:
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
            "volume_shape": [Z, Y, X],
        },
    )
    return merged


__all__ = ["fit_planned"]
