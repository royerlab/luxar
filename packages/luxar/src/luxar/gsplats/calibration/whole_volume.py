"""A whole-volume ``fit --seeds`` budget from a calibration run on a crop.

``cal --region`` / ``cal --auto-region`` measure K* on a crop, so K* is a CROP
budget: passing it to ``fit --seeds`` (a whole-volume budget) under-seeds the
volume. :func:`whole_volume_seeds` restates it for the whole input with the same
density transfer the content planner applies (:meth:`SplatDensity.predict_k`):
the volume is cut into crop-sized boxes, each box's features are counted at the
calibrated absolute level, each box gets ``K*·(n_box/n_crop)^alpha`` (capped at
the measured saturation), and the budgets are summed.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .content import count_features
from .curve_analysis import SplatDensity

#: ``cal --region`` never loaded the whole input; it is read once to size the
#: whole-volume budget only up to this many elements (2 GB as float32).
#: Above it the suggestion is skipped with a message, never risked.
MAX_WHOLE_INPUT_VOXELS = 512 * 1024**2


def input_element_count(path: Any, array_key: Optional[str] = None) -> Optional[int]:
    """Elements in the input's (first) array, from its header only; ``None`` if
    the format cannot be probed without reading it.

    ``.npy`` (memory-mapped header), TIFF (first series) and zarr stores (the
    array at ``array_key``, or a root array). Counts EVERY axis, channels and
    time included, so it errs on the large side.
    """
    from pathlib import Path

    p = Path(path)
    name = p.name.lower()
    try:
        if name.endswith(".npy"):
            return int(np.prod(np.load(p, mmap_mode="r").shape))
        if name.endswith((".tif", ".tiff")):
            import tifffile

            with tifffile.TiffFile(p) as tif:
                return int(np.prod(tif.series[0].shape))
        if p.is_dir():
            import zarr

            node = zarr.open(str(p), mode="r")
            if array_key:
                node = node[array_key]
            shape = getattr(node, "shape", None)
            return int(np.prod(shape)) if shape is not None else None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    return None


def whole_volume_seeds(
    V_full: np.ndarray,
    *,
    crop_shape: Sequence[int],
    k_star_seeds: int,
    splat_density: Dict[str, Any],
    floor_subtracted: Optional[float],
) -> Optional[Dict[str, Any]]:
    """The summed per-box budget for ``V_full``; ``None`` if it cannot be sized.

    ``k_star_seeds`` is the crop K* to transfer — ``gsplat cal`` passes the
    density's effective (post-cull) ``k_star_reference``, the count the content
    planner transfers; ``splat_density`` is the crop calibration's
    ``CalibrationResult.splat_density``
    and ``floor_subtracted`` the floor ``cal`` took off the crop (taken off the
    whole volume too, so features are counted on the same scale).
    """
    density = SplatDensity(**splat_density)
    if not math.isfinite(density.feature_threshold) or V_full.ndim != len(crop_shape):
        return None
    density = replace(
        density,
        k_star_reference=int(k_star_seeds),
        saturation_cap=max(int(density.saturation_cap), int(k_star_seeds)),
    )
    V = np.asarray(V_full, dtype=np.float32)
    if floor_subtracted:
        V = np.clip(V - float(floor_subtracted), 0.0, None)
    steps = [max(1, int(c)) for c in crop_shape]
    starts = [range(0, int(s), st) for s, st in zip(V.shape, steps, strict=True)]
    total = 0
    n_boxes = 0
    n_features = 0
    for origin in itertools.product(*starts):
        box = V[tuple(slice(o, o + st) for o, st in zip(origin, steps, strict=True))]
        n = count_features(
            box, method=density.feature_method, threshold_abs=density.feature_threshold
        )
        n_features += int(n)
        n_boxes += 1
        if n > 0:
            total += density.predict_k(int(n))
    return {
        "seeds": int(total),
        "n_boxes": n_boxes,
        "box_shape": steps,
        "n_features_whole": n_features,
        "n_features_crop": int(splat_density.get("n_features_reference", 0)),
        "k_star_crop_seeds": int(k_star_seeds),
        "whole_shape": [int(s) for s in V.shape],
        "rule": (
            "sum over crop-sized boxes of K*·(n_box/n_crop)^alpha, capped at the "
            "measured saturation (the content planner's density transfer)"
        ),
    }
