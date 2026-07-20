"""Per-splat opacity (color alpha channel) conversions.

The optional 4th color column stores per-splat opacity ``a ∈ [0, 1]``. In the
volumetric blending mode it acts through optical depth ``w = −ln(1 − a)``
(so a splat's peak rendered alpha reproduces ``a`` — the 3DGS-faithful
mapping), and optical depth — not opacity — is the linearly composing
quantity used when splats are merged (LOD reduction).

``ALPHA_CLAMP`` bounds ``a`` away from 1 (infinite optical depth); the value
mirrors the viewer shaders' clamp exactly so Python-side aggregation and
GPU-side rendering agree.
"""

from __future__ import annotations

import numpy as np

# a = 1 means "fully opaque" but maps to w = ∞; clamp one uint9-ish step
# below 1 (w ≈ 6.24, transmittance ≈ 0.2%). Keep in sync with the viewer's
# gsplat shaders (GLSL + TSL), which use the same literal.
ALPHA_CLAMP: float = 1.0 - 1.0 / 512.0


def alpha_to_optical_depth(alpha: np.ndarray) -> np.ndarray:
    """Map opacity ``a`` to optical depth ``w = −ln(1 − a)`` (clamped)."""
    a = np.clip(alpha, 0.0, ALPHA_CLAMP)
    depth: np.ndarray = -np.log1p(-a)
    return depth


def optical_depth_to_alpha(depth: np.ndarray) -> np.ndarray:
    """Map optical depth ``w`` back to opacity ``a = 1 − e^(−w)``."""
    return -np.expm1(-np.asarray(depth, dtype=np.float64)).astype(np.float32)


def effective_amplitudes(data: object) -> np.ndarray:
    """Per-splat emission amplitude including the color alpha factor.

    Every blending mode multiplies a splat's contribution by its alpha
    (per-splat opacity), so ``A·a`` — not raw ``A`` — is the splat's rendered
    mass. Ordering/culling by raw amplitude would misrank imported classical
    splats, whose importer stores amplitude = 1 and carries all per-splat
    weight in alpha. Returns the raw amplitudes unchanged when the data has
    no RGBA colors.

    Args:
        data: Anything with ``amplitudes`` and optional ``colors`` arrays
            (``GSplatData`` / ``AdditiveSubLOD``).
    """
    amps: np.ndarray = np.asarray(data.amplitudes)  # type: ignore[attr-defined]
    colors = getattr(data, "colors", None)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim == 2 and colors.shape[1] == 4:
            weighted: np.ndarray = amps * colors[:, 3].astype(amps.dtype, copy=False)
            return weighted
    return amps
