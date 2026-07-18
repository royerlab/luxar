"""sRGB ↔ linear color-space conversion for classical-splat interop.

Classical 3D Gaussian Splatting stores DC color as a *display-referred* value
(``0.5 + SH_C0·f_dc`` is what reference viewers — SuperSplat, PlayCanvas,
antimatter15 — put straight on an sRGB display, applying no further transfer
function). Luxar's viewer, by contrast, treats every stored per-splat color as
*linear light* and applies the sRGB OETF once, as the final output step. Feeding
a display-referred (sRGB) value into that linear pipeline double-encodes it and
washes the scene toward white.

So the import boundary converts classical DC color sRGB → linear (and the INRIA
exporter inverts it, linear → sRGB), making imported scenes render with the
same colors a reference viewer shows. The transfer functions are the exact IEC
61966-2-1 sRGB piecewise curve.
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB (display) → linear light. Input/return in ``[0, 1]``, elementwise."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear light → sRGB (display). Input/return in ``[0, 1]``, elementwise."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)
    return srgb.astype(np.float32)
