"""Apply a colormap to a scalar array at authoring time (Python).

The viewer normally maps ``scalars`` → RGB at render time (it reads the
``scalar_data_range`` attr, normalises to [0, 1], and indexes the colormap LUT
in the shader). :func:`scalars_to_colors` replicates that exact normalisation in
Python so producers can bake scalar-driven colours when a downstream consumer
cannot carry per-element scalars — e.g. the gsplat coarse levels of a Points
substitutive LOD ladder (gsplats store colours, not scalars).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from luxar.colormaps.registry import resolve_colormap

__all__ = ["scalars_to_colors"]


def scalars_to_colors(
    scalars: NDArray,
    colormap: Union[str, NDArray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> NDArray[np.float32]:
    """Map a scalar array to per-element RGB via a colormap LUT.

    Replicates the viewer's render-time normalisation so baked colours match what
    a scalar-driven node would display *to within one LUT cell*: normalise
    ``scalars`` to [0, 1] over ``[vmin, vmax]``, clamp, index a ``(256, 3)`` LUT
    with nearest rounding. (The viewer samples the LUT texture with linear
    filtering, so interior colours can differ by up to half a LUT cell — sub-
    perceptual; a constant ``gamma`` on the node is also not baked, see callers.)

    Parameters
    ----------
    scalars : array, shape (N,)
        Scalar values.
    colormap : str or (M, 3) array
        Colormap name (built-in / matplotlib / colorcet) or an explicit LUT,
        resolved via :func:`luxar.colormaps.resolve_colormap`.
    vmin, vmax : float, optional
        Normalisation bounds. Default to ``scalars`` min/max — matching the
        ``scalar_data_range`` the writer records for render-time normalisation.

    Returns
    -------
    (N, 3) float32 in [0, 1]
        Per-element RGB. (Float, not uint8, to avoid a round-trip through the
        downstream integer-colour normalisation.)
    """
    s = np.asarray(scalars, dtype=np.float64).reshape(-1)
    lut = resolve_colormap(colormap).astype(np.float64) / 255.0  # (256, 3) in [0,1]

    # Default bounds from the FINITE data only (matching the writer's
    # scalar_data_range, which is finite) — np.isfinite excludes both NaN and
    # Inf, so neither poisons the range; non-finite scalars below map to LUT[0].
    finite = np.isfinite(s)
    s_finite = s[finite]
    if vmin is None:
        lo = float(s_finite.min()) if s_finite.size else 0.0
    else:
        lo = float(vmin)
    if vmax is None:
        hi = float(s_finite.max()) if s_finite.size else 0.0
    else:
        hi = float(vmax)
    rng = hi - lo
    if rng <= 0.0:
        # Degenerate range (all-equal scalars): the viewer normalises with
        # uScalarScale = 1/max(1e-10, max-min) → t = clamp((s-min)*scale, 0, 1)
        # = 0 → it samples LUT[0]. Match that (NOT the LUT centre) so the baked
        # coarse levels agree with the finest scalar-driven node.
        norm = np.zeros(s.shape, dtype=np.float64)
    else:
        norm = np.clip((s - lo) / rng, 0.0, 1.0)
    # Non-finite scalars (NaN/Inf) → LUT[0] (defined, not garbage).
    norm = np.where(finite, norm, 0.0)

    n_lut = lut.shape[0]
    idx = np.minimum((norm * (n_lut - 1)).round().astype(np.intp), n_lut - 1)
    out: NDArray[np.float32] = np.ascontiguousarray(lut[idx], dtype=np.float32)
    return out
