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
    a scalar-driven node would display: normalise ``scalars`` to [0, 1] over
    ``[vmin, vmax]``, clamp, index a ``(256, 3)`` LUT.

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

    lo = float(np.min(s)) if vmin is None else float(vmin)
    hi = float(np.max(s)) if vmax is None else float(vmax)
    rng = hi - lo
    if rng <= 0.0:
        # Degenerate range (all-equal scalars): map everything to the LUT centre,
        # matching a 0-width range that the viewer would render as a flat colour.
        norm = np.full(s.shape, 0.5, dtype=np.float64)
    else:
        norm = np.clip((s - lo) / rng, 0.0, 1.0)

    n_lut = lut.shape[0]
    idx = np.minimum((norm * (n_lut - 1)).round().astype(np.intp), n_lut - 1)
    out: NDArray[np.float32] = np.ascontiguousarray(lut[idx], dtype=np.float32)
    return out
