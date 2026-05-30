"""Colormap resolution and lookup.

Resolves colormap names or numpy arrays to standardized (256, 3) uint8 LUTs.
Resolution order: built-in → matplotlib → colorcet → error.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from luxar.colormaps.builtins import BUILTIN_COLORMAP_NAMES, get_builtin_lut


def resolve_colormap(
    colormap: Union[str, NDArray[np.uint8], NDArray[np.float32], NDArray[np.float64]],
) -> NDArray[np.uint8]:
    """Resolve a colormap name or array to a (256, 3) uint8 LUT.

    Parameters
    ----------
    colormap : str or ndarray
        - String: tries built-in names, then matplotlib, then colorcet.
        - ndarray of shape (N, 3): resampled to (256, 3) uint8.

    Returns
    -------
    NDArray[np.uint8]
        Colormap LUT of shape (256, 3).

    Raises
    ------
    ValueError
        If the name is not found in any source.
    TypeError
        If the input is not a string or array.
    """
    if isinstance(colormap, str):
        return _resolve_named(colormap)
    elif isinstance(colormap, np.ndarray):
        return _resolve_array(colormap)
    else:
        raise TypeError(
            f"Colormap must be a string or numpy array, got {type(colormap).__name__}"
        )


def _resolve_named(name: str) -> NDArray[np.uint8]:
    """Resolve a named colormap."""
    # 1. Built-in
    if name in BUILTIN_COLORMAP_NAMES:
        return get_builtin_lut(name)

    # 2. Try matplotlib
    try:
        return _from_matplotlib(name)
    except (ImportError, ValueError):
        pass

    # 3. Try colorcet
    try:
        return _from_colorcet(name)
    except (ImportError, KeyError):
        pass

    # All sources exhausted
    builtin_list = ", ".join(f"'{n}'" for n in BUILTIN_COLORMAP_NAMES)
    raise ValueError(
        f"Unknown colormap '{name}'. "
        f"Built-in colormaps: [{builtin_list}]. "
        f"Install matplotlib or colorcet for additional colormaps."
    )


def _from_matplotlib(name: str) -> NDArray[np.uint8]:
    """Sample a matplotlib colormap to 256x3 uint8."""
    import matplotlib

    # Use matplotlib.colormaps (available since 3.5, required from 3.11+)
    # Fallback to plt.get_cmap for older versions
    try:
        cmap = matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(name)

    t = np.linspace(0, 1, 256)
    rgba: np.ndarray = cmap(t)
    result: NDArray[np.uint8] = (rgba[:, :3] * 255).round().astype(np.uint8)
    return result


def _from_colorcet(name: str) -> NDArray[np.uint8]:
    """Look up a colorcet palette."""
    import colorcet

    palette = colorcet.palette[name]
    # colorcet palettes are lists of hex strings like '#rrggbb'
    lut = np.zeros((len(palette), 3), dtype=np.uint8)
    for i, hex_color in enumerate(palette):
        hex_color = hex_color.lstrip("#")
        lut[i, 0] = int(hex_color[0:2], 16)
        lut[i, 1] = int(hex_color[2:4], 16)
        lut[i, 2] = int(hex_color[4:6], 16)

    # Resample to 256 if needed
    if len(palette) != 256:
        lut = _resample_lut(lut, 256)
    return lut


def _resolve_array(arr: np.ndarray) -> NDArray[np.uint8]:
    """Validate and normalize a custom colormap array to (256, 3) uint8."""
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Custom colormap must have shape (N, 3), got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(
            f"Custom colormap must have at least 2 entries, got {arr.shape[0]}"
        )

    # Convert float [0, 1] to uint8
    if np.issubdtype(arr.dtype, np.floating):
        # Reject NaN / Inf BEFORE the range check — NaN comparisons are
        # always False, so a NaN entry would slip past `arr < 0` / `arr > 1`
        # and silently produce garbage uint8 via `(NaN * 255).astype(uint8)`.
        # Positive Inf would be caught by `arr > 1`, but the explicit check
        # gives a clearer error than "range [0, 1]" for either case.
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "Float colormap values must be finite (contains NaN or ±Inf)"
            )
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError("Float colormap values must be in [0, 1] range")
        arr = (arr * 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        raise TypeError(
            f"Custom colormap must be float32/float64 (range [0,1]) "
            f"or uint8, got {arr.dtype}"
        )

    # Resample to 256 if needed
    if arr.shape[0] != 256:
        arr = _resample_lut(arr, 256)

    return arr


def _resample_lut(lut: NDArray[np.uint8], n: int) -> NDArray[np.uint8]:
    """Resample a LUT to n entries using linear interpolation."""
    src_len = lut.shape[0]
    src_t = np.linspace(0, 1, src_len)
    dst_t = np.linspace(0, 1, n)
    resampled = np.zeros((n, 3), dtype=np.uint8)
    for c in range(3):
        resampled[:, c] = (
            np.interp(dst_t, src_t, lut[:, c].astype(np.float64))
            .round()
            .astype(np.uint8)
        )
    return resampled
