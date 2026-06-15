"""Colormap (CLUT) support for Luxar.

Provides named colormaps for mapping scalar data to RGB colors.
Built-in colormaps include microscopy linear ramps, perceptually uniform
maps, and diverging maps. Custom colormaps can be provided as numpy arrays
or resolved from matplotlib/colorcet if installed.

Usage::

    from luxar.colormaps import resolve_colormap, BUILTIN_COLORMAP_NAMES

    # Get a built-in colormap as (256, 3) uint8
    lut = resolve_colormap("viridis")

    # Use matplotlib/colorcet name (if installed)
    lut = resolve_colormap("cividis")

    # Use a custom numpy array
    lut = resolve_colormap(my_array)  # shape (N, 3), resampled to 256
"""

from luxar.colormaps.apply import scalars_to_colors
from luxar.colormaps.builtins import BUILTIN_COLORMAP_NAMES
from luxar.colormaps.registry import resolve_colormap

__all__ = [
    "BUILTIN_COLORMAP_NAMES",
    "resolve_colormap",
    "scalars_to_colors",
]
