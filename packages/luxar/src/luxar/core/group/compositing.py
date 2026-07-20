"""Compositing primitives used by Group's partition-wrapping path.

These helpers are shared by the kind=partition and kind=lod wrapper builders
(see ``adders/`` and ``gsplats_pipeline/``). They are pure data
operations — no Group/Node references — and have no side effects.

Exposed:

* :data:`COMPOSITING_ATTRS` — frozenset of attribute names that ride on
  a wrapper Group (where the user thinks of the wrapper as "their
  layer") rather than getting copied onto each internal child.
* :func:`slice_optional_array` — slice an array-valued leaf parameter by
  index, leaving scalars / None / mis-sized inputs untouched.
* :func:`position_bounds_from_array` — per-axis min/max of an (N, D)
  position array, in the writer's shape.
* :func:`sync_custom_colormap_attr` — mirror the writer's custom-colormap
  resolution (`ndarray / non-builtin name -> 'custom'`) into the adder's
  attrs dict so the returned node object matches what zarr stores.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

#: Attrs that ride on a kind=lod / kind=partition wrapper Group (where the user
#: thinks of the wrapper as "their layer") rather than getting copied onto
#: each internal child. Compositing semantics (opacity, gamma, ...) flow
#: down to the children through Group inheritance at render time, so writing
#: them once on the parent is correct. ``colormap`` and ``truncation_radius``
#: are deliberately NOT compositing: the writer auto-defaults them per leaf,
#: which under nearest-ancestor-wins would shadow a parent's setting.
COMPOSITING_ATTRS = frozenset(
    {
        "transform",
        "opacity",
        "absorption",
        "gamma",
        "intensity",
        "offset",
        "blending_mode",
        "layer",
        "visible",
        "nd_transform",
    }
)


def sync_custom_colormap_attr(attrs: Dict[str, Any]) -> None:
    """Sync ``attrs['colormap']`` with what the compiler wrote to zarr.

    The writer resolves any non-builtin colormap — an ndarray LUT or a
    matplotlib/colorcet name — to a ``colormap_lut`` dataset plus
    ``colormap='custom'`` (``io/_compiler/colormap.py``), but it mutates its
    OWN copy of the attrs (the ``**attrs`` packing boundary), so the adder
    must mirror the substitution for the node object it returns. No-op when
    ``colormap`` is absent or a builtin name.
    """
    if "colormap" not in attrs:
        return
    from ...colormaps.builtins import BUILTIN_COLORMAP_NAMES

    cm = attrs["colormap"]
    if not isinstance(cm, str) or cm not in BUILTIN_COLORMAP_NAMES:
        attrs["colormap"] = "custom"


def slice_optional_array(value: Any, indices: np.ndarray, n_elements: int) -> Any:
    """Slice an array-valued leaf parameter by index; pass non-per-element values through.

    Used by the ``partition=`` wrapping path on the leaf adders. Returns
    unchanged when:
      * ``value`` is ``None`` or a scalar (``int`` / ``float`` / ``bool``
        / ``str``) — applies uniformly to every part.
      * ``value`` is a 0-D array.
      * ``value``'s first-axis length doesn't match ``n_elements`` (e.g.
        a 3-vector RGB broadcast, or a length-1 sentinel).
    Slices the first axis when the input is a list of length
    ``n_elements`` (string labels) or an array whose first axis matches.
    """
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, list):
        if len(value) == n_elements:
            return [value[i] for i in indices]
        return value
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if arr.ndim == 0:
        return value
    if arr.shape[0] == n_elements:
        return arr[indices]
    return value


def position_bounds_from_array(positions: np.ndarray) -> Dict[str, List[float]]:
    """Per-axis min/max of an ``(N, D)`` position array, in the writer's shape.

    Matches what the compiler's ``_compute_position_bounds`` writes onto
    each leaf node, so the partition-kind wrapper's ``position_bounds`` is
    the same shape as its children's. Used by the ``partition=`` wrapping
    path to compute the parent bbox directly from the source array
    instead of round-tripping through the per-leaf zarr writes.
    """
    if positions.size == 0:
        raise ValueError("Cannot compute position_bounds from empty array")
    return {
        "min": positions.min(axis=0).astype(float).tolist(),
        "max": positions.max(axis=0).astype(float).tolist(),
    }
