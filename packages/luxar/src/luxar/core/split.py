"""luxar.split – Helpers for the split-kind specialized Group.

A split-kind ``Group`` is a compile-time decomposition of a single large
geometry node (10M+ points / lines / splats) into multiple smaller child nodes
so that per-child frustum culling, per-child LOD, etc. can kick in. The user
does not see the decomposition: they call ``add_points(...)`` (or the like)
with ``split=True`` / ``split=dict(max_elements=N)`` and the layers panel
presents one logical layer of the original geometry type.

The decomposition is **recursive midpoint BSP**: at each step we split along
the longest axis of the current bounding box at the midpoint of that axis,
recursing until each part has at most ``max_elements`` elements. Predictable
axis-aligned tile boundaries; parts may be uneven in size (which is fine —
frustum culling discards empty tiles cheaply).

This module hosts:

* :func:`midpoint_bsp_partition` — the pure-NumPy splitter. Returns a list of
  index arrays into the original positions.
* :func:`validate_split_group` — the well-formedness check (free function,
  matches the validator pattern in ``core/lod.py``).
* :data:`SplitSpec` — the value-vocabulary type alias for the ``split=``
  convenience kwarg on ``add_points`` / ``add_lines`` / ``add_gsplats``.
* :data:`DEFAULT_MAX_ELEMENTS` — the cap used when the user passes
  ``split=True`` without a dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np
from numpy.typing import NDArray

from .lod import resolve_display_type

if TYPE_CHECKING:
    from .node import Node


#: Sentinel-typed alias for the value vocabulary of the ``split=`` kwarg.
#: ``None`` = no split, ``True`` = use :data:`DEFAULT_MAX_ELEMENTS`,
#: ``dict[str, Any]`` = user-supplied (currently only ``max_elements=`` is
#: honored; reserved for future split-algorithm parameters).
SplitSpec = Union[None, bool, dict]


#: Default cap for ``split=True`` (no dict). Sits in the upper half of the
#: 100K–10M smooth-interaction range from ``CLAUDE.md`` — large enough that
#: a single tile is still a comfortable WebGL batch, small enough that
#: splitting is worth it for the 10M+ node sizes the feature targets.
DEFAULT_MAX_ELEMENTS: int = 1_000_000


# ────────────────────────────────────────────────────────────────────────
# Recursive midpoint BSP
# ────────────────────────────────────────────────────────────────────────


def midpoint_bsp_partition(
    positions: NDArray,
    max_elements: int,
) -> List[NDArray[np.intp]]:
    """Recursively split ``positions`` along longest-axis midpoints.

    Args:
        positions: ``(N, d)`` array of element positions. At least 3 spatial
            dimensions are required; only the first 3 are used for splitting
            (extra dims ride along untouched — they don't drive frustum
            culling, which only cares about the screen-projected 3D extent).
        max_elements: Cap on a single part's size. Each returned part has
            ``len(part) <= max_elements`` except in the degenerate case
            where all elements coincide on every axis (then no split makes
            progress and we return the whole input as one part).

    Returns:
        List of index arrays into ``positions``. Concatenating them in
        order yields a permutation of ``np.arange(len(positions))``. The
        list is non-empty for non-empty input; a single-element list is
        returned when ``len(positions) <= max_elements``.

    Notes:
        Pure NumPy, no external deps. The recursion picks the longest of
        ``x``/``y``/``z`` at each level and splits at the midpoint of its
        current bbox. Resulting parts are axis-aligned but not necessarily
        balanced in element count — a feature, since real-world data is
        rarely uniform and forcing balance via median-split would cost an
        ``O(n log n)`` sort per level.
    """
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2-D (N, d); got shape {positions.shape}"
        )
    if positions.shape[1] < 3:
        raise ValueError(
            "midpoint_bsp_partition needs at least 3 spatial dimensions; "
            f"got positions with shape {positions.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n = positions.shape[0]
    if n == 0:
        return []

    # Work over the first 3 spatial dims only (the rest ride along).
    spatial = positions[:, :3]

    result: List[NDArray[np.intp]] = []

    def recurse(indices: NDArray[np.intp]) -> None:
        if indices.size <= max_elements:
            result.append(indices)
            return
        sub = spatial[indices]
        mins = sub.min(axis=0)
        maxs = sub.max(axis=0)
        extents = maxs - mins
        axis = int(np.argmax(extents))
        if extents[axis] == 0:
            # Every element coincides on every spatial axis — splitting
            # cannot make progress. Emit as one (oversized) part and let
            # the caller decide whether to surface a warning.
            result.append(indices)
            return
        mid = (mins[axis] + maxs[axis]) * 0.5
        left_mask = sub[:, axis] < mid
        left = indices[left_mask]
        right = indices[~left_mask]
        # Degenerate partition (everything on one side because of equality
        # at the midpoint): force a single-element move so we make progress.
        if left.size == 0 or right.size == 0:
            half = indices.size // 2
            # Stable: re-sort by the split axis and split at the midpoint
            # of the sorted array. O(n log n) for this edge case only.
            order = np.argsort(sub[:, axis], kind="stable")
            left = indices[order[:half]]
            right = indices[order[half:]]
        recurse(left)
        recurse(right)

    recurse(np.arange(n, dtype=np.intp))
    return result


# ────────────────────────────────────────────────────────────────────────
# Validator
# ────────────────────────────────────────────────────────────────────────


def validate_split_group(group: "Node") -> None:
    """Check that a kind=split ``Group`` is well-formed.

    Raises ``ValueError`` if:

    - the group has zero children;
    - ``display_type`` is missing or empty;
    - ``max_elements`` is missing or < 1;
    - any child's resolved ``display_type`` (per
      :func:`luxar.core.lod.resolve_display_type`) differs from the parent's
      — homogeneity is mandatory for Split (you can't decompose a single
      logical layer into mixed-type parts).

    The ``position_bounds`` union check (parent's bbox = union of
    children's bboxes) is enforced by the compiler at write time, not here
    — the validator runs on an in-memory tree where the parent's bounds
    may not yet have been computed.
    """
    if not group.children:
        raise ValueError(
            f"Split group '{group.path or group.name}' has no children"
        )
    display = group.attrs.get("display_type")
    if not isinstance(display, str) or not display:
        raise ValueError(
            f"Split group '{group.path or group.name}' is missing the "
            "required 'display_type' attribute"
        )
    max_elements = group.attrs.get("max_elements")
    if not isinstance(max_elements, int) or max_elements < 1:
        raise ValueError(
            f"Split group '{group.path or group.name}' has invalid "
            f"max_elements={max_elements!r}; expected an int >= 1"
        )
    for i, child in enumerate(group.children):
        child_display = resolve_display_type(child)
        if child_display != display:
            raise ValueError(
                f"Split group '{group.path or group.name}' is non-homogeneous: "
                f"display_type={display!r} but child {i} ({child.name!r}) "
                f"resolves to display_type={child_display!r}"
            )
