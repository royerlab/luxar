"""luxar.partition – Helpers for the partition-kind specialized Group.

A partition-kind ``Group`` is a compile-time decomposition of a single large
geometry node (10M+ points / lines / splats) into multiple smaller child nodes
so that per-child frustum culling, per-child LOD, etc. can kick in. The user
does not see the decomposition: they call ``add_points(...)`` (or the like)
with ``partition=True`` / ``partition=dict(max_elements=N)`` and the layers
panel presents one logical layer of the original geometry type.

The decomposition is a **recursive BSP**: at each step we split the current
bounding box along an axis and recurse until each part has at most
``max_elements`` elements. Three split rules are available (selected via the
``partition=dict(rule=...)`` kwarg):

* ``"median"`` (default) — split the longest axis at the **median** coordinate,
  giving balanced part counts in O(n) per level. Best for the clustered data
  scientific scenes usually contain.
* ``"midpoint"`` — split the longest axis at the geometric **midpoint**.
  Cheapest; predictable axis-aligned tiles but parts may be very uneven on
  clustered data.
* ``"sah"`` — surface-area-heuristic split-plane selection; best for heavily
  skewed data (one dense cluster + a thin streamer) at higher cost.

This module hosts:

* :func:`median_bsp_partition` / :func:`midpoint_bsp_partition` /
  :func:`sah_bsp_partition` — the pure-NumPy point/gsplats splitters. Each
  returns a list of index arrays into the original positions.
* :func:`median_bsp_polylines` / :func:`midpoint_bsp_polylines` — the
  polyline-atomic variants for ``add_lines``.
* :func:`validate_partition_group` — the well-formedness check (free function,
  matches the validator pattern in ``core/group/lod/gsplats.py``).
* :data:`PartitionSpec` — the value-vocabulary type alias for the
  ``partition=`` convenience kwarg on ``add_points`` / ``add_lines`` /
  ``add_gsplats``.
* :data:`DEFAULT_MAX_ELEMENTS` — the cap used when the user passes
  ``partition=True`` without a dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np
from numpy.typing import NDArray

from .lod.group import resolve_display_type

if TYPE_CHECKING:
    from ..node import Node


#: Sentinel-typed alias for the value vocabulary of the ``partition=`` kwarg.
#: ``None`` = no partition, ``True`` = use :data:`DEFAULT_MAX_ELEMENTS`,
#: ``dict[str, Any]`` = user-supplied (``max_elements=`` and ``rule=`` are
#: honored; reserved for future partition-algorithm parameters).
PartitionSpec = Union[None, bool, dict]


#: Default cap for ``partition=True`` (no dict). Sits in the upper half of the
#: 100K–10M smooth-interaction range from ``CLAUDE.md`` — large enough that
#: a single tile is still a comfortable WebGL batch, small enough that
#: partitioning is worth it for the 10M+ node sizes the feature targets.
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
        balanced in element count. For balanced parts use
        :func:`median_bsp_partition` (the default rule).
    """
    if positions.ndim != 2:
        raise ValueError(f"positions must be 2-D (N, d); got shape {positions.shape}")
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
# Recursive median (balanced) BSP — the default rule
# ────────────────────────────────────────────────────────────────────────


def median_bsp_partition(
    positions: NDArray,
    max_elements: int,
) -> List[NDArray[np.intp]]:
    """Recursively split ``positions`` along longest-axis **medians**.

    Identical contract to :func:`midpoint_bsp_partition`, but each split
    falls at the median coordinate of the longest axis instead of its
    geometric midpoint. This yields parts that are balanced in element
    count (each side gets ~half the elements), which matters for the
    clustered, non-uniform data scientific scenes usually contain: a
    geometric-midpoint split of a tight cluster can put nearly all
    elements on one side and recurse many times, whereas a median split
    halves the count every level (≈ ``ceil(log2(N / max_elements))``
    levels total).

    Args:
        positions: ``(N, d)`` array; at least 3 spatial dims (only the
            first 3 drive the split, the rest ride along).
        max_elements: Cap on a single part's size. Each returned part has
            ``len(part) <= max_elements`` except the degenerate all-coincident
            case.

    Returns:
        List of index arrays into ``positions``; concatenation permutes
        ``np.arange(len(positions))``.

    Notes:
        Pure NumPy. The per-level cost is ``O(n)`` (``np.median`` +
        boolean masking), comparable to midpoint and far below SAH's
        ``O(n · n_candidates · 3)``. Ties at the median are split by
        ``<`` so the left side takes strictly-smaller coordinates; an
        all-on-one-side outcome (every coordinate equal to the median)
        falls back to a stable count-bisection.
    """
    if positions.ndim != 2:
        raise ValueError(f"positions must be 2-D (N, d); got shape {positions.shape}")
    if positions.shape[1] < 3:
        raise ValueError(
            "median_bsp_partition needs at least 3 spatial dimensions; "
            f"got positions with shape {positions.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n = positions.shape[0]
    if n == 0:
        return []

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
            # All elements coincide spatially — no split makes progress.
            result.append(indices)
            return
        coords = sub[:, axis]
        median = float(np.median(coords))
        left_mask = coords < median
        left = indices[left_mask]
        right = indices[~left_mask]
        # All coordinates equal to (or above) the median — the ``<`` test
        # put everything on the right. Fall back to a stable count-bisection.
        if left.size == 0 or right.size == 0:
            order = np.argsort(coords, kind="stable")
            half = indices.size // 2
            left = indices[order[:half]]
            right = indices[order[half:]]
        recurse(left)
        recurse(right)

    recurse(np.arange(n, dtype=np.intp))
    return result


# ────────────────────────────────────────────────────────────────────────
# Polyline-aware BSP (for add_lines partition=)
# ────────────────────────────────────────────────────────────────────────


def midpoint_bsp_polylines(
    vertices: NDArray,
    polyline_indices: List[NDArray[np.intp]],
    max_elements: int,
) -> List[List[int]]:
    """Recursive midpoint BSP over per-polyline centroids.

    Polylines are atomic — every vertex of a polyline lands in exactly
    one part. The BSP is run over the per-polyline **centroids** (mean
    of constituent vertex positions, computed once); the resulting
    partition assigns whole polylines to parts.

    Args:
        vertices: ``(N, d)`` array of vertex positions. At least 3
            spatial dimensions required (only first 3 drive the split).
        polyline_indices: List of per-polyline vertex-index arrays — the
            output of :func:`luxar.core.group.lod.lines.identify_polylines`.
        max_elements: Cap on a single part's vertex count. The BSP
            recurses until each part fits, with the degenerate guarantee
            that a single polyline larger than ``max_elements`` becomes
            its own (oversized) part rather than being broken up.

    Returns:
        List of ``parts``; each ``parts[k]`` is a list of polyline
        indices (into ``polyline_indices``) assigned to part ``k``.
        Concatenating all parts produces a permutation of
        ``range(len(polyline_indices))``.

    Notes:
        Re-uses the axis-selection + midpoint-bisection idea from
        :func:`midpoint_bsp_partition`. The two functions are
        intentionally separate: the points/gsplats version partitions
        individual elements; this one partitions polylines, with the
        accounting done in vertex counts.
    """
    if vertices.ndim != 2:
        raise ValueError(f"vertices must be 2-D (N, d); got shape {vertices.shape}")
    if vertices.shape[1] < 3:
        raise ValueError(
            "midpoint_bsp_polylines needs at least 3 spatial dimensions; "
            f"got vertices with shape {vertices.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n_polylines = len(polyline_indices)
    if n_polylines == 0:
        return []

    # Per-polyline centroid (first 3 spatial dims) and vertex count.
    spatial = vertices[:, :3]
    centroids = np.zeros((n_polylines, 3), dtype=np.float64)
    sizes = np.zeros(n_polylines, dtype=np.intp)
    for p, members in enumerate(polyline_indices):
        if members.size == 0:
            continue
        centroids[p] = spatial[members].mean(axis=0)
        sizes[p] = members.size

    result: List[List[int]] = []

    def recurse(poly_idx: NDArray[np.intp]) -> None:
        total_verts = int(sizes[poly_idx].sum())
        if total_verts <= max_elements or poly_idx.size <= 1:
            result.append(poly_idx.tolist())
            return
        sub = centroids[poly_idx]
        mins = sub.min(axis=0)
        maxs = sub.max(axis=0)
        extents = maxs - mins
        axis = int(np.argmax(extents))
        if extents[axis] == 0:
            # All centroids coincide; cannot make spatial progress.
            result.append(poly_idx.tolist())
            return
        mid = (mins[axis] + maxs[axis]) * 0.5
        left_mask = sub[:, axis] < mid
        left = poly_idx[left_mask]
        right = poly_idx[~left_mask]
        if left.size == 0 or right.size == 0:
            # All on one side of the midpoint — sort by axis and bisect
            # at the median polyline. Stable on ties.
            order = np.argsort(sub[:, axis], kind="stable")
            half = poly_idx.size // 2
            left = poly_idx[order[:half]]
            right = poly_idx[order[half:]]
        recurse(left)
        recurse(right)

    recurse(np.arange(n_polylines, dtype=np.intp))
    return result


def median_bsp_polylines(
    vertices: NDArray,
    polyline_indices: List[NDArray[np.intp]],
    max_elements: int,
) -> List[List[int]]:
    """Recursive **median** BSP over per-polyline centroids.

    Identical contract to :func:`midpoint_bsp_polylines`, but each split
    falls at the median centroid coordinate of the longest axis instead
    of its geometric midpoint, so polylines are balanced across parts.
    Polylines stay atomic (every vertex of a polyline lands in one part);
    the per-part cap is accounted in vertex counts.

    Args:
        vertices: ``(N, d)`` array of vertex positions; at least 3 spatial
            dims (only first 3 drive the split).
        polyline_indices: Per-polyline vertex-index arrays (output of
            :func:`luxar.core.group.lod.lines.identify_polylines`).
        max_elements: Cap on a single part's vertex count.

    Returns:
        List of parts; each part is a list of polyline indices. Concatenation
        permutes ``range(len(polyline_indices))``.
    """
    if vertices.ndim != 2:
        raise ValueError(f"vertices must be 2-D (N, d); got shape {vertices.shape}")
    if vertices.shape[1] < 3:
        raise ValueError(
            "median_bsp_polylines needs at least 3 spatial dimensions; "
            f"got vertices with shape {vertices.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n_polylines = len(polyline_indices)
    if n_polylines == 0:
        return []

    spatial = vertices[:, :3]
    centroids = np.zeros((n_polylines, 3), dtype=np.float64)
    sizes = np.zeros(n_polylines, dtype=np.intp)
    for p, members in enumerate(polyline_indices):
        if members.size == 0:
            continue
        centroids[p] = spatial[members].mean(axis=0)
        sizes[p] = members.size

    result: List[List[int]] = []

    def recurse(poly_idx: NDArray[np.intp]) -> None:
        total_verts = int(sizes[poly_idx].sum())
        if total_verts <= max_elements or poly_idx.size <= 1:
            result.append(poly_idx.tolist())
            return
        sub = centroids[poly_idx]
        mins = sub.min(axis=0)
        maxs = sub.max(axis=0)
        extents = maxs - mins
        axis = int(np.argmax(extents))
        if extents[axis] == 0:
            result.append(poly_idx.tolist())
            return
        coords = sub[:, axis]
        median = float(np.median(coords))
        left_mask = coords < median
        left = poly_idx[left_mask]
        right = poly_idx[~left_mask]
        if left.size == 0 or right.size == 0:
            order = np.argsort(coords, kind="stable")
            half = poly_idx.size // 2
            left = poly_idx[order[:half]]
            right = poly_idx[order[half:]]
        recurse(left)
        recurse(right)

    recurse(np.arange(n_polylines, dtype=np.intp))
    return result


# ────────────────────────────────────────────────────────────────────────
# SAH (Surface-Area Heuristic) BSP — opt-in alternative to median/midpoint
# ────────────────────────────────────────────────────────────────────────


def sah_bsp_partition(
    positions: NDArray,
    max_elements: int,
    n_candidates: int = 32,
) -> List[NDArray[np.intp]]:
    """Recursive BSP using the surface-area heuristic for split-plane selection.

    Standard SAH formulation (Wald 2007 et al.): for each candidate
    split position along each spatial axis, evaluate

        SAH(split) = N_left * SA(box_left) + N_right * SA(box_right)

    and pick the (axis, position) minimizing the heuristic. The
    intuition: an SAH split balances the **work** of further traversal
    (∝ count × surface area) on each side, so non-uniform datasets get a
    better tree than median/midpoint alone.

    Args:
        positions: ``(N, d)`` array. At least 3 spatial dims.
        max_elements: Cap on a single part's size. Recursion stops once
            ``len(part) <= max_elements``.
        n_candidates: Number of uniformly-spaced split positions
            evaluated per axis per recursion (default 32 — the standard
            "binned SAH" budget). Higher = closer to a continuous
            optimum at higher cost.

    Returns:
        Same return shape as :func:`midpoint_bsp_partition` — a list of
        index arrays whose concatenation permutes ``range(N)``.

    Notes:
        Pure NumPy. Cost per recursion is ``O(N * n_candidates * 3)``.
        For the same dataset SAH typically produces fewer but more
        view-frustum-aligned parts than median/midpoint; the practical
        difference shows up on heavily skewed real-world data (one dense
        cluster + a long thin streamer).
    """
    if positions.ndim != 2:
        raise ValueError(f"positions must be 2-D (N, d); got shape {positions.shape}")
    if positions.shape[1] < 3:
        raise ValueError(
            "sah_bsp_partition needs at least 3 spatial dimensions; "
            f"got positions with shape {positions.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")
    if n_candidates < 2:
        raise ValueError(
            f"n_candidates must be >= 2 (need at least one interior split); "
            f"got {n_candidates}"
        )

    n = positions.shape[0]
    if n == 0:
        return []

    spatial = positions[:, :3]

    def surface_area(mins: NDArray, maxs: NDArray) -> float:
        ext = np.maximum(0.0, maxs - mins)
        # 2*(xy + xz + yz) — half-surface-area also works (constant
        # factor washes through the argmin), but full SA matches the
        # textbook form.
        return float(2.0 * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2]))

    result: List[NDArray[np.intp]] = []

    def recurse(indices: NDArray[np.intp]) -> None:
        if indices.size <= max_elements:
            result.append(indices)
            return
        sub = spatial[indices]
        mins = sub.min(axis=0)
        maxs = sub.max(axis=0)
        extents = maxs - mins
        if not np.any(extents > 0):
            result.append(indices)
            return

        best_score = np.inf
        best_axis = -1
        best_pos = 0.0
        for axis in range(3):
            if extents[axis] == 0:
                continue
            # Uniform candidate positions strictly interior to the box.
            cand = np.linspace(mins[axis], maxs[axis], n_candidates + 2)[1:-1]
            for pos in cand:
                left_mask = sub[:, axis] < pos
                n_left = int(left_mask.sum())
                n_right = int(indices.size - n_left)
                if n_left == 0 or n_right == 0:
                    continue
                # Left/right boxes have the same extent on the
                # non-split axes; on the split axis they shrink to
                # [mins[axis], pos] and [pos, maxs[axis]] respectively.
                left_mins = mins.copy()
                left_maxs = maxs.copy()
                left_maxs[axis] = pos
                right_mins = mins.copy()
                right_maxs = maxs.copy()
                right_mins[axis] = pos
                score = n_left * surface_area(
                    left_mins, left_maxs
                ) + n_right * surface_area(right_mins, right_maxs)
                if score < best_score:
                    best_score = score
                    best_axis = axis
                    best_pos = float(pos)

        if best_axis < 0:
            # No interior split made progress — emit as one part.
            result.append(indices)
            return

        left_mask = sub[:, best_axis] < best_pos
        left = indices[left_mask]
        right = indices[~left_mask]
        if left.size == 0 or right.size == 0:
            # Shouldn't happen given the SAH selection, but guard.
            order = np.argsort(sub[:, best_axis], kind="stable")
            half = indices.size // 2
            left = indices[order[:half]]
            right = indices[order[half:]]
        recurse(left)
        recurse(right)

    recurse(np.arange(n, dtype=np.intp))
    return result


# ────────────────────────────────────────────────────────────────────────
# Validator
# ────────────────────────────────────────────────────────────────────────


def validate_partition_group(group: "Node") -> None:
    """Check that a kind=partition ``Group`` is well-formed.

    Raises ``ValueError`` if:

    - the group has zero children;
    - ``display_type`` is missing or empty;
    - ``max_elements`` is missing or < 1;
    - any child's resolved ``display_type`` (per
      :func:`luxar.core.group.lod.group.resolve_display_type`) differs from the parent's
      — homogeneity is mandatory for a partition (you can't decompose a single
      logical layer into mixed-type parts).

    The ``position_bounds`` union check (parent's bbox = union of
    children's bboxes) is enforced by the compiler at write time, not here
    — the validator runs on an in-memory tree where the parent's bounds
    may not yet have been computed.
    """
    if not group.children:
        raise ValueError(
            f"Partition group '{group.path or group.name}' has no children"
        )
    display = group.attrs.get("display_type")
    if not isinstance(display, str) or not display:
        raise ValueError(
            f"Partition group '{group.path or group.name}' is missing the "
            "required 'display_type' attribute"
        )
    max_elements = group.attrs.get("max_elements")
    if not isinstance(max_elements, int) or max_elements < 1:
        raise ValueError(
            f"Partition group '{group.path or group.name}' has invalid "
            f"max_elements={max_elements!r}; expected an int >= 1"
        )
    for i, child in enumerate(group.children):
        child_display = resolve_display_type(child)
        if child_display != display:
            raise ValueError(
                f"Partition group '{group.path or group.name}' is non-homogeneous: "
                f"display_type={display!r} but child {i} ({child.name!r}) "
                f"resolves to display_type={child_display!r}"
            )
