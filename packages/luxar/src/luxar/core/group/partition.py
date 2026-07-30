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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import numpy as np
from arbol import aprint
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
# BSP tree — the recursion structure (split planes) the splitters produce
# ────────────────────────────────────────────────────────────────────────


@dataclass
class BSPNode:
    """A node of the recursive BSP the spatial splitters build.

    An **internal** node carries the split plane it applied: ``axis`` (one of
    the first-3 spatial axes, ``0``/``1``/``2``) and the ``split`` coordinate
    (in the positions' own coordinate space), plus its two children. A
    **leaf** carries the index array of the elements it contains. The split
    convention matches the splitters exactly: the ``left`` subtree holds
    ``coord < split`` and ``right`` holds ``coord >= split``.

    The tree is the split-plane record needed for an exact, camera-position-
    safe back-to-front (painter's) ordering of the leaf parts in the viewer:
    at each node the eye is on one side of ``split`` and everything on the far
    side draws before everything on the near side (Fuchs–Kedem–Naylor).
    """

    # Leaf payload (``None`` on internal nodes).
    indices: Optional[NDArray[np.intp]] = None
    # Internal split (``None`` on leaves).
    axis: Optional[int] = None
    split: Optional[float] = None
    left: Optional["BSPNode"] = None
    right: Optional["BSPNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def leaves(self) -> Iterator["BSPNode"]:
        """Yield leaf nodes in left-first DFS order.

        This is the SAME order the flat splitters append parts in (they
        ``recurse(left); recurse(right)``), so leaf *k* here corresponds to
        flat part *k* — the numbering the serialized tree's ``"part"`` refs and
        the on-disk ``child_index`` both use.
        """
        if self.is_leaf:
            yield self
        else:
            assert self.left is not None and self.right is not None
            yield from self.left.leaves()
            yield from self.right.leaves()

    def to_serializable(self) -> Dict[str, Any]:
        """Serialize to a JSON/zarr-attr-friendly nested dict.

        Leaves are numbered in :meth:`leaves` order (``0, 1, 2, …``) so each
        leaf's ``"part"`` index lines up with the flat parts list and the
        on-disk ``part_<i>`` / ``child_index``. Internal nodes emit
        ``{"axis", "split", "left", "right"}``.
        """
        counter = [0]

        def build(node: "BSPNode") -> Dict[str, Any]:
            if node.is_leaf:
                part = counter[0]
                counter[0] += 1
                return {"part": part}
            assert node.axis is not None and node.split is not None
            assert node.left is not None and node.right is not None
            return {
                "axis": int(node.axis),
                "split": float(node.split),
                "left": build(node.left),
                "right": build(node.right),
            }

        return build(self)


def _bsp_tree_median(
    spatial: NDArray, max_elements: int, indices: NDArray[np.intp]
) -> BSPNode:
    """Median-split BSP tree recursion (see :func:`median_bsp_partition`)."""
    if indices.size <= max_elements:
        return BSPNode(indices=indices)
    sub = spatial[indices]
    mins = sub.min(axis=0)
    maxs = sub.max(axis=0)
    extents = maxs - mins
    axis = int(np.argmax(extents))
    if extents[axis] == 0:
        return BSPNode(indices=indices)
    coords = sub[:, axis]
    median = float(np.median(coords))
    left_mask = coords < median
    left = indices[left_mask]
    right = indices[~left_mask]
    split = median
    if left.size == 0 or right.size == 0:
        order = np.argsort(coords, kind="stable")
        half = indices.size // 2
        left = indices[order[:half]]
        right = indices[order[half:]]
        # The plane that separates the two rank-bisected halves (the first
        # right element's coordinate); ties at it fall in ``left``, so the
        # ``left < split`` convention holds up to coincident coordinates.
        split = float(coords[order[half]])
    return BSPNode(
        axis=axis,
        split=split,
        left=_bsp_tree_median(spatial, max_elements, left),
        right=_bsp_tree_median(spatial, max_elements, right),
    )


def _bsp_tree_midpoint(
    spatial: NDArray, max_elements: int, indices: NDArray[np.intp]
) -> BSPNode:
    """Midpoint-split BSP tree recursion (see :func:`midpoint_bsp_partition`)."""
    if indices.size <= max_elements:
        return BSPNode(indices=indices)
    sub = spatial[indices]
    mins = sub.min(axis=0)
    maxs = sub.max(axis=0)
    extents = maxs - mins
    axis = int(np.argmax(extents))
    if extents[axis] == 0:
        return BSPNode(indices=indices)
    mid = float((mins[axis] + maxs[axis]) * 0.5)
    coords = sub[:, axis]
    left_mask = coords < mid
    left = indices[left_mask]
    right = indices[~left_mask]
    split = mid
    if left.size == 0 or right.size == 0:
        order = np.argsort(coords, kind="stable")
        half = indices.size // 2
        left = indices[order[:half]]
        right = indices[order[half:]]
        split = float(coords[order[half]])
    return BSPNode(
        axis=axis,
        split=split,
        left=_bsp_tree_midpoint(spatial, max_elements, left),
        right=_bsp_tree_midpoint(spatial, max_elements, right),
    )


def _bsp_tree_sah(
    spatial: NDArray,
    max_elements: int,
    indices: NDArray[np.intp],
    n_candidates: int,
) -> BSPNode:
    """SAH-split BSP tree recursion (see :func:`sah_bsp_partition`)."""

    def surface_area(mins: NDArray, maxs: NDArray) -> float:
        """SAH cost proxy: the measure of the box boundary.

        SAH weights a child by the probability a random ray hits it, which is
        proportional to the box's boundary measure — surface area
        ``2(xy + xz + yz)`` in 3D, but **perimeter** ``2(x + y)`` in 2D. Using
        the 3D form on planar data would index a non-existent third extent.
        """
        ext = np.maximum(0.0, maxs - mins)
        if ext.shape[0] == 2:
            return float(2.0 * (ext[0] + ext[1]))
        return float(2.0 * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2]))

    if indices.size <= max_elements:
        return BSPNode(indices=indices)
    sub = spatial[indices]
    mins = sub.min(axis=0)
    maxs = sub.max(axis=0)
    extents = maxs - mins
    if not np.any(extents > 0):
        return BSPNode(indices=indices)

    best_score = np.inf
    best_axis = -1
    best_pos = 0.0
    for axis in range(sub.shape[1]):
        if extents[axis] == 0:
            continue
        cand = np.linspace(mins[axis], maxs[axis], n_candidates + 2)[1:-1]
        for pos in cand:
            left_mask = sub[:, axis] < pos
            n_left = int(left_mask.sum())
            n_right = int(indices.size - n_left)
            if n_left == 0 or n_right == 0:
                continue
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
        return BSPNode(indices=indices)

    coords = sub[:, best_axis]
    left_mask = coords < best_pos
    left = indices[left_mask]
    right = indices[~left_mask]
    split = best_pos
    if left.size == 0 or right.size == 0:
        order = np.argsort(coords, kind="stable")
        half = indices.size // 2
        left = indices[order[:half]]
        right = indices[order[half:]]
        split = float(coords[order[half]])
    return BSPNode(
        axis=best_axis,
        split=split,
        left=_bsp_tree_sah(spatial, max_elements, left, n_candidates),
        right=_bsp_tree_sah(spatial, max_elements, right, n_candidates),
    )


def spatial_bsp_tree(
    positions: NDArray,
    max_elements: int,
    *,
    rule: str = "median",
    n_candidates: int = 32,
) -> BSPNode:
    """Build the BSP **tree** (split planes retained) for ``positions``.

    The tree sibling of the three flat splitters: its :meth:`BSPNode.leaves`
    (left-first DFS) yield exactly the parts (and in the same order) the
    matching ``*_bsp_partition`` returns, but every internal node also records
    the split ``axis``/``split`` — the information a viewer needs for an exact
    back-to-front ordering of the parts. ``rule`` selects the splitter
    (``"median"`` default / ``"midpoint"`` / ``"sah"``); ``n_candidates`` is
    forwarded to the SAH rule only.

    Splits only ever fall on the first up-to-three (spatial) axes, so a
    serialized tree's ``axis`` is always a center-column index below 3 (``0``/
    ``1`` for 2D data, ``0``/``1``/``2`` for 3D+). The viewer maps that column
    through ``displayDims`` to reach its own local axis — see
    ``render-order.ts``; the two coincide only when ``displayDims == [0, 1, 2]``.
    """
    if positions.ndim != 2:
        raise ValueError(f"positions must be 2-D (N, d); got shape {positions.shape}")
    if positions.shape[1] < 2:
        raise ValueError(
            "spatial_bsp_tree needs at least 2 spatial dimensions; "
            f"got positions with shape {positions.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n = positions.shape[0]
    if n == 0:
        raise ValueError("spatial_bsp_tree needs a non-empty positions array")

    spatial = positions[:, : min(3, positions.shape[1])]
    root = np.arange(n, dtype=np.intp)
    if rule == "median":
        return _bsp_tree_median(spatial, max_elements, root)
    if rule == "midpoint":
        return _bsp_tree_midpoint(spatial, max_elements, root)
    if rule == "sah":
        if n_candidates < 2:
            raise ValueError(
                f"n_candidates must be >= 2 (need at least one interior split); "
                f"got {n_candidates}"
            )
        return _bsp_tree_sah(spatial, max_elements, root, n_candidates)
    raise ValueError(f"rule must be 'median', 'midpoint', or 'sah'; got {rule!r}")


def _flat_parts(root: BSPNode) -> List[NDArray[np.intp]]:
    """Flatten a BSP tree to the leaf index-arrays list (the ``*_bsp_partition``
    return shape), in :meth:`BSPNode.leaves` order."""
    parts: List[NDArray[np.intp]] = []
    for leaf in root.leaves():
        assert leaf.indices is not None
        parts.append(leaf.indices)
    return parts


def warn_if_oversized_single_part(
    n_parts: int, part_size: int, max_elements: int, name: str
) -> None:
    """Warn when the BSP could not split below ``max_elements``.

    All three splitters return a single oversized part on fully-coincident
    (or single-atomic-polyline) input. Each adder's ``len(parts) > 1`` gate
    then falls through to a plain single-leaf write with no indication the
    cap was violated; this surfaces that case (shared across points / lines /
    gsplats per the three-geometry symmetry rule). Degraded-but-correct
    render, not data loss.
    """
    if n_parts == 1 and part_size > max_elements:
        aprint(
            f"  ⚠️  partition could not split '{name}' below "
            f"max_elements={max_elements:,}: {part_size:,} coincident/atomic "
            f"elements written as one oversized part."
        )


def warn_if_partition_needs_more_dims(ndim: int, name: str) -> bool:
    """Warn when a requested partition can't run for want of spatial dims.

    Splitting needs at least 2 spatial axes (see :func:`spatial_bsp_tree`), so
    1D data cannot be partitioned. The three adders used to gate their partition
    branch on a bare dimension check, which meant an explicit ``partition=`` —
    or a compiler-level ``auto_partition_max_elements`` — was dropped in silence
    and the caller got one un-partitioned leaf with no clue why. Sibling of
    :func:`warn_if_oversized_single_part`, shared across points / lines /
    gsplats per the three-geometry symmetry rule.

    Returns:
        ``True`` when partitioning can proceed, ``False`` (after warning) when
        there are too few spatial dimensions.
    """
    if ndim < 2:
        aprint(
            f"  ⚠️  partition requested for '{name}' but spatial splitting needs "
            f"at least 2 dimensions (got {ndim}D) — writing a single leaf."
        )
        return False
    return True


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
            dimensions are required (2 for planar data); only the first 3 are
            used for splitting (extra dims ride along untouched — they don't
            drive frustum culling, which only cares about the screen-projected
            3D extent).
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
    if positions.shape[1] < 2:
        raise ValueError(
            "midpoint_bsp_partition needs at least 2 spatial dimensions; "
            f"got positions with shape {positions.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n = positions.shape[0]
    if n == 0:
        return []

    # Work over the first 3 spatial dims only (the rest ride along). The tree
    # builder is the single source of truth; the flat list is its leaves in
    # left-first DFS order (see :func:`spatial_bsp_tree` / :class:`BSPNode`).
    spatial = positions[:, : min(3, positions.shape[1])]
    return _flat_parts(
        _bsp_tree_midpoint(spatial, max_elements, np.arange(n, dtype=np.intp))
    )


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
        positions: ``(N, d)`` array; at least 2 spatial dims (only the
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
    if positions.shape[1] < 2:
        raise ValueError(
            "median_bsp_partition needs at least 2 spatial dimensions; "
            f"got positions with shape {positions.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n = positions.shape[0]
    if n == 0:
        return []

    # The tree builder is the single source of truth; the flat list is its
    # leaves in left-first DFS order (see :func:`spatial_bsp_tree`).
    spatial = positions[:, : min(3, positions.shape[1])]
    return _flat_parts(
        _bsp_tree_median(spatial, max_elements, np.arange(n, dtype=np.intp))
    )


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
            spatial dimensions required, 2 for planar data (only the first 3
            drive the split).
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
    if vertices.shape[1] < 2:
        raise ValueError(
            "midpoint_bsp_polylines needs at least 2 spatial dimensions; "
            f"got vertices with shape {vertices.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n_polylines = len(polyline_indices)
    if n_polylines == 0:
        return []

    # Per-polyline centroid (first 3 spatial dims) and vertex count.
    spatial = vertices[:, : min(3, vertices.shape[1])]
    centroids = np.zeros((n_polylines, spatial.shape[1]), dtype=np.float64)
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
        vertices: ``(N, d)`` array of vertex positions; at least 2 spatial
            dims (only the first 3 drive the split).
        polyline_indices: Per-polyline vertex-index arrays (output of
            :func:`luxar.core.group.lod.lines.identify_polylines`).
        max_elements: Cap on a single part's vertex count.

    Returns:
        List of parts; each part is a list of polyline indices. Concatenation
        permutes ``range(len(polyline_indices))``.
    """
    if vertices.ndim != 2:
        raise ValueError(f"vertices must be 2-D (N, d); got shape {vertices.shape}")
    if vertices.shape[1] < 2:
        raise ValueError(
            "median_bsp_polylines needs at least 2 spatial dimensions; "
            f"got vertices with shape {vertices.shape}"
        )
    if max_elements < 1:
        raise ValueError(f"max_elements must be >= 1, got {max_elements}")

    n_polylines = len(polyline_indices)
    if n_polylines == 0:
        return []

    spatial = vertices[:, : min(3, vertices.shape[1])]
    centroids = np.zeros((n_polylines, spatial.shape[1]), dtype=np.float64)
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
    if positions.shape[1] < 2:
        raise ValueError(
            "sah_bsp_partition needs at least 2 spatial dimensions; "
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

    # The tree builder is the single source of truth; the flat list is its
    # leaves in left-first DFS order (see :func:`spatial_bsp_tree`).
    spatial = positions[:, : min(3, positions.shape[1])]
    return _flat_parts(
        _bsp_tree_sah(spatial, max_elements, np.arange(n, dtype=np.intp), n_candidates)
    )


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
