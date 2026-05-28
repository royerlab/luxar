"""luxar.core.group.lod.spatial_uniform – Stratified-grid sampler for LOD ordering.

Used by ``core/group/lod/points.py`` and ``core/group/lod/lines.py`` to compute a
spatial-uniform additive-LOD permutation: elements are assigned to LOD
levels by which grid resolution first "covers" their position, so any
cumulative prefix of the permutation yields approximately-uniform
density across the dataset's bounding box.

Algorithm — coarsest grid first:

1. Compute the dataset's spatial bounding box (per-axis min / max over
   the first 3 spatial dims).
2. For each LOD level ``i`` from 0 to ``n_lods - 1``, build a grid of
   resolution ``2^(i + 1)`` per axis (so LOD 0 = 2×2×2, LOD 1 = 4×4×4,
   …, LOD k = 2^(k+1) per axis).
3. Walk each LOD coarse-to-fine. For each grid cell at level ``i`` that
   no element has yet been assigned to from any coarser level, pick the
   first unassigned element in that cell (stable scan order) and assign
   it to LOD ``i``. Mark that element as taken.
4. Any element still unassigned after LOD ``n_lods - 1`` (rare: only
   happens when multiple elements fall in the same finest-grid cell) is
   appended to the last LOD's group.

The returned permutation lists every element exactly once, ordered LOD
0 first then LOD 1 etc. The caller (``make_additive_lod_points`` /
``make_additive_lod_lines``) slices the permutation by per-level
``counts`` to produce the per-subgroup index arrays.

Pure NumPy. No external dependencies. O(N · n_lods).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stratified_grid_order(
    positions: NDArray,
    n_lods: int,
) -> tuple[NDArray[np.intp], list[int]]:
    """Compute a stratified-grid LOD permutation.

    Args:
        positions: ``(N, d)`` array of element positions; ``d >= 3``.
            Only the first 3 columns are used (extra dims pass through
            without affecting the spatial binning).
        n_lods: Target number of LOD levels (``>= 1``). The actual
            number of non-empty levels emitted may be smaller if data
            is sparse (some LOD levels can be empty); empty levels are
            still included in the per-level ``counts`` output for
            caller-side handling.

    Returns:
        A tuple ``(permutation, per_level_counts)``:

        - ``permutation``: ``(N,)`` int64. The k-th entry is the
          original index of the element at rank ``k``. Concatenation
          of LOD 0's elements then LOD 1's then ….
        - ``per_level_counts``: list of length ``n_lods``. Entry ``i``
          is the number of elements assigned to LOD ``i``. Sums to
          ``N``. Trailing entries may be 0 if data is sparser than
          ``n_lods`` distinct grid populations.
    """
    if n_lods < 1:
        raise ValueError(f"n_lods must be >= 1, got {n_lods}")
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError(
            f"positions must be (N, d) with d >= 3; got shape {positions.shape}"
        )
    n = positions.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.intp), [0] * n_lods

    # Bbox from the first 3 spatial dims. Extra columns are ignored.
    pos = positions[:, :3].astype(np.float64, copy=False)
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    extents = maxs - mins
    # Avoid div-by-zero on degenerate axes (all elements coincide on
    # an axis); shift the extent to 1.0 so every element lands in
    # bucket 0 along that axis.
    extents[extents == 0.0] = 1.0

    # Per-level assignment. ``assigned[i]`` becomes True when element i
    # has been picked at some level.
    assigned = np.zeros(n, dtype=bool)
    permutation = np.empty(n, dtype=np.intp)
    per_level_counts: list[int] = [0] * n_lods
    cursor = 0

    for level in range(n_lods):
        res = 1 << (level + 1)  # 2^(level+1): 2, 4, 8, 16, ...
        # Compute integer bucket index along each axis. Clamp to
        # [0, res-1] so the inclusive max edge maps inside the grid.
        normalized = (pos - mins) / extents  # in [0, 1]
        bucket_xyz = np.minimum((normalized * res).astype(np.int64), res - 1)
        # Flatten to a single bucket id per element.
        bucket_id = (
            bucket_xyz[:, 0] * (res * res)
            + bucket_xyz[:, 1] * res
            + bucket_xyz[:, 2]
        )

        # Find the first unassigned element in each occupied bucket.
        # Iterate in original-index order so the choice is stable.
        candidates = np.nonzero(~assigned)[0]
        if candidates.size == 0:
            break
        seen_buckets: set[int] = set()
        picked: list[int] = []
        for idx in candidates:
            b = int(bucket_id[idx])
            if b in seen_buckets:
                continue
            seen_buckets.add(b)
            picked.append(int(idx))

        if picked:
            picked_arr = np.asarray(picked, dtype=np.intp)
            assigned[picked_arr] = True
            permutation[cursor : cursor + picked_arr.size] = picked_arr
            cursor += picked_arr.size
            per_level_counts[level] = picked_arr.size

    # Append any element still unassigned (occurs when multiple elements
    # share the finest-grid cell). They land in the last LOD group so
    # the permutation remains a valid bijection.
    if cursor < n:
        remaining = np.nonzero(~assigned)[0]
        permutation[cursor : cursor + remaining.size] = remaining
        per_level_counts[n_lods - 1] += int(remaining.size)
        cursor += int(remaining.size)

    assert cursor == n, f"permutation incomplete: {cursor} / {n}"
    return permutation, per_level_counts
