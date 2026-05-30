"""luxar.core.group.lod.poisson_disk – Bridson Poisson-disk sampler with cell-grid acceleration.

An opt-in alternative to the default stratified-grid spatial-uniform
ordering in ``core/group/lod/points.py`` / ``core/group/lod/lines.py``. Bridson's
algorithm produces a blue-noise distribution (no two retained samples
closer than ``r`` apart), which gives a more perceptually uniform LOD
subset than the deterministic grid binning — at the cost of running an
explicit rejection loop.

This module exposes:

- :func:`poisson_disk_order` — coarse-to-fine LOD permutation built by
  running Bridson at progressively finer radii. Each LOD level
  contributes the points its radius selects that haven't already been
  selected at a coarser level. The output shape mirrors
  :func:`luxar.core.group.lod.spatial_uniform.stratified_grid_order` so the
  caller code in ``make_additive_lod_*`` is symmetric across methods.

Pure NumPy. O(N) expected per level thanks to the cell-grid neighbor
lookup: each level pre-builds a uniform grid sized at ``r / sqrt(d)``,
so the rejection test for a candidate point only needs to scan a 5·5·…
cell neighborhood instead of all already-accepted samples.

Reference: Bridson, "Fast Poisson Disk Sampling in Arbitrary
Dimensions" (SIGGRAPH 2007).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _build_cell_grid(
    pos: NDArray,
    mins: NDArray,
    cell_size: float,
) -> tuple[NDArray[np.intp], tuple[int, ...]]:
    """Bucket positions into a uniform 3D cell grid.

    Returns ``(cell_ids, grid_shape)`` where ``cell_ids[i]`` is the
    flat cell index for ``pos[i]`` and ``grid_shape`` is the per-axis
    cell count.
    """
    extents = pos.max(axis=0) - mins
    grid_shape = tuple(
        max(1, int(np.ceil(extents[a] / cell_size)) + 1) for a in range(3)
    )
    cells = np.clip(
        ((pos - mins) / cell_size).astype(np.intp),
        0,
        np.array(grid_shape, dtype=np.intp) - 1,
    )
    cell_ids = (
        cells[:, 0] * grid_shape[1] * grid_shape[2]
        + cells[:, 1] * grid_shape[2]
        + cells[:, 2]
    ).astype(np.intp)
    return cell_ids, grid_shape


def _select_blue_noise_subset(
    pos: NDArray,
    candidate_order: NDArray[np.intp],
    r: float,
    mins: NDArray,
    available_mask: NDArray[np.bool_],
) -> NDArray[np.intp]:
    """Greedy single-pass Bridson-style selection over already-existing samples.

    Walks ``candidate_order`` (the in-order rejection sampling sequence)
    and accepts each candidate iff no previously-accepted sample is
    within radius ``r``. Uses a uniform cell grid sized at ``r / sqrt(3)``
    so each rejection test scans only the local 5·5·5 neighborhood.

    Args:
        pos: ``(N, 3)`` array of all original positions.
        candidate_order: int array of original indices in the order the
            sampler will consider them (already permuted by caller).
        r: Minimum pairwise distance for accepted samples.
        mins: ``(3,)`` array — bbox lower-left used to map positions to
            cell coordinates.
        available_mask: ``(N,)`` bool — which positions are still
            unassigned at the start (already-taken samples cannot be
            re-accepted, but DO count as occupied neighbors so the new
            level respects the coarser levels' radii too).

    Returns:
        ``(K,)`` int array of original indices accepted at this level,
        in acceptance order.
    """
    n = pos.shape[0]
    if n == 0 or r <= 0:
        return np.empty(0, dtype=np.intp)

    cell_size = r / np.sqrt(3.0)
    cell_ids, grid_shape = _build_cell_grid(pos, mins, cell_size)
    accepted = np.zeros(n, dtype=np.bool_)
    # Pre-seed accepted with positions that were taken by coarser levels —
    # they enforce radius-r exclusion at this finer level too.
    accepted[~available_mask] = True

    # cell_buckets[c] = list of original indices in cell c. Built once.
    buckets: dict[int, list[int]] = {}
    for i, c in enumerate(cell_ids):
        buckets.setdefault(int(c), []).append(int(i))

    selected: list[int] = []
    r_sq = r * r
    gx, gy, gz = grid_shape

    for cand in candidate_order:
        cand = int(cand)
        if not available_mask[cand]:
            continue
        cx = cell_ids[cand] // (gy * gz)
        cy = (cell_ids[cand] // gz) % gy
        cz = cell_ids[cand] % gz
        rejected = False
        for dx in range(-2, 3):
            if rejected:
                break
            ix = int(cx) + dx
            if ix < 0 or ix >= gx:
                continue
            for dy in range(-2, 3):
                if rejected:
                    break
                iy = int(cy) + dy
                if iy < 0 or iy >= gy:
                    continue
                for dz in range(-2, 3):
                    iz = int(cz) + dz
                    if iz < 0 or iz >= gz:
                        continue
                    c = ix * gy * gz + iy * gz + iz
                    bucket = buckets.get(c)
                    if not bucket:
                        continue
                    for j in bucket:
                        if not accepted[j] or j == cand:
                            continue
                        dx_p = pos[j, 0] - pos[cand, 0]
                        dy_p = pos[j, 1] - pos[cand, 1]
                        dz_p = pos[j, 2] - pos[cand, 2]
                        if dx_p * dx_p + dy_p * dy_p + dz_p * dz_p < r_sq:
                            rejected = True
                            break
        if not rejected:
            accepted[cand] = True
            selected.append(cand)

    return np.asarray(selected, dtype=np.intp)


def poisson_disk_order(
    positions: NDArray,
    n_lods: int,
    seed: int = 0,
) -> tuple[NDArray[np.intp], list[int]]:
    """Coarse-to-fine Poisson-disk LOD permutation.

    The radius at LOD level ``i`` is derived from the dataset's bbox
    diagonal and the requested ``n_lods``: the coarsest level uses the
    largest radius, finer levels halve it. Specifically, level ``i``
    uses ``r_i = (diag / 2) * 0.5^i``.

    Args:
        positions: ``(N, d)`` array. Only first 3 spatial dims are used.
        n_lods: Number of LOD levels to emit.
        seed: RNG seed for the candidate-order permutation. Determinism
            matters here because Bridson's accepted set depends on
            which candidate is considered first.

    Returns:
        ``(permutation, per_level_counts)`` matching the contract of
        :func:`luxar.core.group.lod.spatial_uniform.stratified_grid_order`. The
        last level absorbs any leftovers — points that the finest
        Bridson pass rejected (because every neighborhood was already
        occupied) so the permutation still covers every input.
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

    pos = positions[:, :3].astype(np.float64, copy=False)
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    if diag == 0:
        # Degenerate: all points coincide. Stuff everything into LOD 0.
        perm = np.arange(n, dtype=np.intp)
        counts = [n] + [0] * (n_lods - 1)
        return perm, counts

    rng = np.random.RandomState(seed)
    candidate_order = rng.permutation(n).astype(np.intp)
    available = np.ones(n, dtype=np.bool_)

    per_level_indices: list[NDArray[np.intp]] = []
    base_r = diag / 2.0

    for level in range(n_lods):
        if level < n_lods - 1:
            r = base_r * (0.5**level)
            picked = _select_blue_noise_subset(pos, candidate_order, r, mins, available)
        else:
            # Last level: absorb everything still available so the
            # permutation covers every input.
            picked = candidate_order[available[candidate_order]]
        per_level_indices.append(picked)
        available[picked] = False

    perm_parts = [arr for arr in per_level_indices if arr.size > 0]
    perm = np.concatenate(perm_parts) if perm_parts else np.empty(0, dtype=np.intp)
    counts = [int(arr.size) for arr in per_level_indices]
    return perm, counts
