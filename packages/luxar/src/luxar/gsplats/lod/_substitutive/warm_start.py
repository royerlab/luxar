"""Morton (Z-order) space-filling-curve warm start for substitutive LOD.

The ``kmeans*`` methods of :func:`luxar.gsplats.lod.substitutive.make_substitutive_lod`
start from an ``O(N log N)`` Morton-order partition: sort splats along a
Z-order curve and chunk the sorted sequence into ``M = N/K`` contiguous,
balanced, spatially-coherent bins. This replaces a global k-means++ warm
start whose initialisation was ``O(M·N) = O(N²/K)`` — intractable once
``M = N/K`` reached tens of thousands (the substitutive regime).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from luxar.io.ordering import morton_encode_nd, normalize_coords_to_grid


def _morton_order(centres: torch.Tensor) -> np.ndarray:
    """Indices that sort splats along a Morton (Z-order) space-filling curve.

    Returns a length-N int64 numpy array ``order`` such that
    ``centres[order]`` is spatially coherent (neighbours on the curve are
    spatial neighbours). ``O(N log N)``.
    """
    coords = centres.detach().cpu().numpy().astype(np.float64)
    ndim = coords.shape[1]
    bits_per_dim = min(21, 64 // max(ndim, 1))
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    grid = normalize_coords_to_grid(coords, min_c, max_c, 2**bits_per_dim)
    codes = morton_encode_nd(grid, bits_per_dim)
    return np.argsort(codes, kind="stable").astype(np.int64)


def _morton_partition(
    centres: torch.Tensor, M: int, *, order: Optional[np.ndarray] = None
) -> torch.Tensor:
    """O(N log N) space-filling-curve warm start.

    Sort splats along a Morton (Z-order) curve and chunk the sorted
    sequence into ``M`` contiguous, balanced bins of ~``N/M`` splats. The
    result is spatially coherent and empty-bin-free for ``N >= M`` (every
    bin receives at least one splat). Replaces the former global k-means++
    warm start whose initialisation was ``O(M·N) = O(N²/K)`` and therefore
    intractable in the substitutive regime where ``M = N/K`` is large.

    Returns a length-N int64 assignment tensor with values in ``[0, M)``.
    """
    N = centres.shape[0]
    device = centres.device
    M = min(M, N)
    if M <= 1 or N <= 1:
        return torch.zeros(N, dtype=torch.int64, device=device)
    if order is None:
        order = _morton_order(centres)
    # Sorted position p → bin floor(p*M/N): balanced and surjective onto [0, M).
    bin_of_pos = (np.arange(N, dtype=np.int64) * M) // N
    assignments = np.empty(N, dtype=np.int64)
    assignments[order] = bin_of_pos
    return torch.from_numpy(assignments).to(device=device)
