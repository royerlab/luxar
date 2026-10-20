# peak_finding.py
"""Residual peak finding for dynamic seeding operations."""

from __future__ import annotations

import math
import random
from typing import Tuple

import torch


def _separable_nd_max_pool(
    data: torch.Tensor, kernel_size: int, start_dim: int = 0
) -> torch.Tensor:
    """
    Apply separable n-dimensional max pooling using 1D max pooling along each dimension.

    This is an efficient implementation for >3D data where native max_pool functions
    don't exist. The approach applies 1D max pooling sequentially along each dimension,
    which is equivalent to a full nD max pool for local maxima detection.

    Args:
        data: Input tensor of any dimensionality
        kernel_size: Size of the max pooling kernel (same for all dimensions)
        start_dim: First dimension to pool; earlier dimensions are batch axes
            (pooling never mixes values across them)

    Returns:
        Max-pooled tensor with same shape as input
    """
    result = data
    d = data.ndim
    padding = kernel_size // 2

    # Apply 1D max pooling along each dimension sequentially
    for dim in range(start_dim, d):
        # Move the current dimension to the last position for max_pool1d
        # max_pool1d expects input of shape (N, C, L) - we use (1, 1, L) for each slice
        perm = list(range(d))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        result = result.permute(*perm)

        # Store original shape (with swapped dimensions)
        original_shape = result.shape

        # Reshape to (batch, 1, length) for max_pool1d
        # Flatten all dimensions except the last one as batch
        batch_size = result[..., 0].numel()
        length = original_shape[-1]
        result = result.reshape(batch_size, 1, length)

        # Apply 1D max pooling
        result = torch.nn.functional.max_pool1d(
            result, kernel_size=kernel_size, stride=1, padding=padding
        )

        # Handle edge case where output size differs slightly from input
        if result.shape[-1] != length:
            # Truncate or pad to match original size
            if result.shape[-1] > length:
                result = result[..., :length]
            else:
                pad_size = length - result.shape[-1]
                result = torch.nn.functional.pad(
                    result, (0, pad_size), mode="replicate"
                )

        # Reshape back and reverse permutation
        result = result.reshape(original_shape)
        inv_perm = [0] * d
        for i, p in enumerate(perm):
            inv_perm[p] = i
        result = result.permute(*inv_perm)

    return result


def _find_residual_peaks(
    residual: torch.Tensor,
    k_max_residuals: int,
    nms_radius_vox: float,
    enable_tiled: bool = False,
    num_tiles_per_dim: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Find k strongest residual peaks with spatial exclusion (non-maximum suppression).

    IMPORTANT: Only considers POSITIVE residuals (undershoot locations where
    target > prediction). Negative residuals (overshoot) are ignored because
    Gaussian splats can only add to the prediction, not subtract. Relocating
    a splat to an overshoot location would make the error worse.

    Supports two modes:
    - Global mode (enable_tiled=False): Find top k_max_residuals peaks globally
    - Tiled mode (enable_tiled=True): Divide into tiles, use probabilistic/deterministic
      selection to maintain expected count of k_max_residuals peaks

    Args:
        residual: Residual image (target - prediction)
        k_max_residuals: Expected number of peaks to return (controls both modes)
        nms_radius_vox: Minimum distance between peaks
        enable_tiled: If True, use tile-based seeding for spatial fairness
        num_tiles_per_dim: Number of tiles per dimension (e.g., 8 → 8×8=64 tiles
            for 2D, 8³=512 for 3D). None → the auto rule of
            :func:`auto_tiles_per_dim` (16 / 6 / 4 / 2 for 2D / 3D / 4D / 5D+).
        seed: RNG seed for the probabilistic per-tile keep decision (tiled mode
            only, when k_per_tile < 1). None → nondeterministic. Ignored in
            global mode, which is fully deterministic.

    Returns:
        Tensor of peak coordinates, shape (K, d), on same device as residual

    Notes:
        In tiled mode, k_per_tile = k_max_residuals / num_tiles is auto-calculated:
        - If k_per_tile >= 1: Deterministically keep floor(k_per_tile) per tile
        - If k_per_tile < 1: Probabilistically keep each with probability k_per_tile
        This ensures expected count ≈ k_max_residuals regardless of mode.
    """
    if enable_tiled:
        return _find_residual_peaks_tiled(
            residual, num_tiles_per_dim, k_max_residuals, nms_radius_vox, seed=seed
        )
    else:
        return _find_residual_peaks_global(residual, k_max_residuals, nms_radius_vox)


def _find_residual_peaks_global(
    residual: torch.Tensor, k_max_residuals: int, nms_radius_vox: float
) -> torch.Tensor:
    """
    Find k strongest residual peaks globally with spatial exclusion (original method).

    This is the original global peak finding - all regions compete for top k spots.
    Bright/high-residual regions tend to dominate.

    IMPORTANT: Only considers POSITIVE residuals (undershoot locations where
    target > prediction). Negative residuals (overshoot) are ignored because
    Gaussian splats can only add to the prediction, not subtract. Relocating
    a splat to an overshoot location would make the error worse.

    Args:
        residual: Residual image (target - prediction)
        k_max_residuals: Number of peaks to find globally
        nms_radius_vox: Minimum distance between peaks

    Returns:
        Tensor of peak coordinates, shape (K, d), on same device as residual
    """
    # Only consider positive residuals (undershoot: target > prediction)
    # Negative residuals mean overshoot - adding more splats would make it worse
    residual_positive = torch.clamp(residual, min=0)
    d = residual.ndim

    # Create kernel for non-maximum suppression
    kernel_size = int(2 * nms_radius_vox + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply max pooling for NMS
    # Note: max_pool3d is not implemented for MPS, so we use CPU fallback
    if d == 2:
        max_pooled = torch.nn.functional.max_pool2d(
            residual_positive[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0, 0]
    elif d == 3:
        if residual_positive.device.type == "mps":
            # Use separable max pool on MPS (avoids CPU transfer)
            # This is 2-5x faster than transferring to CPU for max_pool3d
            max_pooled = _separable_nd_max_pool(residual_positive, kernel_size)
        else:
            # Native max_pool3d for CUDA/CPU
            max_pooled = torch.nn.functional.max_pool3d(
                residual_positive[None, None],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )[0, 0]
    else:
        # nD NMS using separable 1D max pooling along each dimension
        # This is efficient and works for any dimensionality
        max_pooled = _separable_nd_max_pool(residual_positive, kernel_size)

    # Find local maxima
    is_peak = (residual_positive >= max_pooled) & (residual_positive > 0)
    peak_indices = torch.nonzero(is_peak, as_tuple=False)

    if len(peak_indices) == 0:
        return torch.zeros((0, d), dtype=torch.long, device=residual.device)

    # Get values and sort by magnitude
    peak_values = residual_positive[tuple(peak_indices.T)]
    sorted_indices = torch.sort(peak_values, descending=True, stable=True)[1]

    # Take top k
    top_k = min(k_max_residuals, len(sorted_indices))
    selected_peaks = peak_indices[sorted_indices[:top_k]]

    return selected_peaks


#: Auto ``num_tiles_per_dim`` by dimensionality (``None`` in the config).
_AUTO_TILES_PER_DIM = {2: 16, 3: 6, 4: 4}
_AUTO_TILES_PER_DIM_HIGH_D = 2

#: Gathered-tile elements processed per batch in tiled peak finding. Peak
#: temporaries per element: the int64 gather index (8 B), the gathered tiles
#: (4 B), the pooled copy (4 B), three bool masks (3 B) and the masked values
#: (4 B) -- about 23 B, so ~190 MB per batch; a 256x512x512 residual is ~8
#: batches, all kept on the device (no host sync between them).
_TILE_BATCH_ELEMENTS = 1 << 23


def auto_tiles_per_dim(d: int) -> int:
    """The documented auto tile count: 16 (2D), 6 (3D), 4 (4D), 2 (5D+)."""
    return _AUTO_TILES_PER_DIM.get(d, _AUTO_TILES_PER_DIM_HIGH_D)


def _tile_nms_kernel_size(tile_shape: Tuple[int, ...], nms_radius_vox: float) -> int:
    """Odd NMS kernel size for a tile, clamped to the tile's smallest side."""
    kernel_size = int(2 * nms_radius_vox + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    min_tile_dim = min(tile_shape)
    kernel_size = min(kernel_size, min_tile_dim)
    # Only enforce minimum if tile is large enough
    if min_tile_dim >= 3 and kernel_size < 3:
        kernel_size = 3  # Minimum 3x3(x3) kernel when tile allows it
    elif kernel_size % 2 == 0:
        kernel_size = max(1, kernel_size - 1)  # Ensure odd kernel
    return kernel_size


def _batched_tile_max_pool(tiles: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Per-tile max pool of a ``(T, *tile_shape)`` batch (padding is per tile).

    Same backend choice as the global path: native 2D/3D pooling, except the
    separable 1D passes on MPS (no ``max_pool3d`` there) and above 3D.
    """
    d = tiles.ndim - 1
    pad = kernel_size // 2
    if d == 2:
        return torch.nn.functional.max_pool2d(
            tiles[:, None], kernel_size=kernel_size, stride=1, padding=pad
        )[:, 0]
    if d == 3 and tiles.device.type != "mps":
        return torch.nn.functional.max_pool3d(
            tiles[:, None], kernel_size=kernel_size, stride=1, padding=pad
        )[:, 0]
    # MPS 3D and >3D: separable passes over the tile axes only (axis 0 is the
    # tile batch, which must not be pooled across).
    return _separable_nd_max_pool(tiles, kernel_size, start_dim=1)


def _tile_grid(
    shape: Tuple[int, ...], n: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], torch.Tensor]:
    """Linear tile origins, in-tile offsets, tile shape and per-tile origins.

    Tile ``t`` along an axis of size ``s`` starts at ``s * t // n`` and spans
    ``max(1, s // n)`` voxels, so every tile has the same shape (the end clamp
    of the old per-tile loop never bound). Tiles are enumerated in
    ``itertools.product`` order -- the order the probabilistic keep draws are
    consumed in.
    """
    d = len(shape)
    tile_shape = tuple(max(1, s // n) for s in shape)
    starts = [
        torch.tensor([(s * t) // n for t in range(n)], dtype=torch.long) for s in shape
    ]
    origins = torch.cartesian_prod(*starts).reshape(-1, d)  # (T, d)
    strides = torch.tensor(
        [math.prod(shape[i + 1 :]) for i in range(d)], dtype=torch.long
    )
    offsets = torch.cartesian_prod(
        *[torch.arange(ts, dtype=torch.long) for ts in tile_shape]
    ).reshape(-1, d)
    return (
        (origins @ strides).to(device),
        (offsets @ strides).to(device),
        tile_shape,
        origins.to(device),
    )


def _strong_peak_threshold(residual_positive: torch.Tensor) -> float:
    """75th percentile of the positive residual (strided sample)."""
    # Strided sampling: zero-copy view, no allocation, no GPU kernel
    # (torch.quantile() is limited to ~16M elements: pytorch/pytorch#64947)
    max_samples = 2**20  # ~1M is plenty for a percentile estimate
    stride = max(1, residual_positive.numel() // max_samples)
    sampled = residual_positive.reshape(-1)[::stride]
    positive_sample = sampled[sampled > 0]
    if len(positive_sample) == 0:
        return float("inf")  # No positive residuals
    return float(torch.quantile(positive_sample.float(), 0.75).item())


def _top_k_stable(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-wise top-``k`` of a 2D tensor with ties broken by lowest index.

    ``torch.topk`` leaves the order of equal values unspecified, and it does
    differ between CPU backends (x86 vs arm64), which made relocation peaks --
    and so whole fits -- platform-dependent. ``max`` is documented to return
    the first maximum. For ``k > 1`` the ``topk`` SET is already unique unless
    a row has more than ``k`` elements >= its k-th value; the set is then just
    re-ordered by (value desc, index asc), a sort of only ``k`` columns. Rows
    that do tie at the k-th value instead take everything above it plus the
    lowest-index tied elements. ``-inf`` is padding (non-peaks the caller
    drops), so a row whose k-th value is ``-inf`` does not count as tied and
    its padding indices are unspecified. The extra passes cost about 2-3x a
    bare ``topk`` without ties and 3.5-8x with frequent k-th-value ties
    (milliseconds per relocation step); a full stable row sort was ~20x.
    """
    if k == 1:
        vals, idx = x.max(dim=1, keepdim=True)
        return vals, idx
    vals, idx = torch.topk(x, k, dim=1)
    kth = vals[:, -1:]
    tied_rows = ((x >= kth).sum(dim=1) > k) & (kth[:, 0] > -math.inf)
    if bool(tied_rows.any()):
        above = x > kth
        tied = x == kth
        slots = k - above.sum(dim=1, keepdim=True)
        chosen = above | (tied & (torch.cumsum(tied, dim=1) <= slots))
        idx = torch.nonzero(chosen)[:, 1].view(x.shape[0], k)  # ascending per row
    else:
        idx = idx.sort(dim=1).values
    order = torch.sort(torch.gather(x, 1, idx), dim=1, descending=True, stable=True)
    return order.values, torch.gather(idx, 1, order.indices)


def _top_tile_peaks(
    residual_positive: torch.Tensor,
    num_tiles_per_dim: int,
    k: int,
    nms_radius_vox: float,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """Top-``k`` NMS peaks of every tile, all tiles at once.

    Returns ``(values, coords)`` of shapes ``(T, k)`` and ``(T, k, d)`` in
    global voxel coordinates, with ``values == -inf`` where a tile has fewer
    than ``k`` peaks; ``None`` when tiles are too small to search (< 4
    voxels, as before). Everything stays on the device: the tiles are
    gathered in batches into ``(T_batch, *tile_shape)``, max-pooled once per
    batch and reduced with one stable top-``k`` -- versus one pool, ``nonzero`` and
    ``.tolist()`` host sync per tile in the old loop (~2,500 blocking syncs
    per relocation step on a 3D volume).
    """
    shape = tuple(residual_positive.shape)
    device = residual_positive.device
    tile_base, tile_offsets, tile_shape, origins = _tile_grid(
        shape, num_tiles_per_dim, device
    )
    P = tile_offsets.numel()
    if P < 4:
        return None
    k = min(k, P)
    kernel_size = _tile_nms_kernel_size(tile_shape, nms_radius_vox)
    flat = residual_positive.reshape(-1)
    batch = max(1, _TILE_BATCH_ELEMENTS // P)
    values_parts, local_parts = [], []
    for t0 in range(0, tile_base.numel(), batch):
        lin = tile_base[t0 : t0 + batch, None] + tile_offsets[None, :]
        tiles = flat[lin].view(-1, *tile_shape)
        pooled = _batched_tile_max_pool(tiles, kernel_size)
        is_peak = (tiles >= pooled) & (tiles > 0)
        masked = tiles.masked_fill(~is_peak, -math.inf)
        vals, local = _top_k_stable(masked.reshape(tiles.shape[0], P), k)
        values_parts.append(vals)
        local_parts.append(local)
    values = torch.cat(values_parts)
    local = torch.cat(local_parts)  # (T, k) flat in-tile index
    tile_strides = torch.tensor(
        [math.prod(tile_shape[i + 1 :]) for i in range(len(shape))],
        dtype=torch.long,
        device=device,
    )
    tile_dims = torch.tensor(tile_shape, dtype=torch.long, device=device)
    local_coords = (local[..., None] // tile_strides) % tile_dims  # (T, k, d)
    return values, local_coords + origins[:, None, :]


def _find_residual_peaks_tiled(
    residual: torch.Tensor,
    num_tiles_per_dim: int | None,
    k_max_residuals: int,
    nms_radius_vox: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Find residual peaks using tile-based approach for spatial fairness.

    Divides image into tiles and uses probabilistic/deterministic selection to
    maintain expected count of k_max_residuals peaks while ensuring spatial fairness.

    IMPORTANT: Only considers POSITIVE residuals (undershoot locations where
    target > prediction). Negative residuals (overshoot) are ignored because
    Gaussian splats can only add to the prediction, not subtract. Relocating
    a splat to an overshoot location would make the error worse.

    Args:
        residual: Residual image (target - prediction)
        num_tiles_per_dim: Number of tiles per dimension. If None, auto-selected
            (:func:`auto_tiles_per_dim`):
            2D: 16 (16×16 = 256 tiles)
            3D: 6 (6×6×6 = 216 tiles)
            4D: 4 (4⁴ = 256 tiles)
            5D+: 2 (2^d tiles)
        k_max_residuals: Expected total number of peaks to return
        nms_radius_vox: Minimum distance between peaks (applied within each tile)

    Returns:
        Tensor of peak coordinates, shape (K, d), on same device as residual

    Notes:
        - k_per_tile = k_max_residuals / total_tiles is auto-calculated
        - If k_per_tile >= 1: Keep floor(k_per_tile) peaks per tile (deterministic)
        - If k_per_tile < 1: Keep each peak with probability k_per_tile (random),
          but strong peaks (above the 75th percentile) are always kept
        - Expected total ≈ k_max_residuals regardless of tiling
        - Randomness ensures fairness (no bias toward bright regions)
        - Each tile processed independently (spatial fairness guaranteed)
        - All tiles are searched in one batched pass (:func:`_top_tile_peaks`);
          the probabilistic mode makes ONE host transfer of the per-tile best
          values so the keep draws are consumed in the same tile order, from
          the same seeded RNG, as the old per-tile loop. Peaks of exactly
          equal value are chosen and ordered by lowest index (stable), which
          the old unstable per-tile ``argsort`` did not guarantee.
    """
    # Only consider positive residuals (undershoot: target > prediction)
    # Negative residuals mean overshoot - adding more splats would make it worse
    residual_positive = torch.clamp(residual, min=0)
    d = residual.ndim
    empty = torch.zeros((0, d), dtype=torch.long, device=residual.device)

    if num_tiles_per_dim is None:
        num_tiles_per_dim = auto_tiles_per_dim(d)
    k_per_tile_float = k_max_residuals / num_tiles_per_dim**d
    use_probabilistic = k_per_tile_float < 1.0
    k = 1 if use_probabilistic else int(k_per_tile_float)

    found = _top_tile_peaks(residual_positive, num_tiles_per_dim, k, nms_radius_vox)
    if found is None:
        return empty
    values, coords = found
    keep = torch.isfinite(values)
    if use_probabilistic:
        # Strong peaks are always kept; each weak one survives with probability
        # k_per_tile. The draw is made only for tiles with a weak peak, in tile
        # order, so a seeded run selects what the per-tile loop selected.
        strong = _strong_peak_threshold(residual_positive)
        rng = random.Random(seed)
        best = values[:, 0].cpu().tolist()  # the one host transfer
        mask = [
            math.isfinite(v) and (v >= strong or rng.random() < k_per_tile_float)
            for v in best
        ]
        keep = keep & torch.tensor(mask, device=values.device)[:, None]

    peaks, peak_values = coords[keep], values[keep]
    if peaks.numel() == 0:
        return empty
    # Sort peaks by residual magnitude (descending) to match global mode
    # This ensures convergence guard checks the actual strongest peak
    return peaks[torch.sort(peak_values, descending=True, stable=True)[1]]
