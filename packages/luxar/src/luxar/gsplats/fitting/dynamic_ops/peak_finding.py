# peak_finding.py
"""Residual peak finding for dynamic seeding operations."""

from __future__ import annotations

import itertools
import random
from typing import List, Tuple

import torch


def _separable_nd_max_pool(data: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply separable n-dimensional max pooling using 1D max pooling along each dimension.

    This is an efficient implementation for >3D data where native max_pool functions
    don't exist. The approach applies 1D max pooling sequentially along each dimension,
    which is equivalent to a full nD max pool for local maxima detection.

    Args:
        data: Input tensor of any dimensionality
        kernel_size: Size of the max pooling kernel (same for all dimensions)

    Returns:
        Max-pooled tensor with same shape as input
    """
    result = data
    d = data.ndim
    padding = kernel_size // 2

    # Apply 1D max pooling along each dimension sequentially
    for dim in range(d):
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
    num_tiles_per_dim: int = 8,
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
        num_tiles_per_dim: Number of tiles per dimension (e.g., 8 → 8×8=64 tiles for 2D, 8³=512 for 3D)
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
    sorted_indices = torch.argsort(peak_values, descending=True)

    # Take top k
    top_k = min(k_max_residuals, len(sorted_indices))
    selected_peaks = peak_indices[sorted_indices[:top_k]]

    return selected_peaks


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
        num_tiles_per_dim: Number of tiles per dimension. If None, auto-selected:
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
          but strong peaks (above global median) are always kept
        - Expected total ≈ k_max_residuals regardless of tiling
        - Randomness ensures fairness (no bias toward bright regions)
        - Each tile processed independently (spatial fairness guaranteed)
    """
    # Only consider positive residuals (undershoot: target > prediction)
    # Negative residuals mean overshoot - adding more splats would make it worse
    residual_positive = torch.clamp(residual, min=0)
    shape = residual.shape
    d = residual.ndim

    # Local RNG for the probabilistic per-tile keep decision below. Seeding
    # makes weak-peak selection reproducible run-to-run; a private instance
    # avoids perturbing (and being perturbed by) the global random state.
    rng = random.Random(seed)

    # Auto-select num_tiles_per_dim based on dimensionality
    if num_tiles_per_dim is None:
        if d == 2:
            num_tiles_per_dim = 16  # 16×16 = 256 tiles
        elif d == 3:
            num_tiles_per_dim = 6  # 6×6×6 = 216 tiles
        elif d == 4:
            num_tiles_per_dim = 4  # 4⁴ = 256 tiles
        else:
            num_tiles_per_dim = 2  # 2^d tiles

    # Calculate k_per_tile from k_max_residuals and total tiles
    total_tiles = num_tiles_per_dim**d
    k_per_tile_float = k_max_residuals / total_tiles

    # Determine selection mode
    use_probabilistic = k_per_tile_float < 1.0
    if use_probabilistic:
        # Probabilistic: each peak kept with probability k_per_tile
        keep_probability = k_per_tile_float
        k_deterministic = None
        # Calculate threshold for "strong" peaks that should always be kept
        # Use 75th percentile of positive residual as threshold
        # Strided sampling: zero-copy view, no allocation, no GPU kernel
        # (torch.quantile() is limited to ~16M elements: pytorch/pytorch#64947)
        max_samples = 2**20  # ~1M is plenty for a percentile estimate
        n = residual_positive.numel()
        stride = max(1, n // max_samples)
        sampled = residual_positive.reshape(-1)[::stride]
        positive_sample = sampled[sampled > 0]
        if len(positive_sample) > 0:
            strong_peak_threshold = torch.quantile(positive_sample.float(), 0.75).item()
        else:
            strong_peak_threshold = float("inf")  # No positive residuals
    else:
        # Deterministic: keep floor(k_per_tile) peaks per tile
        k_deterministic = int(k_per_tile_float)
        keep_probability = None
        strong_peak_threshold = float("inf")  # Not used in deterministic mode

    all_peaks = []

    # Calculate tile size for each dimension
    # Divide each dimension into num_tiles_per_dim equal parts
    tile_sizes = [max(1, dim_size // num_tiles_per_dim) for dim_size in shape]

    # Generate tile grid coordinates (start positions)
    tile_starts = []
    for i, dim_size in enumerate(shape):
        # Create num_tiles_per_dim evenly-spaced starting positions
        starts = [
            (dim_size * tile_idx) // num_tiles_per_dim
            for tile_idx in range(num_tiles_per_dim)
        ]
        tile_starts.append(starts)

    # Process each tile
    for tile_coords in itertools.product(*tile_starts):
        # Calculate tile boundaries
        slices = []
        tile_origin = []
        for i, start in enumerate(tile_coords):
            # Calculate end position for this tile
            # Extend to next tile start, or image boundary if last tile
            end = min(start + tile_sizes[i], shape[i])

            slices.append(slice(start, end))
            tile_origin.append(start)

        # Extract tile
        tile = residual_positive[tuple(slices)]

        # Skip very small tiles
        if tile.numel() < 4:
            continue

        # Find peaks within this tile using NMS
        if use_probabilistic:
            # Find best peak in tile
            tile_peaks = _find_peaks_in_tile(
                tile, k_max=1, nms_radius_vox=nms_radius_vox
            )
            if tile_peaks:
                # Get peak value to check if it's a strong peak
                peak_coords = tile_peaks[0]
                peak_value = tile[peak_coords].item()

                # Strong peaks are always kept, others use probabilistic selection
                if peak_value >= strong_peak_threshold:
                    # Keep strong peak unconditionally
                    pass
                elif rng.random() >= keep_probability:
                    # Reject weak peak based on probability
                    tile_peaks = []
        else:
            # Deterministic: keep top k_deterministic peaks per tile
            tile_peaks = _find_peaks_in_tile(
                tile, k_max=k_deterministic, nms_radius_vox=nms_radius_vox
            )

        # Convert local tile coordinates to global image coordinates
        for local_coords in tile_peaks:
            global_coords = torch.tensor(
                [local_coords[i] + tile_origin[i] for i in range(d)],
                dtype=torch.long,
                device=residual.device,
            )
            all_peaks.append(global_coords)

    # Sort peaks by residual magnitude (descending) to match global mode
    # This ensures convergence guard checks the actual strongest peak
    if len(all_peaks) > 0:
        peaks_tensor = torch.stack(all_peaks)  # (K, d)
        peak_residuals = residual_positive[tuple(peaks_tensor.T)]
        sorted_indices = torch.argsort(peak_residuals, descending=True)
        return peaks_tensor[sorted_indices]

    return torch.zeros((0, d), dtype=torch.long, device=residual.device)


def _find_peaks_in_tile(
    tile: torch.Tensor, k_max: int | None, nms_radius_vox: float
) -> List[Tuple[int, ...]]:
    """
    Find up to k_max peaks within a single tile using NMS.

    Args:
        tile: Tile region (positive residual values only, zeros where target <= prediction)
        k_max: Maximum number of peaks to find. If None, return all peaks.
        nms_radius_vox: NMS radius for spatial exclusion

    Returns:
        List of peak coordinates (in local tile coordinates)
    """
    d = tile.ndim

    # Create kernel for non-maximum suppression
    kernel_size = int(2 * nms_radius_vox + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Ensure kernel doesn't exceed tile size
    min_tile_dim = min(tile.shape)
    kernel_size = min(kernel_size, min_tile_dim)

    # Only enforce minimum if tile is large enough
    if min_tile_dim >= 3 and kernel_size < 3:
        kernel_size = 3  # Minimum 3×3(×3) kernel when tile allows it
    elif kernel_size % 2 == 0:
        kernel_size = max(1, kernel_size - 1)  # Ensure odd kernel

    # Apply max pooling for NMS
    # Note: max_pool3d is not implemented for MPS, so we use CPU fallback
    if d == 2:
        max_pooled = torch.nn.functional.max_pool2d(
            tile[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0, 0]
    elif d == 3:
        if tile.device.type == "mps":
            # Use separable max pool on MPS (avoids CPU transfer)
            # This is 2-5x faster than transferring to CPU for max_pool3d
            max_pooled = _separable_nd_max_pool(tile, kernel_size)
        else:
            # Native max_pool3d for CUDA/CPU
            max_pooled = torch.nn.functional.max_pool3d(
                tile[None, None],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )[0, 0]
    else:
        # Use separable n-dimensional max pooling for >3D (consistent with global mode)
        max_pooled = _separable_nd_max_pool(tile, kernel_size)

    # Find local maxima
    is_peak = (tile >= max_pooled) & (tile > 0)
    peak_indices = torch.nonzero(is_peak, as_tuple=False)

    if len(peak_indices) == 0:
        return []

    # Get values and sort by magnitude
    peak_values = tile[tuple(peak_indices.T)]
    sorted_indices = torch.argsort(peak_values, descending=True)

    # Take top k (or all if k_max is None)
    if k_max is None:
        selected_peaks = peak_indices[sorted_indices]
    else:
        top_k = min(k_max, len(sorted_indices))
        selected_peaks = peak_indices[sorted_indices[:top_k]]

    return [tuple(peak.tolist()) for peak in selected_peaks]
