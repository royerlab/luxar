"""
Core CLAHE (Contrast Limited Adaptive Histogram Equalization) implementation.

This module implements CLAHE for arbitrary-dimensional tensors with PyTorch,
providing GPU acceleration and numerical stability.
"""

import itertools
from typing import Tuple

import torch


def apply_clahe(
    V: torch.Tensor,
    tile_size: int = 16,
    clip_limit: float = 2.0,
    nbins: int = 256,
) -> torch.Tensor:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to nD volume.

    CLAHE [CLAHE1994]_ enhances local contrast by performing histogram
    equalization on small tiles, then applying contrast limiting to prevent
    noise amplification.

    Algorithm:

    1. Divide volume into non-overlapping tiles of size tile_size^d
    2. For each tile:

       a. Compute local histogram (nbins bins)
       b. Apply contrast limiting (clip histogram peaks)
       c. Compute CDF mapping (local histogram equalization)
       d. Transform tile intensities

    3. Result: Volume with locally-equalized contrast

    Parameters
    ----------
    V : torch.Tensor
        Input tensor of any dimensionality (1D, 2D, 3D, nD)
    tile_size : int, default=16
        Size of tiles in voxels. Tiles are tile_size^d hypercubes.
        - Too small (< 8): Overfits to noise, over-amplifies uniform regions
        - Too large (> 32): Loses local adaptation, approaches global equalization
        - Recommended: ~2× typical feature diameter
    clip_limit : float, default=2.0
        Contrast limiting factor (range: 1.0-4.0)
        - Low (1.0-2.0): Conservative, closer to original distribution, less noise
        - High (3.0-4.0): Aggressive equalization, more noise amplification
        - Formula: max_histogram_height = clip_limit × (n_pixels_per_tile / nbins)
    nbins : int, default=256
        Number of histogram bins for equalization
        - Too few (< 64): Coarse equalization, loses detail
        - Too many (> 512): Computational cost, no benefit
        - Standard: 256 for 8-16 bit images

    Returns
    -------
    torch.Tensor
        CLAHE-equalized volume with same shape and device as input.
        Intensity range preserved (same min/max as input).

    Examples
    --------
    >>> import torch
    >>> from luxar.gsplats.clahe import apply_clahe
    >>>
    >>> # 2D image with heterogeneous background
    >>> image = torch.randn(256, 256)
    >>> enhanced = apply_clahe(image, tile_size=16, clip_limit=2.0)
    >>>
    >>> # 3D volume
    >>> volume = torch.randn(128, 128, 128)
    >>> enhanced_3d = apply_clahe(volume, tile_size=16, clip_limit=2.0)
    >>>
    >>> # Higher dimensions
    >>> data_4d = torch.randn(64, 64, 64, 64)
    >>> enhanced_4d = apply_clahe(data_4d, tile_size=8, clip_limit=1.5)

    Notes
    -----
    - Output preserves input dtype and device
    - Tiles at boundaries may be smaller than tile_size
    - No interpolation between tiles (for speed and simplicity)
    - For sampling applications, discontinuities are acceptable
    - For visualization, consider adding bilinear/trilinear interpolation

    References
    ----------
    .. [CLAHE1994] Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram
       Equalization." Graphics Gems IV, Academic Press.
    """
    shape = V.shape
    device = V.device
    dtype = V.dtype

    # Calculate number of tiles per dimension
    n_tiles = tuple((s + tile_size - 1) // tile_size for s in shape)

    # Create output tensor
    V_clahe = torch.zeros_like(V)

    # Get global min/max for consistent binning across tiles
    V_min, V_max = V.min().item(), V.max().item()

    if V_max - V_min < 1e-12:
        # Uniform image - return unchanged
        return V.clone()

    # Pre-compute bin edges once (invariant across tiles)
    bin_edges = torch.linspace(V_min, V_max, nbins + 1, device=device)

    # For each tile, compute local histogram equalization
    for tile_idx in itertools.product(*[range(n) for n in n_tiles]):
        # Extract tile boundaries
        tile_slice = tuple(
            slice(t * tile_size, min((t + 1) * tile_size, s))
            for t, s in zip(tile_idx, shape)
        )

        # Get tile data
        tile_data = V[tile_slice]
        tile_flat = tile_data.reshape(-1)

        # Compute histogram
        hist = torch.histc(tile_flat, bins=nbins, min=V_min, max=V_max)

        # Apply contrast limiting
        uniform_height = tile_flat.numel() / nbins
        clip_height = clip_limit * uniform_height
        excess = torch.clamp(hist - clip_height, min=0).sum()
        hist = torch.clamp(hist, max=clip_height)
        hist += excess / nbins  # Redistribute clipped pixels uniformly

        # Compute CDF
        cdf = torch.cumsum(hist, dim=0)
        cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
        cdf_range = cdf[-1] - cdf_min

        if cdf_range > 0:
            cdf_normalized = (cdf - cdf_min) / cdf_range
        else:
            cdf_normalized = cdf

        # Map tile intensities through CDF
        bin_indices = torch.searchsorted(bin_edges[1:], tile_flat.contiguous())
        bin_indices = torch.clamp(bin_indices, 0, nbins - 1)

        # Apply CDF mapping
        tile_equalized = cdf_normalized[bin_indices]

        # Reshape and store
        V_clahe[tile_slice] = tile_equalized.reshape(tile_data.shape)

    # Rescale to original range for consistency
    V_clahe = V_clahe * (V_max - V_min) + V_min

    return V_clahe.to(dtype=dtype)


def compute_clahe_sampling_probabilities(
    V: torch.Tensor,
    tile_size: int = 16,
    clip_limit: float = 2.0,
    nbins: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalized sampling probabilities from CLAHE-equalized volume.

    This is a convenience function for using CLAHE output as a probability
    distribution for importance sampling, commonly used in dynamic operations.

    Parameters
    ----------
    V : torch.Tensor
        Input volume
    tile_size : int, default=16
        CLAHE tile size
    clip_limit : float, default=2.0
        CLAHE contrast limiting factor
    nbins : int, default=256
        CLAHE histogram bins

    Returns
    -------
    probabilities : torch.Tensor
        Flattened probability distribution summing to 1.0, shape (V.numel(),)
    V_clahe : torch.Tensor
        CLAHE-equalized volume (for visualization/debugging)

    Examples
    --------
    >>> import torch
    >>> from luxar.gsplats.clahe import compute_clahe_sampling_probabilities
    >>>
    >>> image = torch.randn(256, 256)
    >>> probs, enhanced = compute_clahe_sampling_probabilities(image)
    >>>
    >>> # Sample locations proportionally to local importance
    >>> k_samples = 100
    >>> sampled_indices = torch.multinomial(probs, k_samples, replacement=True)
    """
    # Apply CLAHE
    V_clahe = apply_clahe(V, tile_size=tile_size, clip_limit=clip_limit, nbins=nbins)

    # Normalize to [0, 1] for probability distribution
    V_min, V_max = V_clahe.min(), V_clahe.max()
    if V_max - V_min < 1e-12:
        # Uniform - use uniform probabilities
        V_norm = torch.ones_like(V_clahe)
    else:
        V_norm = (V_clahe - V_min) / (V_max - V_min)

    # Flatten and normalize to valid probability distribution
    V_flat = V_norm.reshape(-1)
    prob_sum = V_flat.sum()

    if prob_sum < 1e-12:
        # Degenerate case - uniform probabilities
        probabilities = torch.ones_like(V_flat) / V_flat.numel()
    else:
        probabilities = V_flat / prob_sum

    return probabilities, V_clahe
