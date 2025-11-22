# multiscale_gaussian.py
"""
Multiscale Gaussian seed generation for Gaussian splatting.

This module provides multiscale Gaussian-blurred peak detection with optional
CLAHE preprocessing for comprehensive feature coverage.
"""

from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage as ndi

from luxar.gsplats.seeds.utils import dedupe_farthest_first, local_maxima


def find_seeds_multiscale_gaussian(
    V: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    peaks_per_scale: Optional[int] = None,
    percentile_thresh: float = 75.0,
    min_distance: float = 2.0,
    apply_clahe: bool = True,
    clahe_tile_size: int = 32,
    clahe_clip_limit: float = 16.0,
    clahe_nbins: int = 256,
) -> np.ndarray:
    """
    Generate seed set of seed centers for Gaussian splat fitting.

    Uses multiscale Gaussian-blurred peak detection to detect blob-like structures
    at various sizes with optional CLAHE preprocessing to enhances local contrast for balanced detection/


    The resulting seed set is comprehensive across scales - the subsequent fitting
    process will select and refine the most useful subset.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to analyze.
    spacing : Sequence[float], optional
        Physical spacing between voxels along each axis. Currently unused but
        reserved for future physical-space calculations. Default is unit spacing.
    scales : Sequence[float], default=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
        Standard deviations (in voxels) for multiscale Gaussian filtering.
        Should cover the range of expected feature sizes.
    peaks_per_scale : int or None, optional
        Maximum number of peaks to detect at each scale. If None, extracts all
        peaks above threshold. Default: None (unlimited).
    percentile_thresh : float, default=70.0
        Intensity percentile threshold (0-100) for peak detection. Higher values
        are more selective, lower values detect more seeds.
    min_distance : float, default=2.0
        Minimum Euclidean distance (in voxels) between seed centers.
        Used for deduplication to avoid overly dense seeds.
    apply_clahe : bool, default=True
        Whether to apply CLAHE preprocessing before detection. When enabled,
        all detection methods operate on CLAHE-enhanced image, allowing
        discovery of dim structures in heterogeneous data.
    clahe_tile_size : int, default=32
        Tile size for CLAHE preprocessing in voxels.
    clahe_clip_limit : float, default=16.0
        Contrast limiting factor for CLAHE. Higher values provide more
        aggressive enhancement (range: 1.0-40.0, typical: 2.0-16.0).
    clahe_nbins : int, default=256
        Number of histogram bins for CLAHE equalization.

    Returns
    -------
    np.ndarray
        Array of shape (N, ndim) containing seed center coordinates in
        voxel units (float). Coordinates may be sub-voxel due to centroid refinement.

    Notes
    -----
    - When CLAHE is enabled, all processing (detection and centroid refinement)
      uses the CLAHE-enhanced image for consistency.
    - This ensures that sub-voxel refinement matches the features actually detected.
    - Candidates are processed from coarse to fine scales for better spatial distribution.
    """
    # Input validation
    V = np.asarray(V, dtype=float)
    if V.size == 0:
        raise ValueError("Input array V cannot be empty")
    if V.ndim == 0:
        raise ValueError("Input array V must have at least 1 dimension")

    d = V.ndim  # Number of spatial dimensions

    # Validate scales parameter
    if not scales or len(scales) == 0:
        raise ValueError("scales must be a non-empty sequence")
    if any(s <= 0 for s in scales):
        raise ValueError("All scale values must be positive")

    # Validate other parameters
    if peaks_per_scale is not None and peaks_per_scale <= 0:
        raise ValueError("peaks_per_scale must be positive when specified")
    if not (0 <= percentile_thresh <= 100):
        raise ValueError("percentile_thresh must be between 0 and 100")
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")

    # Physical spacing (currently unused, but reserved for future features)
    spacing = np.ones(d, float) if spacing is None else np.asarray(spacing, float)
    if len(spacing) != d:
        raise ValueError(f"spacing must have length {d} to match input dimensions")

    # CLAHE preprocessing (if enabled)
    if apply_clahe:
        import torch

        from luxar.gsplats.clahe import apply_clahe as apply_clahe_torch

        # Convert to torch, apply CLAHE, convert back
        V_torch = torch.tensor(V, dtype=torch.float32)
        V_clahe_torch = apply_clahe_torch(
            V_torch,
            tile_size=clahe_tile_size,
            clip_limit=clahe_clip_limit,
            nbins=clahe_nbins,
        )
        V_work = V_clahe_torch.cpu().numpy()
    else:
        V_work = V

    # Collect coordinates from all detection methods
    all_coords: List[np.ndarray] = []

    # Pre-compute common statistics to avoid redundant calculations
    V_percentile_thresh = np.percentile(
        V_work, percentile_thresh
    )  # Cache base image threshold

    # Multiscale Gaussian-blurred peaks (coarsest to finest)
    # Detect blob-like structures at multiple scales by finding peaks in Gaussian-filtered images
    # Process coarsest scales first for farthest-first priority (large structures before details)
    for s in reversed(scales):
        # Apply Gaussian smoothing at current scale
        img = ndi.gaussian_filter(V_work, sigma=s, mode="nearest")

        # Use adaptive threshold: prefer cached base threshold, fall back to scale-specific
        # For heavily smoothed images, use their own percentile; for mild smoothing, reuse base
        if s <= min(scales) * 2.0:  # For fine scales, use base image threshold
            thr = V_percentile_thresh
        else:  # For coarse scales, compute specific threshold
            thr = np.percentile(img, percentile_thresh)

        # Neighborhood radius for peak detection scales with filter size
        # Factor 1.5 ensures peaks are well-separated relative to blob size
        radius = int(max(1, round(1.5 * s)))

        # Find local maxima in filtered image
        coords = local_maxima(img, radius=radius, thresh=thr, top_k=peaks_per_scale)
        all_coords.append(coords)

    # Handle case where no seeds were found
    if len(all_coords) == 0:
        return np.zeros((0, d), float)

    # Combine all seed coordinate arrays, filtering out empty arrays
    non_empty_coords = [c for c in all_coords if c.size > 0]
    if len(non_empty_coords) == 0:
        return np.zeros((0, d), float)

    coords = np.vstack(non_empty_coords)

    # Remove seeds that are too close to each other (spatial deduplication)
    # This reduces redundancy between different detection methods
    coords = dedupe_farthest_first(coords, min_distance=min_distance)

    # Refine seed positions to sub-voxel precision using intensity-weighted centroids
    # This improves localization accuracy by considering local intensity distribution
    # Use V_work (same image used for detection) for consistency
    centers = []
    for c in coords:
        # Extract 3x3x...x3 neighborhood around each seed (clipped at image borders)
        slices = []
        for ax in range(d):
            # Create slice from c-1 to c+2 (exclusive), clamped to image bounds
            lo = max(0, int(c[ax] - 1))
            hi = min(V_work.shape[ax], int(c[ax] + 2))
            slices.append(slice(lo, hi))
        # Extract intensity patch and create coordinate grids
        patch = V_work[tuple(slices)]
        grids = np.meshgrid(
            *[np.arange(s.start, s.stop) for s in slices], indexing="ij"
        )

        # Compute intensity-weighted centroid
        # Subtract minimum to make weights non-negative (relative intensities)
        w = patch - patch.min()
        W = w.sum() + 1e-12  # Add small epsilon to avoid division by zero

        # Calculate weighted average coordinates across all axes
        mu = np.array(
            [float((w * grids[ax]).sum() / W) for ax in range(d)], dtype=float
        )
        centers.append(mu)

    # Convert list to array and return refined seed positions
    centers = np.array(centers, float)
    return centers
