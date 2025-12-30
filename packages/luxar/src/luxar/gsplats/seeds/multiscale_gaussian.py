# multiscale_gaussian.py
"""
Multiscale Gaussian seed generation for Gaussian splatting.

This module provides multiscale Gaussian-blurred peak detection with optional
CLAHE preprocessing. Returns GSplatData with scale-informed Gaussian shapes
where sigma = blur_scale for each detected seed.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.seeds.utils import (
    local_maxima,
    sigmas_to_cholesky_isotropic,
)


def seed_from_gaussian(
    V: np.ndarray,
    scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    peaks_per_scale: Optional[int] = None,
    percentile_thresh: float = 75.0,
    min_distance: float = 2.0,
    apply_clahe: bool = True,
    clahe_tile_size: int = 32,
    clahe_clip_limit: float = 16.0,
    clahe_nbins: int = 256,
) -> GSplatData:
    """
    Generate seed Gaussian splats using multiscale Gaussian blob detection.

    Uses multiscale Gaussian-blurred peak detection to detect blob-like structures
    at various sizes. Each detected seed is initialized as an isotropic Gaussian
    with sigma equal to the blur scale at which it was detected.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to analyze.
    scales : Sequence[float], default=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
        Standard deviations (in voxels) for multiscale Gaussian filtering.
        Each seed's sigma is set to the scale at which it was detected.
    peaks_per_scale : int or None, optional
        Maximum number of peaks to detect at each scale. If None, extracts all
        peaks above threshold. Default: None (unlimited).
    percentile_thresh : float, default=75.0
        Intensity percentile threshold (0-100) for peak detection. Higher values
        are more selective, lower values detect more seeds.
    min_distance : float, default=2.0
        Minimum Euclidean distance (in voxels) between seed centers.
        Used for deduplication to avoid overly dense seeds.
    apply_clahe : bool, default=True
        Whether to apply CLAHE preprocessing before detection.
    clahe_tile_size : int, default=32
        Tile size for CLAHE preprocessing in voxels.
    clahe_clip_limit : float, default=16.0
        Contrast limiting factor for CLAHE.
    clahe_nbins : int, default=256
        Number of histogram bins for CLAHE equalization.

    Returns
    -------
    GSplatData
        Gaussian splat seeds with:
        - centers: Sub-voxel refined peak positions
        - amplitudes: Peak intensities from original image
        - cholesky_factors: Isotropic Cholesky factors where sigma = scale
        - sharpnesses: All set to 2.0 (standard Gaussian)

    Notes
    -----
    The sigma for each seed is determined by the scale at which it was detected.
    Features detected at scale=4.0 will have sigma=4.0 voxels.
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

    # CLAHE preprocessing (if enabled)
    if apply_clahe:
        import torch

        from luxar.gsplats.clahe import apply_clahe as apply_clahe_torch

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

    # Collect coordinates and their scales from all detection
    all_coords: List[np.ndarray] = []
    all_scales: List[np.ndarray] = []  # Track scale for each seed

    # Pre-compute common statistics
    V_percentile_thresh = np.percentile(V_work, percentile_thresh)

    # Multiscale Gaussian-blurred peaks (coarsest to finest)
    for s in reversed(scales):
        # Apply Gaussian smoothing at current scale
        img = ndi.gaussian_filter(V_work, sigma=s, mode="nearest")

        # Adaptive threshold
        if s <= min(scales) * 2.0:
            thr = V_percentile_thresh
        else:
            thr = np.percentile(img, percentile_thresh)

        # Neighborhood radius scales with filter size
        radius = int(max(1, round(1.5 * s)))

        # Find local maxima
        coords = local_maxima(img, radius=radius, thresh=thr, top_k=peaks_per_scale)

        if len(coords) > 0:
            all_coords.append(coords)
            # Track the scale for each seed detected at this scale
            all_scales.append(np.full(len(coords), s, dtype=np.float32))

    # Handle case where no seeds were found
    if len(all_coords) == 0:
        return GSplatData(
            centers=np.zeros((0, d), dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
            cholesky_factors=np.zeros((0, d * (d + 1) // 2), dtype=np.float32),
            sharpnesses=np.zeros(0, dtype=np.float32),
        )

    # Combine all seeds
    coords = np.vstack(all_coords)
    seed_scales = np.concatenate(all_scales)

    # Deduplicate seeds (keeping track of which scale each came from)
    # Get intensities for deduplication priority
    coords_int = np.clip(np.round(coords).astype(int), 0, np.array(V_work.shape) - 1)
    intensities = V_work[tuple(coords_int.T)]

    # Deduplicate with intensity priority
    coords, seed_scales = _dedupe_with_scales(
        coords, seed_scales, intensities, min_distance
    )

    # Refine positions to sub-voxel precision
    centers = _refine_positions(coords, V_work)

    # Get amplitudes from original image at refined positions
    centers_int = np.clip(
        np.round(centers).astype(int), 0, np.array(V.shape) - 1
    )
    amplitudes = V[tuple(centers_int.T)].astype(np.float32)

    # Build Cholesky factors from scales (sigma = scale)
    cholesky_factors = sigmas_to_cholesky_isotropic(seed_scales, d)

    # Standard Gaussian sharpness
    sharpnesses = np.full(len(centers), 2.0, dtype=np.float32)

    return GSplatData(
        centers=centers.astype(np.float32),
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        sharpnesses=sharpnesses,
    )


def _dedupe_with_scales(
    coords: np.ndarray,
    scales: np.ndarray,
    intensities: np.ndarray,
    min_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate seeds while preserving scale information.

    Uses intensity-based priority for selection.
    """
    if len(coords) == 0:
        return coords, scales

    # Sort by intensity (highest first)
    sort_idx = np.argsort(intensities)[::-1]
    coords_sorted = coords[sort_idx]
    scales_sorted = scales[sort_idx]

    # Greedy deduplication
    kept_mask = np.ones(len(coords_sorted), dtype=bool)

    for i in range(len(coords_sorted)):
        if not kept_mask[i]:
            continue

        # Mark nearby seeds as rejected
        diffs = coords_sorted[i + 1 :] - coords_sorted[i]
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        nearby = distances < min_distance
        kept_mask[i + 1 :][nearby] = False

    return coords_sorted[kept_mask], scales_sorted[kept_mask]


def _refine_positions(coords: np.ndarray, V_work: np.ndarray) -> np.ndarray:
    """
    Refine seed positions to sub-voxel precision using intensity-weighted centroids.
    """
    d = V_work.ndim
    centers = []

    for c in coords:
        # Extract 3x3x...x3 neighborhood
        slices = []
        for ax in range(d):
            lo = max(0, int(c[ax] - 1))
            hi = min(V_work.shape[ax], int(c[ax] + 2))
            slices.append(slice(lo, hi))

        patch = V_work[tuple(slices)]
        grids = np.meshgrid(
            *[np.arange(s.start, s.stop) for s in slices], indexing="ij"
        )

        # Intensity-weighted centroid
        w = patch - patch.min()
        W = w.sum() + 1e-12

        mu = np.array(
            [float((w * grids[ax]).sum() / W) for ax in range(d)], dtype=float
        )
        centers.append(mu)

    return np.array(centers, dtype=float)
