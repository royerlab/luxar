# ====== dependencies ======
from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage as ndi

# ==========================
# 1) Overcomplete candidate centroids in scale-space
# ==========================


def _local_maxima(
    img: np.ndarray, radius: int, thresh: float, top_k: Optional[int]
) -> np.ndarray:
    """
    Find local maxima in n-dimensional image using L∞ (Chebyshev) neighborhood.

    This function identifies peaks by comparing each voxel with all neighbors within
    a hypercube of given radius. A voxel is considered a peak if it equals the
    maximum value in its neighborhood and exceeds the threshold.

    Parameters
    ----------
    img : np.ndarray
        Input n-dimensional image array.
    radius : int
        Half-width of the L∞ neighborhood (hypercube). Minimum value is 1.
    thresh : float
        Minimum intensity threshold for peak detection.
    top_k : int, optional
        Maximum number of strongest peaks to return. If None, returns all peaks.

    Returns
    -------
    np.ndarray
        Array of shape (N, ndim) containing integer coordinates of detected peaks,
        where N is the number of peaks found.
    """
    # Ensure minimum neighborhood size of 3x3x...x3 (radius=1)
    if radius < 1:
        radius = 1

    # Create hypercube footprint for L∞ neighborhood
    # Shape: (2*radius+1)^ndim boolean array
    footprint = np.ones([2 * radius + 1] * img.ndim, dtype=bool)

    # Apply maximum filter to find local maxima
    # Each voxel is compared with all neighbors in the footprint
    max_f = ndi.maximum_filter(img, footprint=footprint, mode="nearest")

    # Peak criteria: voxel equals neighborhood maximum AND exceeds threshold
    peaks_mask = (img == max_f) & (img >= thresh)
    # Extract coordinates of detected peaks
    coords = np.argwhere(peaks_mask)

    # Handle empty result case
    if coords.size == 0:
        return coords

    # Optionally limit to top_k strongest peaks
    if top_k is not None and len(coords) > top_k:
        # Get intensity values at peak locations
        vals = img[tuple(coords.T)]
        # Select indices of top_k strongest peaks (highest intensities)
        keep = np.argsort(vals)[-top_k:]
        coords = coords[keep]
    return coords


def _dog_response(vol: np.ndarray, sigma: float, k: float = 1.6) -> np.ndarray:
    """
    Compute Difference of Gaussians (DoG) response as Laplacian of Gaussian approximation.

    DoG approximates the Laplacian of Gaussian (LoG) operator for blob detection.
    The response is calculated as G(σ) - G(k*σ), where G denotes Gaussian filtering.

    Parameters
    ----------
    vol : np.ndarray
        Input n-dimensional volume/image.
    sigma : float
        Standard deviation for the first Gaussian kernel.
    k : float, default=1.6
        Scale factor for the second Gaussian kernel. Traditional value is 1.6
        which approximates LoG well.

    Returns
    -------
    np.ndarray
        DoG response with same shape as input. Positive values indicate
        bright blobs, negative values indicate dark blobs.
    """
    # Apply Gaussian filtering at two different scales
    g1 = ndi.gaussian_filter(vol, sigma=sigma, mode="nearest")  # Finer scale
    g2 = ndi.gaussian_filter(vol, sigma=sigma * k, mode="nearest")  # Coarser scale

    # Difference of Gaussians: highlights blob-like structures
    return g1 - g2


def _dedupe(coords: np.ndarray, min_dist: float) -> np.ndarray:
    """
    Remove duplicate candidates using greedy spatial deduplication.

    This function eliminates candidates that are too close to each other using
    a greedy algorithm. For each unprocessed candidate, it marks all nearby
    candidates (within min_dist) as used, effectively keeping only spatially
    well-separated points.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, ndim) containing candidate coordinates.
    min_dist : float
        Minimum Euclidean distance required between kept candidates.

    Returns
    -------
    np.ndarray
        Array of deduplicated coordinates, subset of input coords.
        Returns float coordinates for consistency with downstream processing.
    """
    # Handle empty input case
    if len(coords) == 0:
        return coords

    # Track which candidates have been processed/eliminated
    used = np.zeros(len(coords), dtype=bool)
    out = []  # List to collect kept coordinates
    # Process candidates sequentially (could be randomized to avoid spatial bias)
    for i in range(len(coords)):
        # Skip if this candidate was already marked as too close to a kept one
        if used[i]:
            continue

        # Keep this candidate
        c = coords[i]
        out.append(c)
        # Mark as used all candidates within min_dist (Euclidean) of current candidate c
        diffs = coords - c  # Displacement vectors from c to all candidates

        # Compute squared Euclidean distances (avoid sqrt for efficiency)
        if coords.shape[1] == 1:
            # Special case for 1D: just squared difference
            dist2 = diffs[:, 0] ** 2
        else:
            # General nD case: sum of squared differences along all axes
            dist2 = np.sum(diffs**2, axis=1)

        # Mark candidates within exclusion radius as used
        used |= dist2 <= (min_dist**2)
    return np.array(out, dtype=float)


def find_candidates_overcomplete_nd(
    V: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    scales: Sequence[float] = (0.7, 1.0, 1.4, 2.0, 2.8, 4.0),
    peaks_per_scale: int = 1000,
    percentile_thresh: float = 70.0,
    min_dist: float = 2.0,
    add_intensity_grid: bool = True,
    grid_step: Optional[Sequence[int]] = None,
    grid_percentile: float = 60.0,
) -> np.ndarray:
    """
    Generate overcomplete set of candidate centers for Gaussian splat fitting.

    This function combines multiple detection strategies to create a rich set of
    candidate locations where Gaussian splats might be placed:

    1. Multiscale Gaussian-blurred peaks: Detects blob-like structures at various sizes
    2. Multiscale DoG (Difference of Gaussians) peaks: Detects blob boundaries and edges
    3. Intensity-weighted grid sampling: Ensures spatial coverage in high-intensity regions

    The resulting candidate set is intentionally overcomplete - the subsequent fitting
    process will select and refine the most useful subset.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to analyze.
    spacing : Sequence[float], optional
        Physical spacing between voxels along each axis. Currently unused but
        reserved for future physical-space calculations. Default is unit spacing.
    scales : Sequence[float], default=(0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
        Standard deviations (in voxels) for multiscale Gaussian filtering.
        Should cover the range of expected feature sizes.
    peaks_per_scale : int, default=1000
        Maximum number of peaks to detect at each scale for Gaussian filtering.
        DoG detection uses half this number to balance computational cost.
    percentile_thresh : float, default=70.0
        Intensity percentile threshold (0-100) for peak detection. Higher values
        are more selective, lower values detect more candidates.
    min_dist : float, default=2.0
        Minimum Euclidean distance (in voxels) between candidate centers.
        Used for deduplication to avoid overly dense candidates.
    add_intensity_grid : bool, default=True
        Whether to add regular grid samples weighted by local intensity.
        Helps ensure coverage in textured regions.
    grid_step : Sequence[int], optional
        Step size along each axis for grid sampling. If None, automatically
        determined as ~4 times the minimum scale.
    grid_percentile : float, default=60.0
        Intensity percentile threshold for keeping grid samples. Lower than
        percentile_thresh to be more inclusive.

    Returns
    -------
    np.ndarray
        Array of shape (N, ndim) containing candidate center coordinates in
        voxel units (float). Coordinates may be sub-voxel due to centroid refinement.
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
    if peaks_per_scale <= 0:
        raise ValueError("peaks_per_scale must be positive")
    if not (0 <= percentile_thresh <= 100):
        raise ValueError("percentile_thresh must be between 0 and 100")
    if min_dist <= 0:
        raise ValueError("min_dist must be positive")
    if not (0 <= grid_percentile <= 100):
        raise ValueError("grid_percentile must be between 0 and 100")

    # Validate grid_step if provided
    if grid_step is not None:
        if len(grid_step) != d:
            raise ValueError(
                f"grid_step must have length {d} to match input dimensions"
            )
        if any(step <= 0 for step in grid_step):
            raise ValueError("All grid_step values must be positive")

    # Physical spacing (currently unused, but reserved for future features)
    spacing = np.ones(d, float) if spacing is None else np.asarray(spacing, float)
    if len(spacing) != d:
        raise ValueError(f"spacing must have length {d} to match input dimensions")

    # Collect coordinates from all detection methods
    all_coords: List[np.ndarray] = []

    # Pre-compute common statistics to avoid redundant calculations
    V_percentile_thresh = np.percentile(
        V, percentile_thresh
    )  # Cache base image threshold

    # (A) Multiscale Gaussian-blurred peaks
    # Detect blob-like structures at multiple scales by finding peaks in Gaussian-filtered images
    for s in scales:
        # Apply Gaussian smoothing at current scale
        img = ndi.gaussian_filter(V, sigma=s, mode="nearest")

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
        coords = _local_maxima(img, radius=radius, thresh=thr, top_k=peaks_per_scale)
        all_coords.append(coords)

    # (B) Multiscale Difference of Gaussians (DoG) peaks
    # Detect blob boundaries and edge-like structures using DoG filtering
    # Pre-compute DoG responses for all scales to enable threshold optimization
    dog_responses = [_dog_response(V, sigma=s, k=1.6) for s in scales]

    # Compute adaptive thresholds: use global DoG statistics when possible
    if len(dog_responses) > 1:
        # For multiple scales, use combined statistics for more stable thresholding
        combined_dog = np.concatenate([dog.ravel() for dog in dog_responses])
        global_dog_thresh = np.percentile(combined_dog, percentile_thresh)

    for i, s in enumerate(scales):
        dog = dog_responses[i]

        # Use global threshold for consistency, or scale-specific if significantly different
        if len(dog_responses) > 1:
            scale_thresh = np.percentile(dog, percentile_thresh)
            # Use global threshold unless scale-specific differs significantly
            thr = (
                global_dog_thresh
                if abs(scale_thresh - global_dog_thresh)
                / (abs(global_dog_thresh) + 1e-12)
                < 0.5
                else scale_thresh
            )
        else:
            thr = np.percentile(dog, percentile_thresh)

        # Same neighborhood radius as Gaussian peaks
        radius = int(max(1, round(1.5 * s)))

        # Find peaks in DoG response (use fewer peaks to balance computational cost)
        coords = _local_maxima(
            dog, radius=radius, thresh=thr, top_k=peaks_per_scale // 2
        )
        all_coords.append(coords)

    # (C) Optional intensity-weighted grid sampling
    # Ensures spatial coverage by sampling on a regular grid, keeping high-intensity regions
    if add_intensity_grid:
        if grid_step is None:
            # Heuristic: grid spacing should be ~4x the minimum detection scale
            # This ensures we don't miss features between scale-based detections
            step = max(2, int(round(4 * min(scales))))
            grid_step = [step] * d
        grid_step = list(grid_step)
        # Create regular grid points, offset by half-step to avoid image boundaries
        anchors = [
            np.arange(grid_step[i] // 2, V.shape[i], grid_step[i]) for i in range(d)
        ]

        # Generate n-dimensional grid using meshgrid
        grids = np.meshgrid(*anchors, indexing="ij")

        # Flatten grid coordinates into (N_points, ndim) array
        pts = np.stack([g.ravel() for g in grids], axis=1)
        # Compute local intensity at each grid point using small neighborhood averaging
        # Box size scales with grid step to capture local intensity context
        avg_grid_step = int(np.mean(grid_step))
        box = tuple([max(1, avg_grid_step // 3)] * d)
        local = ndi.uniform_filter(V, size=box, mode="nearest")

        # Sample filtered intensities at grid points
        vals = local[tuple(pts.T)]

        # Keep grid points with above-average local intensity
        # Use lower threshold than peak detection to be more inclusive
        thr = np.percentile(vals, grid_percentile)
        coords = pts[vals >= thr]

        # Add to candidate list (convert to int coordinates)
        all_coords.append(coords.astype(int))

    # Handle case where no candidates were found
    if len(all_coords) == 0:
        return np.zeros((0, d), float)

    # Combine all candidate coordinate arrays, filtering out empty arrays
    coords = np.vstack([c for c in all_coords if c.size > 0])

    # Remove candidates that are too close to each other (spatial deduplication)
    # This reduces redundancy between different detection methods
    coords = _dedupe(coords, min_dist=min_dist)

    # Refine candidate positions to sub-voxel precision using intensity-weighted centroids
    # This improves localization accuracy by considering local intensity distribution
    centers = []
    for c in coords:
        # Extract 3x3x...x3 neighborhood around each candidate (clipped at image borders)
        slices = []
        for ax in range(d):
            # Create slice from c-1 to c+2 (exclusive), clamped to image bounds
            lo = max(0, int(c[ax] - 1))
            hi = min(V.shape[ax], int(c[ax] + 2))
            slices.append(slice(lo, hi))
        # Extract intensity patch and create coordinate grids
        patch = V[tuple(slices)]
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

    # Convert list to array and return refined candidate positions
    centers = np.array(centers, float)
    return centers
