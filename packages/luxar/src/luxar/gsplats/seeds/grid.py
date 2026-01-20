# grid.py
"""
Uniform grid seeding for Gaussian splatting.

This module provides simple uniform grid-based seed generation. Seeds are placed
on a regular grid with optional jitter and intensity filtering. Returns GSplatData
with isotropic Gaussian shapes.
"""

from typing import Optional, Sequence, Union

import numpy as np
from scipy import ndimage as ndi

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.seeds.utils import (
    SEED_AMPLITUDE_SCALE,
    sigmas_to_cholesky_isotropic,
)


def seed_from_grid(
    V: np.ndarray,
    spacing: Optional[Union[float, Sequence[float]]] = None,
    jitter: float = 0.0,
    sigma: Optional[float] = None,
    exclude_below: Optional[float] = None,
    exclude_below_percentile: Optional[float] = None,
    device: Optional[str] = None,
) -> GSplatData:
    """
    Generate seed Gaussian splats on a uniform grid.

    This method creates seeds at regular grid positions throughout the image.
    It provides uniform spatial coverage, useful as a baseline or for filling
    gaps left by other seeding methods.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume. Shape: (s_0, s_1, ..., s_{n-1}).
    spacing : float or Sequence[float] or None, optional
        Grid spacing in voxels. Can be:
        - float: Same spacing for all dimensions
        - Sequence[float]: Per-dimension spacing
        - None: Auto-compute with aspect-ratio-aware spacing (default).
          Spacing is proportional to each dimension's size, respecting anisotropy.
          Example: 1000×1000×10 image → [136, 136, 1.4] spacing (not [29, 29, 29])
    jitter : float, default=0.0
        Jitter fraction (0.0 to 0.5). Random offset applied to each grid point
        as a fraction of spacing. 0.0 = no jitter, 0.5 = up to half spacing.
    sigma : float or None, optional
        Gaussian sigma (standard deviation) for all seeds. If None, defaults
        to spacing / 2 (ensures ~95% overlap between adjacent grid points).
    exclude_below : float or None, optional
        Absolute intensity threshold. Grid points where V < threshold are excluded.
        Mutually exclusive with exclude_below_percentile.
    exclude_below_percentile : float or None, optional
        Percentile threshold (0-100). Grid points below this percentile of V
        are excluded. Mutually exclusive with exclude_below.
    device : str, optional
        PyTorch device for GPU acceleration. Options:
        - None (default): CPU using scipy.ndimage
        - 'cpu': Force CPU
        - 'cuda': NVIDIA GPU (if available)
        - 'mps': Apple Metal (if available)
        - 'auto': Auto-detect best device

        GPU acceleration provides 10-30x speedup for amplitude interpolation
        on large volumes (>100³).

    Returns
    -------
    GSplatData
        Gaussian splat seeds with:
        - centers: Grid point positions (possibly jittered)
        - amplitudes: Intensity values at each grid point
        - cholesky_factors: Isotropic Cholesky factors (sigma * I)
        - sharpnesses: All set to 2.0 (standard Gaussian)

    Notes
    -----
    - Grid seeding provides uniform spatial coverage
    - Default spacing is **aspect-ratio-aware**: respects image anisotropy
      (e.g., thin Z slices in microscopy get denser Z spacing)
    - Jitter helps avoid aliasing artifacts
    - Intensity filtering removes seeds in low-signal regions
    - This method is fast and produces many seeds; combine with
      other methods using the "auto" mode in generate_seeds()

    Examples
    --------
    >>> from luxar.gsplats.seeds import generate_seeds
    >>> import numpy as np
    >>>
    >>> # Create test image
    >>> image = np.random.rand(100, 100) + 0.5
    >>>
    >>> # Basic grid seeding (CPU)
    >>> seeds = generate_seeds(image, method="grid")
    >>>
    >>> # Grid seeding with GPU acceleration
    >>> seeds = generate_seeds(image, method="grid", device="cuda")
    >>>
    >>> # Custom spacing with jitter
    >>> seeds = generate_seeds(image, method="grid", spacing=10.0, jitter=0.25)
    >>>
    >>> # Exclude low-intensity regions
    >>> seeds = generate_seeds(image, method="grid", exclude_below_percentile=25.0)
    """
    # Input validation
    V = np.asarray(V, dtype=float)
    if V.size == 0:
        raise ValueError("Input array V cannot be empty")
    if V.ndim == 0:
        raise ValueError("Input array V must have at least 1 dimension")

    ndim = V.ndim
    shape = np.array(V.shape)

    # Validate jitter
    if not 0.0 <= jitter <= 0.5:
        raise ValueError(f"jitter must be in [0.0, 0.5], got {jitter}")

    # Validate mutual exclusivity of thresholds
    if exclude_below is not None and exclude_below_percentile is not None:
        raise ValueError(
            "exclude_below and exclude_below_percentile are mutually exclusive"
        )

    # Compute spacing if not provided
    if spacing is None:
        # Aspect-ratio-aware default: spacing proportional to image shape
        # This respects anisotropy (e.g., 1000×1000×10 thin slices)
        #
        # Strategy: Match old isotropic seed density but distribute proportionally
        # Old: spacing = 5% of smallest dim (same for all dims)
        # New: spacing[i] = shape[i] / k, where k chosen to match old density
        min_dim = float(np.min(shape))
        s_old = max(2.0, min_dim * 0.05)  # Old isotropic spacing

        # Geometric mean of shape (characteristic length scale)
        geom_mean = float(np.prod(shape) ** (1.0 / ndim))

        # Scale factor: distribute old spacing across dimensions proportionally
        k = geom_mean / s_old

        # Spacing proportional to shape (respects aspect ratio)
        spacing_arr = shape.astype(float) / k

        # Ensure minimum 2.0 voxels in each dimension
        spacing_arr = np.maximum(spacing_arr, 2.0)
    elif isinstance(spacing, (int, float)):
        spacing_arr = np.full(ndim, float(spacing))
    else:
        spacing_arr = np.asarray(spacing, dtype=float)
        if len(spacing_arr) != ndim:
            raise ValueError(
                f"spacing has {len(spacing_arr)} elements but V has {ndim} dimensions"
            )

    # Validate spacing
    if np.any(spacing_arr <= 0):
        raise ValueError("All spacing values must be positive")

    # Compute sigma if not provided
    if sigma is None:
        sigma = float(np.mean(spacing_arr)) / 2.0

    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    # Generate grid coordinates
    # Create 1D arrays for each dimension, then meshgrid
    ranges = []
    for dim in range(ndim):
        # Start at half-spacing, end before shape boundary
        start = spacing_arr[dim] / 2.0
        end = float(shape[dim]) - spacing_arr[dim] / 2.0
        if end < start:
            # Image too small for even one grid point in this dimension
            end = start
        dim_coords = np.arange(start, end + spacing_arr[dim] / 2.0, spacing_arr[dim])
        ranges.append(dim_coords)

    # Create meshgrid and flatten to get all grid points
    if ndim == 1:
        grid_coords = ranges[0].reshape(-1, 1)
    else:
        grids = np.meshgrid(*ranges, indexing="ij")
        grid_coords = np.column_stack([g.ravel() for g in grids])

    n_points = len(grid_coords)

    if n_points == 0:
        # Return empty GSplatData
        tril_size = ndim * (ndim + 1) // 2
        return GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
            cholesky_factors=np.zeros((0, tril_size), dtype=np.float32),
            sharpnesses=np.zeros(0, dtype=np.float32),
        )

    # Apply jitter
    if jitter > 0:
        # Random offset in range [-jitter * spacing, +jitter * spacing]
        rng = np.random.default_rng(seed=42)  # Reproducible
        jitter_offsets = rng.uniform(
            -jitter * spacing_arr, jitter * spacing_arr, size=grid_coords.shape
        )
        grid_coords = grid_coords + jitter_offsets

        # Clip to valid bounds
        for dim in range(ndim):
            grid_coords[:, dim] = np.clip(grid_coords[:, dim], 0, shape[dim] - 1)

    # Sample amplitudes from V using interpolation, scaled to avoid overlap overshoot
    # (over-prediction penalty causes divergence with overlapping splats)
    from luxar.gsplats.seeds.edges import _sample_amplitudes

    amplitudes = _sample_amplitudes(V, grid_coords, device=device) * SEED_AMPLITUDE_SCALE

    # Apply intensity threshold
    if exclude_below is not None:
        mask = amplitudes >= exclude_below
    elif exclude_below_percentile is not None:
        threshold = np.percentile(V, exclude_below_percentile)
        mask = amplitudes >= threshold
    else:
        mask = np.ones(n_points, dtype=bool)

    # Filter by mask
    grid_coords = grid_coords[mask]
    amplitudes = amplitudes[mask]

    n_final = len(grid_coords)

    if n_final == 0:
        tril_size = ndim * (ndim + 1) // 2
        return GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
            cholesky_factors=np.zeros((0, tril_size), dtype=np.float32),
            sharpnesses=np.zeros(0, dtype=np.float32),
        )

    # Use spacing-based sigma so splats cover the image
    # σ = spacing/2 ensures ~60% overlap at midpoints between grid points
    sigmas = np.full(n_final, sigma, dtype=np.float32)
    cholesky_factors = sigmas_to_cholesky_isotropic(sigmas, ndim)

    # Standard Gaussian sharpness
    sharpnesses = np.full(n_final, 2.0, dtype=np.float32)

    return GSplatData(
        centers=grid_coords.astype(np.float32),
        amplitudes=amplitudes.astype(np.float32),
        cholesky_factors=cholesky_factors,
        sharpnesses=sharpnesses,
    )
