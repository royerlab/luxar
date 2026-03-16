# edges.py
"""
Edge-based seeding for Gaussian splatting.

This module provides edge-based seed generation using Sobel gradients.
Seeds are placed along edges with isotropic Gaussian shapes (sigma=1.0).
"""

from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.seeds.utils import (
    SEED_AMPLITUDE_SCALE,
    sigmas_to_cholesky_isotropic,
)


def seed_from_edges(
    V: np.ndarray,
    n_seeds: Optional[int] = None,
    min_distance: float = 2.0,
    edge_threshold_rel: float = 0.1,
    device: Optional[str] = None,
) -> GSplatData:
    """
    Generate seed Gaussian splats along edges with isotropic shapes.

    This method detects edges using nD Sobel gradients and samples points along
    edges using weighted Poisson disk sampling. Seeds are initialized with
    isotropic Gaussians (σ=1.0).

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume. Shape: (s_0, s_1, ..., s_{n-1}).
    n_seeds : int or None, optional
        Target number of seeds. If None, auto-estimates based on image size.
    min_distance : float, default=2.0
        Minimum distance between seeds in voxels.
    edge_threshold_rel : float, default=0.1
        Relative edge threshold (0.0-1.0). Fraction of max edge response.
    device : str, optional
        PyTorch device for GPU acceleration. Options:
        - None (default): CPU using scipy.ndimage
        - 'cpu': Force CPU
        - 'cuda': NVIDIA GPU (if available)
        - 'mps': Apple Metal (if available)
        - 'auto': Auto-detect best device

        GPU acceleration provides 10-50x speedup for large volumes (>100³).
        Small volumes (<50³) automatically use CPU due to overhead.

    Returns
    -------
    GSplatData
        Gaussian splat seeds with:
        - centers: Edge point positions
        - amplitudes: Intensity values at each point
        - cholesky_factors: Isotropic Cholesky factors (σ=1.0)
        - sharpnesses: All set to 2.0 (standard Gaussian)

    Notes
    -----
    - Seeds are placed along edges (high gradient magnitude)
    - Gaussian shapes are isotropic (σ=1.0) for all seeds
    - The optimizer will adjust shapes during fitting
    - Previous versions used structure tensor for anisotropic initialization,
      but empirical testing showed no benefit in practice

    Examples
    --------
    >>> from luxar.gsplats.seeds import generate_seeds
    >>> import numpy as np
    >>>
    >>> # Create test image with edges
    >>> image = np.zeros((100, 100))
    >>> image[40:60, 40:60] = 1.0  # Square
    >>>
    >>> # Edge-based seeding (CPU)
    >>> seeds = generate_seeds(image, method="edges")
    >>>
    >>> # Edge-based seeding with GPU acceleration
    >>> seeds = generate_seeds(image, method="edges", device="cuda")
    >>>
    >>> # Custom parameters
    >>> seeds = generate_seeds(
    ...     image, method="edges",
    ...     edge_threshold_rel=0.2,  # Higher threshold
    ...     n_seeds=1000,            # Target seed count
    ...     device="auto",           # Auto-detect GPU
    ... )
    """
    # Input validation
    V = np.asarray(V, dtype=float)
    if V.size == 0:
        raise ValueError("Input array V cannot be empty")
    if V.ndim == 0:
        raise ValueError("Input array V must have at least 1 dimension")

    ndim = V.ndim
    shape = np.array(V.shape)

    # Validate parameters
    if edge_threshold_rel < 0 or edge_threshold_rel > 1:
        raise ValueError(
            f"edge_threshold_rel must be in [0, 1], got {edge_threshold_rel}"
        )

    # Estimate n_seeds if not provided
    if n_seeds is None:
        total_voxels = float(np.prod(shape))
        n_seeds = max(50, int(total_voxels ** (1.0 / ndim) / 4))
        n_seeds = min(n_seeds, 5000)

    # Step 1: Compute nD Sobel gradient magnitude
    edge_response = _compute_nd_sobel_magnitude(V, device=device)

    # Step 2: Normalize to [0, 1]
    edge_max = edge_response.max()
    if edge_max > 0:
        edge_response_norm = edge_response / edge_max
    else:
        # No edges - return empty
        return _empty_gsplatdata(ndim)

    # Step 3: Threshold
    threshold = edge_threshold_rel
    mask = edge_response_norm > threshold

    if not np.any(mask):
        # No edges above threshold - return empty
        return _empty_gsplatdata(ndim)

    # Step 4: Poisson disk sampling weighted by edge response
    centers = _poisson_disk_sample_weighted(
        density=edge_response_norm,
        mask=mask,
        n_samples=n_seeds,
        min_distance=min_distance,
    )

    if len(centers) == 0:
        return _empty_gsplatdata(ndim)

    # Step 5: Use simple isotropic σ=1 initialization (ignore structure tensor estimates)
    # The fancy anisotropic shapes from eigendecomposition don't help in practice
    sigmas_one = np.ones(len(centers), dtype=np.float32)
    cholesky_factors = sigmas_to_cholesky_isotropic(sigmas_one, ndim)

    # Step 6: Sample amplitudes from V, scaled to avoid overlap overshoot
    # (over-prediction penalty causes divergence with overlapping splats)
    amplitudes = _sample_amplitudes(V, centers, device=device) * SEED_AMPLITUDE_SCALE

    # Step 7: Standard Gaussian sharpness
    sharpnesses = np.full(len(centers), 2.0, dtype=np.float32)

    return GSplatData(
        centers=centers.astype(np.float32),
        amplitudes=amplitudes.astype(np.float32),
        cholesky_factors=cholesky_factors.astype(np.float32),
        sharpnesses=sharpnesses,
    )


def _empty_gsplatdata(ndim: int) -> GSplatData:
    """Create empty GSplatData."""
    tril_size = ndim * (ndim + 1) // 2
    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        cholesky_factors=np.zeros((0, tril_size), dtype=np.float32),
        sharpnesses=np.zeros(0, dtype=np.float32),
    )


def _compute_nd_sobel_magnitude(
    V: np.ndarray, device: Optional[str] = None
) -> np.ndarray:
    """
    Compute nD Sobel gradient magnitude.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image.
    device : str, optional
        PyTorch device for GPU acceleration. If None or 'cpu', uses scipy.

    Returns
    -------
    np.ndarray
        Gradient magnitude (same shape as V).
    """
    # Dispatch to GPU if requested and appropriate
    if device is not None and device != "cpu":
        import torch

        from luxar.gsplats.seeds.gpu_ops import (
            _compute_nd_sobel_magnitude_gpu,
            _get_device,
            should_use_gpu,
        )

        resolved_device = _get_device(device)

        if resolved_device != "cpu" and should_use_gpu(V, resolved_device):
            # GPU path (works for arbitrary dimensions)
            V_tensor = torch.tensor(V, device=resolved_device, dtype=torch.float32)
            result_tensor = _compute_nd_sobel_magnitude_gpu(V_tensor)
            return result_tensor.cpu().numpy()

    # CPU path (default)
    ndim = V.ndim
    grad_sq_sum = np.zeros_like(V)

    for axis in range(ndim):
        grad = ndi.sobel(V, axis=axis, mode="nearest")
        grad_sq_sum += grad**2

    result: np.ndarray = np.sqrt(grad_sq_sum)
    return result


def _sample_amplitudes(
    V: np.ndarray, coords: np.ndarray, device: Optional[str] = None
) -> np.ndarray:
    """
    Sample amplitudes from volume at given coordinates.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional volume.
    coords : np.ndarray
        Coordinates to sample at, shape (N, ndim).
    device : str, optional
        PyTorch device for GPU acceleration. If None or 'cpu', uses scipy.

    Returns
    -------
    np.ndarray
        Sampled amplitudes, shape (N,).
    """
    # Dispatch to GPU if requested and appropriate
    if device is not None and device != "cpu":
        import torch

        from luxar.gsplats.seeds.gpu_ops import (
            _get_device,
            sample_amplitudes_gpu,
            should_use_gpu,
        )

        resolved_device = _get_device(device)

        if resolved_device != "cpu" and should_use_gpu(V, resolved_device):
            # GPU path with fallback for unsupported dimensions
            try:
                V_tensor = torch.tensor(V, device=resolved_device, dtype=torch.float32)
                coords_tensor = torch.tensor(
                    coords, device=resolved_device, dtype=torch.float32
                )
                result_tensor = sample_amplitudes_gpu(
                    V_tensor, coords_tensor, mode="bilinear"
                )
                return result_tensor.cpu().numpy().astype(np.float32)
            except NotImplementedError:
                # GPU not supported for this dimensionality, fallback to CPU
                import warnings

                warnings.warn(
                    f"GPU interpolation not supported for {V.ndim}D volumes. Using CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # CPU path (default)
    coords_for_interp = coords.T
    result: np.ndarray = ndi.map_coordinates(
        V, coords_for_interp, order=1, mode="nearest"
    ).astype(np.float32)
    return result


def _poisson_disk_sample_weighted(
    density: np.ndarray,
    mask: np.ndarray,
    n_samples: int,
    min_distance: float,
) -> np.ndarray:
    """
    Weighted Poisson disk sampling.

    Samples points proportional to density while maintaining minimum distance.

    Parameters
    ----------
    density : np.ndarray
        Sampling density (higher = more likely to sample).
    mask : np.ndarray
        Boolean mask of valid sampling locations.
    n_samples : int
        Target number of samples.
    min_distance : float
        Minimum distance between samples.

    Returns
    -------
    np.ndarray
        Sampled point coordinates, shape (N, ndim).
    """
    ndim = density.ndim

    # Get valid candidate positions
    valid_coords = np.argwhere(mask)
    if len(valid_coords) == 0:
        return np.zeros((0, ndim), dtype=float)

    # Get density at valid positions
    valid_density = density[tuple(valid_coords.T)]

    # Normalize to probability
    prob = valid_density / valid_density.sum()

    # Use weighted sampling with rejection for min_distance
    # KD-tree acceleration provides O(N log M) instead of O(N×M) complexity
    rng = np.random.default_rng(seed=42)

    # Pre-allocate array for selected seeds (avoids repeated list→array conversions)
    selected_array = np.empty((n_samples, ndim), dtype=np.float32)
    n_selected = 0

    # Number of candidates to try (oversample)
    n_candidates = min(len(valid_coords), n_samples * 10)

    # Sample candidate indices weighted by density
    candidate_indices = rng.choice(
        len(valid_coords),
        size=n_candidates,
        replace=False if n_candidates <= len(valid_coords) else True,
        p=prob,
    )

    # Use KD-tree for fast distance queries (O(log M) instead of O(M) per query)
    tree = None

    for idx in candidate_indices:
        coord = valid_coords[idx].astype(np.float32)

        # Check distance to existing selected points
        if n_selected > 0:
            # Early termination optimization: for last few candidates, use simple check
            # This avoids tree rebuild overhead when very few slots remain
            if n_samples - n_selected <= 3:
                # Simple distance check for last few slots
                diffs = selected_array[:n_selected] - coord
                min_dist_sq = np.min(np.sum(diffs**2, axis=1))
                if min_dist_sq < min_distance**2:
                    continue  # Too close, reject
            else:
                # Use KD-tree for larger selection sets
                if tree is None:
                    tree = cKDTree(selected_array[:n_selected])
                dist, _ = tree.query(coord, k=1)
                if dist < min_distance:
                    continue  # Too close, reject

        # Add to pre-allocated array
        selected_array[n_selected] = coord
        n_selected += 1

        # Rebuild KD-tree with array slice (no list→array conversion overhead)
        # Still O(M log M) per rebuild, but with lower constant factors
        if n_selected > 0 and n_samples - n_selected > 3:
            tree = cKDTree(selected_array[:n_selected])

        if n_selected >= n_samples:
            break

    if n_selected == 0:
        return np.zeros((0, ndim), dtype=float)

    # Trim to actual size and convert back to float64 for consistency
    return selected_array[:n_selected].astype(float)
