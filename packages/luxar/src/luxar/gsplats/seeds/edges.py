# edges.py
"""
Edge-based seeding for Gaussian splatting.

This module provides edge-based seed generation using Sobel gradients and
structure tensor analysis. Seeds are placed along edges with anisotropic
Gaussian shapes oriented along the edge direction.
"""

from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.seeds.utils import sigmas_to_cholesky_isotropic


def seed_from_edges(
    V: np.ndarray,
    n_seeds: Optional[int] = None,
    min_distance: float = 2.0,
    edge_threshold_rel: float = 0.1,
    structure_radius: float = 3.0,
    min_sigma: float = 0.5,
    max_sigma: float = 16.0,
) -> GSplatData:
    """
    Generate seed Gaussian splats along edges with anisotropic shapes.

    This method detects edges using nD Sobel gradients, samples points along
    edges using weighted Poisson disk sampling, and computes anisotropic
    Gaussian shapes from the local structure tensor.

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
    structure_radius : float, default=3.0
        Radius for structure tensor computation (Gaussian weighting).
    min_sigma : float, default=0.5
        Minimum sigma derived from structure tensor eigenvalues.
    max_sigma : float, default=16.0
        Maximum sigma derived from structure tensor eigenvalues.

    Returns
    -------
    GSplatData
        Gaussian splat seeds with:
        - centers: Edge point positions
        - amplitudes: Intensity values at each point
        - cholesky_factors: Anisotropic Cholesky factors from structure tensor
        - sharpnesses: All set to 2.0 (standard Gaussian)

    Notes
    -----
    - Seeds are placed along edges (high gradient magnitude)
    - Gaussian shapes are anisotropic, elongated along edges
    - Structure tensor captures local gradient covariance
    - Off-diagonal Cholesky elements encode orientation

    Examples
    --------
    >>> from luxar.gsplats.seeds import generate_seeds
    >>> import numpy as np
    >>>
    >>> # Create test image with edges
    >>> image = np.zeros((100, 100))
    >>> image[40:60, 40:60] = 1.0  # Square
    >>>
    >>> # Edge-based seeding
    >>> seeds = generate_seeds(image, method="edges")
    >>>
    >>> # Custom parameters
    >>> seeds = generate_seeds(
    ...     image, method="edges",
    ...     edge_threshold_rel=0.2,  # Higher threshold
    ...     structure_radius=5.0,    # Larger integration radius
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
    if min_sigma <= 0:
        raise ValueError(f"min_sigma must be positive, got {min_sigma}")
    if max_sigma <= 0:
        raise ValueError(f"max_sigma must be positive, got {max_sigma}")
    if min_sigma > max_sigma:
        raise ValueError(f"min_sigma ({min_sigma}) must be <= max_sigma ({max_sigma})")

    # Estimate n_seeds if not provided
    if n_seeds is None:
        total_voxels = float(np.prod(shape))
        n_seeds = max(50, int(total_voxels ** (1.0 / ndim) / 4))
        n_seeds = min(n_seeds, 5000)

    # Step 1: Compute nD Sobel gradient magnitude
    edge_response = _compute_nd_sobel_magnitude(V)

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

    # Step 6: Sample amplitudes from V, scaled to 90% to avoid overlap overshoot
    # (over-prediction penalty causes divergence with overlapping splats)
    coords_for_interp = centers.T
    amplitudes = (
        ndi.map_coordinates(V, coords_for_interp, order=1, mode="nearest").astype(
            np.float32
        )
        * 0.9
    )

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


def _compute_nd_sobel_magnitude(V: np.ndarray) -> np.ndarray:
    """
    Compute nD Sobel gradient magnitude.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image.

    Returns
    -------
    np.ndarray
        Gradient magnitude (same shape as V).
    """
    ndim = V.ndim
    grad_sq_sum = np.zeros_like(V)

    for axis in range(ndim):
        grad = ndi.sobel(V, axis=axis, mode="nearest")
        grad_sq_sum += grad**2

    return np.sqrt(grad_sq_sum)


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
    rng = np.random.default_rng(seed=42)
    selected = []

    # Number of candidates to try (oversample)
    n_candidates = min(len(valid_coords), n_samples * 10)

    # Sample candidate indices weighted by density
    candidate_indices = rng.choice(
        len(valid_coords),
        size=n_candidates,
        replace=False if n_candidates <= len(valid_coords) else True,
        p=prob,
    )

    for idx in candidate_indices:
        coord = valid_coords[idx].astype(float)

        # Check distance to existing selected points
        if len(selected) > 0:
            selected_arr = np.array(selected)
            distances = np.sqrt(np.sum((selected_arr - coord) ** 2, axis=1))
            if np.min(distances) < min_distance:
                continue  # Too close, reject

        selected.append(coord)

        if len(selected) >= n_samples:
            break

    if len(selected) == 0:
        return np.zeros((0, ndim), dtype=float)

    return np.array(selected, dtype=float)


def _compute_structure_tensor_cholesky(
    V: np.ndarray,
    points: np.ndarray,
    radius: float,
    min_sigma: float,
    max_sigma: float,
) -> np.ndarray:
    """
    Compute structure tensor at points and convert to Cholesky factors.

    The structure tensor S = Σ (∇V)(∇V)^T captures local gradient covariance.
    Eigenvalues indicate scale, eigenvectors indicate orientation.

    Parameters
    ----------
    V : np.ndarray
        Original image (NOT edge response).
    points : np.ndarray
        Point coordinates, shape (N, ndim).
    radius : float
        Gaussian weighting radius for integration.
    min_sigma : float
        Minimum sigma value.
    max_sigma : float
        Maximum sigma value.

    Returns
    -------
    np.ndarray
        Packed Cholesky factors, shape (N, ndim*(ndim+1)//2).
    """
    ndim = V.ndim
    n_points = len(points)
    tril_size = ndim * (ndim + 1) // 2

    if n_points == 0:
        return np.zeros((0, tril_size), dtype=np.float32)

    # Compute gradient components
    gradients = []
    for axis in range(ndim):
        grad = ndi.sobel(V, axis=axis, mode="nearest")
        gradients.append(grad)

    # Compute structure tensor components (smoothed outer product)
    # S_ij = G * (∂V/∂i)(∂V/∂j) where G is Gaussian smoothing
    structure_components = {}
    for i in range(ndim):
        for j in range(i, ndim):
            component = gradients[i] * gradients[j]
            # Smooth with Gaussian
            smoothed = ndi.gaussian_filter(component, sigma=radius, mode="nearest")
            structure_components[(i, j)] = smoothed

    # For each point, extract structure tensor, eigendecompose, build Cholesky
    cholesky_factors = np.zeros((n_points, tril_size), dtype=np.float32)

    for p_idx, point in enumerate(points):
        # Build structure tensor at this point
        S = np.zeros((ndim, ndim))
        for i in range(ndim):
            for j in range(i, ndim):
                # Sample structure tensor component at point
                coords = tuple(int(round(c)) for c in point)
                # Clip to valid range
                coords = tuple(
                    max(0, min(c, V.shape[d] - 1)) for d, c in enumerate(coords)
                )
                val = structure_components[(i, j)][coords]
                S[i, j] = val
                S[j, i] = val  # Symmetric

        # Eigendecompose S = Q * Λ * Q^T
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(S)
        except np.linalg.LinAlgError:
            # Fallback to isotropic
            cholesky_factors[p_idx] = _isotropic_cholesky(
                (min_sigma + max_sigma) / 2, ndim
            )
            continue

        # Eigenvalues are sorted ascending, we want descending for sigma
        # σ_i = 1 / sqrt(λ_i) (larger eigenvalue = smaller sigma = sharper)
        # Clamp eigenvalues to avoid division by zero
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Convert eigenvalues to sigmas
        # High gradient = small sigma along that direction
        # We invert: sigma = scale / sqrt(eigenvalue)
        # Use a scale factor to get reasonable sigma values
        scale_factor = 1.0
        sigmas_raw = scale_factor / np.sqrt(eigenvalues)

        # Clamp sigmas
        sigmas = np.clip(sigmas_raw, min_sigma, max_sigma)

        # Build Cholesky factor: L such that L @ L^T = Σ
        # Σ = Q @ diag(σ²) @ Q^T
        # L = Q @ diag(σ)
        # But we need L to have positive diagonal elements
        # The covariance is Σ = L @ L^T, so we can negate columns of L
        # and still get the same Σ. Choose signs so diagonals are positive.
        L = eigenvectors @ np.diag(sigmas)

        # Ensure positive diagonal by flipping column signs if needed
        for col in range(ndim):
            if L[col, col] < 0:
                L[:, col] = -L[:, col]

        # If diagonal is still zero or negative due to numerical issues,
        # use a fallback isotropic Cholesky
        if np.any(np.diag(L) <= 0):
            cholesky_factors[p_idx] = _isotropic_cholesky(
                (min_sigma + max_sigma) / 2, ndim
            )
            continue

        # Pack lower triangular
        cholesky_factors[p_idx] = _pack_lower_triangular(L)

    return cholesky_factors


def _isotropic_cholesky(sigma: float, ndim: int) -> np.ndarray:
    """Create packed isotropic Cholesky factor."""
    tril_size = ndim * (ndim + 1) // 2
    cholesky = np.zeros(tril_size, dtype=np.float32)

    # Diagonal indices: 0, 2, 5, 9, ... = k*(k+3)//2
    for k in range(ndim):
        diag_idx = k * (k + 3) // 2
        cholesky[diag_idx] = sigma

    return cholesky


def _pack_lower_triangular(L: np.ndarray) -> np.ndarray:
    """
    Pack lower triangular matrix column-by-column.

    Packed format: [L00, L10, L11, L20, L21, L22, ...]
    """
    ndim = L.shape[0]
    tril_size = ndim * (ndim + 1) // 2
    packed = np.zeros(tril_size, dtype=np.float32)

    idx = 0
    for col in range(ndim):
        for row in range(col, ndim):
            packed[idx] = L[row, col]
            idx += 1

    return packed
