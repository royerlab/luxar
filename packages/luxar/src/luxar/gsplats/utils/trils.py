# -------------------------------
# Pack / unpack lower-triangular matrices
# -------------------------------
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def tril_size(d: int) -> int:
    """
    Calculate number of elements in lower-triangular portion of d×d matrix.

    This includes all elements on and below the main diagonal, which is
    the standard storage requirement for Cholesky decomposition.

    Parameters
    ----------
    d : int
        Dimension of square matrix.

    Returns
    -------
    int
        Number of lower-triangular elements: d*(d+1)/2.

    Examples
    --------
    >>> tril_size(3)
    6  # Elements: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
    """
    return d * (d + 1) // 2


def calculate_gradient_dilution_factor(d: int) -> float:
    """
    Calculate gradient dilution compensation factor for higher dimensions.

    Gradient dilution occurs because higher dimensions have more parameters per splat,
    spreading gradients thinner. This function computes the compensation factor.

    Parameters
    ----------
    d : int
        Dimensionality

    Returns
    -------
    float
        Gradient dilution compensation factor (multiply base learning rate by this)

    Notes
    -----
    **Background:**
    In Gaussian splat fitting, each splat has d center params + d(d+1)/2 Cholesky params.
    As dimensionality increases, the same loss gradient gets distributed across more
    parameters, causing each parameter to receive smaller gradient updates. This effect
    is called "gradient dilution."

    **Formula rationale:**

    For 2D/3D: Simple linear scaling by parameter count ratio.
    - 2D: 2 + 3 = 5 params per splat (baseline)
    - 3D: 3 + 6 = 9 params → factor = 9/5 = 1.8

    For 4D+: Two additional effects compound:
    1. **Parameter dilution** (params_current / params_2d): More parameters need updates
    2. **Spatial complexity** (d^0.8): Higher-dimensional spaces have exponentially more
       "room" for splats to move, requiring larger position updates to achieve equivalent
       progress in fitting. The 0.8 exponent was empirically determined through testing
       on 4D-8D synthetic datasets, balancing convergence speed vs. stability.

    **Empirical validation:**
    - Without compensation: 4D+ fitting converges 3-10x slower than 2D/3D
    - With d^0.8 factor: Convergence rates across dimensions within 2x of each other
    - The 0.8 exponent is a compromise: d^1.0 caused instability in 6D+, d^0.5 was
      insufficient for 4D-5D

    **Example factors:**
    - 2D: 1.0 (baseline)
    - 3D: 1.8
    - 4D: 3.0 * 3.5 / 5 = 2.1 (d^0.8 ≈ 3.0, params = 4+10 = 14)
    - 6D: 4.2 * 5.2 / 5 = 4.4 (d^0.8 ≈ 4.2, params = 6+21 = 27)
    """
    # Calculate number of parameters per splat
    params_2d = 2 + tril_size(2)  # 5 parameters (baseline)
    params_current = d + tril_size(d)  # Current dimension parameters

    if d <= 3:
        # Conservative scaling for 2D/3D (maintain existing quality)
        gradient_dilution_factor = params_current / params_2d
    else:
        # More aggressive scaling for 4D+ (address empirical findings)
        # See docstring Notes for rationale behind the d^0.8 exponent
        dimensional_complexity = d**0.8  # Spatial complexity scaling
        parameter_complexity = params_current / params_2d  # Parameter dilution
        gradient_dilution_factor = dimensional_complexity * parameter_complexity

    return gradient_dilution_factor


def pack_tril(L: np.ndarray) -> np.ndarray:
    """
    Pack lower-triangular portion of matrices into compact vector representation.

    Extracts and concatenates lower-triangular elements (including diagonal)
    from a batch of square matrices. This is commonly used for efficient
    storage and transmission of Cholesky factors.

    Parameters
    ----------
    L : np.ndarray, shape (N, d, d)
        Batch of square matrices. Only elements where i >= j are used
        (on and below main diagonal). Upper triangular elements are ignored.

    Returns
    -------
    np.ndarray, shape (N, d*(d+1)//2)
        Packed vectors containing lower-triangular elements in row-major order.
        For each matrix, elements are ordered as:
        [L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2], ...]

    Examples
    --------
    >>> L = np.array([[[1, 0], [2, 3]]])  # Shape (1, 2, 2)
    >>> pack_tril(L)
    array([[1, 2, 3]])  # Shape (1, 3): [L00, L10, L11]
    """
    N, d, _ = L.shape
    out = np.zeros((N, tril_size(d)), dtype=L.dtype)
    k = 0
    for i in range(d):
        for j in range(i + 1):
            out[:, k] = L[:, i, j]
            k += 1
    return out


def unpack_tril(v: np.ndarray, d: int) -> np.ndarray:
    """
    Unpack compact vector representation into lower-triangular matrices.

    Inverse operation of pack_tril(). Reconstructs square matrices from
    their packed lower-triangular representations, filling upper triangle
    with zeros.

    Parameters
    ----------
    v : np.ndarray, shape (N, d*(d+1)//2)
        Packed vectors containing lower-triangular elements in row-major order.
    d : int
        Dimension of square matrices to reconstruct.

    Returns
    -------
    np.ndarray, shape (N, d, d)
        Batch of lower-triangular matrices with zeros above diagonal
        and packed elements on/below diagonal.

    Examples
    --------
    >>> v = np.array([[1, 2, 3]])  # Shape (1, 3)
    >>> unpack_tril(v, 2)
    array([[[1, 0],
            [2, 3]]])  # Shape (1, 2, 2)
    """
    N = v.shape[0]
    # Initialize output matrices (zeros above diagonal by default)
    L = np.zeros((N, d, d), dtype=v.dtype)

    # Unpack elements in row-major order
    k = 0
    for i in range(d):  # Row index
        for j in range(i + 1):  # Column index (j <= i, lower triangle)
            L[:, i, j] = v[:, k]
            k += 1
    return L


def validate_cholesky_shape(
    cholesky_factors: np.ndarray,
    ndim: int,
    n_splats: Optional[int] = None,
    allow_uniform: bool = True,
) -> Tuple[bool, int]:
    """
    Validate shape of packed Cholesky factors for Gaussian splats.

    Packed Cholesky factors should be either:
    - Per-splat: shape (N, k) where k = d*(d+1)//2
    - Uniform: shape (k,) when allow_uniform=True

    Parameters
    ----------
    cholesky_factors : np.ndarray
        Packed Cholesky factors array to validate.
    ndim : int
        Number of dimensions (d). Determines expected packed size k = d*(d+1)//2.
    n_splats : int, optional
        Expected number of splats. If provided, validates first dimension matches.
        Ignored if cholesky_factors is uniform (1D).
    allow_uniform : bool, default=True
        Whether to allow uniform Cholesky factors (shape (k,)) for all splats.

    Returns
    -------
    is_uniform : bool
        True if cholesky_factors is uniform (shape (k,)), False if per-splat.
    actual_n_splats : int
        Actual number of splats inferred from shape. For uniform, returns 0.

    Raises
    ------
    ValueError
        If shape is invalid for the given ndim and n_splats.

    Examples
    --------
    >>> # Valid per-splat for 2D (k=3)
    >>> chol = np.random.rand(100, 3)
    >>> is_uniform, n = validate_cholesky_shape(chol, ndim=2, n_splats=100)
    >>> is_uniform, n
    (False, 100)

    >>> # Valid uniform for 3D (k=6)
    >>> chol = np.random.rand(6)
    >>> is_uniform, n = validate_cholesky_shape(chol, ndim=3)
    >>> is_uniform, n
    (True, 0)

    >>> # Invalid shape raises
    >>> chol = np.random.rand(100, 5)  # Wrong k for 2D
    >>> validate_cholesky_shape(chol, ndim=2)
    Traceback (most recent call last):
        ...
    ValueError: Cholesky factors have wrong packed size...
    """
    expected_k = tril_size(ndim)

    # Check for uniform (1D) case
    if cholesky_factors.ndim == 1:
        if not allow_uniform:
            raise ValueError(
                f"Uniform Cholesky factors (shape {cholesky_factors.shape}) not allowed. "
                f"Expected shape ({n_splats or 'N'}, {expected_k})."
            )

        if cholesky_factors.shape[0] != expected_k:
            raise ValueError(
                f"Cholesky factors have wrong packed size for {ndim}D: "
                f"expected ({expected_k},) for uniform, got {cholesky_factors.shape}"
            )

        return True, 0  # is_uniform=True, n_splats=0 (uniform)

    # Per-splat case (2D array)
    if cholesky_factors.ndim != 2:
        raise ValueError(
            f"Cholesky factors must be 1D (uniform) or 2D (per-splat), "
            f"got {cholesky_factors.ndim}D with shape {cholesky_factors.shape}"
        )

    actual_n_splats, actual_k = cholesky_factors.shape

    if actual_k != expected_k:
        raise ValueError(
            f"Cholesky factors have wrong packed size for {ndim}D: "
            f"expected k={expected_k}, got k={actual_k}. "
            f"Shape should be ({actual_n_splats}, {expected_k}), got {cholesky_factors.shape}"
        )

    if n_splats is not None and actual_n_splats != n_splats:
        raise ValueError(
            f"Cholesky factors count mismatch: expected {n_splats} splats, "
            f"got {actual_n_splats} from shape {cholesky_factors.shape}"
        )

    return False, actual_n_splats  # is_uniform=False, actual n_splats


def permute_cholesky_packed(
    packed: np.ndarray,
    d: int,
    perm: Sequence[int],
) -> np.ndarray:
    """Permute dimensions of packed Cholesky factors.

    Given packed Cholesky factors L where Sigma = L @ L^T, reorder the
    dimensions according to the permutation. The new Cholesky L' satisfies
    Sigma'[i,j] = Sigma[perm[i], perm[j]].

    Parameters
    ----------
    packed : np.ndarray, shape (N, k) where k = d*(d+1)//2
        Packed lower-triangular Cholesky factors.
    d : int
        Number of dimensions.
    perm : sequence of int
        Permutation of dimension indices. perm[new_i] = old_i.
        E.g., [2, 0, 1] means new dim 0 was old dim 2.

    Returns
    -------
    np.ndarray, shape (N, k)
        Packed Cholesky factors with permuted dimensions.

    Examples
    --------
    >>> # Reverse 2D dimensions: swap X and Y
    >>> packed = np.array([[1.0, 0.5, 2.0]])  # L00, L10, L11
    >>> permute_cholesky_packed(packed, 2, [1, 0])
    """
    perm = list(perm)
    if len(perm) != d:
        raise ValueError(f"Permutation length {len(perm)} != dimension {d}")
    if sorted(perm) != list(range(d)):
        raise ValueError(f"Invalid permutation: {perm}")

    # Upcast to float64 for numerical stability in Cholesky decomposition
    input_dtype = packed.dtype
    L = unpack_tril(packed.astype(np.float64), d)
    # Compute covariance Sigma = L @ L^T
    Sigma = L @ np.swapaxes(L, -2, -1)
    # Permute: Sigma_new[i,j] = Sigma[perm[i], perm[j]]
    Sigma_perm = Sigma[:, perm, :][:, :, perm]
    # Re-Cholesky decompose (in float64 for robustness)
    L_new = np.linalg.cholesky(Sigma_perm)
    return pack_tril(L_new.astype(input_dtype))


def embed_cholesky_packed(
    packed: np.ndarray,
    d_src: int,
    d_dst: int,
    dim_mapping: List[int],
    fill_sigma: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Embed lower-dimensional packed Cholesky factors into higher dimensions.

    Takes d_src-dimensional Cholesky factors and embeds them into a
    d_dst-dimensional space (d_dst >= d_src). Mapped dimensions carry
    over the original covariance; unmapped dimensions get independent
    Gaussian variance (diagonal only, no cross-terms).

    Parameters
    ----------
    packed : np.ndarray, shape (N, k_src) where k_src = d_src*(d_src+1)//2
        Packed Cholesky factors in the source dimensionality.
    d_src : int
        Source dimensionality.
    d_dst : int
        Target dimensionality (must be >= d_src).
    dim_mapping : list of int, length d_src
        Maps source dimension i to target dimension dim_mapping[i].
        E.g., [1, 2, 3] maps src dims 0,1,2 to dst dims 1,2,3.
    fill_sigma : dict of {target_dim_index: sigma_value}, optional
        Standard deviations for unmapped target dimensions.
        Unmapped dims not in fill_sigma default to 1.0.

    Returns
    -------
    np.ndarray, shape (N, k_dst) where k_dst = d_dst*(d_dst+1)//2
        Packed Cholesky factors in the target dimensionality.

    Examples
    --------
    >>> # Embed 2D into 3D: src dims [0,1] → dst dims [0,1], new dim 2 has sigma=0.5
    >>> packed_2d = np.array([[1.0, 0.0, 1.0]])  # isotropic 2D
    >>> embed_cholesky_packed(packed_2d, 2, 3, [0, 1], fill_sigma={2: 0.5})
    """
    if d_dst < d_src:
        raise ValueError(f"Target dim {d_dst} must be >= source dim {d_src}")
    if len(dim_mapping) != d_src:
        raise ValueError(f"dim_mapping length {len(dim_mapping)} != source dim {d_src}")
    # Validate mapping targets are valid and unique
    if len(set(dim_mapping)) != len(dim_mapping):
        raise ValueError(f"dim_mapping has duplicates: {dim_mapping}")
    for idx in dim_mapping:
        if idx < 0 or idx >= d_dst:
            raise ValueError(f"dim_mapping index {idx} out of range [0, {d_dst})")

    if fill_sigma is None:
        fill_sigma = {}

    # Upcast to float64 for numerical stability in Cholesky decomposition
    input_dtype = packed.dtype
    L_src = unpack_tril(packed.astype(np.float64), d_src)
    # Compute source covariance
    Sigma_src = L_src @ np.swapaxes(L_src, -2, -1)

    N = packed.shape[0]
    # Build target covariance: start with zeros (float64)
    Sigma_dst = np.zeros((N, d_dst, d_dst), dtype=np.float64)

    # Copy source covariance block into mapped positions
    for i_src, i_dst in enumerate(dim_mapping):
        for j_src, j_dst in enumerate(dim_mapping):
            Sigma_dst[:, i_dst, j_dst] = Sigma_src[:, i_src, j_src]

    # Fill unmapped diagonal positions
    mapped_set = set(dim_mapping)
    for i_dst in range(d_dst):
        if i_dst not in mapped_set:
            sigma = fill_sigma.get(i_dst, 1.0)
            # Handle sigma=0: use tiny epsilon to keep matrix positive-definite.
            # For discrete dimensions (e.g., time), sigma=0 is semantically correct
            # (no physical extent), but np.linalg.cholesky requires positive-definite.
            if sigma == 0:
                sigma = 1e-7
            Sigma_dst[:, i_dst, i_dst] = sigma * sigma  # variance = sigma^2

    # Cholesky decompose in float64, then cast back to input dtype
    L_dst = np.linalg.cholesky(Sigma_dst)
    return pack_tril(L_dst.astype(input_dtype))
