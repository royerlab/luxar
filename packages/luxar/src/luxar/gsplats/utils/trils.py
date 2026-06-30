# -------------------------------
# Pack / unpack lower-triangular matrices
# -------------------------------
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    - 3D: 3 + 6 = 9 params, factor = 9/5 = 1.8

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
    - 4D: 3.0 * 3.5 / 5 = 2.1 (d^0.8 approx 3.0, params = 4+10 = 14)
    - 6D: 4.2 * 5.2 / 5 = 4.4 (d^0.8 approx 4.2, params = 6+21 = 27)
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
    rows, cols = np.tril_indices(d)
    return L[:, rows, cols]


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
    L = np.zeros((N, d, d), dtype=v.dtype)
    rows, cols = np.tril_indices(d)
    L[:, rows, cols] = v
    return L


def diag_indices(d: int) -> np.ndarray:
    """Packed-vector positions of the diagonal elements of a d×d tril matrix.

    For row-major lower-triangular packing
    ``[L00, L10, L11, L20, L21, L22, ...]`` the diagonal element ``(i, i)``
    lives at packed position ``(i+1)*(i+2)//2 - 1``.

    Parameters
    ----------
    d : int
        Dimension of the square matrix.

    Returns
    -------
    np.ndarray, shape (d,)
        Integer positions of the diagonal elements within the packed vector.

    Examples
    --------
    >>> diag_indices(3)
    array([0, 2, 5])
    """
    return np.cumsum(np.arange(1, d + 1)) - 1


def offdiag_indices(d: int) -> np.ndarray:
    """Packed-vector positions of the off-diagonal (strictly lower) elements.

    Complement of :func:`diag_indices` within ``range(tril_size(d))``,
    preserving the row-major lower-triangular order. Empty for ``d == 1``.

    Parameters
    ----------
    d : int
        Dimension of the square matrix.

    Returns
    -------
    np.ndarray, shape (d*(d-1)//2,)
        Integer positions of the off-diagonal elements within the packed vector.

    Examples
    --------
    >>> offdiag_indices(3)
    array([1, 3, 4])
    """
    mask = np.ones(tril_size(d), dtype=bool)
    mask[diag_indices(d)] = False
    return np.nonzero(mask)[0]


def split_tril(packed: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split packed Cholesky factors into diagonal and off-diagonal parts.

    The diagonal of a Cholesky factor is positive and scale-like while the
    off-diagonal is signed and zero-centred; splitting them lets each be
    encoded/quantised independently on disk. Operates on the last axis, so
    it accepts per-splat ``(N, k)``, broadcast ``(1, k)`` and uniform
    ``(k,)`` inputs alike.

    Parameters
    ----------
    packed : np.ndarray, shape (..., k) where k = d*(d+1)//2
        Packed lower-triangular Cholesky factors (row-major).
    d : int
        Number of dimensions.

    Returns
    -------
    diag : np.ndarray, shape (..., d)
        Diagonal elements in dimension order.
    offdiag : np.ndarray, shape (..., d*(d-1)//2)
        Off-diagonal elements in row-major lower-triangular order
        (empty trailing axis when ``d == 1``).

    See Also
    --------
    merge_tril : inverse operation.
    """
    expected_k = tril_size(d)
    if packed.shape[-1] != expected_k:
        raise ValueError(
            f"Packed Cholesky factors have wrong size for {d}D: "
            f"expected last axis {expected_k}, got shape {packed.shape}"
        )
    return packed[..., diag_indices(d)], packed[..., offdiag_indices(d)]


def merge_tril(diag: np.ndarray, offdiag: np.ndarray, d: int) -> np.ndarray:
    """Recombine diagonal and off-diagonal parts into packed Cholesky factors.

    Inverse of :func:`split_tril`. Scatters the two column groups back to
    their row-major lower-triangular positions. Operates on the last axis.

    Parameters
    ----------
    diag : np.ndarray, shape (..., d)
        Diagonal elements (as returned by :func:`split_tril`).
    offdiag : np.ndarray, shape (..., d*(d-1)//2)
        Off-diagonal elements (as returned by :func:`split_tril`).
    d : int
        Number of dimensions.

    Returns
    -------
    np.ndarray, shape (..., d*(d+1)//2)
        Packed lower-triangular Cholesky factors (row-major).
    """
    expected_diag = d
    expected_off = tril_size(d) - d
    if diag.shape[-1] != expected_diag:
        raise ValueError(
            f"Diagonal part has wrong size for {d}D: "
            f"expected last axis {expected_diag}, got shape {diag.shape}"
        )
    if offdiag.shape[-1] != expected_off:
        raise ValueError(
            f"Off-diagonal part has wrong size for {d}D: "
            f"expected last axis {expected_off}, got shape {offdiag.shape}"
        )
    out = np.empty(
        diag.shape[:-1] + (tril_size(d),),
        dtype=np.result_type(diag.dtype, offdiag.dtype),
    )
    out[..., diag_indices(d)] = diag
    out[..., offdiag_indices(d)] = offdiag
    return out


def recombine_cholesky(
    decode: Callable[[str], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    """Recombine on-disk Cholesky factors into the packed ``(N, k)`` form.

    Single source of truth for the read side of the v3.1 split layout, shared by
    every reader (the scene reader and the gsplat-tree decoder) so the version
    handling, corruption invariant, and error message live in ONE place.

    ``decode(name)`` returns the named array decoded to a NumPy array, or
    ``None`` when that array is absent from the store. Two layouts are handled:

    - **v3.1 split**: ``cholesky_factors_diag`` ``(N, d)`` +
      ``cholesky_factors_offdiag`` ``(N, k-d)`` → merged via :func:`merge_tril`.
    - **v3.0 single**: ``cholesky_factors`` ``(N, k)`` → returned as-is (the
      fallback taken when no diagonal array is present).

    The off-diagonal array is legitimately absent ONLY for 1D gsplats (no
    off-diagonal terms); for ``d > 1`` its absence means a corrupt or
    partially-written store and raises ``ValueError`` rather than silently
    dropping every splat's off-diagonal covariance. Returns ``None`` when no
    Cholesky array is present at all (matching the legacy single-array reader).

    Parameters
    ----------
    decode : Callable[[str], Optional[np.ndarray]]
        Resolves an array name to its decoded values, or ``None`` if absent.

    Returns
    -------
    np.ndarray or None
        Packed lower-triangular Cholesky factors, or ``None`` if no Cholesky
        array exists in the store.
    """
    diag = decode("cholesky_factors_diag")
    if diag is None:
        # v3.0 single packed array (or None if the store has no Cholesky at all).
        return decode("cholesky_factors")
    ndim = diag.shape[-1]  # the diagonal has exactly d columns
    offdiag = decode("cholesky_factors_offdiag")
    if offdiag is None:
        if ndim > 1:
            raise ValueError(
                f"Cholesky 'cholesky_factors_offdiag' missing for {ndim}D "
                "splats; the .gsplats.zarr is corrupt or partially written."
            )
        # 1D gsplats: no off-diagonal terms were written.
        offdiag = np.empty(diag.shape[:-1] + (0,), dtype=diag.dtype)
    return merge_tril(diag, offdiag, ndim)


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

    # Cholesky decompose in float64, then cast back to input dtype.
    # Some splats may have degenerate covariance matrices (e.g. from
    # numerical issues during fitting), so we selectively regularise
    # only the failing matrices when the batch cholesky fails.
    try:
        L_dst = np.linalg.cholesky(Sigma_dst)
    except np.linalg.LinAlgError:
        # Identify which matrices are not positive-definite by checking
        # eigenvalues (vectorised, much faster than per-splat cholesky).
        eig_min = np.linalg.eigvalsh(Sigma_dst)[:, 0]  # smallest eigenvalue
        bad_mask = eig_min <= 0
        n_bad = int(bad_mask.sum())

        # If eigenvalues all look positive but Cholesky still failed,
        # the matrices are near-singular.  Regularise everything lightly.
        if n_bad == 0:
            for k in range(d_dst):
                Sigma_dst[:, k, k] += 1e-6
        else:
            # Regularise only the bad matrices: add enough to make min
            # eigenvalue positive (with margin).
            deficit = np.abs(eig_min[bad_mask]) + 1e-6
            for k in range(d_dst):
                Sigma_dst[bad_mask, k, k] += deficit

        try:
            L_dst = np.linalg.cholesky(Sigma_dst)
        except np.linalg.LinAlgError:
            # Last resort: per-splat decomposition for all matrices.
            L_dst = np.empty_like(Sigma_dst)
            for i in range(N):
                try:
                    L_dst[i] = np.linalg.cholesky(Sigma_dst[i])
                except np.linalg.LinAlgError:
                    L_dst[i] = np.eye(d_dst) * 1e-3
    return pack_tril(L_dst.astype(input_dtype))
