# -------------------------------
# Pack / unpack lower-triangular matrices
# -------------------------------
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
    - 2D/3D: Conservative scaling based on parameter count
    - 4D+: More aggressive scaling combining spatial and parameter complexity
    """
    # Calculate number of parameters per splat
    params_2d = 2 + tril_size(2)  # 5 parameters (baseline)
    params_current = d + tril_size(d)  # Current dimension parameters

    if d <= 3:
        # Conservative scaling for 2D/3D (maintain existing quality)
        gradient_dilution_factor = params_current / params_2d
    else:
        # More aggressive scaling for 4D+ (address empirical findings)
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
    return out  # type: ignore[no-any-return]


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
    return L  # type: ignore[no-any-return]
