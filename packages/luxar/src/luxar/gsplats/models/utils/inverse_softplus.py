# -------------------------------
# Small utilities
# -------------------------------
import numpy as np


def stable_inverse_softplus(y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Compute numerically stable inverse of softplus function.

    The softplus function is softplus(x) = (1/beta) * log(1 + exp(beta*x)).
    This function computes its inverse: x such that softplus(x) = y.

    Uses expm1 for numerical stability when computing exp(beta*y) - 1,
    which avoids catastrophic cancellation for small y values.

    Parameters
    ----------
    y : np.ndarray
        Input values (must be positive since softplus range is (0, inf)).
    beta : float, default=1.0
        Softplus scaling parameter. Higher values make function steeper.

    Returns
    -------
    np.ndarray
        Inverse softplus values with same shape as input.

    Notes
    -----
    Mathematical relationship:
        softplus(x) = (1/beta) * log(1 + exp(beta*x))
        inverse_softplus(y) = (1/beta) * log(exp(beta*y) - 1)
                            = (1/beta) * log(expm1(beta*y))  # numerically stable
    """
    y = np.asarray(y)
    original_dtype = y.dtype

    # Convert to float for computation (preserving original precision)
    if y.dtype == np.float64:
        y = y.astype(np.float64)
    else:
        y = y.astype(np.float32)

    # Input validation
    if np.any(y <= 0):
        import warnings

        warnings.warn(
            "Inverse softplus input contains non-positive values", RuntimeWarning
        )

    # Compute inverse using numerically stable formula:
    # For large values, use asymptotic approximation to avoid overflow
    beta_y = beta * y

    # For large beta*y (>= 50), use asymptotic approximation: log(exp(z) - 1) ≈ z
    large_mask = beta_y >= 50.0
    result = np.zeros_like(y)

    # For large values: inverse_softplus(y) ≈ y (asymptotically)
    result[large_mask] = y[large_mask]

    # For normal values: use expm1 for numerical stability
    small_mask = ~large_mask
    if np.any(small_mask):
        result[small_mask] = np.log(np.expm1(beta_y[small_mask])) / beta

    # Convert back to original dtype
    return result.astype(original_dtype)
