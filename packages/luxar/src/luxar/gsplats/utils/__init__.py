"""
Utilities for Gaussian splat fitting.

This package provides utility functions for:
- Lower-triangular matrix operations (pack/unpack Cholesky factors)
- Cholesky factor validation for Gaussian splats
- Cholesky dimension permutation and embedding for cross-dimensional scenes
- Gradient dilution compensation for higher-dimensional optimization
- Triangle matrix size calculations
"""

from luxar.gsplats.utils.device import is_mps_available, resolve_torch_device
from luxar.gsplats.utils.trils import (
    calculate_gradient_dilution_factor,
    diag_indices,
    embed_cholesky_packed,
    merge_tril,
    offdiag_indices,
    pack_tril,
    permute_cholesky_packed,
    split_tril,
    tril_size,
    unpack_tril,
    validate_cholesky_shape,
)

__all__ = [
    "is_mps_available",
    "resolve_torch_device",
    "tril_size",
    "calculate_gradient_dilution_factor",
    "pack_tril",
    "unpack_tril",
    "validate_cholesky_shape",
    "permute_cholesky_packed",
    "embed_cholesky_packed",
    "diag_indices",
    "offdiag_indices",
    "split_tril",
    "merge_tril",
]
