"""
Utilities for Gaussian splat fitting.

This package provides utility functions for:
- Lower-triangular matrix operations (pack/unpack Cholesky factors)
- Cholesky factor validation for Gaussian splats
- Cholesky dimension permutation and embedding for cross-dimensional scenes
- Gradient dilution compensation for higher-dimensional optimization
- Triangle matrix size calculations

For detailed documentation, see SPECIFICATIONS.md.
"""

from luxar.gsplats.utils.trils import (
    calculate_gradient_dilution_factor,
    embed_cholesky_packed,
    pack_tril,
    permute_cholesky_packed,
    tril_size,
    unpack_tril,
    validate_cholesky_shape,
)

__all__ = [
    "tril_size",
    "calculate_gradient_dilution_factor",
    "pack_tril",
    "unpack_tril",
    "validate_cholesky_shape",
    "permute_cholesky_packed",
    "embed_cholesky_packed",
]
