"""
Utilities for Gaussian splat fitting.

This package provides utility functions for:
- Lower-triangular matrix operations (pack/unpack Cholesky factors)
- Gradient dilution compensation for higher-dimensional optimization
- Triangle matrix size calculations

For detailed documentation, see SPECIFICATIONS.md.
"""

from luxar.gsplats.utils.trils import (
    calculate_gradient_dilution_factor,
    pack_tril,
    tril_size,
    unpack_tril,
)

__all__ = [
    "tril_size",
    "calculate_gradient_dilution_factor",
    "pack_tril",
    "unpack_tril",
]
