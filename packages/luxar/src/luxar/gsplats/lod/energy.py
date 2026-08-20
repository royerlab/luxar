"""Lightweight artifact-local Gaussian self-energy measurements."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from luxar.gsplats.utils.alpha import effective_amplitudes

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


def total_self_energy(data: "GSplatData") -> float:
    """Exact ``Σ aᵢ²·π^(D/2)·|Σᵢ|^(1/2)`` in O(N), without Torch."""
    if data.n_splats == 0:
        return 0.0
    amplitudes = np.asarray(effective_amplitudes(data), dtype=np.float64)
    diagonal = data._cholesky_diag_elements().astype(np.float64)
    sqrt_determinant = np.abs(np.prod(diagonal, axis=1))
    return float(
        np.sum(amplitudes**2 * sqrt_determinant) * math.pi ** (data.ndim / 2.0)
    )
