"""Lightweight artifact-local Gaussian self-energy measurements.

The viewer combines each level's measured approximation quality ``Q`` with its
additive-prefix energy fraction ``e(k)``. Partition aggregation weights those
qualities by the finest content's self-energy ``w``. Amplitudes use the same
alpha-effective ``A·α`` convention as rendering, so transparent splats carry no
energy weight.

This module re-derives ``Σ aᵢ²·π^(D/2)·|Σᵢ|^(1/2)`` instead of importing
``lod._kernels.gaussian_self_energy_numpy`` because ``_kernels`` imports Torch at
module scope and ordinary filtering/reduction paths must remain Torch-free.
"""

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
