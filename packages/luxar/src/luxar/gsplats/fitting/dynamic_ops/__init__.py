# dynamic_ops/__init__.py
"""
Fixed-Pool Splat Relocation Operations

This package implements fixed-pool splat relocation for adaptive Gaussian splatting.
Instead of adding/removing splats, weak splats are relocated to high-residual regions.
This enables use of standard PyTorch Adam optimizer for much faster optimization.
"""

from luxar.gsplats.fitting.dynamic_ops.config import DynamicOpsConfig
from luxar.gsplats.fitting.dynamic_ops.operations import (
    _calculate_splat_importance,
    _select_weak_splats,
    apply_dynamic_operations,
)
from luxar.gsplats.fitting.dynamic_ops.peak_finding import _find_residual_peaks

__all__ = [
    "DynamicOpsConfig",
    "apply_dynamic_operations",
    # Exported for testing
    "_find_residual_peaks",
    "_calculate_splat_importance",
    "_select_weak_splats",
]
