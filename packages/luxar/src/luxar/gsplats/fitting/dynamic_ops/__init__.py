# dynamic_ops/__init__.py
"""
Dynamic Gaussian Splat Operations

This package implements convergence-driven dynamic operations for adaptive Gaussian splatting,
including tile-based seeding for spatial fairness and principled pruning.
"""

from luxar.gsplats.fitting.dynamic_ops.config import DynamicOpsConfig
from luxar.gsplats.fitting.dynamic_ops.operations import (
    _boost_splat_learning_rate,
    _calculate_splat_importance,
    _select_pruning_candidates,
    apply_dynamic_operations,
)
from luxar.gsplats.fitting.dynamic_ops.peak_finding import _find_residual_peaks

__all__ = [
    "DynamicOpsConfig",
    "apply_dynamic_operations",
    # Exported for testing
    "_find_residual_peaks",
    "_boost_splat_learning_rate",
    "_calculate_splat_importance",
    "_select_pruning_candidates",
]
