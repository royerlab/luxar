"""
Specialized optimizers for Gaussian splatting.
"""

from .integration import ModelOptimizerCoordinator, create_per_splat_optimizer_setup
from .per_splat_adam import PerSplatAdam
from .per_splat_scheduler import PerSplatExponentialLR, PerSplatReduceLROnPlateau

__all__ = [
    "PerSplatAdam",
    "PerSplatReduceLROnPlateau",
    "PerSplatExponentialLR",
    "ModelOptimizerCoordinator",
    "create_per_splat_optimizer_setup",
]
