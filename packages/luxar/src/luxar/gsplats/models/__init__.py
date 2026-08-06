"""Gaussian splat models, rendering, and numerical utilities.

Re-exports key public symbols for convenience::

    from luxar.gsplats.models import GaussianSplatModel, render_gaussians
"""

from luxar.gsplats.models.gsplats import (
    GaussianSplatModel,
    render_gaussians,
    render_gaussians_numpy,
    render_gaussians_pytorch,
)
from luxar.gsplats.models.utils.inverse_softplus import (
    stable_inverse_softplus,
    stable_inverse_softplus_torch,
)
from luxar.gsplats.models.utils.lt_solver import solve_lower_triangular

__all__ = [
    # Core model
    "GaussianSplatModel",
    # Rendering
    "render_gaussians",
    "render_gaussians_numpy",
    "render_gaussians_pytorch",
    # Numerical utilities
    "stable_inverse_softplus",
    "stable_inverse_softplus_torch",
    "solve_lower_triangular",
]
