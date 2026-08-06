"""Gaussian splat model and rendering functions."""

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.models.gsplats.rendering_core import (
    cached_base_and_offsets,
    calculate_optimal_chunk_size,
    clear_grid_cache,
    compute_aabb_with_intensity_floor,
    fwd_norm2_2d,
    fwd_norm2_3d,
    get_grid_cache_stats,
    group_by_box,
    group_by_box_gpu,
    linear_strides,
    render_gaussians,
)
from luxar.gsplats.models.gsplats.rendering_wrappers import (
    render_gaussians_numpy,
    render_gaussians_pytorch,
)

__all__ = [
    "GaussianSplatModel",
    "cached_base_and_offsets",
    "calculate_optimal_chunk_size",
    "clear_grid_cache",
    "compute_aabb_with_intensity_floor",
    "fwd_norm2_2d",
    "fwd_norm2_3d",
    "get_grid_cache_stats",
    "group_by_box",
    "group_by_box_gpu",
    "linear_strides",
    "render_gaussians",
    "render_gaussians_numpy",
    "render_gaussians_pytorch",
]
