"""Utility functions for Luxar."""

from .array import (
    broadcast_color_to_points,
    broadcast_radii_to_points,
    broadcast_scalar_to_points,
    broadcast_sharpness_to_points,
    ensure_float32,
    validate_array_shape,
)
from .demos import (
    create_lorenz_attractor,
    create_random_spheres,
    create_time_series_demo,
)

__all__ = [
    # From array
    "broadcast_color_to_points",
    "broadcast_radii_to_points",
    "broadcast_scalar_to_points",
    "broadcast_sharpness_to_points",
    "ensure_float32",
    "validate_array_shape",
    # From demos
    "create_lorenz_attractor",
    "create_random_spheres",
    "create_time_series_demo",
]
