"""Utility functions for Luxar.

Note: As of v1.4.0, scalar broadcasting is handled by ArrayEncoder in luxar.encoding.
The broadcast_*_to_points() functions have been removed (obsolete).
"""

from .array import (
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
    "ensure_float32",
    "validate_array_shape",
    # From demos
    "create_lorenz_attractor",
    "create_random_spheres",
    "create_time_series_demo",
]
