"""Utility functions for Luxar.

Note: As of v1.4.0, scalar broadcasting is handled by ArrayEncoder in luxar.encoding.
The broadcast_*_to_points() functions have been removed (obsolete).
"""

from .array import (
    ensure_float32,
    validate_array_shape,
)
from .atomic_copy import atomic_copy_file, atomic_copytree
from .demos import (
    create_lorenz_attractor,
    create_random_spheres,
    create_time_series_demo,
)
from .paths import (
    get_datasets_dir,
    get_demos_output_dir,
    get_examples_output_dir,
    get_project_root,
)

__all__ = [
    # From array
    "ensure_float32",
    "validate_array_shape",
    # From atomic_copy
    "atomic_copy_file",
    "atomic_copytree",
    # From demos
    "create_lorenz_attractor",
    "create_random_spheres",
    "create_time_series_demo",
    # From paths
    "get_datasets_dir",
    "get_demos_output_dir",
    "get_examples_output_dir",
    "get_project_root",
]
