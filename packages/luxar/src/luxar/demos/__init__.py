"""Luxar Demos - Self-contained demonstration scripts.

This package contains executable demo scripts that showcase Luxar's capabilities.
Each demo is completely self-contained with all generation code in a single file.

To run a demo:
    hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py
    hatch run python packages/luxar/src/luxar/demos/demo_cubic_array.py

See demos/README.md for more information on creating new demos.
"""

from ..utils.data_fetch import (
    DatasetNotFound,
    LocalComputeDataset,
    dataset_spec,
    ensure_dataset,
    load_dataset_gsplats,
    load_manifest,
)
from ..utils.demos import (
    cache_computed,
    cached_download,
    create_lorenz_attractor,
    create_random_spheres,
    create_time_series_demo,
    detect_device,
    hsv_to_rgb,
    launch_viewer,
    load_precomputed_bundle,
    load_precomputed_gsplats,
    parse_demo_flags,
    parse_int_arg,
    require_local_data,
    stack_colorings,
    warn_if_no_cuda_gpu,
)
from ._dependencies import (
    INSTALL_SPECS,
    DependencySpec,
    DependencyStatus,
    MissingDependencyError,
    extras_for,
    is_installed,
    require_module,
    substitutive_lod_or_flat,
    survey,
)

__all__ = [
    "INSTALL_SPECS",
    "DatasetNotFound",
    "DependencySpec",
    "DependencyStatus",
    "LocalComputeDataset",
    "MissingDependencyError",
    "cache_computed",
    "cached_download",
    "create_lorenz_attractor",
    "create_random_spheres",
    "create_time_series_demo",
    "dataset_spec",
    "detect_device",
    "ensure_dataset",
    "extras_for",
    "hsv_to_rgb",
    "is_installed",
    "launch_viewer",
    "load_dataset_gsplats",
    "load_manifest",
    "load_precomputed_bundle",
    "load_precomputed_gsplats",
    "parse_demo_flags",
    "parse_int_arg",
    "require_local_data",
    "require_module",
    "stack_colorings",
    "substitutive_lod_or_flat",
    "survey",
    "warn_if_no_cuda_gpu",
]
