"""Luxar Demos - Self-contained demonstration scripts.

This package contains executable demo scripts that showcase Luxar's capabilities.
Each demo is completely self-contained with all generation code in a single file.

To run a demo:
    hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py
    hatch run python packages/luxar/src/luxar/demos/demo_cubic_array.py

See demos/README.md for more information on creating new demos.
"""

from ..utils.demos import (
    create_lorenz_attractor,
    create_random_spheres,
    create_time_series_demo,
    detect_device,
    launch_viewer,
    load_precomputed_bundle,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)

__all__ = [
    "create_lorenz_attractor",
    "create_random_spheres",
    "create_time_series_demo",
    "detect_device",
    "launch_viewer",
    "load_precomputed_bundle",
    "load_precomputed_gsplats",
    "parse_demo_flags",
    "warn_if_no_cuda_gpu",
]
