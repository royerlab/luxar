"""Luxar Demos - Self-contained demonstration scripts.

This package contains executable demo scripts that showcase Luxar's capabilities.
Each demo keeps the code that makes it distinctive in its own file, and reaches
into this package for the shared plumbing helpers re-exported below.

To run a demo:
    hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py
    hatch run python packages/luxar/src/luxar/demos/demo_cubic_array.py

See demos/README.md for more information on creating new demos.
"""

from ..utils.bundles import (
    BundleMemberNotFound,
    load_dataset_bundle,
    load_precomputed_bundle,
    load_precomputed_gsplats,
)
from ..utils.cache import cache_computed, cached_download
from ..utils.colors import hsv_to_rgb, stack_colorings
from ..utils.data_fetch import (
    LOCAL_FIT_DIRNAME,
    DatasetNotFound,
    DatasetUnavailable,
    LocalComputeDataset,
    dataset_spec,
    ensure_dataset,
    load_dataset_gsplats,
    load_local_fit_gsplats,
    load_local_fit_gsplats_at,
    load_manifest,
    local_fit_path,
)
from ..utils.device import detect_device, warn_if_no_cuda_gpu
from ..utils.flags import parse_demo_flags, parse_int_arg, parse_path_arg
from ..utils.lfs import is_lfs_pointer, require_local_data
from ..utils.payload_agreement import voxel_sampled_payload_agreement
from ..utils.provenance import (
    BUILDER_FINGERPRINT_ATTR,
    demo_source_fingerprint,
    print_data_provenance,
    scene_is_current,
)
from ..utils.scenes import (
    create_lorenz_attractor,
    create_random_spheres,
    create_time_series_demo,
)
from ..utils.viewer import launch_viewer
from ._caption import add_demo_caption, format_demo_caption
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
    "BUILDER_FINGERPRINT_ATTR",
    "INSTALL_SPECS",
    "LOCAL_FIT_DIRNAME",
    "BundleMemberNotFound",
    "DatasetNotFound",
    "DatasetUnavailable",
    "DependencySpec",
    "DependencyStatus",
    "LocalComputeDataset",
    "MissingDependencyError",
    "add_demo_caption",
    "cache_computed",
    "cached_download",
    "create_lorenz_attractor",
    "create_random_spheres",
    "create_time_series_demo",
    "dataset_spec",
    "demo_source_fingerprint",
    "detect_device",
    "ensure_dataset",
    "extras_for",
    "format_demo_caption",
    "hsv_to_rgb",
    "is_installed",
    "is_lfs_pointer",
    "launch_viewer",
    "load_dataset_bundle",
    "load_dataset_gsplats",
    "load_local_fit_gsplats",
    "load_local_fit_gsplats_at",
    "load_manifest",
    "load_precomputed_bundle",
    "load_precomputed_gsplats",
    "local_fit_path",
    "parse_demo_flags",
    "parse_int_arg",
    "parse_path_arg",
    "print_data_provenance",
    "require_local_data",
    "require_module",
    "scene_is_current",
    "stack_colorings",
    "substitutive_lod_or_flat",
    "survey",
    "voxel_sampled_payload_agreement",
    "warn_if_no_cuda_gpu",
]
