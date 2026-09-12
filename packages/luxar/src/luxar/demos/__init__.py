"""Luxar Demos - Self-contained demonstration scripts.

This package contains executable demo scripts that showcase Luxar's capabilities.
Each demo keeps the code that makes it distinctive in its own file, and reaches
into this package for the shared plumbing helpers re-exported below.

To run a demo:
    hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py
    hatch run python packages/luxar/src/luxar/demos/demo_cubic_array.py

See demos/README.md for more information on creating new demos.
"""

from ..utils.colors import hsv_to_rgb, stack_colorings
from ..utils.scenes import (
    create_lorenz_attractor,
    create_random_spheres,
    create_time_series_demo,
)
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
from ._support._fields import (
    FlowField,
    add_reference_cube_to_scene,
    cubic_bounds,
    rk4_step,
    trilinear_vector,
    unit_flow,
)
from ._support.datasets.bundles import (
    BundleMemberNotFound,
    load_dataset_bundle,
    load_precomputed_bundle,
    load_precomputed_gsplats,
)
from ._support.datasets.cache import cache_computed, cached_download
from ._support.datasets.data_fetch import (
    LOCAL_FIT_DIRNAME,
    DatasetNotFound,
    DatasetUnavailable,
    LocalComputeDataset,
    ResolvedDataset,
    dataset_spec,
    declared_file_names,
    ensure_dataset,
    load_dataset_gsplats,
    load_local_fit_gsplats,
    load_local_fit_gsplats_at,
    load_manifest,
    local_fit_path,
)
from ._support.datasets.lfs import is_lfs_pointer, require_local_data
from ._support.datasets.payload_agreement import voxel_sampled_payload_agreement
from ._support.downloads.download import (
    QUARANTINE_SUFFIX,
    download_with_checksum,
    find_quarantined_files,
    format_quarantine_notice,
    quarantine_file,
    robust_download,
    warn_if_quarantined,
)
from ._support.downloads.remote_zip import download_zip_member
from ._support.runtime.cli import run_luxar_cli
from ._support.runtime.device import detect_device, warn_if_no_cuda_gpu
from ._support.runtime.flags import (
    control_serve_args,
    parse_demo_flags,
    parse_int_arg,
    parse_path_arg,
    parse_str_arg,
)
from ._support.runtime.provenance import (
    BUILDER_FINGERPRINT_ATTR,
    INPUT_DIGESTS_ATTR,
    demo_source_fingerprint,
    print_data_provenance,
    scene_is_current,
    stamp_input_digests,
)
from ._support.runtime.viewer import launch_viewer


def window_attrs(window: tuple[float, float]) -> dict[str, float]:
    """Intensity and offset storing a Layers-panel display window."""
    lo, hi = window
    return {"intensity": 1.0 / (hi - lo), "offset": -lo / (hi - lo)}


__all__ = [
    "BUILDER_FINGERPRINT_ATTR",
    "INSTALL_SPECS",
    "INPUT_DIGESTS_ATTR",
    "LOCAL_FIT_DIRNAME",
    "BundleMemberNotFound",
    "DatasetNotFound",
    "DatasetUnavailable",
    "DependencySpec",
    "DependencyStatus",
    "FlowField",
    "LocalComputeDataset",
    "MissingDependencyError",
    "QUARANTINE_SUFFIX",
    "ResolvedDataset",
    "add_demo_caption",
    "add_reference_cube_to_scene",
    "cache_computed",
    "cached_download",
    "create_lorenz_attractor",
    "create_random_spheres",
    "create_time_series_demo",
    "cubic_bounds",
    "dataset_spec",
    "declared_file_names",
    "demo_source_fingerprint",
    "detect_device",
    "download_with_checksum",
    "download_zip_member",
    "ensure_dataset",
    "extras_for",
    "find_quarantined_files",
    "format_demo_caption",
    "format_quarantine_notice",
    "hsv_to_rgb",
    "is_installed",
    "is_lfs_pointer",
    "control_serve_args",
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
    "parse_str_arg",
    "print_data_provenance",
    "quarantine_file",
    "require_local_data",
    "require_module",
    "rk4_step",
    "robust_download",
    "run_luxar_cli",
    "scene_is_current",
    "stack_colorings",
    "stamp_input_digests",
    "substitutive_lod_or_flat",
    "survey",
    "trilinear_vector",
    "unit_flow",
    "voxel_sampled_payload_agreement",
    "warn_if_no_cuda_gpu",
    "warn_if_quarantined",
    "window_attrs",
]
