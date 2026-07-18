"""Interoperability adapters between Luxar Gaussian splats and external tools.

Provides readers for the classical (photogrammetric) Gaussian-splat file
formats — INRIA ``point_cloud.ply``, antimatter15 ``.splat``, Niantic
``.spz``, and SuperSplat compressed ``.ply`` — plus a bridge to `tracksdata
<https://github.com/royerlab/tracksdata>`_ (the Royer-lab multi-object-tracking
data structure). Adapters import optional external dependencies lazily, so
this package imports cleanly without the extras installed.
"""

from luxar.gsplats.interop.classical_splats import (
    CLASSICAL_FORMATS,
    ClassicalSplats,
    classical_to_gsplat_data,
    detect_classical_format,
    import_gsplats,
    quat_to_rotmat,
    read_antimatter_splat,
    read_inria_ply,
    read_sog,
    read_spz,
    read_supersplat_ply,
    rotmat_to_quat,
)
from luxar.gsplats.interop.inria_export import (
    export_inria_ply,
    gsplat_data_to_inria_ply,
)
from luxar.gsplats.interop.tracksdata import (
    gsplats_to_tracksdata_graph,
    splat_mask_and_bbox,
)

__all__ = [
    "CLASSICAL_FORMATS",
    "ClassicalSplats",
    "classical_to_gsplat_data",
    "detect_classical_format",
    "export_inria_ply",
    "gsplat_data_to_inria_ply",
    "gsplats_to_tracksdata_graph",
    "import_gsplats",
    "quat_to_rotmat",
    "read_antimatter_splat",
    "read_inria_ply",
    "read_sog",
    "read_spz",
    "read_supersplat_ply",
    "rotmat_to_quat",
    "splat_mask_and_bbox",
]
