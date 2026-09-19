"""Mesh-specific tooling that sits beside the ``Mesh`` node type.

Mirrors :mod:`luxar.gsplats`: the node class itself lives in :mod:`luxar.core.mesh`,
while everything *about* meshes that is not the scene-graph object lives here. Today
that is :mod:`luxar.mesh.interop`, which imports classical mesh files;
:mod:`luxar.mesh.split`, the by-face re-indexing behind ``add_mesh(partition=…)``;
:mod:`luxar.mesh.decimate`, which produces substitutive mesh LOD levels; and
:mod:`luxar.mesh.primitives`, analytic shapes (the welded ``icosphere``) for demos
and marker geometry.

Deliberately does NOT re-export ``Mesh``. ``luxar.mesh`` and ``luxar.core.mesh`` are
distinct modules, and pulling the node class up here would make this package import
``luxar.core`` — an edge that buys nothing and risks a cycle, since ``luxar.core``
already reaches a great deal of the package.
"""

from .decimate import (
    DECIMATION_METHODS,
    DecimatedMesh,
    decimate,
    decimate_cluster,
    decimate_ladder,
    resolve_decimation_method,
)
from .interop import (
    TriangleMesh,
    detect_mesh_format,
    import_mesh,
    import_mesh_directory,
)
from .primitives import icosphere

__all__ = [
    "DECIMATION_METHODS",
    "DecimatedMesh",
    "TriangleMesh",
    "decimate",
    "decimate_cluster",
    "decimate_ladder",
    "detect_mesh_format",
    "icosphere",
    "import_mesh",
    "import_mesh_directory",
    "resolve_decimation_method",
]
