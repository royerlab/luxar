"""Mesh-specific tooling that sits beside the ``Mesh`` node type.

Mirrors :mod:`luxar.gsplats`: the node class itself lives in :mod:`luxar.core.mesh`,
while everything *about* meshes that is not the scene-graph object lives here. Today
that is :mod:`luxar.mesh.interop`, which imports classical mesh files.

Deliberately does NOT re-export ``Mesh``. ``luxar.mesh`` and ``luxar.core.mesh`` are
distinct modules, and pulling the node class up here would make this package import
``luxar.core`` — an edge that buys nothing and risks a cycle, since ``luxar.core``
already reaches a great deal of the package.
"""

from .interop import TriangleMesh, detect_mesh_format, import_mesh

__all__ = ["TriangleMesh", "detect_mesh_format", "import_mesh"]
