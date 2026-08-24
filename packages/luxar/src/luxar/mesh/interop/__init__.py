"""Classical mesh-format interchange (PLY / OBJ / STL / VTP / glTF → Luxar).

Everything here is NumPy + stdlib — no new dependency, matching
:mod:`luxar.gsplats.interop`, so ``luxar mesh import`` works on a bare
``pip install luxar``. The VTK XML reader holds to the same bar: ``xml.etree`` +
``base64`` + ``zlib``, all standard library.
"""

from .mesh_import import (
    MESH_FORMATS,
    TriangleMesh,
    detect_mesh_format,
    import_mesh,
)

__all__ = [
    "MESH_FORMATS",
    "TriangleMesh",
    "detect_mesh_format",
    "import_mesh",
]
