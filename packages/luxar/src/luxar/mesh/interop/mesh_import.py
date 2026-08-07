"""Import classical mesh files into a Luxar-ready triangle mesh.

The structural sibling of :mod:`luxar.gsplats.interop.classical_splats`, and
deliberately the same shape: one frozen intermediate that validates its own invariants,
one hand-written sniffer, a ``_READERS`` dispatch dict keyed by the same names as
:data:`MESH_FORMATS`, and an :func:`import_mesh` that does exists-check → sniff →
validate → read → normalize.

It diverges from that module in one visible way. ``classical_splats`` keeps all five
readers inline; here each format lives in its own private module, because each is
120–330 lines of independent parsing with no shared decode step. (An earlier
version of this note claimed ruff's ``max-complexity = 10`` "would force the
extraction anyway". It would not: ``C901`` is not in ``[tool.ruff.lint] select``,
so that setting is inert — see the comment beside it in ``pyproject.toml``. The
split stands on the readers being independent, which is reason enough.)

**There is no ``MeshData`` here on purpose.** ``luxar.io.reader`` already defines a
``MeshData`` for reading a written node back; the intermediate a reader produces is a
different thing, so it is :class:`TriangleMesh`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ._gltf import read_gltf
from ._obj import read_obj
from ._ply_mesh import parse_ply_header, read_ply_mesh
from ._stl import is_binary_stl, read_stl
from ._weld import drop_degenerate_faces, fan_triangulate, weld_vertices

#: Formats accepted by :func:`import_mesh`'s ``format`` argument.
MESH_FORMATS = ("ply", "obj", "stl", "gltf")


@dataclass(frozen=True)
class TriangleMesh:
    """A decoded triangle mesh in the source file's own coordinate frame.

    Every reader returns this, welded and triangulated, so the CLI and the scene
    embed have exactly one shape to handle.
    """

    #: ``(V, 3)`` float32 vertex positions.
    vertices: NDArray[np.float32]
    #: ``(F, 3)`` uint32 triangle indices into ``vertices``.
    faces: NDArray[np.uint32]
    #: ``(V, 3)`` float32 unit normals, or None.
    normals: Optional[NDArray[np.float32]] = None
    #: ``(V, 3)`` or ``(V, 4)`` uint8 per-vertex colour, or None.
    colors: Optional[NDArray[np.uint8]] = None
    #: One of :data:`MESH_FORMATS`, for provenance.
    source_format: str = ""

    def __post_init__(self) -> None:
        v = self.vertices.shape[0]
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3); got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3); got {self.faces.shape}")
        if self.faces.size and int(self.faces.max()) >= v:
            # Checked here rather than left to `add_mesh`, because at that point the
            # message can no longer name which file produced the bad index.
            raise ValueError(
                f"face index {int(self.faces.max())} is out of range for {v} vertices"
            )
        for name, arr, widths in (
            ("normals", self.normals, (3,)),
            ("colors", self.colors, (3, 4)),
        ):
            if arr is None:
                continue
            if arr.shape[0] != v or arr.shape[1] not in widths:
                raise ValueError(
                    f"{name} must be (V, {'|'.join(map(str, widths))}) with V={v}; "
                    f"got {arr.shape}"
                )

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def detect_mesh_format(path: Union[str, Path]) -> str:
    """Detect which classical mesh dialect ``path`` holds.

    Extension-first, because four distinct extensions map to four distinct formats and
    only ``.ply`` is genuinely ambiguous — it is shared with the Gaussian-splat
    importer. A splat PLY has ``scale_0`` / ``rot_0`` / ``opacity`` on its vertex
    element and no ``face`` element; a mesh PLY has the reverse. Recognising the wrong
    one and saying so beats a parse error thirty lines deeper.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".gltf", ".glb"):
        return "gltf"
    if suffix == ".obj":
        return "obj"
    if suffix == ".stl":
        return "stl"
    if suffix == ".ply":
        with open(path, "rb") as handle:
            head = handle.read(64 * 1024)
        elements, _fmt, _offset = parse_ply_header(head)
        by_name = {e.name: e for e in elements}
        if "face" in by_name:
            return "ply"
        vertex = by_name.get("vertex")
        props = {n for n, _ in (vertex.properties if vertex else [])}
        if {"scale_0", "rot_0", "opacity"} <= props:
            raise ValueError(
                f"{path.name}: this is a Gaussian-splat PLY, not a mesh — it has "
                "scale_0/rot_0/opacity and no 'face' element. Import it with "
                "`luxar gsplat import` instead."
            )
        raise ValueError(
            f"{path.name}: PLY has no 'face' element, so it is a point cloud rather "
            "than a mesh. Load the points with `scene.add_points(...)`."
        )
    raise ValueError(
        f"{path.name}: unrecognized extension {suffix!r} — expected .ply, .obj, .stl, "
        ".gltf or .glb"
    )


_READERS: dict[str, Callable[[Path], dict]] = {
    "ply": read_ply_mesh,
    "obj": read_obj,
    "stl": read_stl,
    "gltf": read_gltf,
}


def _faces_from_rows(rows: list[list[int]]) -> NDArray[np.uint32]:
    """Fan-triangulate PLY's variable-length face rows into ``(F, 3)``."""
    tris: list[tuple[int, int, int]] = []
    for row in rows:
        tris.extend(fan_triangulate(row))
    if not tris:
        return np.zeros((0, 3), dtype=np.uint32)
    return np.asarray(tris, dtype=np.uint32)


def import_mesh(
    path: Union[str, Path],
    *,
    format: str = "auto",
    weld: bool = True,
) -> TriangleMesh:
    """Read a classical mesh file into a :class:`TriangleMesh`.

    Args:
        path: The file. ``.ply`` / ``.obj`` / ``.stl`` / ``.gltf`` / ``.glb``.
        format: Source dialect, or ``"auto"`` to sniff.
        weld: Merge duplicate vertex positions and reindex. On by default because an
            unwelded surface (always, for STL; often, for glTF without indices) has no
            shared vertices, which defeats per-vertex normals, trips the writer's
            authoring lint, and gives picking a different vertex ordinal for the same
            corner depending on which triangle was hit. Pass False to keep the vertex
            list the reader produced. That is not always the file's own list: an OBJ
            that indexes normals independently of positions has no per-vertex normal
            array to begin with, so the reader splits vertices per distinct
            (position, normal) pair and welding is what merges them back.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: For an unknown format, or a file that is not the dialect claimed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    fmt = detect_mesh_format(path) if format == "auto" else format
    if fmt not in _READERS:
        raise ValueError(
            f"Unknown format {fmt!r}; expected 'auto' or one of {MESH_FORMATS}"
        )

    parsed = _READERS[fmt](path)
    vertices = np.ascontiguousarray(parsed["vertices"], dtype=np.float32)
    faces = (
        _faces_from_rows(parsed["face_rows"])
        if "face_rows" in parsed
        else np.ascontiguousarray(parsed["faces"], dtype=np.uint32)
    )
    normals = parsed.get("normals")
    colors = parsed.get("colors")

    if weld:
        extras: dict[str, NDArray | None] = {"normals": normals, "colors": colors}
        vertices, faces, welded = weld_vertices(vertices, faces, extras=extras)
        normals, colors = welded.get("normals"), welded.get("colors")
    faces = drop_degenerate_faces(faces)

    if faces.shape[0] == 0:
        raise ValueError(
            f"{path.name}: no non-degenerate triangles survived import "
            f"({vertices.shape[0]} vertices read)"
        )

    return TriangleMesh(
        vertices=vertices,
        faces=faces,
        normals=None
        if normals is None
        else np.ascontiguousarray(normals, dtype=np.float32),
        colors=None if colors is None else np.ascontiguousarray(colors, dtype=np.uint8),
        source_format=fmt,
    )


__all__ = [
    "MESH_FORMATS",
    "TriangleMesh",
    "detect_mesh_format",
    "import_mesh",
    "is_binary_stl",
]
