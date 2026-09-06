"""Analytic mesh primitives for demos, tests and marker geometry.

One shape today, the icosphere, because it is the one every mesh-touching corner
of the repo kept re-deriving privately (the viewer fixtures, the core tests, the
protein-stories marker shells): a welded, closed, smoothly varying surface whose
normals have an analytic answer. Anything that wants to *show* a material rather
than *reconstruct* a dataset wants exactly that.

Nothing here touches zarr or the scene graph — plain NumPy in, plain NumPy out —
which is why it lives beside :mod:`luxar.mesh.split` rather than in
:mod:`luxar.core`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

__all__ = ["icosphere"]


def icosphere(
    subdivisions: int = 3, radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A welded, closed icosphere with outward unit normals.

    Starts from the regular icosahedron and splits every face into four
    ``subdivisions`` times, sharing each new edge midpoint between the two faces
    that own it. That sharing is the whole point of building it this way rather
    than emitting a triangle soup:

    * **Welded** — adjacent faces reference the same vertex rows, so the surface
      is one connected mesh (smooth shading interpolates across edges, and a
      per-vertex pick lands on a genuine many-to-one target).
    * **Closed** — no boundary, so back faces exist to be culled or exposed.
    * **Analytic normals** — after projection onto the sphere the unit position
      *is* the outward normal, which makes the shading of the result something
      one can reason about rather than merely record.

    Faces are wound counter-clockwise seen from outside, so they are front-facing
    under ``normal_dims=[0, 1, 2]`` with the returned normals.

    Args:
        subdivisions: Refinement steps. ``0`` is the 12-vertex icosahedron; each
            step quadruples the face count (``20 * 4**subdivisions`` faces,
            ``10 * 4**subdivisions + 2`` vertices). ``3`` gives 642 vertices /
            1280 faces, smooth enough to read as a sphere at marker size.
        radius: Sphere radius in the caller's units.

    Returns:
        ``(vertices, faces, normals)`` — ``(V, 3) float32``, ``(F, 3) uint32``,
        ``(V, 3) float32`` unit normals — in the shapes and dtypes ``add_mesh``
        takes directly.

    Raises:
        ValueError: If ``subdivisions`` is negative or ``radius`` is not
            strictly positive and finite.
    """
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be >= 0, got {subdivisions}")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be finite and > 0, got {radius}")

    t = (1.0 + 5.0**0.5) / 2.0
    verts: List[List[float]] = [
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    faces: List[List[int]] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    def midpoint(i: int, j: int, cache: Dict[Tuple[int, int], int]) -> int:
        """Index of the midpoint of edge ``(i, j)``, created once per edge."""
        key = (min(i, j), max(i, j))
        index = cache.get(key)
        if index is None:
            vi, vj = verts[i], verts[j]
            verts.append(
                [(vi[0] + vj[0]) / 2, (vi[1] + vj[1]) / 2, (vi[2] + vj[2]) / 2]
            )
            index = len(verts) - 1
            cache[key] = index
        return index

    for _ in range(subdivisions):
        cache: Dict[Tuple[int, int], int] = {}
        out: List[List[int]] = []

        for a, b, c in faces:
            ab = midpoint(a, b, cache)
            bc = midpoint(b, c, cache)
            ca = midpoint(c, a, cache)
            out += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = out

    positions = np.asarray(verts, dtype=np.float64)
    normals = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    return (
        (normals * radius).astype(np.float32),
        np.asarray(faces, dtype=np.uint32),
        normals.astype(np.float32),
    )
