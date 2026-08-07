"""Topology normalization shared by the mesh readers.

Classical mesh formats disagree about whether vertices are shared. PLY and glTF are
indexed, OBJ is indexed but 1-based and may use polygons, and STL is a pure triangle
soup with no index at all. :func:`weld_vertices` and :func:`fan_triangulate` bring all
four to the one representation ``add_mesh`` wants: a shared vertex array plus a
``(F, 3)`` index array.

Welding matters beyond tidiness. An unwelded surface has no shared vertices, so
per-vertex normals cannot be averaged across faces, the writer's authoring lint flags
it, and picking — which returns a VERTEX ordinal — reports a different id for the same
corner depending on which triangle was hit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fan_triangulate(polygon: list[int]) -> list[tuple[int, int, int]]:
    """Split a convex polygon's vertex loop into a triangle fan.

    ``[a, b, c, d]`` becomes ``[(a, b, c), (a, c, d)]``. Correct for convex polygons,
    which is what OBJ quads from any modelling package are in practice; a concave
    polygon fans into triangles that overlap its exterior. Detecting concavity needs a
    plane fit and a winding test per face, and the failure is visible rather than
    silent, so this stays the cheap version.
    """
    if len(polygon) < 3:
        return []
    return [
        (polygon[0], polygon[i], polygon[i + 1]) for i in range(1, len(polygon) - 1)
    ]


def weld_vertices(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    *,
    extras: dict[str, NDArray | None] | None = None,
    decimals: int = 6,
) -> tuple[NDArray[np.float32], NDArray[np.uint32], dict[str, NDArray | None]]:
    """Merge duplicate vertices and reindex the faces onto the survivors.

    Two vertices merge only when their position **and every supplied per-vertex
    attribute** agree. Keying on position alone would be wrong for the indexed formats:
    PLY, OBJ and glTF all express a HARD EDGE as coincident positions carrying
    different normals (a cube's corner appears three times, once per face). Merging
    those and keeping one side's normal turns every crease into arbitrarily-shaded
    nonsense — the corruption is worst on exactly the CAD and modelling output people
    would import first.

    Including the attributes is what keeps STL working too, and is why the key is not
    instead narrowed to soup-only inputs: ``_stl`` deliberately returns
    ``normals=None`` (STL's normal is per-FACET, and no per-vertex choice is
    non-arbitrary), so a soup's key degenerates to position and the soup welds as
    before. The rule needs no per-format special case.

    Args:
        vertices: ``(V, D)`` positions.
        faces: ``(F, 3)`` indices into ``vertices``.
        extras: Per-vertex arrays (normals, colors) to carry through. They participate
            in the merge key, so each welded group is attribute-IDENTICAL and taking
            the first row is exact rather than a choice. Nothing is averaged: averaging
            normals across a hard edge would round it off, and a reader cannot know
            which edges were meant to be hard.
        decimals: Rounding applied before comparison. Coordinates that differ below
            this survive as one vertex. 6 is ~1 nm at metre scale and ~1e-4 of a unit
            cube, comfortably below any format's own precision. Applied to the
            attribute columns as well, so a normal that differs only in float noise
            does not split a vertex that should weld.

    Returns:
        ``(welded_vertices, remapped_faces, welded_extras)``.
    """
    if vertices.shape[0] == 0:
        return vertices, faces, dict(extras or {})

    # Round for the comparison only — the SURVIVING rows keep full precision. Welding
    # on rounded values and then storing them would quantize every coordinate in the
    # file to `decimals`, which for a millimetre-scale scan is a visible loss.
    key_columns = [np.round(vertices.astype(np.float64), decimals)]
    # Sorted by name so the key layout does not depend on dict insertion order — the
    # grouping is identical either way, but a stable layout keeps `first_index`
    # (and therefore the output vertex order) reproducible across callers.
    for _, arr in sorted((extras or {}).items()):
        if arr is None:
            continue
        column = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        if column.shape[0] != vertices.shape[0]:
            raise ValueError(
                f"per-vertex attribute has {column.shape[0]} rows but there are "
                f"{vertices.shape[0]} vertices"
            )
        key_columns.append(np.round(column, decimals))

    keyed = key_columns[0] if len(key_columns) == 1 else np.hstack(key_columns)
    _, first_index, inverse = np.unique(
        keyed, axis=0, return_index=True, return_inverse=True
    )

    # `np.unique` returns groups in sorted-key order; `first_index` maps each group to
    # the row it first appeared at. Take the survivors in that same order so `inverse`
    # indexes them directly.
    welded = vertices[first_index]
    remapped = inverse[faces.reshape(-1)].reshape(faces.shape).astype(np.uint32)

    welded_extras: dict[str, NDArray | None] = {}
    for key, arr in (extras or {}).items():
        welded_extras[key] = arr[first_index] if arr is not None else None

    return welded, remapped, welded_extras


def drop_degenerate_faces(faces: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Remove triangles with a repeated vertex index.

    Welding creates these: a sliver triangle whose two corners round to the same
    position collapses to a line. They contribute nothing to the render (zero area)
    but they do inflate the face count and, in a `flat` shading pass, produce a
    degenerate cross product whose normalization is undefined.
    """
    if faces.shape[0] == 0:
        return faces
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    keep = (a != b) & (b != c) & (a != c)
    kept: NDArray[np.uint32] = faces[keep]
    return kept
