"""Split a triangle mesh into spatially disjoint parts, by face.

The mesh counterpart of the flat index-array split that Points / Lines /
GSplats get from :mod:`luxar.core.group.partition`. Those three geometries can
be partitioned by handing each part a slice of the element arrays, because
their elements are *independent*: element ``i`` owns its own row in every
attribute. A triangle does not — it owns three **references** into a shared
vertex table, and two triangles on opposite sides of a BSP cut routinely share
a vertex.

So a mesh part is not a slice. It is a re-indexing:

* Faces are assigned whole to one part (the split is on face CENTROIDS), so no
  triangle is ever geometrically cut and no new geometry is invented.
* Each part then gathers the vertices its own faces reference and renumbers
  those faces to index the gathered table.
* A vertex referenced from both sides of a cut is therefore **duplicated** —
  it appears, byte-identical, in both parts' vertex arrays.

Duplication is what makes the parts independently renderable, which is the
whole point of a partition: each part must stand alone as a drawable leaf. It
costs vertices (bounded by ``3F`` in the pathological case, and in practice a
few percent — only the cut surface duplicates), and it is invisible in the
render because both copies carry identical position AND identical stored
normal, so the shared edge stays seamless. Derivative (flat) shading is
per-fragment and therefore part-agnostic, so it is seamless too.

This module is deliberately geometry-only: it returns index bookkeeping and
never touches vertices, normals, colors or scalars. The caller gathers each
per-vertex attribute through :attr:`MeshPart.vertex_index`, which keeps this
function ignorant of the attribute set (and so immune to a new attribute being
added without updating it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MeshPart:
    """One part of a face-partitioned mesh: renumbered faces + a gather map."""

    #: ``(Fi, 3)`` triangle indices into this part's OWN gathered vertex table
    #: (i.e. into ``vertices[vertex_index]``), not into the original vertices.
    faces: NDArray[np.uint32]

    #: ``(Vi,)`` indices into the ORIGINAL vertex array, ascending. Gather any
    #: per-vertex attribute with ``attr[part.vertex_index]``. Values may repeat
    #: ACROSS parts (a boundary vertex) but never within one.
    vertex_index: NDArray[np.intp]

    #: ``(Fi,)`` indices into the ORIGINAL face array — which input triangles
    #: this part received. Lets a caller carry per-FACE data across the split.
    face_index: NDArray[np.intp]


def face_centroids(
    vertices: NDArray, faces: NDArray[np.uint32]
) -> NDArray[np.floating]:
    """Centroid of each triangle, ``(F, D)``.

    The quantity the BSP splits on. A centroid is used rather than any single
    corner so a part's spatial extent is symmetric about the cut — splitting on
    (say) the first vertex biases every part toward one corner of its own box.
    """
    gathered = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    centroids: NDArray[np.floating] = gathered.mean(axis=1)
    return centroids


def split_mesh_by_faces(
    faces: NDArray, face_parts: Sequence[NDArray[np.intp]]
) -> List[MeshPart]:
    """Re-index ``faces`` into one independently-drawable :class:`MeshPart` each.

    Args:
        faces: ``(F, 3)`` triangle indices into a shared vertex table.
        face_parts: One index array per part, each holding indices into
            ``faces``. Must be a true partition — every face exactly once. This
            is what the ``*_bsp_partition`` splitters return when handed
            :func:`face_centroids`.

    Returns:
        One :class:`MeshPart` per entry of ``face_parts``, in the same order.

    Raises:
        ValueError: If ``faces`` is not ``(F, 3)``, or ``face_parts`` is not a
            partition of ``range(F)``. The partition check is cheap next to the
            gather and catches the one bug that would otherwise be silent —
            dropped or double-counted triangles, which render as holes or as
            invisible double-drawn surfaces rather than as an error.
    """
    faces_arr = np.asarray(faces)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces_arr.shape}")

    n_faces = int(faces_arr.shape[0])
    assigned = np.concatenate(
        [np.asarray(p, dtype=np.intp).reshape(-1) for p in face_parts]
        or [np.empty(0, dtype=np.intp)]
    )
    if assigned.size != n_faces or np.unique(assigned).size != n_faces:
        raise ValueError(
            f"face_parts must be a partition of the {n_faces} faces: got "
            f"{assigned.size} assignments covering {np.unique(assigned).size} "
            "distinct faces. Every face must appear in exactly one part — a "
            "dropped face renders as a hole and a duplicated one as an "
            "invisible double-draw, so neither fails loudly downstream."
        )

    parts: List[MeshPart] = []
    for part in face_parts:
        face_index = np.asarray(part, dtype=np.intp).reshape(-1)
        part_faces = faces_arr[face_index]
        # Flatten BEFORE np.unique: with a 2-D input, numpy >= 2.0 returns
        # `inverse` shaped like the input while numpy < 2.0 returns it flat.
        # A 1-D input behaves identically on both, so the reshape is ours.
        used, inverse = np.unique(part_faces.reshape(-1), return_inverse=True)
        parts.append(
            MeshPart(
                faces=inverse.reshape(-1, 3).astype(np.uint32, copy=False),
                vertex_index=used.astype(np.intp, copy=False),
                face_index=face_index,
            )
        )
    return parts


def duplication_factor(parts: Sequence[MeshPart], n_vertices: int) -> float:
    """Total gathered vertices across ``parts`` divided by the original count.

    ``1.0`` means the cut fell entirely between connected components (no shared
    vertex crossed it); larger means boundary vertices were duplicated. Reported
    by the writer as a diagnostic — it is the honest cost of the partition, and
    a surprising value (say > 1.5) usually means the cap is far too small for
    the mesh's connectivity rather than that anything is wrong.
    """
    if n_vertices <= 0:
        return 1.0
    return sum(int(p.vertex_index.size) for p in parts) / float(n_vertices)
