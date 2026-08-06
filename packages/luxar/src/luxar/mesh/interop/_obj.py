"""Wavefront OBJ reader.

Text format, so no byte-order or binary layout to get wrong — but three traps that a
naive reader silently gets wrong instead of failing:

* **Indices are 1-BASED**, and may be NEGATIVE (relative to the end of the list so
  far). Reading them as 0-based shifts the whole surface by one vertex.
* **Faces may be polygons**, not just triangles. Quads are the common case from any
  modelling package; taking only the first three indices drops half of every quad.
* **Normals are indexed per CORNER, independently of the positions.** Every mainstream
  exporter deduplicates the ``vn`` pool, so it is neither the same length as the ``v``
  pool nor parallel to it. Assuming parallel indexing (or giving up and dropping the
  pool) throws away the normals of essentially every smooth-shaded export.

Materials (``usemtl`` / ``mtllib``) are ignored: they name a separate ``.mtl`` file
whose model is per-FACE, while ``add_mesh`` colours per VERTEX. Translating one to the
other means splitting vertices at material boundaries, which is a different feature.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._weld import fan_triangulate

#: Corner with no `vn` binding. A resolved normal reference is always >= 0 (see
#: `_resolve`), so a negative value is unambiguous and keeps the ref arrays plain ints.
_UNBOUND = -1


def _resolve(index: int, count: int, kind: str, name: str) -> int:
    """OBJ index → 0-based. Positive is 1-based; negative counts back from the end.

    Range-checked, because an out-of-range reference would otherwise surface as an
    ``IndexError`` on the normal pool or as an ``OverflowError`` casting a negative
    index to ``uint32`` — neither of which is a ``ValueError``, so a malformed file
    escapes the CLI's error funnel as a raw traceback.
    """
    resolved = index - 1 if index > 0 else count + index
    if not 0 <= resolved < count:
        raise ValueError(
            f"{name}: a face references {kind} index {index}, but only {count} '{kind}' "
            f"lines precede it — the file is malformed; re-export it."
        )
    return resolved


def _bind_normals(
    n_positions: int,
    faces: list[tuple[int, int, int]],
    face_normal_refs: list[tuple[int, int, int]],
    normals_raw: list[tuple[float, float, float]],
) -> tuple[list[int], list[tuple[int, int, int]], NDArray[np.float32] | None]:
    """Turn OBJ's per-CORNER normal indexing into a per-VERTEX normal array.

    Returns ``(vertex_sources, faces, normals)``: which position row each output vertex
    draws from, the faces reindexed onto those vertices, and the per-vertex normals.
    """

    def corners() -> Iterator[tuple[int, int]]:
        """Every (position_ref, normal_ref) corner, lazily — this is 3F entries."""
        for tri, refs in zip(faces, face_normal_refs):
            yield from zip(tri, refs)

    identity = list(range(n_positions))

    # The pool must actually be REFERENCED, by EVERY corner. A `vn` block no face points
    # at (every corner written as plain `f 1 2 3`) says nothing about which vertex each
    # normal belongs to — matching counts is a coincidence, and honouring it invents
    # per-vertex normals the file never asked for, shading the surface by data the
    # exporter left unbound. A partial binding is the same problem for the unbound
    # corners, so it is dropped whole rather than filled in.
    if not normals_raw or not faces or any(ref == _UNBOUND for _, ref in corners()):
        return identity, faces, None

    if len(normals_raw) == n_positions and all(
        ref == corner for corner, ref in corners()
    ):
        # `f v//v` with one `vn` per `v`: the pool already IS a per-vertex array.
        return identity, faces, np.asarray(normals_raw, dtype=np.float32)

    # Independent normal indexing — what Blender and most exporters write, with one
    # deduplicated `vn` per DISTINCT normal, so the pool is neither the same length as
    # the positions nor parallel to them. A normal belongs to a corner rather than to a
    # vertex, so the faithful per-vertex form is one vertex per distinct
    # (position, normal) pair. Dropping the normals instead would put every smooth-shaded
    # export on the derivative flat-normal path, which is the common case, not the
    # exotic one. The duplication is undone downstream: `weld_vertices` keys on position
    # AND normal, so corners that agree merge back into one shared vertex and only a
    # genuine crease stays split.
    pairs: dict[tuple[int, int], int] = {}
    sources: list[int] = []
    normal_rows: list[int] = []
    remapped: list[tuple[int, int, int]] = []
    for tri, refs in zip(faces, face_normal_refs):
        slots: list[int] = []
        for corner, ref in zip(tri, refs):
            slot = pairs.get((corner, ref))
            if slot is None:
                slot = len(sources)
                pairs[(corner, ref)] = slot
                sources.append(corner)
                normal_rows.append(ref)
            slots.append(slot)
        remapped.append((slots[0], slots[1], slots[2]))

    pool = np.asarray(normals_raw, dtype=np.float32)
    return sources, remapped, pool[np.asarray(normal_rows, dtype=np.intp)]


def read_obj(path: Path) -> dict[str, object]:
    """Decode an OBJ into raw component arrays.

    Only ``v`` (positions), ``vn`` (normals) and ``f`` (faces) are consumed. Texture
    coordinates (``vt``) are parsed off the face triples and discarded — Luxar meshes
    have no UV channel.
    """
    positions: list[tuple[float, ...]] = []
    normals_raw: list[tuple[float, float, float]] = []
    # (vertex_index, normal_index_or_UNBOUND) per corner, still in file order.
    faces: list[tuple[int, int, int]] = []
    face_normal_refs: list[tuple[int, int, int]] = []
    vertex_colors: list[tuple[float, float, float] | None] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]

            if tag == "v":
                vals = [float(v) for v in parts[1:]]
                positions.append(tuple(vals[:3]))
                # The unofficial-but-widespread `v x y z r g b` extension, emitted by
                # MeshLab and several scanners. Six floats means per-vertex colour.
                vertex_colors.append(
                    (vals[3], vals[4], vals[5]) if len(vals) >= 6 else None
                )
            elif tag == "vn":
                vals = [float(v) for v in parts[1:4]]
                normals_raw.append((vals[0], vals[1], vals[2]))
            elif tag == "f":
                corners: list[int] = []
                normal_refs: list[int] = []
                for token in parts[1:]:
                    # v, v/vt, v//vn, or v/vt/vn
                    fields = token.split("/")
                    corners.append(
                        _resolve(int(fields[0]), len(positions), "v", path.name)
                    )
                    if len(fields) == 3 and fields[2]:
                        normal_refs.append(
                            _resolve(int(fields[2]), len(normals_raw), "vn", path.name)
                        )
                    else:
                        normal_refs.append(_UNBOUND)
                # Fan over CORNER POSITIONS, then map each to its vertex and normal.
                # Fanning over the vertex indices themselves and looking them back up
                # would attribute the wrong normal reference to a polygon that repeats
                # a vertex, since the lookup finds the first occurrence.
                for tri in fan_triangulate(list(range(len(corners)))):
                    faces.append((corners[tri[0]], corners[tri[1]], corners[tri[2]]))
                    face_normal_refs.append(
                        (
                            normal_refs[tri[0]],
                            normal_refs[tri[1]],
                            normal_refs[tri[2]],
                        )
                    )

    if not positions:
        raise ValueError(f"{path.name}: OBJ has no 'v' vertex lines")
    if not faces:
        raise ValueError(
            f"{path.name}: OBJ has no 'f' face lines — this is a point cloud, not a "
            "mesh. Load it with `scene.add_points(...)` instead."
        )

    sources, faces, normals = _bind_normals(
        len(positions), faces, face_normal_refs, normals_raw
    )
    vertices = np.asarray(positions, dtype=np.float32)[np.asarray(sources, np.intp)]
    face_arr = np.asarray(faces, dtype=np.uint32)

    # Colour is per POSITION, so it follows the same gather as the positions when a
    # corner split duplicated them.
    per_vertex = [vertex_colors[i] for i in sources]
    colors = None
    present = [c for c in per_vertex if c is not None]
    if present:
        # The `v x y z r g b` extension is unofficial and exporters disagree about the
        # range: MeshLab writes 0..1, several scanners write 0..255. Distinguish by the
        # observed peak, exactly as the PLY reader does for `red/green/blue` — assuming
        # 0..1 and scaling unconditionally would clip every nonzero channel of a 0..255
        # file to 255, turning a coloured mesh into a white one. Measured over the rows
        # that HAVE colour, so the fill value below cannot skew the verdict.
        unit_range = float(np.max(np.asarray(present, dtype=np.float32))) <= 1.0
        # A `v` line may omit the colour while others carry it. Fill those with white in
        # WHICHEVER convention the file uses — a fixed 1.0 is white only in the 0..1
        # convention; in a 0..255 file it survives unscaled as RGB(1, 1, 1), i.e. black.
        white = 1.0 if unit_range else 255.0
        raw = np.asarray(
            [c if c is not None else (white, white, white) for c in per_vertex],
            dtype=np.float32,
        )
        scaled = raw * 255.0 if unit_range else raw
        colors = np.clip(scaled, 0, 255).astype(np.uint8)

    return {
        "vertices": vertices,
        "faces": face_arr,
        "normals": normals,
        "colors": colors,
    }
