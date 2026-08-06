"""Wavefront OBJ reader.

Text format, so no byte-order or binary layout to get wrong — but two traps that a
naive reader silently gets wrong instead of failing:

* **Indices are 1-BASED**, and may be NEGATIVE (relative to the end of the list so
  far). Reading them as 0-based shifts the whole surface by one vertex.
* **Faces may be polygons**, not just triangles. Quads are the common case from any
  modelling package; taking only the first three indices drops half of every quad.

Materials (``usemtl`` / ``mtllib``) are ignored: they name a separate ``.mtl`` file
whose model is per-FACE, while ``add_mesh`` colours per VERTEX. Translating one to the
other means splitting vertices at material boundaries, which is a different feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._weld import fan_triangulate


def _resolve(index: int, count: int) -> int:
    """OBJ index → 0-based. Positive is 1-based; negative counts back from the end."""
    return index - 1 if index > 0 else count + index


def read_obj(path: Path) -> dict[str, object]:
    """Decode an OBJ into raw component arrays.

    Only ``v`` (positions), ``vn`` (normals) and ``f`` (faces) are consumed. Texture
    coordinates (``vt``) are parsed off the face triples and discarded — Luxar meshes
    have no UV channel.
    """
    positions: list[tuple[float, ...]] = []
    normals_raw: list[tuple[float, float, float]] = []
    # (vertex_index, normal_index_or_None) per corner, still in file order.
    faces: list[tuple[int, int, int]] = []
    face_normal_refs: list[tuple[int | None, int | None, int | None]] = []
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
                normal_refs: list[int | None] = []
                for token in parts[1:]:
                    # v, v/vt, v//vn, or v/vt/vn
                    fields = token.split("/")
                    corners.append(_resolve(int(fields[0]), len(positions)))
                    if len(fields) == 3 and fields[2]:
                        normal_refs.append(_resolve(int(fields[2]), len(normals_raw)))
                    else:
                        normal_refs.append(None)
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

    vertices = np.asarray(positions, dtype=np.float32)
    face_arr = np.asarray(faces, dtype=np.uint32)

    # Per-vertex normals only when the file's normal indexing is parallel to its
    # position indexing (`f v//v` with one vn per v). Anything else — a shared normal
    # pool indexed independently — cannot become a per-vertex array without splitting
    # vertices, so it is dropped and the shader derives flat normals instead.
    normals = None
    if normals_raw and len(normals_raw) == len(positions):
        refs = [
            (corner, ref)
            for tri, tri_refs in zip(faces, face_normal_refs)
            for corner, ref in zip(tri, tri_refs)
        ]
        # The pool must actually be REFERENCED, and referenced index-parallel to the
        # positions. A `vn` block no face points at (every corner written as plain
        # `f 1 2 3`) says nothing about which vertex each normal belongs to — matching
        # counts is a coincidence, and honouring it invents per-vertex normals the file
        # never asked for, shading the surface by data the exporter left unbound.
        referenced = any(ref is not None for _, ref in refs)
        parallel = all(ref is None or ref == corner for corner, ref in refs)
        if referenced and parallel:
            normals = np.asarray(normals_raw, dtype=np.float32)

    colors = None
    if any(c is not None for c in vertex_colors):
        filled = [c if c is not None else (1.0, 1.0, 1.0) for c in vertex_colors]
        raw = np.asarray(filled, dtype=np.float32)
        # The `v x y z r g b` extension is unofficial and exporters disagree about the
        # range: MeshLab writes 0..1, several scanners write 0..255. Distinguish by the
        # observed peak, exactly as the PLY reader does for `red/green/blue` — assuming
        # 0..1 and scaling unconditionally would clip every nonzero channel of a 0..255
        # file to 255, turning a coloured mesh into a white one.
        peak = float(np.max(raw)) if raw.size else 0.0
        scaled = raw * 255.0 if peak <= 1.0 else raw
        colors = np.clip(scaled, 0, 255).astype(np.uint8)

    return {
        "vertices": vertices,
        "faces": face_arr,
        "normals": normals,
        "colors": colors,
    }
