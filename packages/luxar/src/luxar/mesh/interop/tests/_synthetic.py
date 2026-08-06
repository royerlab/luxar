"""Byte-exact miniature files of every mesh dialect, from one shared ground truth.

Mirrors :mod:`luxar.gsplats.interop.tests._synthetic`: readers are tested for
round-trip parity without shipping binary fixtures, and each writer emits what a real
encoder would (OBJ 1-based, STL an unwelded soup, PLY's faces as a ``property list``),
so the readers are exercised against the formats' actual quirks rather than a
convenient subset.

The ground truth is a **unit tetrahedron**: the smallest closed surface, so every
vertex is shared by three faces and welding is genuinely load-bearing. A cube would
hide index bugs behind axis-aligned symmetry, and a single triangle would hide welding
entirely.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GroundTruth:
    """The mesh every writer below encodes, and every reader must reproduce."""

    vertices: NDArray[np.float32]
    faces: NDArray[np.uint32]
    normals: NDArray[np.float32]
    colors: NDArray[np.uint8]


def make_ground_truth() -> GroundTruth:
    """A unit tetrahedron: 4 vertices, 4 faces, every vertex shared by 3 faces."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    # Per-vertex normals: the normalized position offset from the centroid. Arbitrary
    # but deterministic, non-axis-aligned, and unit-length — so a reader that drops the
    # inverse-transpose or forgets to renormalize is visible.
    offsets = vertices - vertices.mean(axis=0)
    normals = (offsets / np.linalg.norm(offsets, axis=1, keepdims=True)).astype(
        np.float32
    )
    colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8
    )
    return GroundTruth(vertices=vertices, faces=faces, normals=normals, colors=colors)


def write_ply_binary(path: Path, gt: GroundTruth, *, big_endian: bool = False) -> None:
    """Binary PLY with normals, uchar colours, and faces as a ``property list``."""
    order = "binary_big_endian" if big_endian else "binary_little_endian"
    prefix = ">" if big_endian else "<"
    header = (
        "ply\n"
        f"format {order} 1.0\n"
        f"element vertex {len(gt.vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(gt.faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    vdt = np.dtype(
        [(n, f"{prefix}f4") for n in ("x", "y", "z", "nx", "ny", "nz")]
        + [(n, "u1") for n in ("red", "green", "blue")]
    )
    rows = np.zeros(len(gt.vertices), dtype=vdt)
    for i, n in enumerate(("x", "y", "z")):
        rows[n] = gt.vertices[:, i]
    for i, n in enumerate(("nx", "ny", "nz")):
        rows[n] = gt.normals[:, i]
    for i, n in enumerate(("red", "green", "blue")):
        rows[n] = gt.colors[:, i]

    body = bytearray(rows.tobytes())
    for tri in gt.faces:
        body += struct.pack("B", 3)
        body += np.asarray(tri, dtype=f"{prefix}i4").tobytes()
    path.write_bytes(header + bytes(body))


def write_ply_ascii(path: Path, gt: GroundTruth) -> None:
    """ASCII PLY — the variant the gsplat PLY parser refuses outright."""
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(gt.vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(gt.faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v, c in zip(gt.vertices, gt.colors):
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}")
    for tri in gt.faces:
        lines.append(f"3 {tri[0]} {tri[1]} {tri[2]}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_ply_quads(path: Path, gt: GroundTruth) -> None:
    """A PLY whose faces are QUADS, to exercise fan triangulation.

    One unit square as two-triangles-worth of one quad row.
    """
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        "element face 1",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v in verts:
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    lines.append("4 0 1 2 3")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_obj(path: Path, gt: GroundTruth) -> None:
    """OBJ with 1-BASED indices and `v//vn` face triples."""
    lines = ["# synthetic tetrahedron"]
    for v in gt.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for n in gt.normals:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for tri in gt.faces:
        a, b, c = (int(i) + 1 for i in tri)
        lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_obj_negative_indices(path: Path, gt: GroundTruth) -> None:
    """OBJ using NEGATIVE (end-relative) indices — legal, and easy to read wrong."""
    lines = []
    for v in gt.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    n = len(gt.vertices)
    for tri in gt.faces:
        # -n maps to index 0 when n vertices have been declared.
        lines.append(" ".join(["f"] + [str(int(i) - n) for i in tri]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_obj_quad(path: Path) -> None:
    """OBJ with a QUAD face, to exercise fan triangulation."""
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", encoding="utf-8")


def write_stl_binary(path: Path, gt: GroundTruth) -> None:
    """Binary STL — a triangle SOUP, and a header that begins with the word 'solid'.

    The header is the trap: 'solid' at byte 0 is what makes magic-word sniffing
    misclassify real binary STLs as ASCII.
    """
    header = b"solid exported by a tool that writes 'solid' into a BINARY header".ljust(
        80, b"\0"
    )
    body = bytearray(header)
    body += struct.pack("<I", len(gt.faces))
    for tri in gt.faces:
        corners = gt.vertices[np.asarray(tri)]
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm > 0 else normal
        body += np.asarray(normal, dtype="<f4").tobytes()
        body += np.asarray(corners, dtype="<f4").tobytes()
        body += struct.pack("<H", 0)
    path.write_bytes(bytes(body))


def write_stl_ascii(path: Path, gt: GroundTruth) -> None:
    """ASCII STL."""
    out = ["solid tetra"]
    for tri in gt.faces:
        corners = gt.vertices[np.asarray(tri)]
        out.append("  facet normal 0 0 0")
        out.append("    outer loop")
        for c in corners:
            out.append(f"      vertex {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
        out.append("    endloop")
        out.append("  endfacet")
    out.append("endsolid tetra")
    path.write_text("\n".join(out) + "\n", encoding="ascii")


def _accessor_blob(gt: GroundTruth) -> tuple[bytes, list[dict], list[dict]]:
    """Pack positions, normals and indices into one buffer with three views."""
    pos = gt.vertices.astype("<f4").tobytes()
    nrm = gt.normals.astype("<f4").tobytes()
    idx = gt.faces.reshape(-1).astype("<u4").tobytes()
    blob = pos + nrm + idx
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
        {"buffer": 0, "byteOffset": len(pos), "byteLength": len(nrm)},
        {"buffer": 0, "byteOffset": len(pos) + len(nrm), "byteLength": len(idx)},
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": len(gt.vertices),
            "type": "VEC3",
        },
        {
            "bufferView": 1,
            "componentType": 5126,
            "count": len(gt.vertices),
            "type": "VEC3",
        },
        {
            "bufferView": 2,
            "componentType": 5125,
            "count": gt.faces.size,
            "type": "SCALAR",
        },
    ]
    return blob, views, accessors


def write_glb(
    path: Path, gt: GroundTruth, *, translation: list[float] | None = None
) -> None:
    """A GLB with one mesh under one node, optionally translated.

    ``translation`` exercises the node-transform composition: skip it in the reader and
    the imported mesh sits at the origin instead of where the file put it.
    """
    blob, views, accessors = _accessor_blob(gt)
    node: dict = {"mesh": 0}
    if translation is not None:
        node["translation"] = translation
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [node],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "mode": 4,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_chunk = json.dumps(doc).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = blob + b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    path.write_bytes(out)


def write_glb_interleaved(path: Path, gt: GroundTruth) -> None:
    """A GLB whose POSITION and NORMAL share one bufferView via ``byteStride``.

    Common in real exports, and the case a reader that ignores ``byteStride`` decodes
    as garbage rather than failing.
    """
    stride = 24  # 3 floats position + 3 floats normal
    packed = np.empty((len(gt.vertices), 6), dtype="<f4")
    packed[:, :3] = gt.vertices
    packed[:, 3:] = gt.normals
    blob = packed.tobytes()
    idx = gt.faces.reshape(-1).astype("<u4").tobytes()
    full = blob + idx
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "mode": 4,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(full)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(blob),
                "byteStride": stride,
            },
            {"buffer": 0, "byteOffset": len(blob), "byteLength": len(idx)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": len(gt.vertices),
                "type": "VEC3",
            },
            {
                "bufferView": 0,
                "byteOffset": 12,
                "componentType": 5126,
                "count": len(gt.vertices),
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": gt.faces.size,
                "type": "SCALAR",
            },
        ],
    }
    json_chunk = json.dumps(doc).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = full + b"\0" * (-len(full) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    path.write_bytes(out)


def write_gltf_draco(path: Path) -> None:
    """A .gltf declaring Draco as REQUIRED, to pin the named refusal."""
    doc = {
        "asset": {"version": "2.0"},
        "extensionsRequired": ["KHR_draco_mesh_compression"],
        "extensionsUsed": ["KHR_draco_mesh_compression"],
        "scene": 0,
        "scenes": [{"nodes": []}],
        "meshes": [],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def write_gsplat_ply(path: Path) -> None:
    """An INRIA Gaussian-splat PLY — the cross-importer case the sniffer must name."""
    props = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + np.zeros((2, len(props)), dtype="<f4").tobytes())


#: Dialect name → writer, mirroring `_READERS` on the production side.
WRITERS = {
    "ply": write_ply_binary,
    "obj": write_obj,
    "stl": write_stl_binary,
    "gltf": lambda p, gt: write_glb(p, gt),
}

#: Dialect name → the extension its writer emits.
SUFFIXES = {"ply": ".ply", "obj": ".obj", "stl": ".stl", "gltf": ".glb"}


def _glb_with_nodes(
    path: Path, gt: GroundTruth, nodes: list[dict], roots: list[int]
) -> None:
    """A GLB carrying one mesh and a caller-supplied node graph.

    Factored out so the malformed-graph fixtures below differ only in the graph, which
    is the variable under test.
    """
    blob, views, accessors = _accessor_blob(gt)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": roots}],
        "nodes": nodes,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "mode": 4,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_chunk = json.dumps(doc).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = blob + b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    path.write_bytes(out)


def write_glb_cyclic(path: Path, gt: GroundTruth) -> None:
    """A GLB whose node graph contains a cycle (0 -> 1 -> 0).

    Malformed — glTF nodes form a forest — and the case that recursed to
    ``RecursionError`` before the reader tracked ancestors.
    """
    _glb_with_nodes(path, gt, [{"children": [1], "mesh": 0}, {"children": [0]}], [0])


def write_glb_shared_child(path: Path, gt: GroundTruth) -> None:
    """Two parents pointing at one mesh-bearing leaf — a DAG, not a cycle."""
    _glb_with_nodes(
        path, gt, [{"children": [2]}, {"children": [2]}, {"mesh": 0}], [0, 1]
    )


def write_glb_bad_node_index(path: Path, gt: GroundTruth) -> None:
    """A child edge pointing past the end of the node array."""
    _glb_with_nodes(path, gt, [{"children": [99], "mesh": 0}], [0])


def write_glb_interleaved_at_buffer_end(path: Path, gt: GroundTruth) -> None:
    """Interleaved POSITION/NORMAL placed LAST in the buffer, with no trailing padding.

    The layout that exposes an accessor span computed as ``count * stride`` instead of
    ``(count - 1) * stride + element``: the final element occupies only its own width,
    so the padding a naive span assumes simply is not there. ``write_glb_interleaved``
    hides the defect because index data follows the interleaved view in the same buffer
    and absorbs the over-read.
    """
    stride = 24
    packed = np.empty((len(gt.vertices), 6), dtype="<f4")
    packed[:, :3] = gt.vertices
    packed[:, 3:] = gt.normals
    # Drop the trailing 12 bytes the last element does not need: the buffer ends exactly
    # at the end of the final NORMAL.
    inter = packed.tobytes()[: (len(gt.vertices) - 1) * stride + 24]
    idx = gt.faces.reshape(-1).astype("<u4").tobytes()
    full = idx + inter  # indices FIRST, interleaved view at the very end
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 1, "NORMAL": 2},
                        "indices": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(full)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(idx)},
            {
                "buffer": 0,
                "byteOffset": len(idx),
                "byteLength": len(inter),
                "byteStride": stride,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5125,
                "count": gt.faces.size,
                "type": "SCALAR",
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": len(gt.vertices),
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "byteOffset": 12,
                "componentType": 5126,
                "count": len(gt.vertices),
                "type": "VEC3",
            },
        ],
    }
    json_chunk = json.dumps(doc).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = full + b"\0" * (-len(full) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    path.write_bytes(out)


def write_glb_mirrored(path: Path, gt: GroundTruth) -> None:
    """A GLB whose single node applies a reflecting scale of ``[-1, 1, 1]``."""
    _glb_with_nodes(path, gt, [{"mesh": 0, "scale": [-1.0, 1.0, 1.0]}], [0])


#: A MID-RANGE palette for the OBJ colour-convention fixtures.
#:
#: Deliberately not :attr:`GroundTruth.colors`, whose channels are all 0 or 255. Under
#: the bug these fixtures exist to catch — scaling a 0..255 file by 255 and clipping —
#: 255 maps back to 255 and 0 to 0, so an all-extremes palette produces the CORRECT
#: answer through the wrong code path and the test cannot fail. Every channel here is
#: strictly between 1 and 254, so 255x-and-clip lands on 255 and the difference shows.
OBJ_MID_COLORS = np.array(
    [[128, 64, 32], [10, 200, 90], [77, 77, 77], [3, 250, 128]], dtype=np.uint8
)


def write_obj_colors_0_255(path: Path, gt: GroundTruth) -> None:
    """OBJ vertex colours in the 0..255 convention (scanners), not 0..1 (MeshLab)."""
    lines = []
    for v, c in zip(gt.vertices, OBJ_MID_COLORS):
        lines.append(
            f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}"
        )
    for tri in gt.faces:
        lines.append(" ".join(["f"] + [str(int(i) + 1) for i in tri]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_obj_colors_0_1(path: Path, gt: GroundTruth) -> None:
    """The same colours in the 0..1 convention."""
    lines = []
    for v, c in zip(gt.vertices, OBJ_MID_COLORS):
        r, g, b = (float(x) / 255.0 for x in c)
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.6f} {g:.6f} {b:.6f}")
    for tri in gt.faces:
        lines.append(" ".join(["f"] + [str(int(i) + 1) for i in tri]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
