"""glTF 2.0 / GLB reader — hand-written, no dependency.

The house precedent is unambiguous: all five Gaussian-splat dialects decode on NumPy +
stdlib, and ``read_spz`` hand-decodes a gzip container with a bit-packed smallest-three
quaternion — harder than a glTF accessor. Taking ``pygltflib`` would save the JSON→
object mapping and none of the accessor decode, ``byteStride`` handling or node-transform
composition; taking ``trimesh`` would put the WHOLE importer behind an optional extra,
so ``luxar mesh import`` would fail on a bare install while ``luxar gsplat import``
works. That asymmetry is the trade the gsplat importer already declined.

Three things here are correctness, not polish:

* **Interleaved accessors.** ``bufferView.byteStride`` is common in real files
  (POSITION and NORMAL packed in one view). Ignoring it decodes garbage rather than
  failing.
* **Node transforms.** Meshes hang off ``nodes`` with a ``matrix`` or TRS, rooted in a
  scene. Skip the graph and a multi-part model imports with every part stacked at the
  origin — a plausible-looking, entirely wrong result.
* **Compressed geometry.** Draco and meshopt are not hand-rollable at reasonable cost,
  so they are refused BY NAME with the fix, mirroring ``read_spz``'s SPZ-v4 refusal.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from numpy.typing import NDArray

_GLB_MAGIC = 0x46546C67  # 'glTF'
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

#: glTF componentType → numpy dtype (all little-endian per spec).
_COMPONENT_TYPES: dict[int, str] = {
    5120: "<i1",
    5121: "<u1",
    5122: "<i2",
    5123: "<u2",
    5125: "<u4",
    5126: "<f4",
}
#: glTF accessor type → component count.
_TYPE_COUNTS: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}
#: Extensions whose geometry we cannot decode. Refused by name.
_COMPRESSION_EXTENSIONS = {
    "KHR_draco_mesh_compression": "Draco",
    "EXT_meshopt_compression": "meshopt",
}


def _parse_glb(raw: bytes) -> tuple[dict, bytes | None]:
    """Split a GLB container into its JSON chunk and its optional BIN chunk."""
    if len(raw) < 12:
        raise ValueError("GLB is shorter than its 12-byte header")
    magic, version, _total = struct.unpack_from("<III", raw, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("not a GLB (bad magic)")
    if version != 2:
        raise ValueError(f"GLB version {version} is not supported (only glTF 2.0)")

    doc: dict | None = None
    binary: bytes | None = None
    offset = 12
    while offset + 8 <= len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        payload = raw[offset : offset + length]
        if kind == _CHUNK_JSON:
            doc = json.loads(payload.decode("utf-8"))
        elif kind == _CHUNK_BIN:
            binary = payload
        # Unknown chunk types are skipped, as the spec requires.
        offset += length + (-length % 4)  # chunks are 4-byte aligned
    if doc is None:
        raise ValueError("GLB has no JSON chunk")
    return doc, binary


def _load_buffers(doc: dict, base_dir: Path, glb_binary: bytes | None) -> list[bytes]:
    """Resolve every declared buffer to bytes: the GLB chunk, a data URI, or a file."""
    buffers: list[bytes] = []
    for i, buf in enumerate(doc.get("buffers", [])):
        uri = buf.get("uri")
        if uri is None:
            if glb_binary is None:
                raise ValueError(f"buffer {i} has no uri and there is no GLB BIN chunk")
            buffers.append(glb_binary)
        elif uri.startswith("data:"):
            _, _, payload = uri.partition(",")
            buffers.append(base64.b64decode(payload))
        else:
            target = base_dir / unquote(uri)
            if not target.exists():
                raise ValueError(
                    f"glTF buffer {i} points at {uri!r}, which does not exist next to "
                    "the .gltf file. Use a .glb (self-contained) or keep the .bin beside it."
                )
            buffers.append(target.read_bytes())
    return buffers


def _read_accessor(doc: dict, buffers: list[bytes], index: int) -> NDArray:
    """Decode one accessor to an ``(count, ncomp)`` array, honouring ``byteStride``."""
    accessor = doc["accessors"][index]
    if "sparse" in accessor:
        raise ValueError(
            "glTF sparse accessors are not supported. Re-export without sparse "
            "storage, or run the file through `gltf-transform resample`."
        )
    count = int(accessor["count"])
    ncomp = _TYPE_COUNTS[accessor["type"]]
    dtype = np.dtype(_COMPONENT_TYPES[int(accessor["componentType"])])

    if "bufferView" not in accessor:
        # A bufferView-less accessor is defined to be all zeros.
        return np.zeros((count, ncomp), dtype=dtype)

    view = doc["bufferViews"][int(accessor["bufferView"])]
    blob = buffers[int(view.get("buffer", 0))]
    base = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", 0)) or dtype.itemsize * ncomp

    if stride == dtype.itemsize * ncomp:
        # Tightly packed — one frombuffer.
        flat = np.frombuffer(blob, dtype=dtype, count=count * ncomp, offset=base)
        return flat.reshape(count, ncomp)

    # Interleaved: take a strided view over raw bytes, then reinterpret each row.
    #
    # The span is `(count - 1) * stride + element`, NOT `count * stride`: the last
    # element occupies only its own width, and the padding that would follow it need not
    # exist. Asking for `count * stride` over-reads by `stride - element` bytes, which is
    # invisible whenever anything follows the view in the same buffer — index data, say —
    # and raises only when the interleaved view sits at the very end of the buffer. That
    # is exactly the layout a tightly-packed exporter produces.
    element = dtype.itemsize * ncomp
    span = (count - 1) * stride + element if count else 0
    window = np.frombuffer(blob, dtype=np.uint8, offset=base, count=span)
    rows = np.lib.stride_tricks.as_strided(
        window, shape=(count, element), strides=(stride, 1)
    )
    return np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)


def _node_matrix(node: dict) -> NDArray[np.float64]:
    """A node's local 4x4, from either ``matrix`` or TRS. Returned ROW-major (numpy)."""
    if "matrix" in node:
        # glTF stores column-major; transpose into numpy's row-major convention.
        m: NDArray[np.float64] = (
            np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
        )
        return m
    out = np.eye(4)
    if "scale" in node:
        out[:3, :3] = out[:3, :3] @ np.diag(node["scale"])
    if "rotation" in node:
        x, y, z, w = node["rotation"]  # glTF quaternions are xyzw
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        out[:3, :3] = rot @ out[:3, :3]
    if "translation" in node:
        out[:3, 3] = node["translation"]
    return out


def read_gltf(path: Path) -> dict[str, object]:
    """Decode a .gltf or .glb into merged component arrays.

    Every triangle primitive reachable from the default scene is transformed into world
    space by its node chain and concatenated into one mesh. Materials, textures,
    animations, skins, cameras and morph targets are ignored — Luxar meshes carry
    geometry plus per-vertex colour, and nothing else in the file maps onto that.
    """
    raw = path.read_bytes()
    if raw[:4] == b"glTF":
        doc, glb_binary = _parse_glb(raw)
    else:
        doc, glb_binary = json.loads(raw.decode("utf-8")), None

    for ext in doc.get("extensionsRequired", []):
        if ext in _COMPRESSION_EXTENSIONS:
            name = _COMPRESSION_EXTENSIONS[ext]
            raise ValueError(
                f"{path.name} requires {ext} ({name}-compressed geometry), which this "
                f"reader cannot decode. Decompress first — e.g. "
                f"`gltf-transform {'dedup' if name == 'Draco' else 'meshopt'} in.glb out.glb` "
                "with the extension removed, or re-export from Blender with compression off."
            )

    buffers = _load_buffers(doc, path.parent, glb_binary)
    meshes = doc.get("meshes", [])
    nodes = doc.get("nodes", [])

    vert_blocks: list[NDArray[np.float32]] = []
    norm_blocks: list[NDArray[np.float32] | None] = []
    color_blocks: list[NDArray[np.uint8] | None] = []
    face_blocks: list[NDArray[np.uint32]] = []
    offset = 0
    skipped_modes = 0

    def emit(mesh_index: int, world: NDArray[np.float64]) -> None:
        nonlocal offset, skipped_modes
        for prim in meshes[mesh_index].get("primitives", []):
            if int(prim.get("mode", 4)) != 4:
                skipped_modes += 1
                continue
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = _read_accessor(doc, buffers, attrs["POSITION"]).astype(np.float64)
            # Affine transform into world space.
            pos = (world[:3, :3] @ pos.T).T + world[:3, 3]

            nrm = None
            if "NORMAL" in attrs:
                n = _read_accessor(doc, buffers, attrs["NORMAL"]).astype(np.float64)
                # Normals transform by the inverse-transpose, or a non-uniform scale
                # tilts them off the surface.
                nrm = (np.linalg.inv(world[:3, :3]).T @ n.T).T
                lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
                nrm = np.divide(nrm, lengths, out=np.zeros_like(nrm), where=lengths > 0)
                nrm = nrm.astype(np.float32)

            col = None
            if "COLOR_0" in attrs:
                c = _read_accessor(doc, buffers, attrs["COLOR_0"])
                if c.dtype == np.float32:
                    col = np.clip(c * 255.0, 0, 255).astype(np.uint8)
                elif c.dtype == np.uint16:
                    col = (c >> 8).astype(np.uint8)
                else:
                    col = c.astype(np.uint8)

            if "indices" in prim:
                idx = _read_accessor(doc, buffers, prim["indices"]).reshape(-1)
            else:
                # No index: the primitive IS a soup. Welding happens downstream.
                idx = np.arange(pos.shape[0], dtype=np.uint32)
            usable = (idx.shape[0] // 3) * 3
            faces = idx[:usable].reshape(-1, 3).astype(np.uint32) + offset

            # A REFLECTING transform (negative determinant — e.g. `scale: [-1, 1, 1]`,
            # which glTF exporters emit routinely for mirrored parts) reverses a
            # triangle's geometric winding. Positions and normals above are transformed
            # correctly, but the index ORDER is untouched, so the face's implied winding
            # would now disagree with its own normal. Under single-sided rendering the
            # mirrored part faces away and vanishes; with stored normals it lights from
            # the wrong side.
            #
            # Swapping two of the three indices restores the winding. Done per primitive
            # rather than globally because the sign is a property of that node's chain —
            # a file can mirror one part and not another.
            if np.linalg.det(world[:3, :3]) < 0:
                faces = faces[:, [0, 2, 1]]

            vert_blocks.append(pos.astype(np.float32))
            norm_blocks.append(nrm)
            color_blocks.append(col)
            face_blocks.append(faces)
            offset += pos.shape[0]

    def walk(
        node_index: int, parent: NDArray[np.float64], ancestors: frozenset[int]
    ) -> None:
        # glTF node graphs are a strict forest — a node has at most one parent — so a
        # child edge back into an ancestor is a malformed file. Without this the walk
        # recurses until Python's stack limit and raises RecursionError, which is not
        # a ValueError and so escapes the CLI's error funnel as a raw traceback.
        #
        # ANCESTOR-scoped rather than a global visited set: the same node reached twice
        # by different paths is a DAG, technically invalid but harmless to us (it just
        # emits the mesh twice, which is what the file asks for). Only a true cycle is
        # unrecoverable, so only a true cycle is refused.
        if node_index in ancestors:
            raise ValueError(
                f"{path.name}: node {node_index} is its own ancestor — the glTF node "
                "graph has a cycle, which the format does not permit (nodes form a "
                "forest). The file is malformed; re-export it."
            )
        if node_index < 0 or node_index >= len(nodes):
            raise ValueError(
                f"{path.name}: node index {node_index} is out of range "
                f"({len(nodes)} nodes declared)"
            )
        node = nodes[node_index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            emit(int(node["mesh"]), world)
        descended = ancestors | {node_index}
        for child in node.get("children", []):
            walk(int(child), world, descended)

    scenes = doc.get("scenes", [])
    scene_index = int(doc.get("scene", 0))
    if scenes and scene_index < len(scenes):
        for root in scenes[scene_index].get("nodes", []):
            walk(int(root), np.eye(4), frozenset())
    elif nodes:
        for i in range(len(nodes)):
            walk(i, np.eye(4), frozenset())
    else:
        # No node graph at all — take the meshes as authored.
        for i in range(len(meshes)):
            emit(i, np.eye(4))

    if not vert_blocks:
        raise ValueError(
            f"{path.name}: no triangle geometry found"
            + (
                f" ({skipped_modes} non-triangle primitive(s) skipped)"
                if skipped_modes
                else ""
            )
        )

    vertices = np.concatenate(vert_blocks, axis=0)
    faces = np.concatenate(face_blocks, axis=0)

    # A partial attribute is worse than none: filling the gaps with a default would
    # produce a surface that is smooth-shaded in one region and flat in another with no
    # indication why. Keep the attribute only when EVERY primitive supplied it.
    normals = (
        np.concatenate([n for n in norm_blocks], axis=0)
        if all(n is not None for n in norm_blocks)
        else None
    )
    colors: NDArray[np.uint8] | None = None
    present = [c for c in color_blocks if c is not None]
    if color_blocks and len(present) == len(color_blocks):
        # Mixed RGB/RGBA across primitives: promote everything to RGBA (opaque).
        if len({c.shape[1] for c in present}) > 1:
            present = [
                c
                if c.shape[1] == 4
                else np.hstack([c, np.full((c.shape[0], 1), 255, np.uint8)])
                for c in present
            ]
        colors = np.concatenate(present, axis=0)

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "colors": colors,
        "skipped_primitives": skipped_modes,
    }
