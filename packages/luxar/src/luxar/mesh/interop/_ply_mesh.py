"""PLY mesh reader — ascii, binary little-endian, and binary big-endian.

Deliberately NOT built on ``luxar.gsplats.interop._ply``. That parser exists to read
Gaussian-splat PLYs and rejects, by design, the two things every classical mesh PLY
needs: ``property list`` (which is how a ``face`` element declares its vertex indices)
and any format other than ``binary_little_endian``. Generalizing it would put list
handling and three format branches into a module whose only other consumer is a
fixed-layout float table, so this is a second parser rather than a shared one — a
deliberate duplication, recorded here so it reads as a choice.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

#: PLY scalar type names → numpy dtypes, including the pre-1.0 short spellings that
#: MeshLab and older exporters still emit.
_PLY_TYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_BYTE_ORDER = {
    "ascii": "=",
    "binary_little_endian": "<",
    "binary_big_endian": ">",
}


class PlyProperty(NamedTuple):
    """One declared property, in the order the header declared it.

    ``count_type`` is ``None`` for a plain scalar property and the list's length type
    for a ``property list``, in which case ``type`` is the ITEM type.
    """

    name: str
    type: str
    count_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.count_type is not None


class PlyElement:
    """One declared element: its name, its row count, and its properties IN ORDER.

    Declaration order is the row layout — a ``face`` element may carry a scalar before
    ``vertex_indices`` (a per-face flag) or a second list after it (Blender writes
    ``texcoord`` that way). Splitting the properties into "scalars" and "lists" loses
    that order, and a reader that then assumes list-first-scalars-after walks the body
    at the wrong offsets, silently decoding one element's bytes as another's.
    """

    def __init__(self, name: str, count: int) -> None:
        self.name = name
        self.count = count
        #: Every property, in declaration order.
        self.entries: list[PlyProperty] = []

    @property
    def properties(self) -> list[tuple[str, str]]:
        """``(prop_name, scalar_type)`` for the plain properties, in order."""
        return [(p.name, p.type) for p in self.entries if not p.is_list]

    @property
    def lists(self) -> list[PlyProperty]:
        """The ``property list`` declarations, in order."""
        return [p for p in self.entries if p.is_list]

    @property
    def has_lists(self) -> bool:
        return any(p.is_list for p in self.entries)


def parse_ply_header(raw: bytes) -> tuple[list[PlyElement], str, int]:
    """Parse a PLY header.

    Returns ``(elements, fmt, body_offset)`` where ``fmt`` is one of ``ascii`` /
    ``binary_little_endian`` / ``binary_big_endian`` and ``body_offset`` is the byte
    offset just past ``end_header``.
    """
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("not a PLY file (no 'end_header' in the first block)")
    # Past the marker and its line terminator, which may be \n or \r\n.
    body_offset = raw.index(b"\n", end) + 1
    text = raw[:end].decode("ascii", errors="replace")

    if not text.lstrip().startswith("ply"):
        raise ValueError("not a PLY file (missing the 'ply' magic)")

    fmt_match = re.search(r"^format\s+(\S+)\s+([\d.]+)", text, re.M)
    if not fmt_match:
        raise ValueError("PLY header has no 'format' line")
    fmt = fmt_match.group(1)
    if fmt not in _BYTE_ORDER:
        raise ValueError(f"unsupported PLY format {fmt!r}")

    elements: list[PlyElement] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            elements.append(PlyElement(parts[1], int(parts[2])))
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                # property list <count_type> <item_type> <name>
                elements[-1].entries.append(
                    PlyProperty(parts[4], parts[3], count_type=parts[2])
                )
            else:
                elements[-1].entries.append(PlyProperty(parts[2], parts[1]))
    if not elements:
        raise ValueError("PLY header declares no elements")
    return elements, fmt, body_offset


def _scalar(fmt: str, ply_type: str) -> np.dtype:
    if ply_type not in _PLY_TYPES:
        raise ValueError(f"unknown PLY property type {ply_type!r}")
    return np.dtype(_BYTE_ORDER[fmt] + _PLY_TYPES[ply_type])


def _index_list(element: PlyElement) -> str:
    """Which of an element's lists holds its vertex indices.

    By name when one of the conventional spellings is present, and only then by
    position: an element carrying both ``texcoord`` and ``vertex_indices`` may declare
    them in either order, and taking the first list would import UV data as topology.
    """
    for prop in element.lists:
        if prop.name in ("vertex_indices", "vertex_index"):
            return prop.name
    return element.lists[0].name


def _read_fixed_element(
    buf: bytes, offset: int, element: PlyElement, fmt: str
) -> tuple[np.ndarray, int]:
    """Read a list-free element as one structured block. Returns ``(array, new_offset)``."""
    dtype = np.dtype([(n, _scalar(fmt, t)) for n, t in element.properties])
    nbytes = dtype.itemsize * element.count
    if offset + nbytes > len(buf):
        raise ValueError(
            f"PLY body is truncated: element '{element.name}' needs {nbytes} bytes "
            f"at offset {offset}, file has {len(buf) - offset}"
        )
    arr = np.frombuffer(buf, dtype=dtype, count=element.count, offset=offset)
    return arr, offset + nbytes


def _read_list_element(
    buf: bytes, offset: int, element: PlyElement, fmt: str
) -> tuple[list[list[int]], int]:
    """Read an element that carries a ``property list``, row by row.

    Row-by-row rather than vectorized because a PLY list is variable-length: a `face`
    element may mix triangles and quads, so the stride is not knowable up front. Face
    counts are orders of magnitude below vertex counts, so this loop is not the cost.

    Every property is consumed in DECLARATION order and only the vertex-index list's
    rows are returned. Walking the declared order is what the format requires: a scalar may
    precede the list (a per-face flag or colour) and a second list may follow it (an
    exporter writing `texcoord` after `vertex_indices`). Assuming list-first would
    consume the wrong bytes from row two onward, and the desynchronized offset then
    decodes the rest of the file — including any following element — as garbage.
    """
    wanted = _index_list(element)
    # Dtypes resolved once, not per row: this loop runs per face.
    plan = [
        (
            prop.name,
            _scalar(fmt, prop.count_type) if prop.count_type else None,
            _scalar(fmt, prop.type),
        )
        for prop in element.entries
    ]

    rows: list[list[int]] = []
    for _ in range(element.count):
        for name, count_dt, item_dt in plan:
            if count_dt is None:
                offset += item_dt.itemsize
                if offset > len(buf):
                    raise ValueError(
                        f"PLY body truncated reading '{element.name}.{name}'"
                    )
                continue
            if offset + count_dt.itemsize > len(buf):
                raise ValueError(f"PLY body truncated reading '{element.name}.{name}'")
            n = int(np.frombuffer(buf, dtype=count_dt, count=1, offset=offset)[0])
            offset += count_dt.itemsize
            if offset + item_dt.itemsize * n > len(buf):
                raise ValueError(f"PLY body truncated reading '{element.name}.{name}'")
            if name == wanted:
                rows.append(
                    np.frombuffer(buf, dtype=item_dt, count=n, offset=offset)
                    .astype(int)
                    .tolist()
                )
            offset += item_dt.itemsize * n
    return rows, offset


def _read_ascii_body(
    text: str, elements: list[PlyElement]
) -> dict[str, tuple[np.ndarray | None, list[list[int]] | None]]:
    """Read every element from an ASCII body in declaration order."""
    tokens_by_line = [ln.split() for ln in text.splitlines() if ln.strip()]
    out: dict[str, tuple[np.ndarray | None, list[list[int]] | None]] = {}
    line = 0
    for element in elements:
        # Checked up front so a truncated body raises a ValueError naming the element
        # rather than an IndexError, which is not what the CLI's error funnel catches.
        if line + element.count > len(tokens_by_line):
            raise ValueError(
                f"PLY body is truncated: element '{element.name}' declares "
                f"{element.count} rows, only {max(0, len(tokens_by_line) - line)} remain"
            )
        if element.has_lists:
            wanted = _index_list(element)
            rows: list[list[int]] = []
            for _ in range(element.count):
                parts = tokens_by_line[line]
                line += 1
                # Declaration order again — see `_read_list_element`. ASCII rows are
                # whitespace-delimited rather than fixed-width, so the cursor walks
                # tokens instead of bytes, but the rule is identical.
                cursor = 0
                for prop in element.entries:
                    if not prop.is_list:
                        cursor += 1
                        continue
                    n = int(parts[cursor])
                    cursor += 1
                    if prop.name == wanted:
                        rows.append([int(v) for v in parts[cursor : cursor + n]])
                    cursor += n
            out[element.name] = (None, rows)
        else:
            names = [n for n, _ in element.properties]
            block = (
                np.array(
                    [
                        [float(v) for v in tokens_by_line[line + i]]
                        for i in range(element.count)
                    ],
                    dtype=np.float64,
                )
                if element.count
                else np.zeros((0, len(names)), dtype=np.float64)
            )
            line += element.count
            rec = np.zeros(
                element.count, dtype=np.dtype([(col, "f8") for col in names])
            )
            for j, col in enumerate(names):
                if j < block.shape[1]:
                    rec[col] = block[:, j]
            out[element.name] = (rec, None)
    return out


def read_ply_mesh(path: Path) -> dict[str, object]:
    """Decode a classical mesh PLY into raw component arrays.

    Returns a dict with ``vertices``, ``faces``, ``normals``, ``colors`` — the shape
    :func:`luxar.mesh.interop.mesh_import.import_mesh` assembles a ``TriangleMesh``
    from. Kept as a plain dict so the reader owns no policy.
    """
    raw = path.read_bytes()
    elements, fmt, body_offset = parse_ply_header(raw)
    by_name = {e.name: e for e in elements}

    if "vertex" not in by_name:
        raise ValueError(f"{path.name}: PLY has no 'vertex' element")
    if "face" not in by_name:
        raise ValueError(
            f"{path.name}: PLY has no 'face' element — this is a point cloud, not a "
            "mesh. Load it with `scene.add_points(...)` instead."
        )

    face_rows: list[list[int]] = []
    if fmt == "ascii":
        parsed = _read_ascii_body(
            raw[body_offset:].decode("ascii", errors="replace"), elements
        )
        vertex_rec = parsed["vertex"][0]
        face_rows = parsed["face"][1] or []
    else:
        offset = body_offset
        vertex_rec = None
        for element in elements:
            if element.has_lists:
                rows, offset = _read_list_element(raw, offset, element, fmt)
                if element.name == "face":
                    face_rows = rows
            else:
                arr, offset = _read_fixed_element(raw, offset, element, fmt)
                if element.name == "vertex":
                    vertex_rec = arr
    if vertex_rec is None:
        raise ValueError(f"{path.name}: PLY 'vertex' element could not be read")

    names = set(vertex_rec.dtype.names or ())
    missing = {"x", "y", "z"} - names
    if missing:
        raise ValueError(
            f"{path.name}: PLY vertex element is missing {sorted(missing)} — "
            "expected x/y/z position properties"
        )
    vertices = np.stack(
        [vertex_rec["x"], vertex_rec["y"], vertex_rec["z"]], axis=1
    ).astype(np.float32)

    normals = None
    if {"nx", "ny", "nz"} <= names:
        normals = np.stack(
            [vertex_rec["nx"], vertex_rec["ny"], vertex_rec["nz"]], axis=1
        ).astype(np.float32)

    colors = None
    if {"red", "green", "blue"} <= names:
        channels = ["red", "green", "blue"] + (["alpha"] if "alpha" in names else [])
        colors = np.stack([vertex_rec[c] for c in channels], axis=1)
        # PLY colours are uchar by convention but `float` is legal and some exporters
        # emit 0..1. Distinguish by range rather than by declared dtype, because the
        # ascii path widens everything to f8 and would otherwise lose the distinction.
        peak = float(np.max(colors)) if colors.size else 0.0
        colors = (
            np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            if peak <= 1.0
            else np.clip(colors, 0, 255).astype(np.uint8)
        )

    return {
        "vertices": vertices,
        "face_rows": face_rows,
        "normals": normals,
        "colors": colors,
    }
