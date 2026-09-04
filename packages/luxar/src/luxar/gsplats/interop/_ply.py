"""Binary-little-endian PLY header parsing (shared by the INRIA and SuperSplat
Gaussian-splat dialects). Extracted from ``classical_splats.py`` as part of the
per-concern module split; ``classical_splats`` re-exports nothing from here
(these helpers are private to the readers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PLY_DTYPES = {
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


@dataclass
class _PlyElement:
    name: str
    count: int
    properties: list[tuple[str, str]] = field(default_factory=list)  # (name, np dtype)

    def dtype(self) -> np.dtype:
        return np.dtype([(name, "<" + dt) for name, dt in self.properties])


def _parse_ply_header(raw: bytes) -> tuple[list[_PlyElement], int]:
    """Parse a binary-little-endian PLY header.

    Returns the declared elements (in file order) and the byte offset of the
    binary body. Only scalar properties are supported (3DGS dialects never use
    ``property list``).
    """
    end = raw.find(b"end_header\n")
    if not raw.startswith(b"ply") or end < 0:
        raise ValueError("Not a PLY file (missing 'ply' magic or 'end_header')")
    header = raw[:end].decode("ascii", errors="replace")
    body_offset = end + len(b"end_header\n")

    if not re.search(r"^format\s+binary_little_endian\s+1\.0\s*$", header, re.M):
        raise ValueError(
            "Only binary_little_endian PLY is supported (ASCII / big-endian "
            "Gaussian-splat PLY files are not produced by any known tool)"
        )

    elements: list[_PlyElement] = []
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "element":
            elements.append(_PlyElement(name=parts[1], count=int(parts[2])))
        elif parts[0] == "property":
            if not elements:
                raise ValueError("PLY property declared before any element")
            if parts[1] == "list":
                raise ValueError("PLY list properties are not supported")
            dt = _PLY_DTYPES.get(parts[1])
            if dt is None:
                raise ValueError(f"Unsupported PLY property type: {parts[1]}")
            elements[-1].properties.append((parts[-1], dt))
    return elements, body_offset


def _read_ply_elements(path: Path) -> dict[str, np.ndarray]:
    """Read all elements of a binary PLY into structured arrays keyed by name."""
    with open(path, "rb") as f:
        head = f.read(64 * 1024)
        elements, body_offset = _parse_ply_header(head)
        f.seek(body_offset)
        out: dict[str, np.ndarray] = {}
        for el in elements:
            dtype = el.dtype()
            arr = np.fromfile(f, dtype=dtype, count=el.count)
            if arr.shape[0] != el.count:
                raise ValueError(
                    f"PLY element '{el.name}' truncated: expected {el.count} "
                    f"records, read {arr.shape[0]}"
                )
            out[el.name] = arr
    return out


def _stack_fields(arr: np.ndarray, names: list[str]) -> np.ndarray:
    """Stack structured-array fields into a float32 (N, len(names)) array."""
    return np.stack([arr[name].astype(np.float32) for name in names], axis=1)
