"""STL reader — binary and ASCII.

STL is a pure triangle SOUP: every triangle carries its own three vertices, so a closed
surface stores each corner once per incident face and no index exists. Left that way it
would trip the writer's unwelded-mesh authoring lint, defeat per-vertex normals, and
give picking a different vertex ordinal for the same corner depending on which triangle
was hit. So this reader always welds — see :mod:`._weld`.

**The ascii-vs-binary discriminator is the classic trap.** An ASCII STL begins with the
word ``solid``; a binary STL begins with an 80-byte header that is *also allowed to*,
and plenty of exporters write "solid" into it. Sniffing the magic word therefore
misclassifies real files. The reliable test is arithmetic: a binary STL is exactly
``84 + 50 * n`` bytes for the ``n`` declared at offset 80.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: 80-byte header + uint32 triangle count.
_BINARY_PREAMBLE = 84
#: normal (3f) + 3 vertices (9f) + attribute byte count (u2) = 50 bytes.
_BINARY_STRIDE = 50

_BINARY_RECORD = np.dtype(
    [("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
)


def is_binary_stl(raw: bytes) -> bool:
    """Decide binary vs ASCII by SIZE, not by the leading magic word.

    Returns False for anything too short to carry the preamble, which lets the ASCII
    path produce the better error for a truncated or non-STL file.
    """
    if len(raw) < _BINARY_PREAMBLE:
        return False
    count = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    return len(raw) == _BINARY_PREAMBLE + _BINARY_STRIDE * count


def read_stl(path: Path) -> dict[str, object]:
    """Decode an STL into a soup of triangle corners, pre-weld.

    Per-facet normals are deliberately DISCARDED. STL stores one normal per facet, but
    ``add_mesh`` takes per-VERTEX normals; after welding, one vertex belongs to several
    facets with different normals and there is no non-arbitrary way to pick. Dropping
    them puts the mesh on the shader's derivative flat-normal path, which reproduces
    exactly the faceted look STL describes — the same picture, without inventing data.
    """
    raw = path.read_bytes()

    if is_binary_stl(raw):
        count = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
        records = np.frombuffer(
            raw, dtype=_BINARY_RECORD, count=count, offset=_BINARY_PREAMBLE
        )
        soup = records["vertices"].reshape(-1, 3).astype(np.float32)
    else:
        text = raw.decode("ascii", errors="replace")
        if "facet" not in text:
            raise ValueError(
                f"{path.name}: not an STL — no 'facet' keyword, and the file size is "
                f"not the 84 + 50n a binary STL would be ({len(raw)} bytes)"
            )
        coords = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text, re.I)
        if not coords:
            raise ValueError(f"{path.name}: ASCII STL has no 'vertex' lines")
        soup = np.asarray(coords, dtype=np.float32)

    if soup.shape[0] % 3 != 0:
        raise ValueError(
            f"{path.name}: STL has {soup.shape[0]} vertices, not a multiple of 3 — "
            "the triangle soup is incomplete"
        )

    faces = np.arange(soup.shape[0], dtype=np.uint32).reshape(-1, 3)
    return {"vertices": soup, "faces": faces, "normals": None, "colors": None}
