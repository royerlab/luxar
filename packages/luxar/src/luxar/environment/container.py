"""The on-the-wire container a baked environment travels in.

``luxar env bake`` drives the viewer headlessly; the viewer captures six cube
faces as HALF floats (a PNG would clip everything above 1.0 and destroy the
highlights that make metals read) and hands back ONE binary blob. This module is
the single definition of that blob, shared by the Python reader
(:func:`unpack`), the Python writer used by tests and by ``luxar env attach``
(:func:`pack`), and — by contract, mirrored in
``packages/luxar-viewer/src/rendering/environment/bake.ts`` — the viewer.

Layout, all little-endian::

    8 bytes   magic  ``b"LXENV001"``
    4 bytes   uint32 header length ``n``
    n bytes   UTF-8 JSON header (see :data:`REQUIRED_HEADER_KEYS`)
    rest      uint16 samples, ``(6, H, W, 4)`` C-order, IEEE half-float bits

The samples are IEEE half-float BITS stored as ``uint16`` end to end: the GPU
readback yields them, three's ``HalfFloatType`` cube texture consumes them, and a
zarr ``uint16`` array needs no ``Float16Array`` on the reading engine (zarrita
throws on ``<f2`` without one). ``np.asarray(faces).view(np.float16)`` is the
one-line way to look at the values from Python.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Tuple

import numpy as np

#: Magic prefix; the trailing digits are the container version.
MAGIC: bytes = b"LXENV001"

#: The ``format`` value the header must carry.
ENVIRONMENT_FORMAT: str = "cube-faces-half"

#: The sample encoding stamped on the zarr array by ``attach``.
SAMPLE_FORMAT: str = "half-float-bits"

#: Face order of the samples — three's ``CubeTexture`` order.
FACE_ORDER: Tuple[str, ...] = ("px", "nx", "py", "ny", "pz", "nz")

#: Default cube face size for a bake — what the viewer defaults to as well. Lives
#: here rather than in ``bake.py`` so the CLI can read it without importing the
#: bake module (which imports ``luxar.cli`` and would close an import cycle).
DEFAULT_RESOLUTION: int = 128

#: Header keys :func:`unpack` insists on. ``scene_content_hash`` is the guard
#: ``luxar env attach`` and the viewer compare against the scene's own digest;
#: ``coordinate_system`` and ``face_order`` are what lets the viewer rebuild the
#: EXACT texture the capture produced; ``probe`` and ``resolution`` are the
#: capture's parameters, kept so a re-bake can be judged idempotent.
REQUIRED_HEADER_KEYS: Tuple[str, ...] = (
    "format",
    "face_order",
    "coordinate_system",
    "probe",
    "resolution",
    "scene_content_hash",
)

_LEN = struct.Struct("<I")


def pack(header: Dict[str, Any], faces: np.ndarray) -> bytes:
    """Serialize ``header`` + ``faces`` into one container blob.

    ``faces`` must be ``uint16`` of shape ``(6, H, W, 4)`` with ``H == W ==
    header["resolution"]``; the header must carry :data:`REQUIRED_HEADER_KEYS`.
    Validated up front so a malformed blob is never produced.
    """
    _validate_header(header)
    faces = np.ascontiguousarray(faces)
    _validate_faces(faces, int(header["resolution"]))
    body = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + _LEN.pack(len(body)) + body + faces.tobytes()


def unpack(data: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
    """Parse a container blob into ``(header, faces)``.

    Raises ``ValueError`` naming the first thing wrong: magic, header length,
    header JSON, a missing required key, or a sample payload whose length does
    not match ``6 * resolution² * 4`` halves.
    """
    view = memoryview(data)
    if len(view) < len(MAGIC) + _LEN.size or bytes(view[: len(MAGIC)]) != MAGIC:
        raise ValueError(
            f"Not a Luxar environment container (expected the {MAGIC!r} magic prefix)."
        )
    offset = len(MAGIC)
    (header_len,) = _LEN.unpack_from(view, offset)
    offset += _LEN.size
    if offset + header_len > len(view):
        raise ValueError(
            f"Environment container header claims {header_len} bytes but only "
            f"{len(view) - offset} remain."
        )
    try:
        header = json.loads(bytes(view[offset : offset + header_len]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Environment container header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("Environment container header must be a JSON object.")
    _validate_header(header)
    offset += header_len
    resolution = int(header["resolution"])
    expected = 6 * resolution * resolution * 4
    payload = view[offset:]
    if len(payload) != expected * 2:
        raise ValueError(
            f"Environment container carries {len(payload)} sample bytes; a "
            f"{resolution}px cube needs {expected * 2} (6 faces x {resolution}^2 x "
            "RGBA halves)."
        )
    faces = np.frombuffer(payload, dtype="<u2").reshape(6, resolution, resolution, 4)
    # A copy, so the array does not alias the (possibly memory-mapped) input.
    return header, np.array(faces, dtype=np.uint16)


def _validate_header(header: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_HEADER_KEYS if key not in header]
    if missing:
        raise ValueError(f"Environment header is missing {missing}.")
    if header["format"] != ENVIRONMENT_FORMAT:
        raise ValueError(
            f"Environment header format must be {ENVIRONMENT_FORMAT!r}, got "
            f"{header['format']!r}."
        )
    order = header["face_order"]
    if not isinstance(order, list) or tuple(order) != FACE_ORDER:
        raise ValueError(
            f"Environment header face_order must be {list(FACE_ORDER)}, got {order!r}."
        )
    resolution = header["resolution"]
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution < 1
    ):
        raise ValueError(
            f"Environment header resolution must be a positive int, got {resolution!r}."
        )
    if (
        not isinstance(header["scene_content_hash"], str)
        or not header["scene_content_hash"]
    ):
        raise ValueError(
            "Environment header scene_content_hash must be a non-empty string."
        )
    for key in ("type", "kind"):
        if key in header:
            raise ValueError(
                f"Environment header must not carry a {key!r} key: the viewer skips "
                "the environment group as a metadata sidecar only while it has "
                "neither."
            )


def _validate_faces(faces: np.ndarray, resolution: int) -> None:
    if faces.dtype != np.uint16:
        raise ValueError(
            f"Environment faces must be uint16 half-float bits, got {faces.dtype}."
        )
    if faces.shape != (6, resolution, resolution, 4):
        raise ValueError(
            f"Environment faces must have shape (6, {resolution}, {resolution}, 4), "
            f"got {faces.shape}."
        )
