"""Import classical (photogrammetric) Gaussian-splat files as :class:`GSplatData`.

Classical 3D Gaussian Splatting — the INRIA reference implementation and the
ecosystem around it (SuperSplat, Scaniverse, antimatter15's web viewer, …) —
stores splats as *position + per-axis scale + rotation quaternion + opacity +
spherical-harmonics color*. Luxar stores *center + packed lower-triangular
Cholesky factor + amplitude + RGB color*. This module reads the four common
on-disk dialects into a shared :class:`ClassicalSplats` intermediate and
converts it to a :class:`~luxar.gsplats.gsplat_data.GSplatData`, after which
the entire Luxar toolchain (LOD recipes, partition, filter, scenes, viewer)
applies unchanged.

Supported dialects
------------------
- ``inria``      — the reference ``point_cloud.ply`` written by INRIA-style trainers
- ``splat``      — antimatter15 ``.splat`` (flat 32-byte records)
- ``spz``        — Niantic/Scaniverse ``.spz`` (gzipped, quantized)
- ``supersplat`` — PlayCanvas/SuperSplat *compressed* ``.ply`` (chunked, bit-packed)

Spherical harmonics are reduced to the DC band: view-dependent ``f_rest``
coefficients are dropped and the DC term is baked to per-splat RGB. Opacity
maps to Luxar ``amplitudes`` (both live in ``[0, 1]`` after the sigmoid).

Everything here is NumPy + stdlib only — no torch, no external parsers.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from luxar.gsplats.gsplat_data import GSplatData

__all__ = [
    "ClassicalSplats",
    "read_inria_ply",
    "read_antimatter_splat",
    "read_spz",
    "read_supersplat_ply",
    "read_sog",
    "detect_classical_format",
    "classical_to_gsplat_data",
    "import_gsplats",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "CLASSICAL_FORMATS",
]

#: Formats accepted by :func:`import_gsplats`'s ``format`` argument.
CLASSICAL_FORMATS = ("inria", "splat", "spz", "supersplat", "sog")

#: The degree-0 real spherical-harmonics basis constant Y_0^0 = 1/(2*sqrt(pi)).
SH_C0 = 0.28209479177387814


# ─────────────────────────────────────────────────────────────────────────────
# Shared intermediate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassicalSplats:
    """Decoded classical splats in the source file's world (x, y, z) frame.

    All decode-time nonlinearities are already applied: ``scales`` are linear
    standard deviations (``exp`` applied), ``opacities`` are in ``[0, 1]``
    (sigmoid applied), ``quaternions`` are unit-norm w-first, and ``colors``
    are DC-baked RGB in ``[0, 1]``.
    """

    positions: np.ndarray  # (N, 3) float32, world x, y, z
    scales: np.ndarray  # (N, 3) float32, linear sigmas
    quaternions: np.ndarray  # (N, 4) float32, (w, x, y, z), unit norm
    opacities: np.ndarray  # (N,)  float32 in [0, 1]
    colors: np.ndarray  # (N, 3) float32 in [0, 1]
    sh_degree: int = 0  # SH degree declared by the source (f_rest dropped)
    source_format: str = ""  # one of CLASSICAL_FORMATS
    y_up: bool = False  # True if the source is already Y-up (SPZ/RUB); COLMAP-style
    #                     y-down sources (INRIA/.splat/SuperSplat) leave this False

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        if self.positions.shape != (n, 3):
            raise ValueError(f"positions must be (N, 3); got {self.positions.shape}")
        for name, arr, shape in (
            ("scales", self.scales, (n, 3)),
            ("quaternions", self.quaternions, (n, 4)),
            ("opacities", self.opacities, (n,)),
            ("colors", self.colors, (n, 3)),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name} must be {shape}; got {arr.shape}")

    @property
    def n_splats(self) -> int:
        return int(self.positions.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion / rotation helpers (w-first convention throughout)
# ─────────────────────────────────────────────────────────────────────────────


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions ``(N, 4)`` (w, x, y, z) to rotation matrices ``(N, 3, 3)``.

    Quaternions are re-normalized defensively; zero-norm quaternions decode to
    the identity rotation.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"q must be (N, 4); got {q.shape}")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.divide(q, norm, out=np.zeros_like(q), where=norm > 0)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    identity = norm[:, 0] == 0
    w = np.where(identity, 1.0, w)

    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices ``(N, 3, 3)`` to unit quaternions ``(N, 4)`` (w, x, y, z).

    Uses Shepperd's method (branch on the largest diagonal combination) for
    numerical stability near 180° rotations. Inputs must be proper rotations
    (``det = +1``); the caller is responsible for reflection correction.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"R must be (N, 3, 3); got {R.shape}")
    n = R.shape[0]
    q = np.empty((n, 4), dtype=np.float64)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    # Candidate squared components (all >= 0 up to rounding); branch on the
    # largest so the divisor 4s below is always well-conditioned.
    qw2 = np.maximum(0.0, 1.0 + trace) / 4.0
    qx2 = np.maximum(0.0, 1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]) / 4.0
    qy2 = np.maximum(0.0, 1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]) / 4.0
    qz2 = np.maximum(0.0, 1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]) / 4.0
    branch = np.argmax(np.stack([qw2, qx2, qy2, qz2], axis=1), axis=1)

    def _fill(
        mask: np.ndarray, sq: np.ndarray, cols: list[Optional[np.ndarray]]
    ) -> None:
        if np.any(mask):
            s = np.sqrt(sq[mask])
            for target, col in enumerate(cols):
                q[mask, target] = s if col is None else col[mask] / (4 * s)

    r = R  # column shorthands (differences/sums of off-diagonal entries)
    wx = r[:, 2, 1] - r[:, 1, 2]
    wy = r[:, 0, 2] - r[:, 2, 0]
    wz = r[:, 1, 0] - r[:, 0, 1]
    xy = r[:, 0, 1] + r[:, 1, 0]
    xz = r[:, 0, 2] + r[:, 2, 0]
    yz = r[:, 1, 2] + r[:, 2, 1]
    _fill(branch == 0, qw2, [None, wx, wy, wz])
    _fill(branch == 1, qx2, [wx, None, xy, xz])
    _fill(branch == 2, qy2, [wy, xy, None, yz])
    _fill(branch == 3, qz2, [wz, xz, yz, None])

    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Canonical sign: w >= 0.
    q[q[:, 0] < 0] *= -1
    return q


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe sigmoid (clips the exponent to avoid overflow warnings)."""
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions to unit length (zero-norm rows become identity)."""
    q = q.astype(np.float32, copy=True)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.divide(q, norm, out=q, where=norm > 0)
    q[norm[:, 0] == 0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q


# ─────────────────────────────────────────────────────────────────────────────
# PLY header parsing (shared by the INRIA and SuperSplat dialects)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Dialect readers
# ─────────────────────────────────────────────────────────────────────────────


def read_inria_ply(path: Union[str, Path]) -> ClassicalSplats:
    """Read an INRIA-style 3DGS ``point_cloud.ply``.

    The header drives the layout: the SH degree is derived from the number of
    ``f_rest_*`` properties, so degree-0..3 files all parse. ``f_rest`` bands
    (view-dependent color) are dropped; the DC band is baked to RGB.
    """
    path = Path(path)
    elements = _read_ply_elements(path)
    if "vertex" not in elements:
        raise ValueError(f"{path.name}: PLY has no 'vertex' element")
    v = elements["vertex"]
    names = set(v.dtype.names or ())

    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            f"{path.name}: not an INRIA 3DGS PLY (missing properties: {missing})"
        )

    positions = _stack_fields(v, ["x", "y", "z"])
    scales = np.exp(_stack_fields(v, ["scale_0", "scale_1", "scale_2"]))
    quaternions = _normalize_quat(
        _stack_fields(v, ["rot_0", "rot_1", "rot_2", "rot_3"])
    )
    opacities = _sigmoid(v["opacity"].astype(np.float32))

    n_rest = sum(1 for name in names if name.startswith("f_rest_"))
    # 3 * ((degree+1)^2 - 1) rest coefficients for RGB SH.
    sh_degree = int(round(np.sqrt(n_rest / 3 + 1))) - 1 if n_rest else 0

    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        dc = _stack_fields(v, ["f_dc_0", "f_dc_1", "f_dc_2"])
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0).astype(np.float32)
    else:
        colors = np.full((v.shape[0], 3), 0.5, dtype=np.float32)

    return ClassicalSplats(
        positions=positions,
        scales=scales.astype(np.float32),
        quaternions=quaternions,
        opacities=opacities.astype(np.float32),
        colors=colors,
        sh_degree=sh_degree,
        source_format="inria",
    )


_SPLAT_RECORD = np.dtype(
    [
        ("position", "<f4", 3),
        ("scale", "<f4", 3),
        ("rgba", "u1", 4),
        ("rot", "u1", 4),
    ]
)


def read_antimatter_splat(path: Union[str, Path]) -> ClassicalSplats:
    """Read an antimatter15 ``.splat`` file (flat 32-byte records).

    Scales are stored linear (``exp`` already applied by the converter), color
    is DC-baked RGB with opacity in the alpha byte, and the rotation is the
    unit quaternion quantized as ``round(q * 128 + 128)`` in (w, x, y, z) order.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) == 0 or len(raw) % _SPLAT_RECORD.itemsize != 0:
        raise ValueError(
            f"{path.name}: size {len(raw)} is not a multiple of 32 bytes — "
            "not a .splat file"
        )
    records = np.frombuffer(raw, dtype=_SPLAT_RECORD)

    quaternions = _normalize_quat((records["rot"].astype(np.float32) - 128.0) / 128.0)
    rgba = records["rgba"].astype(np.float32) / 255.0

    return ClassicalSplats(
        positions=records["position"].astype(np.float32),
        scales=records["scale"].astype(np.float32),
        quaternions=quaternions,
        opacities=rgba[:, 3].copy(),
        colors=rgba[:, :3].copy(),
        sh_degree=0,
        source_format="splat",
    )


_SPZ_MAGIC = 0x5053474E  # "NGSP"
_SPZ_SH_DIM = {0: 0, 1: 3, 2: 8, 3: 15, 4: 24}
_SPZ_COLOR_SCALE = 0.15  # Niantic quantizes SH-DC with 0.15, not SH_C0


def _spz_unpack_smallest_three(comp: np.ndarray) -> np.ndarray:
    """Decode SPZ v3+ smallest-three packed rotations to (N, 4) (x, y, z, w).

    Bit layout per uint32 (little-endian value): bits 31..30 = index of the
    largest-magnitude component (into x,y,z,w); then three 10-bit fields
    (1 sign bit + 9-bit magnitude scaled by sqrt(1/2)/511) holding the
    remaining components in ascending index order, packed from high to low bits.
    """
    n = comp.shape[0]
    largest = (comp >> 30).astype(np.int64)
    quat = np.zeros((n, 4), dtype=np.float64)
    sq_sum = np.zeros(n, dtype=np.float64)
    remaining = comp.astype(np.uint64)
    # The packer shifted components in ascending index order, so the LAST
    # non-largest index sits in the LOW bits: consume indices 3 → 0.
    for idx in (3, 2, 1, 0):
        sel = largest != idx
        mag = np.sqrt(0.5) * (remaining & 511).astype(np.float64) / 511.0
        sign = ((remaining >> np.uint64(9)) & np.uint64(1)).astype(bool)
        value = np.where(sign, -mag, mag)
        quat[sel, idx] = value[sel]
        sq_sum += np.where(sel, value * value, 0.0)
        remaining = np.where(sel, remaining >> np.uint64(10), remaining)
    quat[np.arange(n), largest] = np.sqrt(np.maximum(0.0, 1.0 - sq_sum))
    return quat


def read_spz(path: Union[str, Path]) -> ClassicalSplats:
    """Read a Niantic/Scaniverse ``.spz`` file (gzipped quantized splats).

    Supports the legacy gzip container (versions 1–3, the format written by
    Scaniverse and shipped as the official samples). The v4 "NGSP" container
    (per-attribute ZSTD streams) is detected and rejected with a clear error.
    SPZ data is RUB (right-up-back, the three.js convention) — already Y-up.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) >= 4 and int.from_bytes(raw[:4], "little") == _SPZ_MAGIC:
        raise ValueError(
            f"{path.name}: SPZ v4 (NGSP/zstd container) is not supported yet — "
            "re-export as legacy .spz (v2/v3) or convert to .ply first"
        )
    if len(raw) < 2 or raw[:2] != b"\x1f\x8b":
        raise ValueError(f"{path.name}: not an SPZ file (no gzip or NGSP magic)")
    stream = gzip.decompress(raw)

    if len(stream) < 16:
        raise ValueError(f"{path.name}: SPZ stream truncated (no header)")
    magic = int.from_bytes(stream[0:4], "little")
    version = int.from_bytes(stream[4:8], "little")
    n = int.from_bytes(stream[8:12], "little")
    sh_degree, fractional_bits = stream[12], stream[13]
    if magic != _SPZ_MAGIC:
        raise ValueError(f"{path.name}: bad SPZ magic 0x{magic:08x}")
    if not 1 <= version <= 3:
        raise ValueError(f"{path.name}: unsupported SPZ version {version}")
    if sh_degree not in _SPZ_SH_DIM:
        raise ValueError(f"{path.name}: invalid SPZ SH degree {sh_degree}")
    if n == 0:
        raise ValueError(f"{path.name}: SPZ file holds zero splats")

    sh_dim = _SPZ_SH_DIM[sh_degree]
    pos_bytes = n * 3 * (2 if version == 1 else 3)
    rot_bytes = n * (4 if version >= 3 else 3)
    offset = 16

    def take(count: int) -> np.ndarray:
        nonlocal offset
        arr = np.frombuffer(stream, np.uint8, count, offset)
        if arr.shape[0] != count:
            raise ValueError(f"{path.name}: SPZ stream truncated mid-body")
        offset += count
        return arr

    pos_u8 = take(pos_bytes)
    alpha_u8 = take(n)
    color_u8 = take(n * 3)
    scale_u8 = take(n * 3)
    rot_u8 = take(rot_bytes)
    take(n * sh_dim * 3)  # f_rest bands — read past, dropped (DC-only policy)

    if version == 1:
        positions = pos_u8.view("<f2").astype(np.float32).reshape(n, 3)
    else:
        b = pos_u8.reshape(-1, 3).astype(np.int32)
        fixed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        fixed = np.where(fixed & 0x800000, fixed | ~0xFFFFFF, fixed)  # sign-extend
        positions = (
            fixed.astype(np.float32) / np.float32(1 << fractional_bits)
        ).reshape(n, 3)

    scales = np.exp(scale_u8.astype(np.float32).reshape(n, 3) / 16.0 - 10.0)

    if version >= 3:
        c = rot_u8.reshape(n, 4).astype(np.uint32)
        comp = c[:, 0] | (c[:, 1] << 8) | (c[:, 2] << 16) | (c[:, 3] << 24)
        quat_xyzw = _spz_unpack_smallest_three(comp)
    else:
        xyz = rot_u8.reshape(n, 3).astype(np.float64) / 127.5 - 1.0
        w = np.sqrt(np.maximum(0.0, 1.0 - np.square(xyz).sum(axis=1)))
        quat_xyzw = np.concatenate([xyz, w[:, None]], axis=1)
    quaternions = _normalize_quat(np.roll(quat_xyzw, 1, axis=1))  # → (w, x, y, z)

    # Alpha bytes are quantized sigmoid(logit): opacity = byte / 255 exactly.
    opacities = alpha_u8.astype(np.float32) / 255.0
    # Color bytes hold the SH-DC coefficient scaled by 0.15 (not SH_C0).
    f_dc = (color_u8.astype(np.float32).reshape(n, 3) / 255.0 - 0.5) / _SPZ_COLOR_SCALE
    colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)

    return ClassicalSplats(
        positions=positions,
        scales=scales.astype(np.float32),
        quaternions=quaternions,
        opacities=opacities,
        colors=colors,
        sh_degree=int(sh_degree),
        source_format="spz",
        y_up=True,
    )


_SUPERSPLAT_CHUNK = 256


def _unpack_11_10_11(word: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split uint32 words into three unorm fields: bits [31:21], [20:11], [10:0]."""
    a = (word >> 21).astype(np.float32) / 2047.0
    b = ((word >> 11) & 0x3FF).astype(np.float32) / 1023.0
    c = (word & 0x7FF).astype(np.float32) / 2047.0
    return a, b, c


def read_supersplat_ply(path: Union[str, Path]) -> ClassicalSplats:
    """Read a PlayCanvas/SuperSplat compressed ``.ply`` (chunked, bit-packed).

    Layout (splat-transform reference): a ``chunk`` element with 12 or 18
    float32 min/max bounds per 256-splat chunk, and a ``vertex`` element of
    four uint32 bitfields per splat (position 11-10-11, rotation
    2+10-10-10 smallest-three, scale 11-10-11 in log space, color 8-8-8-8).
    The optional ``sh`` element (f_rest bands) is dropped (DC-only policy).
    """
    path = Path(path)
    elements = _read_ply_elements(path)
    if "chunk" not in elements or "vertex" not in elements:
        raise ValueError(
            f"{path.name}: not a SuperSplat compressed PLY "
            "(missing 'chunk'/'vertex' elements)"
        )
    chunk_struct = elements["chunk"]
    vertex = elements["vertex"]
    chunk_names = list(chunk_struct.dtype.names or ())
    if len(chunk_names) not in (12, 18):
        raise ValueError(
            f"{path.name}: chunk element has {len(chunk_names)} properties; "
            "expected 12 or 18"
        )
    n = vertex.shape[0]
    num_chunks = chunk_struct.shape[0]
    if num_chunks != -(-n // _SUPERSPLAT_CHUNK):
        raise ValueError(
            f"{path.name}: {num_chunks} chunks for {n} splats "
            f"(expected ceil(n/{_SUPERSPLAT_CHUNK}))"
        )
    chunk = _stack_fields(chunk_struct, chunk_names)  # (num_chunks, 12|18)
    ci = np.arange(n) // _SUPERSPLAT_CHUNK

    def lerp(col_min: int, col_max: int, t: np.ndarray) -> np.ndarray:
        return np.asarray(chunk[ci, col_min] * (1.0 - t) + chunk[ci, col_max] * t)

    # Chunk property order: min_xyz(0-2), max_xyz(3-5), min_scale(6-8),
    # max_scale(9-11), then optionally min_rgb(12-14), max_rgb(15-17).
    px, py, pz = _unpack_11_10_11(vertex["packed_position"])
    positions = np.stack(
        [lerp(0, 3, px), lerp(1, 4, py), lerp(2, 5, pz)], axis=1
    ).astype(np.float32)

    sx, sy, sz = _unpack_11_10_11(vertex["packed_scale"])
    scales = np.exp(
        np.stack([lerp(6, 9, sx), lerp(7, 10, sy), lerp(8, 11, sz)], axis=1)
    ).astype(np.float32)

    # Rotation: smallest-three over the stored (rot_0..rot_3) = (w, x, y, z)
    # order — bits 31:30 give the largest component's index, the three 10-bit
    # fields hold the rest in ascending index order, each mapped to
    # [-sqrt(1/2), +sqrt(1/2)].
    word = vertex["packed_rotation"]
    which = (word >> 30).astype(np.int64)
    fields = [
        (((word >> 20) & 0x3FF).astype(np.float64) / 1023.0 - 0.5) * np.sqrt(2.0),
        (((word >> 10) & 0x3FF).astype(np.float64) / 1023.0 - 0.5) * np.sqrt(2.0),
        ((word & 0x3FF).astype(np.float64) / 1023.0 - 0.5) * np.sqrt(2.0),
    ]
    quaternions = np.zeros((n, 4), dtype=np.float64)
    rows = np.arange(n)
    for which_val in range(4):
        mask = which == which_val
        if not np.any(mask):
            continue
        others = [i for i in range(4) if i != which_val]
        for slot, target in enumerate(others):
            quaternions[mask, target] = fields[slot][mask]
    sq = np.square(quaternions).sum(axis=1)
    quaternions[rows, which] = np.sqrt(np.maximum(0.0, 1.0 - sq))

    word = vertex["packed_color"]
    r8 = ((word >> 24) & 0xFF).astype(np.float32) / 255.0
    g8 = ((word >> 16) & 0xFF).astype(np.float32) / 255.0
    b8 = ((word >> 8) & 0xFF).astype(np.float32) / 255.0
    alpha = ((word & 0xFF).astype(np.float32) / 255.0).copy()  # never chunk-lerped
    if len(chunk_names) == 18:
        rgb = np.stack([lerp(12, 15, r8), lerp(13, 16, g8), lerp(14, 17, b8)], axis=1)
    else:
        rgb = np.stack([r8, g8, b8], axis=1)
    colors = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    sh_element = elements.get("sh")
    n_rest = len(sh_element.dtype.names or ()) if sh_element is not None else 0
    sh_degree = int(round(np.sqrt(n_rest / 3 + 1))) - 1 if n_rest else 0

    return ClassicalSplats(
        positions=positions,
        scales=scales,
        quaternions=quaternions.astype(np.float32),
        opacities=alpha,
        colors=colors,
        sh_degree=sh_degree,
        source_format="supersplat",
    )


def _sog_accessor(path: Path) -> "Callable[[str], bytes]":
    """Return a ``read(name) -> bytes`` accessor for a SOG bundle.

    Accepts a directory of loose files (``meta.json`` + ``*.webp``, how
    SuperSplat/PlayCanvas serve them), a path to that ``meta.json``, or a
    single ``.sog`` ZIP archive bundling the same members.
    """
    if path.suffix.lower() == ".sog":
        import zipfile

        zf = zipfile.ZipFile(path)
        names = {Path(n).name: n for n in zf.namelist()}
        return lambda name: zf.read(names.get(name, name))
    # A directory is the bundle; a meta.json (or any member) → its parent.
    base = path if path.is_dir() else path.parent
    return lambda name: (base / name).read_bytes()


def _sog_load_image(
    read: "Callable[[str], bytes]",
    filename: str,
    count: int,
    min_channels: int = 1,
) -> np.ndarray:
    """Decode one SOG WebP to a ``(count, C)`` uint8 array (row-major, top-left).

    All property images share the pixel→Gaussian layout: the same pixel across
    images is the same Gaussian, so only the first ``count`` row-major pixels
    are valid (the tail is padding up to W×H). ``min_channels`` guards the
    per-attribute channel requirement (e.g. quats/sh0 need RGBA) so a malformed
    image fails with a clear message instead of a bare ``IndexError`` later.
    """
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - exercised via the guard test
        raise ImportError(
            "Reading the SOG format needs Pillow with WebP support: "
            "`pip install 'Pillow>=9.0.0'` (bundled in the `luxar[demos]` extra)."
        ) from e

    import io

    arr = np.asarray(Image.open(io.BytesIO(read(filename))))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    flat = arr.reshape(-1, arr.shape[-1])
    if flat.shape[1] < min_channels:
        raise ValueError(
            f"SOG image {filename!r} has {flat.shape[1]} channel(s); "
            f"expected at least {min_channels}"
        )
    if flat.shape[0] < count:
        raise ValueError(
            f"SOG image {filename!r} has {flat.shape[0]} pixels < declared "
            f"count {count}"
        )
    return flat[:count]


def read_sog(path: Union[str, Path]) -> ClassicalSplats:
    """Read a PlayCanvas SOG (Spatially Ordered Gaussians) bundle → ClassicalSplats.

    SOG v2 is a ``meta.json`` referencing lossless WebP images; ``path`` may be
    the bundle directory, its ``meta.json``, or a ``.sog`` ZIP. Per-Gaussian
    attributes are co-located across images (same pixel = same Gaussian):

      - ``means_l``/``means_u`` — 16-bit-per-axis position, dequantized into the
        per-axis ``[mins, maxs]`` log domain, then the symmetric log is undone
        (``sign(n)·(exp|n|−1)``).
      - ``scales`` — RGB indices into a 256-entry log-domain codebook (``exp``).
      - ``quats`` — smallest-three: three stored components in (w,x,y,z) order
        mapped to ``[−√½, +√½]``, the omitted (largest) component recovered as
        ``√(1−Σ)`` and its slot given by ``alpha − 252``.
      - ``sh0`` — RGB indices into a DC codebook (``0.5 + c·SH_C0``) + opacity in
        alpha.

    Higher-order SH (``shN``) is intentionally dropped — the DC-only policy
    shared with the other classical dialects.
    """
    import json

    path = Path(path)
    read = _sog_accessor(path)
    meta = json.loads(read("meta.json"))
    version = meta.get("version")
    if version != 2:
        raise ValueError(
            f"Unsupported SOG version {version!r} (this reader implements v2)"
        )
    count = int(meta["count"])
    if count <= 0:
        raise ValueError(f"SOG meta declares a non-positive count: {count}")

    # Positions: 16-bit per axis → per-axis log-domain lerp → undo symmetric log.
    lo = _sog_load_image(read, meta["means"]["files"][0], count, 3).astype(np.uint16)
    hi = _sog_load_image(read, meta["means"]["files"][1], count, 3).astype(np.uint16)
    q = ((hi << 8) | lo)[:, :3].astype(np.float64) / 65535.0
    mins = np.asarray(meta["means"]["mins"], dtype=np.float64)
    maxs = np.asarray(meta["means"]["maxs"], dtype=np.float64)
    n = mins + (maxs - mins) * q
    positions = (np.sign(n) * np.expm1(np.abs(n))).astype(np.float32)

    # Scales: per-channel codebook index → exp(log-sigma).
    sc = _sog_load_image(read, meta["scales"]["files"][0], count, 3)[:, :3]
    sbook = np.asarray(meta["scales"]["codebook"], dtype=np.float64)
    scales = np.exp(sbook[sc]).astype(np.float32)

    # Quaternions: smallest-three (three stored comps in w,x,y,z order; alpha
    # byte 252..255 names the omitted largest component).
    qz = _sog_load_image(read, meta["quats"]["files"][0], count, 4)
    comp = (qz[:, :3].astype(np.float64) / 255.0 - 0.5) * (2.0 / np.sqrt(2.0))
    d = np.sqrt(np.maximum(0.0, 1.0 - np.square(comp).sum(axis=1)))
    mode = qz[:, 3].astype(np.int64) - 252
    quat = np.zeros((count, 4), dtype=np.float64)  # (w, x, y, z)
    for m in range(4):
        sel = mode == m
        if not np.any(sel):
            continue
        others = [i for i in range(4) if i != m]
        for slot, tgt in enumerate(others):
            quat[sel, tgt] = comp[sel, slot]
        quat[sel, m] = d[sel]
    quaternions = _normalize_quat(quat).astype(np.float32)

    # Base color + opacity: RGB codebook indices (DC) + alpha opacity.
    s0 = _sog_load_image(read, meta["sh0"]["files"][0], count, 4)
    c0book = np.asarray(meta["sh0"]["codebook"], dtype=np.float64)
    dc = c0book[s0[:, :3]]
    colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0).astype(np.float32)
    opacities = (s0[:, 3].astype(np.float32) / 255.0).copy()

    sh_degree = int(meta.get("shN", {}).get("bands", 0))
    return ClassicalSplats(
        positions=positions,
        scales=scales,
        quaternions=quaternions,
        opacities=opacities,
        colors=colors,
        sh_degree=sh_degree,
        source_format="sog",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Format detection + dispatch
# ─────────────────────────────────────────────────────────────────────────────


def detect_classical_format(path: Union[str, Path]) -> str:
    """Detect which classical dialect ``path`` holds.

    ``.splat`` and ``.spz`` are keyed on the extension (``.spz`` additionally
    verified by the gzip magic); ``.ply`` is sniffed from the header — a
    ``chunk`` element marks the SuperSplat compressed dialect, INRIA properties
    (``f_dc_0`` / ``scale_0`` / ``rot_0``) mark the reference dialect.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    # SOG: a bundle directory (has meta.json), a meta.json file, or a .sog zip.
    if path.is_dir():
        if (path / "meta.json").is_file():
            return "sog"
        raise ValueError(f"{path.name}: directory has no meta.json — not a SOG bundle")
    if path.name == "meta.json" or suffix == ".sog":
        return "sog"
    if suffix == ".splat":
        return "splat"
    if suffix == ".spz":
        return "spz"
    if suffix == ".ply":
        with open(path, "rb") as f:
            head = f.read(64 * 1024)
        elements, _ = _parse_ply_header(head)
        by_name = {el.name: el for el in elements}
        if "chunk" in by_name:
            return "supersplat"
        vertex = by_name.get("vertex")
        prop_names = {name for name, _ in (vertex.properties if vertex else [])}
        if {"scale_0", "rot_0", "opacity"} <= prop_names:
            return "inria"
        raise ValueError(
            f"{path.name}: PLY is not a recognized Gaussian-splat dialect "
            "(no 'chunk' element and no INRIA scale_0/rot_0/opacity properties)"
        )
    raise ValueError(
        f"{path.name}: unrecognized extension {suffix!r} — expected "
        ".ply, .splat, .spz, .sog, or a SOG bundle directory"
    )


_READERS = {
    "inria": read_inria_ply,
    "splat": read_antimatter_splat,
    "spz": read_spz,
    "supersplat": read_supersplat_ply,
    "sog": read_sog,
}


# ─────────────────────────────────────────────────────────────────────────────
# Conversion to GSplatData
# ─────────────────────────────────────────────────────────────────────────────


def _orientation_matrix(rotate_x180: bool, flip: str) -> np.ndarray:
    """Build the (3, 3) orientation matrix applied to imported world coordinates.

    ``rotate_x180`` is the canonical COLMAP fix (captures store +Y down / +Z
    forward, so they appear upside-down in Y-up viewers); ``flip`` mirrors the
    named axes on top of that (a reflection — allowed here because the
    covariance is rebuilt from scratch rather than routed through
    ``GSplatData.transform``, whose diagonal path rejects negative scales).
    """
    M = np.eye(3, dtype=np.float64)
    if rotate_x180:
        M = np.diag([1.0, -1.0, -1.0]) @ M
    for axis in flip:
        try:
            idx = "xyz".index(axis.lower())
        except ValueError:
            raise ValueError(f"flip axes must be drawn from 'xyz'; got {flip!r}")
        M[idx] *= -1.0
    return M


def _robust_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Batch Cholesky with a per-splat eigenvalue-clamp fallback for non-PD input.

    Mirrors the regularization strategy of
    :func:`luxar.gsplats.utils.trils.embed_cholesky_packed`: quantized or
    degenerate source files can yield covariance matrices that are only
    positive *semi*-definite; those get their eigenvalues floored and are
    re-factorized individually.
    """
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        pass

    # Fully vectorized detect-and-repair (no per-splat Python loop, so a single
    # degenerate splat among millions doesn't drop the whole import to O(N)
    # scalar LAPACK calls). Batch eigvalsh finds the non-PD subset; only those
    # get their eigenvalues floored and recomposed via batch eigh.
    eigvals_all = np.linalg.eigvalsh(sigma)  # ascending per splat
    scale = np.maximum(np.abs(eigvals_all[:, -1]), 1e-14)
    floor = scale * 1e-9  # per-splat relative floor
    bad = eigvals_all[:, 0] < floor
    if not np.any(bad):
        # PD everywhere but the batch call still failed (rare numerical noise) —
        # symmetrize and retry once.
        sym = (sigma + np.swapaxes(sigma, -2, -1)) / 2.0
        return np.linalg.cholesky(sym)

    fixed = sigma.copy()
    eigvals, eigvecs = np.linalg.eigh(sigma[bad])
    eigvals = np.maximum(eigvals, floor[bad, None])
    fixed[bad] = eigvecs @ (eigvals[..., None] * np.swapaxes(eigvecs, -2, -1))
    return np.linalg.cholesky(fixed)


def classical_to_gsplat_data(
    cs: ClassicalSplats,
    *,
    rotate_x180: Optional[bool] = None,
    flip: str = "",
) -> "GSplatData":
    """Convert decoded classical splats to a :class:`GSplatData`.

    The covariance is rebuilt as ``Σ = (M·R) · diag(scales²) · (M·R)ᵀ`` where
    ``R`` comes from the quaternion and ``M`` is the orientation matrix
    (:func:`_orientation_matrix`), then factorized to Luxar's packed
    lower-triangular Cholesky form. Opacities become ``amplitudes``; the DC
    color becomes per-splat SDR RGB. Columns stay in world (x, y, z) order —
    that is what downstream dimension inference labels x/y/z.

    ``rotate_x180=None`` (default) applies the 180°-about-X COLMAP → Y-up fix
    exactly when the source dialect needs it (``cs.y_up`` False); SPZ declares
    RUB/Y-up data and is left untouched. Pass an explicit bool to override.

    The applied orientation and source dialect are recorded under
    ``stats["interop"]`` so an eventual export can invert them.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.utils.trils import pack_tril

    if cs.n_splats == 0:
        raise ValueError("Cannot convert an empty splat set")

    if rotate_x180 is None:
        rotate_x180 = not cs.y_up
    M = _orientation_matrix(rotate_x180, flip)

    positions = (cs.positions.astype(np.float64) @ M.T).astype(np.float32)
    R = M @ quat_to_rotmat(cs.quaternions)  # (N, 3, 3), orientation folded in
    s2 = (cs.scales.astype(np.float64) ** 2)[:, None, :]  # (N, 1, 3)
    sigma = (R * s2) @ np.swapaxes(R, -2, -1)  # R · diag(s²) · Rᵀ
    L = _robust_cholesky(sigma)
    cholesky_factors = pack_tril(L).astype(np.float32)

    amplitudes = np.ascontiguousarray(cs.opacities, dtype=np.float32)
    colors = np.clip(cs.colors, 0.0, 1.0).astype(np.float32)

    stats = {
        "interop": {
            "source_format": cs.source_format,
            "source_sh_degree": int(cs.sh_degree),
            "orientation_matrix": M.tolist(),
        }
    }
    return GSplatData(
        centers=positions,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        colors=colors,
        stats=stats,
    )


def import_gsplats(
    path: Union[str, Path],
    *,
    format: str = "auto",
    rotate_x180: Optional[bool] = None,
    flip: str = "",
) -> "GSplatData":
    """Read a classical Gaussian-splat file into a :class:`GSplatData`.

    Args:
        path: Source file (``.ply`` — INRIA or SuperSplat compressed,
            ``.splat``, ``.spz``) or a PlayCanvas SOG bundle (a directory with
            ``meta.json`` + WebPs, that ``meta.json``, or a ``.sog`` ZIP).
        format: One of ``auto`` (default, sniffed via
            :func:`detect_classical_format`) or an explicit dialect name from
            :data:`CLASSICAL_FORMATS`.
        rotate_x180: Apply the canonical COLMAP → Y-up orientation fix
            (180° rotation about X). Default ``None`` = per-dialect (on for
            the Y-down dialects INRIA/.splat/SuperSplat, off for Y-up SPZ).
        flip: Additional axes to mirror, e.g. ``"x"`` or ``"xz"``.

    Returns:
        A single-leaf :class:`GSplatData` with per-splat colors, ready for
        ``.save()``, LOD recipes, or ``Scene.add_gsplats_from_data``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Splat file not found: {path}")
    fmt = detect_classical_format(path) if format == "auto" else format
    if fmt not in _READERS:
        raise ValueError(
            f"Unknown format {fmt!r}; expected 'auto' or one of {CLASSICAL_FORMATS}"
        )
    cs = _READERS[fmt](path)
    return classical_to_gsplat_data(cs, rotate_x180=rotate_x180, flip=flip)
