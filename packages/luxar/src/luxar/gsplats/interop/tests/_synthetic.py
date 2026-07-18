"""Synthetic classical-splat file writers for interop tests.

Each writer emits a byte-exact miniature file of one dialect from a shared
ground truth, so readers can be tested for round-trip parity without shipping
binary fixtures. Written as encoders would (INRIA stores logits/log-scales/
unnormalized quats; .splat/SPZ/SuperSplat quantize), so tolerances in the
tests mirror each format's real quantization error.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GroundTruth:
    """Shared ground truth in the source (pre-orientation) world frame."""

    positions: np.ndarray  # (N, 3) float32
    scales: np.ndarray  # (N, 3) float32, linear sigmas
    quaternions: np.ndarray  # (N, 4) float32, (w, x, y, z), unit, w >= 0
    opacities: np.ndarray  # (N,) float32 in (0, 1)
    colors: np.ndarray  # (N, 3) float32 in (0, 1)


def make_ground_truth(n: int = 16, seed: int = 7) -> GroundTruth:
    """Random but quantization-friendly splats (all values well inside range)."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32)
    scales = rng.uniform(0.05, 0.8, (n, 3)).astype(np.float32)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1  # w >= 0 (canonical; required by SPZ first-three)
    opacities = rng.uniform(0.15, 0.85, n).astype(np.float32)
    colors = rng.uniform(0.1, 0.9, (n, 3)).astype(np.float32)
    return GroundTruth(
        positions=positions,
        scales=scales,
        quaternions=q.astype(np.float32),
        opacities=opacities,
        colors=colors,
    )


def write_inria_ply(path: Path, gt: GroundTruth, *, sh_degree: int = 1) -> None:
    """Write an INRIA-style point_cloud.ply (float32, logits/log/unnormalized)."""
    n = gt.positions.shape[0]
    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    props = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(n_rest)]
        + [
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )
    body = np.zeros((n, len(props)), dtype="<f4")
    body[:, 0:3] = gt.positions
    body[:, 6:9] = (gt.colors - 0.5) / SH_C0  # f_dc
    off = 9 + n_rest
    body[:, off] = np.log(gt.opacities / (1.0 - gt.opacities))  # logit
    body[:, off + 1 : off + 4] = np.log(gt.scales)
    # Store the quaternion unnormalized (scaled by 1.7) as real trainers do.
    body[:, off + 4 : off + 8] = gt.quaternions * 1.7
    path.write_bytes(header.encode("ascii") + body.tobytes())


def write_antimatter_splat(path: Path, gt: GroundTruth) -> None:
    """Write an antimatter15 .splat (32-byte records)."""
    n = gt.positions.shape[0]
    rec = np.zeros(
        n,
        dtype=np.dtype(
            [
                ("position", "<f4", 3),
                ("scale", "<f4", 3),
                ("rgba", "u1", 4),
                ("rot", "u1", 4),
            ]
        ),
    )
    rec["position"] = gt.positions
    rec["scale"] = gt.scales
    rec["rgba"][:, :3] = np.round(gt.colors * 255).astype(np.uint8)
    rec["rgba"][:, 3] = np.round(gt.opacities * 255).astype(np.uint8)
    rec["rot"] = np.clip(np.round(gt.quaternions * 128.0 + 128.0), 0, 255).astype(
        np.uint8
    )
    path.write_bytes(rec.tobytes())


def write_spz(path: Path, gt: GroundTruth, *, sh_degree: int = 1) -> None:
    """Write a legacy SPZ v2 (gzip container, first-three rotations)."""
    n = gt.positions.shape[0]
    fractional_bits = 12
    sh_dim = {0: 0, 1: 3, 2: 8, 3: 15}[sh_degree]

    header = struct.pack("<IIIBBBB", 0x5053474E, 2, n, sh_degree, fractional_bits, 0, 0)
    fixed = np.round(gt.positions.astype(np.float64) * (1 << fractional_bits))
    fixed = np.clip(fixed, -(1 << 23), (1 << 23) - 1).astype(np.int32)
    pos = np.zeros((n, 3, 3), dtype=np.uint8)
    pos[..., 0] = fixed & 0xFF
    pos[..., 1] = (fixed >> 8) & 0xFF
    pos[..., 2] = (fixed >> 16) & 0xFF

    alphas = np.round(gt.opacities * 255).astype(np.uint8)
    f_dc = (gt.colors.astype(np.float64) - 0.5) / SH_C0
    colors = np.clip(np.round((f_dc * 0.15 + 0.5) * 255), 0, 255).astype(np.uint8)
    scales = np.clip(
        np.round((np.log(gt.scales.astype(np.float64)) + 10.0) * 16.0), 0, 255
    ).astype(np.uint8)
    # v2 first-three: store (x, y, z); w >= 0 holds by ground-truth construction.
    xyz = gt.quaternions[:, 1:4].astype(np.float64)
    rots = np.clip(np.round(xyz * 127.5 + 127.5), 0, 255).astype(np.uint8)
    sh = np.full((n, sh_dim * 3), 128, dtype=np.uint8)  # zero f_rest

    stream = (
        header
        + pos.tobytes()
        + alphas.tobytes()
        + colors.tobytes()
        + scales.tobytes()
        + rots.tobytes()
        + sh.tobytes()
    )
    path.write_bytes(gzip.compress(stream))


def _pack_unorm(t: np.ndarray, bits: int) -> np.ndarray:
    return np.clip(np.round(t * ((1 << bits) - 1)), 0, (1 << bits) - 1).astype(
        np.uint32
    )


def write_supersplat_ply(
    path: Path, gt: GroundTruth, *, color_bounds: bool = True
) -> None:
    """Write a SuperSplat compressed .ply (single chunk).

    ``color_bounds=True`` (default) writes the modern 18-property chunk layout
    (per-chunk min/max RGB, colors lerped against them); ``color_bounds=False``
    writes the older 12-property layout where the 8-bit color is a raw unorm
    (no per-chunk color lerp) — the reader must accept both.
    """
    n = gt.positions.shape[0]
    if n > 256:
        raise ValueError("synthetic writer supports a single 256-splat chunk")
    log_scales = np.log(gt.scales.astype(np.float64))

    def bounds(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mn, mx = a.min(axis=0), a.max(axis=0)
        mx = np.where(mx - mn < 1e-6, mn + 1e-6, mx)  # avoid degenerate lerp
        return mn, mx

    pmin, pmax = bounds(gt.positions.astype(np.float64))
    smin, smax = bounds(log_scales)
    cmin, cmax = bounds(gt.colors.astype(np.float64))

    def norm(a: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
        return (a - mn) / (mx - mn)

    tp = norm(gt.positions.astype(np.float64), pmin, pmax)
    packed_position = (
        (_pack_unorm(tp[:, 0], 11) << 21)
        | (_pack_unorm(tp[:, 1], 10) << 11)
        | _pack_unorm(tp[:, 2], 11)
    )
    ts = norm(log_scales, smin, smax)
    packed_scale = (
        (_pack_unorm(ts[:, 0], 11) << 21)
        | (_pack_unorm(ts[:, 1], 10) << 11)
        | _pack_unorm(ts[:, 2], 11)
    )

    q = gt.quaternions.astype(np.float64)  # (w, x, y, z) = stored rot_0..3 order
    packed_rotation = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        a = q[i].copy()
        largest = int(np.argmax(np.abs(a)))
        if a[largest] < 0:
            a = -a
        word = np.uint32(largest)
        for j in range(4):
            if j != largest:
                word = (word << np.uint32(10)) | _pack_unorm(
                    np.asarray(a[j] * np.sqrt(0.5) + 0.5), 10
                )
        packed_rotation[i] = word

    # 18-prop: color lerped against per-chunk bounds; 12-prop: raw unorm color.
    tc = (
        norm(gt.colors.astype(np.float64), cmin, cmax)
        if color_bounds
        else gt.colors.astype(np.float64)
    )
    packed_color = (
        (_pack_unorm(tc[:, 0], 8) << 24)
        | (_pack_unorm(tc[:, 1], 8) << 16)
        | (_pack_unorm(tc[:, 2], 8) << 8)
        | _pack_unorm(gt.opacities.astype(np.float64), 8)
    )

    chunk_props = [
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z",
        "min_scale_x",
        "min_scale_y",
        "min_scale_z",
        "max_scale_x",
        "max_scale_y",
        "max_scale_z",
    ]
    if color_bounds:
        chunk_props += ["min_r", "min_g", "min_b", "max_r", "max_g", "max_b"]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Generated by luxar synthetic test writer\n"
        "element chunk 1\n"
        + "".join(f"property float {p}\n" for p in chunk_props)
        + f"element vertex {n}\n"
        + "".join(
            f"property uint {p}\n"
            for p in (
                "packed_position",
                "packed_rotation",
                "packed_scale",
                "packed_color",
            )
        )
        + "end_header\n"
    )
    chunk_arrays = [pmin, pmax, smin, smax]
    if color_bounds:
        chunk_arrays += [cmin, cmax]
    chunk = np.concatenate(chunk_arrays).astype("<f4")
    vertex = np.stack(
        [packed_position, packed_rotation, packed_scale, packed_color], axis=1
    ).astype("<u4")
    path.write_bytes(header.encode("ascii") + chunk.tobytes() + vertex.tobytes())


def _codebook_encode(values: np.ndarray, n_entries: int = 256):
    """Build an ``n_entries`` linspace codebook over ``values`` and index into it.

    Returns ``(codebook, indices)`` — the SOG scheme for scales/sh0. A linspace
    codebook is not what the real (k-means) encoder produces, but it round-trips
    within the 8-bit resolution the reader must tolerate.
    """
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    codebook = np.linspace(lo, hi, n_entries)
    idx = np.clip(np.round((values - lo) / (hi - lo) * (n_entries - 1)), 0, n_entries - 1)
    return codebook.tolist(), idx.astype(np.uint8)


def write_sog(directory: Path, gt: GroundTruth) -> None:
    """Write a synthetic PlayCanvas SOG v2 bundle (meta.json + lossless WebPs).

    Encodes exactly the four decoded groups (means/scales/quats/sh0); no shN.
    Uses a square-ish row-major image layout with tail padding, matching the
    reader's ``first-count-pixels`` contract.
    """
    import io
    import json

    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    n = gt.positions.shape[0]
    w = int(np.ceil(np.sqrt(n)))
    h = int(np.ceil(n / w))

    def save(name: str, flat: np.ndarray) -> None:  # flat: (n, C) uint8
        c = flat.shape[1]
        full = np.zeros((w * h, c), dtype=np.uint8)
        full[:n] = flat
        img = full.reshape(h, w, c)
        mode = {3: "RGB", 4: "RGBA"}[c]
        buf = io.BytesIO()
        Image.fromarray(img, mode=mode).save(buf, format="WEBP", lossless=True)
        (directory / name).write_bytes(buf.getvalue())

    # Means: symmetric log → per-axis [min,max] → 16-bit split low/high.
    log = np.sign(gt.positions) * np.log1p(np.abs(gt.positions.astype(np.float64)))
    mins = log.min(axis=0)
    maxs = np.where(log.max(axis=0) - mins < 1e-9, mins + 1e-9, log.max(axis=0))
    q16 = np.round((log - mins) / (maxs - mins) * 65535).astype(np.uint16)  # (n,3)
    save("means_l.webp", (q16 & 0xFF).astype(np.uint8))
    save("means_u.webp", (q16 >> 8).astype(np.uint8))

    # Scales: log-domain codebook.
    sbook, sidx = _codebook_encode(np.log(gt.scales.astype(np.float64)).ravel())
    save("scales.webp", sidx.reshape(n, 3))

    # Quats: smallest-three (store the three non-largest in w,x,y,z order).
    qd = gt.quaternions.astype(np.float64)
    quats_rgba = np.zeros((n, 4), dtype=np.uint8)
    for i in range(n):
        a = qd[i].copy()
        largest = int(np.argmax(np.abs(a)))
        if a[largest] < 0:
            a = -a
        others = [j for j in range(4) if j != largest]
        for slot, j in enumerate(others):
            c = (a[j] * (np.sqrt(2.0) / 2.0) + 0.5) * 255.0
            quats_rgba[i, slot] = int(np.clip(round(c), 0, 255))
        quats_rgba[i, 3] = 252 + largest
    save("quats.webp", quats_rgba)

    # sh0: DC codebook + opacity in alpha.
    dc = (gt.colors.astype(np.float64) - 0.5) / SH_C0
    cbook, cidx = _codebook_encode(dc.ravel())
    sh0 = np.zeros((n, 4), dtype=np.uint8)
    sh0[:, :3] = cidx.reshape(n, 3)
    sh0[:, 3] = np.clip(np.round(gt.opacities.astype(np.float64) * 255), 0, 255)
    save("sh0.webp", sh0)

    meta = {
        "version": 2,
        "count": n,
        "means": {"mins": mins.tolist(), "maxs": maxs.tolist(),
                  "files": ["means_l.webp", "means_u.webp"]},
        "scales": {"codebook": sbook, "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": cbook, "files": ["sh0.webp"]},
    }
    (directory / "meta.json").write_text(json.dumps(meta))


WRITERS = {
    "inria": write_inria_ply,
    "splat": write_antimatter_splat,
    "spz": write_spz,
    "supersplat": write_supersplat_ply,
    "sog": write_sog,
}

SUFFIXES = {
    "inria": ".ply",
    "splat": ".splat",
    "spz": ".spz",
    "supersplat": ".ply",
    # SOG is a bundle DIRECTORY (meta.json + WebPs), not a single file — the
    # empty suffix makes `_write_fixture` hand write_sog a directory path.
    "sog": "",
}
