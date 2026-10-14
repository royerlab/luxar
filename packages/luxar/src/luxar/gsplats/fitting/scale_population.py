"""Carry voxel-space scale classifications through a fit's closing reductions."""

from __future__ import annotations

import base64
import zlib
from typing import Any, Sequence

import numpy as np

MASK_KEY = "_fit_scale_population_bits"
_CANDIDATES = (
    ("fit_init_sigma", "fit_init_sigma_diag_vox", 1),
    ("relocation_init_sigma", "relocation_init_sigma_vox", 2),
    ("sigma_min", "sigma_min_diag_vox", 4),
)
_RELOCATED = 8


def encode_population_bits(bits: np.ndarray) -> str:
    """Compact one byte per splat for intermediate tile-store metadata."""
    return base64.b64encode(zlib.compress(bits.tobytes())).decode("ascii")


def decode_population_bits(value: str, n_splats: int) -> np.ndarray:
    bits = np.frombuffer(zlib.decompress(base64.b64decode(value)), dtype=np.uint8)
    if len(bits) != n_splats:
        raise ValueError("Scale-population mask length does not match splat count")
    return bits


def classify_scales(
    Ls: np.ndarray, stats: dict[str, Any], relocated: np.ndarray | None = None
) -> np.ndarray:
    """Classify optimizer-space scales before output transforms alter them."""
    marginal = np.sqrt(np.einsum("nij,nij->ni", Ls, Ls, optimize=True))
    bits = np.zeros(len(Ls), dtype=np.uint8)
    tolerance = float(stats["scale_diagnostic_tolerance_vox"])
    for _, target_key, flag in _CANDIDATES:
        target = stats.get(target_key)
        if target is not None:
            near = np.all(np.abs(marginal - np.asarray(target)) <= tolerance, axis=1)
            bits[near] |= flag
    if relocated is not None:
        if relocated.shape != (len(bits),):
            raise ValueError("Relocation mask length does not match splat count")
        bits[np.asarray(relocated, dtype=bool)] |= _RELOCATED
    return bits


def stamp_population_stats(stats: dict[str, Any], bits: np.ndarray) -> None:
    """Stamp counts against the exact splat set represented by ``bits``."""
    n = len(bits)
    for name, target_key, flag in _CANDIDATES:
        if target_key not in stats:
            continue
        count_key = f"splats_near_{name}_count"
        fraction_key = f"splats_near_{name}_fraction"
        if stats[target_key] is None:
            stats[count_key] = None
            stats[fraction_key] = None
            continue
        near = (bits & flag) != 0
        count = int(np.count_nonzero(near))
        stats[count_key] = count
        stats[fraction_key] = count / n if n else 0.0
        if name == "relocation_init_sigma" and (
            "splats_near_relocation_init_sigma_relocated_count" in stats
            or np.any(bits & _RELOCATED)
        ):
            relocated_count = int(np.count_nonzero(near & ((bits & _RELOCATED) != 0)))
            stats[f"splats_near_{name}_relocated_count"] = relocated_count
            stats[f"splats_near_{name}_unrelocated_count"] = count - relocated_count


def reduce_population_stats(
    source: dict[str, Any], target: dict[str, Any], keep: np.ndarray
) -> None:
    """Reduce the source's original-scale classification by delivered indices."""
    encoded = source.get(MASK_KEY)
    if encoded is None:
        return
    bits = decode_population_bits(encoded, len(keep))[keep]
    for _, target_key, _ in _CANDIDATES:
        if target_key in source:
            target[target_key] = source[target_key]
    target[MASK_KEY] = encode_population_bits(bits)
    stamp_population_stats(target, bits)


def merge_population_stats(
    sources: Sequence[dict[str, Any]], target: dict[str, Any]
) -> None:
    """Sum local classifications for a flat or partition fit root."""
    for key in ("dynamic_ops_relocation_events", "dynamic_ops_unique_splats_relocated"):
        if sources and all(source.get(key) is not None for source in sources):
            target[key] = sum(int(source[key]) for source in sources)
    nonempty = [source for source in sources if int(source.get("n_splats", 0)) > 0]
    if not nonempty or any(MASK_KEY not in source for source in nonempty):
        return
    bits = np.concatenate(
        [decode_population_bits(s[MASK_KEY], int(s["n_splats"])) for s in nonempty]
    )
    for _, target_key, _ in _CANDIDATES:
        values = [source.get(target_key) for source in nonempty]
        if values and all(value == values[0] for value in values):
            target[target_key] = values[0]
    target[MASK_KEY] = encode_population_bits(bits)
    stamp_population_stats(target, bits)


def sum_population_counts(
    sources: Sequence[dict[str, Any]], target: dict[str, Any], n_splats: int
) -> None:
    """Aggregate already-reduced parts without restoring stale pre-cull masks."""
    if not sources:
        return
    for name, target_key, _ in _CANDIDATES:
        count_key = f"splats_near_{name}_count"
        counts = [source.get(count_key) for source in sources]
        if any(count is None for count in counts):
            continue
        count = sum(int(value) for value in counts)
        target[count_key] = count
        target[f"splats_near_{name}_fraction"] = count / n_splats if n_splats else 0.0
        values = [source.get(target_key) for source in sources]
        if all(value == values[0] for value in values):
            target[target_key] = values[0]
        if name == "relocation_init_sigma":
            for suffix in ("relocated_count", "unrelocated_count"):
                key = f"splats_near_{name}_{suffix}"
                if all(source.get(key) is not None for source in sources):
                    target[key] = sum(int(source[key]) for source in sources)
    for key in ("dynamic_ops_relocation_events", "dynamic_ops_unique_splats_relocated"):
        if all(source.get(key) is not None for source in sources):
            target[key] = sum(int(source[key]) for source in sources)


def finish_population_stats(stats: dict[str, Any]) -> None:
    """Remove the fitting-only per-splat bits from a delivered artifact."""
    stats.pop(MASK_KEY, None)
