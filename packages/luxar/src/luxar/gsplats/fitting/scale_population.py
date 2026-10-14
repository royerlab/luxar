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
_EFFECTIVE_RELOCATION_KEY = "relocation_reseed_sigma_vox_effective"


def _target(stats: dict[str, Any], key: str) -> Any:
    if key == "relocation_init_sigma_vox":
        return stats.get(_EFFECTIVE_RELOCATION_KEY, stats.get(key))
    return stats.get(key)


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
        target = _target(stats, target_key)
        if target is not None:
            near = np.all(np.abs(marginal - np.asarray(target)) <= tolerance, axis=1)
            bits[near] |= flag
    if relocated is not None and relocated.shape == (len(bits),):
        bits[np.asarray(relocated, dtype=bool)] |= _RELOCATED
    return bits


def stamp_population_stats(stats: dict[str, Any], bits: np.ndarray) -> None:
    """Stamp counts against the exact splat set represented by ``bits``."""
    n = len(bits)
    for name, target_key, flag in _CANDIDATES:
        count_key = f"splats_near_{name}_count"
        if (
            target_key not in stats
            and count_key not in stats
            and not (
                name == "relocation_init_sigma" and _EFFECTIVE_RELOCATION_KEY in stats
            )
        ):
            continue
        fraction_key = f"splats_near_{name}_fraction"
        if target_key in stats and _target(stats, target_key) is None:
            stats[count_key] = None
            stats[fraction_key] = None
            continue
        near = (bits & flag) != 0
        count = int(np.count_nonzero(near))
        stats[count_key] = count
        stats[fraction_key] = count / n if n else 0.0
        if name == "relocation_init_sigma" and (
            stats.get("splats_near_relocation_init_sigma_relocated_count") is not None
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
    if _EFFECTIVE_RELOCATION_KEY in source:
        target[_EFFECTIVE_RELOCATION_KEY] = source[_EFFECTIVE_RELOCATION_KEY]
    split_key = "splats_near_relocation_init_sigma_relocated_count"
    if source.get(split_key) is not None:
        target[split_key] = 0
    elif split_key in source:
        target[split_key] = None
        target["splats_near_relocation_init_sigma_unrelocated_count"] = None
    target[MASK_KEY] = encode_population_bits(bits)
    stamp_population_stats(target, bits)


def _merge_candidate_stamps(
    nonempty: Sequence[dict[str, Any]], target: dict[str, Any]
) -> None:
    for _, target_key, _ in _CANDIDATES:
        values = [source.get(target_key) for source in nonempty]
        if values and all(value == values[0] for value in values):
            target[target_key] = values[0]
    for name, _, _ in _CANDIDATES:
        count_key = f"splats_near_{name}_count"
        if all(source.get(count_key) is not None for source in nonempty):
            target[count_key] = 0
    effective = [source.get(_EFFECTIVE_RELOCATION_KEY) for source in nonempty]
    if (
        effective
        and effective[0] is not None
        and all(value == effective[0] for value in effective)
    ):
        target[_EFFECTIVE_RELOCATION_KEY] = effective[0]
    if all(
        s.get("splats_near_relocation_init_sigma_relocated_count") is not None
        for s in nonempty
    ):
        target["splats_near_relocation_init_sigma_relocated_count"] = 0


def merge_population_stats(
    sources: Sequence[dict[str, Any]], target: dict[str, Any]
) -> None:
    """Sum local classifications for a flat or partition fit root."""
    sum_run_history(sources, target)
    nonempty = [source for source in sources if int(source.get("n_splats", 0)) > 0]
    if not nonempty or any(MASK_KEY not in source for source in nonempty):
        return
    bits = np.concatenate(
        [decode_population_bits(s[MASK_KEY], int(s["n_splats"])) for s in nonempty]
    )
    _merge_candidate_stamps(nonempty, target)
    target[MASK_KEY] = encode_population_bits(bits)
    stamp_population_stats(target, bits)
    split_key = "splats_near_relocation_init_sigma_relocated_count"
    if any(split_key in source for source in nonempty) and not all(
        source.get(split_key) is not None for source in nonempty
    ):
        target[split_key] = None
        target["splats_near_relocation_init_sigma_unrelocated_count"] = None


def _sum_present(sources: Sequence[dict[str, Any]], key: str) -> int | None:
    values = [source.get(key) for source in sources]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values if value is not None)


def sum_run_history(sources: Sequence[dict[str, Any]], target: dict[str, Any]) -> None:
    """Retain optimizer event totals even when a closing cull removes a part."""
    for key in ("dynamic_ops_relocation_events", "dynamic_ops_unique_splats_relocated"):
        relevant = [
            source
            for source in sources
            if int(source.get("n_splats", 0)) > 0 or key in source
        ]
        count = _sum_present(relevant, key) if relevant else None
        if count is not None:
            target[key] = count


def _sum_candidate(
    sources: Sequence[dict[str, Any]],
    target: dict[str, Any],
    n_splats: int,
    name: str,
    target_key: str,
) -> None:
    count_key = f"splats_near_{name}_count"
    count = _sum_present(sources, count_key)
    if count is None:
        if all(count_key in source for source in sources):
            target[count_key] = None
            target[f"splats_near_{name}_fraction"] = None
        return
    target[count_key] = count
    target[f"splats_near_{name}_fraction"] = count / n_splats if n_splats else 0.0
    values = [source.get(target_key) for source in sources]
    if all(value == values[0] for value in values):
        target[target_key] = values[0]
    if name == "relocation_init_sigma":
        effective = [source.get(_EFFECTIVE_RELOCATION_KEY) for source in sources]
        if effective[0] is not None and all(
            value == effective[0] for value in effective
        ):
            target[_EFFECTIVE_RELOCATION_KEY] = effective[0]
    if name == "relocation_init_sigma":
        for suffix in ("relocated_count", "unrelocated_count"):
            key = f"splats_near_{name}_{suffix}"
            split_count = _sum_present(sources, key)
            if split_count is not None:
                target[key] = split_count
            elif all(key in source for source in sources):
                target[key] = None


def sum_population_counts(
    sources: Sequence[dict[str, Any]], target: dict[str, Any], n_splats: int
) -> None:
    """Aggregate already-reduced parts without restoring stale pre-cull masks."""
    if not sources:
        return
    sum_run_history(sources, target)
    if (
        all("n_splats" in source for source in sources)
        and sum(int(source["n_splats"]) for source in sources) != n_splats
    ):
        return
    for name, target_key, _ in _CANDIDATES:
        _sum_candidate(sources, target, n_splats, name, target_key)


def finish_population_stats(stats: dict[str, Any]) -> None:
    """Remove the fitting-only per-splat bits from a delivered artifact."""
    stats.pop(MASK_KEY, None)
