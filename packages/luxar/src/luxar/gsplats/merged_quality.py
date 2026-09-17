"""Whole-volume quality scoring shared by merged gsplat fit paths."""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

import numpy as np
from arbol import aprint

from luxar.gsplats.fit_basis import MISSING_BASIS_HINT, reference_on_fit_basis
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import _FITTING_INFO_KEYS
from luxar.typing_utils.json_safe import json_safe_value

_FIT_REFERENCE_KINDS = frozenset(("acquisition", "preprocessed", "synthetic"))

#: Upper bound, in GiB, on the memory a merged-quality score may hold resident.
#: Above it the score is SKIPPED — and says so out loud, because an archive that
#: silently carries no PSNR is the failure this scoring exists to end. Override
#: with ``LUXAR_TILED_QUALITY_MAX_GB`` when the machine can take more, or set it
#: to ``0`` to decline scoring outright. This is a CEILING, not the budget: the
#: default is additionally held under a share of the host memory or CUDA memory
#: actually free, since a fixed number describes whichever machine it was
#: written on and not the one running the fit.
_QUALITY_BUDGET_GB = 24.0

#: Share of currently-allocatable host memory or free CUDA memory the default
#: budget will commit to a score. Deliberately well under 1: the peaks below are
#: estimates,
#: the fit process is holding the merged splats too, and being wrong in this
#: direction costs a metric while being wrong in the other costs the whole fit.
_QUALITY_BUDGET_MEM_FRACTION = 0.5

#: Host-side float32 copies held before the reference reaches the render device:
#: ``np.asarray`` materializes a lazy source, then a non-zero ``image_min`` makes
#: ``reference_on_fit_basis`` allocate the shifted/clipped array.
_QUALITY_HOST_REFERENCE_VOLUMES = 2

#: Full-size float32 volumes estimated at the scoring-device peak. This retains
#: the existing conservative count until the compiled CUDA renderer is measured
#: independently; CUDA SSIM already tiles itself from free VRAM, so this is now
#: a device-side render/metric allowance rather than a host-memory proxy.
_QUALITY_DEVICE_PEAK_VOLUMES = 8

#: Local ``batch-fit run`` workers share one CUDA device. Its parent records that
#: concurrency here so each process admits only its share of device VRAM.
QUALITY_WORKERS_PER_DEVICE_ENV = "LUXAR_QUALITY_WORKERS_PER_DEVICE"

#: Concurrent fit workers across every device share the host's RAM. Local and
#: packed Slurm launchers set this count before each worker starts.
QUALITY_WORKERS_PER_HOST_ENV = "LUXAR_QUALITY_WORKERS_PER_HOST"


def _available_ram_gb() -> "float | None":
    """Allocatable host memory in GiB, or ``None`` where it cannot be measured."""
    from luxar.gsplats.utils.device import available_host_memory_bytes

    available = available_host_memory_bytes()
    if available is None:
        return None
    return available / 1024**3


def _default_quality_budget_gb() -> float:
    """The default budget: the ceiling, held under a share of free memory.

    The ceiling alone is a number about some other machine. Scoring materializes
    the whole volume — during the fit it is only ever read tile by tile — so on a
    host smaller than the ceiling the guard would wave through a peak the machine
    cannot hold, and the OOM kill lands BEFORE the archive is written, losing the
    finished fit. That is the one outcome this budget exists to prevent, so the
    default is the smaller of the two. An explicit override still wins outright:
    the operator knows what the machine can take.
    """
    available = _available_ram_gb()
    if available is None:  # pragma: no cover - platform
        return _QUALITY_BUDGET_GB
    return min(_QUALITY_BUDGET_GB, _QUALITY_BUDGET_MEM_FRACTION * available)


def _quality_budget_gb() -> float:
    """Resident-memory budget for scoring — the env override, or the default.

    A malformed override falls back to the default with a note rather than
    raising: this runs after every tile has been fitted, so an unparseable
    environment variable must not be what loses a finished fit.
    """
    default = _default_quality_budget_gb()
    raw = os.environ.get("LUXAR_TILED_QUALITY_MAX_GB")
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    # NaN is the one malformed value that would DISABLE the guard instead of
    # tripping it: `float("nan")` parses, and every comparison against it is
    # False, so the over-budget test would silently pass whatever the volume
    # size. Treated like any other unusable override. `inf` is left alone — it
    # is a coherent way to say "score it no matter how big".
    if math.isnan(value):
        aprint(
            f"⚠️  Ignoring LUXAR_TILED_QUALITY_MAX_GB={raw!r} (not a usable "
            f"number) — using the {default:g} GiB budget"
        )
        return default
    return value


def _quality_worker_count(env_name: str, *, default: int = 1) -> int:
    """Concurrent scoring workers sharing the named memory resource."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _has_usable_quality_override() -> bool:
    raw = os.environ.get("LUXAR_TILED_QUALITY_MAX_GB")
    if raw is None:
        return False
    try:
        return not math.isnan(float(raw))
    except ValueError:
        return False


def _quality_memory_guard(
    volume_shape: tuple[int, ...], device: Optional[str]
) -> "str | None":
    """Return the resource that cannot safely admit scoring, if any."""
    voxel_gb = 4 * float(np.prod(volume_shape)) / 1024**3

    from luxar.gsplats.utils.device import resolve_torch_device

    try:
        resolved = resolve_torch_device(device)
    except Exception:
        # Preserve the scoring path's existing failure contract: device errors
        # are reported by the guarded render attempt, never raised after fitting.
        return None

    device_workers = _quality_worker_count(QUALITY_WORKERS_PER_DEVICE_ENV)
    host_workers = _quality_worker_count(
        QUALITY_WORKERS_PER_HOST_ENV, default=device_workers
    )
    has_override = _has_usable_quality_override()
    # CPU renders and MPS unified device memory both consume host RAM at peak.
    peak_gb = (
        _QUALITY_HOST_REFERENCE_VOLUMES
        if resolved.type == "cuda"
        else _QUALITY_DEVICE_PEAK_VOLUMES
    ) * voxel_gb
    host_budget_gb = _quality_budget_gb()
    if has_override:
        host_budget_description = (
            f"{host_budget_gb:g} GiB LUXAR_TILED_QUALITY_MAX_GB cap"
        )
    else:
        host_budget_gb /= host_workers
        host_budget_description = (
            f"{host_budget_gb:g} GiB per-worker budget "
            f"({host_workers} worker(s) sharing the host)"
        )
    if peak_gb > host_budget_gb:
        return (
            f"needs ~{peak_gb:.1f} GiB of host memory, over the "
            f"{host_budget_description}"
        )

    if resolved.type == "cuda":
        from luxar.gsplats.metrics import _gpu_free_memory

        free_bytes = _gpu_free_memory(resolved)
        if has_override:
            device_budget_gb = host_budget_gb
            budget_description = (
                f"{device_budget_gb:g} GiB LUXAR_TILED_QUALITY_MAX_GB cap"
            )
        elif free_bytes is None:
            device_budget_gb = _QUALITY_BUDGET_GB / device_workers
            budget_description = (
                f"{device_budget_gb:g} GiB per-worker budget "
                f"({device_workers} worker(s) sharing the device)"
            )
        else:
            device_budget_gb = min(
                _QUALITY_BUDGET_GB,
                _QUALITY_BUDGET_MEM_FRACTION * free_bytes / device_workers / 1024**3,
            )
            budget_description = (
                f"{device_budget_gb:g} GiB per-worker budget "
                f"({device_workers} worker(s) sharing the device)"
            )
        device_peak_gb = _QUALITY_DEVICE_PEAK_VOLUMES * voxel_gb
        if device_peak_gb > device_budget_gb:
            return (
                f"needs ~{device_peak_gb:.1f} GiB of {resolved} memory, over "
                f"the {budget_description}"
            )
    return None


#: What to do about an archive this module could not score. One wording for
#: every shape: ``luxar gsplat compare`` reads the written archive whatever tree
#: it is, ``kind=partition`` included (#1978), so nothing needs flattening first.
_COMPARE_RECOURSE = "Run `luxar gsplat compare` on the written archive instead."


def _validated_fit_reference(
    fit_reference: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if fit_reference is None:
        return None
    kind = fit_reference.get("kind")
    if kind not in _FIT_REFERENCE_KINDS:
        allowed = ", ".join(sorted(_FIT_REFERENCE_KINDS))
        raise ValueError(f"fit_reference.kind must be one of: {allowed}")
    note = fit_reference.get("note")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise ValueError("fit_reference.note must be a non-empty string when provided")
    safe_reference = {"kind": kind}
    if note is not None:
        safe_reference["note"] = note
    return safe_reference


def _json_safe_fitting(stats: dict[str, Any]) -> dict[str, Any]:
    fitting: dict[str, Any] = {}
    for key in _FITTING_INFO_KEYS:
        if key not in stats:
            continue
        ok, safe_value = json_safe_value(stats[key])
        if ok:
            fitting[key] = safe_value
    return fitting


def collect_part_provenance(
    datasets: Sequence[GSplatData],
    *,
    values: Sequence[float],
    fit_reference: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect JSON-safe per-fit stamps for caller-defined component coordinates.

    The caller supplies the reference classification when it knows what each fit
    was scored against; ``None`` records the contract's unknown-reference case.
    The returned records describe the component fits; they are not a scalar
    quality claim for the transformed union. Existing component provenance is
    retained recursively when composed datasets are collected again.
    """
    if len(values) != len(datasets):
        raise ValueError(
            f"Number of values ({len(values)}) must match number of datasets "
            f"({len(datasets)})"
        )
    safe_reference = _validated_fit_reference(fit_reference)

    records: list[dict[str, Any]] = []
    for coordinate, dataset in zip(values, datasets):
        ok, safe_coordinate = json_safe_value(coordinate)
        if not ok or not isinstance(safe_coordinate, (int, float)):
            raise ValueError(f"stack coordinate {coordinate!r} is not a finite number")
        record = {
            "coordinate": safe_coordinate,
            "fitting": _json_safe_fitting(dataset.stats),
        }
        if safe_reference is not None:
            record["fit_reference"] = dict(safe_reference)
        records.append(record)
    return records


def _summary_fitting(record: Any) -> dict[str, Any]:
    """Resolve source/timing fields from a possibly nested component record."""
    if not isinstance(record, dict):
        return {}
    fitting = record.get("fitting")
    if not isinstance(fitting, dict):
        return {}

    nested = summarize_part_provenance(fitting.get("part_provenance"))
    nested_fitting = nested[0]["fitting"] if nested is not None else {}
    resolved: dict[str, Any] = {}
    for key in (
        "source_bytes",
        "source_voxels",
        "time_seconds",
        "source_shape",
        "source_dtype",
    ):
        if key in fitting:
            resolved[key] = fitting[key]
        elif key in nested_fitting:
            resolved[key] = nested_fitting[key]
    return resolved


def summarize_part_provenance(
    value: Any, *, shared_source: bool = False
) -> Optional[list[dict[str, Any]]]:
    """Collapse component records into one coordinate-free source summary.

    ``shared_source`` is for spatial partition records whose repeated source
    sizes describe the same parent volume. Independent merge inputs use the
    default additive policy.
    """
    if not isinstance(value, list) or not value:
        return None
    fittings = [_summary_fitting(record) for record in value]

    summary: dict[str, Any] = {}
    for key in ("source_bytes", "source_voxels", "time_seconds"):
        values = [fitting.get(key) for fitting in fittings]
        if all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in values
        ):
            if (
                shared_source
                and key != "time_seconds"
                and all(item == values[0] for item in values[1:])
            ):
                summary[key] = values[0]
            else:
                summary[key] = sum(values)

    for key in ("source_shape", "source_dtype"):
        values = [fitting.get(key) for fitting in fittings]
        if values[0] is not None and all(item == values[0] for item in values[1:]):
            summary[key] = values[0]

    return [{"part_count": len(value), "fitting": summary}]


def announce_unscored_merge(reason: str) -> None:
    """Explain why a merged result carries no whole-volume quality metrics.

    Parameters
    ----------
    reason : str
        The reason scoring was unavailable.
    """
    aprint(f"No merged quality metrics: {reason}. {_COMPARE_RECOURSE}")


def resolve_merged_reference(
    volume: "Any | None",
    expected_shape: tuple[int, ...],
    *,
    grid_name: str,
    missing_reason: str,
) -> "tuple[Any | None, str | None]":
    """Validate that a merged-quality reference matches its fitting grid."""
    if volume is None:
        return None, missing_reason

    shape = getattr(volume, "shape", None)
    if shape is None:
        return None, "the supplied reference volume does not expose a shape"

    reference_shape = tuple(int(size) for size in shape)
    if reference_shape != expected_shape:
        return (
            None,
            f"reference shape {reference_shape} does not match the {grid_name} "
            f"{expected_shape}",
        )

    return volume, None


def _to_voxel_frame(merged: GSplatData, scale: Optional[Sequence[float]]) -> GSplatData:
    """The same mixture expressed on the tile grid's own voxel frame.

    A real-space tiled fit emits physical coordinates, so the merged splats do
    not sit on ``volume_shape``'s grid and cannot be rendered against it. The
    frames differ by one per-axis factor (see :func:`resolve_grid_scale`), which
    scales centers directly and Cholesky ROW ``i`` by ``scale[i]`` — so dividing
    both undoes it exactly. Amplitudes are untouched by the conversion.
    """
    if scale is None:
        return merged
    voxel_scale = np.asarray(scale, dtype=np.float64)
    dimensions = merged.centers.shape[1] if merged.n_splats else len(voxel_scale)
    tril_scales = np.concatenate(
        [[voxel_scale[axis]] * (axis + 1) for axis in range(dimensions)]
    )
    return GSplatData(
        centers=(merged.centers / voxel_scale).astype(np.float32),
        amplitudes=merged.amplitudes,
        cholesky_factors=(merged.cholesky_factors / tril_scales).astype(np.float32),
        truncation_radius=merged.truncation_radius,
    )


def _announce_missing_basis(image_min: Optional[float]) -> None:
    if image_min is None:
        aprint(f"WARNING: {MISSING_BASIS_HINT}")


def stamp_merged_quality(
    merged: "GSplatData | Sequence[GSplatData]",
    volume: Any,
    *,
    volume_shape: tuple[int, ...],
    grid_scale: Optional[Sequence[float]],
    device: Optional[str],
    verbose: bool,
    image_min: Optional[float],
    stats: "dict[str, Any] | None" = None,
) -> None:
    """Score the MERGED reconstruction against the whole volume, in place.

    Each tile already scores itself, but those numbers are about crops of an
    apodized decomposition: the tiles overlap, so their errors do not compose
    into the merged one, and none of them can speak for the archive that
    actually ships. Without this a tiled archive carries no PSNR at all — which
    is exactly what a published dataset is asked for.

    The reference is shifted onto the merged fit's background-relative basis
    using the explicitly resolved ``image_min``. The tiles reconstruct
    ``V - image_min``, not the raw acquisition, so leaving the pedestal in the
    reference would make tiled and non-tiled fits publish different metrics for
    the same signal (#1173).
    Under ``--denoise`` that parity ends, and not in this path's favor: the tiles
    reconstruct denoised data while the reference here keeps its noise, so the
    score is capped by that noise, whereas ``--tiling none`` denoises the whole
    volume up front and scores against its own smoothed copy. Neither number is
    wrong, but they are not the same measurement — a gap between them under
    ``--denoise`` is not a tiling artifact. A lazy source is materialized here — during the fit it
    is only ever read tile-by-tile — which is what the budget below bounds.
    """
    if stats is not None:
        parts = [merged] if isinstance(merged, GSplatData) else list(merged)
        target_stats = stats
    elif isinstance(merged, GSplatData):
        parts = [merged]
        target_stats = merged.stats
    else:
        announce_unscored_merge(
            "partition merged-quality scoring was not given a stats target"
        )
        return
    if not parts or sum(part.n_splats for part in parts) == 0:
        return
    _announce_missing_basis(image_min)
    over_budget = _quality_memory_guard(volume_shape, device)
    if over_budget is not None:
        aprint(
            f"Merged quality metrics skipped: scoring {volume_shape} {over_budget}. Raise "
            f"LUXAR_TILED_QUALITY_MAX_GB to score it anyway. {_COMPARE_RECOURSE}"
        )
        return

    try:
        import torch

        from luxar.gsplats.metrics import compute_quality_metrics
        from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

        rendered: Any = None
        reference: Any = None
        try:
            with torch.no_grad():
                for part in parts:
                    scored = _to_voxel_frame(part, grid_scale)
                    part_render = render_to_volume_tensor(
                        scored,
                        shape=volume_shape,
                        device=device,
                        truncate=scored.truncation_radius,
                    )
                    if rendered is None:
                        rendered = part_render
                    else:
                        rendered.add_(part_render)
                        del part_render
                reference_np = reference_on_fit_basis(
                    np.asarray(volume, dtype=np.float32), image_min
                )
                reference = torch.as_tensor(reference_np, device=rendered.device)
                quality = compute_quality_metrics(rendered, reference)
        finally:
            # Released whether or not the score succeeded: the failure this most
            # often takes is an OOM inside SSIM, and leaving the peak reserved
            # would carry it into whatever the caller does next (a `--recipe`
            # reduction runs on the same device seconds later).
            del rendered, reference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        target_stats.update(
            {
                "mse": quality["mse"],
                "psnr_db": quality["psnr_db"],
                "ssim": quality["ssim"],
                "foreground_psnr_db": quality["foreground_psnr_db"],
                "foreground_threshold": quality["foreground_threshold"],
                "foreground_fraction": quality["foreground_fraction"],
            }
        )
        if verbose:
            aprint(
                f"Merged quality: PSNR={quality['psnr_db']:.1f} dB, "
                f"foreground PSNR={quality['foreground_psnr_db']:.1f} dB "
                f"(over {quality['foreground_fraction'] * 100:.2f}% of voxels), "
                f"SSIM={quality['ssim']:.4f}"
            )
    except Exception as exc:  # pragma: no cover - device/memory dependent
        aprint(f"⚠️  Merged quality metrics failed ({exc}) — archive carries no PSNR")


__all__ = [
    "announce_unscored_merge",
    "collect_part_provenance",
    "resolve_merged_reference",
    "stamp_merged_quality",
    "summarize_part_provenance",
]
