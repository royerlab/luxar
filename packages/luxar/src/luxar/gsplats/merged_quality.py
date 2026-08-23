"""Whole-volume quality scoring shared by merged gsplat fit paths."""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

import numpy as np
from arbol import aprint

from luxar.gsplats.fit_basis import fit_image_min, reference_on_fit_basis
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.tree import GSplatNode, GSplatPartition

#: Upper bound, in GiB, on the memory a merged-quality score may hold resident.
#: Above it the score is SKIPPED — and says so out loud, because an archive that
#: silently carries no PSNR is the failure this scoring exists to end. Override
#: with ``LUXAR_TILED_QUALITY_MAX_GB`` when the machine can take more, or set it
#: to ``0`` to decline scoring outright. This is a CEILING, not the budget: the
#: default is additionally held under a share of the memory actually free (see
#: :func:`_default_quality_budget_gb`), since a fixed number describes whichever
#: machine it was written on and not the one running the fit.
_QUALITY_BUDGET_GB = 24.0

#: Share of currently-free physical memory the default budget will commit to a
#: score. Deliberately well under 1: the peak below is an estimate, the fit
#: process is holding the merged splats too, and being wrong in this direction
#: costs a metric while being wrong in the other costs the whole fit.
_QUALITY_BUDGET_MEM_FRACTION = 0.5

#: Full-size float32 volumes live at the scoring peak, which sits inside SSIM
#: rather than at the render: the reconstruction and the reference, plus the
#: convolution intermediates :func:`luxar.gsplats.metrics._ssim_nd` keeps live
#: (``_SSIM_PEAK_TENSOR_COUNT``, the same count that function's own tiled
#: fallback is sized by — and that fallback only engages on CUDA, so on CPU this
#: is the true peak). Counting only the reconstruction and the reference
#: under-reports it fourfold, and the shortfall does not merely cost a metric:
#: scoring runs BEFORE the archive is written, so thrashing or an OOM kill here
#: loses the whole fit.
_QUALITY_PEAK_VOLUMES = 8


def _available_ram_gb() -> "float | None":
    """Free physical memory in GiB, or ``None`` where it cannot be measured."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):  # pragma: no cover - platform
        return None
    if pages <= 0 or page_size <= 0:  # pragma: no cover - platform
        return None
    return pages * page_size / 1024**3


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


def _compare_recourse(*, partition: "bool | None") -> str:
    if partition:
        return (
            "Flatten the written archive with `luxar gsplat flatten`, then run "
            "`luxar gsplat compare`."
        )
    if partition is None:
        return (
            "Run `luxar gsplat compare` on the written archive; if it is "
            "`kind=partition`, run `luxar gsplat flatten` first."
        )
    return "Run `luxar gsplat compare` on the written archive instead."


def announce_unscored_merge(reason: str, *, partition: "bool | None" = False) -> None:
    """Explain why a merged result carries no whole-volume quality metrics.

    Parameters
    ----------
    reason : str
        The reason scoring was unavailable.
    partition : bool or None, default False
        ``False`` when the result is known to be flat, ``True`` when it is a
        partition, and ``None`` when the caller cannot determine its kind.
    """
    aprint(
        f"No merged quality metrics: {reason}. {_compare_recourse(partition=partition)}"
    )


def announce_unscored_partition_merge(
    node: "GSplatData | GSplatNode", *, suffix: str = ""
) -> None:
    """Explain why a requested partition merge was not scored.

    Derives both the reason and compare recourse from the node the merge
    actually returned. A single surviving part is deliberately returned without
    a partition wrapper and may be either a leaf or an LOD group.

    Parameters
    ----------
    node : GSplatData or GSplatNode
        The node returned by the requested partition merge.
    suffix : str, default ""
        Caller-specific detail appended to the shared reason.
    """
    is_partition = isinstance(node, GSplatPartition)
    reason = (
        "the requested content-partition merge produced a kind=partition tree, "
        "but that path does not yet compute a whole-tree score"
        if is_partition
        else "the requested content-partition merge collapsed to a single "
        "matrix-shaped part, but that path does not yet compute a merged score"
    )
    announce_unscored_merge(f"{reason}{suffix}", partition=is_partition)


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


def stamp_merged_quality(
    merged: "GSplatData | Sequence[GSplatData]",
    volume: Any,
    *,
    volume_shape: tuple[int, ...],
    grid_scale: Optional[Sequence[float]],
    device: Optional[str],
    verbose: bool,
    stats: "dict[str, Any] | None" = None,
) -> None:
    """Score the MERGED reconstruction against the whole volume, in place.

    Each tile already scores itself, but those numbers are about crops of an
    apodized decomposition: the tiles overlap, so their errors do not compose
    into the merged one, and none of them can speak for the archive that
    actually ships. Without this a tiled archive carries no PSNR at all — which
    is exactly what a published dataset is asked for.

    The reference is shifted onto the merged fit's background-relative basis
    using the normalization level in ``target_stats``. The tiles reconstruct
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
        is_partition = not isinstance(merged, GSplatData)
        parts = [merged] if isinstance(merged, GSplatData) else list(merged)
        target_stats = stats
    elif isinstance(merged, GSplatData):
        is_partition = False
        parts = [merged]
        target_stats = merged.stats
    else:
        announce_unscored_merge(
            "partition merged-quality scoring was not given a stats target",
            partition=True,
        )
        return
    if not parts or sum(part.n_splats for part in parts) == 0:
        return
    budget_gb = _quality_budget_gb()
    needed_gb = _QUALITY_PEAK_VOLUMES * 4 * float(np.prod(volume_shape)) / 1024**3
    if needed_gb > budget_gb:
        recourse = _compare_recourse(partition=is_partition)
        aprint(
            f"Merged quality metrics skipped: scoring {volume_shape} peaks at "
            f"~{needed_gb:.1f} GiB (the reconstruction, the reference, and "
            f"SSIM's intermediates), over the {budget_gb:g} GiB budget. Raise "
            f"LUXAR_TILED_QUALITY_MAX_GB to score it anyway. {recourse}"
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
                    np.asarray(volume, dtype=np.float32), fit_image_min(target_stats)
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
    "announce_unscored_partition_merge",
    "resolve_merged_reference",
    "stamp_merged_quality",
]
