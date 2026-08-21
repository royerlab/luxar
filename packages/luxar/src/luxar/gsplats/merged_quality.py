"""Whole-volume quality scoring shared by merged gsplat fit paths."""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

import numpy as np
from arbol import aprint

from luxar.gsplats.gsplat_data import GSplatData

#: Upper bound, in GiB, on the memory a merged-quality score may hold resident.
#: Above it the score is skipped and announced. An explicit
#: ``LUXAR_TILED_QUALITY_MAX_GB`` overrides this ceiling and the free-memory
#: fraction below.
_QUALITY_BUDGET_GB = 24.0

#: Share of currently-free physical memory the default budget may consume.
_QUALITY_BUDGET_MEM_FRACTION = 0.5

#: Float32 volume equivalents resident at the SSIM scoring peak.
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
    """The default budget: the ceiling, held under a share of free memory."""
    available = _available_ram_gb()
    if available is None:  # pragma: no cover - platform
        return _QUALITY_BUDGET_GB
    return min(_QUALITY_BUDGET_GB, _QUALITY_BUDGET_MEM_FRACTION * available)


def _quality_budget_gb() -> float:
    """Resident-memory budget for scoring — the env override, or the default."""
    default = _default_quality_budget_gb()
    raw = os.environ.get("LUXAR_TILED_QUALITY_MAX_GB")
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
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
    """Explain why a merged result carries no whole-volume quality metrics."""
    aprint(
        f"No merged quality metrics: {reason}. {_compare_recourse(partition=partition)}"
    )


def _to_voxel_frame(merged: GSplatData, scale: Optional[Sequence[float]]) -> GSplatData:
    """Return the same mixture expressed on the fit grid's voxel frame."""
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
    merged: GSplatData,
    volume: Any,
    *,
    volume_shape: tuple[int, ...],
    grid_scale: Optional[Sequence[float]],
    device: Optional[str],
    verbose: bool,
) -> None:
    """Score the merged reconstruction against the whole reference in place.

    Regional scores do not compose after overlap blending or content-box core
    masking, so this renders the result that will actually be written. The
    reference is ``volume`` exactly as supplied: a subtracted pedestal is not
    restored, and per-region denoising is not applied to it. Under denoising,
    the reconstruction therefore targets denoised data while this reference
    retains acquisition noise; that is a different measurement from a
    whole-volume denoise that scores against its own smoothed reference.
    """
    if merged.n_splats == 0:
        return
    budget_gb = _quality_budget_gb()
    needed_gb = _QUALITY_PEAK_VOLUMES * 4 * float(np.prod(volume_shape)) / 1024**3
    if needed_gb > budget_gb:
        recourse = _compare_recourse(partition=False)
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

        scored = _to_voxel_frame(merged, grid_scale)
        rendered: Any = None
        reference: Any = None
        try:
            with torch.no_grad():
                rendered = render_to_volume_tensor(
                    scored,
                    shape=volume_shape,
                    device=device,
                    truncate=scored.truncation_radius,
                )
                reference = torch.as_tensor(
                    np.asarray(volume, dtype=np.float32), device=rendered.device
                )
                quality = compute_quality_metrics(rendered, reference)
        finally:
            del rendered, reference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        merged.stats.update(
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


__all__ = ["announce_unscored_merge", "stamp_merged_quality"]
