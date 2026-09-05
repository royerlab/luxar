"""Flat CLI flags -> the four ``plan_batch`` config dataclasses.

``batch-fit run`` and ``batch-fit submit`` plan the *same* job — the whole
discover/decompose/manifest half is shared through :func:`plan_batch` — and
differ only in who executes the resulting tasks. So the mapping from their flag
lists onto :mod:`planning`'s ``FitConfig`` / ``DenoiseConfig`` / ``ContentKnobs``
/ ``MergeConfig`` belongs in ONE place that neither command owns.

It briefly did not. Each command built the four dataclasses itself, field for
field, and the copies drifted: ``--calibration-samples`` was declared on
``submit`` only, while the local path consumed ``manifest.calibration_samples``
(``run_orchestration._resolve_local_deferred_floor`` hands it to
``calibrate_all_channels``). A local ``--denoise`` run therefore always
calibrated on 5 timepoints with no way to ask for more.

``cli/tests/test_batch_config_agreement.py`` now fails if a second construction
site appears, if the mapper consumes a flag one command does not expose, or if
the two commands' shared flags disagree on a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PlanConfigs:
    """The four ``plan_batch`` config dataclasses, assembled from CLI flags."""

    fit: Any
    denoise: Any
    content: Any
    merge: Any


def build_plan_configs(
    *,
    preset: str,
    seeds: Optional[str],
    iters: Optional[int],
    config: Optional[Path],
    floor: Optional[str],
    batch_progressive: bool,
    batch_splats_per_pass: Optional[int],
    batch_psnr_patience: Optional[float],
    batch_max_passes: Optional[int],
    batch_cull_retention: Optional[float],
    batch_denoise: bool,
    batch_denoise_h: Optional[float],
    batch_denoise_2d: bool,
    batch_denoise_patch_size: int,
    batch_denoise_search_distance: int,
    batch_denoise_backend: str,
    batch_calibration_samples: int,
    batch_preprocess: Optional[bool],
    cal: Optional[Path],
    k_star_ref: Optional[int],
    n_features_ref: Optional[int],
    saturation_exponent: float,
    saturation_cap: Optional[int],
    feature_threshold: Optional[float],
    feature_metric: Optional[str],
    cell: int,
    target_features: Optional[int],
    min_leaf: int,
    max_leaf: int,
    plan_timepoint: Optional[int],
    plan_samples: int,
    merge_recipe: Optional[str],
    channel_colors: Optional[str],
    merge_n_lods: Optional[int],
    merge_additive_method: Optional[str],
    merge_breakpoints: Optional[str],
    merge_target_ms: Optional[float],
    merge_bandwidth_mbps: Optional[float],
    merge_bytes_per_splat: Optional[float],
    merge_compression_factor: Optional[int],
    merge_levels: Optional[int],
    merge_refine: Optional[str],
    merge_refine_iters: Optional[int],
    merge_substitutive_method: Optional[str],
    merge_coarsen_dims: Optional[str],
) -> PlanConfigs:
    """Map the submit CLI flags onto the ``plan_batch`` config dataclasses."""
    from luxar.cli.gsplat_ops.batch.planning import (
        ContentKnobs,
        DenoiseConfig,
        FitConfig,
        MergeConfig,
    )

    fit_cfg = FitConfig(
        preset=preset,
        seeds=seeds,
        iters=iters,
        config=config,
        floor=floor,
        progressive=batch_progressive,
        splats_per_pass=batch_splats_per_pass,
        psnr_patience=batch_psnr_patience,
        max_passes=batch_max_passes,
        cull_retention=batch_cull_retention,
    )
    denoise_cfg = DenoiseConfig(
        denoise=batch_denoise,
        denoise_h=batch_denoise_h,
        denoise_2d=batch_denoise_2d,
        patch_size=batch_denoise_patch_size,
        search_distance=batch_denoise_search_distance,
        backend=batch_denoise_backend,
        calibration_samples=batch_calibration_samples,
        preprocess=batch_preprocess,
    )
    content_cfg = ContentKnobs(
        cal=cal,
        k_star_ref=k_star_ref,
        n_features_ref=n_features_ref,
        saturation_exponent=saturation_exponent,
        saturation_cap=saturation_cap,
        feature_threshold=feature_threshold,
        feature_metric=feature_metric,
        cell=cell,
        target_features=target_features,
        min_leaf=min_leaf,
        max_leaf=max_leaf,
        plan_timepoint=plan_timepoint,
        plan_samples=plan_samples,
    )
    merge_cfg = MergeConfig(
        recipe=merge_recipe,
        channel_colors=channel_colors,
        n_lods=merge_n_lods,
        additive_method=merge_additive_method,
        breakpoints=merge_breakpoints,
        target_ms=merge_target_ms,
        bandwidth_mbps=merge_bandwidth_mbps,
        bytes_per_splat=merge_bytes_per_splat,
        compression_factor=merge_compression_factor,
        levels=merge_levels,
        refine=merge_refine,
        refine_iters=merge_refine_iters,
        substitutive_method=merge_substitutive_method,
        coarsen_dims=merge_coarsen_dims,
    )
    return PlanConfigs(
        fit=fit_cfg, denoise=denoise_cfg, content=content_cfg, merge=merge_cfg
    )
