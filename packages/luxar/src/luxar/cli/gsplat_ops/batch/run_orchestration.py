"""Orchestration helper for local ``batch-fit run`` execution."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from arbol import aprint

if TYPE_CHECKING:
    from luxar.cli.gsplat_ops.batch.plan_configs import PlanConfigs
    from luxar.gsplats.batch.manifest import BatchManifest


def _refuse_resume_grid_mismatch(
    output_dir: Path,
    fresh: "BatchManifest",
    *,
    resume: bool,
    mismatch_help: str = "pass --no-resume to refit every tile",
    legacy_weight_help: str = "Pass --no-resume to refit every tile",
) -> None:
    """Refuse index-based resume when completed tiles use another grid."""
    if not resume or not (output_dir / "manifest.json").exists():
        return

    from luxar.gsplats.batch.manifest import load_manifest

    existing = load_manifest(output_dir)
    tiles_dir = output_dir / "tiles"
    completed_count = sum(
        (tiles_dir / job.output_filename).exists()
        or Path(str(tiles_dir / job.output_filename) + ".empty").exists()
        for job in existing.jobs
    )
    if completed_count == 0:
        return

    def grid_signature(manifest: "BatchManifest") -> tuple[Any, ...]:
        if manifest.mode != "uniform":
            return (
                manifest.mode,
                tuple(manifest.spatial_shape),
                manifest.tile_size,
                manifest.tile_overlap,
                manifest.n_tiles,
            )

        from luxar.gsplats.tiling import compute_tile_specs

        specs = compute_tile_specs(
            tuple(manifest.spatial_shape),
            manifest.tile_size,
            manifest.tile_overlap,
            fold_slivers=manifest.fold_tile_slivers,
        )
        regions = tuple(
            tuple((span.start, span.stop) for span in spec.slices) for spec in specs
        )
        return manifest.mode, regions

    old_grid = grid_signature(existing)
    new_grid = grid_signature(fresh)
    if old_grid != new_grid:
        raise typer.BadParameter(
            "cannot resume: the existing tile outputs use a different grid; "
            f"{mismatch_help}"
        )
    if (
        existing.tile_occupancy_weights is None
        and fresh.tile_occupancy_weights is not None
    ):
        tile_word = "tile" if completed_count == 1 else "tiles"
        aprint(
            f"Warning: {completed_count} completed {tile_word} predating occupancy "
            "weighting will keep equal-share budgets. "
            f"{legacy_weight_help}."
        )
    elif (
        existing.tile_occupancy_weights is not None
        and fresh.tile_occupancy_weights is not None
        and (
            len(existing.tile_occupancy_weights) != len(fresh.tile_occupancy_weights)
            or any(
                len(old_row) != len(new_row)
                or any(
                    not math.isclose(old_weight, new_weight)
                    for old_weight, new_weight in zip(old_row, new_row, strict=True)
                )
                for old_row, new_row in zip(
                    existing.tile_occupancy_weights,
                    fresh.tile_occupancy_weights,
                    strict=True,
                )
            )
        )
    ):
        tile_word = "tile" if completed_count == 1 else "tiles"
        budget_word = "budget" if completed_count == 1 else "budgets"
        owner_word = "its" if completed_count == 1 else "their"
        aprint(
            f"Warning: {completed_count} completed {tile_word} planned with different "
            f"occupancy weights will keep {owner_word} existing seed {budget_word}. "
            f"{legacy_weight_help}."
        )


def _resolve_local_gpu_profile(
    gpus: str,
) -> tuple[list[int], str, Optional[list[int]], Optional[list[dict[str, Any]]]]:
    """Resolve local devices and the conservative profile used for planning."""
    # Keep runtime discovery imports local so CLI tests can isolate hardware and
    # profile dependencies without importing this orchestration module differently.
    import torch

    from luxar.gsplats.gpu_profile import get_gpu_summary, get_gpu_throughput_table
    from luxar.gsplats.utils.device import resolve_gpu_selection

    try:
        selected = resolve_gpu_selection(gpus)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not selected:
        summary = get_gpu_summary()
        if summary is None:
            return selected, "CPU", None, None
        recommendations = summary.get("recommendations", {})
        peak = recommendations.get("peak_throughput_3d", {})
        oom = summary.get("oom_boundaries", {}).get("3d", {})
        max_shape = oom.get("max_successful_shape", peak.get("shape", []))
        return selected, "CPU", max_shape, get_gpu_throughput_table()

    device_memory_by_name: dict[str, int] = {}
    for index in selected:
        properties = torch.cuda.get_device_properties(index)
        name = str(properties.name)
        total_memory = int(properties.total_memory)
        device_memory_by_name[name] = min(
            device_memory_by_name.get(name, total_memory), total_memory
        )

    devices = list(device_memory_by_name.items())
    manifest_name = ", ".join(name for name, _ in devices)
    profiled: list[tuple[int, str, dict[str, Any]]] = []
    for name, total_memory in devices:
        summary = get_gpu_summary(gpu_name=name)
        if summary is None:
            aprint(
                f"no benchmark profile for {name} — tile auto-sizing off; "
                "pass --tile-size or run 'luxar gsplat benchmark'"
            )
            return selected, manifest_name, None, None
        profiled.append((total_memory, name, summary))

    _, profile_name, summary = min(profiled, key=lambda item: item[0])
    recommendations = summary.get("recommendations", {})
    peak = recommendations.get("peak_throughput_3d", {})
    oom = summary.get("oom_boundaries", {}).get("3d", {})
    max_shape = oom.get("max_successful_shape", peak.get("shape", []))
    throughput_table = get_gpu_throughput_table(gpu_name=profile_name)
    return selected, manifest_name, max_shape, throughput_table


def _resolve_local_deferred_floor(
    manifest: "BatchManifest", output_dir: Path
) -> "BatchManifest":
    """Resolve a local denoise-dependent floor after calibrating NLM if needed.

    A deferred percentile floor pins one calibrated ``h`` per selected channel
    into both the global floor resolver and subsequent local workers. Non-deferred
    ``auto`` runs retain their historical per-worker self-calibration.
    """
    from luxar.cli.gsplat_ops.batch.denoise_workers import resolve_deferred_batch_floor
    from luxar.gsplats.batch.manifest import load_manifest, save_manifest
    from luxar.gsplats.preprocessing.denoise_pipeline import calibrate_all_channels

    if manifest.denoise_h is None:
        h_values = calibrate_all_channels(
            input_path=Path(manifest.input_path),
            n_timepoints=manifest.n_timepoints,
            n_channels=manifest.n_channels,
            channel_indices=manifest.channel_indices,
            timepoint_indices=manifest.timepoint_indices,
            array_key=manifest.array_key,
            axes=manifest.axes,
            calibration_samples=manifest.calibration_samples,
            patch_size=manifest.denoise_patch_size,
            search_distance=manifest.denoise_search_distance,
            backend=manifest.denoise_backend,
        )
        manifest.denoise_h_values = {str(key): value for key, value in h_values.items()}
    save_manifest(manifest, output_dir)
    resolve_deferred_batch_floor(output_dir)
    return load_manifest(output_dir)


def run_batch_local_orchestration(
    *,
    input_path: Path,
    output_dir: Path,
    tiling: str,
    tile_size: Optional[int],
    tile_overlap: int,
    axes: Optional[str],
    array_key: Optional[str],
    timepoints_slice: Optional[str],
    channels_slice: Optional[str],
    cfgs: "PlanConfigs",
    gpus: str,
    jobs_per_gpu: str,
    no_resume: bool,
    dry_run: bool,
) -> None:
    """Build a local batch plan, print summary, and execute local workers."""
    from luxar.cli.gsplat_config import parse_hex_color
    from luxar.cli.gsplat_ops.batch.planning import plan_batch
    from luxar.cli.gsplat_ops.batch.recipe_args import (
        build_merge_recipe_params as _build_merge_recipe_params_impl,
    )
    from luxar.gsplats.batch.local_runner import run_batch_local

    axes_list = [a.strip() for a in axes.split(",")] if axes else None

    # Plan against the selected execution device(s), not torch's default GPU 0.
    selected_gpus, resolved_gpu, max_shape, throughput_table = (
        _resolve_local_gpu_profile(gpus)
    )

    fit_cfg = cfgs.fit
    denoise_cfg = cfgs.denoise
    content_cfg = cfgs.content
    merge_cfg = cfgs.merge

    # Merge-recipe knobs (incl. --merge-target-ms sizing) are resolved
    # INSIDE plan_batch, after shape discovery — so the ladder is sized
    # with the true merged ndim, matching `batch-fit merge`.
    plan = plan_batch(
        input_path=input_path,
        output_dir=output_dir,
        tiling=tiling,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        axes_list=axes_list,
        array_key=array_key,
        timepoints_slice=timepoints_slice,
        channels_slice=channels_slice,
        fit=fit_cfg,
        denoise=denoise_cfg,
        content=content_cfg,
        merge=merge_cfg,
        max_shape=max_shape,
        throughput_table=throughput_table,
        resolved_gpu=resolved_gpu,
    )
    manifest = plan.manifest
    _refuse_resume_grid_mismatch(output_dir, manifest, resume=not no_resume)

    # Resolve merge colors + recipe params for the streaming merge.
    colors = None
    if manifest.channel_colors:
        colors = [parse_hex_color(c.strip()) for c in manifest.channel_colors]
    recipe_params = None
    if manifest.merge_recipe is not None:
        recipe_params = _build_merge_recipe_params_impl(manifest.merge_recipe_args)

    # Plan summary (which GPUs, how many tasks already done).
    gpu_desc = "CPU" if not selected_gpus else f"GPU(s) {selected_gpus}"
    slot = "boxes" if manifest.mode == "content" else "tiles"
    aprint("")
    aprint("=" * 60)
    aprint("LOCAL BATCH FIT")
    aprint("=" * 60)
    aprint(
        f"  Input: {input_path.name} "
        f"(T={manifest.n_timepoints}, C={manifest.n_channels}, "
        f"spatial={'x'.join(str(s) for s in manifest.spatial_shape)})"
    )
    aprint(
        f"  Decomposition: {manifest.mode}, {manifest.n_tiles} {slot}/volume "
        f"(overlap={tile_overlap})"
    )
    aprint(
        f"  Tasks: {manifest.n_timepoints} x {manifest.n_channels} x "
        f"{manifest.n_tiles} = {manifest.total_tasks} fits"
    )
    aprint(f"  Devices: {gpu_desc} (--jobs-per-gpu {jobs_per_gpu})")
    if manifest.merge_recipe:
        aprint(f"  Merge recipe: {manifest.merge_recipe}")
    aprint(f"  Output: {output_dir}")
    aprint("")

    if dry_run:
        aprint("Dry run -- omit --dry-run to actually fit.")
        raise typer.Exit(0)

    if manifest.floor_deferred:
        manifest = _resolve_local_deferred_floor(manifest, output_dir)

    final_path = run_batch_local(
        manifest,
        output_dir,
        gpus=gpus,
        jobs_per_gpu=jobs_per_gpu,
        resume=not no_resume,
        channel_colors=colors,
        recipe=manifest.merge_recipe,
        recipe_params=recipe_params,
    )
    aprint(f"\nFinal output: {final_path}")
    aprint(f"Inspect: luxar gsplat info {final_path}")
    aprint(f"Validate tiles: luxar gsplat batch-fit validate {output_dir}")
