"""Scheduler-agnostic batch planning shared by ``batch-fit submit`` and ``run``.

``batch_submit`` (Slurm) and ``batch_run`` (local) share everything *up to and
including* building the :class:`BatchManifest` + its job list: dataset discovery
+ T/C slicing, uniform-tile / content-box decomposition (and the shared
``plan.json``), fit-arg assembly, and per-part merge-recipe validation.  Only the
*execution* differs (sbatch submit vs local subprocess pool).

This module owns that shared half so the two commands don't duplicate ~300 lines.
It lives in the CLI layer (it drives CLI helpers like ``load_volume`` /
``_resolve_density`` and raises :class:`typer.BadParameter`); the local engine
(:mod:`luxar.gsplats.batch.local_runner`) consumes only the returned manifest, so
there is no upward dependency from the gsplats package into the CLI.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import typer
from arbol import aprint, asection

from luxar.gsplats.batch.manifest import BatchJob, BatchManifest, output_filename

# ---------------------------------------------------------------------------
# Grouped option contracts (the shared surface both commands populate)
# ---------------------------------------------------------------------------


@dataclass
class FitConfig:
    """Per-tile fit parameters shared by both commands."""

    preset: str = "standard"
    seeds: Optional[str] = None
    iters: Optional[int] = None
    config: Optional[Path] = None
    floor: Optional[str] = "auto"
    progressive: bool = False
    splats_per_pass: Optional[int] = None
    psnr_patience: Optional[float] = None
    max_passes: Optional[int] = None
    cull_retention: Optional[float] = None


@dataclass
class DenoiseConfig:
    """NLM denoise parameters (off by default)."""

    denoise: bool = False
    denoise_h: Optional[float] = None
    denoise_2d: bool = False
    patch_size: int = 3
    search_distance: int = 5
    backend: str = "auto"
    calibration_samples: int = 5
    preprocess: Optional[bool] = None


@dataclass
class ContentKnobs:
    """Content-adaptive box-plan knobs (used only when ``tiling == 'content'``)."""

    cal: Optional[Path] = None
    k_star_ref: Optional[int] = None
    n_features_ref: Optional[int] = None
    saturation_exponent: float = 0.44
    saturation_cap: Optional[int] = None
    feature_threshold: Optional[float] = None
    feature_metric: Optional[str] = None
    cell: int = 16
    target_features: Optional[int] = None
    min_leaf: int = 256
    max_leaf: int = 512
    plan_timepoint: Optional[int] = None
    plan_samples: int = 16


@dataclass
class MergeConfig:
    """Per-part LOD merge recipe + its knobs (validated by :func:`resolve_merge_recipe_args`)."""

    recipe: Optional[str] = None
    channel_colors: Optional[str] = None
    n_lods: Optional[int] = None
    additive_method: Optional[str] = None
    breakpoints: Optional[str] = None
    # Streaming sizing (--merge-target-ms trio): resolved at plan time into a
    # concrete `breakpoints="stream:<c>"` string, so the manifest schema is
    # unchanged and the merge job just re-parses the stored breakpoints.
    target_ms: Optional[float] = None
    bandwidth_mbps: Optional[float] = None
    bytes_per_splat: Optional[float] = None
    compression_factor: Optional[int] = None
    levels: Optional[int] = None
    substitutive_method: Optional[str] = None
    coarsen_dims: Optional[str] = None


@dataclass
class PlanResult:
    """Output of :func:`plan_batch`: the manifest + decomposition facts."""

    manifest: BatchManifest
    content_plan: Optional[Any]  # FitPlan in content mode, else None
    tile_voxels: int
    needs_tiling: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_plan_timepoints(
    t_indices: list[int], plan_timepoint: Optional[int], plan_samples: int
) -> list[int]:
    """Pick which timepoints to scan when building the shared content box plan.

    A pinned ``plan_timepoint`` scans just that timepoint (it must be one of the
    selected ``t_indices``); otherwise return up to ``plan_samples`` evenly-spaced
    timepoints (endpoints included, deduplicated) to max-project — so the plan
    covers any region with signal at ANY timepoint rather than leaving holes where
    content moved over time. Raises :class:`typer.BadParameter` on invalid input.
    """
    if plan_timepoint is not None:
        if plan_timepoint not in t_indices:
            raise typer.BadParameter(
                f"--plan-timepoint {plan_timepoint} is not in the selected "
                f"timepoints {t_indices[0]}..{t_indices[-1]} "
                f"({len(t_indices)} total)."
            )
        return [plan_timepoint]
    if plan_samples < 1:
        raise typer.BadParameter(f"--plan-samples must be >= 1, got {plan_samples}.")
    if len(t_indices) <= plan_samples:
        return list(t_indices)
    import numpy as _np

    idx = _np.unique(
        _np.linspace(0, len(t_indices) - 1, plan_samples).round().astype(int)
    )
    return [t_indices[i] for i in idx]


def _parse_slice(s: str, max_val: int) -> list[int]:
    """Parse a Python-style ``start:stop:step`` slice (or a single index)."""
    parts = s.split(":")
    if len(parts) == 1:
        return [int(parts[0])]
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if len(parts) > 1 and parts[1] else max_val
    step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return list(range(start, stop, step))


def resolve_merge_recipe_args(
    merge: MergeConfig, *, merged_ndim: int = 4, merged_has_colors: bool = False
) -> dict:
    """Validate the per-part merge recipe + knobs into the manifest dict.

    Fail-fast (before any expensive planning / fitting): rejects an unsupported
    recipe, cross-recipe knobs, and merge knobs given WITHOUT a ``--merge-recipe``
    (previously silently dropped), and validates method/breakpoint spellings —
    mirroring ``fit --recipe`` and ``gsplat lod``. The streaming trio
    (``--merge-target-ms``/``--merge-bandwidth-mbps``/``--merge-bytes-per-splat``)
    is resolved HERE into a concrete ``breakpoints="stream:<c>"`` string (analytic
    bytes/splat for ``merged_ndim`` — no gsplat store exists yet at plan time), so
    the manifest schema is unchanged. ``merged_ndim`` defaults to 4 (3 spatial +
    the stacked-timepoint axis, the whole-timelapse norm); pass the value the
    merge itself will compute — ``len(spatial_shape) + (n_timepoints > 1)`` — so
    plan-time and merge-time size the SAME ladder for the same data.
    ``merged_has_colors`` marks a merge that will write per-splat colors (a
    multi-channel merge with channel colors). Returns ``{}`` when no recipe and
    no knobs are requested. Raises :class:`typer.BadParameter` on any problem.
    """
    if merge.recipe is None:
        # A merge knob without a recipe is a silent no-op — reject it loudly so
        # the user adds --merge-recipe (mirrors `batch-fit merge`'s runtime check).
        orphaned = [
            flag
            for flag, val in {
                "--merge-n-lods": merge.n_lods,
                "--merge-additive-method": merge.additive_method,
                "--merge-breakpoints": merge.breakpoints,
                "--merge-target-ms": merge.target_ms,
                "--merge-bandwidth-mbps": merge.bandwidth_mbps,
                "--merge-bytes-per-splat": merge.bytes_per_splat,
                "--merge-compression-factor": merge.compression_factor,
                "--merge-levels": merge.levels,
                "--merge-substitutive-method": merge.substitutive_method,
                "--merge-coarsen-dims": merge.coarsen_dims,
            }.items()
            if val is not None
        ]
        if orphaned:
            raise typer.BadParameter(
                f"option(s) {', '.join(sorted(orphaned))} require a "
                f"--merge-recipe but none was given; pass --merge-recipe "
                f"stream|levels (without one the merge writes bare-leaf "
                f"parts, so these knobs would be ignored)."
            )
        return {}

    from luxar.gsplats.lod.recipes import LEGACY_RECIPE_NAMES, PER_PART_RECIPES

    if merge.recipe in LEGACY_RECIPE_NAMES:
        raise typer.BadParameter(
            f"recipe {merge.recipe!r} was renamed to "
            f"{LEGACY_RECIPE_NAMES[merge.recipe]!r}; use --merge-recipe "
            f"{LEGACY_RECIPE_NAMES[merge.recipe]}."
        )
    if merge.recipe not in PER_PART_RECIPES:
        raise typer.BadParameter(
            f"--merge-recipe {merge.recipe!r} is not supported; choose from "
            f"{', '.join(sorted(PER_PART_RECIPES))} (the composed recipes "
            "re-partition their input, but each tile is already one part)."
        )

    additive_only = {
        "--merge-n-lods": merge.n_lods,
        "--merge-additive-method": merge.additive_method,
        "--merge-breakpoints": merge.breakpoints,
        "--merge-target-ms": merge.target_ms,
        "--merge-bandwidth-mbps": merge.bandwidth_mbps,
        "--merge-bytes-per-splat": merge.bytes_per_splat,
    }
    substitutive_only = {
        "--merge-compression-factor": merge.compression_factor,
        "--merge-levels": merge.levels,
        "--merge-substitutive-method": merge.substitutive_method,
        "--merge-coarsen-dims": merge.coarsen_dims,
    }
    irrelevant = substitutive_only if merge.recipe == "stream" else additive_only
    provided = [flag for flag, val in irrelevant.items() if val is not None]
    if provided:
        other = "levels" if merge.recipe == "stream" else "stream"
        raise typer.BadParameter(
            f"option(s) {', '.join(provided)} are not used by "
            f"--merge-recipe {merge.recipe} (they configure "
            f"--merge-recipe {other}). Remove them or switch recipe."
        )

    # Streaming trio → a concrete stream:<c> breakpoints string, resolved at
    # plan time (the stored string round-trips the manifest untouched).
    from luxar.cli.lod import validate_streaming_knobs

    validate_streaming_knobs(
        merge.target_ms,
        merge.bandwidth_mbps,
        merge.bytes_per_splat,
        merge.breakpoints,
        prefix="--merge-",
    )
    eff_breakpoints = merge.breakpoints
    if merge.target_ms is not None:
        from luxar.cli.lod import (
            estimate_bytes_per_splat,
            resolve_streaming_breakpoints,
        )

        eff_breakpoints = resolve_streaming_breakpoints(
            merge.target_ms,
            merge.bandwidth_mbps,
            merge.bytes_per_splat,
            analytic_bps=estimate_bytes_per_splat(
                merged_ndim, has_colors=merged_has_colors
            ),
        )

    args: dict = {}
    if merge.n_lods is not None:
        args["n-lods"] = str(merge.n_lods)
    if merge.additive_method is not None:
        from luxar.cli.lod import VALID_ADDITIVE_METHODS

        am_norm = merge.additive_method.strip().replace("-", "_")
        if am_norm not in VALID_ADDITIVE_METHODS:
            raise typer.BadParameter(
                f"--merge-additive-method must be one of "
                f"{list(VALID_ADDITIVE_METHODS)}; got {merge.additive_method!r}"
            )
        args["additive-method"] = am_norm
    if eff_breakpoints is not None:
        from luxar.cli.lod import parse_lod_breakpoints

        parse_lod_breakpoints(eff_breakpoints)
        args["breakpoints"] = eff_breakpoints
    if merge.compression_factor is not None:
        args["compression-factor"] = str(merge.compression_factor)
    if merge.levels is not None:
        args["levels"] = str(merge.levels)
    if merge.substitutive_method is not None:
        from luxar.cli.lod import VALID_SUBSTITUTIVE_METHODS

        sm_norm = merge.substitutive_method.strip().replace("-", "_")
        if sm_norm not in VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--merge-substitutive-method must be one of "
                f"{list(VALID_SUBSTITUTIVE_METHODS)}; "
                f"got {merge.substitutive_method!r}"
            )
        args["substitutive-method"] = sm_norm
    if merge.coarsen_dims is not None:
        args["coarsen-dims"] = merge.coarsen_dims
    return args


def _assemble_fit_args(
    fit: FitConfig, denoise: DenoiseConfig
) -> Tuple[dict, Optional[str], Optional[str]]:
    """Build the ``fit_args`` dict + denoise mode from the fit/denoise configs.

    Returns ``(fit_args, denoise_mode, denoised_zarr_path)``. ``denoise_mode`` is
    set later against ``output_dir`` by the caller for preprocess mode; here it is
    resolved to ``'preprocess'`` / ``'on-the-fly'`` / ``None``.
    """
    fit_args: dict = {}
    if fit.seeds:
        fit_args["seeds"] = fit.seeds
    if fit.iters is not None:
        fit_args["iters"] = str(fit.iters)
    if fit.config:
        fit_args["config"] = str(fit.config)
    if fit.floor is not None:
        fit_args["floor"] = str(fit.floor)
    if fit.progressive:
        fit_args["progressive"] = ""  # boolean flag, no value
    if fit.splats_per_pass is not None:
        fit_args["splats-per-pass"] = str(fit.splats_per_pass)
    if fit.psnr_patience is not None:
        fit_args["psnr-patience"] = str(fit.psnr_patience)
    if fit.max_passes is not None:
        fit_args["max-passes"] = str(fit.max_passes)
    if fit.cull_retention is not None:
        fit_args["cull-retention"] = str(fit.cull_retention)

    denoise_mode: Optional[str] = None
    if denoise.denoise:
        if denoise.preprocess is True:
            denoise_mode = "preprocess"
        else:
            # Default (and explicit --no-preprocess): denoise per-tile in each fit.
            denoise_mode = "on-the-fly"
        if denoise_mode == "on-the-fly":
            fit_args["denoise"] = ""
            if denoise.denoise_2d:
                fit_args["denoise-2d"] = ""
            if denoise.patch_size != 3:
                fit_args["denoise-patch-size"] = str(denoise.patch_size)
            if denoise.search_distance != 5:
                fit_args["denoise-search-distance"] = str(denoise.search_distance)
            if denoise.backend != "auto":
                fit_args["denoise-backend"] = denoise.backend
            # --denoise-h is injected at runtime from the calibrated h values.
    return fit_args, denoise_mode, None


def plan_batch(
    *,
    input_path: Path,
    output_dir: Path,
    tiling: str,
    tile_size: Optional[int],
    tile_overlap: int,
    axes_list: Optional[List[str]],
    array_key: Optional[str],
    timepoints_slice: Optional[str],
    channels_slice: Optional[str],
    fit: FitConfig,
    denoise: DenoiseConfig,
    content: ContentKnobs,
    merge: MergeConfig,
    merge_recipe_args: Optional[dict] = None,
    max_shape: Optional[List[int]] = None,
    throughput_table: Optional[list] = None,
    resolved_gpu: str = "unknown",
) -> PlanResult:
    """Discover, decompose, and build the manifest + job list (no execution).

    The scheduler-agnostic half of ``batch-fit``: stages shared by ``submit``
    (Slurm) and ``run`` (local). Slurm-only manifest fields (partition, packing,
    preemptible, …) are left at defaults for the caller to populate.

    Parameters
    ----------
    merge_recipe_args
        Pre-resolved manifest dict (tests). Default ``None`` resolves ``merge``
        via :func:`resolve_merge_recipe_args` HERE, after shape discovery — so
        ``--merge-target-ms`` is sized with the true merged ndim
        (``len(spatial) + (n_t > 1)``) and color-carrying multi-channel merges,
        exactly matching what ``batch-fit merge`` computes at merge time.
    max_shape
        GPU-profile "largest shape that fits" (uniform auto tile-size). ``None``
        in content mode, or locally without a benchmark profile — then an explicit
        ``tile_size`` is required for a multi-tile uniform fit.
    throughput_table
        GPU-profile throughput (for the wall-time estimate); ``None`` -> a flat
        default (irrelevant locally).
    resolved_gpu
        GPU name recorded in the manifest.
    """
    from luxar.cli.gsplat_config import (
        PRESETS,
        decode_flat_channel_index,
        discover_ome_zarr_shape,
    )
    from luxar.gsplats.batch.time_estimate import estimate_tile_wall_seconds
    from luxar.gsplats.tiling import compute_tile_specs

    # batch-fit fans a whole nD dataset across its axes — it needs a chunked,
    # randomly-addressable OME-Zarr store (so each task reads only its tile/box,
    # and a huge movie is never fully materialized). A flat .npy/.tiff/.h5 would
    # force a full in-RAM load per task. Fail fast with a clear pointer.
    if input_path.suffix.lower() in (".npy", ".npz", ".tif", ".tiff", ".h5", ".hdf5"):
        raise typer.BadParameter(
            f"batch-fit needs an OME-Zarr input (.zarr or .zarr.zip); got "
            f"'{input_path.name}'. Convert it to zarr first, or use `gsplat fit` "
            f"for a single {input_path.suffix.lower()} volume."
        )

    fit_args, denoise_mode, _ = _assemble_fit_args(fit, denoise)
    denoised_zarr_path = None
    if denoise_mode == "preprocess":
        denoised_zarr_path = str(output_dir.resolve() / "denoised.zarr")

    # 2. Discover dataset shape + apply --timepoints/--channels slicing.
    with asection("Discovering dataset shape"):
        ome_info = discover_ome_zarr_shape(
            input_path, axes_override=axes_list, array_key=array_key
        )
        n_t_full = ome_info.n_timepoints
        n_c_full = ome_info.n_channels
        spatial = ome_info.spatial_shape
        aprint(f"Axes: {ome_info.axes}")
        aprint(f"Shape: {ome_info.shape}")
        aprint(f"T={n_t_full}, C={n_c_full}, spatial={'x'.join(map(str, spatial))}")

        t_indices = (
            _parse_slice(timepoints_slice, n_t_full)
            if timepoints_slice
            else list(range(n_t_full))
        )
        c_indices = (
            _parse_slice(channels_slice, n_c_full)
            if channels_slice
            else list(range(n_c_full))
        )
        if not t_indices:
            raise ValueError("--timepoints selected no timepoints")
        if not c_indices:
            raise ValueError("--channels selected no channels")
        bad_t = [i for i in t_indices if i < 0 or i >= n_t_full]
        bad_c = [i for i in c_indices if i < 0 or i >= n_c_full]
        if bad_t:
            raise ValueError(
                f"--timepoints selected out-of-range indices {bad_t}; "
                f"valid range is 0..{n_t_full - 1}"
            )
        if bad_c:
            raise ValueError(
                f"--channels selected out-of-range flat channel indices {bad_c}; "
                f"valid range is 0..{n_c_full - 1}"
            )
        n_t = len(t_indices)
        n_c = len(c_indices)
        if timepoints_slice or channels_slice:
            aprint(f"Sliced: T={n_t} (of {n_t_full}), C={n_c} (of {n_c_full})")

    # Resolve the merge recipe knobs now that the merged output's shape facts
    # are known: the merge stacks timepoints onto an extra axis (so merged ndim
    # is spatial + 1 only when T > 1) and writes per-splat colors only for a
    # multi-channel merge with channel colors. Sizing --merge-target-ms here
    # with the same inputs `batch-fit merge` uses guarantees plan-time and
    # merge-time produce the SAME ladder for the same data.
    if merge_recipe_args is None:
        merge_recipe_args = resolve_merge_recipe_args(
            merge,
            merged_ndim=len(spatial) + (1 if n_t > 1 else 0),
            merged_has_colors=bool(merge.channel_colors) and n_c > 1,
        )

    # 3. Decompose the spatial volume into the slots fanned across (t, c).
    mode = "content" if tiling == "content" else "uniform"
    content_plan = None
    plan_path_str: Optional[str] = None
    total_voxels = math.prod(spatial)

    if mode == "content":
        import numpy as _np

        from luxar.cli.gsplat_config import load_volume
        from luxar.cli.gsplat_ops.planner import _resolve_density
        from luxar.gsplats.planner import plan_volume
        from luxar.gsplats.planner.fit_planned_parallel import max_padded_box_voxels

        rep_c = c_indices[0]
        plan_t_samples = _select_plan_timepoints(
            t_indices, content.plan_timepoint, content.plan_samples
        )
        scan_desc = (
            f"t={plan_t_samples[0]}"
            if len(plan_t_samples) == 1
            else f"max-proj of {len(plan_t_samples)} timepoints"
        )
        with asection(f"Content plan (scan {scan_desc}, c={rep_c})"):
            rep_vol = load_volume(
                input_path,
                channel=rep_c,
                timepoint=plan_t_samples[0],
                array_key=array_key,
                axes=",".join(axes_list) if axes_list else None,
            )
            for _t in plan_t_samples[1:]:
                _v = load_volume(
                    input_path,
                    channel=rep_c,
                    timepoint=_t,
                    array_key=array_key,
                    axes=",".join(axes_list) if axes_list else None,
                )
                rep_vol = _np.maximum(rep_vol, _v)
            density = _resolve_density(
                content.cal,
                content.k_star_ref,
                content.n_features_ref,
                content.saturation_exponent,
                content.saturation_cap,
                content.feature_metric,
                content.feature_threshold,
            )
            if (
                content.feature_metric is not None
                and content.cal is not None
                and content.feature_metric != density.feature_method
            ):
                aprint(
                    f"⚠ --feature-metric '{content.feature_metric}' differs from the "
                    f"calibrated density.feature_method '{density.feature_method}' — "
                    "per-box budgets will be mis-scaled. Use matching metrics."
                )
            # Scan the floor-suppressed volume the boxes will fit: cal records
            # `density.feature_threshold` on floor-subtracted data, and each box
            # fits with `fit.floor`. Subtract the same floor from the (max-proj)
            # scan volume so the content field is on the calibration's scale —
            # otherwise the raw pedestal counts as signal and flattens the plan.
            from luxar.gsplats.fitting.preprocessing import _resolve_floor

            scan_floor = _resolve_floor(
                rep_vol, "auto" if fit.floor is None else fit.floor
            )
            if scan_floor is not None:
                rep_vol = _np.clip(rep_vol.astype(_np.float32) - scan_floor, 0.0, None)
            content_plan = plan_volume(
                rep_vol,
                density,
                feature_method=(content.feature_metric or density.feature_method),
                cell=content.cell,
                target_features=content.target_features,
                min_leaf=content.min_leaf,
                max_leaf=content.max_leaf,
                overlap=tile_overlap,
            )
            kept_boxes = [b for b in content_plan.boxes if b.budget > 0]
            if not kept_boxes:
                raise typer.BadParameter("content plan has no boxes with budget > 0")
            content_plan = dataclasses.replace(content_plan, boxes=kept_boxes)
            output_dir.mkdir(parents=True, exist_ok=True)
            plan_path_obj = output_dir.resolve() / "plan.json"
            content_plan.to_json(plan_path_obj)
            plan_path_str = str(plan_path_obj)
            med, mx = content_plan.overlap_fraction()
            aprint(
                f"Plan: {content_plan.n_boxes} boxes, total budget "
                f"{content_plan.total_budget:,} splats, overlap median "
                f"{med:.0%} / max {mx:.0%} → {plan_path_obj}"
            )
        n_tiles = content_plan.n_boxes
        needs_tiling = n_tiles > 1
        tile_size_resolved = 0  # sentinel; content tasks omit --tile-size
        tile_voxels = max_padded_box_voxels(content_plan)
    else:
        auto_tile = tile_size is None
        if auto_tile:
            if max_shape is None:
                raise typer.BadParameter(
                    "uniform auto tile-size needs a GPU benchmark profile; "
                    "pass --tile-size explicitly (or use --tiling content)."
                )
            max_safe_voxels = math.prod(max_shape) if max_shape else 256**3
            if total_voxels <= max_safe_voxels:
                tile_size = max(spatial) + tile_overlap
            else:
                tile_edge = int(max_safe_voxels ** (1.0 / len(spatial)))
                tile_size = min(tile_edge, max(spatial))
        assert tile_size is not None
        tile_size_resolved = tile_size
        specs = compute_tile_specs(spatial, tile_size, tile_overlap)
        n_tiles = len(specs)
        needs_tiling = n_tiles > 1
        tile_voxels = tile_size ** len(spatial) if needs_tiling else total_voxels

    total_tasks = n_t * n_c * n_tiles

    preset_config = PRESETS.get(fit.preset, PRESETS["standard"])
    n_iters = fit.iters if fit.iters is not None else preset_config.get("n_iters", 3000)
    if throughput_table:
        est_seconds = estimate_tile_wall_seconds(tile_voxels, n_iters, throughput_table)
    else:
        est_seconds = 600.0

    colors_list = (
        [c.strip() for c in merge.channel_colors.split(",")]
        if merge.channel_colors
        else None
    )
    # NOTE: the uniform+per-part-LOD warning fires inside merge_batch_results
    # (the library boundary), so it is NOT emitted here — every caller (Slurm
    # merge, local run, direct API) gets it exactly once at merge time.

    manifest = BatchManifest(
        version=1,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_path=str(input_path.resolve()),
        output_dir=str(output_dir.resolve()),
        array_key=array_key,
        axes=",".join(axes_list) if axes_list else None,
        n_timepoints=n_t,
        n_channels=n_c,
        channel_axes=ome_info.channel_axes,
        channel_shape=ome_info.channel_shape,
        spatial_shape=spatial,
        mode=mode,
        tile_size=tile_size_resolved,
        tile_overlap=tile_overlap,
        n_tiles=n_tiles,
        plan_path=plan_path_str,
        total_tasks=total_tasks,
        preset=fit.preset,
        fit_args=fit_args,
        gpu_name=resolved_gpu,
        estimated_seconds_per_task=est_seconds,
        timepoint_indices=t_indices if timepoints_slice else None,
        channel_indices=c_indices if channels_slice else None,
        channel_colors=colors_list,
        merge_recipe=merge.recipe,
        merge_recipe_args=merge_recipe_args,
        denoise=denoise.denoise,
        denoise_2d=denoise.denoise_2d,
        denoise_h=denoise.denoise_h,
        denoise_patch_size=denoise.patch_size,
        denoise_search_distance=denoise.search_distance,
        denoise_backend=denoise.backend,
        denoise_mode=denoise_mode,
        denoised_zarr_path=denoised_zarr_path,
        calibration_samples=denoise.calibration_samples,
    )

    # Job list: store REAL dataset indices in filenames so status/merge/scripts
    # agree when --timepoints/--channels select non-contiguous values.
    jobs: list[BatchJob] = []
    t_width_base = max(t_indices) + 1
    c_width_base = max(c_indices) + 1
    for task_id in range(total_tasks):
        t_seq = task_id // (n_c * n_tiles)
        r = task_id % (n_c * n_tiles)
        c_seq = r // n_tiles
        k = r % n_tiles
        t_real = t_indices[t_seq]
        c_real = c_indices[c_seq]
        jobs.append(
            BatchJob(
                task_id=task_id,
                timepoint=t_real,
                channel=c_real,
                tile_index=k,
                output_filename=output_filename(
                    t_real,
                    c_real,
                    k,
                    t_width_base,
                    c_width_base,
                    n_tiles,
                    label="box" if mode == "content" else "tile",
                ),
                estimated_wall_seconds=est_seconds,
                channel_coords=decode_flat_channel_index(
                    c_real, ome_info.channel_shape
                ),
            )
        )
    manifest.jobs = jobs

    return PlanResult(
        manifest=manifest,
        content_plan=content_plan,
        tile_voxels=tile_voxels,
        needs_tiling=needs_tiling,
    )
