"""Shared utility helpers for gsplat fit command implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import typer
from arbol import aprint, asection

from ..utils import format_memory_size
from .fitting_recipe_args import (
    build_fit_recipe_params as _build_fit_recipe_params_impl,
)

if TYPE_CHECKING:
    pass


@dataclass
class FitPipelineCtx:
    """CLI parameter state threaded through the ``run_fit_volume`` pipeline.

    ``run_fit_volume`` has ~60 typer parameters; the pipeline helpers below
    each consume overlapping subsets, so the command builds this ctx once
    (right after the tiling strategy is resolved) instead of every helper
    taking a dozen positional arguments. Field values are the command's
    parameters verbatim; ``denoise_effective_h`` is the one field written
    later (by the denoise-calibration step).
    """

    input_path: Path
    output_path: Path
    seeds: Optional[str]
    iters: Optional[int]
    device: Optional[str]
    preset: Optional[str]
    loss: Optional[str]
    config: Optional[Path]
    compress: "Optional[Literal['zip', 'tar.gz']]"
    channel: Optional[int]
    timepoint: Optional[int]
    array_key: Optional[str]
    axes: Optional[str]
    lr: Optional[float]
    floor: Optional[str]
    seed_method: Optional[str]
    verbose: bool
    downscale: Optional[str]
    resolved_tiling: str
    flat: bool
    tile_size: int
    tile_overlap: int
    tile: Optional[str]
    jobs: str
    keep_tiles: bool
    allow_empty_tile: bool
    recipe: Optional[str]
    recipe_n_lods: Optional[int]
    recipe_additive_method: Optional[str]
    recipe_breakpoints: Optional[str]
    recipe_target_ms: Optional[float]
    recipe_bandwidth_mbps: Optional[float]
    recipe_bytes_per_splat: Optional[float]
    recipe_compression_factor: Optional[int]
    recipe_levels: Optional[int]
    recipe_substitutive_method: Optional[str]
    recipe_coarsen_dims: Optional[str]
    cal: Optional[Path]
    k_star_ref: Optional[int]
    n_features_ref: Optional[int]
    feature_threshold: Optional[float]
    feature_metric: Optional[str]
    target_features: Optional[int]
    plan_only: bool
    plan_box: Optional[int]
    progressive: bool
    max_splats_per_pass: int
    psnr_patience: float
    max_passes: Optional[int]
    cull_retention: Optional[float]
    denoise: bool
    denoise_h: Optional[float]
    denoise_2d: bool
    denoise_patch_size: int
    denoise_search_distance: int
    denoise_backend: str
    denoise_effective_h: Optional[float] = None


def warn_ignored_density_flags(ctx: FitPipelineCtx) -> None:
    """Warn about content-density knobs given without ``--tiling content``.

    Content density knobs only apply to content tiling — warn if the
    decomposition didn't resolve to content (e.g. an explicit
    ``--tiling uniform/none``), so the flags aren't silently no-ops.
    """
    if ctx.resolved_tiling != "content" and ctx.plan_box is None:
        _density_flags = [
            name
            for name, on in (
                ("--cal", ctx.cal is not None),
                ("--k-star-ref", ctx.k_star_ref is not None),
                ("--n-features-ref", ctx.n_features_ref is not None),
                ("--feature-threshold", ctx.feature_threshold is not None),
                ("--feature-metric", ctx.feature_metric is not None),
                ("--target-features", ctx.target_features is not None),
            )
            if on
        ]
        if _density_flags:
            aprint(
                f"⚠ {', '.join(_density_flags)} apply only to "
                f"--tiling content; ignored under --tiling {ctx.resolved_tiling}."
            )


# Whole-volume voxel budget for ``--tiling auto``: below this a volume fits
# whole (no tiling), avoiding spurious tile seams on small/medium stacks. ~4x a
# 256^3 tile — a 28M-voxel neuromast stack fits whole; gigavoxel volumes tile.
_AUTO_WHOLE_VOLUME_VOXELS = 64_000_000


def resolve_tiling(
    tiling: str, shape: "tuple[int, ...]", tile_size: int, has_density: bool
) -> str:
    """Resolve ``--tiling`` to a concrete strategy: ``none | uniform | content``.

    ``auto`` fits the whole volume unless it is genuinely large — i.e. some
    dimension exceeds ``tile_size`` AND the total voxel count exceeds
    :data:`_AUTO_WHOLE_VOLUME_VOXELS`. A single dim over ``tile_size`` is not
    enough on its own (that needlessly tiled small stacks and produced visible
    background seams). When tiling IS selected, ``content`` is used when a
    transferable density (``--cal`` / ``--k-star-ref`` / ``--plan``) is
    available, else ``uniform``.
    """
    t = tiling.lower()
    if t not in ("auto", "none", "uniform", "content"):
        raise typer.BadParameter(
            f"--tiling must be one of auto|none|uniform|content, got {tiling!r}"
        )
    if t != "auto":
        return t
    n_voxels = 1
    for s in shape:
        n_voxels *= int(s)
    exceeds_dim = any(int(s) > int(tile_size) for s in shape)
    large = exceeds_dim and n_voxels > _AUTO_WHOLE_VOLUME_VOXELS
    if not large:
        return "none"
    return "content" if has_density else "uniform"


def validate_and_build_recipe(ctx: FitPipelineCtx, volume_ndim: int) -> "Any":
    """Validate per-part ``--recipe`` usage and build its ``RecipeParams``.

    Returns ``None`` when no ``--recipe`` was given. Raises
    ``typer.BadParameter`` for the flag combinations a per-part LOD recipe
    cannot serve (a --flat single leaf, a whole-volume fit, a single-tile
    worker, or the plan-only paths).
    """
    if ctx.recipe is None:
        return None
    if ctx.flat:
        raise typer.BadParameter(
            "--recipe needs a partition output; it is incompatible "
            "with --flat (which merges to a single leaf)."
        )
    if ctx.resolved_tiling == "none":
        raise typer.BadParameter(
            "--recipe needs a tiled fit (--tiling uniform/content); a "
            "whole-volume fit is a single leaf. Run `gsplat lod` on it "
            "instead."
        )
    if ctx.tile is not None:
        raise typer.BadParameter(
            "--recipe is applied when the parts are merged; it cannot "
            "be combined with single-tile --tile (a worker fits one "
            "bare leaf)."
        )
    if ctx.plan_only or ctx.plan_box is not None:
        raise typer.BadParameter(
            "--recipe is incompatible with --plan-only / --plan-box."
        )
    recipe_params = build_fit_recipe_params(
        ctx.recipe,
        n_lods=ctx.recipe_n_lods,
        additive_method=ctx.recipe_additive_method,
        breakpoints=ctx.recipe_breakpoints,
        target_ms=ctx.recipe_target_ms,
        bandwidth_mbps=ctx.recipe_bandwidth_mbps,
        bytes_per_splat=ctx.recipe_bytes_per_splat,
        compression_factor=ctx.recipe_compression_factor,
        levels=ctx.recipe_levels,
        substitutive_method=ctx.recipe_substitutive_method,
        coarsen_dims=ctx.recipe_coarsen_dims,
        device=ctx.device,
        volume_ndim=volume_ndim,
    )
    from luxar.gsplats.lod.recipes import uniform_per_part_lod_warning

    _w = uniform_per_part_lod_warning(ctx.resolved_tiling, ctx.recipe)
    if _w:
        aprint(f"⚠ {_w}")
    return recipe_params


def resolve_denoise_h(ctx: FitPipelineCtx, volume: "Any") -> Optional[float]:
    """Resolve the effective NLM ``h`` (manual value, or auto-calibrated).

    Returns ``None`` when ``--denoise`` is off. The caller stores the result
    on ``ctx.denoise_effective_h`` and then either denoises the full volume
    now (non-tiled fitting, :func:`maybe_denoise_full_volume`) or passes
    ``h`` + params through to ``fit_tile`` (tiled fitting, per-tile denoise).
    """
    if not ctx.denoise:
        return None
    if ctx.denoise_h is not None:
        effective_h = ctx.denoise_h
        aprint(f"Denoise: using manual h={effective_h:.4f}")
        return effective_h
    import torch

    from luxar.gsplats.preprocessing import calibrate_nlm_h
    from luxar.gsplats.preprocessing.denoise_pipeline import (
        normalize_volume,
    )
    from luxar.gsplats.utils.device import resolve_torch_device

    with asection("Calibrating NLM h"):
        norm_vol, _, _ = normalize_volume(volume)
        t_vol = torch.from_numpy(norm_vol)
        # Auto-select CUDA > MPS > CPU when --device is omitted.
        dev = resolve_torch_device(ctx.device) if ctx.device else resolve_torch_device()
        effective_h = calibrate_nlm_h(
            t_vol,
            patch_size=ctx.denoise_patch_size,
            search_distance=ctx.denoise_search_distance,
            backend=ctx.denoise_backend,
            device=dev,
            use_2d_slice=True,
        )
        aprint(f"Calibrated h={effective_h:.4f}")
    return effective_h


def maybe_denoise_full_volume(
    ctx: FitPipelineCtx, volume: "Any", is_tiled: bool
) -> "Any":
    """Denoise the full volume now for the non-tiled paths.

    For tiled paths, denoise is deferred to per-tile (see ``fit_tile``);
    the volume is returned unchanged.
    """
    if ctx.denoise and ctx.denoise_effective_h is not None and not is_tiled:
        from luxar.gsplats.preprocessing.denoise_pipeline import (
            denoise_volume_array,
        )

        with asection("Denoising (NLM)"):
            volume = denoise_volume_array(
                volume,
                h=ctx.denoise_effective_h,
                patch_size=ctx.denoise_patch_size,
                search_distance=ctx.denoise_search_distance,
                backend=ctx.denoise_backend,
                device=ctx.device,
                use_2d=ctx.denoise_2d,
            )
            aprint(f"Denoised volume shape: {volume.shape}")
    return volume


def assemble_fit_config(ctx: FitPipelineCtx, is_tiled: bool) -> "tuple[dict, Any, Any]":
    """Build the merged fit config and parse the seed / downscale options.

    Returns ``(fit_config, parsed_seeds, effective_downscale)``: the merged
    preset/YAML/CLI config dict (with per-tile denoise params injected for
    tiled fitting and ``downscale`` popped out), the parsed ``--seeds``
    value, and the effective downscale (CLI flag winning over YAML/preset).
    """
    from luxar.cli.gsplat_config import load_fit_config, parse_seeds

    # 2. Parse downscale option
    parsed_downscale = None
    if ctx.downscale is not None:
        ds_parts = [int(x.strip()) for x in ctx.downscale.split(",")]
        parsed_downscale = ds_parts[0] if len(ds_parts) == 1 else tuple(ds_parts)

    # 3. Build merged config
    cli_overrides = {
        "n_iters": ctx.iters,
        "device": ctx.device,
        "loss_type": ctx.loss,
        "lr": ctx.lr,
        "floor": ctx.floor,
        "seed_method": ctx.seed_method,
        "verbose": ctx.verbose,
        "cull_retention": ctx.cull_retention,
    }
    fit_config = load_fit_config(ctx.preset, ctx.config, cli_overrides)

    if ctx.preset:
        aprint(f"Preset: {ctx.preset}")
    if ctx.config:
        aprint(f"Config: {ctx.config}")
    aprint(f"Iterations: {fit_config.get('n_iters')}")

    # 4. Parse seeds
    parsed_seeds = parse_seeds(ctx.seeds)
    if parsed_seeds is not None:
        aprint(f"Seeds: {parsed_seeds}")
    else:
        aprint("Seeds: auto")

    # 4b. Inject per-tile denoise params for tiled fitting
    if ctx.denoise and ctx.denoise_effective_h is not None and is_tiled:
        fit_config["_denoise_h"] = ctx.denoise_effective_h
        fit_config["_denoise_params"] = {
            "patch_size": ctx.denoise_patch_size,
            "search_distance": ctx.denoise_search_distance,
            "backend": ctx.denoise_backend,
            "device": ctx.device,
            "use_2d": ctx.denoise_2d,
        }
        aprint(f"Denoise: per-tile on-the-fly (h={ctx.denoise_effective_h:.4f})")

    # 5. Apply downscaling
    # Pop downscale from fit_config to avoid "multiple values" conflict
    # (get_fit_defaults extracts it from the fit_gaussian_splats signature)
    fc_downscale = fit_config.pop("downscale", None)
    # CLI --downscale flag takes priority over YAML/preset config
    effective_downscale = (
        parsed_downscale if parsed_downscale is not None else fc_downscale
    )
    return fit_config, parsed_seeds, effective_downscale


def dispatch_parallel_tiled(
    ctx: FitPipelineCtx,
    volume: "Any",
    fit_config: dict,
    effective_downscale: "Any",
    recipe_params: "Any",
) -> bool:
    """Parallel tiled fitting: spawn one subprocess per tile.

    Branches BEFORE the in-memory downscale in the command body — the parent
    skips the in-memory downscale (it only needs the shape to compute the
    grid); each worker re-invokes ``fit --tile i/M``, loading and downscaling
    its own region and rescaling back to original coords, then we reload +
    merge. (The parent still holds the loaded volume — only its shape is used
    here.) When ``--jobs`` resolves to 1 (e.g. ``-j auto`` on a CPU/MPS box,
    or an explicit ``-j 0/1``), falls through to the in-process sequential
    path instead of spawning a subprocess.

    Returns ``True`` when the parallel path ran to completion (the caller
    raises ``typer.Exit(0)``); ``False`` when the run should fall through to
    the sequential paths.
    """
    tiled = ctx.resolved_tiling == "uniform"
    if not (tiled and ctx.tile is None and ctx.jobs != "1"):
        return False

    import math

    from luxar.gsplats.fit_tiled_parallel import (
        build_worker_cmd,
        fit_tiled_parallel,
        luxar_argv0,
        resolve_jobs,
    )
    from luxar.gsplats.fitting.downscale import normalize_downscale
    from luxar.gsplats.tiling import compute_tile_specs

    # Compute the tile grid on the POST-downscale shape (shape math
    # only — decimation is volume[::f]) so the parent and workers
    # agree on the tile count M.
    ds_factors = (
        normalize_downscale(effective_downscale, volume.ndim)
        if effective_downscale is not None
        else None
    )
    if ds_factors is not None:
        grid_shape = tuple(
            len(range(0, s, f)) for s, f in zip(volume.shape, ds_factors)
        )
    else:
        grid_shape = tuple(volume.shape)

    specs = compute_tile_specs(grid_shape, ctx.tile_size, ctx.tile_overlap)
    n_tiles = len(specs)
    tile_voxels = max((int(math.prod(s.shape)) for s in specs), default=1)

    try:
        n_jobs = resolve_jobs(
            ctx.jobs,
            tile_voxels=tile_voxels,
            num_tiles=n_tiles,
            device=ctx.device,
        )
    except ValueError:
        aprint(f"Error: --jobs must be an integer or 'auto', got '{ctx.jobs}'")
        raise typer.Exit(1)

    # Only spawn workers when there is genuine concurrency to gain.
    # Otherwise (n_jobs == 1) fall through to the sequential tiled
    # path below — no subprocess overhead for a single worker.
    if n_jobs > 1:
        aprint(
            f"Parallel tiled fitting: {n_tiles} tiles, grid={grid_shape}, "
            f"{n_jobs} concurrent worker(s)"
        )

        # Format downscale for worker argv (scalar or per-axis).
        ds_arg: Optional[str] = None
        if effective_downscale is not None:
            if isinstance(effective_downscale, (list, tuple)):
                ds_arg = ",".join(str(int(x)) for x in effective_downscale)
            else:
                ds_arg = str(int(effective_downscale))

        argv0 = luxar_argv0()

        def _worker_cmd(i: int, m: int, out_path: Path) -> list[str]:
            return build_worker_cmd(
                argv0,
                ctx.input_path,
                out_path,
                i,
                m,
                ctx.tile_size,
                ctx.tile_overlap,
                seeds=ctx.seeds,
                iters=ctx.iters,
                device=ctx.device,
                preset=ctx.preset,
                config=ctx.config,
                loss=ctx.loss,
                lr=ctx.lr,
                # Forward the user's SPEC verbatim: each worker resolves it
                # against the same volume with the deterministic sampler, so
                # every worker subtracts one identical level. (An unset floor
                # lets each worker apply its own --config/--preset merge.)
                floor=ctx.floor,
                seed_method=ctx.seed_method,
                downscale=ds_arg,
                channel=ctx.channel,
                timepoint=ctx.timepoint,
                array_key=ctx.array_key,
                axes=ctx.axes,
                progressive=ctx.progressive,
                max_splats_per_pass=ctx.max_splats_per_pass,
                psnr_patience=ctx.psnr_patience,
                max_passes=ctx.max_passes,
                denoise=ctx.denoise,
                denoise_h=ctx.denoise_effective_h,
                denoise_patch_size=ctx.denoise_patch_size,
                denoise_search_distance=ctx.denoise_search_distance,
                denoise_backend=ctx.denoise_backend,
                denoise_2d=ctx.denoise_2d,
                # Empty (windowed-to-zero) tiles must not crash the
                # whole run: the worker writes an .empty marker and
                # exits 0; the orchestrator skips it at merge.
                allow_empty_tile=True,
            )

        tmp_dir = ctx.output_path.parent / f".{ctx.output_path.name}.tiles"
        merge_cull = fit_config.get("cull_retention")

        with asection("Optimization (parallel tiles)"):
            result = fit_tiled_parallel(
                num_tiles=n_tiles,
                jobs=n_jobs,
                tmp_dir=tmp_dir,
                worker_cmd_builder=_worker_cmd,
                volume_shape=grid_shape,
                tile_size=ctx.tile_size,
                overlap=ctx.tile_overlap,
                progressive=ctx.progressive,
                cull_retention=merge_cull,
                verbose=ctx.verbose,
                keep_tiles=ctx.keep_tiles,
                partition=not ctx.flat,
                recipe=ctx.recipe,
                recipe_params=recipe_params,
            )

        with asection(f"Saving to {ctx.output_path.name}"):
            n_splats = save_fit_output(
                result, ctx.output_path, compress=ctx.compress, verbose=ctx.verbose
            )

        aprint(f"\nDone: {n_splats:,} splats")
        return True

    aprint("--jobs resolved to 1 worker; using sequential tiled fitting")
    return False


def fit_single_tile(
    ctx: FitPipelineCtx, volume: "Any", fit_config: dict, parsed_seeds: "Any"
) -> "Any":
    """Single-tile mode (Slurm-ready): fit tile ``--tile N/M`` of the grid."""
    from luxar.gsplats.fit_tiled_gsplats import fit_tile
    from luxar.gsplats.tiling import compute_tile_specs

    assert ctx.tile is not None
    tile_parts = ctx.tile.split("/")
    if len(tile_parts) != 2:
        aprint("Error: --tile must be N/M format (e.g., '3/16')")
        raise typer.Exit(1)
    try:
        tile_idx, tile_total = int(tile_parts[0]), int(tile_parts[1])
    except ValueError:
        aprint("Error: --tile N/M requires integer values")
        raise typer.Exit(1)

    specs = compute_tile_specs(volume.shape, ctx.tile_size, ctx.tile_overlap)
    if tile_total != len(specs):
        aprint(
            f"Note: --tile specifies {tile_total} tiles but "
            f"grid has {len(specs)} tiles for this volume. "
            f"Using actual grid count."
        )
    if tile_idx < 0 or tile_idx >= len(specs):
        aprint(f"Error: tile index {tile_idx} out of range [0, {len(specs)})")
        raise typer.Exit(1)

    # Extract params that are explicit in fit_tile to avoid
    # "got multiple values" conflicts with **fit_config
    fc_voxel_size = fit_config.pop("voxel_size", None)
    fc_output_space = fit_config.pop("output_space", "real")

    # This is the standalone worker's own user-spec entry point: validate the
    # spec (rejecting e.g. a negative --floor, as every other entry point
    # does), then resolve it GUARDED so fit_tile is never handed an unguarded
    # numeric — a too-high explicit floor is warned about and dropped instead
    # of silently erasing the tile. ``None`` in the merged config (a
    # ``floor: null`` YAML) means DISABLED, exactly as on the sequential
    # tiled and non-tiled paths.
    from luxar.gsplats.fitting.preprocessing import resolve_volume_floor
    from luxar.gsplats.fitting.validation import _validate_floor

    floor_spec = fit_config.get("floor", "auto")
    _validate_floor(floor_spec)
    resolved_floor = resolve_volume_floor(volume, floor_spec, guard_numeric=True)
    fit_config["floor"] = resolved_floor if resolved_floor is not None else "none"

    with asection(
        f"Fitting tile {tile_idx}/{len(specs)} grid={specs[tile_idx].grid_index}"
    ):
        return fit_tile(
            volume,
            specs[tile_idx],
            voxel_size=fc_voxel_size,
            output_space=fc_output_space,
            progressive=ctx.progressive,
            max_splats_per_pass=ctx.max_splats_per_pass,
            psnr_patience=ctx.psnr_patience,
            max_passes=ctx.max_passes,
            seeds=parsed_seeds,
            **fit_config,
        )


def fit_sequential_tiled(
    ctx: FitPipelineCtx,
    volume: "Any",
    fit_config: dict,
    parsed_seeds: "Any",
    tiled_downscale_factors: "Any",
    recipe_params: "Any",
) -> "Any":
    """Full (in-process, sequential) tiled fitting."""
    from luxar.gsplats.fit_tiled_gsplats import fit_tiled

    # Extract params that are explicit in fit_tiled to avoid
    # "got multiple values" conflicts with **fit_config
    fc_voxel_size = fit_config.pop("voxel_size", None)
    fc_output_space = fit_config.pop("output_space", "real")
    fc_verbose = fit_config.pop("verbose", True)

    # Partition by default (one part per tile), unless --flat. With
    # --downscale this sequential path rescales a flat merged result
    # back to original coords below, so partition is only offered
    # here when not downscaling (use -j>1 for a downscaled partition,
    # whose workers rescale themselves).
    seq_partition = (not ctx.flat) and tiled_downscale_factors is None
    if (not ctx.flat) and tiled_downscale_factors is not None:
        if ctx.recipe is not None:
            raise typer.BadParameter(
                "--recipe needs a partition, but the sequential tiled "
                "path writes a flat leaf under --downscale. Use -j>1 "
                "(parallel tiles) for a downscaled partition with LOD."
            )
        aprint(
            "Note: --downscale on the sequential tiled path writes a "
            "flat leaf; use -j>1 for a downscaled partition."
        )
    return fit_tiled(
        volume,
        tile_size=ctx.tile_size,
        overlap=ctx.tile_overlap,
        voxel_size=fc_voxel_size,
        output_space=fc_output_space,
        verbose=fc_verbose,
        progressive=ctx.progressive,
        max_splats_per_pass=ctx.max_splats_per_pass,
        psnr_patience=ctx.psnr_patience,
        max_passes=ctx.max_passes,
        seeds=parsed_seeds,
        partition=seq_partition,
        recipe=ctx.recipe,
        recipe_params=recipe_params,
        **fit_config,
    )


def fit_progressive(
    ctx: FitPipelineCtx, volume: "Any", fit_config: dict, parsed_seeds: "Any"
) -> "Any":
    """Progressive fitting: multiple passes on residuals."""
    from luxar.gsplats.fit_progressive_gsplats import (
        fit_progressive_gaussian_splats,
    )

    # max_splats = seeds (total budget), or use seeds as max
    prog_max_splats = (
        parsed_seeds
        if isinstance(parsed_seeds, int)
        else fit_config.pop("seeds", 50000)
    )
    # Map --iters to iters_per_pass for progressive mode
    prog_iters = fit_config.pop("n_iters", 1000)
    # Remove params that progressive handles differently
    fit_config.pop("downscale", None)
    fit_config.pop("seeds", None)
    # voxel_size/output_space are passed through — progressive
    # handles them internally (voxel space for passes, converts final result)

    with asection("Progressive Optimization"):
        return fit_progressive_gaussian_splats(
            volume,
            max_splats=prog_max_splats,
            max_splats_per_pass=ctx.max_splats_per_pass,
            iters_per_pass=prog_iters,
            psnr_patience=ctx.psnr_patience,
            max_passes=ctx.max_passes,
            **fit_config,
        )


def rescale_and_save(
    ctx: FitPipelineCtx, result: "Any", tiled_downscale_factors: "Any"
) -> "tuple[Any, int, bool]":
    """Rescale a downscaled tiled result back to original coords, then save.

    Returns ``(result, n_splats, is_leaf)`` for the command's summary line.
    An empty single-tile result under ``--allow-empty-tile`` writes an
    ``.empty`` marker instead of a store (the gsplats writer enforces a
    no-empty policy; the parallel orchestrator skips the marker at merge).
    """
    # Rescale tiled results back to original coordinates if downscaled
    if tiled_downscale_factors is not None and result.n_splats > 0:
        from luxar.gsplats.fitting.downscale import (
            rescale_centers,
            rescale_cholesky_packed,
        )
        from luxar.gsplats.gsplat_data import GSplatData

        result = GSplatData(
            centers=rescale_centers(result.centers, tiled_downscale_factors),
            amplitudes=result.amplitudes,
            cholesky_factors=rescale_cholesky_packed(
                result.cholesky_factors, tiled_downscale_factors
            ),
            colors=result.colors,
            stats=result.stats,
        )
        aprint(f"Rescaled {result.n_splats} splats to original coordinates")

    # 7. Save
    from luxar.gsplats.gsplat_data import GSplatData

    is_leaf = isinstance(result, GSplatData)
    with asection(f"Saving to {ctx.output_path.name}"):
        if (
            is_leaf
            and ctx.allow_empty_tile
            and ctx.tile is not None
            and result.n_splats == 0
        ):
            # Empty tile (windowed to near-zero signal): the gsplats
            # writer enforces a no-empty policy, so instead of erroring
            # we drop an .empty marker that the parallel orchestrator
            # treats as a legitimately-skipped tile at merge time.
            marker = Path(str(ctx.output_path) + ".empty")
            marker.write_text("0 splats\n")
            aprint("Empty tile (0 splats): wrote marker, skipped save")
            n_splats = 0
        else:
            # leaf → .save; partition node → write_gsplats_tree
            n_splats = save_fit_output(
                result, ctx.output_path, compress=ctx.compress, verbose=ctx.verbose
            )
    return result, n_splats, is_leaf


def build_fit_recipe_params(
    recipe: str,
    *,
    n_lods: Optional[int],
    additive_method: Optional[str],
    breakpoints: Optional[str],
    target_ms: Optional[float] = None,
    bandwidth_mbps: Optional[float] = None,
    bytes_per_splat: Optional[float] = None,
    compression_factor: Optional[int],
    levels: Optional[int],
    substitutive_method: Optional[str],
    coarsen_dims: Optional[str],
    device: Optional[str],
    volume_ndim: int,
) -> "Any":
    """Build optional fit-time LOD recipe parameters from CLI arguments."""
    return _build_fit_recipe_params_impl(
        recipe,
        n_lods=n_lods,
        additive_method=additive_method,
        breakpoints=breakpoints,
        target_ms=target_ms,
        bandwidth_mbps=bandwidth_mbps,
        bytes_per_splat=bytes_per_splat,
        compression_factor=compression_factor,
        levels=levels,
        substitutive_method=substitutive_method,
        coarsen_dims=coarsen_dims,
        device=device,
        volume_ndim=volume_ndim,
    )


def save_fit_output(
    result: Any,
    output_path: Path,
    *,
    compress: "Optional[Literal['zip', 'tar.gz']]",
    verbose: bool,
) -> int:
    """Save a flat ``GSplatData`` leaf or a ``kind=partition`` tree node.

    Returns the splat count for the summary line.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    if isinstance(result, GSplatData):
        result.save(output_path, compress=compress)
        n = int(result.n_splats)
    else:  # a partition / tree node has no flat-matrix equivalent
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        write_gsplats_tree(output_path, result, compress=compress)
        n = int(getattr(result, "n_splats", 0))
    if verbose:
        aprint(f"Saved {n:,} splats")
        if output_path.exists():
            aprint(f"File size: {format_memory_size(output_path.stat().st_size)}")
    return n
