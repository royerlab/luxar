"""Shared utility helpers for gsplat fit command implementation."""

from __future__ import annotations

import math
import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import typer
from arbol import aprint, asection

from ...utils import format_memory_size
from .recipe_args import (
    build_fit_recipe_params as _build_fit_recipe_params_impl,
)

if TYPE_CHECKING:
    pass


CONTENT_UNSUPPORTED_FIT_FLAGS = ("--denoise", "--downscale", "--progressive")


def _invocation_token() -> str:
    """Unique per-invocation token for parallel-fit staging paths.

    host+pid alone can collide — two threads in one process share a pid, and
    containers with identical (user-set) hostnames and PID-namespaced pids can
    mount the same output directory — so a random component guarantees the
    token is unique per invocation. host+pid are kept for diagnosability
    (a retained staging dir names the run that produced it).
    """
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _parallel_staging_dir(output_path: Path, token: str) -> Path:
    """Per-invocation staging dir for the parallel tiled fit.

    ``fit_tiled_parallel`` clean-slates (``rmtree`` + ``mkdir``) whatever
    ``tmp_dir`` it is handed, so a directory derived only from the output
    path would let two concurrent ``fit -j`` runs to the SAME output delete
    each other's in-progress tiles. Appending a unique per-invocation
    ``token`` gives each invocation its own dir, so it only ever cleans its
    OWN tiles.
    """
    return output_path.parent / f".{output_path.name}.tiles.{token}"


@dataclass
class FitPipelineCtx:
    """CLI parameter state threaded through the ``run_fit_volume`` pipeline.

    ``run_fit_volume`` has ~60 typer parameters; the pipeline helpers below
    each consume overlapping subsets, so the command builds this ctx once
    (right after the tiling strategy is resolved) instead of every helper
    taking a dozen positional arguments. Field values are the command's
    parameters verbatim; ``denoise_effective_h`` and ``denoise_norm_range``
    are written later (by the denoise-calibration step,
    :func:`resolve_denoise_h`).
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
    norm_range: "Optional[tuple[float, float]]"
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
    recipe_compression_factor: Optional[int]
    recipe_levels: Optional[int]
    recipe_substitutive_method: Optional[str]
    recipe_coarsen_dims: Optional[str]
    recipe_refine: Optional[str]
    recipe_refine_iters: Optional[int]
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
    denoise_norm_range: "Optional[tuple[float, float]]" = None


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


def validate_and_build_recipe(
    ctx: FitPipelineCtx, volume_ndim: int, volume: "Any" = None
) -> "Any":
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
        compression_factor=ctx.recipe_compression_factor,
        levels=ctx.recipe_levels,
        substitutive_method=ctx.recipe_substitutive_method,
        coarsen_dims=ctx.recipe_coarsen_dims,
        refine=ctx.recipe_refine,
        refine_iters=ctx.recipe_refine_iters,
        # `refine="volume"` re-fits against the array being fitted. Unlike
        # `gsplat lod --target`, no path and no axis map are needed: the volume is
        # already in hand and the fit emits splats in its own voxel frame, so the
        # identity map is correct by construction.
        volume=volume,
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

    The non-tiled path then resolves the background floor on DENOISED data as a
    matter of ordering (the fit sees the denoised volume). The uniform tiled
    paths get there by correcting the whole-volume level onto the denoised basis
    with a bounded probe
    (:func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_floor_denoised`,
    #1178). A volume ABOVE the probe budget keeps its raw-basis level under the
    default ``--floor auto`` because the histogram-mode shift is not measurable
    on a bounded crop; a ``pNN`` spec is corrected there, and a failed probe
    keeps the raw level with a note. Uniform batch plans defer that ``pNN``
    correction until calibrated ``h`` exists, while uniform preprocess-mode plans
    resolve every volume-derived spec directly from the selected denoised store.
    Content plans need the floor while placing boxes and therefore keep the
    plan-time raw-basis resolution.
    """
    if not ctx.denoise:
        return None
    if ctx.denoise_h is not None:
        effective_h = ctx.denoise_h
        # Record the WHOLE-volume range so per-tile normalization matches the
        # scale h was chosen for (a fixed h is not scale-invariant).
        ctx.denoise_norm_range = (float(volume.min()), float(volume.max()))
        aprint(f"Denoise: using manual h={effective_h:.4f}")
        return effective_h
    import torch

    from luxar.gsplats.preprocessing import calibrate_nlm_h
    from luxar.gsplats.preprocessing.denoise_pipeline import (
        normalize_volume,
    )
    from luxar.gsplats.utils.device import resolve_torch_device

    with asection("Calibrating NLM h"):
        # Capture the global range h is calibrated against so every tile
        # normalizes against EXACTLY that range (scale-consistent smoothing).
        norm_vol, _gmin, _gmax = normalize_volume(volume)
        ctx.denoise_norm_range = (_gmin, _gmax)
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

    Ordering note: the non-tiled fit therefore resolves its ``--floor`` from the
    DENOISED volume (``_normalize_data`` runs on what this returns), always and
    exactly. The tiled paths cannot reorder the passes that way, so they aim at
    the same basis with a denoise-corrected whole-volume estimator instead
    (#1178) — exactly within the probe budget, and above it only for a ``pNN``
    spec (see :func:`resolve_denoise_h`).
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
        "norm_range": ctx.norm_range,
        "seed_method": ctx.seed_method,
        "verbose": ctx.verbose,
        "cull_retention": ctx.cull_retention,
    }
    fit_config = load_fit_config(ctx.preset, ctx.config, cli_overrides)

    # Progressive mode disables relocation by design (see fit_progressive_gsplats:
    # each pass seeds at residual peaks, no measured quality benefit). The general
    # single-pass fit carries enable_dynamic_ops=True as its default, so it must
    # not leak in here and silently flip relocation ON for a progressive fit. This
    # single choke point covers every progressive CLI path (non-tiled, tiled,
    # tile-workers). Direct Python callers of fit_progressive_gaussian_splats can
    # still pass enable_dynamic_ops themselves.
    if ctx.progressive:
        # Only warn if the user EXPLICITLY set the key (in --config YAML); the
        # inherited single-pass default is dropped silently. Mirrors the
        # "⚠ --seeds is ignored ..." notices. Guarded so a bad/parse-time YAML
        # never breaks the fit over a status notice.
        if ctx.config is not None:
            try:
                from luxar.cli.gsplat_config import _load_yaml_config

                raw_yaml = _load_yaml_config(ctx.config)
                if "enable_dynamic_ops" in raw_yaml:
                    aprint(
                        "⚠ enable_dynamic_ops is ignored with --progressive; "
                        "progressive passes disable relocation by design."
                    )
            except Exception:
                pass
        fit_config.pop("enable_dynamic_ops", None)

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
            "norm_range": ctx.denoise_norm_range,
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


def split_seeds_across_tiles(
    parsed_seeds: "int | float | None",
    n_tiles: int,
    *,
    grid_tiles: "int | None" = None,
) -> "int | float | None":
    """Split a whole-volume ``--seeds`` budget across a tiled fit's tiles.

    A default ``luxar gsplat cal`` reports a WHOLE-VOLUME K*, and the documented
    pipeline is ``cal`` -> ``fit --seeds K*``. A tiled fit hands the same ``seeds``
    value to EVERY tile, so an undivided integer budget realizes roughly
    ``K x n_tiles`` splats (issue #1556: ``--seeds 256000`` on a volume
    auto-tiled into 21 tiles produced 2.4M splats). An integer ``--seeds K`` is
    therefore a whole-volume budget: each of ``n_tiles`` non-empty tiles is
    seeded with ``ceil(K / n_tiles)``, so the budget is divided across the
    tiles instead of being multiplied by them.

    ``ceil`` itself guarantees at least one seed per tile for any positive K, so
    ``0 < K < n_tiles`` gives 1 per tile and realizes ``n_tiles``, not ``K``.
    Empty tiles are excluded from the divisor using the fitter's
    floor-subtracted, Hann-windowed predicate. The scan deliberately does not
    replay optional per-tile denoising, so denoising can still skip a tile that
    the budget scan counted.

    A **non-positive** K is returned UNCHANGED: it is invalid input, and the
    fitter rejects it with "seeds as int must be positive" exactly as it does on
    the whole-volume path — rounding it up to 1 would swallow that error and
    silently fit.

    A ``float`` ratio (the ``0 < r <= 1`` compression-ratio form) is returned
    UNCHANGED. A ratio is a fraction of the voxels it is applied to, so it is
    already scale-free — per tile it means exactly the density it means
    whole-volume — and must never be divided by the tile count. ``None``
    (auto) is likewise unchanged. A non-positive ``n_tiles`` is a defensive
    no-op. When one non-empty tile survives a larger grid, K is unchanged but
    the sparse-grid divisor is still announced.
    """
    if not isinstance(parsed_seeds, int) or isinstance(parsed_seeds, bool):
        return parsed_seeds
    if n_tiles <= 0 or parsed_seeds <= 0:
        return parsed_seeds
    # Integer ceiling division (not math.ceil on a quotient): a budget is an
    # arbitrary-precision Python int, and going through a float would round
    # wrong above 2**53 (and raise OverflowError on an absurdly large one).
    per_tile = parsed_seeds if n_tiles == 1 else -(-parsed_seeds // n_tiles)
    if n_tiles > 1 or (grid_tiles is not None and grid_tiles > n_tiles):
        tile_label = "tile" if n_tiles == 1 else "tiles"
        aprint(
            f"Seeds: {parsed_seeds:,} whole-volume budget -> {per_tile:,} per tile "
            f"across {n_tiles} non-empty {tile_label}"
            + (
                f" ({grid_tiles} grid tiles)"
                if grid_tiles is not None and grid_tiles != n_tiles
                else ""
            )
        )
    return per_tile


def announce_seed_split_lower_bound(
    parsed_seeds: "int | float | None", n_tiles: int
) -> None:
    """Announce the conservative split when plan time cannot inspect content."""
    if (
        not isinstance(parsed_seeds, int)
        or isinstance(parsed_seeds, bool)
        or parsed_seeds <= 0
        or n_tiles <= 1
    ):
        return
    per_tile = -(-parsed_seeds // n_tiles)
    aprint(
        f"Seeds: {parsed_seeds:,} whole-volume budget -> at least "
        f"{per_tile:,} per non-empty tile across {n_tiles} grid tiles "
        "(each worker resolves the exact non-empty count)"
    )


def _needs_nonempty_tile_scan(parsed_seeds: "int | float | None", n_tiles: int) -> bool:
    """Whether an integer budget will actually be divided across this grid."""
    return (
        isinstance(parsed_seeds, int)
        and not isinstance(parsed_seeds, bool)
        and parsed_seeds > 0
        and n_tiles > 1
    )


def reject_rescaled_volume_refit(
    recipe_params: "Any",
    effective_downscale: "Any",
    *,
    voxel_size: "Any" = None,
    output_space: "Any" = "real",
) -> None:
    """Refuse ``--refine volume`` when the tile grid and the splats differ in frame.

    A per-part volume re-fit crops the source to the part's own tile — the cell
    of the partition's ``bsp_tree`` — and uses that cell as VOXEL INDICES into
    the volume. That only holds while the tile grid and the splats share a
    coordinate frame. Two independent factors break it, and they are exactly the
    two :func:`~luxar.gsplats.tiling.resolve_grid_scale` reconciles for the
    split planes (#1587):

    * ``--downscale``: the grid is computed on the DOWNSCALED shape (so the
      parent and its workers agree on the tile count) while every worker
      rescales its splats back to full resolution.
    * a ``voxel_size`` with ``output_space="real"`` (the default): the grid is
      in voxels while every tile's splats are offset by ``origin * voxel_size``
      and emitted in PHYSICAL units.

    Either way each crop is a factor off and in the wrong place, and the
    never-worse guard cannot tell — it compares against that same wrong crop.
    (Physical-unit centers additionally defeat the re-fit itself: it renders on
    an origin-anchored voxel grid, which is what
    :class:`~luxar.gsplats.lod.volume_refit.VolumeRefitConfig`'s
    ``frame_tolerance`` heuristic exists to notice after the fact.) Checked
    before any fitting, since the alternative is discovering it after the whole
    fit.

    Called with the RESOLVED values, so a factor coming from ``--config`` /
    ``--preset`` is caught as well as the flag. The sequential tiled path
    refuses ``--recipe`` outright under ``--downscale`` (it writes a flat leaf
    there), so ``-j>1`` is what makes that combination otherwise reachable; a
    ``voxel_size`` is reachable on both the sequential and the parallel
    partition path. It is checked on the NON-tiled path too, where there are no
    parts to crop but the re-fit still renders on the source's voxel grid, so a
    physical-unit ladder can only ever be discarded — hence the message speaks
    of the crop's frame rather than of tiles.
    """
    if recipe_params is None or getattr(recipe_params, "refine", "none") != "volume":
        return
    if effective_downscale is not None:
        raise typer.BadParameter(
            "--refine volume cannot be combined with --downscale: the tile grid "
            "is computed on the downscaled shape while the fitted splats are "
            "rescaled back to full resolution, so each tile's crop of the volume "
            "would land in the wrong place. Drop --downscale, or use --refine l2."
        )
    if voxel_size is not None and output_space == "real":
        spacing = (
            [float(voxel_size)]
            if isinstance(voxel_size, (int, float))
            else [float(v) for v in voxel_size]
        )
        if any(v != 1.0 for v in spacing):
            raise typer.BadParameter(
                f"--refine volume cannot be combined with a real-space "
                f"voxel_size ({spacing} with output_space='real'): the re-fit "
                "crops and renders on the source's VOXEL grid (per part, from "
                "the tile grid, when there are parts) while the fitted splats "
                "are in physical units, so every crop would be a factor off. "
                "Set output_space: voxel in the config, drop voxel_size, or "
                "use --refine l2."
            )


def dispatch_parallel_tiled(
    ctx: FitPipelineCtx,
    volume: "Any",
    fit_config: dict,
    effective_downscale: "Any",
    recipe_params: "Any",
) -> bool:
    """Parallel tiled fitting: spawn one subprocess per tile.

    Branches BEFORE the in-memory downscale in the command body. Each worker
    re-invokes ``fit --tile i/M``, loading and downscaling the whole grid before
    selecting its tile and rescaling the fitted splats to original coordinates;
    the parent then reloads and merges the outputs. Once concurrency is confirmed,
    the parent also decimates one matching reference for whole-merge quality
    scoring. When ``--jobs``
    resolves to 1 (e.g. ``-j auto`` on a CPU/MPS box, or an explicit ``-j
    0/1``), it falls through without materializing that reference so the
    in-process sequential path performs the downscale only once.

    Returns ``True`` when the parallel path ran to completion (the caller
    raises ``typer.Exit(0)``); ``False`` when the run should fall through to
    the sequential paths.
    """
    tiled = ctx.resolved_tiling == "uniform"
    if not (tiled and ctx.tile is None and ctx.jobs != "1"):
        return False

    import numpy as np

    from luxar.gsplats.fit_tiled_parallel import (
        build_worker_cmd,
        fit_tiled_parallel,
        luxar_argv0,
        report_auto_jobs,
        resolve_jobs_with_limit,
    )
    from luxar.gsplats.fitting.downscale import downscale_volume, normalize_downscale
    from luxar.gsplats.tiling import compute_tile_specs, resolve_grid_scale

    # Compute the tile grid on the POST-downscale shape without materializing
    # the decimated reference yet, so a one-job fallback does not downscale the
    # volume twice. Decimation is volume[::f], so shape math is exact here.
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
        worker_limit = resolve_jobs_with_limit(
            ctx.jobs,
            tile_voxels=tile_voxels,
            num_tiles=n_tiles,
            device=ctx.device,
        )
        n_jobs = worker_limit.count
    except ValueError:
        aprint(f"Error: --jobs must be an integer or 'auto', got '{ctx.jobs}'")
        raise typer.Exit(1)

    report_auto_jobs(ctx.jobs, worker_limit)

    # Only spawn workers when there is genuine concurrency to gain.
    # Otherwise (n_jobs == 1) fall through to the sequential tiled
    # path below — no subprocess overhead for a single worker.
    if n_jobs > 1:
        reference_volume = (
            np.ascontiguousarray(downscale_volume(volume, ds_factors))
            if ds_factors is not None
            else volume
        )
        aprint(
            f"Parallel tiled fitting: {n_tiles} tiles, grid={grid_shape}, "
            f"{n_jobs} concurrent worker(s)"
        )
        # Announce a conservative seed split HERE, in the parent. Each worker
        # resolves the exact non-empty count itself, but workers run under
        # subprocess.run(capture_output=True), so on success their stdout is
        # discarded and the user would only ever see the parent's undivided
        # "Seeds: K" line. The raw whole-volume count is still forwarded below;
        # dividing it here as well would double-divide in each worker.
        from luxar.cli.gsplat_config import parse_seeds

        parent_seeds = parse_seeds(ctx.seeds)
        if _needs_nonempty_tile_scan(parent_seeds, n_tiles):
            announce_seed_split_lower_bound(parent_seeds, n_tiles)

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
                # The RAW whole-volume --seeds string, on purpose: each worker
                # re-enters `fit --tile i/M` and splits it per tile itself (see
                # split_seeds_across_tiles in fit_single_tile). Dividing here
                # too would double-divide the budget.
                seeds=ctx.seeds,
                iters=ctx.iters,
                device=ctx.device,
                preset=ctx.preset,
                config=ctx.config,
                loss=ctx.loss,
                lr=ctx.lr,
                # Forward the user's SPEC verbatim: for auto/pNN each worker
                # resolves it against the same volume with the deterministic
                # sampler, so every worker subtracts one identical level. (An
                # unset floor lets each worker apply its own --config/--preset
                # merge.) A user NUMERIC passes straight through to the worker,
                # which now applies it unvetoed — see fit_single_tile: a numeric
                # above the volume's max windows every tile to zero, warned about
                # per tile rather than silently ignored as it once was.
                floor=ctx.floor,
                norm_range=ctx.norm_range,
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

        # Per-invocation staging (unique token): so concurrent `fit -j`
        # runs targeting one output can't clobber each other's in-progress
        # tiles — the helper clean-slates only its OWN unique dir.
        tmp_dir = _parallel_staging_dir(ctx.output_path, _invocation_token())
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
                # The grid above is in DOWNSCALED voxels while every worker
                # rescales its splats back to full resolution AND (with a
                # voxel_size from --config, unless output_space is "voxel")
                # emits physical coordinates. Both factors compose, so the
                # merge needs their product to place the partition's split
                # planes in the splats' own frame (#1587).
                grid_scale=resolve_grid_scale(
                    volume.ndim,
                    downscale_factors=ds_factors,
                    voxel_size=fit_config.get("voxel_size"),
                    output_space=fit_config.get("output_space", "real"),
                ),
                # What the merged result is a representation OF. The workers hold
                # the volume, so this process is the only one that can say: the
                # grid above is post-downscale, and the stored element type is
                # gone by the time `load_volume` has handed back float32. Under
                # --downscale the acquisition grid is declared; without one the
                # two grids agree and there is nothing to declare.
                source_shape=(
                    [int(s) for s in volume.shape] if ds_factors is not None else None
                ),
                source_dtype=fit_config.get("source_dtype"),
                volume=reference_volume,
                device=ctx.device,
            )

        with asection(f"Saving to {ctx.output_path.name}"):
            n_splats = save_fit_output(
                result, ctx.output_path, compress=ctx.compress, verbose=ctx.verbose
            )

        aprint(f"\nDone: {n_splats:,} splats")
        return True

    aprint("--jobs resolved to 1 worker; using sequential tiled fitting")
    return False


def validate_floor_spec(floor_spec: "str | float | None") -> None:
    """Reject a malformed/negative floor spec as a clean usage error.

    Meant to run BEFORE any volume is touched, so a typo (``--floor potato``) or
    an out-of-range percentile (``--floor p150``) costs no read, and surfaces as a
    :class:`typer.BadParameter` (which Typer renders as a usage error) rather than
    a bare ``ValueError`` traceback from deep inside the fit. Numeric specs are
    validated too — a ``floor: -5.0`` in a YAML config would otherwise reach a
    worker's argv as ``--floor -5.0``, which click parses as an option, not a
    value.
    """
    from luxar.gsplats.fitting.validation import _validate_floor

    try:
        _validate_floor(floor_spec)
    except ValueError as exc:
        raise typer.BadParameter(f"--floor: {exc}") from exc


def floor_spec_needs_volume(floor_spec: "str | float | None") -> bool:
    """Whether resolving this ``--floor`` spec has to read the volume.

    ``auto`` / ``pNN`` are volume-derived; ``none`` / ``None`` / a numeric spec
    are already concrete, so a caller that would have to LOAD data purely to
    resolve them can skip the load entirely.
    """
    if not isinstance(floor_spec, str):
        return False
    f = floor_spec.strip().lower()
    return f == "auto" or f.startswith("p")


def _calibration_floor_level(cal: "Path") -> "Optional[float]":
    """The CONCRETE level a ``cal.json`` records having subtracted, if any.

    ``None`` means "this calibration has nothing usable to say" and covers five
    cases deliberately treated alike: the file is unreadable, it predates the
    stamp, it records ``null``, it records something that is not a number at all
    (a hand-edited ``floor_subtracted: "auto"``), or it records a non-finite one
    (``json`` round-trips ``NaN``/``Infinity`` happily, while ``--floor`` refuses
    them — so forwarding one would abort the fit with an error about a flag the
    user never passed).

    A recorded ``null`` is NOT adopted as ``--floor none``. ``cal`` writes it
    both when the user asked for ``none`` and when its own too-high guard
    REFUSED the level (see ``gsplats/calibration/driver.py``), and the file
    cannot distinguish the two — so honouring it would silently disable the
    ``auto`` default on the strength of a guard that fired. A malformed file is
    not diagnosed here either: the density resolver opens the same path and
    reports it properly.
    """
    from luxar.gsplats.calibration import CalibrationResult

    try:
        fit_config = CalibrationResult.from_json(cal).fit_config or {}
        level = fit_config.get("floor_subtracted")
        if level is None:
            return None
        # float() inside the try on purpose: a hand-edited "auto" would
        # otherwise raise a bare ValueError out of a helper whose whole contract
        # is to say nothing when it has nothing to say.
        value = float(level)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _config_pins_floor(config: "Optional[Path]") -> bool:
    """Whether a YAML ``--config`` states a ``floor:`` of its own."""
    if config is None:
        return False
    from luxar.cli.gsplat_config import _load_yaml_config

    return "floor" in _load_yaml_config(config)


def resolve_floor_with_calibration(
    cal: "Optional[Path]",
    floor: "Optional[str]",
    config: "Optional[Path]",
    *,
    tiling: str,
    verbose: bool = True,
) -> "Optional[str]":
    """The ``--floor`` spec to fit with, adopting a ``--cal``'s level (#1175).

    ``gsplat cal`` subtracts a floor ONCE up front and measures the density's
    ``feature_threshold`` on that floor-suppressed volume, recording the level
    it used in ``fit_config.floor_subtracted``. Nothing consumed it: a
    ``fit --tiling content --cal cal.json`` re-derived its own floor, so a
    ``cal --floor p20`` was followed by a fit that scanned and fitted at
    ``auto`` — the density's threshold and the volume it is applied to on two
    different scales.

    ``tiling`` is the RESOLVED decomposition, and anything but ``content`` is a
    no-op: that is the one mode which honours ``--cal`` at all, and it is
    required rather than defaulted, because the permissive value is the one
    that CHANGES the fit — a caller who forgot it would adopt the calibration
    floor under a decomposition that ignores everything else about the cal. And
    :func:`warn_ignored_density_flags` announces the flag as ignored under every
    other one. Silently changing the floor from a flag the CLI has just called
    ignored would be the command contradicting itself.

    Only fills a gap, never overrides. ``--floor`` is a Typer option whose
    default is ``None`` (NOT ``"auto"`` — the "unset" state is representable),
    so an explicit ``--floor auto`` still means the user asked for auto and
    wins; a ``floor:`` in a YAML ``--config`` wins too. No preset sets a floor,
    so a preset can never shadow this.

    Only a CONCRETE non-negative level is adopted — see
    :func:`_calibration_floor_level` for why a recorded ``null`` is silence
    rather than ``"none"``. A NEGATIVE recorded level (dark-frame-corrected
    data) is declined out loud: ``--floor`` cannot express it, and forwarding
    one would abort the fit.

    NOTE the adopted level is ABSOLUTE, in the volume's own units. Reusing one
    ``cal.json`` across a timelapse therefore applies timepoint 0's pedestal to
    every timepoint, where the ``auto`` default re-estimates per volume — which
    is the point when the pedestal is an instrument offset, and wrong when it
    drifts. Pass ``--floor auto`` to opt back out.

    Returns ``floor`` unchanged whenever the calibration has nothing to add.
    """
    if (
        tiling != "content"
        or cal is None
        or floor is not None
        or _config_pins_floor(config)
    ):
        return floor
    level = _calibration_floor_level(cal)
    if level is None:
        return floor
    if level < 0.0:
        if verbose:
            aprint(
                f"⚠ {cal.name} recorded a negative floor ({level:.6g}); "
                "--floor cannot express it, so this fit resolves its own."
            )
        return floor
    spec = repr(level)
    if verbose:
        aprint(
            f"Floor from calibration: --floor {spec} (recorded by {cal.name} as "
            "floor_subtracted; its density was calibrated on that scale). "
            "The level is absolute, so re-calibrate rather than reusing this "
            "cal.json on a volume with a different pedestal. Pass --floor "
            "explicitly to override."
        )
    return spec


def resolve_shared_floor(
    volume: "Any",
    floor_spec: "str | float | None",
    *,
    guard_numeric: bool = True,
    scope: str = "every tile",
    verbose: bool = True,
) -> "tuple[float | None, str | float]":
    """Resolve a user ``--floor`` spec ONCE into the level every worker subtracts.

    The multi-worker counterpart of what :func:`luxar.gsplats.fit_tiled_gsplats.fit_tiled`
    does inline: a spec is resolved against the WHOLE ``volume`` (via
    :func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_floor`, a bounded
    deterministic sample — never a full ``np.percentile``) so independent
    consumers — content boxes, ``-j`` box subprocesses, every batch ``(t, c)``
    task — all subtract one identical pedestal instead of each re-estimating its
    own.

    The level is resolved on the RAW volume. ``--tiling content`` is unaffected
    because it ignores ``--denoise`` outright (warned about in ``fit``).
    ``batch-fit`` under ``--denoise`` does NOT get a denoised-basis level in
    either mode, and that is a known gap rather than a covered case: the plan
    resolves the level from the RAW ``input_path`` (``batch/planning.py``,
    :func:`~luxar.cli.gsplat_ops.batch.planning._resolve_and_record_floor`)
    before the denoise job has written anything, and the default on-the-fly mode
    then forwards that raw-basis NUMBER to tasks which denoise each tile
    themselves — a numeric spec being precisely what
    :func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_floor_denoised`
    passes through uncorrected. Out of scope for #1178 and tracked separately.
    The paths that DO resolve on the denoised basis — uniform tiling, sequential
    and ``-j``/``--tile k/M`` alike (for any volume-derived spec within the
    denoise probe's budget, and above it for a ``pNN`` spec only) — call
    :func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_floor_denoised`
    instead of this function (see :func:`fit_single_tile` and
    :func:`luxar.gsplats.fit_tiled_gsplats.fit_tiled`). Add
    ``denoise_h``/``denoise_params`` passthrough here if a consumer of this
    function ever gains denoising.

    Parameters
    ----------
    volume
        The volume the level is a property of. May be ``None`` when
        :func:`floor_spec_needs_volume` is ``False`` (nothing is read).
    guard_numeric
        Apply the "floor >= max would erase all signal" guard to a NUMERIC spec
        too (one bounded read). ``True`` where a user spec first becomes a level;
        ``False`` for a level a parent already resolved and guarded.
    scope
        Phrase naming who subtracts it, for the log line ("every box", ...).

    Returns
    -------
    (level, forward)
        ``level`` is the concrete level to subtract locally (``None`` = disabled,
        or the guard refused it). ``forward`` is what to hand a worker — the same
        number, the string ``"none"``, or, for the rare NEGATIVE resolved level
        (dark-frame-corrected data), the original spec: neither ``--floor`` nor
        ``fit_gaussian_splats`` accepts a negative level, so that one case keeps
        forwarding the spec exactly as before this function existed — and, being
        a spec again, it is re-resolved wherever it lands:

        * ``fit --tiling uniform -j N`` / ``--tile k/M``: exact. Each worker
          re-resolves the spec against the SAME whole volume with the same
          deterministic sampler, so they all reach the same level.
        * ``fit --tiling content``: DEGENERATES to per-box resolution. The spec
          goes into ``box_fit_kwargs["floor"]`` and reaches
          ``fit_gaussian_splats(crop, floor=<spec>)`` per box, which resolves it
          against that BOX CROP — i.e. the #1174 per-box pedestal, and a
          violation of :func:`~luxar.gsplats.planner.fit_planned.fit_planned`'s
          "must already be a CONCRETE level" contract. It is accepted only
          because refusing would make dark-frame-corrected data unfittable.
        * ``batch-fit``: each ``(t, c)`` task resolves it on its own timepoint, so
          pedestals may differ across the run; that is said loudly at plan time
          and no level is recorded in the manifest (see
          :func:`luxar.cli.gsplat_ops.batch.planning.resolve_batch_floor`).
    """
    from luxar.gsplats.fitting.preprocessing import resolve_volume_floor
    from luxar.gsplats.fitting.validation import _validate_floor

    if isinstance(floor_spec, str):
        _validate_floor(floor_spec)  # reject a malformed/negative USER spec early
    if volume is None:
        # Nothing to sample: a concrete spec resolves without data (the caller
        # checked `floor_spec_needs_volume`), so the guard has to be skipped.
        guard_numeric = False
    level = resolve_volume_floor(volume, floor_spec, guard_numeric=guard_numeric)
    if level is None:
        # Disabled, resolved to 0 (the "0 disables" rule), or refused by the
        # guard — all three mean "subtract nothing", which is what the workers
        # must be told explicitly so they don't re-resolve the spec themselves.
        return None, "none"
    if level < 0.0:
        aprint(
            f"Note: resolved background floor {level:.6g} is negative and cannot "
            f"be forwarded as --floor; {scope} resolves the spec itself."
        )
        # `floor_spec` cannot be None here: a None spec resolves to level None
        # and already returned above.
        assert floor_spec is not None
        return level, floor_spec
    if verbose:
        aprint(
            f"Floor suppression: {scope} subtracts background level {level:.6g} "
            f"(resolved once for the whole volume)"
        )
    return level, level


def _tile_worker_label(ctx: "Any", tile_idx: int, n_tiles: int) -> str:
    """Name this worker's sub-volume for a diagnostic: ``tile 3/16 (t=7, c=1)``."""
    label = f"tile {tile_idx}/{n_tiles}"
    t_idx = getattr(ctx, "timepoint", None)
    c_idx = getattr(ctx, "channel", None)
    if t_idx is not None or c_idx is not None:
        label += f" (t={t_idx}, c={c_idx})"
    return label


def warn_if_level_erases_volume(volume: "Any", level: float, *, what: str) -> None:
    """Loudly flag a concrete floor level that clips ``what``'s volume to zero.

    A concrete numeric ``--floor`` reaching a worker is applied UNVETOED, by
    design: the number is a property of the whole volume the workers share, and
    re-guarding it against one worker's sub-volume is exactly the per-sub-volume
    pedestal disagreement #1174 removes (see :func:`fit_single_tile`). But a level
    at or above THIS sub-volume's maximum windows it entirely to zero — 0 splats,
    an ``.empty`` marker, and a merge that skips it while the run reports success
    — and nothing downstream mentions the floor. So it is announced here.

    Warn-only: the level is still applied. Reuses the same bounded deterministic
    sample :func:`~luxar.gsplats.fitting.preprocessing.resolve_volume_floor`
    would draw, so this costs one bounded read and never a second one.
    """
    from luxar.gsplats.fitting.preprocessing import (
        FLOOR_SAMPLE_BUDGET_VOXELS,
        _sample_volume_for_floor,
    )

    sample = _sample_volume_for_floor(volume, int(FLOOR_SAMPLE_BUDGET_VOXELS))
    if sample is None or sample.size == 0:
        return
    sample_max = float(sample.max())
    if level < sample_max:
        return
    aprint(
        f"⚠ Background floor level {level:.6g} is NOT below {what}'s sampled "
        f"maximum ({sample_max:.6g}): subtracting it clips this whole sub-volume "
        f"to zero, so it will fit 0 SPLATS and be skipped by the merge (an "
        f".empty marker). The level is applied AS GIVEN — a concrete numeric "
        f"--floor is deliberately not re-guarded per sub-volume, so no worker "
        f"disagrees about the pedestal. Pass --floor none, or a lower "
        f"--floor N, if this sub-volume must survive."
    )


def fit_single_tile(
    ctx: FitPipelineCtx, volume: "Any", fit_config: dict, parsed_seeds: "Any"
) -> "Any":
    """Single-tile mode (Slurm-ready): fit tile ``--tile N/M`` of the grid."""
    from luxar.gsplats.fit_tiled_gsplats import count_nonempty_tiles, fit_tile
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
    # spec (rejecting e.g. a negative --floor, as every other entry point does),
    # then resolve it. ``None`` in the merged config (a ``floor: null`` YAML)
    # means DISABLED, exactly as on the sequential tiled and non-tiled paths.
    #
    # A CONCRETE numeric level is applied UNGUARDED (#1174). This worker holds
    # only ONE sub-volume — one tile of one (t, c) — so re-guarding the number
    # here would drop it on a dim/bleached timepoint (``floor=None`` → hard-min
    # normalization) while every sibling task subtracts it: precisely the
    # per-timepoint pedestal difference the shared resolution removes. The number
    # normally comes from a parent that already resolved and guarded it against
    # the whole volume the tiles belong to (`batch-fit` plan time; the
    # `fit --tiling uniform -j N` parent, which forwards its ``--floor`` spec
    # verbatim, so each worker re-resolves the same spec against the same volume).
    #
    # BEHAVIOUR CHANGE vs main: a USER numeric on this path — `fit --tile k/M
    # --floor 110`, or `-j N --floor 110` — used to be guarded here too and would
    # be reported-and-ignored when it exceeded the volume's sampled max. It is now
    # APPLIED, so a too-high number windows the tile to zero. The loud warn-only
    # check below names exactly that, since the resulting ``.empty`` tile would
    # otherwise never mention the floor. A volume-derived spec (``auto``/``pNN``)
    # is still resolved against this whole volume WITH the guard, because it
    # becomes a level here for the first time.
    # With --denoise the tile is denoised BEFORE the level is subtracted, so a
    # volume-derived spec is resolved on the DENOISED basis (#1178) — the same
    # correction `fit_tiled` applies, from the same deterministic probe of the
    # same whole volume, so this worker and its siblings still agree on one
    # level. The denoise keys are PEEKED at: they stay in `fit_config` for
    # `fit_tile` (which pops them) to denoise the tile with.
    from luxar.gsplats.fitting.preprocessing import resolve_volume_floor_denoised
    from luxar.gsplats.fitting.validation import _validate_floor

    floor_spec = fit_config.get("floor", "auto")
    probe_cache = fit_config.setdefault("_denoise_probe_cache", {})
    _validate_floor(floor_spec)
    resolved_floor = resolve_volume_floor_denoised(
        volume,
        floor_spec,
        denoise_h=fit_config.get("_denoise_h"),
        denoise_params=fit_config.get("_denoise_params"),
        probe_cache=probe_cache,
        guard_numeric=False,
        # Log the raw level and the measured denoise shift — but ONLY where
        # denoising made this a new resolution to report. With `--denoise` off
        # this worker's log stays byte-identical to what it printed before #1178
        # (and it would not be read anyway: `build_worker_cmd` hardcodes
        # ``--quiet`` and the parent discards a successful worker's stdout).
        # Gated on a volume-derived spec too: an absolute level is not resolved
        # here, and `warn_if_level_erases_volume` below is the line that matters
        # for one of those.
        verbose=bool(fit_config.get("verbose", False))
        and floor_spec_needs_volume(floor_spec)
        and fit_config.get("_denoise_h") is not None
        and fit_config.get("_denoise_params") is not None,
    )
    if resolved_floor is not None and not floor_spec_needs_volume(floor_spec):
        warn_if_level_erases_volume(
            volume,
            resolved_floor,
            what=_tile_worker_label(ctx, tile_idx, len(specs)),
        )
    fit_config["floor"] = resolved_floor if resolved_floor is not None else "none"

    # Every independent worker scans the same volume, grid and resolved floor,
    # so all of them derive one identical divisor without parent-only state.
    if _needs_nonempty_tile_scan(parsed_seeds, len(specs)):
        nonempty_tiles = count_nonempty_tiles(volume, specs, resolved_floor)
        tile_seeds = split_seeds_across_tiles(
            parsed_seeds, nonempty_tiles, grid_tiles=len(specs)
        )
    else:
        tile_seeds = split_seeds_across_tiles(parsed_seeds, len(specs))

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
            seeds=tile_seeds,
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
    from luxar.gsplats.fit_tiled_gsplats import count_nonempty_tiles, fit_tiled
    from luxar.gsplats.fitting.preprocessing import resolve_volume_floor_denoised
    from luxar.gsplats.fitting.validation import _validate_floor
    from luxar.gsplats.tiling import compute_tile_specs

    # An integer --seeds is a WHOLE-VOLUME budget (what `gsplat cal` reports);
    # fit_tiled hands its `seeds` to EVERY tile, so split it across the grid
    # first. ``volume`` is already downscaled when --downscale is in play,
    # which is exactly the grid fit_tiled will build below.
    specs = compute_tile_specs(volume.shape, ctx.tile_size, ctx.tile_overlap)
    floor_spec = fit_config.get("floor", "auto")
    probe_cache = fit_config.setdefault("_denoise_probe_cache", {})
    _validate_floor(floor_spec)
    if _needs_nonempty_tile_scan(parsed_seeds, len(specs)):
        resolved_floor = resolve_volume_floor_denoised(
            volume,
            floor_spec,
            denoise_h=fit_config.get("_denoise_h"),
            denoise_params=fit_config.get("_denoise_params"),
            probe_cache=probe_cache,
            guard_numeric=True,
            verbose=bool(fit_config.get("verbose", True))
            and floor_spec_needs_volume(floor_spec)
            and fit_config.get("_denoise_h") is not None
            and fit_config.get("_denoise_params") is not None,
        )
        nonempty_tiles = count_nonempty_tiles(volume, specs, resolved_floor)
        tile_seeds = split_seeds_across_tiles(
            parsed_seeds, nonempty_tiles, grid_tiles=len(specs)
        )
        fit_config["floor"] = resolved_floor if resolved_floor is not None else "none"
        fit_config["_floor_resolved"] = True
    else:
        tile_seeds = split_seeds_across_tiles(parsed_seeds, len(specs))

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
        seeds=tile_seeds,
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
        import numpy as np

        # A per-axis rescale IS a diagonal linear transform, and going through
        # `transform` carries everything the leaf holds through it. Rebuilding a
        # plain GSplatData from the concatenated top-level arrays instead RESET
        # `truncation_radius` to the default (#1624): `truncate:` is a documented
        # YAML key (`gsplat fit --dump-config` emits it) that lands on the result
        # in `fitting/results.py`, so `fit --tile k/M --downscale N` with a
        # `--config` holding `truncate: 3.5` stored 2.75 — a wrong radius in the
        # tile's own store, on the plain non-progressive path too, and one the
        # merge's `concatenate` requires the non-empty tiles to AGREE on
        # (`_data/composition.py`), so a downscaled tile also disagreed with an
        # un-downscaled sibling.
        # `transform`'s diagonal fast path multiplies centers by these factors and
        # the packed Cholesky by the same per-row `tril_scales` vector as
        # `rescale_centers` / `rescale_cholesky_packed`. It also maps per sub-LOD,
        # which keeps an additive ladder's rungs (colors, stats, radius) intact;
        # that is a by-construction guarantee rather than a fixed symptom — every
        # fitter reachable here flattens first (`fit_progressive_gsplats` returns
        # `final_result.flattened()`), so no ladder arrives at this line today.
        result = result.transform(
            np.diag(np.asarray(tiled_downscale_factors, dtype=np.float64))
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
    compression_factor: Optional[int],
    levels: Optional[int],
    substitutive_method: Optional[str],
    coarsen_dims: Optional[str],
    refine: Optional[str],
    refine_iters: Optional[int],
    volume: "Any",
    device: Optional[str],
    volume_ndim: int,
) -> "Any":
    """Build optional fit-time LOD recipe parameters from CLI arguments."""
    return _build_fit_recipe_params_impl(
        recipe,
        n_lods=n_lods,
        additive_method=additive_method,
        breakpoints=breakpoints,
        compression_factor=compression_factor,
        levels=levels,
        substitutive_method=substitutive_method,
        coarsen_dims=coarsen_dims,
        refine=refine,
        refine_iters=refine_iters,
        volume=volume,
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
        from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree

        fitting, config, provenance, pipeline = split_fitting_info(
            result.meta.get("fit_stats", {}), include_fitting_info=True
        )
        write_gsplats_tree(
            output_path,
            result,
            compress=compress,
            fitting_info=fitting,
            fitting_config=config,
            provenance_info=provenance,
            pipeline_info=pipeline,
        )
        n = int(getattr(result, "n_splats", 0))
    if verbose:
        aprint(f"Saved {n:,} splats")
        if output_path.exists():
            aprint(f"File size: {format_memory_size(output_path.stat().st_size)}")
    return n
