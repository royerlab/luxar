"""``luxar gsplat batch-fit`` — cluster-scale fitting via Slurm.

Owns the ``app_batch`` Typer sub-app and its commands; the aggregator
(``cli/gsplat_commands.py``) mounts it via ``add_typer``. Extracted from the
former monolithic ``gsplat_commands.py`` (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import typer
from arbol import aprint, asection

if TYPE_CHECKING:
    from luxar.gsplats.lod.recipes import RecipeParams

# Re-exported from helpers (single source); kept importable here for
# back-compat with callers/tests that import these names from this module.
from luxar.cli.gsplat_ops.batch_measurement import (
    measure_tiles_bytes_per_splat as _measure_tiles_bytes_per_splat_impl,
)
from luxar.cli.gsplat_ops.batch_planning import _select_plan_timepoints  # noqa: F401
from luxar.cli.gsplat_ops.batch_recipe_args import (
    _MERGE_ALLOWED_TOKENS as _MERGE_ALLOWED_TOKENS_IMPL,
)
from luxar.cli.gsplat_ops.batch_recipe_args import (
    _MERGE_OPTION_TOKENS as _MERGE_OPTION_TOKENS_IMPL,
)
from luxar.cli.gsplat_ops.batch_recipe_args import (
    build_merge_recipe_params as _build_merge_recipe_params_impl,
)
from luxar.cli.gsplat_ops.batch_run import run_batch_run
from luxar.cli.gsplat_ops.batch_status_validate_cancel import (
    run_batch_cancel_cmd,
    run_batch_status_cmd,
    run_batch_validate_cmd,
)
from luxar.cli.gsplat_ops.batch_submit import run_batch_submit
from luxar.cli.gsplat_ops.batch_validation import (
    validate_leaf_arrays as _validate_leaf_arrays_impl,
)
from luxar.cli.gsplat_ops.batch_validation import (
    validate_node_dir as _validate_node_dir_impl,
)
from luxar.cli.gsplat_ops.batch_validation import validate_tile as _validate_tile_impl
from luxar.encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor

app_batch = typer.Typer(
    help="Fit a whole nD dataset across its axes — locally across GPUs "
    "(`batch-fit run`) or on a Slurm cluster (`batch-fit submit`). The "
    "scheduler-agnostic, scaled-up sibling of `gsplat fit`."
)


app_batch.command("submit")(run_batch_submit)


def batch_submit(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around batch submit command function."""
    return run_batch_submit(*args, **kwargs)


app_batch.command("run")(run_batch_run)


def batch_run(*args: Any, **kwargs: Any) -> None:
    """Back-compat wrapper around batch run command function."""
    return run_batch_run(*args, **kwargs)


@app_batch.command("status")
def batch_status_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check status of a batch fitting job.

    Reads the manifest, checks for output files, and queries sacct/squeue
    for job states.

    Examples:
        luxar gsplat batch-fit status output_dir/
    """
    return run_batch_status_cmd(output_dir=output_dir, verbose=verbose)


@app_batch.command("validate")
def batch_validate_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    fix: bool = typer.Option(
        False, "--fix", help="Delete corrupt/incomplete tiles so they get re-fitted"
    ),
) -> None:
    """Validate integrity of all tiles in a batch output directory.

    Checks each tile for completeness (metadata, arrays, shapes).
    Reports OK, MISSING, CORRUPT, and STALE_TMP counts.

    Use --fix to delete corrupt tiles and leftover .tmp directories,
    so they get re-fitted on the next submit.

    Examples:
        luxar gsplat batch-fit validate output_dir/

        luxar gsplat batch-fit validate output_dir/ --fix
    """
    return run_batch_validate_cmd(output_dir=output_dir, fix=fix)


def _validate_leaf_arrays(node_dir: Path, label: str) -> str:
    """Back-compat wrapper around leaf-array validation helper."""
    return _validate_leaf_arrays_impl(node_dir, label)


def _validate_node_dir(node_dir: Path, label: str) -> str:
    """Back-compat wrapper around node-tree validation helper."""
    return _validate_node_dir_impl(node_dir, label)


def _validate_tile(tile_path: Path) -> str:
    """Back-compat wrapper around full tile validation helper."""
    return _validate_tile_impl(tile_path)


@app_batch.command("cancel")
def batch_cancel_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """Cancel all Slurm jobs for a batch fitting run.

    Reads the manifest to find job IDs (calibrate, denoise, fit array,
    merge) and cancels them via scancel.

    Examples:
        luxar gsplat batch-fit cancel output_dir/
    """
    return run_batch_cancel_cmd(output_dir=output_dir)


# Per-part merge recipe parsing lives in ``batch_recipe_args.py``; keep these
# names at module scope for back-compat with tests/importers.
_MERGE_OPTION_TOKENS = _MERGE_OPTION_TOKENS_IMPL
_MERGE_ALLOWED_TOKENS = _MERGE_ALLOWED_TOKENS_IMPL


def _build_merge_recipe_params(
    stored: dict,
    *,
    n_lods: Optional[int] = None,
    additive_method: Optional[str] = None,
    breakpoints: Optional[str] = None,
    compression_factor: Optional[int] = None,
    levels: Optional[int] = None,
    substitutive_method: Optional[str] = None,
    coarsen_dims: Optional[str] = None,
) -> "RecipeParams":
    return _build_merge_recipe_params_impl(
        stored,
        n_lods=n_lods,
        additive_method=additive_method,
        breakpoints=breakpoints,
        compression_factor=compression_factor,
        levels=levels,
        substitutive_method=substitutive_method,
        coarsen_dims=coarsen_dims,
    )


def _measure_tiles_bytes_per_splat(
    tiles_dir: Path, tile_names: List[str]
) -> Tuple[Optional[float], int]:
    """Back-compat wrapper around tile-bytes measurement helper."""
    return _measure_tiles_bytes_per_splat_impl(tiles_dir, tile_names)


@app_batch.command("merge")
def batch_merge_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    channel_colors: Optional[str] = typer.Option(
        None, "--channel-colors", help="Hex colors for channel merge"
    ),
    force: bool = typer.Option(False, "--force", help="Re-merge even if outputs exist"),
    flat: bool = typer.Option(
        False,
        "--flat",
        help=(
            "Concatenate all tiles into a single flat leaf (legacy). Default is a "
            "memory-safe kind=partition with one part per spatial tile."
        ),
    ),
    recipe: Optional[str] = typer.Option(
        None,
        "--recipe",
        help=(
            "Per-part LOD recipe applied to each spatial tile-part as it streams: "
            "'stream' (each part a prefix-sum ladder → tiles topology) or "
            "'levels' (each part its own coarse↔fine lod group → adaptive). "
            "Default: bare-leaf parts (no per-part LOD). Closes the tiled-data LOD "
            "gap without re-loading the whole volume. Falls back to the recipe "
            "recorded at plan time. Mutually exclusive with --flat."
        ),
    ),
    no_recipe: bool = typer.Option(
        False,
        "--no-recipe",
        help=(
            "Force a recipe-less merge (bare-leaf parts), overriding any "
            "merge_recipe recorded at plan time. Use this to merge without LOD "
            "when the manifest defaulted to a recipe."
        ),
    ),
    n_lods: Optional[int] = typer.Option(
        None, "--n-lods", help="Additive ladder depth (stream recipe)."
    ),
    additive_method: Optional[str] = typer.Option(
        None,
        "--additive-method",
        help="Additive ladder method: auto (default) | greedy | self_energy "
        "(stream recipe).",
    ),
    breakpoints: Optional[str] = typer.Option(
        None,
        "--breakpoints",
        help="Additive ladder breakpoints: 'equal-count' (default), 'stream:C', "
        "'counts:...' or 'energy:...' (stream recipe).",
    ),
    target_ms: Optional[float] = typer.Option(
        None,
        "--target-ms",
        min=1.0,
        help="[stream recipe] streaming sizing: derive 'stream:<c>' "
        "breakpoints so each part's first additive chunk downloads in ~this "
        "many ms at --bandwidth-mbps. Mutually exclusive with --breakpoints.",
    ),
    bandwidth_mbps: Optional[float] = typer.Option(
        None,
        "--bandwidth-mbps",
        min=0.1,
        help="Assumed downlink for --target-ms sizing (default 25).",
    ),
    bytes_per_splat: Optional[float] = typer.Option(
        None,
        "--bytes-per-splat",
        min=0.1,
        help="Override the on-wire bytes/splat for --target-ms sizing "
        "(default: analytic estimate for the merged parts).",
    ),
    compression_factor: Optional[int] = typer.Option(
        None, "-K", "--compression-factor", help="Substitutive reduction factor."
    ),
    levels: Optional[int] = typer.Option(
        None, "-L", "--levels", help="Substitutive level count (levels recipe)."
    ),
    substitutive_method: Optional[str] = typer.Option(
        None, "--substitutive-method", help="Substitutive coarsening method."
    ),
    coarsen_dims: Optional[str] = typer.Option(
        None,
        "--coarsen-dims",
        help=(
            "Comma-separated center-column indices substitutive coarsening may "
            "merge over; the rest become hard barriers. Default: spatial dims only "
            "(stacked-timepoint axis is a barrier)."
        ),
    ),
) -> None:
    """Run the merge step for a completed batch job.

    Normally runs as a dependent Slurm job, but this command allows
    running it manually or re-running if the merge job failed.

    By default the tiles are assembled into a ``kind=partition`` file (one part
    per spatial tile) — streamed tile-by-tile so peak memory is a single
    tile-region, and the spatial structure is preserved for per-part frustum
    culling. Pass ``--flat`` for the legacy single-leaf concatenation (reloads
    every tile into memory).

    Pass ``--recipe`` to give each tile-part its own LOD ladder as it streams —
    the memory-safe way to add level-of-detail to tiled output (the canonical
    ``cal → fit → lod`` chain otherwise can't, since ``lod`` rejects a partition).

    Examples:
        luxar gsplat batch-fit merge output_dir/

        luxar gsplat batch-fit merge output_dir/ --flat

        luxar gsplat batch-fit merge output_dir/ --recipe stream --n-lods 6

        luxar gsplat batch-fit merge output_dir/ --recipe levels -K 4 -L 3

        luxar gsplat batch-fit merge output_dir/ --no-recipe   # override a manifest recipe

        luxar gsplat batch-fit merge output_dir/ --channel-colors "#ff0080,#00ff00"
    """
    try:
        from luxar.cli.gsplat_config import parse_hex_color
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.batch.merge_orchestrator import merge_batch_results

        manifest = load_manifest(output_dir)

        colors = None
        color_source = channel_colors or (
            ",".join(manifest.channel_colors) if manifest.channel_colors else None
        )
        if color_source:
            colors = [parse_hex_color(c.strip()) for c in color_source.split(",")]

        from luxar.cli.lod import reject_irrelevant_recipe_options
        from luxar.gsplats.lod.recipes import PER_PART_RECIPES

        # ── usage validation (up front, before the streaming writer runs) ──
        if recipe is not None and no_recipe:
            raise typer.BadParameter("--recipe and --no-recipe are mutually exclusive.")
        if flat and recipe is not None:
            raise typer.BadParameter(
                "--flat and --recipe are mutually exclusive; --flat concatenates "
                "all tiles into a single bare leaf (no per-part LOD)."
            )

        # Resolve the per-part recipe + its knobs, CLI overriding the values
        # recorded at plan time (manifest.merge_recipe / merge_recipe_args).
        # --no-recipe (or --flat) forces a recipe-less merge regardless of the
        # manifest default.
        # NOTE: the uniform+per-part-LOD warning now fires inside
        # merge_batch_results (the library boundary), so every caller — this CLI,
        # the Slurm merge job, and any direct API use — gets it exactly once.
        from luxar.gsplats.lod.recipes import (
            LEGACY_RECIPE_NAMES,
            canonical_recipe_name,
        )

        if recipe in LEGACY_RECIPE_NAMES:
            raise typer.BadParameter(
                f"recipe {recipe!r} was renamed to "
                f"{LEGACY_RECIPE_NAMES[recipe]!r}; use --recipe "
                f"{LEGACY_RECIPE_NAMES[recipe]}."
            )
        # Manifests written before the rename carry legacy spellings —
        # translate those silently (data compat, not CLI compat).
        manifest_recipe = (
            canonical_recipe_name(manifest.merge_recipe)
            if manifest.merge_recipe
            else None
        )
        eff_recipe = None if (no_recipe or flat) else (recipe or manifest_recipe)

        # Validate the effective recipe NAME before the knob-relevance check —
        # mirrors `gsplat lod`'s RECIPE_NAMES guard (cli/lod.py). Without this an
        # unknown recipe (a typo like `--recipe addative`, or a stale manifest
        # value) reaches reject_irrelevant_recipe_options, whose allowed-token
        # lookup returns empty and misreports a VALID knob as "not used by
        # --recipe addative" — hiding the real error (the recipe name).
        if eff_recipe is not None and eff_recipe not in PER_PART_RECIPES:
            raise typer.BadParameter(
                f"unknown per-part recipe {eff_recipe!r}; choose from "
                f"{', '.join(sorted(PER_PART_RECIPES))}"
                + (" (recorded at plan time in the manifest)" if recipe is None else "")
            )

        # Reject recipe knobs that are irrelevant to (or given without) the
        # effective recipe — previously such knobs were silently dropped. The
        # no-recipe hint depends on WHY there's no recipe: a forced recipe-less
        # merge (--no-recipe/--flat) must not tell the user to "pass --recipe"
        # (it would contradict the flag they just typed).
        if no_recipe or flat:
            forced = "--no-recipe" if no_recipe else "--flat"
            no_recipe_hint = (
                f"{forced} forces a recipe-less (bare-leaf) merge — drop these "
                f"knobs, or drop {forced} and pass --recipe stream|levels "
                f"for per-part LOD."
            )
        else:
            no_recipe_hint = (
                "Pass --recipe stream|levels (without one the merge "
                "writes bare-leaf parts, so these knobs would be ignored)."
            )
        reject_irrelevant_recipe_options(
            eff_recipe,
            {
                "--n-lods": n_lods,
                "--additive-method": additive_method,
                "--breakpoints": breakpoints,
                "--target-ms": target_ms,
                "--bandwidth-mbps": bandwidth_mbps,
                "--bytes-per-splat": bytes_per_splat,
                "--compression-factor": compression_factor,
                "--levels": levels,
                "--substitutive-method": substitutive_method,
                "--coarsen-dims": coarsen_dims,
            },
            _MERGE_OPTION_TOKENS,
            _MERGE_ALLOWED_TOKENS,
            no_recipe_hint=no_recipe_hint,
        )

        # Streaming trio → a concrete stream:<c> breakpoints string. Bytes/splat
        # is MEASURED from the completed tile stores when possible (they exist
        # on disk at merge time), falling back to the analytic estimate for the
        # merged parts (spatial dims + the stacked-timepoint axis when
        # timepoints were stacked; colors when a multi-channel color merge will
        # write them). Mutually exclusive with an explicit --breakpoints; the
        # supporting knobs need --target-ms.
        from luxar.cli.lod import validate_streaming_knobs

        validate_streaming_knobs(
            target_ms, bandwidth_mbps, bytes_per_splat, breakpoints
        )
        eff_breakpoints = breakpoints
        if target_ms is not None:
            from luxar.cli.lod import (
                estimate_bytes_per_splat,
                resolve_streaming_breakpoints,
            )

            merged_ndim = len(manifest.spatial_shape) + (
                1 if manifest.n_timepoints > 1 else 0
            )
            merged_has_colors = bool(colors) and manifest.n_channels > 1
            measured, n_measured = _measure_tiles_bytes_per_splat(
                output_dir / "tiles",
                [job.output_filename for job in manifest.jobs],
            )
            eff_breakpoints = resolve_streaming_breakpoints(
                target_ms,
                bandwidth_mbps,
                bytes_per_splat,
                measured_bps=measured,
                measured_label=(f"measured from {n_measured} completed tile store(s)"),
                analytic_bps=estimate_bytes_per_splat(
                    merged_ndim, has_colors=merged_has_colors
                ),
            )

        recipe_params = None
        if eff_recipe is not None:
            recipe_params = _build_merge_recipe_params(
                manifest.merge_recipe_args,
                n_lods=n_lods,
                additive_method=additive_method,
                breakpoints=eff_breakpoints,
                compression_factor=compression_factor,
                levels=levels,
                substitutive_method=substitutive_method,
                coarsen_dims=coarsen_dims,
            )

        with asection(f"Merging batch results: {output_dir}"):
            final_path = merge_batch_results(
                manifest=manifest,
                output_dir=output_dir,
                channel_colors=colors,
                force=force,
                flat=flat,
                recipe=eff_recipe,
                recipe_params=recipe_params,
            )
            aprint(f"\nFinal output: {final_path}")

    except (typer.Exit, typer.BadParameter):
        # Usage errors (e.g. an invalid --substitutive-method) surface cleanly
        # instead of being swallowed into an "Error: ..." traceback below.
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


# ── Hidden batch worker commands for denoise pipeline ────────────


@app_batch.command("denoise-calibrate", hidden=True)
def batch_denoise_calibrate_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """[Internal] Run NLM calibration for batch denoise pipeline.

    Reads manifest, calibrates h per channel, writes results back.
    Called by the calibration Slurm job.
    """
    try:
        import json

        from luxar.gsplats.batch.manifest import load_manifest, save_manifest
        from luxar.gsplats.preprocessing.denoise_pipeline import calibrate_all_channels

        manifest = load_manifest(output_dir)

        if not manifest.denoise:
            aprint("Error: denoise not enabled in manifest")
            raise typer.Exit(1)

        with asection("NLM Calibration"):
            h_values = calibrate_all_channels(
                input_path=Path(manifest.input_path),
                n_timepoints=manifest.n_timepoints,
                n_channels=manifest.n_channels,
                channel_indices=(
                    manifest.channel_indices
                    if manifest.channel_indices
                    else list(range(manifest.n_channels))
                ),
                timepoint_indices=manifest.timepoint_indices,
                array_key=manifest.array_key,
                calibration_samples=manifest.calibration_samples,
                patch_size=manifest.denoise_patch_size,
                search_distance=manifest.denoise_search_distance,
                backend=manifest.denoise_backend,
                h_override=manifest.denoise_h,
            )

            # Write h_values to manifest (string keys for JSON)
            manifest.denoise_h_values = {str(k): v for k, v in h_values.items()}
            save_manifest(manifest, output_dir)

            # Also write standalone JSON for easy reading by other jobs
            h_path = output_dir / "denoise_h_values.json"
            h_path.write_text(json.dumps(h_values, indent=2))

            aprint(f"Calibrated h values: {h_values}")
            aprint(f"Saved to {h_path}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


@app_batch.command("denoise-preprocess", hidden=True)
def batch_denoise_preprocess_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    task_id: int = typer.Argument(..., help="Array task ID (encodes T*n_c + C)"),
) -> None:
    """[Internal] Denoise one (T,C) volume for batch preprocess pipeline.

    Called by the denoise Slurm array job, one task per (timepoint, channel).
    """
    try:
        import json

        import zarr

        from luxar.cli.gsplat_config import load_volume
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

        manifest = load_manifest(output_dir)

        # Read calibrated h values
        h_path = output_dir / "denoise_h_values.json"
        if not h_path.exists():
            aprint("Error: denoise_h_values.json not found. Run calibration first.")
            raise typer.Exit(1)
        h_values = json.loads(h_path.read_text())

        # Decode task_id -> (t_idx, c_idx) within selected indices
        n_c = manifest.n_channels
        t_idx = task_id // n_c
        c_idx = task_id % n_c

        # Map to real dataset indices
        t_indices = manifest.timepoint_indices or list(range(manifest.n_timepoints))
        c_indices = manifest.channel_indices or list(range(manifest.n_channels))
        t_real = t_indices[t_idx]
        c_real = c_indices[c_idx]

        h = h_values.get(str(c_real), 0.04)

        with asection(f"Denoising T={t_real} C={c_real} (h={h:.4f})"):
            # Load volume
            volume = load_volume(
                Path(manifest.input_path),
                channel=c_real if manifest.n_channels > 1 else None,
                timepoint=t_real if manifest.n_timepoints > 1 else None,
                array_key=manifest.array_key,
            )
            aprint(f"Loaded: shape={volume.shape}")

            # Denoise
            denoised = denoise_volume_array(
                volume,
                h=h,
                patch_size=manifest.denoise_patch_size,
                search_distance=manifest.denoise_search_distance,
                backend=manifest.denoise_backend,
                use_2d=manifest.denoise_2d,
            )

            # Write to denoised.zarr
            zarr_path = output_dir / "denoised.zarr"
            store = zarr.open(str(zarr_path), mode="a")

            spatial = denoised.shape
            full_shape = (len(t_indices), len(c_indices), *spatial)
            chunks = (1, 1, *[min(s, 128) for s in spatial])

            if "data" not in store:
                store.create_dataset(
                    "data",
                    shape=full_shape,
                    chunks=chunks,
                    # string dtype: numpy is only imported under TYPE_CHECKING
                    # in this module (dtype=np.float32 here was a latent
                    # NameError before this change).
                    dtype="float32",
                    compressor=resolve_compressor(WIDTH_AWARE_DEFAULT, "float32"),
                )
            store["data"][t_idx, c_idx] = denoised
            aprint(f"Written to denoised.zarr[{t_idx}, {c_idx}]")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e
