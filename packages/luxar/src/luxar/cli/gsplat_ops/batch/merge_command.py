"""Implementation helper for ``batch-fit merge`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint, asection

from luxar.cli.gsplat_ops.batch.measurement import (
    measure_tiles_bytes_per_splat as _measure_tiles_bytes_per_splat_impl,
)
from luxar.cli.gsplat_ops.batch.recipe_args import (
    _MERGE_ALLOWED_TOKENS as _MERGE_ALLOWED_TOKENS_IMPL,
)
from luxar.cli.gsplat_ops.batch.recipe_args import (
    _MERGE_OPTION_TOKENS as _MERGE_OPTION_TOKENS_IMPL,
)
from luxar.cli.gsplat_ops.batch.recipe_args import (
    build_merge_recipe_params as _build_merge_recipe_params_impl,
)


def run_batch_merge_cmd(
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

        from luxar.cli.gsplat_ops.recipe_shared import reject_irrelevant_recipe_options
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
            _MERGE_OPTION_TOKENS_IMPL,
            _MERGE_ALLOWED_TOKENS_IMPL,
            no_recipe_hint=no_recipe_hint,
        )

        # Streaming trio → a concrete stream:<c> breakpoints string. Bytes/splat
        # is MEASURED from the completed tile stores when possible (they exist
        # on disk at merge time), falling back to the analytic estimate for the
        # merged parts (spatial dims + the stacked-timepoint axis when
        # timepoints were stacked; colors when a multi-channel color merge will
        # write them). Mutually exclusive with an explicit --breakpoints; the
        # supporting knobs need --target-ms.
        from luxar.cli.gsplat_ops.recipe_shared import validate_streaming_knobs

        validate_streaming_knobs(
            target_ms, bandwidth_mbps, bytes_per_splat, breakpoints
        )
        eff_breakpoints = breakpoints
        if target_ms is not None:
            from luxar.cli.gsplat_ops.recipe_shared import (
                estimate_bytes_per_splat,
                resolve_streaming_breakpoints,
            )

            merged_ndim = len(manifest.spatial_shape) + (
                1 if manifest.n_timepoints > 1 else 0
            )
            merged_has_colors = bool(colors) and manifest.n_channels > 1
            measured, n_measured = _measure_tiles_bytes_per_splat_impl(
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
            recipe_params = _build_merge_recipe_params_impl(
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
