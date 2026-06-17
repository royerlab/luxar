"""``luxar gsplat lod --recipe`` — build a representation topology from a fit.

A thin CLI wrapper over :func:`luxar.gsplats.lod.recipes.build_recipe`. It parses
the option superset, validates that the options given are relevant to the chosen
recipe, fills scale-derived defaults, loads the input, builds the recipe, and
writes the output ``.gsplats.zarr``.

The single ``--recipe`` flag replaces the historical ``lod additive`` /
``lod substitutive`` / ``lod pyramid`` subcommands (those three are now recipe
values: ``additive`` / ``substitutive`` / ``pyramid``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import typer
from arbol import aprint, asection

# Valid ordering methods for the additive (prefix-sum) axis.
_VALID_ADDITIVE_METHODS = (
    "greedy",
    "self_energy",
    "mass",
    "amplitude",
    "spectral",
    "random",
)

# Valid substitutive partition algorithms.
_VALID_SUBSTITUTIVE_METHODS = (
    "auto",
    "kmeans",
    "kmeans_lloyd",
    "greedy",
    "greedy_lloyd",
)

_VALID_PARTITION_RULES = ("median", "midpoint", "sah")

# Per-recipe relevance tokens. Each tuning option belongs to a token group; a
# recipe only accepts options whose token is in its allowed set. ``--levels`` is
# its own token because ``multiscale`` accepts the other substitutive options
# (for its coarse cap) but fixes the cap at a single level.
_OPTION_TOKENS = {
    "--n-lods": "additive",
    "--method": "additive",
    "--breakpoints": "additive",
    "--truncation-sigmas": "additive",
    "--max-n-dense": "additive",
    "--max-elements": "partition",
    "--parts": "partition",
    "--partition-rule": "partition",
    "--compression-factor": "substitutive",
    "--substitutive-method": "substitutive",
    "--lloyd-iters": "substitutive",
    "--candidate-bins-k": "substitutive",
    "--levels": "levels",
    "--base-pixel-size": "lod_selector",
    "--lod-method": "lod_selector",
    "--extent-percentile": "lod_selector",
    "--extent-anisotropy": "lod_selector",
}

_ALLOWED_TOKENS = {
    "flat": frozenset(),
    "additive": frozenset({"additive"}),
    "partitioned": frozenset({"additive", "partition"}),
    # ``lod_selector`` (--lod-method/--extent-percentile/--extent-anisotropy/
    # --base-pixel-size) tunes the coarse↔fine switch of any kind=lod group:
    # the multiscale cap, and the substitutive/pyramid/mosaic lod ladders.
    "multiscale": frozenset({"additive", "partition", "substitutive", "lod_selector"}),
    # mosaic: BSP partition + a substitutive lod group per part — partition knobs
    # plus the substitutive ones (and --levels for per-part depth) and the lod
    # selector. No additive ladder (parts replace, not accumulate).
    "mosaic": frozenset({"partition", "substitutive", "levels", "lod_selector"}),
    "substitutive": frozenset({"substitutive", "levels", "lod_selector"}),
    "pyramid": frozenset({"additive", "substitutive", "levels", "lod_selector"}),
}


def _parse_lod_breakpoints(spec: str) -> "str | list[int] | list[float]":
    """Parse the ``--breakpoints`` string for :func:`make_additive_lod`.

    Accepted forms: ``equal-count`` → literal; ``counts:5,10,15`` → ``[int]``
    (cumulative splat counts); ``energy:0.5,0.9,1.0`` → ``[float]`` (cumulative
    energy fractions in (0, 1]).
    """
    s = spec.strip()
    if s == "equal-count":
        return "equal-count"
    if s.startswith("counts:"):
        body = s[len("counts:") :]
        try:
            values_int = [int(p.strip()) for p in body.split(",") if p.strip()]
        except ValueError as e:
            raise typer.BadParameter(
                f"counts breakpoints must be ints; got {body!r}"
            ) from e
        if not values_int:
            raise typer.BadParameter("counts breakpoints list is empty")
        if any(v <= 0 for v in values_int):
            raise typer.BadParameter(
                f"counts breakpoints must be positive; got {values_int}"
            )
        return values_int
    if s.startswith("energy:"):
        body = s[len("energy:") :]
        try:
            values_flt = [float(p.strip()) for p in body.split(",") if p.strip()]
        except ValueError as e:
            raise typer.BadParameter(
                f"energy breakpoints must be floats; got {body!r}"
            ) from e
        if not values_flt:
            raise typer.BadParameter("energy breakpoints list is empty")
        # Energy fractions are cumulative in (0, 1] — checkable here (no N needed).
        if any(not (0.0 < v <= 1.0) for v in values_flt):
            raise typer.BadParameter(
                f"energy breakpoints must lie in (0, 1]; got {values_flt}"
            )
        return values_flt
    raise typer.BadParameter(
        f"breakpoints must be 'equal-count', 'counts:...', or 'energy:...'; "
        f"got {spec!r}"
    )


def _resolve_encoding(mode: str) -> Any:
    from luxar.encoding import EncodingMode

    try:
        return {
            "auto": EncodingMode.AUTO,
            "precision": EncodingMode.PRECISION,
            "memory": EncodingMode.MEMORY,
        }[mode]
    except KeyError:
        raise typer.BadParameter(
            f"--encoding must be auto|precision|memory; got {mode!r}"
        ) from None


def register_lod_command(app: typer.Typer) -> None:
    """Attach the unified ``lod`` command to the ``gsplat`` Typer app."""
    app.command("lod")(lod_recipe)


def lod_recipe(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr (a fitted / flat dataset)."
    ),
    output_path: Path = typer.Argument(..., help="Output .gsplats.zarr."),
    recipe: Optional[str] = typer.Option(
        None,
        "--recipe",
        "-r",
        help=(
            "Representation topology to build (REQUIRED). Scale-ordered: "
            "flat | additive | partitioned | multiscale | mosaic; plus primitives "
            "substitutive | pyramid."
        ),
    ),
    # ── additive ladder (additive / partitioned / multiscale parts / pyramid) ──
    n_lods: Optional[int] = typer.Option(
        None, "--n-lods", min=1, help="Additive LOD levels (default 4)."
    ),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        "-m",
        help="Additive ordering: greedy (default) | self_energy | mass | "
        "amplitude | spectral | random.",
    ),
    breakpoints: Optional[str] = typer.Option(
        None,
        "--breakpoints",
        "-b",
        help="'equal-count' (default) | 'counts:N1,N2,...' | 'energy:f1,f2,...'.",
    ),
    truncation_sigmas: Optional[float] = typer.Option(
        None, "--truncation-sigmas", help="Mahalanobis cutoff for greedy (default 3.0)."
    ),
    max_n_dense: Optional[int] = typer.Option(
        None, "--max-n-dense", help="Greedy dense-Gram threshold (default 2000)."
    ),
    # ── spatial partition (partitioned / multiscale / mosaic) ──
    max_elements: Optional[int] = typer.Option(
        None,
        "--max-elements",
        min=1,
        help="Per-part splat cap for the BSP partition (default 1,000,000). "
        "Mutually exclusive with --parts.",
    ),
    parts: Optional[int] = typer.Option(
        None,
        "--parts",
        min=1,
        help="Target number of parts; sets max_elements = ceil(N / parts).",
    ),
    partition_rule: Optional[str] = typer.Option(
        None, "--partition-rule", help="BSP rule: median (default) | midpoint | sah."
    ),
    # ── substitutive reduction (substitutive / pyramid / multiscale cap) ──
    compression_factor: Optional[int] = typer.Option(
        None,
        "--compression-factor",
        "-K",
        min=2,
        help="Substitutive per-level compression factor (default 4).",
    ),
    levels: Optional[int] = typer.Option(
        None,
        "--levels",
        "-L",
        min=1,
        help="Substitutive coarser levels (default 3). Not used by multiscale "
        "(its cap is a single level).",
    ),
    substitutive_method: Optional[str] = typer.Option(
        None,
        "--substitutive-method",
        help="Substitutive algorithm: auto (default) | kmeans | kmeans_lloyd | "
        "greedy | greedy_lloyd.",
    ),
    lloyd_iterations: Optional[int] = typer.Option(
        None, "--lloyd-iters", min=0, help="Lloyd refinement passes (default 5)."
    ),
    candidate_bins_k: Optional[int] = typer.Option(
        None, "--candidate-bins-k", min=1, help="Lloyd spatial-hash top-k (default 12)."
    ),
    lod_method: Optional[str] = typer.Option(
        None,
        "--lod-method",
        help="multiscale only: coarse↔fine threshold method. 'extent' (default) "
        "anchors the switch in physical element size (W/r, self-calibrating); "
        "'count' is the legacy scene-relative √N proxy.",
    ),
    extent_percentile: Optional[float] = typer.Option(
        None,
        "--extent-percentile",
        min=0.0,
        max=100.0,
        help="multiscale only: percentile of per-level element radius used by "
        "--lod-method extent (default 90).",
    ),
    extent_anisotropy: Optional[bool] = typer.Option(
        None,
        "--extent-anisotropy/--no-extent-anisotropy",
        help="multiscale only: use the largest principal semi-axis (anisotropy-"
        "aware, default) vs the isotropic-equivalent radius for --lod-method extent.",
    ),
    base_pixel_size: Optional[float] = typer.Option(
        None,
        "--base-pixel-size",
        help="multiscale only: LOD selector pixel anchor. In 'extent' mode the "
        "target element pixel size T (~1.5 px, self-calibrating); in 'count' mode "
        "the √N anchor (~10 px). The default rarely needs tuning under 'extent'.",
    ),
    # ── universal ──
    ordering: str = typer.Option(
        "hilbert", "--ordering", help="Spatial ordering: hilbert | morton | none."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Compute device for substitutive: auto|cpu|cuda|mps."
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output if it exists."
    ),
    encoding: str = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode: auto|precision|memory."
    ),
    compress: Optional[str] = typer.Option(
        None, "--compress", help="Optional output compression: 'zip' or 'tar.gz'."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress detail lines."),
) -> None:
    """Build a representation topology from a fitted gsplat dataset.

    One recipe, scale-ordered. Pick with ``--recipe``:

    \b
      flat          single leaf (no LOD, no partition)
      additive      one leaf with an additive (prefix-sum) ladder
      partitioned   BSP parts, each with its own additive ladder
      multiscale    a coarse substitutive cap + a partitioned fine branch
                    (unbalanced by design: detail only where you look closely)
      mosaic        BSP parts, each its own substitutive lod group
                    (per-part coarse<->fine swap: locally adaptive detail)
      substitutive  pure substitutive pyramid (synthesised coarse levels)
      pyramid       balanced substitutive x additive matrix

    Input must be a fitted / flat .gsplats.zarr (output of ``luxar gsplat fit``).
    Canonical pipeline: ``cal`` -> ``fit --seeds K*`` -> ``lod --recipe ...``.

    \b
    Examples:
        luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe additive --n-lods 6
        luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe partitioned \\
            --max-elements 250000
        luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe multiscale \\
            --compression-factor 8
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
    from luxar.gsplats.lod.recipes import (
        RECIPE_NAMES,
        RecipeParams,
        build_recipe,
    )

    try:
        # ── validate --recipe ──
        if recipe is None:
            raise typer.BadParameter(
                "--recipe is required; choose one of: " + ", ".join(RECIPE_NAMES)
            )
        if recipe not in RECIPE_NAMES:
            raise typer.BadParameter(
                f"unknown recipe {recipe!r}; choose one of: " + ", ".join(RECIPE_NAMES)
            )

        # ── option-relevance check (reject options irrelevant to the recipe) ──
        provided = {
            "--n-lods": n_lods,
            "--method": method,
            "--breakpoints": breakpoints,
            "--truncation-sigmas": truncation_sigmas,
            "--max-n-dense": max_n_dense,
            "--max-elements": max_elements,
            "--parts": parts,
            "--partition-rule": partition_rule,
            "--compression-factor": compression_factor,
            "--substitutive-method": substitutive_method,
            "--lloyd-iters": lloyd_iterations,
            "--candidate-bins-k": candidate_bins_k,
            "--levels": levels,
            "--base-pixel-size": base_pixel_size,
            "--lod-method": lod_method,
            "--extent-percentile": extent_percentile,
            "--extent-anisotropy": extent_anisotropy,
        }
        allowed = _ALLOWED_TOKENS[recipe]
        irrelevant = [
            flag
            for flag, value in provided.items()
            if value is not None and _OPTION_TOKENS[flag] not in allowed
        ]
        if irrelevant:
            msg = (
                f"option(s) {', '.join(sorted(irrelevant))} are not used by "
                f"--recipe {recipe}."
            )
            # The substitutive *algorithm* moved to --substitutive-method; -m/--method
            # now means the additive ordering. Point substitutive/pyramid users there.
            if "--method" in irrelevant and recipe == "substitutive":
                msg += " (for the substitutive algorithm use --substitutive-method)"
            raise typer.BadParameter(msg)

        # ── validate values ──
        method_norm = (method or "greedy").strip().replace("-", "_")
        if method_norm not in _VALID_ADDITIVE_METHODS:
            raise typer.BadParameter(
                f"--method must be one of {list(_VALID_ADDITIVE_METHODS)}; got {method!r}"
            )
        sub_norm = (substitutive_method or "auto").strip().replace("-", "_")
        if sub_norm not in _VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--substitutive-method must be one of "
                f"{list(_VALID_SUBSTITUTIVE_METHODS)}; got {substitutive_method!r}"
            )
        rule = partition_rule or "median"
        if rule not in _VALID_PARTITION_RULES:
            raise typer.BadParameter(
                f"--partition-rule must be one of {list(_VALID_PARTITION_RULES)}; "
                f"got {partition_rule!r}"
            )
        if compress not in (None, "zip", "tar.gz"):
            raise typer.BadParameter(
                f"--compress must be 'zip' or 'tar.gz'; got {compress!r}"
            )
        if ordering not in ("hilbert", "morton", "none"):
            raise typer.BadParameter(
                f"--ordering must be hilbert|morton|none; got {ordering!r}"
            )
        if parts is not None and max_elements is not None:
            raise typer.BadParameter(
                "--parts and --max-elements are mutually exclusive."
            )
        bp = _parse_lod_breakpoints(breakpoints or "equal-count")
        encoding_obj = _resolve_encoding(encoding)

        if output_path.exists() and not overwrite:
            raise typer.BadParameter(
                f"Output {output_path} exists; pass --overwrite to replace it."
            )

        with asection(f"LOD recipe '{recipe}': {input_path.name}"):
            with asection("Loading dataset"):
                try:
                    data = GSplatData.load(input_path, include_stats=True)
                except ValueError as e:
                    raise typer.BadParameter(
                        f"recipe input must be a fitted / flat (matrix-shaped) "
                        f".gsplats.zarr; could not load {input_path.name}: {e}"
                    ) from e
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

            # ── scale-derived defaults (logged) ──
            eff_max_elements: Optional[int] = max_elements
            if recipe in ("partitioned", "multiscale", "mosaic"):
                # BSP partitioning needs >= 3 spatial dims; fail cleanly (the
                # rest of the command's validation style) rather than letting
                # the deeper ValueError surface as a raw traceback.
                if data.ndim < 3:
                    raise typer.BadParameter(
                        f"recipe '{recipe}' requires >=3 spatial dimensions for "
                        f"BSP partitioning; got {data.ndim}D. Use --recipe "
                        f"additive/substitutive/pyramid for {data.ndim}D data."
                    )
                if parts is not None:
                    eff_max_elements = -(-data.n_splats // parts)  # ceil
                    aprint(
                        f"--parts {parts} -> max_elements={eff_max_elements:,} "
                        f"(ceil of {data.n_splats:,}/{parts})"
                    )
                elif max_elements is None:
                    from luxar.core.group.partition import DEFAULT_MAX_ELEMENTS

                    eff_max_elements = DEFAULT_MAX_ELEMENTS
                    aprint(f"max_elements defaulting to {DEFAULT_MAX_ELEMENTS:,}")
                else:
                    aprint(f"max_elements={eff_max_elements:,}")

            if lod_method is not None and lod_method not in ("extent", "count"):
                raise typer.BadParameter(
                    f"--lod-method must be 'extent' or 'count'; got {lod_method!r}"
                )

            params = RecipeParams(
                n_lods=n_lods if n_lods is not None else 4,
                additive_method=method_norm,  # type: ignore[arg-type]
                breakpoints=bp,  # type: ignore[arg-type]
                truncation_sigmas=(
                    truncation_sigmas if truncation_sigmas is not None else 3.0
                ),
                max_n_dense=max_n_dense if max_n_dense is not None else 2000,
                max_elements=eff_max_elements,
                partition_rule=rule,  # type: ignore[arg-type]
                compression_factor=(
                    compression_factor if compression_factor is not None else 4
                ),
                levels=levels if levels is not None else 3,
                substitutive_method=sub_norm,
                lloyd_iterations=lloyd_iterations
                if lloyd_iterations is not None
                else 5,
                candidate_bins_k=candidate_bins_k
                if candidate_bins_k is not None
                else 12,
                # multiscale-only LOD threshold knobs; None → RecipeParams defaults
                # (extent method, p90, anisotropy-aware, ~1.5px target anchor).
                lod_method=lod_method if lod_method is not None else "extent",
                extent_percentile=extent_percentile
                if extent_percentile is not None
                else 90.0,
                extent_anisotropy=extent_anisotropy
                if extent_anisotropy is not None
                else True,
                base_pixel_size=base_pixel_size,
                device=device or "auto",
                seed=seed,
            )

            with asection(f"Building '{recipe}'"):
                try:
                    result = build_recipe(data, recipe, params)  # type: ignore[arg-type]
                except ValueError as e:
                    # Builders raise ValueError for input-driven mistakes that
                    # depend on the data (e.g. a counts breakpoint exceeding N);
                    # surface these as a clean BadParameter rather than a traceback.
                    raise typer.BadParameter(str(e)) from e

            with asection("Saving"):
                if output_path.exists() and overwrite:
                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        output_path.unlink()
                if isinstance(result, GSplatData):
                    # Matrix recipe — identical write path to the absorbed subcommands.
                    # The LOD-threshold knobs are derived at tree-build time (here),
                    # so forward them so substitutive/pyramid honor --lod-method etc.
                    # (inert for flat/additive, which have no kind=lod group).
                    result.save(
                        output_path,
                        ordering=ordering,  # type: ignore[arg-type]
                        encoding_mode=encoding_obj,
                        compress=compress,  # type: ignore[arg-type]
                        lod_method=params.lod_method,
                        extent_percentile=params.extent_percentile,
                        extent_anisotropy=params.extent_anisotropy,
                        base_pixel_size=params.base_pixel_size,
                    )
                else:
                    # Composed recipe — write the node tree, carrying input provenance.
                    fitting_info, fitting_config, provenance_info = split_fitting_info(
                        data.stats
                    )
                    write_gsplats_tree(
                        output_path,
                        result,
                        ordering=ordering,  # type: ignore[arg-type]
                        encoding_mode=encoding_obj,
                        fitting_info=fitting_info,
                        fitting_config=fitting_config,
                        provenance_info=provenance_info,
                        compress=compress,  # type: ignore[arg-type]
                    )
                if not quiet:
                    aprint(f"Saved to {output_path}")

    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


# Re-exported for tests / introspection.
__all__ = ["lod_recipe", "register_lod_command"]
