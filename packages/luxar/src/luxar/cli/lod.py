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
from typing import Any, Mapping, Optional

import typer
from arbol import aprint, asection

# Valid ordering methods for the additive (prefix-sum) axis.
_VALID_ADDITIVE_METHODS = (
    "auto",
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

#: Target splat count for the multiscale coarse cap when ``--compression-factor``
#: is not given. The cap is a SINGLE substitutive level of ``ceil(N / K)`` splats,
#: so a fixed default K scales badly: on a 23 M fit, the historical K=4/8 left a
#: 2.9 M-splat cap (far too heavy to load first). Instead derive
#: ``K = max(2, round(N / target))`` so the cap lands near this size regardless of
#: N. ~256 K keeps the coarsest level light enough to stream instantly while still
#: carrying enough detail to be a useful overview. Mirrors the N-aware
#: ``max_elements`` default used for the partitioned branch.
_MULTISCALE_CAP_TARGET = 256_000

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
    "--coarsen-dims": "substitutive",
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


def reject_irrelevant_recipe_options(
    recipe: Optional[str],
    provided: Mapping[str, Any],
    option_tokens: Mapping[str, str],
    allowed_tokens: Mapping[str, "frozenset[str]"],
    *,
    hints: Optional[Mapping[str, str]] = None,
    no_recipe_hint: str = "",
) -> None:
    """Raise ``typer.BadParameter`` for options irrelevant to ``recipe``.

    Shared by ``gsplat lod`` and ``batch-fit merge`` so both reject — rather
    than silently ignore — a recipe-specific knob. A flag is *provided* when its
    value is not ``None``; it is *irrelevant* when its token
    (``option_tokens[flag]``) is not in ``allowed_tokens[recipe]``. When
    ``recipe is None`` (no recipe in effect) every recipe-specific knob is
    irrelevant; the message is the neutral prefix plus ``no_recipe_hint`` — the
    caller supplies the *why* (e.g. "no recipe given, pass --recipe ..." vs
    "--no-recipe forces a recipe-less merge, drop these knobs"), since the helper
    can't tell why the recipe is absent. ``hints`` maps a flag to an extra clause
    appended when that flag is among the irrelevant ones (recipe-present case).
    """
    allowed: "frozenset[str]" = (
        allowed_tokens.get(recipe, frozenset()) if recipe is not None else frozenset()
    )
    irrelevant = sorted(
        flag
        for flag, value in provided.items()
        if value is not None and option_tokens[flag] not in allowed
    )
    if not irrelevant:
        return
    if recipe is None:
        msg = f"option(s) {', '.join(irrelevant)} are recipe-specific but no recipe is in effect."
        if no_recipe_hint:
            msg += " " + no_recipe_hint
    else:
        msg = (
            f"option(s) {', '.join(irrelevant)} are not used by --recipe {recipe}."
        )
        if hints:
            for flag, clause in hints.items():
                if flag in irrelevant:
                    msg += " " + clause
    raise typer.BadParameter(msg)


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
        help="Additive ordering: auto (default; greedy at small N, self_energy "
        "for large N to avoid greedy's O(N·nnz·logN) blowup) | greedy | "
        "self_energy | mass | amplitude | spectral | random.",
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
        help="Substitutive per-level compression factor (default 4; for "
        "--recipe multiscale, auto-scaled from N to a ~256K coarse cap "
        "when omitted).",
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
    coarsen_dims: Optional[str] = typer.Option(
        None,
        "--coarsen-dims",
        help="Comma-separated center-column indices coarsening may merge over; "
        "the remaining dims become hard grouping barriers (e.g. a categorical / "
        "timepoint / channel axis). Default: all dims. (Standalone gsplats carry "
        "no display info, so pass explicit indices here.)",
    ),
    lod_method: Optional[str] = typer.Option(
        None,
        "--lod-method",
        help="lod recipes (multiscale/substitutive/pyramid/mosaic): coarse↔fine "
        "threshold method. 'extent' (default) anchors the switch in physical element "
        "size (W/r, self-calibrating); 'count' is the legacy scene-relative √N proxy.",
    ),
    extent_percentile: Optional[float] = typer.Option(
        None,
        "--extent-percentile",
        min=0.0,
        max=100.0,
        help="lod recipes (multiscale/substitutive/pyramid/mosaic): percentile of "
        "per-level element radius used by --lod-method extent (default 90).",
    ),
    extent_anisotropy: Optional[bool] = typer.Option(
        None,
        "--extent-anisotropy/--no-extent-anisotropy",
        help="lod recipes (multiscale/substitutive/pyramid/mosaic): use the largest "
        "principal semi-axis (anisotropy-aware, default) vs the isotropic-equivalent "
        "radius for --lod-method extent.",
    ),
    base_pixel_size: Optional[float] = typer.Option(
        None,
        "--base-pixel-size",
        help="lod recipes (multiscale/substitutive/pyramid/mosaic): LOD selector "
        "pixel anchor. In 'extent' mode the target element pixel size T (~1.5 px, "
        "self-calibrating); in 'count' mode the √N anchor (~10 px). The default "
        "rarely needs tuning under 'extent'.",
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
            "--coarsen-dims": coarsen_dims,
            "--levels": levels,
            "--base-pixel-size": base_pixel_size,
            "--lod-method": lod_method,
            "--extent-percentile": extent_percentile,
            "--extent-anisotropy": extent_anisotropy,
        }
        # ``hints`` add a recipe-specific clause when a given flag is rejected:
        #  - --method moved to --substitutive-method for the substitutive recipe;
        #  - multiscale's coarse cap is single-level, so --levels has no meaning
        #    there (size the cap with -K, auto-scaled by default).
        hints: dict[str, str] = {}
        if recipe == "substitutive":
            hints["--method"] = (
                "(for the substitutive algorithm use --substitutive-method)"
            )
        if recipe == "multiscale":
            hints["--levels"] = (
                "(multiscale's coarse cap is single-level; size it with "
                "--compression-factor/-K, auto-scaled by default)"
            )
        reject_irrelevant_recipe_options(
            recipe, provided, _OPTION_TOKENS, _ALLOWED_TOKENS, hints=hints
        )

        # ── validate values ──
        method_norm = (method or "auto").strip().replace("-", "_")
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
                        f".gsplats.zarr; could not load {input_path.name}: {e}. "
                        f"If this is a kind=partition (e.g. a tiled `batch-fit "
                        f"merge` output), collapse it to a single leaf first with "
                        f"`luxar gsplat flatten {input_path.name} flat.gsplats.zarr`."
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

            # multiscale's coarse cap is a SINGLE substitutive level of ceil(N/K)
            # splats. A fixed default K scales badly (K=8 on 23 M → a 2.9 M cap),
            # so when --compression-factor is not given, size K to land the cap
            # near _MULTISCALE_CAP_TARGET. Explicit -K always wins.
            eff_compression_factor: Optional[int] = compression_factor
            if recipe == "multiscale" and compression_factor is None:
                eff_compression_factor = max(
                    2, round(data.n_splats / _MULTISCALE_CAP_TARGET)
                )
                cap_n = -(-data.n_splats // eff_compression_factor)  # ceil
                aprint(
                    f"--compression-factor defaulting to {eff_compression_factor} "
                    f"(coarse cap ~{cap_n:,} splats, target "
                    f"~{_MULTISCALE_CAP_TARGET:,}); pass -K to override"
                )

            if lod_method is not None and lod_method not in ("extent", "count"):
                raise typer.BadParameter(
                    f"--lod-method must be 'extent' or 'count'; got {lod_method!r}"
                )

            # Barrier dims for substitutive coarsening. Standalone gsplats carry
            # no display metadata, so this path takes explicit column indices and
            # warns (rather than auto-grouping) when the input is >3D.
            parsed_coarsen: Optional[tuple] = None
            if coarsen_dims is not None:
                try:
                    idxs = sorted(
                        {int(t) for t in coarsen_dims.split(",") if t.strip() != ""}
                    )
                except ValueError as e:
                    raise typer.BadParameter(
                        f"--coarsen-dims must be comma-separated integers; "
                        f"got {coarsen_dims!r}"
                    ) from e
                if not idxs:
                    raise typer.BadParameter("--coarsen-dims must list >=1 index")
                for i in idxs:
                    if i < 0 or i >= data.ndim:
                        raise typer.BadParameter(
                            f"--coarsen-dims index {i} out of range for "
                            f"{data.ndim}D data"
                        )
                parsed_coarsen = tuple(idxs) if len(idxs) < data.ndim else None
            elif data.ndim > 3 and recipe in (
                "substitutive",
                "pyramid",
                "multiscale",
                "mosaic",
            ):
                aprint(
                    f"  ⚠ {data.ndim}D input with no --coarsen-dims: substitutive "
                    "coarsening will merge across ALL dims. If some dims are "
                    "categorical/sliced (time/channel/...), pass --coarsen-dims "
                    "with the spatial column indices to keep them as barriers."
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
                    eff_compression_factor
                    if eff_compression_factor is not None
                    else 4
                ),
                levels=levels if levels is not None else 3,
                substitutive_method=sub_norm,
                lloyd_iterations=lloyd_iterations
                if lloyd_iterations is not None
                else 5,
                candidate_bins_k=candidate_bins_k
                if candidate_bins_k is not None
                else 12,
                coarsen_dims=parsed_coarsen,
                # LOD threshold knobs for any kind=lod recipe (multiscale/
                # substitutive/pyramid/mosaic); None → RecipeParams defaults
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
