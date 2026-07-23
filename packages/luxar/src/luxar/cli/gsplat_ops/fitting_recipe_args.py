"""Recipe-argument parsing for fit-time per-part LOD."""

from __future__ import annotations

from typing import Any, Optional

import typer


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
) -> Any:
    """Validate per-part ``--recipe`` knobs and build a ``RecipeParams``."""
    from luxar.cli.lod import (
        VALID_ADDITIVE_METHODS,
        VALID_SUBSTITUTIVE_METHODS,
        estimate_bytes_per_splat,
        parse_lod_breakpoints,
        resolve_streaming_breakpoints,
        validate_streaming_knobs,
    )
    from luxar.gsplats.lod.recipes import (
        LEGACY_RECIPE_NAMES,
        PER_PART_RECIPES,
        RecipeParams,
    )

    if recipe in LEGACY_RECIPE_NAMES:
        raise typer.BadParameter(
            f"recipe {recipe!r} was renamed to "
            f"{LEGACY_RECIPE_NAMES[recipe]!r}; use --recipe "
            f"{LEGACY_RECIPE_NAMES[recipe]}."
        )
    if recipe not in PER_PART_RECIPES:
        raise typer.BadParameter(
            f"--recipe must be one of {list(PER_PART_RECIPES)} for a fit "
            f"(stream → tiles, levels → adaptive); got {recipe!r}. "
            f"For other topologies run `gsplat lod` on a flat (--flat) fit."
        )

    # Reject knobs that don't apply to the chosen recipe (mirrors `gsplat lod`,
    # which raises on irrelevant options rather than silently dropping them).
    additive_only = {
        "--n-lods": n_lods,
        "--additive-method": additive_method,
        "--breakpoints": breakpoints,
        "--target-ms": target_ms,
        "--bandwidth-mbps": bandwidth_mbps,
        "--bytes-per-splat": bytes_per_splat,
    }
    substitutive_only = {
        "--compression-factor": compression_factor,
        "--levels": levels,
        "--substitutive-method": substitutive_method,
        "--coarsen-dims": coarsen_dims,
    }
    irrelevant = substitutive_only if recipe == "stream" else additive_only
    provided = [flag for flag, val in irrelevant.items() if val is not None]
    if provided:
        other = "levels" if recipe == "stream" else "stream"
        raise typer.BadParameter(
            f"option(s) {', '.join(provided)} are not used by --recipe {recipe} "
            f"(they configure --recipe {other}). Remove them or switch recipe."
        )

    add_norm = (additive_method or "auto").strip().replace("-", "_")
    if add_norm not in VALID_ADDITIVE_METHODS:
        raise typer.BadParameter(
            f"--additive-method must be one of {list(VALID_ADDITIVE_METHODS)}; "
            f"got {additive_method!r}"
        )
    sub_norm = (substitutive_method or "auto").strip().replace("-", "_")
    if sub_norm not in VALID_SUBSTITUTIVE_METHODS:
        raise typer.BadParameter(
            f"--substitutive-method must be one of "
            f"{list(VALID_SUBSTITUTIVE_METHODS)}; got {substitutive_method!r}"
        )
    validate_streaming_knobs(target_ms, bandwidth_mbps, bytes_per_splat, breakpoints)
    if target_ms is not None:
        bp: Any = resolve_streaming_breakpoints(
            target_ms,
            bandwidth_mbps,
            bytes_per_splat,
            analytic_bps=estimate_bytes_per_splat(volume_ndim),
        )
    else:
        bp = parse_lod_breakpoints(breakpoints) if breakpoints else "equal-count"

    parsed_coarsen: Optional[tuple] = None
    if coarsen_dims is not None:
        try:
            idxs = sorted({int(t) for t in coarsen_dims.split(",") if t.strip() != ""})
        except ValueError as e:
            raise typer.BadParameter(
                f"--coarsen-dims must be comma-separated integers; got {coarsen_dims!r}"
            ) from e
        if not idxs:
            raise typer.BadParameter("--coarsen-dims must list >=1 index")
        for i in idxs:
            if i < 0 or i >= volume_ndim:
                raise typer.BadParameter(
                    f"--coarsen-dims index {i} out of range for {volume_ndim}D data"
                )
        parsed_coarsen = tuple(idxs) if len(idxs) < volume_ndim else None

    return RecipeParams(
        n_lods=n_lods if n_lods is not None else 4,
        additive_method=add_norm,  # type: ignore[arg-type]
        breakpoints=bp,
        compression_factor=(
            compression_factor if compression_factor is not None else 4
        ),
        levels=levels if levels is not None else 3,
        substitutive_method=sub_norm,
        coarsen_dims=parsed_coarsen,
        device=device or "auto",
    )
