"""Recipe-argument parsing for ``luxar gsplat batch-fit merge``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import typer

if TYPE_CHECKING:
    from luxar.gsplats.lod.recipes import RecipeParams


# Per-part merge recipes accept only ``additive`` / ``substitutive`` (the
# composed topologies are reached via `gsplat flatten` → `gsplat lod`). These
# mirror lod.py's ``_OPTION_TOKENS`` / ``_ALLOWED_TOKENS`` so the merge path
# rejects — rather than silently ignores — a knob irrelevant to (or given
# without) a recipe. ``--coarsen-dims`` belongs to the substitutive (mosaic)
# per-part lod group.
_MERGE_OPTION_TOKENS = {
    "--n-lods": "additive",
    "--additive-method": "additive",
    "--breakpoints": "additive",
    "--target-ms": "additive",
    "--bandwidth-mbps": "additive",
    "--bytes-per-splat": "additive",
    "--compression-factor": "substitutive",
    "--levels": "substitutive",
    "--substitutive-method": "substitutive",
    "--coarsen-dims": "substitutive",
}
_MERGE_ALLOWED_TOKENS = {
    "stream": frozenset({"additive"}),
    "levels": frozenset({"substitutive"}),
}


def build_merge_recipe_params(
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
    """Build a ``RecipeParams`` for the per-part merge recipe.

    Each knob is resolved CLI-first, then the value recorded at plan time
    (``manifest.merge_recipe_args``, string-valued), then the ``RecipeParams``
    default. ``coarsen_dims`` is parsed from a comma string to a tuple of ints;
    leaving it unset lets the merge default it per part (spatial dims only). The
    additive knobs (``additive_method`` / ``breakpoints``) bring
    the merge recipe to parity with ``fit --recipe`` and ``gsplat lod``.

    NOTE: the L2 refine knobs (``refine`` / ``refine_iters`` on ``RecipeParams``)
    are deliberately NOT exposed at the merge yet — per-part refits across
    hundreds of tiles need their own perf validation first; the RecipeParams
    defaults ("none") keep the merge byte-identical to before.
    """
    from luxar.cli.lod import (
        VALID_ADDITIVE_METHODS,
        VALID_SUBSTITUTIVE_METHODS,
        parse_lod_breakpoints,
    )
    from luxar.gsplats.lod.recipes import RecipeParams

    def _resolve(key: str, cli: Any, cast: Callable[[Any], Any]) -> Any:
        if cli is not None:
            return cli
        raw = stored.get(key)
        return cast(raw) if raw is not None else None

    def _parse_dims(raw: Any) -> tuple:
        try:
            return tuple(int(x) for x in str(raw).split(",") if x.strip() != "")
        except ValueError as e:
            raise typer.BadParameter(
                f"--coarsen-dims must be comma-separated integers; got {raw!r}"
            ) from e

    overrides: dict = {}
    nl = _resolve("n-lods", n_lods, int)
    if nl is not None:
        overrides["n_lods"] = nl
    am = _resolve("additive-method", additive_method, str)
    if am is not None:
        am_norm = am.strip().replace("-", "_")
        if am_norm not in VALID_ADDITIVE_METHODS:
            raise typer.BadParameter(
                f"--additive-method must be one of "
                f"{list(VALID_ADDITIVE_METHODS)}; got {am!r}"
            )
        overrides["additive_method"] = am_norm
    bp = _resolve("breakpoints", breakpoints, str)
    if bp is not None:
        overrides["breakpoints"] = parse_lod_breakpoints(bp)
    cf = _resolve("compression-factor", compression_factor, int)
    if cf is not None:
        overrides["compression_factor"] = cf
    lv = _resolve("levels", levels, int)
    if lv is not None:
        overrides["levels"] = lv
    sm = _resolve("substitutive-method", substitutive_method, str)
    if sm is not None:
        # Normalise hyphens to underscores so the documented CLI spelling
        # (`kmeans-lloyd`) maps to the canonical method name (`kmeans_lloyd`),
        # then validate up front — both halves of the `gsplat lod` contract
        # (cli/lod.py:381-386). Validating here means a bad method fails cleanly
        # BEFORE the streaming writer overwrites final.gsplats.zarr, rather than
        # raising deep in the merge and leaving a stub a non-`--force` re-run skips.
        sm_norm = sm.strip().replace("-", "_")
        if sm_norm not in VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--substitutive-method must be one of "
                f"{list(VALID_SUBSTITUTIVE_METHODS)}; got {sm!r}"
            )
        overrides["substitutive_method"] = sm_norm
    cd = coarsen_dims if coarsen_dims is not None else stored.get("coarsen-dims")
    if cd is not None:
        overrides["coarsen_dims"] = _parse_dims(cd)

    return RecipeParams(**overrides)
