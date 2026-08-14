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
    "--add-method": "additive",
    "--breakpoints": "additive",
    "--target-ms": "additive",
    "--bandwidth-mbps": "additive",
    "--bytes-per-splat": "additive",
    "--compression-factor": "substitutive",
    "--levels": "substitutive",
    "--subst-method": "substitutive",
    "--coarsen-dims": "substitutive",
    "--refine": "substitutive",
    "--refine-iters": "substitutive",
}
_MERGE_ALLOWED_TOKENS = {
    "stream": frozenset({"additive"}),
    "levels": frozenset({"substitutive"}),
}


def _apply_refine(
    overrides: dict, refine: "Optional[str]", refine_iters: "Optional[int]"
) -> None:
    """Validate and record the refine knobs, leaving them unset when not given.

    Unset means the ``RecipeParams`` default ("none"), which keeps a merge
    byte-identical to one planned before these knobs existed.

    The resolved PAIR is validated whenever either half is given, so
    ``--refine-iters`` on its own gets the shared orphan-option error rather than
    being recorded against a ``refine`` of "none" that never reads it.
    """
    from luxar.cli.gsplat_ops.recipe_shared import validate_refine

    if refine is None and refine_iters is None:
        return
    norm = validate_refine(refine, refine_iters)
    if refine is not None:
        overrides["refine"] = norm
    if refine_iters is not None:
        overrides["refine_iters"] = refine_iters


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
    refine: Optional[str] = None,
    refine_iters: Optional[int] = None,
) -> "RecipeParams":
    """Build a ``RecipeParams`` for the per-part merge recipe.

    Each knob is resolved CLI-first, then the value recorded at plan time
    (``manifest.merge_recipe_args``, string-valued), then the ``RecipeParams``
    default. ``coarsen_dims`` is parsed from a comma string to a tuple of ints;
    leaving it unset lets the merge default it per part (spatial dims only). The
    additive knobs (``additive_method`` / ``breakpoints``) bring
    the merge recipe to parity with ``fit --recipe`` and ``gsplat lod``.

    ``refine`` / ``refine_iters`` ARE exposed. ``refine="volume"`` re-opens the
    source the manifest recorded and crops it to each tile as that tile streams,
    so the merge is no longer volume-free; ``merge_batch_results`` validates the
    source up front, before any part is written, because a per-part failure
    mid-stream would leave a half-written store. Left unset, the ``RecipeParams``
    default ("none") keeps the merge byte-identical to before.
    """
    from luxar.cli.gsplat_ops.recipe_shared import (
        VALID_ADDITIVE_METHODS,
        VALID_SUBSTITUTIVE_METHODS,
        parse_lod_breakpoints,
    )
    from luxar.gsplats.lod.recipes import RecipeParams
    from luxar.utils.lod_methods import canonical_method_token

    # A manifest written before the 2026-08 method-flag rename stores
    # `substitutive-method` / `additive-method`. Left alone, `_resolve` would look
    # up the NEW token, miss, and silently fall back to the recipe default — the
    # planned method quietly replaced by `auto` on a resumed run. Normalising the
    # whole dict once covers every knob without per-key special-casing; the emit
    # path in `slurm_gen` translates for the same reason.
    stored = {canonical_method_token(k): v for k, v in stored.items()}

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
    am = _resolve("add-method", additive_method, str)
    if am is not None:
        am_norm = am.strip().replace("-", "_")
        if am_norm not in VALID_ADDITIVE_METHODS:
            raise typer.BadParameter(
                f"--add-method must be one of "
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
    sm = _resolve("subst-method", substitutive_method, str)
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
                f"--subst-method must be one of "
                f"{list(VALID_SUBSTITUTIVE_METHODS)}; got {sm!r}"
            )
        overrides["substitutive_method"] = sm_norm
    cd = coarsen_dims if coarsen_dims is not None else stored.get("coarsen-dims")
    if cd is not None:
        overrides["coarsen_dims"] = _parse_dims(cd)
    _apply_refine(
        overrides,
        _resolve("refine", refine, str),
        _resolve("refine-iters", refine_iters, int),
    )

    return RecipeParams(**overrides)
