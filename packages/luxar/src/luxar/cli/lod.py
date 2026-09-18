"""``luxar gsplat lod --recipe`` — build a representation topology from a fit.

A thin CLI wrapper over :func:`luxar.gsplats.lod.recipes.build_recipe`. It parses
the option superset, validates that the options given are relevant to the chosen
recipe, fills scale-derived defaults, loads the input, builds the recipe, and
writes the output ``.gsplats.zarr``.

The single ``--recipe`` flag builds one of the intent-first topologies
(``flat`` / ``stream`` / ``levels`` / ``tiles`` / ``overview`` / ``adaptive``);
it replaced the historical ``lod additive`` / ``lod substitutive`` /
``lod pyramid`` subcommands (now the ``stream`` / ``levels`` recipes).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import typer
from arbol import aprint, asection

# Under its former private spelling, so every call site in this module is
# untouched — the extraction is a move, not a rename.
from luxar.cli.reveal_options import parse_reveal_knobs as _parse_reveal_knobs
from luxar.utils.lod_methods import (
    GSPLAT_ADDITIVE_CHOICES_HELP,
    LEGACY_METHOD_FLAGS,
)

from ._traceback import exit_with_error

# Shared recipe/streaming validation surface — moved to
# gsplat_ops/recipe_shared.py (consumed by six gsplat_ops modules);
# re-imported here so the lod command and historical
# ``from luxar.cli.lod import <name>`` imports keep working.
from .gsplat_ops.recipe_shared import (
    VALID_ADDITIVE_METHODS,
    VALID_SUBSTITUTIVE_METHODS,
    carried_appearance,
    detect_store_encoding,
    estimate_bytes_per_splat,
    measure_store_bytes,
    parse_lod_breakpoints,
    reject_irrelevant_recipe_options,
    resolve_streaming_breakpoints,
    survey_gsplat_streaming_layout,
    validate_refine,
    validate_streaming_knobs,
)

_VALID_PARTITION_RULES = ("median", "midpoint", "sah")

#: Target splat count for the overview coarse cap when ``--compression-factor``
#: is not given. The cap is a SINGLE substitutive level of ``ceil(N / K)`` splats,
#: so a fixed default K scales badly: on a 23 M fit, the historical K=4/8 left a
#: 2.9 M-splat cap (far too heavy to load first). Instead derive
#: ``K = max(2, round(N / target))`` so the cap lands near this size regardless of
#: N. ~256 K keeps the coarsest level light enough to stream instantly while still
#: carrying enough detail to be a useful overview. Mirrors the N-aware
#: ``max_elements`` default used for the tiles branch.
_MULTISCALE_CAP_TARGET = 256_000

#: What ``--coarsen-dims`` does BEYOND steering the reduction, per recipe — the
#: second half of the >3D no-barrier warning below. ``levels`` stamps the
#: resolved choice and the writer reads it back, so the flag also fixes the
#: CHUNK layout — for the whole ladder, finest level included, even though that
#: level is the input unreduced and its own grid would have survived a per-level
#: guess. ``overview`` / ``adaptive`` publish no stamp at all, so the choice
#: never lands on disk and their layout stays guessed per level (#1600). A
#: lookup rather than a branch: this command is already at the C901 ceiling.
_COARSEN_STAMP_NOTES = {
    "levels": (
        "    With --recipe levels that choice is also published and fixes the "
        "chunk layout of the WHOLE ladder — the finest level included, though "
        "it is your input unreduced."
    ),
    "overview": (
        "    Note that this recipe does not publish the choice: it steers the "
        "reduction, but the chunk layout is still guessed per level."
    ),
}
_COARSEN_STAMP_NOTES["adaptive"] = _COARSEN_STAMP_NOTES["overview"]


def _resolve_lod_streaming_breakpoints(
    *,
    data: Any,
    input_path: Path,
    recipe: str,
    encoding: str,
    target_ms: float,
    bandwidth_mbps: Optional[float],
    bytes_per_splat: Optional[float],
    eff_max_elements: Optional[int],
    parts: Optional[int],
    parsed_coarsen: Optional[tuple],
) -> str:
    """Size a stream ladder from the barrier and partition this build writes."""
    import numpy as np

    stored_total = sum(
        data.at_substitutive(index).n_splats for index in range(data.n_substitutive)
    )
    store_bytes = measure_store_bytes(input_path)
    measured = (
        store_bytes / stored_total if store_bytes > 0 and stored_total > 0 else None
    )
    if measured is not None and encoding != "auto":
        input_encoding = detect_store_encoding(input_path)
        if input_encoding != encoding:
            aprint(
                f"--encoding {encoding} re-encodes the output "
                f"(input store looks {input_encoding or 'unknown'}-encoded); sizing "
                f"--target-ms from the analytic {encoding} estimate instead of "
                "the measured input bytes"
            )
            measured = None

    partitioned = recipe in ("tiles", "overview", "adaptive")
    part_count = (
        parts
        if partitioned and parts is not None
        else max(1, -(-data.n_splats // eff_max_elements))
        if partitioned and eff_max_elements is not None
        else 1
    )
    if recipe in ("levels", "overview", "adaptive"):
        hidden_dims = sorted(
            set(range(data.ndim)) - set(parsed_coarsen or range(data.ndim))
        )
        slice_count = (
            len(np.unique(data.centers[:, hidden_dims], axis=0)) if hidden_dims else 1
        )
    else:
        slice_count = survey_gsplat_streaming_layout(input_path)[0]
    return resolve_streaming_breakpoints(
        target_ms,
        bandwidth_mbps,
        bytes_per_splat,
        measured_bps=measured,
        analytic_bps=estimate_bytes_per_splat(
            data.ndim, data.colors is not None, encoding=encoding
        ),
        slice_count=max(1, slice_count),
        part_count=part_count,
    )


# Per-recipe relevance tokens. Each tuning option belongs to a token group; a
# recipe only accepts options whose token is in its allowed set. ``--levels`` is
# its own token because ``overview`` accepts the other substitutive options
# (for its coarse cap) but fixes the cap at a single level.
_OPTION_TOKENS = {
    "--n-lods": "additive",
    "--add-method": "additive",
    "--breakpoints": "additive",
    "--additive": "additive",
    "--target-ms": "additive",
    "--bandwidth-mbps": "additive",
    "--bytes-per-splat": "additive",
    "--truncation-sigmas": "additive",
    "--max-n-dense": "additive",
    "--reveal-center": "additive",
    "--spatial-dims": "additive",
    "--max-elements": "partition",
    "--parts": "partition",
    "--partition-rule": "partition",
    "--compression-factor": "substitutive",
    "--subst-method": "substitutive",
    "--lloyd-iters": "substitutive",
    "--candidate-bins-k": "substitutive",
    "--coverage-inflation": "substitutive",
    "--conserve-mass": "substitutive",
    "--refine": "substitutive",
    "--refine-iters": "substitutive",
    "--target": "substitutive",
    "--target-axes": "substitutive",
    "--channel": "substitutive",
    "--timepoint": "substitutive",
    "--array-key": "substitutive",
    "--coarsen-dims": "substitutive",
    "--quality-stamps": "substitutive",
    "--quality-max-pair-splats": "substitutive",
    "--levels": "levels",
}

_ALLOWED_TOKENS = {
    "flat": frozenset(),
    "stream": frozenset({"additive"}),
    "tiles": frozenset({"additive", "partition"}),
    # LOD switch thresholds are auto-derived as viewport-relative coverage
    # fractions (screen-occupancy halving) for every kind=lod group (the overview
    # cap and the levels/adaptive lod groups) — no threshold knob.
    "overview": frozenset({"additive", "partition", "substitutive"}),
    # adaptive: spatial tiles + a levels group per tile — partition +
    # level-merge knobs (--levels for per-tile depth), plus ladder knobs:
    # every per-tile level carries a stream ladder by default.
    "adaptive": frozenset({"additive", "partition", "substitutive", "levels"}),
    # levels: ladder knobs accepted too — every level is stream-laddered by
    # default (project convention: additive/stream LODs everywhere;
    # --no-additive restores bare per-level leaves).
    "levels": frozenset({"additive", "substitutive", "levels"}),
}


def _resolve_encoding(mode: str) -> Any:
    """Map an ``--encoding`` string to the ``EncodingMode`` enum.

    Args:
        mode: One of ``"auto"``, ``"precision"``, or ``"memory"``.

    Returns:
        The matching :class:`~luxar.encoding.EncodingMode` member.

    Raises:
        typer.BadParameter: If ``mode`` is not a recognised encoding name.
    """
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


def _reject_renamed_method_flags(*supplied: tuple[Optional[str], str]) -> None:
    """Reject a renamed method flag with a pointer naming its replacement.

    Extracted from ``lod_recipe`` rather than inlined: that function is baselined
    at C901 44 and the loop pushed it to 46, which the complexity ratchet reports
    as a regression (the gate is "no worse", not "under the limit"). Lifting it out
    keeps the caller's score unchanged and the check independently testable.

    Each argument pairs the value typer parsed for a hidden legacy option with that
    option's spelling. A non-``None`` value means the user typed the old flag.
    """
    for value, legacy in supplied:
        if value is None:
            continue
        new = LEGACY_METHOD_FLAGS[legacy]
        raise typer.BadParameter(
            f"{legacy} was renamed to {new} (2026-08: the additive and "
            f"substitutive method flags are now named symmetrically, and no "
            f"method flag is bare); use {new} {value}."
        )


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
            "flat | stream | levels | tiles | overview | adaptive."
        ),
    ),
    # ── stream (additive) ladder — every recipe ladders by default ──
    n_lods: Optional[int] = typer.Option(
        None, "--n-lods", min=1, help="Additive LOD levels (default 4)."
    ),
    method: Optional[str] = typer.Option(
        None,
        "--add-method",
        "-m",
        help=f"Additive ordering: {GSPLAT_ADDITIVE_CHOICES_HELP}. auto (the default) is "
        "greedy at small N, self_energy for large N to avoid greedy's "
        "O(N·nnz·logN) blowup. radial orders concentric shells around the "
        "bbox center, so a streaming prefix grows outward from the middle "
        "(a reveal); its ladder carries no energy stamps.",
    ),
    # Renamed spellings, declared only so the body can raise a pointer. Typer
    # rejects an unknown option before this function runs, so the
    # LEGACY_RECIPE_NAMES idiom (validate the VALUE, name the replacement) cannot
    # reach a removed FLAG at all — the option must exist to be reachable.
    #
    # Stated honestly: dropping these would NOT leave the user with nothing. Typer
    # already emits "No such option: --substitutive-method (Possible options:
    # --subst-method)", which is a usable hint. What these add is the *why* (a
    # deliberate rename, dated) and a copy-pasteable replacement carrying the
    # user's own value — worth ~6 lines, but an improvement on a decent default
    # rather than a rescue from silence.
    #
    # Hidden so they stay out of `--help`; any value at all triggers the pointer.
    legacy_method: Optional[str] = typer.Option(None, "--method", hidden=True),
    legacy_substitutive_method: Optional[str] = typer.Option(
        None, "--substitutive-method", hidden=True
    ),
    breakpoints: Optional[str] = typer.Option(
        None,
        "--breakpoints",
        "-b",
        help="'equal-count' (default) | 'stream:C' (geometric streaming ladder, "
        "first chunk C splats then doubling; sized per part/level) | "
        "'equi-energy:N' (N rungs at equal shares of cumulative self-energy: few "
        "heavy splats first, fatter rungs later; commit-capped) | "
        "'counts:N1,N2,...' | 'energy:f1,f2,...'.",
    ),
    target_ms: Optional[float] = typer.Option(
        None,
        "--target-ms",
        min=1.0,
        help="Streaming sizing: derive 'stream:<c>' breakpoints so the first "
        "additive chunk downloads in ~this many ms at --bandwidth-mbps "
        "(bytes/splat measured from the input store; override with "
        "--bytes-per-splat). Mutually exclusive with --breakpoints.",
    ),
    bandwidth_mbps: Optional[float] = typer.Option(
        None,
        "--bandwidth-mbps",
        min=0.1,
        help="Assumed downlink for --target-ms sizing (default 25, a typical "
        "broadband connection).",
    ),
    bytes_per_splat: Optional[float] = typer.Option(
        None,
        "--bytes-per-splat",
        min=0.1,
        help="Override the on-wire bytes/splat used by --target-ms sizing "
        "(default: measured from the input store).",
    ),
    truncation_sigmas: Optional[float] = typer.Option(
        None,
        "--truncation-sigmas",
        help="Mahalanobis cutoff for greedy (default: the dataset's own "
        "truncation radius).",
    ),
    max_n_dense: Optional[int] = typer.Option(
        None, "--max-n-dense", help="Greedy dense-Gram threshold (default 2000)."
    ),
    # ── spatial partition (tiles / overview / adaptive) ──
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
    # ── level merge (levels / overview cap / adaptive tiles) ──
    compression_factor: Optional[int] = typer.Option(
        None,
        "--compression-factor",
        "-K",
        min=2,
        help="Substitutive per-level compression factor (default 4; for "
        "--recipe overview, auto-scaled from N to a ~256K coarse cap "
        "when omitted).",
    ),
    levels: Optional[int] = typer.Option(
        None,
        "--levels",
        "-L",
        min=1,
        help="Coarser LOD levels (default 3). Not used by overview "
        "(its cap is a single level).",
    ),
    substitutive_method: Optional[str] = typer.Option(
        None,
        "--subst-method",
        help="Substitutive algorithm: auto (default) | kmeans | kmeans_lloyd | "
        "greedy | greedy_lloyd.",
    ),
    lloyd_iterations: Optional[int] = typer.Option(
        None, "--lloyd-iters", min=0, help="Lloyd refinement passes (default 5)."
    ),
    candidate_bins_k: Optional[int] = typer.Option(
        None, "--candidate-bins-k", min=1, help="Lloyd spatial-hash top-k (default 12)."
    ),
    coverage_inflation: Optional[float] = typer.Option(
        None,
        "--coverage-inflation",
        min=1.0,
        help="Widen each merged representative's inter-center spread by this "
        "factor (mass-preserving). Default 3.0 — calibrated so neighbouring "
        "representatives sum flat, suppressing the grid-pattern ripple that "
        "pure moment matching produces at coarse levels. 1.0 disables.",
    ),
    additive_ladders: Optional[bool] = typer.Option(
        None,
        "--additive/--no-additive",
        help="Additive ladder inside every substitutive level / part / cap "
        "(streaming-friendly first paint). ON by default everywhere; "
        "--no-additive emits bare leaves. Rejected for the stream and "
        "tiles recipes (their ladders are definitional).",
    ),
    conserve_mass: Optional[bool] = typer.Option(
        None,
        "--conserve-mass/--no-conserve-mass",
        help="Rescale each reduced level so its total mass over the coarsened "
        "dims matches its fine input (per barrier group) — keeps additive-"
        "render brightness constant across LOD switches. Default on.",
    ),
    refine: Optional[str] = typer.Option(
        None,
        "--refine",
        help="Post-merge refinement of each substitutive level: none (default) "
        "| l2 (Adam-optimize the level against its fine input under the "
        "closed-form mixture L2 — slower, higher fidelity, peak-preserving; "
        "total mass pinned so brightness never pops across levels) "
        "| volume (warm-start re-fit each level against the source volume "
        "given via --target — the highest-fidelity option; each level keeps "
        "whichever of merge/re-fit renders closer to the volume).",
    ),
    refine_iters: Optional[int] = typer.Option(
        None,
        "--refine-iters",
        min=1,
        help="Refinement steps per level (default 120 for --refine l2, 300 "
        "for --refine volume; requires --refine l2|volume).",
    ),
    target_path: Optional[Path] = typer.Option(
        None,
        "--target",
        exists=True,
        help="Source volume for --refine volume (.npy/.npz/.tiff/.zarr/"
        ".zarr.zip; the volume the splats were fitted from). Its voxel "
        "coordinate frame must match the splats'.",
    ),
    target_channel: Optional[int] = typer.Option(
        None,
        "--channel",
        help="Channel to extract from a multi-channel --target volume.",
    ),
    target_timepoint: Optional[int] = typer.Option(
        None,
        "--timepoint",
        help="Timepoint to extract from a time-series --target volume.",
    ),
    target_axes: Optional[str] = typer.Option(
        None,
        "--target-axes",
        help="Per-dimension labels for a --target that KEEPS its stacked axis "
        "(e.g. 'time,z,y,x'), so a --refine volume of a stacked timelapse can "
        "walk that axis one slice per timepoint. Without this a >3D target is "
        "assumed to be in the splats' own dim order. Contrast --timepoint, "
        "which slices a single timepoint out instead.",
    ),
    target_array_key: Optional[str] = typer.Option(
        None,
        "--array-key",
        help="Array path inside a nested --target zarr group (e.g. 'a/fused').",
    ),
    reveal_center: Optional[str] = typer.Option(
        None,
        "--reveal-center",
        help="[-m radial] Comma-separated center of the concentric shells, one "
        "coordinate per spatial axis. Default: the dataset's own bounding-box "
        "center — NOT the scene origin, so a dataset far from the origin still "
        "reveals from its own middle. On a PARTITIONED recipe "
        "(tiles/overview/adaptive) the ladder is built per part, so the default "
        "centers EACH PART on itself (N local reveals); pass this flag to make "
        "the whole object grow from one point.",
    ),
    spatial_dims: Optional[str] = typer.Option(
        None,
        "--spatial-dims",
        help="[-m radial] Comma-separated center-column indices the shell "
        "distance is measured over. Default: the non-degenerate axes, so a "
        "stacked time/channel axis cannot become a shell dimension (shells "
        "would otherwise expand through time as well as space).",
    ),
    coarsen_dims: Optional[str] = typer.Option(
        None,
        "--coarsen-dims",
        help="Comma-separated center-column indices coarsening may merge over; "
        "the remaining dims become hard grouping barriers (e.g. a categorical / "
        "timepoint / channel axis). Default: all dims. (Standalone gsplats carry "
        "no display info, so pass explicit indices here.)",
    ),
    quality_stamps: Optional[bool] = typer.Option(
        None,
        "--quality-stamps/--no-quality-stamps",
        help="Measure each coarse substitutive level's mixture-L2 quality Q vs "
        "its group's finest content and stamp it (with the reference-energy "
        "weight w) into the level stats — the viewer folds Q with the per-chunk "
        "committed-energy fraction e(k) into a recursive quality estimate. "
        "Constant-cost sampled estimator; ON by default.",
    ),
    quality_max_pair_splats: Optional[int] = typer.Option(
        None,
        "--quality-max-pair-splats",
        min=1,
        help="Subsample cap per mixture for the quality measurement "
        "(default 2,000,000 splats; lower = faster, noisier Q).",
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

    One recipe, scale-ordered; every recipe streams by default (each leaf
    carries a progressive "stream" ladder unless --no-additive). Pick by what
    you need::

      \b
        recipe     structure                        use when
        flat       one bare leaf                    tiny data / debugging
        stream     one leaf + progressive ladder    small data, fast first paint
        levels     coarse->fine replacement levels  zooming across scales
        tiles      spatial tiles (culled), each     large scene, one scale
                   with its own ladder
        overview   instant coarse overview level    huge scene, "see everything
                   + fine tiles on zoom             first" (detail where you look)
        adaptive   tiles where EVERY tile picks     largest scenes, locally
                   its own detail level             adaptive detail

    Scale is not the only axis. When first paint is request-constrained, the
    recipe is picked by how many levels and tiles the viewer fetches EAGERLY,
    not by N: overview defers its whole fine branch behind a selector, while
    tiles and adaptive load every part at once. Measured on one 1.58 GiB 4D
    timelapse, an adaptive build (44 parts) needed ~18x more requests to first
    paint than the same data as one stacked leaf with a stream ladder.

    (Renamed 2026-07: additive->stream, substitutive/pyramid->levels,
    partitioned->tiles, multiscale->overview, mosaic->adaptive.)

    Input must be a fitted / flat .gsplats.zarr (output of ``luxar gsplat fit``).
    Canonical pipeline: ``cal`` -> ``fit --seeds K*`` -> ``lod --recipe ...``.

    \b
    Examples:
        luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe stream --n-lods 6

        luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe tiles \\
            --max-elements 250000

        luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe overview \\
            --compression-factor 8
    """
    from luxar.gsplats.gsplat_data import GSplatData, stats_after_structure_change
    from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
    from luxar.gsplats.lod.recipes import (
        RECIPE_NAMES,
        RecipeParams,
        build_recipe,
    )

    try:
        # ── validate --recipe ──
        # Renamed method flags, before anything else — a run that named an old
        # spelling asked for something this command no longer has, and letting it
        # proceed under the flag's DEFAULT would silently do different work than
        # requested (`--method greedy` would become `auto`). Mirrors the legacy
        # recipe-name rejection below; the difference is that a flag has to be
        # DECLARED to be diagnosable at all, so the two hidden options above exist
        # purely to reach this check.
        _reject_renamed_method_flags(
            (legacy_method, "--method"),
            (legacy_substitutive_method, "--substitutive-method"),
        )
        if recipe is None:
            raise typer.BadParameter(
                "--recipe is required; choose one of: " + ", ".join(RECIPE_NAMES)
            )
        if recipe not in RECIPE_NAMES:
            from luxar.gsplats.lod.recipes import LEGACY_RECIPE_NAMES

            if recipe in LEGACY_RECIPE_NAMES:
                raise typer.BadParameter(
                    f"recipe {recipe!r} was renamed to "
                    f"{LEGACY_RECIPE_NAMES[recipe]!r} (2026-07 intent-first "
                    f"vocabulary); use --recipe {LEGACY_RECIPE_NAMES[recipe]}."
                )
            raise typer.BadParameter(
                f"unknown recipe {recipe!r}; choose one of: " + ", ".join(RECIPE_NAMES)
            )

        # ── option-relevance check (reject options irrelevant to the recipe) ──
        provided = {
            "--n-lods": n_lods,
            "--add-method": method,
            "--breakpoints": breakpoints,
            "--additive": additive_ladders,
            "--target-ms": target_ms,
            "--bandwidth-mbps": bandwidth_mbps,
            "--bytes-per-splat": bytes_per_splat,
            "--truncation-sigmas": truncation_sigmas,
            "--max-n-dense": max_n_dense,
            "--max-elements": max_elements,
            "--parts": parts,
            "--partition-rule": partition_rule,
            "--compression-factor": compression_factor,
            "--subst-method": substitutive_method,
            "--lloyd-iters": lloyd_iterations,
            "--candidate-bins-k": candidate_bins_k,
            "--coverage-inflation": coverage_inflation,
            "--conserve-mass": conserve_mass,
            "--refine": refine,
            "--refine-iters": refine_iters,
            "--target": target_path,
            "--channel": target_channel,
            "--timepoint": target_timepoint,
            "--array-key": target_array_key,
            "--target-axes": target_axes,
            "--reveal-center": reveal_center,
            "--spatial-dims": spatial_dims,
            "--coarsen-dims": coarsen_dims,
            "--quality-stamps": quality_stamps,
            "--quality-max-pair-splats": quality_max_pair_splats,
            "--levels": levels,
        }
        # ``hints`` add a recipe-specific clause when a given flag is rejected:
        #  - --add-method's value naming a substitutive algorithm points at
        #    --subst-method (see the cross-hint below);
        #  - overview's coarse cap is single-level, so --levels has no meaning
        #    there (size the cap with -K, auto-scaled by default).
        hints: dict[str, str] = {}
        if recipe == "overview":
            hints["--levels"] = (
                "(overview's coarse cap is single-level; size it with "
                "--compression-factor/-K, auto-scaled by default)"
            )
        reject_irrelevant_recipe_options(
            recipe, provided, _OPTION_TOKENS, _ALLOWED_TOKENS, hints=hints
        )

        # ── validate values ──
        method_norm = (method or "auto").strip().replace("-", "_")
        if method_norm not in VALID_ADDITIVE_METHODS:
            msg = (
                f"--add-method must be one of {list(VALID_ADDITIVE_METHODS)}; "
                f"got {method!r}"
            )
            if method_norm in VALID_SUBSTITUTIVE_METHODS:
                # `-m kmeans_lloyd` etc.: the user almost certainly meant the
                # substitutive partition algorithm (--add-method is the ADDITIVE
                # ordering — every substitutive level is laddered by default).
                msg += " (for the substitutive algorithm use --subst-method)"
            raise typer.BadParameter(msg)
        sub_norm = (substitutive_method or "auto").strip().replace("-", "_")
        if sub_norm not in VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--subst-method must be one of "
                f"{list(VALID_SUBSTITUTIVE_METHODS)}; got {substitutive_method!r}"
            )
        if additive_ladders is False and recipe in ("stream", "tiles"):
            raise typer.BadParameter(
                f"--no-additive contradicts --recipe {recipe}: its additive "
                "ladder is the recipe's definition."
            )
        # Same mode vocabulary and orphan-option rule as `fit --recipe` and the
        # batch merge — one validator, so the three spellings cannot drift.
        refine_norm = validate_refine(refine, refine_iters)
        if refine_norm == "volume" and target_path is None:
            raise typer.BadParameter(
                "--refine volume needs the source volume: pass --target <volume>."
            )
        if target_path is not None and refine_norm != "volume":
            raise typer.BadParameter(
                "--target is only consumed by --refine volume; pass --refine "
                "volume to re-fit the coarse levels against it."
            )
        orphan_selectors = [
            flag
            for flag, value in (
                ("--channel", target_channel),
                ("--timepoint", target_timepoint),
                ("--array-key", target_array_key),
                ("--target-axes", target_axes),
            )
            if value is not None
        ]
        if orphan_selectors and target_path is None:
            raise typer.BadParameter(
                f"option(s) {', '.join(orphan_selectors)} select a sub-volume "
                "of --target, but no --target was given."
            )
        if target_axes is not None and (
            target_channel is not None or target_timepoint is not None
        ):
            raise typer.BadParameter(
                "--target-axes KEEPS the target's stacked axis so --refine volume "
                "can walk it one barrier group at a time, while --channel/"
                "--timepoint slice one index out and drop it — the labels would "
                "no longer describe the array. Use one or the other."
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
        validate_streaming_knobs(
            target_ms, bandwidth_mbps, bytes_per_splat, breakpoints
        )
        bp = parse_lod_breakpoints(breakpoints or "equal-count")
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
                # A recipe rebuild owns the STRUCTURE, not the appearance: carry
                # the source root's authored attrs across or they are silently
                # replaced by the writer's defaults (#1600).
                source_appearance = carried_appearance(input_path)
                # ...and owning the structure is exactly why the input's own
                # TOPOLOGY record must not ride along. An already-LOD store is a
                # legal input here (the gate is matrix-shaped-ness, so a
                # partition and a lod group with non-leaf children are out), and every
                # recipe starts from `data.flattened()` — so nothing this command
                # publishes preserves the input's shape. Scrubbed ONCE, on the
                # loaded input, which covers BOTH write paths below: the matrix
                # builders derive `result.stats` from `dict(src.stats)` and then
                # stamp their own true record over it, and the composed path
                # hands this same dict to `split_fitting_info`. Without it a
                # `--recipe flat` leaf published `lod_kind: substitutive` /
                # `n_substitutive_levels: 4` / `lod_cutpoints: [...]` one line
                # above its own `recipe: flat` (#1600). Only the builder's own
                # stamps survive, so a recipe that stamps nothing says nothing —
                # absence is the format's "this artifact does not know".
                # `coarsen_dims` is exempt from the scrub, so a `levels` build
                # keeps its barrier provenance.
                data = GSplatData.from_tree(
                    data.tree, stats=stats_after_structure_change(data.stats)
                )

            # ── --refine volume: load the source volume (shared loader) ──
            target_volume: Any = None
            target_volume_axes = None
            if target_path is not None:
                from luxar.cli.gsplat_config import load_volume

                # A --target-axes target keeps its stacked axis, and the re-fit
                # then only ever SLICES it (one barrier group at a time). Open it
                # lazily so that stays true on disk as well as in principle: a
                # 253-timepoint 407x2048x2048 uint16 timelapse is 431 GB while one
                # timepoint is 3.4 GB. The selector flags are the eager case by
                # definition (they reduce the array before the fit sees it), and
                # they cannot be combined with --target-axes anyway.
                lazy = (
                    target_axes is not None
                    and target_channel is None
                    and target_timepoint is None
                )
                with asection(f"Loading target volume: {target_path.name}"):
                    if lazy:
                        from luxar.io.volume import open_volume_lazy

                        target_volume = open_volume_lazy(
                            target_path, array_key=target_array_key
                        )
                    else:
                        target_volume = load_volume(
                            target_path,
                            channel=target_channel,
                            timepoint=target_timepoint,
                            array_key=target_array_key,
                        )
                    aprint(f"Volume shape: {target_volume.shape}")
                if len(target_volume.shape) != data.ndim:
                    raise typer.BadParameter(
                        f"--target volume is {len(target_volume.shape)}D but the "
                        f"splats are {data.ndim}D; select a matching sub-volume "
                        f"with --channel/--timepoint/--array-key."
                    )
                if target_axes is not None:
                    from luxar.io.volume import volume_axes_from_spec

                    target_volume_axes = volume_axes_from_spec(target_axes, data.ndim)
                    aprint(
                        f"Target axis map (center dim -> volume axis): "
                        f"{target_volume_axes}"
                    )

            # ── scale-derived defaults (logged) ──
            eff_max_elements: Optional[int] = max_elements
            if recipe in ("tiles", "overview", "adaptive"):
                # BSP partitioning needs >= 2 spatial dims (matching
                # spatial_bsp_tree's shape[1] < 2 guard); only 1D is rejected.
                # Fail cleanly (the rest of the command's validation style)
                # rather than letting the deeper ValueError surface as a raw
                # traceback.
                if data.ndim < 2:
                    raise typer.BadParameter(
                        f"recipe '{recipe}' requires >=2 spatial dimensions for "
                        f"BSP partitioning; got {data.ndim}D. Use --recipe "
                        f"stream or levels for {data.ndim}D data."
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

            # overview's coarse cap is a SINGLE merged level of ceil(N/K)
            # splats. A fixed default K scales badly (K=8 on 23 M → a 2.9 M cap),
            # so when --compression-factor is not given, size K to land the cap
            # near _MULTISCALE_CAP_TARGET. Explicit -K always wins.
            eff_compression_factor: Optional[int] = compression_factor
            if recipe == "overview" and compression_factor is None:
                eff_compression_factor = max(
                    2, round(data.n_splats / _MULTISCALE_CAP_TARGET)
                )
                cap_n = -(-data.n_splats // eff_compression_factor)  # ceil
                aprint(
                    f"--compression-factor defaulting to {eff_compression_factor} "
                    f"(coarse cap ~{cap_n:,} splats, target "
                    f"~{_MULTISCALE_CAP_TARGET:,}); pass -K to override"
                )

            parsed_reveal_center, parsed_spatial_dims = _parse_reveal_knobs(
                reveal_center, spatial_dims, method_norm, data.ndim
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
            elif data.ndim > 3 and recipe in _COARSEN_STAMP_NOTES:
                aprint(
                    f"  ⚠ {data.ndim}D input with no --coarsen-dims: substitutive "
                    "coarsening will merge across ALL dims. If some dims are "
                    "categorical/sliced (time/channel/...), pass --coarsen-dims "
                    "with the spatial column indices to keep them as barriers."
                )
                # ...and what the flag does beyond that, which differs by
                # recipe — see `_COARSEN_STAMP_NOTES`.
                aprint(_COARSEN_STAMP_NOTES[recipe])

            # ── streaming breakpoints from --target-ms (measured B/splat) ──
            if target_ms is not None:
                bp = _resolve_lod_streaming_breakpoints(
                    data=data,
                    input_path=input_path,
                    recipe=recipe,
                    encoding=encoding,
                    target_ms=target_ms,
                    bandwidth_mbps=bandwidth_mbps,
                    bytes_per_splat=bytes_per_splat,
                    eff_max_elements=eff_max_elements,
                    parts=parts,
                    parsed_coarsen=parsed_coarsen,
                )

            params = RecipeParams(
                n_lods=n_lods if n_lods is not None else 4,
                additive_method=method_norm,  # type: ignore[arg-type]
                reveal_center=parsed_reveal_center,
                spatial_dims=parsed_spatial_dims,
                breakpoints=bp,
                # Passed through as-is: None means "the dataset's own
                # truncation_radius", which the additive builders resolve.
                truncation_sigmas=truncation_sigmas,
                max_n_dense=max_n_dense if max_n_dense is not None else 2000,
                max_elements=eff_max_elements,
                partition_rule=rule,  # type: ignore[arg-type]
                compression_factor=(
                    eff_compression_factor if eff_compression_factor is not None else 4
                ),
                levels=levels if levels is not None else 3,
                substitutive_method=sub_norm,
                lloyd_iterations=lloyd_iterations
                if lloyd_iterations is not None
                else 5,
                candidate_bins_k=candidate_bins_k
                if candidate_bins_k is not None
                else 12,
                coverage_inflation=coverage_inflation
                if coverage_inflation is not None
                else 3.0,
                additive_ladders=additive_ladders
                if additive_ladders is not None
                else True,
                conserve_mass=conserve_mass if conserve_mass is not None else True,
                refine=refine_norm,
                # None resolves inside make_substitutive_lod to the engine's
                # own default (l2: 120, volume: 300) — single source of truth.
                refine_iters=refine_iters,
                volume=target_volume,
                volume_axes=target_volume_axes,
                coarsen_dims=parsed_coarsen,
                quality_stamps=quality_stamps if quality_stamps is not None else True,
                quality_max_pair_splats=(
                    quality_max_pair_splats
                    if quality_max_pair_splats is not None
                    else 2_000_000
                ),
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
                    # A multi-level result gets per-level coverage_fraction
                    # thresholds derived at tree-build time in save(). Record which
                    # RECIPE built this (build provenance) alongside the mechanism
                    # ``lod_kind`` the builder already stamped — the viewer consumes
                    # the on-disk ``kind`` attrs, not this recipe name.
                    result.stats["recipe"] = recipe
                    result.save(
                        output_path,
                        ordering=ordering,  # type: ignore[arg-type]
                        encoding_mode=encoding_obj,
                        amplitude_bits="auto",
                        compress=compress,  # type: ignore[arg-type]
                        root_attrs=source_appearance,
                    )
                else:
                    # Composed recipe — write the node tree, carrying input
                    # provenance plus the recipe that built it.
                    fitting_info, fitting_config, provenance_info, pipeline_info = (
                        split_fitting_info(data.stats)
                    )
                    pipeline_info = {**(pipeline_info or {}), "recipe": recipe}
                    write_gsplats_tree(
                        output_path,
                        result,
                        ordering=ordering,  # type: ignore[arg-type]
                        encoding_mode=encoding_obj,
                        amplitude_bits="auto",
                        source_dtype=data.stats.get("source_dtype"),
                        fitting_info=fitting_info,
                        fitting_config=fitting_config,
                        provenance_info=provenance_info,
                        pipeline_info=pipeline_info,
                        compress=compress,  # type: ignore[arg-type]
                        root_attrs=source_appearance,
                    )
                if not quiet:
                    aprint(f"Saved to {output_path}")

    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as e:  # noqa: BLE001 - exit_with_error reports the cause and exits.
        exit_with_error(f"Error: {e}", e)


# Re-exported for tests / introspection.
__all__ = ["lod_recipe", "register_lod_command"]
