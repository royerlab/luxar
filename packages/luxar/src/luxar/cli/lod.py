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

import os
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

#: Target splat count for the overview coarse cap when ``--compression-factor``
#: is not given. The cap is a SINGLE substitutive level of ``ceil(N / K)`` splats,
#: so a fixed default K scales badly: on a 23 M fit, the historical K=4/8 left a
#: 2.9 M-splat cap (far too heavy to load first). Instead derive
#: ``K = max(2, round(N / target))`` so the cap lands near this size regardless of
#: N. ~256 K keeps the coarsest level light enough to stream instantly while still
#: carrying enough detail to be a useful overview. Mirrors the N-aware
#: ``max_elements`` default used for the tiles branch.
_MULTISCALE_CAP_TARGET = 256_000

# Per-recipe relevance tokens. Each tuning option belongs to a token group; a
# recipe only accepts options whose token is in its allowed set. ``--levels`` is
# its own token because ``overview`` accepts the other substitutive options
# (for its coarse cap) but fixes the cap at a single level.
_OPTION_TOKENS = {
    "--n-lods": "additive",
    "--method": "additive",
    "--breakpoints": "additive",
    "--additive": "additive",
    "--target-ms": "additive",
    "--bandwidth-mbps": "additive",
    "--bytes-per-splat": "additive",
    "--truncation-sigmas": "additive",
    "--max-n-dense": "additive",
    "--max-elements": "partition",
    "--parts": "partition",
    "--partition-rule": "partition",
    "--compression-factor": "substitutive",
    "--substitutive-method": "substitutive",
    "--lloyd-iters": "substitutive",
    "--candidate-bins-k": "substitutive",
    "--coverage-inflation": "substitutive",
    "--conserve-mass": "substitutive",
    "--refine": "substitutive",
    "--refine-iters": "substitutive",
    "--target": "substitutive",
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
    # fractions (sqrt(N_i/N_finest)) for every kind=lod group (the overview
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
        msg = f"option(s) {', '.join(irrelevant)} are not used by --recipe {recipe}."
        if hints:
            for flag, clause in hints.items():
                if flag in irrelevant:
                    msg += " " + clause
    raise typer.BadParameter(msg)


def _parse_lod_breakpoints(spec: str) -> "str | list[int] | list[float]":
    """Parse the ``--breakpoints`` string for :func:`make_additive_lod`.

    Accepted forms: ``equal-count`` → literal; ``stream:14000`` → passed
    through as a string (a bandwidth-derived geometric ladder, resolved
    per-N inside the builder — the first chunk is ``<c>`` splats, then
    doubling); ``counts:5,10,15`` → ``[int]`` (cumulative splat counts);
    ``energy:0.5,0.9,1.0`` → ``[float]`` (cumulative energy fractions in
    (0, 1]).
    """
    s = spec.strip()
    if s == "equal-count":
        return "equal-count"
    if s.startswith("stream:"):
        body = s[len("stream:") :]
        try:
            first_chunk = int(body)
        except ValueError as e:
            raise typer.BadParameter(
                f"stream breakpoints must be 'stream:<integer>'; got {body!r}"
            ) from e
        if first_chunk < 1:
            raise typer.BadParameter(
                f"stream first-chunk size must be >= 1; got {first_chunk}"
            )
        # Pass the validated string through — it is resolved per-N inside
        # _resolve_breakpoints (each part/level sizes its own ladder), and the
        # string form round-trips the batch manifest verbatim.
        return s
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
        f"breakpoints must be 'equal-count', 'stream:<c>', 'counts:...', or "
        f"'energy:...'; got {spec!r}"
    )


#: Assumed stored bytes per scalar for (centers, amplitudes, cholesky) under
#: each encoding mode (see :func:`estimate_bytes_per_splat`). AUTO and MEMORY
#: write the SAME widths: centers u16 (coordinates never drop to u8),
#: amplitude ~u16 (width picked from dynamic range identically in both modes),
#: cholesky u8 (AUTO certified — escalation to u16 is the exception, not the
#: model; MEMORY unconditional). PRECISION: float32 everywhere.
_ENCODING_ARRAY_BYTES = {
    "auto": (2.0, 2.0, 1.0),
    "precision": (4.0, 4.0, 4.0),
    "memory": (2.0, 2.0, 1.0),
}


def estimate_bytes_per_splat(
    ndim: int, has_colors: bool = False, encoding: str = "auto"
) -> float:
    """Analytic on-wire bytes/splat estimate for an encoding mode.

    Per-array model: centers (d scalars), amplitude (1), split-Cholesky
    diag/offdiag (d(d+1)/2) each get the per-mode byte width from
    ``_ENCODING_ARRAY_BYTES`` (AUTO cholesky is u8 under the covariance
    certificate), and the store adds zarr/blosc/chunk-bounds overhead of
    roughly ×1.5 — calibrated against a real 4D fit that measured
    ~45 B/splat when everything was u16. Colors add ~4 B (u8 RGB + overhead;
    ~18 B as float32 under PRECISION). A crude estimate by design: used only
    when no matching store exists to measure (``fit --recipe``, or when
    ``--encoding`` re-encodes the output); the ``--bytes-per-splat`` override
    is the escape hatch, and the assumed value is always logged.
    """
    k = ndim * (ndim + 1) // 2
    center_b, amp_b, chol_b = _ENCODING_ARRAY_BYTES.get(encoding, (2.0, 2.0, 1.0))
    color_bytes = 18.0 if encoding == "precision" else 4.0
    return round(
        1.5 * (center_b * ndim + amp_b + chol_b * k)
        + (color_bytes if has_colors else 0.0),
        1,
    )


def measure_store_bytes(path: Path) -> int:
    """Total on-disk bytes of a ``.gsplats.zarr`` store (dir walk; ≈ wire cost).

    Zarr chunks are served as-is over HTTP, so store bytes / stored splats is
    the true average network cost per splat. Returns 0 for a non-directory
    (e.g. a ``.zip`` archive path) — callers fall back to the analytic estimate.
    """
    if not path.is_dir():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:  # pragma: no cover - race with concurrent writers
                pass
    return total


def detect_store_encoding(path: Path) -> Optional[str]:
    """Classify a ``.gsplats.zarr`` store's encoding mode from its on-disk attrs.

    Reads the split-Cholesky arrays' ``encoding`` attrs. The AUTO writer
    quantizes to u8 (escalating to u16 when its covariance certificate
    demands) and records the measured ``certificate`` as provenance; MEMORY is
    u8 WITHOUT a certificate; PRECISION stores plain ``float32``. So the
    certificate key — not the bit width — separates AUTO from MEMORY, and a
    bare u16 (legacy pre-certificate store) is AUTO. Uniform-cholesky stores
    broadcast the factors (no signal), so the amplitudes array is the
    fallback — it still separates PRECISION (``float32``) from the quantized
    modes (u8 in both auto and memory, hence not discriminative). Returns
    ``None`` when the store cannot be classified (zip archive, legacy layout,
    broadcast-only quantized store) — callers should then not assume a mode.
    """
    import json
    from typing import Iterator

    if not path.is_dir():
        return None

    def _encodings(array: str) -> Iterator[dict]:
        for zattrs in sorted(path.rglob(f"{array}/.zattrs")):
            try:
                enc = json.loads(zattrs.read_text()).get("encoding", {})
            except (OSError, json.JSONDecodeError, AttributeError):
                continue
            if isinstance(enc, dict) and enc.get("name"):
                yield enc

    for array in ("cholesky_factors_diag", "cholesky_factors_offdiag"):
        for enc in _encodings(array):
            name = str(enc["name"])
            certified = "certificate" in enc
            if name.endswith("_u16"):
                return "auto"  # escalated-AUTO, or legacy AUTO (pre-certificate)
            if name.endswith("_u8"):
                return "auto" if certified else "memory"
            if name == "float32":
                return "auto" if certified else "precision"
    for enc in _encodings("amplitudes"):
        if str(enc["name"]) == "float32":
            return "precision"
    return None


def validate_streaming_knobs(
    target_ms: Optional[float],
    bandwidth_mbps: Optional[float],
    bytes_per_splat: Optional[float],
    breakpoints: Optional[str],
    *,
    prefix: str = "--",
) -> None:
    """Reject contradictory / orphaned streaming-sizing knobs.

    Shared by every surface exposing the ``--target-ms`` trio (``gsplat lod``,
    ``gsplat additive``, ``fit --recipe``, ``batch-fit submit/run/merge``):
    ``--target-ms`` *derives* the breakpoints, so an explicit ``--breakpoints``
    alongside it is contradictory; and the supporting knobs are meaningless
    without ``--target-ms`` (loud, not silently ignored). ``prefix`` renames
    the options in the messages (e.g. ``"--merge-"`` for the batch plan-time
    surface). Raises :class:`typer.BadParameter` on violation.
    """
    if target_ms is not None and breakpoints is not None:
        raise typer.BadParameter(
            f"{prefix}target-ms and {prefix}breakpoints are mutually exclusive "
            f"({prefix}target-ms derives the breakpoints)."
        )
    if target_ms is None and (
        bandwidth_mbps is not None or bytes_per_splat is not None
    ):
        raise typer.BadParameter(
            f"{prefix}bandwidth-mbps/{prefix}bytes-per-splat only apply with "
            f"{prefix}target-ms."
        )


def resolve_streaming_breakpoints(
    target_ms: float,
    bandwidth_mbps: Optional[float],
    bytes_per_splat: Optional[float],
    *,
    measured_bps: Optional[float] = None,
    analytic_bps: Optional[float] = None,
    measured_label: str = "measured from input store",
) -> str:
    """Resolve ``--target-ms``/``--bandwidth-mbps`` into a ``stream:<c>`` spec.

    Shared by every CLI surface that builds additive ladders. Bytes/splat
    priority: explicit ``--bytes-per-splat`` override > measured from the
    input store(s) > analytic estimate. Logs the derivation (mirrors the
    N-aware overview-K message) so the assumed numbers are always visible;
    ``measured_label`` names the measurement source in that log line (e.g.
    "measured from 8 completed tile stores" for ``batch-fit merge``).
    """
    from luxar.gsplats.lod.additive import (
        DEFAULT_BANDWIDTH_MBPS,
        streaming_chunk_splats,
    )

    bw = bandwidth_mbps if bandwidth_mbps is not None else DEFAULT_BANDWIDTH_MBPS
    if bytes_per_splat is not None:
        bps, source = float(bytes_per_splat), "override"
    elif measured_bps is not None and measured_bps > 0:
        bps, source = float(measured_bps), measured_label
    elif analytic_bps is not None and analytic_bps > 0:
        bps, source = float(analytic_bps), "analytic estimate"
    else:
        raise typer.BadParameter(
            "--target-ms needs a bytes-per-splat figure; pass --bytes-per-splat"
        )
    try:
        c = streaming_chunk_splats(target_ms, bw, bps)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    aprint(
        f"--target-ms {target_ms:g} @ {bw:g} Mbps, {bps:.1f} B/splat "
        f"({source}) -> stream:{c} (first chunk ~{c:,} splats, "
        f"~{c * bps / 1024:.0f} KB)"
    )
    return f"stream:{c}"


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
            "flat | stream | levels | tiles | overview | adaptive."
        ),
    ),
    # ── stream (additive) ladder — every recipe ladders by default ──
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
        help="'equal-count' (default) | 'stream:C' (geometric streaming ladder, "
        "first chunk C splats then doubling; sized per part/level) | "
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
        None, "--truncation-sigmas", help="Mahalanobis cutoff for greedy (default 3.0)."
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
    target_array_key: Optional[str] = typer.Option(
        None,
        "--array-key",
        help="Array path inside a nested --target zarr group (e.g. 'a/fused').",
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
            "--method": method,
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
            "--substitutive-method": substitutive_method,
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
            "--coarsen-dims": coarsen_dims,
            "--quality-stamps": quality_stamps,
            "--quality-max-pair-splats": quality_max_pair_splats,
            "--levels": levels,
        }
        # ``hints`` add a recipe-specific clause when a given flag is rejected:
        #  - --method moved to --substitutive-method for the substitutive recipe;
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
        if method_norm not in _VALID_ADDITIVE_METHODS:
            msg = (
                f"--method must be one of {list(_VALID_ADDITIVE_METHODS)}; "
                f"got {method!r}"
            )
            if method_norm in _VALID_SUBSTITUTIVE_METHODS:
                # `-m kmeans_lloyd` etc.: the user almost certainly meant the
                # substitutive partition algorithm (--method is the ADDITIVE
                # ordering — every substitutive level is laddered by default).
                msg += " (for the substitutive algorithm use --substitutive-method)"
            raise typer.BadParameter(msg)
        sub_norm = (substitutive_method or "auto").strip().replace("-", "_")
        if sub_norm not in _VALID_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"--substitutive-method must be one of "
                f"{list(_VALID_SUBSTITUTIVE_METHODS)}; got {substitutive_method!r}"
            )
        if additive_ladders is False and recipe in ("stream", "tiles"):
            raise typer.BadParameter(
                f"--no-additive contradicts --recipe {recipe}: its additive "
                "ladder is the recipe's definition."
            )
        refine_norm = (refine or "none").strip()
        if refine_norm not in ("none", "l2", "volume"):
            raise typer.BadParameter(
                f"--refine must be 'none', 'l2', or 'volume'; got {refine!r}"
            )
        if refine_iters is not None and refine_norm == "none":
            raise typer.BadParameter(
                "--refine-iters only applies with --refine l2|volume."
            )
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
            )
            if value is not None
        ]
        if orphan_selectors and target_path is None:
            raise typer.BadParameter(
                f"option(s) {', '.join(orphan_selectors)} select a sub-volume "
                "of --target, but no --target was given."
            )
        if refine_norm == "volume" and recipe == "adaptive":
            raise typer.BadParameter(
                "--refine volume is not supported for --recipe adaptive: each "
                "tile's levels would re-fit against the full volume, pulling "
                "splats out of their tile. Use --recipe levels/overview, or "
                "--refine l2."
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

            # ── --refine volume: load the source volume (shared loader) ──
            target_volume = None
            if target_path is not None:
                from luxar.cli.gsplat_config import load_volume

                with asection(f"Loading target volume: {target_path.name}"):
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

            # ── streaming breakpoints from --target-ms (measured B/splat) ──
            if target_ms is not None:
                stored_total = sum(
                    data.at_substitutive(s).n_splats for s in range(data.n_substitutive)
                )
                store_bytes = measure_store_bytes(input_path)
                measured = (
                    store_bytes / stored_total
                    if store_bytes > 0 and stored_total > 0
                    else None
                )
                # The output is re-encoded per --encoding: when an explicit
                # non-default mode differs from the input's stored encoding,
                # the measured input bytes misstate the on-wire OUTPUT cost
                # (e.g. u16 input + --encoding precision ≈ 2× the measured
                # figure), so size against the analytic estimate for the
                # target encoding instead.
                if measured is not None and encoding != "auto":
                    input_encoding = detect_store_encoding(input_path)
                    if input_encoding != encoding:
                        aprint(
                            f"--encoding {encoding} re-encodes the output "
                            f"(input store looks "
                            f"{input_encoding or 'unknown'}-encoded); sizing "
                            f"--target-ms from the analytic {encoding} "
                            f"estimate instead of the measured input bytes"
                        )
                        measured = None
                bp = resolve_streaming_breakpoints(
                    target_ms,
                    bandwidth_mbps,
                    bytes_per_splat,
                    measured_bps=measured,
                    analytic_bps=estimate_bytes_per_splat(
                        data.ndim, data.colors is not None, encoding=encoding
                    ),
                )

            # ── scale-derived defaults (logged) ──
            eff_max_elements: Optional[int] = max_elements
            if recipe in ("tiles", "overview", "adaptive"):
                # BSP partitioning needs >= 3 spatial dims; fail cleanly (the
                # rest of the command's validation style) rather than letting
                # the deeper ValueError surface as a raw traceback.
                if data.ndim < 3:
                    raise typer.BadParameter(
                        f"recipe '{recipe}' requires >=3 spatial dimensions for "
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
                "levels",
                "overview",
                "adaptive",
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
                breakpoints=bp,
                truncation_sigmas=(
                    truncation_sigmas if truncation_sigmas is not None else 3.0
                ),
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
                        compress=compress,  # type: ignore[arg-type]
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
                        fitting_info=fitting_info,
                        fitting_config=fitting_config,
                        provenance_info=provenance_info,
                        pipeline_info=pipeline_info,
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
