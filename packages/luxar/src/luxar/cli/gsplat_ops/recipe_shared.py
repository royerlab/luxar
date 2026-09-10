"""Shared recipe/streaming validation for the gsplat CLI surfaces.

Consumed by the fit / batch / transform command modules AND the ``lod``
command itself: valid-method registries, recipe option-relevance rejection,
breakpoint parsing, bytes-per-splat estimation/measurement, store-encoding
detection, and the ``--target-ms`` streaming-knob validation/resolution.

Moved from ``cli/lod.py`` (which keeps the ``lod`` command and re-imports
these names) — six ``gsplat_ops`` modules previously had to import a
sub-command module to reach this shared validation surface.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import AbstractSet, Any, Mapping, Optional, Sequence

import typer
from arbol import aprint

from luxar.utils.lod_breakpoints import (
    estimate_bytes_per_splat as estimate_bytes_per_splat,
)
from luxar.utils.lod_methods import GSPLAT_ADDITIVE_CHOICES

#: Valid ordering methods for the additive (prefix-sum) axis.
#:
#: DERIVED from the shared registry, not hand-copied. This used to be a literal
#: tuple, and it had already rotted: ``radial`` was invisible to every gsplat CLI
#: surface until this line changed, and three ``--method`` help strings still
#: advertised only ``auto|greedy|self_energy``. Adding a method to
#: :mod:`luxar.utils.lod_methods` now reaches the CLI with no second edit.
#:
#: The registry deliberately lives under ``utils`` rather than in
#: :mod:`luxar.gsplats.lod.additive`, which owns the implementation: importing
#: anything under ``luxar.gsplats`` executes its ``__init__``, which adds ~600 ms
#: on top of the CLI's own ~250 ms import — a 3.4x multiplier on ``luxar --help``.
VALID_ADDITIVE_METHODS: tuple[str, ...] = GSPLAT_ADDITIVE_CHOICES

# Valid substitutive partition algorithms.
VALID_SUBSTITUTIVE_METHODS = (
    "auto",
    "kmeans",
    "kmeans_lloyd",
    "greedy",
    "greedy_lloyd",
)


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


def _validated_passthrough_breakpoints(s: str) -> str:
    """Validate a size-adaptive string spec and hand it back verbatim.

    ``stream:<c>`` and ``equi-energy:<n>`` are resolved per-N inside
    ``_resolve_breakpoints`` (each part/level sizes its own ladder), and the
    string form round-trips the batch manifest verbatim — so only the payload
    is checked here, with the shared parsers' own messages.
    """
    from luxar.utils.lod_breakpoints import parse_equi_energy_rungs, parse_stream_chunk

    try:
        if s.startswith("stream:"):
            parse_stream_chunk(s)
        else:
            parse_equi_energy_rungs(s)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    return s


def parse_lod_breakpoints(spec: str) -> "str | list[int] | list[float]":
    """Parse the ``--breakpoints`` string for :func:`make_additive_lod`.

    Accepted forms: ``equal-count`` → literal; ``stream:14000`` → passed
    through as a string (a bandwidth-derived geometric ladder, resolved
    per-N inside the builder — the first chunk is ``<c>`` splats, then
    doubling); ``equi-energy:4`` → passed through as a string (``n`` rungs at
    equal shares of cumulative self-energy along the ordering, commit-capped —
    few heavy splats first, fatter rungs later); ``counts:5,10,15`` →
    ``[int]`` (cumulative splat counts); ``energy:0.5,0.9,1.0`` → ``[float]``
    (cumulative energy fractions in (0, 1]).
    """
    s = spec.strip()
    if s == "equal-count":
        return "equal-count"
    if s.startswith(("stream:", "equi-energy:")):
        return _validated_passthrough_breakpoints(s)
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


def carried_appearance(input_path: Path) -> dict:
    """The source root's authored appearance, announced as it is picked up.

    A rebuilding command owns the STRUCTURE, not the look: the builders make
    fresh nodes that know nothing about the input, so without this the writer's
    own defaults take over and every authored value is lost (#1600). Pass the
    result to ``write_gsplats_tree(root_attrs=...)`` /
    ``GSplatData.save(root_attrs=...)``, which seeds the output root at LOWEST
    precedence so the command's own structural attrs still win.

    The announcement lives here rather than at each call site so a command reads
    as one statement (and so ``lod_recipe`` stays under the complexity ratchet).
    Quiet when the input authored nothing.
    """
    from luxar.gsplats.io.load_gsplats import read_authored_appearance

    return _announce_carried(read_authored_appearance(input_path))


def carried_appearance_from_inputs(
    input_paths: "Sequence[Path]",
    *,
    exclude: "Mapping[str, str] | AbstractSet[str]" = frozenset(),
    input_has_colors: "Sequence[bool] | None" = None,
    output_has_colors: "bool | None" = None,
) -> dict:
    """The appearance N inputs AGREE on, announced as it is picked up.

    The multi-input form of :func:`carried_appearance`, for ``gsplat merge``:
    same lowest-precedence ``root_attrs`` channel, but the value has to be
    agreed rather than simply read — see
    :func:`~luxar.gsplats.io.load_gsplats.agreed_authored_appearance` for the
    unanimity rule, what does and does not count as an OPINION (an absent key,
    and a value equal to the one the writer manufactures, are both silence —
    with the single ``visible`` exception), and ``exclude`` (keys the merge MODE
    itself invalidates). The helper does the warning; this adds the same
    one-line "carrying" announcement its single-input sibling prints, so the two
    commands read alike.

    ``exclude`` is a SET, not a ``Collection``, so a bare ``str`` cannot be
    passed by accident — ``set("colormap")`` is a set of characters and would
    exclude nothing.

    ``input_has_colors`` supplies the color state already measured by merge.
    A colored input with no authored palette is not silent: it is asking the
    viewer to use its per-splat RGB, so a sibling's palette cannot be carried.
    ``output_has_colors`` supplies the merged state so a dropped colormap's
    warning can distinguish no palette from the writer's colorless ``gray``.
    """
    from luxar.gsplats.io.load_gsplats import agreed_authored_appearance

    return _announce_carried(
        agreed_authored_appearance(
            input_paths,
            exclude=exclude,
            input_has_colors=input_has_colors,
            output_has_colors=output_has_colors,
        )
    )


def _announce_carried(carried: dict) -> dict:
    """Print the one-line carry announcement (quiet when nothing is carried)."""
    if carried:
        aprint("Carrying authored appearance: " + ", ".join(sorted(carried)))
    return carried


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

    Attributes are read through :func:`luxar._zarr_compat.read_node_attrs`, so
    both on-disk formats are classified. Globbing for the format-2 ``.zattrs``
    document by name (the previous implementation) matched nothing in a format-3
    store and returned the perfectly ordinary ``None``, which callers read as
    "unclassifiable" rather than as the failure it was.
    """
    from typing import Iterator

    from luxar._zarr_compat import read_node_attrs

    if not path.is_dir():
        return None

    def _encodings(array: str) -> Iterator[dict]:
        # Glob the array DIRECTORY, not its metadata document: the document is
        # named differently per format, the directory is not.
        for node in sorted(path.rglob(array)):
            if not node.is_dir():
                continue
            attrs = read_node_attrs(node)
            enc = (attrs or {}).get("encoding", {})
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
    slice_count: int = 1,
    part_count: int = 1,
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
    from luxar.utils.lod_breakpoints import scaled_streaming_chunk

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
        whole_node_c = streaming_chunk_splats(target_ms, bw, bps)
        c = scaled_streaming_chunk(
            whole_node_c, slice_count=slice_count, part_count=part_count
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    aprint(
        f"--target-ms {target_ms:g} @ {bw:g} Mbps, {bps:.1f} B/splat "
        f"({source}) x {slice_count} slice(s) / {part_count} part(s) "
        f"-> stream:{c} (first chunk ~{c:,} splats/part, "
        f"~{c * bps / 1024:.0f} KB)"
    )
    return f"stream:{c}"


def survey_gsplat_streaming_layout(input_path: Path) -> tuple[int, int]:
    """Return ``(hidden slices, simultaneously drawn leaves)`` for a store."""
    from luxar._zarr_compat import open_group
    from luxar.encoding.decoder import decode_coordinate_columns
    from luxar.gsplats.io._archive import resolve_store_path

    resolved, temp_dir = resolve_store_path(input_path)
    try:
        root = open_group(str(resolved), mode="r")
        hidden_rows: set[tuple[object, ...]] = set()

        def survey(group: Any) -> int:
            attrs = dict(group.attrs)
            if attrs.get("type") == "gsplats":
                n_sub = int(attrs.get("n_additive_sublods", 1) or 1)
                levels = (
                    [group[f"additive_{index}"] for index in range(n_sub)]
                    if "additive_0" in group
                    else [group]
                )
                for level in levels:
                    if "centers" not in level:
                        continue
                    centers = level["centers"]
                    slice_dims = [
                        int(dim)
                        for dim in level.attrs.get("slice_dims", [])
                        if 0 <= int(dim) < centers.shape[1]
                    ]
                    if len(slice_dims) >= centers.shape[1]:
                        dims = tuple(dim for dim in slice_dims if dim >= 3)
                    else:
                        first_hidden_dim = 2 if centers.shape[1] == 3 else 3
                        dims = tuple(
                            dim for dim in slice_dims if dim >= first_hidden_dim
                        )
                    if not dims:
                        continue
                    columns = decode_coordinate_columns(centers, dims, root)
                    hidden_rows.update(tuple(row) for row in columns)
                return 1
            children = [group[name] for name in group.group_keys()]
            if not children:
                return 0
            counts = [survey(child) for child in children]
            if attrs.get("kind") == "partition":
                return sum(counts)
            return max(counts, default=0)

        part_count = survey(root)
        return max(1, len(hidden_rows)), max(1, part_count)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


#: The post-merge refinement modes, in the order they trade time for fidelity.
VALID_REFINE_MODES = ("none", "l2", "volume")


def validate_refine(
    refine: "Optional[str]",
    refine_iters: "Optional[int]",
    *,
    flag: str = "--refine",
    iters_flag: str = "--refine-iters",
    volume: "Any" = None,
    require_volume: bool = False,
) -> str:
    """Normalise a ``--refine`` value and reject an orphan iteration count.

    Shared by `gsplat fit --recipe`, `batch-fit merge` and `gsplat lod`, which
    each expose the same pair under their own flag spellings — hence ``flag`` /
    ``iters_flag``, so the message names what the user actually typed.
    """
    norm = (refine or "none").strip()
    if norm not in VALID_REFINE_MODES:
        raise typer.BadParameter(
            f"{flag} must be one of {list(VALID_REFINE_MODES)}; got {refine!r}"
        )
    if refine_iters is not None and norm == "none":
        raise typer.BadParameter(f"{iters_flag} only applies with {flag} l2|volume.")
    if require_volume and norm == "volume" and volume is None:
        raise typer.BadParameter(
            f"{flag} volume needs the source volume; none was supplied."
        )
    return norm
