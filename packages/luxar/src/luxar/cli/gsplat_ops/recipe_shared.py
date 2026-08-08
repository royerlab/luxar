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
from pathlib import Path
from typing import Any, Mapping, Optional

import typer
from arbol import aprint

# Valid ordering methods for the additive (prefix-sum) axis.
VALID_ADDITIVE_METHODS = (
    "auto",
    "greedy",
    "self_energy",
    "mass",
    "amplitude",
    "spectral",
    "random",
)

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


def parse_lod_breakpoints(spec: str) -> "str | list[int] | list[float]":
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
