"""Reveal-ladder option parsing, shared by every command that authors one.

``gsplat lod`` and ``luxar mesh lod`` both accept ``--reveal-center`` and
``--spatial-dims``, and both need them parsed and cross-validated identically —
the flags name the same concept and a divergence between the two would be a
silent difference in what a user's ladder actually orders by.

Its own module rather than :mod:`luxar.cli.common_options`, whose docstring scopes
it to the serve-family network/CORS options; widening it to ladder vocabulary
would make that statement false. The precedent is
:mod:`luxar.utils.lod_methods` — one shared home for a vocabulary two surfaces
must agree on, rather than a copy per surface.

Every function here moved VERBATIM from ``cli/lod.py``, where it was private, so
``gsplat lod``'s behaviour and error strings are unchanged by the extraction.
"""

from __future__ import annotations

import math
from typing import Optional

import typer

from luxar.utils.lod_methods import REVEAL_METHODS, is_reveal_method


def parse_reveal_center(spec: Optional[str]) -> Optional["list[float]"]:
    """Parse ``--reveal-center`` — a comma-separated shell centre, or ``None``."""
    if spec is None:
        return None
    try:
        parsed = [float(t) for t in spec.split(",") if t.strip() != ""]
    except ValueError as e:
        raise typer.BadParameter(
            f"--reveal-center must be comma-separated numbers; got {spec!r}"
        ) from e
    if not parsed:
        raise typer.BadParameter("--reveal-center must list >=1 coordinate")
    if not all(math.isfinite(c) for c in parsed):
        # `float("nan")` / `float("inf")` parse happily. Every distance would then
        # be non-finite, all comparing equal under the stable sort, so the ladder
        # would come out in input order with nothing to say the centre was junk.
        raise typer.BadParameter(
            f"--reveal-center must be finite numbers; got {spec!r}"
        )
    return parsed


def parse_reveal_spatial_dims(spec: Optional[str], ndim: int) -> Optional["list[int]"]:
    """Parse ``--spatial-dims`` — the columns the shell distance spans, or ``None``.

    Keeps the listed ORDER and rejects a repeat (see the comment below: the order
    pairs with ``--reveal-center``, deliberately unlike ``--coarsen-dims``), and
    bounds-checks each index against the dataset's own ``ndim`` so a typo is
    caught before any work.
    """
    if spec is None:
        return None
    try:
        parsed = [int(t) for t in spec.split(",") if t.strip() != ""]
    except ValueError as e:
        raise typer.BadParameter(
            f"--spatial-dims must be comma-separated integers; got {spec!r}"
        ) from e
    # Order is PRESERVED and duplicates REJECTED, deliberately unlike
    # `--coarsen-dims` (which sorts, because a barrier set is order-free). Here the
    # order is load-bearing: `--reveal-center` supplies one coordinate per LISTED
    # axis, so `sorted(set(...))` made `--spatial-dims 2,0 --reveal-center 10,20`
    # silently mean "axis 0 centred at 10" rather than the pairing the user typed.
    if not parsed:
        raise typer.BadParameter("--spatial-dims must list >=1 index")
    if len(set(parsed)) != len(parsed):
        raise typer.BadParameter(
            f"--spatial-dims must not repeat an axis (a repeat would count it "
            f"twice in the distance); got {parsed}"
        )
    for i in parsed:
        if i < 0 or i >= ndim:
            raise typer.BadParameter(
                f"--spatial-dims index {i} out of range for {ndim}D data"
            )
    return parsed


def parse_reveal_knobs(
    reveal_center: Optional[str],
    spatial_dims: Optional[str],
    method_norm: Optional[str],
    ndim: int,
) -> "tuple[Optional[list[float]], Optional[list[int]]]":
    """Parse and validate ``--reveal-center`` / ``--spatial-dims``.

    Kept out of the command bodies, which are already the most complex functions
    in their modules; inlining this validation pushed `lod_recipe` past the C901
    ratchet. Everything it needs is passed in, so it stays independently testable.

    Both knobs apply only to the ``radial`` ordering, and passing either under
    another method is an ERROR rather than a silent no-op: a user who types
    ``--reveal-center`` with the default ``auto`` wants a reveal, and would
    otherwise get an energy-ordered ladder with nothing to indicate the flag was
    dropped.
    """
    if (reveal_center is not None or spatial_dims is not None) and not is_reveal_method(
        str(method_norm)
    ):
        # Asked of the shared registry, not compared against a literal, so a
        # second reveal ordering needs no edit here.
        bad = "--reveal-center" if reveal_center is not None else "--spatial-dims"
        listed = " / ".join(sorted(REVEAL_METHODS))
        raise typer.BadParameter(
            f"{bad} only applies to a reveal ordering ({listed}); pass "
            f"-m {sorted(REVEAL_METHODS)[0]} (got -m {method_norm})."
        )

    parsed_centre = parse_reveal_center(reveal_center)
    parsed_dims = parse_reveal_spatial_dims(spatial_dims, ndim)

    # The centre carries one coordinate per axis the distance is measured over,
    # so its length must match --spatial-dims when both are given. Checked here
    # rather than deep in the scorer so the error names the flags the user typed.
    if parsed_centre is not None and parsed_dims is not None:
        if len(parsed_centre) != len(parsed_dims):
            raise typer.BadParameter(
                f"--reveal-center has {len(parsed_centre)} coordinates but "
                f"--spatial-dims lists {len(parsed_dims)} axes; they must match."
            )
    return parsed_centre, parsed_dims
