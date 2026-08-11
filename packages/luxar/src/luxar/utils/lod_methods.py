"""Registry of additive-LOD ordering methods — the single source of truth.

This exists so the implementation (:mod:`luxar.gsplats.lod.additive`) and the
CLI (:mod:`luxar.cli.gsplat_ops.recipe_shared`) can share one list instead of
hand-copying it. They used to hold two literal tuples with no consistency test,
and the copies had already rotted: ``radial`` was invisible to every gsplat CLI
surface, and three ``--method`` help strings still advertised only
``auto|greedy|self_energy`` long after six methods existed.

**Why it lives under ``utils`` rather than beside the implementation.** Importing
anything under ``luxar.gsplats`` executes ``luxar/gsplats/__init__.py``, which
adds ~600 ms on top of the CLI's own ~250 ms import (a 3.4x multiplier on
``luxar --help``, stable over 3 runs) — for the sake of one tuple. ``luxar.utils`` is already on the CLI's import path, so a leaf module
here is free. The same reasoning already put the shared breakpoint vocabulary in
``luxar.utils.lod_breakpoints``.

Keep this module dependency-free (stdlib ``typing`` only) — that property is what
makes it importable from either side.
"""

from __future__ import annotations

from typing import Literal

#: Ordering methods the additive (prefix-sum) axis implements, in the order they
#: are presented to users. ``auto`` is deliberately absent — it is a resolution
#: sentinel, not an implementation, and only some surfaces accept it.
GSPLAT_ADDITIVE_METHODS: tuple[str, ...] = (
    "greedy",
    "self_energy",
    "mass",
    "amplitude",
    "spectral",
    "random",
    "radial",
)

#: Accepted at the API/CLI boundary: the implementations plus the size-adaptive
#: ``auto`` sentinel resolved by
#: :func:`luxar.gsplats.lod.additive.resolve_additive_method`.
GSPLAT_ADDITIVE_CHOICES: tuple[str, ...] = ("auto", *GSPLAT_ADDITIVE_METHODS)

#: Rendered into ``--method`` help strings so they cannot fall out of date the
#: way the five hand-written ones did.
GSPLAT_ADDITIVE_CHOICES_HELP: str = "|".join(GSPLAT_ADDITIVE_CHOICES)

#: A resolved method — no ``auto``.
MethodName = Literal[
    "greedy", "self_energy", "mass", "amplitude", "spectral", "random", "radial"
]

#: A method as accepted from a user, ``auto`` included.
AutoOrMethod = Literal[
    "auto", "greedy", "self_energy", "mass", "amplitude", "spectral", "random", "radial"
]

#: Methods that order a REVEAL rather than ranking by contribution.
#:
#: A reveal's prefix is a *partial object at full brightness*, not a dim version
#: of the whole, so its ladder must carry no energy stamps: the viewer multiplies
#: brightness by ``1/e(k)`` while a ladder is incomplete, gated on the blending
#: mode and never on geometry type, which would blow out the innermost shell and
#: then dim it as the object completes — the inverse of growing in. Enforced at
#: authoring time in :mod:`luxar.gsplats.lod.additive` and in
#: :func:`luxar.core.group.lod.group.additive_level_stats`.
#:
#: MEASURED SCOPE (2026-08-10): the compensation is applied by ``applyLodFade``,
#: whose only caller is the viewer's ``kind=lod`` group registry. So the stamps
#: are read for a ladder that sits *inside* a lod group (``levels`` / ``adaptive``
#: / ``overview``, all of which carry stream ladders by default) and are inert on
#: a bare ``stream``/``flat`` leaf — a stamped and an unstamped bare leaf render
#: byte-identically.
#:
#: That does NOT make the rule conditional, because the method is the only thing
#: an authoring call has to key on and it already distinguishes both cases:
#: ``lod --recipe levels|adaptive|overview -m radial`` writes reveal ladders
#: *directly inside* a lod group (where the stamps bite), and
#: ``--recipe stream -m radial`` writes a bare one (where they are inert). One
#: predicate covers both. Nor is there a route that turns a suppressed ladder
#: into a stamped one behind your back: every ladder-producing path rebuilds
#: through :func:`~luxar.gsplats.lod.additive.make_additive_lod` (or
#: :func:`~luxar.core.group.lod.group.additive_level_stats`) and so re-consults
#: this set — verified for ``gsplat additive`` and ``lod --recipe levels``, both
#: of which DISCARD an input ladder and re-derive from the method they are given.
REVEAL_METHODS: frozenset[str] = frozenset({"radial"})


def is_reveal_method(method: str) -> bool:
    """Whether ``method`` orders a reveal, so its ladder carries no energy stamps."""
    return method in REVEAL_METHODS
