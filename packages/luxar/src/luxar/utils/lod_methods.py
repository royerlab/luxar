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


#: Old → new spellings of the LOD *method* flags (renamed 2026-08-11), so the
#: additive and substitutive knobs are named symmetrically. Modelled on
#: :data:`luxar.gsplats.lod.recipes.LEGACY_RECIPE_NAMES` for the CLI half — an old
#: spelling is rejected with a pointer naming its replacement.
#:
#: MANIFEST TRANSLATION IS REQUIRED, and the reason is easy to get backwards (I
#: did): batch plans persist these knobs to disk under a DASH-LESS key
#: (``planning.py`` writes ``args["subst-method"]``), which *looks* like a
#: vocabulary independent of the flags — but ``slurm_gen.py`` turns each stored key
#: straight into a flag with ``f" --{flag} ..."``. So a manifest written before this
#: rename emits ``--substitutive-method`` into its merge sbatch command and the
#: merge job dies on an unknown option. :func:`canonical_method_token` is applied
#: at EMIT time, exactly as ``canonical_recipe_name`` already is four lines above
#: it in that file for the identical reason.
#:
#: WHY the rename. ``gsplat lod`` had ``-m/--method`` for the ADDITIVE ordering but
#: ``--substitutive-method`` for the reduction — one qualified, one not — so the
#: bare name silently meant "additive" on every gsplat surface while ``mesh lod``
#: used the same bare ``--method`` for its DECIMATION algorithm. Two commands, one
#: flag name, two different mechanisms. Both are now qualified and neither is bare.
#:
#: Spelling notes, since shorter candidates were considered and rejected:
#: ``--sub-method`` would read as *sub-LOD* (used throughout); ``--stream-method`` /
#: ``--levels-method`` name one recipe each when the additive ordering applies to
#: every recipe that ladders (all of them) and ``--levels`` already means a count;
#: ``--order``/``--ordering`` is taken (``--ordering`` is the spatial index,
#: ``hilbert|morton|none``).
#: Keyed on the DASH-LESS token, because that is the form batch manifests persist
#: and the form ``slurm_gen`` prefixes ``--`` onto. :data:`LEGACY_METHOD_FLAGS` is
#: derived from it so the CLI and the manifest path cannot disagree about what an
#: old spelling maps to.
#:
#: WHERE a hand-written pointer is warranted, and where typer's own is enough.
#: Typer already answers an unknown option with "No such option:
#: --substitutive-method (Possible options: --subst-method)", which is sufficient
#: whenever the new spelling is a pure abbreviation of the old one. A hidden option
#: raising a hand-written pointer is added only where the OLD spelling would, or
#: will, name something with a different meaning — ``gsplat lod``, where bare
#: ``--method`` meant "additive" and a sibling substitutive flag now exists to be
#: confused with it, and ``mesh lod``, where ``-m`` returns later bound to the
#: additive ordering. The other surfaces (``fit``, ``additive``, the three
#: ``batch-fit`` commands) keep their ``-m`` short form and lean on typer.
#:
#: ``luxar mesh lod`` is DELIBERATELY not served by this table and raises its own
#: pointer. ``method`` maps to ``add-method`` here because on every *gsplat*
#: surface the bare flag meant the additive ordering — but mesh's bare ``--method``
#: was its *substitutive* decimation knob, so the same token has two replacements
#: chosen by the command. A single dict cannot express that, and routing mesh
#: through it would send users to a flag mesh does not yet have. Do not "unify"
#: them.
LEGACY_METHOD_TOKENS: dict[str, str] = {
    "method": "add-method",
    "additive-method": "add-method",
    "substitutive-method": "subst-method",
    "merge-additive-method": "merge-add-method",
    "merge-substitutive-method": "merge-subst-method",
}

#: Flag form of :data:`LEGACY_METHOD_TOKENS` — old → new ``--spelling``.
LEGACY_METHOD_FLAGS: dict[str, str] = {
    f"--{old}": f"--{new}" for old, new in LEGACY_METHOD_TOKENS.items()
}


def canonical_method_token(token: str) -> str:
    """Translate a legacy dash-less method token to its current spelling.

    Identity for current tokens and for unrelated knobs, so a caller can map a
    whole stored ``merge_recipe_args`` dict through it. Applied at sbatch EMIT
    time (:mod:`luxar.gsplats.batch.slurm_gen`), mirroring
    :func:`luxar.gsplats.lod.recipes.canonical_recipe_name`: a manifest from
    before the rename would otherwise emit ``--substitutive-method`` and the merge
    job would die on an unknown option.
    """
    return LEGACY_METHOD_TOKENS.get(token, token)
