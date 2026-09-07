"""``luxar mesh lod`` — a mesh scene → a substitutive LOD ladder.

The mesh peer of ``luxar gsplat lod``, and shaped differently for one structural
reason: ``gsplat lod`` reads and writes ``.gsplats.zarr``, a standalone store, so
it is a store-to-store transform. **There is no standalone mesh store** — the only
sink for a mesh is ``Scene.add_mesh`` — so this reads a ``.luxar.zarr`` SCENE,
takes one mesh node out of it, and writes a new scene whose node is a ``kind=lod``
group. Same asymmetry ``luxar mesh import`` already carries, for the same reason.

The ladder itself is built by ``add_mesh(substitutive_lod=…)``; this command is an
option surface plus a read/write shell around it, so the CLI and the Python API
cannot disagree about what a level is.

The output scene contains ONLY the picked mesh's ladder: every other node in the
source scene — other points/lines/gsplats/mesh nodes, other groups, user-authored
overlays — is not carried across; nor is any placement/compositing an ancestor
group genuinely changes (its own attrs are forwarded, but a group's are not — see
``_lost_compositing_keys``, which warns only on a key that is not sitting at its
neutral default, not the identity transform, and not already overridden by the
picked mesh's own attrs or by a nearer group — so neither a bare namespace
group nor an existing ladder's own bookkeeping wrapper, nor this command's own
re-stamped defaults, trigger a false alarm); nor are the picked mesh's own
per-vertex ``labels`` / ``image_labels`` / ``keys`` (``MeshData`` has no field for
them, so the reader never surfaces them), nor its texture / UV coordinates (mesh
LOD does not preserve node-level images). All of this is reported
with an explicit warning naming what is dropped, after every validator that can
still abort the run and before anything is written.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import typer
from arbol import aprint, asection

from ...core.group.lod.group import MESH_SUBSTITUTIVE_METHODS
from ...core.group.lod.mesh import (
    DEFAULT_MESH_ADDITIVE_METHOD,
    MESH_ADDITIVE_METHODS,
)


def _ancestor_group_paths(node_path: str) -> List[str]:
    """Full paths of every group ``node_path`` is nested under, root-first.

    Empty for a top-level node. A group's ``COMPOSITING_ATTRS`` (``transform``,
    ``opacity``, ``blending_mode``, …) are inherited by every descendant,
    exactly the way the ``kind=lod`` wrapper this command itself writes passes
    them down to its children (see the note below on ``forwarded``). `run_lod`
    only reads the picked mesh LEAF's own ``data.metadata`` — an ancestor's
    attrs never reach it. Whether that is a REAL loss is a much narrower
    question than "did the ancestor set the key" — see `_lost_compositing_keys`,
    which does the actual filtering.
    """
    ancestors = []
    prefix = ""
    for part in node_path.split("/")[:-1]:
        prefix = f"{prefix}/{part}" if prefix else part
        ancestors.append(prefix)
    return ancestors


_NEUTRAL_COMPOSITING_DEFAULTS = {
    # Measured on a bare `add_mesh("surf", V, F)`: the writer stamps these
    # FIVE keys onto every leaf regardless of whether the caller set them, at
    # values that compose as a no-op (multiplicatively for the first four,
    # additively for `offset`). `run_lod` forwards them (they are in
    # `KNOWN_RENDER_ATTRS`) and the adder splits `COMPOSITING_ATTRS` onto the
    # `kind=lod` wrapper it builds — so every ladder THIS COMMAND writes has a
    # wrapper carrying all five at these exact values, and a key sitting at
    # its neutral default changes nothing regardless of which layer sets it.
    "opacity": 1.0,
    "gamma": 1.0,
    "intensity": 1.0,
    "absorption": 1.0,
    "offset": 0.0,
    # Not auto-stamped, but these two have a DEFAULT the reader applies when
    # the key is absent (`Node.layer` -> False, `Node.visible` -> True), so an
    # ancestor authoring the default explicitly changes nothing either: the
    # group is not a Layers-panel entry, and it is not hidden. Only the other
    # value is a real loss (a dropped layer grouping / a surface that was
    # authored hidden and comes back shown).
    "layer": False,
    "visible": True,
}
# `blending_mode` is NOT auto-stamped (a bare leaf has none), so presence
# alone means the ancestor authored one — but the viewer's attrs composer is
# nearest-setter-wins, so it only matters when neither the picked LEAF nor a
# NEARER ancestor group also sets the same key (the nearest setter's value
# wins regardless of an outer group's). The leaf half of that rule lives in
# `_is_lost_compositing_key`; the ancestor half in `_ancestor_compositing_loss`.
#
# `join` is COMPOSITING_ATTRS too, but is handled separately in
# `_lost_compositing_keys` rather than through this leaf-overrides rule:
# `add_mesh` REFUSES `join` outright (`reject_lines_only_join` in
# `core/group/adders/mesh.py` — it is lines-only), so a mesh LEAF can never
# carry it, and the "skip when the leaf also sets it" test could never fire.
# An ancestor's `join` cannot affect a mesh ladder either way, so it is
# always skipped, not just when the leaf happens to override it.
#
# `layer_order` joins `blending_mode` for exactly the same reasons: not
# auto-stamped (its absence on disk is genuine silence, deliberately, so that an
# explicit level stays distinguishable from a default), and nearest-setter-wins
# in the composer. Classified explicitly rather than left to the
# report-by-default fall-through at the end of `_is_lost_compositing_key`, which
# would over-report an ancestor level the picked leaf already overrides.
_LEAF_OVERRIDES_ANCESTOR = frozenset({"blending_mode", "layer_order"})


def _is_identity_transform(raw_transform: Any) -> bool:
    """Whether a group's RAW stored ``transform`` attr is the identity.

    ``source.get_node_metadata`` — unlike ``get_mesh`` — hands back the
    column-major 16-list exactly as stored (see its own docstring); reshaping
    that list directly, row-major, would silently swap the translation into
    the wrong slots (the CLAUDE.md transpose gotcha). `read_transform_from_zarr`
    is the same conversion `get_mesh` itself applies, so this compares like
    with like against :func:`luxar.core.transforms.identity`.
    """
    from ...core.transforms import identity, read_transform_from_zarr

    matrix = read_transform_from_zarr(raw_transform)
    return bool((matrix == identity()).all())


def _is_identity_nd_transform(nd_transform: Optional[Dict[str, Any]]) -> bool:
    """Whether a group's ``nd_transform`` dict changes nothing.

    Per-dimension: a ``{"scale": 1.0, "offset": 0.0}`` (or absent) affine entry,
    or a ``permutation`` that maps every index to itself, is a no-op.
    """
    if not nd_transform:
        return True
    for entry in nd_transform.values():
        permutation = entry.get("permutation")
        if permutation is not None:
            if list(permutation) != list(range(len(permutation))):
                return False
        elif entry.get("scale", 1.0) != 1.0 or entry.get("offset", 0.0) != 0.0:
            return False
    return True


def _is_lost_compositing_key(
    key: str, value: Any, leaf_metadata: Dict[str, Any]
) -> bool:
    """Whether ONE ancestor ``COMPOSITING_ATTRS`` key is a REAL loss.

    Key-presence alone over-reports: see the module-level notes above each
    exception this makes. Split out of `_lost_compositing_keys` itself so
    that function's own branching (now just a comprehension) stays well
    under the C901 ratchet.
    """
    if key in _NEUTRAL_COMPOSITING_DEFAULTS:
        return bool(value != _NEUTRAL_COMPOSITING_DEFAULTS[key])
    if key == "transform":
        return not _is_identity_transform(value)
    if key == "nd_transform":
        return not _is_identity_nd_transform(value)
    if key == "join":
        # A mesh leaf can never carry `join` (see the module-level note on
        # `_LEAF_OVERRIDES_ANCESTOR`), so an ancestor's `join` is never
        # something a mesh ladder loses — this path is a mesh command, not
        # a lines one.
        return False
    if key in _LEAF_OVERRIDES_ANCESTOR:
        # The leaf half of nearest-setter-wins; the nearer-GROUP half is
        # `_ancestor_compositing_loss`'s, which sees the whole chain.
        return key not in leaf_metadata
    # Every member of `COMPOSITING_ATTRS` is covered above; a future one
    # defaults to being reported rather than silently skipped.
    return True


def _lost_compositing_keys(
    ancestor_attrs: Dict[str, Any], leaf_metadata: Dict[str, Any]
) -> List[str]:
    """Which of an ancestor's ``COMPOSITING_ATTRS`` are a REAL loss.

    Iterates the intersection in sorted order so the result (and the message
    built from it) is deterministic.
    """
    from ...core.group.compositing import COMPOSITING_ATTRS

    return [
        key
        for key in sorted(COMPOSITING_ATTRS & ancestor_attrs.keys())
        if _is_lost_compositing_key(key, ancestor_attrs[key], leaf_metadata)
    ]


def _ancestor_compositing_loss(
    source: Any, node_path: str, leaf_metadata: Dict[str, Any]
) -> List[Tuple[str, List[str]]]:
    """``(group_path, lost_keys)`` for each ancestor that actually loses something.

    A bare namespace group, and the ``kind=lod`` / ``kind=partition`` wrapper
    groups this command's own supported ``--node surf/child_0`` /
    ``--node surf/part_0`` re-laddering workflows nest a mesh under, carry
    ONLY structural stamps (``kind``, ``child_index``, ``content_hash``,
    ``selector``, ``default_level``, ``max_elements``, ``position_bounds``,
    ``display_type``, …) — none of them a member of ``COMPOSITING_ATTRS``. But
    key PRESENCE is not enough either: see `_lost_compositing_keys`, which
    this delegates to for the real filter.

    Walked NEAREST-ANCESTOR-FIRST for the sake of the
    `_LEAF_OVERRIDES_ANCESTOR` keys, which the viewer composes
    nearest-setter-wins: under
    ``outer(blending_mode='additive') / inner(blending_mode='normal') / mesh``
    only ``inner``'s mode ever reached the mesh, so ``outer``'s was ALREADY
    shadowed in the source scene and is not something this rewrite loses.
    Every other compositing key composes across the whole chain (the scalars
    multiply, the transforms concatenate), so each ancestor that sets one
    genuinely loses it and is reported. The list is flipped back to
    root-first before returning, so the warnings still read outermost-first.
    """
    losses = []
    shadowed: Set[str] = set()
    for group_path in reversed(_ancestor_group_paths(node_path)):
        attrs = source.get_node_metadata(group_path)
        lost_keys = [
            key
            for key in _lost_compositing_keys(attrs, leaf_metadata)
            if key not in shadowed
        ]
        shadowed |= _LEAF_OVERRIDES_ANCESTOR & attrs.keys()
        if lost_keys:
            losses.append((group_path, lost_keys))
    losses.reverse()
    return losses


def _dropped_sibling_nodes(source: Any, node_path: str) -> List[Tuple[str, str]]:
    """Every OTHER node in ``source`` the rewrite will not carry.

    The output is a brand-new scene containing only the ladder built from
    ``node_path`` — every other points/lines/gsplats/mesh node, and every other
    group, is left out (the caller is responsible for warning about it; this
    just enumerates it). Ancestor groups of ``node_path`` are excluded HERE —
    see `_ancestor_group_paths`, which the caller uses instead to warn about
    them with a more specific message; they are not omitted because they are
    harmless. The reserved ``overlays`` container is also excluded: it is not
    a user node (`Scene` refuses to let anyone create a top-level node by that
    name), and the overlay entries actually living under it don't match any
    ``list_*`` filter in the first place — `_dropped_overlay_nodes` reports
    those, the real content, separately.
    """
    ancestors = set(_ancestor_group_paths(node_path))
    dropped = [(n, "points") for n in sorted(source.list_points())]
    dropped += [(n, "lines") for n in sorted(source.list_lines())]
    dropped += [(n, "gsplats") for n in sorted(source.list_gsplats())]
    dropped += [(n, "mesh") for n in sorted(source.list_meshes()) if n != node_path]
    dropped += [
        (n, "group")
        for n in sorted(source.list_groups())
        if n not in ancestors and n != "overlays"
    ]
    return dropped


_OVERLAY_TYPES = ("overlay_text", "overlay_html", "overlay_image", "overlay_video")
"""The four user-authored overlay types: `Scene.add_text` / `.add_html` /
`.add_image` / `.add_video` (`core/scene/overlays/adders.py`). Anything else under
`overlays/` is not a real overlay this command needs to report."""


_AUTO_HOVER_TEMPLATES = {
    "__hover_text": ("text", "{hover_label}"),
    "__hover_image": ("html", "{hover_image_label}"),
}
"""The two overlays ``auto_inject_hover_overlay`` writes, each mapped to the
attr carrying its content and the exact placeholder it fills that attr with
(``core/scene/overlays/hover_inject.py``). That payload is the part of an
injected overlay a user does not arrive at by accident — the NAME is
unreserved (`next_overlay_name` lets anyone claim it) and ``hover=True`` is a
public ``Scene.add_text`` kwarg."""


def _is_auto_injected_hover_overlay(
    source: Any, overlay_path: str, mesh_is_labelled: bool
) -> bool:
    """Whether ``overlay_path`` is the hover overlay finalize auto-injects.

    FOUR conditions, all required:

    1. Gated on the picked mesh actually HAVING ``labels``/``image_labels``:
       that is the only reason finalize ever injects a hover overlay
       (``auto_inject_hover_overlay`` returns immediately otherwise), and it
       is the per-vertex-label warning's presence — not this overlay's own
       name or attrs — that justifies skipping it here. Without this gate, an
       UNLABELLED scene with a user overlay named ``__hover_text`` AND
       ``hover=True`` (a public kwarg of ``Scene.add_text``) satisfies the
       checks below yet is genuine content nothing else warns about.
    2. Name matches ``__hover_text`` / ``__hover_image``.
    3. Its own ``hover`` attr is actually ``True`` (`next_overlay_name` does
       not reserve either name, so a same-named, non-hover user overlay must
       still be reported).
    4. Its content is the injected PLACEHOLDER (`_AUTO_HOVER_TEMPLATES`).
       Conditions 1-3 are not provenance on a LABELLED scene: finalize skips
       the injection entirely when any overlay already sets ``hover``
       (`auto_inject_hover_overlay` condition 2), so a user's own
       ``add_text('…', name='__hover_text', hover=True)`` is the ONLY hover
       overlay in the store and would be dropped with nothing said. An
       overlay whose text is verbatim ``{hover_label}`` either IS the
       injected one or is indistinguishable from it in what it renders.
    """
    if not mesh_is_labelled:
        return False
    template = _AUTO_HOVER_TEMPLATES.get(overlay_path.rsplit("/", 1)[-1])
    if template is None:
        return False
    attrs = source.get_node_metadata(overlay_path)
    if not attrs.get("hover"):
        return False
    content_attr, placeholder = template
    return bool(attrs.get(content_attr) == placeholder)


def _dropped_overlay_nodes(source: Any, mesh_is_labelled: bool) -> List[str]:
    """Full paths of the genuine user-authored overlays under ``overlays/``.

    Overlay nodes match none of the five ``list_*`` filters, so
    `_dropped_sibling_nodes` never sees them — only ``LuxarScene.nodes`` types
    them. Excludes the auto-injected hover overlay (see
    `_is_auto_injected_hover_overlay`): it exists purely because the picked
    node has ``labels``/``image_labels``, and the per-vertex-label warning
    already covers that loss — naming the hover overlay too would just be the
    same drop said twice.
    """
    return sorted(
        n["name"]
        for n in source.nodes
        if n["name"].startswith("overlays/")
        and n["type"] in _OVERLAY_TYPES
        and not _is_auto_injected_hover_overlay(source, n["name"], mesh_is_labelled)
    )


def _format_dropped_node_lines(
    dropped: List[Tuple[str, str]], leaf_name: str
) -> List[str]:
    """One warning line per dropped node — collapsed where that would flood.

    A large spatial partition (`--node surf/part_0` against a scene with
    hundreds of `surf/part_N` siblings — a workflow `_pick_mesh` itself
    forces once there is more than one mesh) would otherwise print one
    `aprint` per sibling. Grouped by (parent path, kind); a group with MORE
    THAN THREE members collapses to one line stating the full count (never a
    silent truncation) — e.g. "4 other points nodes in 'stuff'" — and a
    smaller group keeps its individual named lines, as before.

    Deliberately NOT worded "parts": that is Luxar's term of art for
    `kind=partition` children specifically, and this groups by parent PATH
    alone — any sibling nodes sharing an ordinary namespace group are not
    "parts" of anything.
    """
    groups: Dict[Tuple[str, str], List[str]] = {}
    for name, kind in dropped:
        parent = name.rsplit("/", 1)[0] if "/" in name else ""
        groups.setdefault((parent, kind), []).append(name)

    lines = []
    for (parent, kind), names in sorted(groups.items()):
        if len(names) > 3:
            where = f" in {parent!r}" if parent else ""
            lines.append(
                f"⚠️  Not carried into the new scene: {len(names)} other "
                f"{kind} nodes{where} — only {leaf_name}'s ladder is written."
            )
        else:
            for name in names:
                lines.append(
                    f"⚠️  Not carried into the new scene: {name!r} ({kind}) — "
                    f"only {leaf_name}'s ladder is written."
                )
    return lines


def _report_drops(source: Any, node_path: str, leaf_name: str, data: Any) -> None:
    """Print every warning for what this rewrite will not carry across.

    Pulled out of `run_lod` so that function's own branching stays under the
    C901 ratchet — this is pure reporting (four independent drop categories,
    each already reduced to a plain list/tuple by its own helper), not
    control flow the caller needs inline.
    """
    dropped_siblings = _dropped_sibling_nodes(source, node_path)
    for line in _format_dropped_node_lines(dropped_siblings, leaf_name):
        aprint(line)
    for group_path, lost_keys in _ancestor_compositing_loss(
        source, node_path, data.metadata
    ):
        aprint(
            f"⚠️  {node_path!r} is nested under group {group_path!r}, which "
            f"sets {', '.join(lost_keys)}; that inherited placement/"
            "compositing is NOT forwarded and does not travel into the "
            "new scene."
        )
    dropped_label_channels = [
        channel
        for channel, present in (
            ("labels", data.metadata.get("has_labels")),
            ("image_labels", data.metadata.get("has_image_labels")),
        )
        if present
    ]
    # `keys` (#1917) is dropped for exactly the same reason — `MeshData` has no
    # field for it, so the reader never surfaces it — and must be reported, or a
    # keyed mesh loses every link target here in silence.
    #
    # Kept OUT of `dropped_label_channels` rather than appended to it, because
    # that list feeds `mesh_is_labelled` below and keys are not a hover-overlay
    # trigger: finalize auto-injects an overlay from labels alone, so a
    # keys-only mesh has no overlay to lose and must not take that branch.
    dropped_channels = dropped_label_channels + (
        ["keys"] if data.metadata.get("has_keys") else []
    )
    # Threaded into the overlay lookup because the hover-overlay skip is
    # justified ONLY by this: finalize auto-injects a hover overlay purely
    # because the picked node has labels, so unlabelled it never exists and
    # the "the label warning already covers it" reasoning does not apply.
    mesh_is_labelled = bool(dropped_label_channels)
    for overlay_path in _dropped_overlay_nodes(source, mesh_is_labelled):
        aprint(
            f"⚠️  Not carried into the new scene: {overlay_path!r} (overlay) "
            f"— only {leaf_name}'s ladder is written."
        )
    if dropped_channels:
        aprint(
            f"⚠️  {node_path!r} has per-vertex "
            f"{' and '.join(dropped_channels)}; `add_mesh` has no field "
            "for them, so they will NOT be carried into the new scene."
        )
    if data.metadata.get("has_texture"):
        aprint(
            f"⚠️  {node_path!r} has a texture and per-vertex UV coordinates; "
            "mesh LOD does not preserve node-level images, so they will NOT be "
            "carried into the new scene."
        )


def _pick_mesh(scene: Any, input_path: Path, node_name: Optional[str]) -> str:
    """Which mesh node to coarsen.

    Naming it is optional when the scene has exactly one mesh — which is what
    ``luxar mesh import`` produces, and therefore the common input. With several,
    the ambiguity is refused rather than guessed at: picking "the first" would
    silently coarsen a different surface than the user meant, on a scene whose
    node order they never see.
    """
    meshes = scene.list_meshes()
    if not meshes:
        raise ValueError(
            f"{input_path} contains no mesh node. `luxar mesh lod` coarsens a "
            "surface; use `luxar gsplat lod` for splats, or `luxar mesh import` "
            "to bring a mesh file in first."
        )
    if node_name is not None:
        if node_name not in meshes:
            raise ValueError(
                f"No mesh node named {node_name!r} in {input_path}. Available: "
                f"{', '.join(sorted(meshes))}"
            )
        return node_name
    if len(meshes) > 1:
        raise ValueError(
            f"{input_path} has {len(meshes)} mesh nodes and none was named. Pass "
            f"--node to choose one: {', '.join(sorted(meshes))}"
        )
    return str(meshes[0])


#: The two ladder flavours `luxar mesh lod` can write. Named rather than spelled
#: inline because every gate below compares against them, and `gsplat lod` already
#: taught that a mode implied by "which flags you happened to pass" is the thing
#: worth removing from a flag surface.
RECIPE_LEVELS = "levels"
RECIPE_REVEAL = "reveal"
MESH_LOD_RECIPES = (RECIPE_LEVELS, RECIPE_REVEAL)


def _was_supplied(ctx: Optional[typer.Context], name: str) -> bool:
    """Whether ``name`` came from the COMMAND LINE, not from its default.

    Comparing the value against the default cannot answer this, and the gap is not
    theoretical: `--recipe reveal --levels 3` types a levels-only flag whose value
    happens to BE the default, so a value comparison sees "not given" and the flag
    is silently ignored — the exact silent-drop the gate exists to prevent, hit by
    the user most likely to be surprised (someone spelling out a default).

    Click records where each parameter's value came from, which is the real signal.
    `ctx` is optional because the gate is also called without one (from `run_lod`,
    which has no parser to ask, and from the unit tests); with no context every
    knob reads as supplied, which is the safe direction — a direct caller passing a
    knob means it, having no defaults to fall back on.
    """
    if ctx is None:
        return True
    source = ctx.get_parameter_source(name)
    if source is None:
        return False
    return getattr(source, "name", str(source)) == "COMMANDLINE"


def _reject_cross_recipe_flags(
    *,
    ctx: Optional[typer.Context] = None,
    recipe: str,
    add_method: Optional[str],
    n_lods: Optional[int],
    counts: Optional[str],
    reveal_centre: Optional[str],
    spatial_dims: Optional[str],
    levels_given: bool,
    compression_given: bool,
    subst_method_given: bool,
) -> None:
    """Refuse a knob used against the recipe it does not belong to.

    An error rather than a silent no-op, and the message names BOTH flags — the
    one typed and the one that does the same job under the chosen recipe. A user
    who passes ``--n-lods`` has said what they want; dropping it and writing a
    3-level decimation instead would answer a different question without saying so.

    Checked here, before ``run_lod`` opens anything, so a rejected invocation
    cannot have deleted an existing output — the same pre-deletion discipline the
    method validation follows.

    ``run_lod`` calls this too, with the three ``*_given`` flags off: those three
    are the substitutive knobs, which it takes as PLAIN (non-optional) arguments
    and so cannot tell apart from their defaults. The five reveal-only ones it can
    — they are ``Optional`` and default to ``None`` — so a direct caller who passes
    one under ``recipe='levels'`` gets the same refusal a CLI user does, instead of
    a substitutive ladder built as if the argument had never been typed.
    """
    if recipe not in MESH_LOD_RECIPES:
        raise typer.BadParameter(
            f"--recipe must be one of {' | '.join(MESH_LOD_RECIPES)}; got {recipe!r}."
        )

    if recipe == RECIPE_LEVELS:
        reveal_only = [
            ("--add-method", add_method is not None, "-m/--add-method"),
            ("--n-lods", n_lods is not None, "--n-lods"),
            ("--counts", counts is not None, "--counts/--breakpoints"),
            ("--reveal-centre", reveal_centre is not None, "--reveal-centre"),
            ("--spatial-dims", spatial_dims is not None, "--spatial-dims"),
        ]
        substitutive_equivalent = {
            "--n-lods": "-L/--levels",
            "--add-method": "--subst-method",
        }
        # `-m cluster` is the pre-rename spelling of `--subst-method cluster`, and
        # `-m` now belongs to the additive ordering — so this gate is where that
        # migration lands. Carry the VALUE into the replacement rather than naming
        # the flag alone: a pointer the user can paste is the whole point, and the
        # long `--method` form has always carried it.
        if add_method is not None and add_method in MESH_SUBSTITUTIVE_METHODS:
            raise typer.BadParameter(
                f"-m/--add-method names the reveal ordering, but {add_method!r} is a "
                f"decimation method. -m was the old short form of --method; use "
                f"--subst-method {add_method} (with the default --recipe "
                f"{RECIPE_LEVELS}), or pass --recipe {RECIPE_REVEAL} for a reveal."
            )
        for flag, given, shown in reveal_only:
            if not given:
                continue
            equivalent = substitutive_equivalent.get(flag)
            instead = (
                f"; --recipe levels uses {equivalent} instead"
                if equivalent
                else "; --recipe levels has no equivalent"
            )
            raise typer.BadParameter(
                f"{shown} applies to --recipe {RECIPE_REVEAL}{instead}. "
                f"Pass --recipe {RECIPE_REVEAL}, or drop {shown}."
            )
        return

    # recipe == reveal: the substitutive knobs are the ones that do not apply.
    for flag, given, equivalent in (
        ("-L/--levels", levels_given, "--n-lods"),
        ("-K/--compression-factor", compression_given, "no equivalent"),
        ("--subst-method", subst_method_given, "-m/--add-method"),
    ):
        if not given:
            continue
        instead = (
            f"; --recipe {RECIPE_REVEAL} uses {equivalent} instead"
            if equivalent != "no equivalent"
            else f"; --recipe {RECIPE_REVEAL} has no equivalent (its levels are a "
            "face PARTITION, not a reduction, so there is no per-level factor)"
        )
        raise typer.BadParameter(
            f"{flag} applies to --recipe {RECIPE_LEVELS}{instead}. "
            f"Pass --recipe {RECIPE_LEVELS}, or drop {flag}."
        )
    if n_lods is not None and counts is not None:
        raise typer.BadParameter(
            "--n-lods and --counts both size the ladder; pass one. --n-lods asks "
            "for N equal-count levels, --counts gives the cumulative boundaries."
        )


def _parse_counts_spec(spec: str) -> Any:
    """``--counts`` as the resolver wants it: a list of ints, or a tagged string.

    A ``<word>:`` prefix marks the resolver's own string vocabulary and is handed
    over verbatim so its diagnostics survive. Everything else is a bare comma list
    of CUMULATIVE face counts, which the resolver takes only as a list — passing
    that form as a string reaches it as an unrecognized tag and reports the
    vocabulary rather than the mistake.
    """
    text = spec.strip()
    if ":" in text.split(",", 1)[0]:
        return text
    try:
        parsed = [int(t) for t in text.split(",") if t.strip() != ""]
    except ValueError as e:
        raise typer.BadParameter(
            f"--counts must be comma-separated integers (cumulative face counts) "
            f"or a tagged spec like 'stream:2000'; got {spec!r}"
        ) from e
    if not parsed:
        raise typer.BadParameter("--counts must list >=1 boundary")
    return parsed


def _build_reveal_spec(
    *,
    add_method: Optional[str],
    n_lods: Optional[int],
    counts: Optional[str],
    reveal_centre: Optional[str],
    spatial_dims: Optional[str],
    ndim: int,
) -> Dict[str, Any]:
    """Assemble the ``additive_lod=`` spec for ``--recipe reveal``.

    Only keys the caller actually gave are included, so the resolver's own
    defaults (``method="radial"``, four levels) stay the single source of truth for
    what an unqualified reveal means — a CLI that restated them would be a second
    place for that answer to live.

    ``counts`` is split between the two forms the resolver actually takes. Its
    STRING vocabulary is only ``stream:<c>`` / ``energy:<fractions>``; a bare
    comma list has to arrive as a list of ints, so a plain ``--counts 100,300,600``
    is parsed here and anything with a ``<word>:`` prefix is passed through
    untouched. The prefixed forms keep their own diagnostics that way — including
    the mesh-specific refusal of ``energy:`` (a mesh has no energy to take
    fractions of), which this must not pre-empt with a parse error.
    """
    from ..reveal_options import parse_reveal_knobs

    method = (add_method or DEFAULT_MESH_ADDITIVE_METHOD).replace("_", "-")
    if method not in MESH_ADDITIVE_METHODS:
        # Pre-validated for the same reason `--subst-method` is: the output
        # deletion below is irreversible, and a bad method should not cost an
        # existing store. Derived from the registry, never a literal.
        raise typer.BadParameter(
            f"--add-method must be one of {', '.join(sorted(MESH_ADDITIVE_METHODS))}; "
            f"got {method!r}. A mesh ladder is a REVEAL, so only orderings whose "
            "every prefix is one connected patch are admitted."
        )

    # The same parser `gsplat lod` uses, so the two commands cannot disagree about
    # what `--reveal-centre 1,2,3 --spatial-dims 2,0,1` means — including that the
    # dims order is load-bearing because it pairs with the centre's coordinates.
    parsed_centre, parsed_dims = parse_reveal_knobs(
        reveal_centre, spatial_dims, method, ndim
    )

    spec: Dict[str, Any] = {"method": method}
    if n_lods is not None:
        spec["n_lods"] = n_lods
    if counts is not None:
        spec["counts"] = _parse_counts_spec(counts)
    if parsed_centre is not None:
        spec["reveal_centre"] = parsed_centre
    if parsed_dims is not None:
        spec["spatial_dims"] = parsed_dims
    return spec


def _require_known_recipe(recipe: str) -> None:
    """Reject a recipe name neither ladder arm serves.

    A function rather than an inline check at each site: both callers below are
    already at the complexity ratchet's limit, and one more branch in `run_lod`
    tipped it over. The check is one line here and zero branches there.
    """
    if recipe not in MESH_LOD_RECIPES:
        raise ValueError(
            f"recipe must be one of {' | '.join(MESH_LOD_RECIPES)}; got {recipe!r}."
        )


def _build_ladder_specs(
    *,
    recipe: str,
    levels: int,
    compression_factor: int,
    method: str,
    add_method: Optional[str],
    n_lods: Optional[int],
    counts: Optional[str],
    reveal_centre: Optional[str],
    spatial_dims: Optional[str],
    ndim: int,
    n_vertices: int,
    coarsen_ndim: int,
) -> "tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]":
    """Build the ``(substitutive, additive)`` spec pair for ``recipe``.

    Exactly one is ever non-``None``: `add_mesh` refuses `additive_lod=` alongside
    `substitutive_lod=`, so `--recipe` selects rather than the two composing.

    Both arms VALIDATE before returning, and that is the load-bearing part — the
    caller deletes an existing output under `--overwrite` before it writes, so a
    knob rejected only by the adder would be diagnosed with the previous store
    already gone. `--subst-method qem` did exactly that once. These are the very
    resolvers `add_mesh` runs, so the two cannot disagree about what is accepted.

    Extracted from :func:`run_lod`, which the complexity ratchet flagged at 11 the
    moment it grew a second recipe.
    """
    from ...core.group.lod.mesh import (
        resolve_additive_axis_mesh,
        resolve_substitutive_axis_mesh,
    )
    from ...mesh.decimate import resolve_decimation_method

    if recipe == RECIPE_REVEAL:
        additive_spec = _build_reveal_spec(
            add_method=add_method,
            n_lods=n_lods,
            counts=counts,
            reveal_centre=reveal_centre,
            spatial_dims=spatial_dims,
            ndim=ndim,
        )
        # A COPY: this call only validates, and the same spec still has to reach
        # `add_mesh` whole. (The resolver copies before it pops, so the copy is
        # insurance rather than load-bearing — the substitutive arm below relies
        # on exactly that property and passes its spec straight in.)
        resolve_additive_axis_mesh(dict(additive_spec))
        return None, additive_spec

    # Reached only when the recipe is not `reveal`, and the substitutive arm is
    # entered by NAME rather than by falling through to it: an unrecognized recipe
    # here would quietly write the OTHER flavour, and `recipe="reveaal"` producing
    # a decimated ladder is a different product, not a near miss.
    _require_known_recipe(recipe)

    substitutive_spec: Dict[str, Any] = {
        "levels": levels,
        "compression_factor": compression_factor,
        "method": method,
    }
    resolve_substitutive_axis_mesh(substitutive_spec)
    resolve_decimation_method(
        method, n_vertices, spatial_ndim=coarsen_ndim, announce=False
    )
    return substitutive_spec, None


def run_lod(
    *,
    input_path: Path,
    output_path: Path,
    node_name: Optional[str],
    levels: int,
    compression_factor: int,
    method: str,
    overwrite: bool,
    recipe: str = RECIPE_LEVELS,
    add_method: Optional[str] = None,
    n_lods: Optional[int] = None,
    counts: Optional[str] = None,
    reveal_centre: Optional[str] = None,
    spatial_dims: Optional[str] = None,
) -> List[int]:
    """Write ``input_path``'s mesh as a ladder of ``recipe``. Returns the counts.

    ``levels`` selects the SUBSTITUTIVE flavour — decimated levels that REPLACE one
    another — and ``reveal`` the additive one, whose levels are disjoint face groups
    the viewer concatenates, so a prefix is a partial surface rather than a coarse
    one. ``add_mesh`` refuses the two together, which is why one ``--recipe``
    selects rather than the knobs composing.

    ``output_path`` is normalized to ``<stem>.luxar.zarr`` — the store the
    compiler actually writes — before any guard or deletion looks at it.
    """
    # `--recipe` is validated at the CLI in `_reject_cross_recipe_flags`, but
    # `run_lod` is a public entry point too, so it states its own contract rather
    # than trusting its caller. Stated HERE, at the top, so the rejection does not
    # depend on where `_build_ladder_specs` happens to sit relative to the
    # `--overwrite` deletion — that ordering is correct today (specs are built
    # ~100 lines before the `rmtree`) and this keeps it from becoming load-bearing.
    _require_known_recipe(recipe)
    # The rest of that contract: the five reveal-only arguments do not reach the
    # substitutive arm, which reads none of them, so under `recipe='levels'` a
    # direct `run_lod(recipe="levels", n_lods=6)` would build a 3-level decimation
    # and say nothing — the silent drop `--recipe` exists to prevent, reachable
    # again one layer below the flag surface. Same gate as the CLI's, so the
    # refusal and its "use -L/--levels instead" pointer are the same sentence.
    _reject_cross_recipe_flags(
        recipe=recipe,
        add_method=add_method,
        n_lods=n_lods,
        counts=counts,
        reveal_centre=reveal_centre,
        spatial_dims=spatial_dims,
        # The substitutive knobs are plain arguments here, not `Optional` ones, so
        # "did the caller pass -L?" is unanswerable — left off rather than guessed
        # at from the value, which is exactly the inference `_was_supplied` exists
        # to avoid.
        levels_given=False,
        compression_given=False,
        subst_method_given=False,
    )

    from luxar import LuxarZarrCompiler

    from ...core.group.adders.mesh import validate_scalar_data_range
    from ...core.viewer_config import ViewerConfig
    from ...io.reader import LuxarScene
    from ...utils.paths import normalize_zarr_path
    from ...validation.writing import (
        KNOWN_RENDER_ATTRS,
    )

    # NORMALIZE FIRST, and guard the normalized path only. `LuxarZarrCompiler`
    # applies exactly this normalization to whatever it is handed, so the store
    # it writes is `<stem>.luxar.zarr` — guarding the raw argument guards a path
    # nothing ever writes to. Three real data-loss cases came of that:
    # `--output scene` next to `scene.luxar.zarr` slipped the same-path check and
    # rewrote the INPUT; an existing `out.luxar.zarr` slipped the exists check and
    # was replaced without `--overwrite`; and an output normalizing onto the
    # directory that holds the input rmtree'd the source. Each reported itself
    # only afterwards, as "Scene not found" from the read-back below.
    output_path = normalize_zarr_path(output_path, ".luxar.zarr")

    # Before the deletion, for the reason `mesh import` documents: `--overwrite`
    # removes the output first, so an output that IS the input — or a directory
    # CONTAINING it, where `rmtree` takes the whole tree — would delete the
    # source and only then discover there is nothing to read. The relation is
    # checked BOTH ways, because the message promises a path outside the input's
    # tree: a destination INSIDE the source store survived the one-directional
    # check and wrote a whole nested scene into it.
    source_resolved = input_path.resolve()
    destination = output_path.resolve()
    if (
        destination == source_resolved
        or destination in source_resolved.parents
        or source_resolved in destination.parents
    ):
        raise ValueError(
            f"Output {output_path} is the input scene itself, or a directory "
            "containing it, or a path inside it. Writing there would destroy or "
            "corrupt the source; choose an output path outside the input's tree."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )

    with asection(f"Building a mesh LOD ladder from {input_path}"):
        source = LuxarScene.load(input_path)
        # A scene with no dimensions block cannot be rewritten: every node's
        # coordinates are interpreted against it, so inventing a default here
        # would silently relabel the axes of whatever we wrote out.
        if source.dimensions is None:
            raise ValueError(
                f"{input_path} declares no scene dimensions, so its mesh cannot be "
                "rewritten into a new scene without inventing axes for it."
            )
        node_path = _pick_mesh(source, input_path, node_name)
        leaf_name = node_path.rsplit("/", 1)[-1]
        data = source.get_mesh(node_path)
        aprint(
            f"Source {node_path!r}: {data.vertices.shape[0]:,} vertices, "
            f"{data.faces.shape[0]:,} faces"
        )

        substitutive_spec, additive_spec = _build_ladder_specs(
            recipe=recipe,
            levels=levels,
            compression_factor=compression_factor,
            method=method,
            add_method=add_method,
            n_lods=n_lods,
            counts=counts,
            reveal_centre=reveal_centre,
            spatial_dims=spatial_dims,
            ndim=int(data.vertices.shape[1]),
            n_vertices=int(data.vertices.shape[0]),
            coarsen_ndim=(
                len(source.dimensions.displayed)
                if int(source.dimensions.ndim) == int(data.vertices.shape[1])
                else int(data.vertices.shape[1])
            ),
        )

        # The authored ATTRS of the source node: the placement (transform /
        # nd_transform), the nD visibility broadcast, and the render attrs.
        # Forwarding only `shading`/`double_sided` dropped all of it silently —
        # most sharply the transform, which put the coarsened surface somewhere
        # else in the scene with nothing saying so.
        #
        # What still does NOT come across is the per-vertex STRING channels
        # (`labels` / `image_labels` / `keys`): `MeshData` has no field for them,
        # so the reader never surfaces them and this round trip cannot carry what
        # it cannot read. A labelled source therefore comes back unlabelled
        # rather than half-labelled, which is at least uniform across every
        # level. The warning printed below (from `data.metadata["has_labels"]` /
        # `["has_image_labels"]` / `["has_keys"]`) is the only place this is
        # reported now.
        #
        # An allow-list rather than "everything outside MESH_RESERVED_ATTRS":
        # that set is the keys the writer refuses FROM A CALLER, and the store
        # carries stamps that never pass through that gate — the derived
        # `scalar_data_range` / `color_data_range` the array writers put straight
        # onto the group, and the `content_hash` finalize adds to every node.
        # Each of those is rejected as an unknown attribute on the way back in,
        # so an exclusion list would break this command every time a new stamp
        # lands. `shading` / `double_sided` stay explicit arguments below.
        #
        # `data.metadata`, NOT `get_node_metadata`: `get_mesh` has already turned
        # the stored column-major 16-list back into a 4x4 matrix, which is the
        # convention `add_mesh` expects — handing back the raw list would
        # transpose the transform on the round trip.
        #
        # The compositing keys among these (transform, opacity, blending_mode, …)
        # land on the `kind=lod` WRAPPER group rather than the children: the
        # adder splits `COMPOSITING_ATTRS` onto the wrapper, which is where a
        # per-layer setting belongs and where the viewer inherits it from.
        forwardable = KNOWN_RENDER_ATTRS | {
            "transform",
            "nd_transform",
            "extend_to_all",
        }
        forwarded = {k: v for k, v in data.metadata.items() if k in forwardable}
        # A texture cannot travel through either structural LOD route, so its
        # sampling attrs cannot travel alone either. Leaving them in `forwarded`
        # makes the adder reject the rewrite for an orphaned texture setting.
        forwarded.pop("texture_filter", None)
        forwarded.pop("texture_wrap", None)

        # The stored scalar window, FORWARDED rather than left to be recomputed
        # from the decoded values. The two agree for an ordinary mesh, but the
        # stamp can legitimately be wider than the values it describes, and then
        # they do not: a coarse child of an existing ladder carries the whole
        # ladder's shared window while its own cluster-averaged values are
        # contracted, so `--node surf/child_0` recomputed a narrower window and
        # the surface came back recoloured — the exact pop the shared window
        # exists to prevent, reintroduced by the rewrite. It rides the private
        # `_scalar_data_range` plumbing key rather than `forwarded` because the
        # public stamp is not an accepted attribute (see the note above).
        #
        # Validated HERE, with the adder's own validator, for the same reason
        # `--method` is: the deletion below is irreversible, and a corrupt stamp
        # would otherwise be diagnosed with the output already gone.
        scalar_range = (
            validate_scalar_data_range(
                node_path, data.metadata.get("scalar_data_range")
            )
            if data.scalars is not None
            else None
        )

        # `colormap='custom'` is a SENTINEL, not a name: the writer resolves any
        # non-builtin colormap — an ndarray LUT, but also a plain matplotlib or
        # colorcet name like 'magma' — to a `colormap_lut` dataset plus that
        # word. Forwarding the word alone reaches the resolver as a name and
        # raises "Unknown colormap 'custom'", so a `magma` mesh could not be
        # laddered at all; accepting it would have been worse, silently
        # substituting the default LUT. Handing back the ARRAY is lossless: the
        # writer re-resolves it to the same LUT plus the same sentinel on every
        # child.
        if forwarded.get("colormap") == "custom":
            lut = source.get_colormap_lut(node_path)
            if lut is None:
                raise ValueError(
                    f"Mesh node {node_path!r} declares colormap='custom' but has no "
                    "'colormap_lut' dataset, so its colors cannot be reproduced. "
                    "The store is inconsistent; re-write the source scene."
                )
            forwarded["colormap"] = lut

        # Everything above this line reads or validates; everything below is
        # either the drop report or destroys/writes the destination. The
        # report sits exactly HERE, in both directions: after
        # `resolve_substitutive_axis_mesh`, `validate_scalar_data_range` and
        # the missing-LUT check above — each can still raise and abort the
        # command, and printing "will not be carried" before an abort that
        # means nothing was carried (or written) at all was actively
        # misleading — and before the `rmtree`/write below, which is the
        # last point where anything is still reversible.
        _report_drops(source, node_path, leaf_name, data)

        # The destination is destroyed only once the write is certain to be
        # attempted — nothing below this point can still abort.
        if output_path.exists():
            shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()

        # The LEAF name, not the full path: the ladder is written at the scene
        # root, so a nested source node keeps its own name rather than inventing
        # a group hierarchy the user did not ask for.
        # The scene-level viewer config travels with the scene, and dropping it
        # undoes the colormap fidelity above on exactly the scenes that have a
        # custom LUT: with no `tone_mapping` the viewer applies its ACES default,
        # which intentionally shifts hues, and the compiler re-emits the notice
        # saying so. `luxar mesh import` states ACES explicitly, so the
        # documented import → lod pipeline lost it too. Same fallback and same
        # reasoning as that command: ACES is the house default, and saying so
        # keeps the "nothing was chosen" notice quiet.
        viewer_config = source.viewer_config or ViewerConfig(tone_mapping="ACES")

        with LuxarZarrCompiler(str(output_path)) as compiler:
            scene = compiler.create_scene(
                dimensions=source.dimensions, viewer_config=viewer_config
            )
            scene.add_mesh(
                leaf_name,
                data.vertices,
                data.faces,
                normals=data.normals,
                normal_dims=data.normal_dims,
                colors=data.colors,
                scalars=data.scalars,
                _scalar_data_range=scalar_range,
                shading=data.metadata.get("shading"),
                double_sided=bool(data.metadata.get("double_sided", True)),
                substitutive_lod=substitutive_spec,
                additive_lod=additive_spec,
                **forwarded,
            )

    # Read back rather than trusting the write: the ladder's whole value is that
    # the viewer can select among its levels, and a store that came out with one
    # level (or with duplicate counts) still "succeeds" at the API boundary.
    written = LuxarScene.load(output_path)
    if recipe == RECIPE_REVEAL:
        # A reveal is ONE node carrying `additive_<i>` subgroups, not N sibling
        # nodes, so the substitutive read-back below would count 1 and report a
        # perfectly good ladder as "too coarse to reduce". The levels are read
        # from the parent's own declaration and each subgroup's face count.
        (mesh_path,) = written.list_meshes()
        parent = written.get_node_metadata(mesh_path)
        n_levels = int(parent.get("n_additive_sublods", 0))
        if n_levels < 2:
            aprint(
                "Note: the surface produced fewer than two levels, so a plain mesh "
                "leaf was written instead of a reveal ladder."
            )
            return []
        level_faces = [
            int(written.get_node_metadata(f"{mesh_path}/additive_{i}")["n_faces"])
            for i in range(n_levels)
        ]
        aprint(
            f"✓ {n_levels} reveal levels on disk, face counts {level_faces} "
            f"(sum {sum(level_faces):,} = the source's faces; the levels are a "
            "PARTITION, not copies)"
        )
        return level_faces

    # `vertex_counts`, not `counts`: the latter is now a PARAMETER (the reveal
    # breakpoint spec), and reusing the name shadowed it. mypy caught it, which is
    # the whole reason the parameter is typed rather than left as Any.
    vertex_counts = sorted(
        int(written.get_node_metadata(m)["n_vertices"]) for m in written.list_meshes()
    )
    if len(vertex_counts) < 2:
        aprint(
            "Note: the surface was too coarse to reduce, so a plain mesh leaf was "
            "written instead of a ladder."
        )
    else:
        aprint(f"✓ {len(vertex_counts)} levels on disk, vertex counts {vertex_counts}")
    return vertex_counts


def lod_command(
    ctx: typer.Context,
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .luxar.zarr scene containing a mesh node."
    ),
    output_path: Path = typer.Argument(..., help="Output .luxar.zarr scene."),
    node: Optional[str] = typer.Option(
        None,
        "--node",
        help="Which mesh node to coarsen. Optional when the scene has exactly one.",
    ),
    levels: int = typer.Option(
        3, "--levels", "-L", help="Number of coarse levels below the original."
    ),
    compression_factor: int = typer.Option(
        4,
        "--compression-factor",
        "-K",
        help="Vertex-count reduction per level: level i targets V / K**i.",
    ),
    method: str = typer.Option(
        "auto",
        "--subst-method",
        help=(
            "Decimation method, one of "
            f"{', '.join(sorted(MESH_SUBSTITUTIVE_METHODS))}. 'auto' resolves to "
            "'qem' through 10,000 vertices and 'cluster' above that. Named "
            "`--subst-method` to match `gsplat lod`, where "
            "it selects the substitutive (level-replacing) reduction — though the "
            "algorithms differ: these decimate a surface, those reduce a Gaussian "
            "mixture."
        ),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing output."
    ),
    recipe: str = typer.Option(
        RECIPE_LEVELS,
        "--recipe",
        help=(
            f"Which ladder to build: {' | '.join(MESH_LOD_RECIPES)}. "
            "'levels' decimates the surface into coarse levels that REPLACE one "
            "another (the default, and what this command has always done). "
            "'reveal' writes an additive ladder of disjoint face groups the viewer "
            "concatenates, so a prefix is a partial surface rather than a coarse "
            "one — a mesh has no coarse prefix, which is why the two are separate "
            "recipes and `add_mesh` refuses them together."
        ),
    ),
    add_method: Optional[str] = typer.Option(
        None,
        "--add-method",
        "-m",
        help=(
            "[--recipe reveal] Reveal ordering, one of "
            f"{', '.join(sorted(MESH_ADDITIVE_METHODS))} "
            f"(default {DEFAULT_MESH_ADDITIVE_METHOD}). Only orderings whose every "
            "prefix is ONE connected patch are admitted — that restriction is what "
            "makes a partial load a growing surface rather than lace."
        ),
    ),
    n_lods: Optional[int] = typer.Option(
        None,
        "--n-lods",
        help="[--recipe reveal] Number of levels in the ladder (default 4).",
    ),
    counts: Optional[str] = typer.Option(
        None,
        "--counts",
        "--breakpoints",
        help=(
            "[--recipe reveal] Explicit level boundaries as CUMULATIVE face counts "
            "(e.g. 500,2000,10000), or a streaming ladder as 'stream:C'. Mutually "
            "exclusive with --n-lods."
        ),
    ),
    reveal_centre: Optional[str] = typer.Option(
        None,
        "--reveal-centre",
        help=(
            "[--recipe reveal] Comma-separated centre the shells grow from. "
            "Defaults to the mesh's own bounding-box centre, so a surface far from "
            "the origin still grows from its middle."
        ),
    ),
    spatial_dims: Optional[str] = typer.Option(
        None,
        "--spatial-dims",
        help=(
            "[--recipe reveal] Comma-separated columns the shell distance spans. "
            "ORDER IS SIGNIFICANT — it pairs with --reveal-centre's coordinates. "
            "Use it to keep a stacked time/channel column out of the distance."
        ),
    ),
    # Declared only so the body can raise a pointer; see `gsplat lod`'s pair of
    # hidden options for the general reason (typer rejects an unknown option
    # before the body runs, so an undeclared flag is undiagnosable).
    #
    # Mesh needs the pointer MORE than gsplat does, and for a reason unique to it:
    # `-m` is not merely gone, it is RESERVED — should mesh ever gain an additive
    # ordering knob it would be `-m/--add-method`, matching gsplat. So a script that
    # says `-m qem` would, after that, be naming the additive-ordering flag with a
    # decimation value. Typer's own "No such option: -m" says nothing about that.
    #
    # ONLY the long form now. `-m` was declared here while it was RESERVED; it has
    # since been claimed by `--add-method` above, which is exactly what the
    # reservation was for — so `-m` is a live flag again rather than a migration
    # pointer, and declaring it twice would be a typer conflict.
    #
    # A legacy `-m cluster` therefore reaches `--add-method` and is refused there.
    # That refusal names `--subst-method` when the value is a decimation method, so
    # the migration is still spelled out; see `_build_reveal_spec`. This is the
    # "one token, two replacements" case the rename comment predicted, now resolved
    # by the recipe: `-m` belongs to the additive ordering on both commands.
    legacy_method: Optional[str] = typer.Option(None, "--method", hidden=True),
) -> None:
    """Build a LOD ladder for a mesh scene — decimated levels, or a reveal.

    Writes a `kind=lod` group whose coarse children are progressively decimated
    copies of the surface and whose finest child is the original. The viewer shows
    exactly one at a time, chosen by how much of the screen the object covers.

    Levels that cannot reduce the surface are dropped, so a small mesh may come
    back with fewer than `--levels` — or, if it cannot be reduced at all, as a
    plain leaf.

    The output scene contains ONLY the picked mesh's ladder: every other node in
    the input scene (other points/lines/gsplats/mesh nodes, other groups,
    user-authored overlays), any placement/compositing an ancestor group actually
    set (transform/opacity/blending_mode/…; a bare namespace group or an existing
    ladder's own wrapper sets none and is not reported), and the picked mesh's
    own per-vertex `labels`/`image_labels`/`keys` are not carried across. Each is
    named in a warning before anything is written.

    \b
    Examples:
      luxar mesh lod bunny.luxar.zarr bunny_lod.luxar.zarr
      luxar mesh lod scan.luxar.zarr scan_lod.luxar.zarr -L 4 -K 3
      luxar mesh lod multi.luxar.zarr out.luxar.zarr --node surfaces/skull
      luxar mesh lod bunny.luxar.zarr bunny_lod.luxar.zarr --subst-method qem
    """
    # NOT routed through `LEGACY_METHOD_FLAGS`, deliberately: that table maps
    # `--method` → `--add-method`, which is right for every gsplat surface and
    # WRONG here — mesh's bare `--method` was the substitutive one. One token, two
    # replacements, decided by the command; a shared table cannot express that,
    # which is precisely the collision this rename removes from the flag surface.
    if legacy_method is not None:
        raise typer.BadParameter(
            "--method was renamed to --subst-method (2026-08: no method flag is "
            "bare). -m is no longer an alias for it — it now names the reveal "
            f"ordering, as on `gsplat lod`; use --subst-method {legacy_method}."
        )
    _reject_cross_recipe_flags(
        ctx=ctx,
        recipe=recipe,
        add_method=add_method,
        n_lods=n_lods,
        counts=counts,
        reveal_centre=reveal_centre,
        spatial_dims=spatial_dims,
        levels_given=_was_supplied(ctx, "levels"),
        compression_given=_was_supplied(ctx, "compression_factor"),
        subst_method_given=_was_supplied(ctx, "method"),
    )
    try:
        run_lod(
            input_path=input_path,
            output_path=output_path,
            node_name=node,
            levels=levels,
            compression_factor=compression_factor,
            method=method,
            overwrite=overwrite,
            recipe=recipe,
            add_method=add_method,
            n_lods=n_lods,
            counts=counts,
            reveal_centre=reveal_centre,
            spatial_dims=spatial_dims,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1) from exc


def register_lod_commands(app: typer.Typer) -> None:
    """Attach the mesh LOD commands to the mesh CLI."""
    app.command("lod")(lod_command)
