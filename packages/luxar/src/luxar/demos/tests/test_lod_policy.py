"""Every fitting demo must choose an LOD topology deliberately.

The five shipped archives that had no ladder at all did not lose one: they never
had it, because ``fit_gaussian_splats`` returns a single additive sub-LOD while
the progressive fitter happens to return several. So the topology a demo shipped
was decided by which fitter it called. This gate turns that into a choice: a
demo either routes its cache write through :mod:`luxar.demos._lod_policy`, or
appears below with a reason.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import luxar.demos._lod_policy as lod_policy
from luxar.demos import registry
from luxar.demos._lod_policy import (
    _RECIPE_DEFAULTS,
    SCENE_TOPOLOGY_RECIPES,
    SLICED_LADDER_MAX_DEPTH,
    TOPOLOGY_PRESERVING_ADDERS,
    TREE_RECIPES,
    DemoRecipe,
    hidden_axis_stops,
    save_with_lod,
    stream_ladder,
)

#: How a demo may read a cached artifact back. ``GSplatData.load`` flattens and
#: so cannot see a partition; ``load_dataset_gsplats`` ends in that same call.
_FLAT_LOADERS = ("GSplatData.load", "load_dataset_gsplats")

_FIT_CALLS = {
    "fit_gaussian_splats",
    "fit_progressive_gaussian_splats",
    "fit_tiled",
}

#: Demos that fit but write no cached artifact of their own, and why.
_NO_CACHED_ARTIFACT = {
    "demo_quantum_orbitals.py": (
        "the volume is evaluated from a closed-form wavefunction on every run, "
        "so there is no fitted archive to ship or to give a topology to"
    ),
}

#: Demos that choose the topology of their shipped artifact themselves, and why.
#:
#: What qualifies is narrow: the demo names the topology it ships in its own
#: source, with a literal ``build_recipe(data, "<recipe>", ...)`` call, instead of
#: letting :func:`save_with_lod` make the choice. That is still a deliberate,
#: reviewable decision — the helper simply is not the one carrying it out. A demo
#: that keeps whatever topology its fitter happened to produce has made no choice
#: at all and must route its cache write through :func:`save_with_lod` instead.
#:
#: This excuses a member from the *choosing* gate only. The topology it names is
#: still put through the two round-trip gates (see :func:`_chosen_recipes`), since
#: naming ``levels`` yourself costs exactly what asking the helper for it costs.
_CHOOSES_OUTSIDE_THE_POLICY = {
    "demo_gsplats_4d_cell_tracking_challenge.py": (
        "build_lod() asks build_recipe for 'levels' with time as a hard coarsening "
        "barrier (coarsen_dims=(0, 1, 2)), and the precomputed crop it ships is "
        "that result; its two bare .save() calls are the per-timepoint refit cache "
        "and that already-laddered crop"
    ),
    "demo_gsplats_4d_zebrafish_timelapse.py": (
        "same shape as its cell-tracking sibling: build_lod() asks build_recipe "
        "for 'levels' with time as a hard coarsening barrier "
        "(coarsen_dims=(0, 1, 2)) and the 4D archive it ships is that result; its "
        "two bare .save() calls are the per-timepoint fit cache, which is scratch "
        "consumed by the stack, and that already-laddered archive"
    ),
}

#: Sliced Points/Lines calls whose additive ladder does not paint first, and why.
_SLICED_ADDITIVE_LOD_EXEMPTIONS = {
    ("demo_biodiversity_planetary_scale.py", 2657): (
        "the eager coarsest substitutive level paints first; the additive ladder "
        "only refines that already-visible partition in the background"
    ),
}

#: Sliced GSplats calls exempt from the slice-even ladder policy, and why.
#:
#: The gsplats peer of :data:`_SLICED_ADDITIVE_LOD_EXEMPTIONS`, and empty on
#: purpose: swept over the whole corpus, ``add_gsplats``/``_from_data``/
#: ``_from_file`` pass ``additive_lod=`` exactly once (the NEXRAD supercell), and
#: it names ``slice_dims=`` and ``recompute=True``. The empty dict is the
#: tripwire — a new sliced gsplat ladder must satisfy both, or land here with a
#: reason (one entry excuses a location from BOTH requirements).
#:
#: Unlike its Points/Lines counterpart an entry stands on its REASON alone; it
#: carries no structural co-requirement. That counterpart demands
#: ``substitutive_lod=``, because the one excuse it recognises is "an eager coarse
#: level paints first". Here there are TWO legitimate excuses and only one of them
#: is visible in the call: ``lod_group=`` (the gsplats spelling of
#: ``substitutive_lod=``, same reasoning), and a node that carries a hidden
#: dimension but is NOT sliced on it because ``extend_to_all=`` replicates it
#: across every coordinate — for which one interleave group is the correct and
#: meaningless answer. :func:`_declares_a_hidden_dimension` is FILE-level and this
#: gate deliberately does not try to read ``extend_to_all`` (at
#: ``demo_gsplats_3d_kidney_multichannel_toggles.py:535`` it is a VARIABLE), so
#: requiring ``lod_group=`` would leave that shape with no way out at all.
_SLICED_GSPLATS_ADDITIVE_LOD_EXEMPTIONS: dict[tuple[str, int], str] = {}

#: No fitting demo may be parked here instead of choosing a topology.
#:
#: This empty set is a tripwire: new fitting demos must route through the policy
#: or qualify for one of the explicit exemptions above. The downloadable archives
#: still need regeneration after policy changes; that publication pass is #1879.
_NOT_YET_ROUTED: set[str] = set()


def _demo_sources() -> dict[str, str]:
    return {
        p.name: p.read_text() for p in sorted(registry._DEMOS_DIR.glob("demo_*.py"))
    }


def _additive_lod_calls(src: str) -> list[tuple[ast.Call, ast.expr]]:
    """The ``add_points``/``add_lines`` calls with ``additive_lod=`` in *src*."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr not in ("add_points", "add_lines")
        ):
            continue
        out.extend(
            (node, keyword.value)
            for keyword in node.keywords
            if keyword.arg == "additive_lod"
        )
    return out


#: The gsplats adders this gate walks.
_GSPLATS_ADDERS = ("add_gsplats", "add_gsplats_from_data", "add_gsplats_from_file")


def _gsplats_additive_lod_calls(src: str) -> list[tuple[ast.AST, ast.expr]]:
    """Every ``additive_lod=`` bound for a gsplats adder in *src*.

    The peer of :func:`_additive_lod_calls`, split out rather than folded into
    it because the two families do not share a spelling: Points/Lines route
    through :func:`stream_ladder`, whose ``slices=`` term is a POLICY resolved at
    authoring time, while gsplats pass ``make_additive_lod`` kwargs straight
    through and get slice-evenness from ``slice_dims=`` instead. A gsplats call
    was simply invisible to the older gate — which is the hole #2485 fell into.

    TWO shapes are collected, because the direct keyword is not the only route:

    1. ``add_gsplats*(..., additive_lod=<spec>)`` — the keyword on the call.
    2. ``additive_lod`` as an entry of ANY dict literal in the file — a
       ``{"additive_lod": …}`` key or a ``dict(additive_lod=…)`` keyword,
       wherever it is built.

    Shape 2 exists because a ``**`` spread is opaque to AST:
    ``demo_gsplats_4d_drosophila_embryogenesis.py:748`` already calls
    ``add_gsplats_from_data(..., **appearance)`` in a hidden-dim demo, so moving
    ``additive_lod`` into ``appearance`` would escape the requirements entirely.
    Failing every spread call instead would red the corpus today for no defect
    (two demos spread, neither passes a ladder). Collecting the dict entry is the
    cheap middle: swept over the corpus there are ZERO such entries, so it costs
    nothing now and closes the hole for later. It over-reaches slightly by
    design — a dict entry bound for some other consumer would also be gated —
    which is the safe direction, and the exemption dict is the escape hatch.
    """
    out: list[tuple[ast.AST, ast.expr]] = []
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _GSPLATS_ADDERS
        ):
            out.extend(
                (node, keyword.value)
                for keyword in node.keywords
                if keyword.arg == "additive_lod"
            )
        elif isinstance(node, ast.Dict):
            out.extend(
                (node, value)
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and key.value == "additive_lod"
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            out.extend(
                (node, keyword.value)
                for keyword in node.keywords
                if keyword.arg == "additive_lod"
            )
    return out


def _dict_literal_items(node: ast.expr) -> dict[str, ast.expr] | None:
    """The keyword name → value map of a readable ``dict(...)`` / ``{...}`` literal.

    ``None`` means "not readable here" — a variable, a helper call, a ``**spread``
    — and the caller must FAIL on that rather than wave it through: a spec whose
    keys cannot be read off the source is a spec no reviewer can check either.
    Values are returned, not just names, because one of the two gate requirements
    is on a VALUE (``recompute=True``; ``recompute=False`` is a key that satisfies
    nothing).
    """
    if isinstance(node, ast.Dict):
        if not all(
            isinstance(key, ast.Constant) and isinstance(key.value, str)
            for key in node.keys
        ):
            return None
        return {
            key.value: value  # type: ignore[union-attr,misc]
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
        and all(keyword.arg is not None for keyword in node.keywords)
    ):
        return {str(keyword.arg): keyword.value for keyword in node.keywords}
    return None


def _is_literal_true(node: ast.expr | None) -> bool:
    """Whether *node* is the literal ``True``."""
    return isinstance(node, ast.Constant) and node.value is True


def _is_literal_none(node: ast.expr | None) -> bool:
    """Whether *node* is the literal ``None``."""
    return isinstance(node, ast.Constant) and node.value is None


def _declares_a_hidden_dimension(src: str) -> bool:
    """Whether *src* declares a ``Dimension(..., display=False)``."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Dimension"
        and any(
            keyword.arg == "display"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        )
        for node in ast.walk(ast.parse(src))
    )


def _fitting_demos() -> dict[str, str]:
    """Demo file name -> source, for demos that call a fitter."""
    out = {}
    for name, src in _demo_sources().items():
        tree = ast.parse(src, filename=name)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FIT_CALLS
            for node in ast.walk(tree)
        ):
            out[name] = src
    return out


def _policy_recipes(src: str) -> list[str]:
    """The ``recipe=`` literals passed to ``save_with_lod`` in *src*.

    ``recipe`` is keyword-only on :func:`save_with_lod`, so a real call always
    contributes one entry here — which makes a non-empty result the definition
    of "routed" everywhere below. A substring test for the helper's NAME would
    instead be satisfied by the import line alone, passing a demo that imports
    the policy and then still writes its cache with a bare ``.save()``.
    """
    recipes = []
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "save_with_lod"
        ):
            for kw in node.keywords:
                if kw.arg == "recipe" and isinstance(kw.value, ast.Constant):
                    recipes.append(kw.value.value)
    return recipes


def _literal_recipes(src: str) -> list[str]:
    """The recipe literals *src* passes to ``build_recipe`` itself.

    ``recipe`` is the SECOND positional parameter of
    :func:`luxar.gsplats.lod.recipes.build_recipe` and positional-or-keyword, so
    both spellings count. Parsed, not grepped, for :func:`_policy_recipes`'
    reason plus one of its own: the import line already contains the name, and a
    ``build_recipe(data, recipe, params)`` that takes its recipe from a VARIABLE
    (the recipes-gallery demo loops over all of them) names no topology a
    reviewer can read off the source. Only a constant in the recipe slot is the
    visible choice ``_CHOOSES_OUTSIDE_THE_POLICY`` accepts in place of the helper.
    """
    recipes = []
    for node in ast.walk(ast.parse(src)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_recipe"
        ):
            continue
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            recipes.append(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "recipe" and isinstance(kw.value, ast.Constant):
                recipes.append(kw.value.value)
    return recipes


def _chosen_recipes(name: str, src: str) -> list[str]:
    """Every topology *name* chose, through whichever door it chose it.

    The two round-trip gates below ask a different question from the one the
    exemption answers: not "who applied the recipe?" but "do the extra bytes it
    costs reach the scene, or are they discarded?". Keying those on
    :func:`_policy_recipes` alone would let ``_CHOOSES_OUTSIDE_THE_POLICY``
    excuse a demo from them too, which is not what it is for — a demo that names
    ``levels`` itself pays the same ~38% and can throw it away just as easily.
    """
    if name in _CHOOSES_OUTSIDE_THE_POLICY:
        return _literal_recipes(src)
    return _policy_recipes(src)


def _scene_adders(src: str) -> set[str]:
    """The ``add_gsplats*`` methods *src* builds its scene with.

    Parsed, not substring-matched, for the reason :func:`_policy_recipes` gives
    in reverse: ``"add_gsplats" in src`` is true of every demo here, since
    ``add_gsplats_from_data`` contains it. Only the resolved attribute name
    tells the flattening adder from the two that preserve a topology.
    """
    out = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name is not None and name.startswith("add_gsplats"):
            out.add(name)
    return out


def _bare_saves(src: str) -> int:
    """Count ``<something>.save(...)`` calls — a cache write that skipped the policy.

    ``numpy.save`` is excluded by name: the sidecar writers spell it ``np.save``
    / ``np.savez_compressed`` and have nothing to do with a gsplat archive.
    """
    n = 0
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "save":
            continue
        value = node.func.value
        if isinstance(value, ast.Name) and value.id in ("np", "numpy"):
            continue
        n += 1
    return n


def test_every_fitting_demo_chooses_a_topology_or_is_listed() -> None:
    unrouted = {
        name for name, src in _fitting_demos().items() if not _policy_recipes(src)
    }
    accounted = (
        set(_NO_CACHED_ARTIFACT) | set(_CHOOSES_OUTSIDE_THE_POLICY) | _NOT_YET_ROUTED
    )
    assert unrouted <= accounted, (
        "a fitting demo writes its cache without choosing an LOD topology:\n  "
        + "\n  ".join(sorted(unrouted - accounted))
        + "\nThere are three ways out. Route its save through "
        "luxar.demos._lod_policy.save_with_lod; or, if it ships no artifact at "
        "all, add it to _NO_CACHED_ARTIFACT with that reason; or, if it already "
        "names the topology of the artifact it ships in its own source with a "
        "literal build_recipe(...) call, add it to _CHOOSES_OUTSIDE_THE_POLICY "
        "with that reason."
    )


def test_the_pending_list_stays_empty() -> None:
    """The former backlog stays empty now that every fitting demo has a policy."""
    assert not _NOT_YET_ROUTED, (
        "route the demo through save_with_lod or take an explicit exemption — "
        "_NOT_YET_ROUTED is closed"
    )
    fitting = _fitting_demos()
    for name in sorted(_NO_CACHED_ARTIFACT):
        assert name in fitting, f"{name} no longer fits — drop the exemption"


def test_the_outside_the_policy_exemptions_still_earn_their_place() -> None:
    """An exemption granted for a reason must not outlive it.

    Its members are excused only because their source states a topology, so
    measure that claim rather than take it: the demo must still fit, must still
    NOT use the helper (once it does, it is routed, and every other assertion
    here starts applying to it — the exemption would hide them), and the literal
    ``build_recipe`` call the whole excuse rests on must still be present.
    """
    from luxar.gsplats.lod import RECIPE_NAMES

    fitting = _fitting_demos()
    for name, reason in sorted(_CHOOSES_OUTSIDE_THE_POLICY.items()):
        assert name in fitting, f"{name} no longer fits — drop the exemption"
        assert not _policy_recipes(fitting[name]), (
            f"{name} now routes through save_with_lod — drop it from "
            "_CHOOSES_OUTSIDE_THE_POLICY so the routed assertions cover it"
        )
        literals = _literal_recipes(fitting[name])
        assert literals, (
            f"{name} no longer names a topology with a literal build_recipe() "
            f"call, so its exemption ({reason}) no longer describes it"
        )
        # A name is only a topology if a builder answers to it. The recipes have
        # been renamed once already (substitutive -> levels), so a stale or
        # typo'd literal is a real way for this excuse to stop meaning anything
        # while still parsing.
        unknown = sorted(set(literals) - set(RECIPE_NAMES))
        assert not unknown, (
            f"{name} names {unknown}, which build_recipe does not define "
            f"(it knows {list(RECIPE_NAMES)}) — an unbuildable recipe is not the "
            "choice the exemption rests on"
        )


def test_a_routed_demo_writes_every_archive_through_the_policy() -> None:
    """One policy save does not licence a second, bare one in the same demo.

    A demo with several archives (cmu1's three channels, nexrad's per-frame
    files) would otherwise satisfy the gate above with a single routed call
    while its remaining cache writes kept whatever topology their fitter
    produced — the defect the policy exists to remove, hidden behind a demo
    that looks routed.
    """
    for name, src in sorted(_fitting_demos().items()):
        if not _policy_recipes(src):
            continue
        assert _bare_saves(src) == 0, (
            f"{name} chooses a topology but still writes {_bare_saves(src)} "
            "archive(s) with a bare .save() — route those through "
            "luxar.demos._lod_policy.save_with_lod too."
        )


def test_every_chosen_recipe_is_one_the_policy_defines() -> None:
    """A typo'd recipe name would otherwise fail only at refit time."""
    known = set(_RECIPE_DEFAULTS)
    for name, src in sorted(_fitting_demos().items()):
        for recipe in _policy_recipes(src):
            assert recipe in known, (
                f"{name}: unknown recipe {recipe!r}; the policy defines {sorted(known)}"
            )


def test_a_demo_does_not_mix_topologies_across_its_own_archives() -> None:
    """Channels of one dataset must agree, or the record cannot state a topology.

    A demo writing some archives as `levels` and others as `stream` would give
    its Zenodo entry no single answer to "what LOD does this carry?".
    """
    for name, src in sorted(_fitting_demos().items()):
        recipes = set(_policy_recipes(src))
        assert len(recipes) <= 1, f"{name} writes mixed topologies: {sorted(recipes)}"


def test_the_policy_recipes_are_all_reachable_from_a_demo() -> None:
    """A recipe no demo uses is dead configuration; say so early."""
    used = {r for src in _fitting_demos().values() for r in _policy_recipes(src)}
    unused = set(_RECIPE_DEFAULTS) - used
    assert not unused, (
        f"the policy defines {sorted(unused)} but no demo asks for it — either "
        "wire a demo to it or drop it"
    )


def test_the_detector_actually_finds_routed_demos() -> None:
    """A scan that matched nothing would pass every assertion above."""
    fitting = _fitting_demos()
    assert len(fitting) >= 10, f"only found {len(fitting)} fitting demos"
    routed = {n for n, s in fitting.items() if _policy_recipes(s)}
    assert len(routed) >= 5, f"only {len(routed)} demos routed through the policy"
    assert any(_policy_recipes(fitting[n]) for n in routed), (
        "no recipe literal parsed out — the recipe assertions are vacuous"
    )


def test_the_routing_detectors_are_not_fooled_by_the_import_line() -> None:
    """Run both detectors over planted sources, since both gate everything else.

    The import-only case is the one that matters: it is what a half-finished
    migration looks like, and a name-substring test reads it as routed.
    """
    import_only = (
        "from luxar.demos._lod_policy import save_with_lod\n"
        "result.save(cache_file, compress='zip')\n"
    )
    assert _policy_recipes(import_only) == []
    assert _bare_saves(import_only) == 1

    routed_plus_bare = (
        "save_with_lod(a, p0, recipe='stream')\n"
        "b.save(p1, compress='zip')\n"
        "np.save(p2, arr)\n"  # a sidecar, not an archive
    )
    assert _policy_recipes(routed_plus_bare) == ["stream"]
    assert _bare_saves(routed_plus_bare) == 1


def test_the_literal_recipe_detector_demands_a_readable_choice() -> None:
    """Same treatment for the third door's detector, which gates an exemption.

    A detector that answered "yes" too easily would excuse a demo that in fact
    chose nothing — the defect the whole file exists to catch, wearing the
    exemption as cover.
    """
    import_only = (
        "from luxar.gsplats.lod import RecipeParams, build_recipe\n"
        "result.save(cache_file, compress='zip')\n"
    )
    assert _literal_recipes(import_only) == []
    # A recipe read out of a variable is not a choice anyone can review.
    assert _literal_recipes("built = build_recipe(base, recipe, _params())") == []
    # The real shape, positionally and by keyword.
    assert _literal_recipes("build_recipe(d, 'levels', RecipeParams(levels=3))") == [
        "levels"
    ]
    assert _literal_recipes("build_recipe(d, recipe='levels', params=p)") == ["levels"]


def test_the_demo_recipe_type_and_defaults_agree() -> None:
    """`DemoRecipe` and `_RECIPE_DEFAULTS` must list the same recipes."""
    from typing import get_args

    assert set(get_args(DemoRecipe)) == set(_RECIPE_DEFAULTS)


def test_a_demo_choosing_a_tree_recipe_does_not_read_its_cache_flat() -> None:
    """The gate the AST checks above could not see.

    ``adaptive`` writes a ``kind=partition`` store, which ``GSplatData.load``
    refuses ("node is not matrix-shaped"). A demo that picks it and still loads
    its cache flat is broken on its DEFAULT path — the one every user takes —
    while ``--recompute``, the path an author runs, stays green. That is exactly
    how it shipped to review once.
    """
    for name, src in sorted(_fitting_demos().items()):
        tree = sorted(set(_chosen_recipes(name, src)) & TREE_RECIPES)
        if not tree:
            continue
        for loader in _FLAT_LOADERS:
            assert loader not in src, (
                f"{name} writes a {tree} "
                f"artifact but reads it back through {loader}, which cannot open a "
                "multi-part store. Fetch the PATH (ensure_dataset) and graft it with "
                "Group.add_gsplats_from_file()."
            )


def test_a_costly_recipe_is_only_chosen_where_the_scene_can_carry_it() -> None:
    """The other half of "does the topology survive the round trip?".

    ``test_a_demo_choosing_a_tree_recipe_does_not_read_its_cache_flat`` catches
    the loud failure: ``adaptive`` plus a flat READ raises. This catches the
    silent one. A demo can read its archive back perfectly and still throw the
    topology away at the next step, by rebuilding the scene from loose
    ``centers=``/``amplitudes=`` arrays — every one of which is a view of the
    finest level. ``add_gsplats`` then writes a flat leaf and the extra bytes
    (+38% for ``levels`` on a real fit) buy nothing. Nothing raises; the demo
    simply pays for a ladder no viewer will ever be offered.
    """
    for name, src in sorted(_fitting_demos().items()):
        costly = set(_chosen_recipes(name, src)) & SCENE_TOPOLOGY_RECIPES
        if not costly:
            continue
        adders = _scene_adders(src)
        flattening = adders - TOPOLOGY_PRESERVING_ADDERS
        assert not flattening, (
            f"{name} writes {sorted(costly)} but builds its scene with "
            f"{sorted(flattening)}, which flattens to a single leaf — the extra "
            "bytes are spent and then discarded. Either hand the whole "
            "GSplatData to add_gsplats_from_data / graft the path with "
            "add_gsplats_from_file, or choose 'stream'."
        )


def test_the_scene_adder_detector_tells_the_three_adders_apart() -> None:
    """A substring test would read all three as the flattening one."""
    assert _scene_adders("scene.add_gsplats(name='g', centers=c)") == {"add_gsplats"}
    assert _scene_adders("scene.add_gsplats_from_data(name='g', result=r)") == {
        "add_gsplats_from_data"
    }
    assert _scene_adders("scene.add_gsplats_from_file(name='g', path=p)") == {
        "add_gsplats_from_file"
    }
    # Mentioning an adder without calling it must not count.
    assert _scene_adders("x = 'add_gsplats'\nimport add_gsplats_from_data\n") == set()
    assert TOPOLOGY_PRESERVING_ADDERS.isdisjoint({"add_gsplats"})


def test_the_costly_recipe_gate_is_not_vacuous() -> None:
    """It must actually be watching a demo, and a real adder call."""
    fitting = _fitting_demos()
    watched = {
        n
        for n, s in fitting.items()
        if set(_chosen_recipes(n, s)) & SCENE_TOPOLOGY_RECIPES
    }
    assert watched, "no demo chooses a scene-topology recipe — the gate is vacuous"
    # Negative control for the third door: a demo excused from CHOOSING through
    # the policy is not excused from spending the bytes wisely, so a costly
    # topology named in its own source must still land in `watched`.
    for n in sorted(_CHOOSES_OUTSIDE_THE_POLICY):
        if set(_literal_recipes(fitting[n])) & SCENE_TOPOLOGY_RECIPES:
            assert n in watched, (
                f"{n} names a costly topology but the gate does not see it — its "
                "exemption would licence spending those bytes and discarding them"
            )
    for n in sorted(watched):
        assert _scene_adders(fitting[n]), (
            f"{n} is watched by the gate but no add_gsplats* call parsed out of "
            "it — the assertion cannot fail for the right reason"
        )
    assert SCENE_TOPOLOGY_RECIPES <= set(_RECIPE_DEFAULTS)


def _tiny_fit(rng: np.random.Generator, n: int, ndim: int):
    """A small well-formed GSplatData — enough splats for a real BSP split."""
    from luxar.gsplats.gsplat_data import GSplatData

    tril = ndim * (ndim + 1) // 2
    chol = np.zeros((n, tril), dtype=np.float32)
    # Unit-diagonal Cholesky: the diagonal entries sit at the triangular-number
    # offsets, everything off-diagonal stays 0.
    for d in range(ndim):
        chol[:, d * (d + 1) // 2 + d] = 1.0
    return GSplatData(
        centers=(rng.random((n, ndim)) * 100.0).astype(np.float32),
        amplitudes=(rng.random(n) + 0.1).astype(np.float32),
        cholesky_factors=chol,
    )


@pytest.mark.parametrize("recipe", sorted(_RECIPE_DEFAULTS))
def test_the_tree_recipe_list_matches_what_the_builders_write(
    recipe: str, tmp_path: Path
) -> None:
    """``TREE_RECIPES`` is load-bearing, so measure it rather than trust it.

    A recipe added to the policy without being classified here would sail past
    the AST gate above.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    out = tmp_path / f"{recipe}.gsplats.zarr"
    save_with_lod(
        _tiny_fit(np.random.default_rng(0), 600, 2), out, recipe=recipe, quiet=True
    )

    if recipe in TREE_RECIPES:
        with pytest.raises(ValueError, match="matrix-shaped"):
            GSplatData.load(out, include_stats=False)
    else:
        assert len(GSplatData.load(out, include_stats=False).amplitudes) > 0


@pytest.mark.parametrize("recipe", sorted(_RECIPE_DEFAULTS))
def test_every_recipe_grafts_into_a_scene_from_its_file(
    recipe: str, tmp_path: Path
) -> None:
    """The load path a demo actually uses, for every topology the policy offers.

    Mirrors cmu1's call shape (2D, per-channel colormapped layer), because that
    is the one that broke: a partition reaches the scene by PATH, not by array.

    Also pins the kwarg asymmetry that came with it — a grafted subtree refuses
    ``dim_order``, since the file is already authored in its own dims. cmu1
    passed one, so this is the second half of the same defect.
    """
    from luxar import Dimension, Dimensions, LuxarZarrCompiler

    artifact = tmp_path / f"{recipe}.gsplats.zarr"
    save_with_lod(
        _tiny_fit(np.random.default_rng(1), 600, 2), artifact, recipe=recipe, quiet=True
    )

    def _scene(tag: str):
        compiler = LuxarZarrCompiler(tmp_path / f"scene_{recipe}_{tag}.luxar.zarr")
        return compiler, compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension("x", unit="px", display=True),
                    Dimension("y", unit="px", display=True),
                ]
            )
        )

    compiler, scene = _scene("ok")
    with compiler:
        scene.add_gsplats_from_file(
            name="ch",
            path=str(artifact),
            opacity=1.0,
            blending_mode="additive",
            layer=True,
            colormap="red",
        )

    if recipe in TREE_RECIPES:
        compiler, scene = _scene("dimorder")
        with pytest.raises(ValueError, match="dim_order"), compiler:
            scene.add_gsplats_from_file(
                name="ch", path=str(artifact), dim_order=["x", "y"]
            )


def test_a_multi_part_adaptive_tree_grafts_with_its_colormap(tmp_path: Path) -> None:
    """The many-part shape the real datasets have.

    ``save_with_lod`` uses the adaptive recipe's 1,000,000-splat tile cap, so a
    test-sized fit always comes out as ONE part — which would leave the
    interesting case (a partition with many children, each its own level ladder)
    untested. Build that shape directly and check both that it grafts and that
    the per-channel colormap reaches the levels, since ``colormap`` is copied
    down to the leaves while ``blending_mode`` stays on the wrapper.
    """
    from luxar import Dimension, Dimensions, LuxarZarrCompiler
    from luxar._zarr_compat import open_group
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod import RecipeParams, build_recipe

    tree = build_recipe(
        _tiny_fit(np.random.default_rng(3), 1600, 2),
        "adaptive",
        # Four parts is enough to be multi-part; the per-part reduction and the
        # zarr group count both scale with it, so do not raise this casually.
        RecipeParams(n_lods=4, max_elements=400, compression_factor=4, levels=2),
    )
    artifact = tmp_path / "multi.gsplats.zarr"
    write_gsplats_tree(artifact, tree)

    scene_path = tmp_path / "scene_multi.luxar.zarr"
    with LuxarZarrCompiler(scene_path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions(
                [
                    Dimension("x", unit="px", display=True),
                    Dimension("y", unit="px", display=True),
                ]
            )
        )
        scene.add_gsplats_from_file(
            name="ch",
            path=str(artifact),
            opacity=1.0,
            blending_mode="additive",
            layer=True,
            colormap="red",
        )

    root = open_group(str(scene_path), mode="r")["ch"]
    assert dict(root.attrs).get("kind") == "partition"
    parts = sorted(root.group_keys())
    assert len(parts) > 1, f"expected a multi-part graft, got {parts}"
    levels = sorted(root[parts[0]].group_keys())
    assert levels, "a part with no levels means the ladder was dropped"
    assert dict(root[parts[0]][levels[0]].attrs).get("colormap") == "red"


def test_source_tree_is_the_one_being_tested() -> None:
    """Guard against the AST scan reading an installed copy instead of the repo."""
    assert (Path(registry._DEMOS_DIR) / "_lod_policy.py").exists()


def _returns_the_prelod_fit(src: str, name: str) -> list[str]:
    """Names returned after being passed to ``save_with_lod`` in a function.

    The third way to throw a topology away, after the loud one (a tree read
    flat) and the silent one (a scene rebuilt from loose arrays): write the
    ladder to the cache and then hand the SCENE the pre-LOD variable that was
    passed IN. ``save_with_lod`` returns nothing, so the name still refers to the
    flat fit. The warm path loads the archive back and gets the levels, so only
    a local refit (``--recompute``, or no precomputed archive) is flat.
    """

    def function_body_nodes(function: ast.FunctionDef) -> Iterator[ast.AST]:
        stack = list(function.body)
        while stack:
            node = stack.pop()
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue
            yield node
            stack.extend(ast.iter_child_nodes(node))

    out = []
    for function in ast.walk(ast.parse(src, filename=name)):
        if not isinstance(function, ast.FunctionDef):
            continue
        saved = set()
        returned = []
        for node in function_body_nodes(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "save_with_lod"
            ):
                data = (
                    node.args[0]
                    if node.args
                    else next(
                        (kw.value for kw in node.keywords if kw.arg == "data"), None
                    )
                )
                if isinstance(data, ast.Name):
                    saved.add(data.id)
            elif isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                returned.append(node.value.id)
        out.extend(
            returned_name for returned_name in returned if returned_name in saved
        )
    return out


def test_a_costly_recipe_is_not_handed_to_the_scene_as_the_pre_lod_fit() -> None:
    """A demo must not return the variable it just passed to ``save_with_lod``.

    Measured on milkyway before the fix: the archive carried four substitutive
    levels while a ``--recompute`` scene was a flat leaf, because the cache
    branch ended ``return result`` — the same flat fit it had handed in. The
    warm branch loaded LOCAL_FIT back and got the levels, so cold and warm runs
    rendered DIFFERENT scenes from the same code.

    Returning the stored artifact instead fixes both at once, and picks up the
    lossy cache encoding as a bonus, so the two paths agree on bytes as well as
    topology (the rationale spelled out in demo_gsplats_4d_nexrad_supercell).
    """
    for name, src in sorted(_fitting_demos().items()):
        if not set(_chosen_recipes(name, src)) & SCENE_TOPOLOGY_RECIPES:
            continue
        adders = _scene_adders(src)
        if adders - TOPOLOGY_PRESERVING_ADDERS:
            continue  # already flattens; the sibling gate above owns that case
        if adders == {"add_gsplats_from_file"}:
            # Grafts the PATH, so the topology comes off disk and whatever the
            # fit step returned never carries scene geometry. cmu1 returns its
            # in-memory `result` for exactly this reason and is not a bug.
            continue
        offenders = _returns_the_prelod_fit(src, name)
        assert not offenders, (
            f"{name} returns {offenders} after passing them to save_with_lod, "
            "which hands the scene the FLAT pre-LOD fit — the scene loses the "
            "levels the archive carries, and a local refit renders differently "
            "from a later cache hit. Return what was stored instead."
        )


def test_the_pre_lod_return_detector_catches_the_shape_it_is_meant_to() -> None:
    """A detector that never fires would let the bug back in silently."""
    bad = {
        "adjacent": (
            "def f():\n"
            "    result = fit()\n"
            "    save_with_lod(result, path, recipe='levels')\n"
            "    return result\n"
        ),
        "intervening statement": (
            "def f():\n"
            "    result = fit()\n"
            "    save_with_lod(result, path, recipe='levels')\n"
            "    aprint('cached')\n"
            "    return result\n"
        ),
        "dedented return": (
            "def f():\n"
            "    result = fit()\n"
            "    with section():\n"
            "        save_with_lod(result, path, recipe='levels')\n"
            "    return result\n"
        ),
        "keyword data": (
            "def f():\n"
            "    result = fit()\n"
            "    save_with_lod(data=result, path=path, recipe='levels')\n"
            "    return result\n"
        ),
    }
    good = (
        "def f():\n"
        "    result = fit()\n"
        "    save_with_lod(result, path, recipe='levels')\n"
        "    stored = load_local_fit_gsplats_at([path], label='d')\n"
        "    return result if stored is None else stored[0]\n"
    )
    shadowed = (
        "def f():\n"
        "    result = fit()\n"
        "    def cache_other_result():\n"
        "        result = fit_other()\n"
        "        save_with_lod(result, path, recipe='levels')\n"
        "    return result\n"
    )
    for shape, source in bad.items():
        assert _returns_the_prelod_fit(source, f"bad {shape}.py") == ["result"]
    assert _returns_the_prelod_fit(good, "good.py") == []
    assert _returns_the_prelod_fit(shadowed, "shadowed.py") == []


# ─────────────────────────────────────────────────────────────────────
# stream_ladder — the Points/Lines half of the policy
# ─────────────────────────────────────────────────────────────────────


class TestStreamLadder:
    """The ladder spec a Points/Lines leaf shown whole carries.

    Two properties are load-bearing and neither is self-evident from the call
    site, which is why they are pinned here rather than left to the demos.
    """

    def test_the_first_rung_is_the_download_budget_not_a_zarr_chunk(self) -> None:
        # 39,062 = 200 ms at 25 Mbps and 16 B/element. NOT desi's 2,000, which is
        # sized to land an eager coarsest SUBSTITUTIVE level in one zarr chunk; an
        # additive-only leaf has no coarse level, so its first rung IS first
        # paint, and at 2,000 the demos' group counts do not fall at all.
        counts = stream_ladder(6_248_730)["counts"]
        assert counts[0] == 39_062

    def test_increments_stay_capped_for_points(self) -> None:
        from luxar.utils.lod_breakpoints import DEFAULT_MAX_ADDITIVE_COMMIT

        for n in (1_153_506, 3_000_000, 6_248_730, 9_874_128):
            counts = stream_ladder(n)["counts"]
            increments = [b - a for a, b in zip([0, *counts], counts)]
            assert max(increments) <= DEFAULT_MAX_ADDITIVE_COMMIT, n
            assert counts[-1] == n

    def test_lines_get_the_string_form_because_the_units_differ(self) -> None:
        # THE TRAP: on add_lines an explicit `counts` LIST is in POLYLINES while
        # "stream:<c>" is in VERTICES. A vertex-sized list therefore clamps to the
        # polyline count and writes NO rungs — silently. Measured: 4,000 polylines
        # x 27 vertices with counts=[39062, 78124, 108000] produced 0 rungs, while
        # "stream:39062" produced 3. So the Lines spelling must be the string.
        spec = stream_ladder(318_078, geometry="lines")
        assert spec["counts"] == "stream:39062"
        assert isinstance(spec["counts"], str)

    def test_lines_refuse_the_first_size_whose_increment_breaks_the_cap(self) -> None:
        with pytest.raises(
            ValueError, match="resolved chunk 39,062 exceeds the 900,000-vertex"
        ):
            stream_ladder(2_149_985, geometry="lines")

        assert stream_ladder(2_149_984, geometry="lines")["counts"] == "stream:39062"

    def test_points_and_lines_do_not_return_the_same_shape(self) -> None:
        assert isinstance(stream_ladder(500_000)["counts"], list)
        assert isinstance(stream_ladder(500_000, geometry="lines")["counts"], str)

    def test_method_and_seed_are_pinned_so_a_rebuild_reproduces(self) -> None:
        for spec in (stream_ladder(500_000), stream_ladder(500_000, geometry="lines")):
            assert spec["method"] == "random"
            assert spec["seed"] == 0

    def test_an_unknown_geometry_raises_rather_than_guessing(self) -> None:
        # Guessing would silently pick the wrong UNIT, which is the failure this
        # parameter exists to prevent.
        with pytest.raises(ValueError, match="must be 'points' or 'lines'"):
            stream_ladder(1000, geometry="gsplats")

    @pytest.mark.parametrize("geometry", ["points", "lines"])
    def test_non_positive_leaf_size_raises(self, geometry: str) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            stream_ladder(0, geometry=geometry)

    def test_a_leaf_below_the_first_rung_collapses_to_one_level(self) -> None:
        # Not a defect: a node smaller than one first-paint chunk has nothing to
        # stream. NPC relies on this, since `_validate_counts` clamps the shared
        # spec to each part's own count.
        assert stream_ladder(1_000)["counts"] == [1_000]


# ─────────────────────────────────────────────────────────────────────
# hidden_axis_stops + the per-slice first rung (#2374's authoring half)
# ─────────────────────────────────────────────────────────────────────


class TestHiddenAxisStops:
    """Counts hidden coordinate combinations used to detect sliced nodes."""

    def test_counts_distinct_values_on_one_hidden_axis(self) -> None:
        pos = np.array([[0, 0, 0, 5], [0, 0, 0, 7], [0, 0, 0, 7]], dtype=np.float32)
        assert hidden_axis_stops(pos, [3]) == 2

    def test_counts_OCCURRING_combinations_not_the_product(self) -> None:
        # The distinction that matters on sparse data, and the one a
        # product-of-cardinalities definition gets wrong: axis 2 has two values
        # and axis 3 has two, so the product is 4 — but only 3 pairs occur, and
        # only 3 slices exist to divide a rung between.
        pos = np.array([[0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 2, 1]], dtype=np.float32)
        assert hidden_axis_stops(pos, [2, 3]) == 3

    def test_no_hidden_axis_is_one_slice(self) -> None:
        assert hidden_axis_stops(np.zeros((4, 3), dtype=np.float32), []) == 1

    def test_empty_node_is_one_slice_not_zero(self) -> None:
        # A zero would propagate into `slices=0` and raise from stream_ladder,
        # turning an empty layer into a build failure.
        assert hidden_axis_stops(np.zeros((0, 4), dtype=np.float32), [3]) == 1

    def test_rejects_a_column_index_past_the_end(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            hidden_axis_stops(np.zeros((3, 4), dtype=np.float32), [9])

    def test_rejects_non_2d_positions(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            hidden_axis_stops(np.zeros(5, dtype=np.float32), [0])


class TestASlicedNodeGetsAShareOfItsFrame:
    """#2374: a sliced node's first rung is a share of the frame, not a budget."""

    def test_mean_first_paint_is_one_over_the_max_depth_of_the_resident_slice(
        self,
    ) -> None:
        # The contract Loic ruled for. Checked as a SHARE, which is the quantity
        # that predicts whether the opening frame is recognisable — and which is
        # slice-invariant, so it holds at every stop count.
        #
        # MEAN, not per-stop: rung 0 is a prefix of a global ordering, so it
        # concentrates where the signal is rather than spreading in proportion to
        # slice size. `n / stops` below is the mean resident slice; on a
        # non-uniform axis the sparsest stop gets less. This bound is sized for
        # the 2-7 stop categorical axes these demos have; longer axes need their
        # per-stop histogram checked. See SLICED_LADDER_MAX_DEPTH.
        for total, stops in (
            (1_153_506, 6),  # mouse_multiome
            (1_151_006, 2),  # esm3
            (4_485_810, 7),  # zebrahub
            (3_000_000, 3),  # cellxgene
            (6_248_730, 6),  # human_multiome
            (6_572_730, 2),  # arxiv
        ):
            rung0 = stream_ladder(total, slices=stops)["counts"][0]
            resident = total / stops
            assert rung0 / stops / resident == pytest.approx(
                1 / SLICED_LADDER_MAX_DEPTH, rel=1e-3
            )

    def test_the_share_clears_the_viewer_side_floor_with_margin(self) -> None:
        # The gate for #2374 fails a sliced node below 10% of its frame. Authored
        # ladders must not sit on that boundary.
        rung0 = stream_ladder(6_248_730, slices=6)["counts"][0]
        assert rung0 / 6_248_730 > 0.10

    def test_the_floor_needs_no_slice_term(self) -> None:
        # `first_chunk/S >= share * (n/S)` cancels to `first_chunk >= share * n`,
        # so the same total yields the same rung 0 at every stop count. This is
        # why a demo gaining a dimension cannot regress its first paint.
        counts = {
            stream_ladder(4_485_810, slices=s)["counts"][0] for s in (2, 3, 6, 7, 50)
        }
        assert len(counts) == 1

    def test_default_is_unchanged_for_an_unsliced_node(self) -> None:
        # Regression guard: every non-sliced caller must keep its ladder, so the
        # ruling costs nothing on demos that are shown whole.
        assert (
            stream_ladder(1_000_000)["counts"]
            == stream_ladder(1_000_000, slices=1)["counts"]
        )

    def test_every_sliced_additive_ladder_uses_the_slice_policy(self) -> None:
        missing = []
        seen_exemptions = set()
        for name, src in _demo_sources().items():
            if not _declares_a_hidden_dimension(src):
                continue
            for call, additive_lod in _additive_lod_calls(src):
                location = (name, call.lineno)
                if location in _SLICED_ADDITIVE_LOD_EXEMPTIONS:
                    seen_exemptions.add(location)
                    if not any(
                        keyword.arg == "substitutive_lod" for keyword in call.keywords
                    ):
                        missing.append(
                            f"{name}:{call.lineno} is exempt but has no substitutive_lod="
                        )
                    continue
                if not (
                    isinstance(additive_lod, ast.Call)
                    and isinstance(additive_lod.func, ast.Name)
                    and additive_lod.func.id == "stream_ladder"
                ):
                    missing.append(
                        f"{name}:{call.lineno} uses additive_lod="
                        f"{ast.unparse(additive_lod)}"
                    )
                    continue
                call = additive_lod
                if not any(keyword.arg == "slices" for keyword in call.keywords):
                    missing.append(f"{name}:{call.lineno}")
        stale_exemptions = set(_SLICED_ADDITIVE_LOD_EXEMPTIONS) - seen_exemptions
        missing.extend(f"stale exemption {location}" for location in stale_exemptions)
        assert not missing, (
            "Points/Lines additive ladders in demos with hidden dimensions must use "
            f"stream_ladder(..., slices=...) or an explicit exemption: {missing}"
        )

    def test_every_sliced_gsplats_additive_ladder_is_slice_even(self) -> None:
        """A sliced gsplats ladder must be slice-aware AND must actually run.

        #2485, and the concrete testable half of #2482. The sibling gate above
        walks Points/Lines only, so ``add_gsplats_from_data(...,
        additive_lod=dict(breakpoints="stream:20000"))`` was invisible to it — an
        ABSOLUTE first rung on a node the viewer slices 82 ways, whose sparsest
        scan came out holding 4 splats.

        TWO independent requirements, asserted separately because they fail
        separately and a caller fixes them separately:

        ``slice_dims=``
            The ladder is authored slice-aware. On the gsplats side the fix for a
            sliced node is not a sizing policy but an ORDERING one: the ladder is
            interleaved round-robin across the hidden coordinates, so every rung
            carries an equal absolute per-slice budget.

        ``recompute=True``
            The spec actually RUNS rather than being shadowed. This looks
            redundant and is not: ``resolve_additive_axis_gsplats`` computes only
            when ``recompute`` is set or the level holds ``<= 1`` rung, and
            ``GSplatData.combine_as_new_dimension`` MERGES its sources' ladders
            instead of dropping them — so on a stack of already-laddered
            per-timepoint fits the whole spec is a silent no-op and a proportional
            per-source ladder ships instead. That is #2485's own root cause, which
            the first requirement alone does not catch.

            Required unconditionally, including where the fitter output is flat.
            It costs nothing there — ``needs_compute = recompute or
            n_additive_sublods <= 1`` makes it an exact no-op on a single-rung
            level — and a static AST check cannot tell a stacked ``result=`` from
            a flat one, so it cannot rule the hazard out. A harmless flag that
            closes an invisible failure mode is the right default; the alternative
            is a gate that passes the very call site it was written for.

        Both requirements apply to a spec routed through a ``**`` spread as well
        as to one written on the call — see :func:`_gsplats_additive_lod_calls`.
        A node that declares a hidden dimension but is not actually sliced on it
        (``extend_to_all=``) takes the exemption dict, which is why an entry there
        stands on its reason alone.
        """
        not_slice_aware: list[str] = []
        shadowable: list[str] = []
        seen_exemptions = set()
        for name, src in _demo_sources().items():
            if not _declares_a_hidden_dimension(src):
                continue
            for site, additive_lod in _gsplats_additive_lod_calls(src):
                location = (name, site.lineno)  # type: ignore[attr-defined]
                if location in _SLICED_GSPLATS_ADDITIVE_LOD_EXEMPTIONS:
                    # No structural co-requirement: the reason carries it. See
                    # the exemption dict for why (two legitimate excuses, and
                    # only one of them is visible in the call).
                    seen_exemptions.add(location)
                    continue
                if isinstance(additive_lod, ast.Constant) and additive_lod.value in (
                    False,
                    None,
                ):
                    continue  # deliberately no ladder at all
                spelled = f"{name}:{location[1]} uses additive_lod="
                items = _dict_literal_items(additive_lod)
                if items is None:
                    not_slice_aware.append(
                        f"{spelled}{ast.unparse(additive_lod)}, whose keys cannot "
                        "be read from the source"
                    )
                    continue
                if "slice_dims" not in items or _is_literal_none(
                    items.get("slice_dims")
                ):
                    not_slice_aware.append(f"{spelled}{ast.unparse(additive_lod)}")
                if not _is_literal_true(items.get("recompute")):
                    shadowable.append(f"{spelled}{ast.unparse(additive_lod)}")
        stale_exemptions = (
            set(_SLICED_GSPLATS_ADDITIVE_LOD_EXEMPTIONS) - seen_exemptions
        )
        not_slice_aware.extend(
            f"stale exemption {location}" for location in stale_exemptions
        )
        assert not not_slice_aware, (
            "GSplats additive ladders in demos with hidden dimensions must pass "
            "additive_lod=dict(..., slice_dims=[<raw PRE-dim_order centre "
            "columns>]) — NOT the scene's post-dim_order dimension positions, "
            "which are a different frame of reference. An absolute first rung "
            "(breakpoints='stream:<c>' or explicit counts) is divided across the "
            "slices and starves the sparsest. A spec routed through a ** spread "
            f"is gated the same way: {not_slice_aware}"
        )
        assert not shadowable, (
            "GSplats additive ladders in demos with hidden dimensions must also "
            "pass additive_lod=dict(..., recompute=True), or the spec is silently "
            "shadowed by the merged per-source ladder of a stacked dataset and "
            "never reaches make_additive_lod at all. A spec routed through a ** "
            f"spread is gated the same way: {shadowable}"
        )

    def test_a_literal_none_slice_dims_does_not_satisfy_the_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "demo_none_slice_dims.py").write_text(
            """
Dimension("Time", values=[0], display=False)
scene.add_gsplats_from_data(
    data,
    additive_lod=dict(slice_dims=None, recompute=True),
)
"""
        )
        monkeypatch.setattr(registry, "_DEMOS_DIR", tmp_path)

        with pytest.raises(AssertionError, match="slice_dims"):
            self.test_every_sliced_gsplats_additive_ladder_is_slice_even()

    def test_a_small_sliced_node_keeps_its_budget_ladder(self) -> None:
        # The floor is a max(), so a node whose budget rung already exceeds
        # n/L keeps the finer ladder rather than being coarsened to meet a share
        # it already clears.
        assert (
            stream_ladder(200_000, slices=3)["counts"]
            == stream_ladder(200_000)["counts"]
        )

    def test_lines_floors_a_safe_string_form_too(self) -> None:
        whole = int(stream_ladder(1_800_001, geometry="lines")["counts"].split(":")[1])
        sliced = int(
            stream_ladder(1_800_001, geometry="lines", slices=4)["counts"].split(":")[1]
        )
        assert sliced > whole
        assert sliced == -(-1_800_001 // SLICED_LADDER_MAX_DEPTH)

    def test_lines_reject_a_resolved_ladder_over_the_commit_ceiling(self) -> None:
        with pytest.raises(ValueError, match="900,000-vertex commit ceiling"):
            stream_ladder(1_800_005, geometry="lines", slices=4)

        assert (
            stream_ladder(1_800_004, geometry="lines", slices=4)["counts"]
            == "stream:225001"
        )

    def test_rejects_a_sliced_node_too_large_to_deliver_the_share(self) -> None:
        with pytest.raises(ValueError, match="cannot deliver its 12.5% first rung"):
            stream_ladder(7_200_001, slices=2)

        assert stream_ladder(7_200_000, slices=2)["counts"][0] == 900_000

    def test_the_rejection_reports_the_configured_share(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lod_policy, "SLICED_LADDER_MAX_DEPTH", 10)
        with pytest.raises(ValueError, match="cannot deliver its 10% first rung"):
            stream_ladder(9_000_001, slices=2)

    def test_rejects_a_slice_count_below_one(self) -> None:
        with pytest.raises(ValueError, match="slices must be >= 1"):
            stream_ladder(1_000, slices=0)


@pytest.mark.parametrize(
    ("filename", "node_expression", "keyword", "value_expression"),
    [
        ("demo_lorenz.py", "'LorenzAttractor'", "additive_lod", "stream_ladder"),
        ("demo_mandelbulb.py", "'Mandelbulb'", "additive_lod", "stream_ladder"),
        ("demo_rainbow_sphere.py", "'RainbowSphere'", "additive_lod", "stream_ladder"),
        (
            "demo_exotic_surfaces.py",
            "FAMILY_NAMES[family]",
            "additive_lod",
            "stream_ladder",
        ),
        (
            "demo_galaxy_simulation.py",
            "f'Disc {label}'",
            "additive_lod",
            "stream_ladder",
        ),
        (
            "demo_galaxy_simulation.py",
            "'HII regions'",
            "additive_lod",
            "stream_ladder",
        ),
        (
            "demo_galaxy_simulation.py",
            "'Bulge'",
            "additive_lod",
            "stream_ladder",
        ),
        (
            "demo_ppi_flow_field.py",
            "'Advected protein streamlines'",
            "additive_lod",
            "stream_ladder",
        ),
        (
            "demo_4d_fractals.py",
            "'Fractals4D'",
            "max_elements",
            "TARGET_MAX_POINTS_PER_PLANE",
        ),
    ],
)
def test_gallery_oversized_nodes_bound_individual_commits(
    filename: str,
    node_expression: str,
    keyword: str,
    value_expression: str,
) -> None:
    """Every known oversized gallery node must bound one viewer commit."""
    source = (Path(__file__).parents[1] / filename).read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"add_points", "add_lines", "add_partition_group"}
        and node.args
        and ast.unparse(node.args[0]) == node_expression
    ]
    assert len(calls) == 1, f"expected one {node_expression} adder in {filename}"

    values = {item.arg: item.value for item in calls[0].keywords if item.arg}
    assert keyword in values, f"{filename} {node_expression} has no {keyword}="
    assert value_expression in ast.unparse(values[keyword])
