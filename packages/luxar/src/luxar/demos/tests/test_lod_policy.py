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
from pathlib import Path

from luxar.demos import registry
from luxar.demos._lod_policy import _RECIPE_DEFAULTS, DemoRecipe

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

#: Fitting demos not yet routed through the policy. SHRINKS to empty.
#:
#: These keep whatever topology their fitter happens to produce — which is the
#: defect the policy exists to remove, not a configuration. Every one of them is
#: cache-only or local-compute, which is why they are not urgent.
_NOT_YET_ROUTED = {
    "demo_gsplats_2d_codex_pancreas.py",
    "demo_gsplats_3d_acto3d_heart.py",
    "demo_gsplats_3d_cells3d_multichannel.py",
    "demo_gsplats_3d_flylight_mcfo_neurons.py",
    "demo_gsplats_3d_kidney_multichannel_layers.py",
    "demo_gsplats_3d_kidney_multichannel_toggles.py",
    "demo_gsplats_3d_opencell_map4.py",
    "demo_gsplats_3d_organoid_dapi_nuclei.py",
    "demo_gsplats_3d_organoid_multichannel.py",
    "demo_gsplats_3d_tng_cosmic_web.py",
    "demo_gsplats_3d_tribolium_embryo.py",
    "demo_gsplats_4d_celegans_tracking.py",
    "demo_gsplats_4d_zebrafish_timelapse.py",
}


def _demo_sources() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(registry._DEMOS_DIR.glob("demo_*.py"))}


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
    """The ``recipe=`` literals passed to ``save_with_lod`` in *src*."""
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


def test_every_fitting_demo_chooses_a_topology_or_is_listed() -> None:
    unrouted = {
        name
        for name, src in _fitting_demos().items()
        if "save_with_lod" not in src
    }
    accounted = set(_NO_CACHED_ARTIFACT) | _NOT_YET_ROUTED
    assert unrouted <= accounted, (
        "a fitting demo writes its cache without choosing an LOD topology:\n  "
        + "\n  ".join(sorted(unrouted - accounted))
        + "\nRoute its save through luxar.demos._lod_policy.save_with_lod, or add "
        "it to _NO_CACHED_ARTIFACT with the reason it ships no artifact."
    )


def test_the_pending_list_shrinks_and_does_not_go_stale() -> None:
    """A demo that has been routed must leave the list."""
    fitting = _fitting_demos()
    for name in sorted(_NOT_YET_ROUTED):
        assert name in fitting, f"{name} no longer fits — drop it from the list"
        assert "save_with_lod" not in fitting[name], (
            f"{name} now chooses a topology — remove it from _NOT_YET_ROUTED"
        )
    for name in sorted(_NO_CACHED_ARTIFACT):
        assert name in fitting, f"{name} no longer fits — drop the exemption"


def test_every_chosen_recipe_is_one_the_policy_defines() -> None:
    """A typo'd recipe name would otherwise fail only at refit time."""
    known = set(_RECIPE_DEFAULTS)
    for name, src in sorted(_fitting_demos().items()):
        for recipe in _policy_recipes(src):
            assert recipe in known, (
                f"{name}: unknown recipe {recipe!r}; the policy defines "
                f"{sorted(known)}"
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
    routed = {n for n, s in fitting.items() if "save_with_lod" in s}
    assert len(routed) >= 5, f"only {len(routed)} demos routed through the policy"
    assert any(_policy_recipes(fitting[n]) for n in routed), (
        "no recipe literal parsed out — the recipe assertions are vacuous"
    )


def test_the_demo_recipe_type_and_defaults_agree() -> None:
    """`DemoRecipe` and `_RECIPE_DEFAULTS` must list the same recipes."""
    from typing import get_args

    assert set(get_args(DemoRecipe)) == set(_RECIPE_DEFAULTS)


def test_source_tree_is_the_one_being_tested() -> None:
    """Guard against the AST scan reading an installed copy instead of the repo."""
    assert (Path(registry._DEMOS_DIR) / "_lod_policy.py").exists()
