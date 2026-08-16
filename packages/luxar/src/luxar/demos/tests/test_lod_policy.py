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

import numpy as np
import pytest

from luxar.demos import registry
from luxar.demos._lod_policy import (
    _RECIPE_DEFAULTS,
    TREE_RECIPES,
    DemoRecipe,
    save_with_lod,
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
    return {
        p.name: p.read_text() for p in sorted(registry._DEMOS_DIR.glob("demo_*.py"))
    }


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
        name for name, src in _fitting_demos().items() if "save_with_lod" not in src
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
    routed = {n for n, s in fitting.items() if "save_with_lod" in s}
    assert len(routed) >= 5, f"only {len(routed)} demos routed through the policy"
    assert any(_policy_recipes(fitting[n]) for n in routed), (
        "no recipe literal parsed out — the recipe assertions are vacuous"
    )


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
        if not (set(_policy_recipes(src)) & TREE_RECIPES):
            continue
        for loader in _FLAT_LOADERS:
            assert loader not in src, (
                f"{name} writes a {sorted(set(_policy_recipes(src)) & TREE_RECIPES)} "
                f"artifact but reads it back through {loader}, which cannot open a "
                "multi-part store. Fetch the PATH (ensure_dataset) and graft it with "
                "Group.add_gsplats_from_file()."
            )


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

    ``save_with_lod`` caps tiles at 250,000 splats, so a test-sized fit always
    comes out as ONE part — which would leave the interesting case (a partition
    with many children, each its own level ladder) untested. Build that shape
    directly and check both that it grafts and that the per-channel colormap
    reaches the levels, since ``colormap`` is copied down to the leaves while
    ``blending_mode`` stays on the wrapper.
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
