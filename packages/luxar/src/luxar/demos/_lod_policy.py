"""The LOD topology each cached demo artifact ships with.

A ladder is not something a fit acquires by default. ``GSplatData`` carries
``additive_sublods``, and the progressive fitter happens to produce several as a
by-product of its passes while ``fit_gaussian_splats`` produces exactly one — so
which demos shipped a streaming ladder was decided by which fitter they called,
not by what their data needed. Five of the twenty-one shipped archives had no
ladder at all and none had a substitutive level, for that reason alone.

This module makes the choice explicit and reviewable, and gives the one call
demos need between fitting and saving. The recipe a dataset gets is a property
of the dataset, so it is recorded here rather than buried in each demo:

``stream``
    One leaf plus a progressive prefix ladder. Fast first paint; the right
    answer whenever the whole object is loaded anyway.
``levels``
    Coarse→fine replacement levels *and* a ladder per level. Worth its ~40%
    extra bytes on a large single object that is orbited, because the finest
    level still shows whenever the object fills half the screen or more.
``adaptive``
    Spatial tiles, each choosing its own level. For data panned and zoomed
    rather than orbited, where most tiles are off-screen most of the time.

Measured cost of ``levels`` over ``stream``, on ct_atlas (660,934 splats):
6,966 KB → 9,635 KB, i.e. +38%. Do not extrapolate that from a small fit — the
same comparison on a 743-splat fit reads 3.6x, which is per-zarr-group overhead
rather than data.

A recipe costlier than ``stream`` is only worth choosing if the demo's SCENE can
carry it, and that depends on which adder the demo builds with — see
:data:`TOPOLOGY_PRESERVING_ADDERS`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from arbol import aprint

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData

#: Recipes usable from a demo. A subset of the full recipe set on purpose: the
#: rest need knobs (a density, a source volume, a tiled fit) that a demo's cache
#: step does not have in hand.
DemoRecipe = Literal["stream", "levels", "adaptive"]

#: Recipes whose artifact is a node TREE with no flat matrix form.
#:
#: This is not a detail of the writer — it decides how the demo READS its own
#: cache back. ``GSplatData.load`` (and therefore
#: :func:`luxar.utils.data_fetch.load_dataset_gsplats`) refuses a
#: ``kind=partition`` store outright, so a demo choosing one of these must graft
#: the PATH with :meth:`~luxar.core.group.Group.add_gsplats_from_file` instead.
#: Getting this wrong breaks the demo's DEFAULT path while leaving the
#: ``--recompute`` path — the one an author exercises — perfectly green.
TREE_RECIPES: frozenset[str] = frozenset({"adaptive"})

#: Scene adders that carry a stored topology into the scene.
#:
#: ``add_gsplats_from_data`` is handed the whole ``GSplatData``, whose matrix form
#: holds the substitutive levels, and re-emits them as a ``kind=lod`` group;
#: ``add_gsplats_from_file`` grafts the stored subtree node-for-node. The third
#: adder, plain ``add_gsplats(centers=…, amplitudes=…)``, is handed loose arrays
#: — every one of which is a view of the FINEST level only — so it writes a flat
#: leaf. Measured on a 4,000-splat fit: the same archive added via
#: ``add_gsplats`` gives a node with no ``kind`` and no child groups whether it
#: was written ``stream`` or ``levels``, while the ``levels`` archive is 4.1x the
#: bytes on disk (the small-fit figure — +38% on a real one, above).
#:
#: So a demo that rebuilds its scene from arrays gets nothing for the extra bytes
#: and should stay on ``stream``. It is not a rule the choice below can enforce
#: on its own — the archive is written in the fit step and read in the scene step,
#: often hundreds of lines apart — so ``tests/test_lod_policy.py`` gates it.
TOPOLOGY_PRESERVING_ADDERS: frozenset[str] = frozenset(
    {"add_gsplats_from_data", "add_gsplats_from_file"}
)

#: Recipes whose extra bytes only pay off if the topology reaches the scene.
#:
#: ``stream`` is excluded because a prefix ladder is the same splats regrouped —
#: it costs essentially nothing, so it is the right floor even for a demo whose
#: scene flattens it (the archive is still downloadable and ``luxar gsplat view``
#: honours the ladder).
SCENE_TOPOLOGY_RECIPES: frozenset[str] = frozenset({"levels", "adaptive"})

#: Per-recipe parameters, so two demos choosing ``levels`` cannot drift apart.
#:
#: ``n_lods=4`` throughout: four rungs is the shipped default and what every
#: laddered archive already carries, so it keeps regenerated artifacts
#: comparable to the ones they replace.
_RECIPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "stream": {"n_lods": 4},
    "levels": {"n_lods": 4, "compression_factor": 4, "levels": 3},
    # max_elements caps splats per tile; K/L are the per-tile level ladder.
    # L=2 rather than 3 because a tile is already a fraction of the object, so
    # a third level would coarsen past anything a viewer requests.
    "adaptive": {
        "n_lods": 4,
        "max_elements": 250_000,
        "compression_factor": 4,
        "levels": 2,
    },
}


def save_with_lod(
    data: "GSplatData",
    path: Path,
    *,
    recipe: DemoRecipe,
    device: Optional[str] = None,
    quiet: bool = False,
    **save_kwargs: Any,
) -> None:
    """Apply ``recipe`` to ``data`` and write the cached artifact.

    Replaces a bare ``data.save(path, ...)`` in a demo's cache step. Handles
    both shapes a recipe can return: the matrix recipes give back a
    ``GSplatData`` (whose own ``save`` is used, so ``save_kwargs`` pass through
    untouched), while ``adaptive`` gives back a node tree that needs
    ``write_gsplats_tree`` and an explicitly split stats dict.

    The fit's ``stats`` survive the recipe unchanged, which is what keeps the
    foreground PSNR and the source-grid provenance on the artifact — a LOD step
    that dropped them would leave the published compression figure and quality
    numbers with nothing behind them.

    Parameters
    ----------
    data
        The fitted splats.
    path
        Destination, e.g. ``.../name.gsplats.zarr.zip``.
    recipe
        Which topology to build. See the module docstring for why each exists.
    device
        Passed to the substitutive reduction (``levels``/``adaptive`` only).
        ``None`` lets it choose; pass ``"cpu"`` on Apple silicon, where the MPS
        backend lacks the float64 support the reduction wants and warns as it
        falls back anyway.
    quiet
        Suppress the one-line report of what was built.
    **save_kwargs
        Forwarded to the writer: ``encoding_mode``, ``compress``,
        ``zip_deflate``, ``ordering``, ``include_fitting_info``, ``description``.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
    from luxar.gsplats.lod import RecipeParams, build_recipe

    params_kwargs = dict(_RECIPE_DEFAULTS[recipe])
    if device is not None:
        params_kwargs["device"] = device
    result = build_recipe(data, recipe, RecipeParams(**params_kwargs))

    # Build provenance: which recipe made this. Distinct from the `lod_kind` the
    # builder stamps, which is the mechanism the viewer actually reads.
    if isinstance(result, GSplatData):
        result.stats["recipe"] = recipe
        result.save(path, **save_kwargs)
    else:
        fitting_info, fitting_config, provenance_info, pipeline_info = (
            split_fitting_info(
                data.stats,
                include_fitting_info=save_kwargs.pop("include_fitting_info", True),
                include_provenance=save_kwargs.pop("include_provenance", False),
            )
        )
        # `include_*` are consumed above; the rest are writer kwargs.
        write_gsplats_tree(
            path,
            result,
            fitting_info=fitting_info,
            fitting_config=fitting_config,
            provenance_info=provenance_info,
            pipeline_info={**(pipeline_info or {}), "recipe": recipe},
            **save_kwargs,
        )

    if not quiet:
        aprint(f"  LOD topology: {recipe} ({_describe(recipe)})")


def _describe(recipe: str) -> str:
    """One phrase per recipe, for the demo's console output."""
    defaults = _RECIPE_DEFAULTS[recipe]
    if recipe == "stream":
        return f"{defaults['n_lods']}-rung progressive ladder"
    if recipe == "levels":
        return (
            f"{defaults['levels']} coarse levels at 1/{defaults['compression_factor']} "
            f"each, {defaults['n_lods']}-rung ladder per level"
        )
    return (
        f"spatial tiles of <={defaults['max_elements']:,} splats, "
        f"{defaults['levels']} levels per tile"
    )
