"""Overlapping gsplat layers must not rely on an INFERRED draw order.

The viewer gives each depth-sorted gsplat node one global back-to-front order
slot. When a demo stacks several gsplat *layers* over the same specimen, that
slot has to come from somewhere, and by default it is *inferred* — from mean
view-z, and then from bounding-sphere containment (a container is forced to
draw before its contents). Both are properties of where the splats happened to
land, not of what the author meant, and the containment relation between two
co-located channels can rest on a sub-percent radius difference.

So an overlapping-layer demo has exactly two sound options, and this test pins
that it takes one of them:

1. **Commutative blending** — ``additive`` (or ``max`` / ``luminous``), where
   the order genuinely does not matter, so there is nothing to get wrong.
2. **A depth-sorted mode WITH an explicit ``depth_level`` on every layer** —
   the order is then stated by the author and camera-independent
   (``docs/guides/specs/LAYER_DEPTH_LEVEL_SPEC.md``).

What is refused is the third case: ``normal`` / ``volumetric`` with no level,
which silently accepts whatever order the geometry implies.

This SUPERSEDES the earlier rule (#1964), which allowed only option 1 because
option 2 did not exist yet. Relaxing it is deliberate — the old rule would now
forbid the correct thing — but it is relaxed in one specific direction and
tightened in another: a depth-sorted layer must carry a level, which the old
rule had no way to require.

The demos listed here were audited against their built scenes: each authors two
or more co-visible gsplat layers whose bounding boxes overlap. The test reads
the *source*, not a built scene, so it needs neither the demo data nor a fit.

Demos deliberately NOT listed, and why they are not violations:

- ``demo_gsplats_3d_decimation_study`` -- four variants placed side by side; the
  bounding boxes are disjoint, so nothing overlaps.
- ``demo_gsplats_lod_embryo_line`` -- one node per embryo, placed by disjoint
  transforms.
- ``demo_gsplats_recipes_tribolium`` -- one node per recipe, placed by disjoint
  transforms.
- ``demo_gsplats_4d_cell_tracking_challenge`` -- one node per crop, placed by
  disjoint transforms.
- ``demo_storm_3d_microtubules`` -- two nodes separated on a ``view`` axis, so
  only one is co-visible.
- ``demo_gsplats_4d_zebrafish_timelapse`` -- one node per timepoint, separated
  on the time axis, so only one is ever co-visible.
- the ``gsplats_interop_*`` demos -- ``kind=partition`` parts and ``kind=lod``
  levels of a SINGLE node, not independent layers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_DEMOS = Path(__file__).resolve().parents[1]

# Modes the viewer must depth-sort (viewer: needsDepthSort).
DEPTH_SORTED_MODES = ("normal", "volumetric")

# Commutative modes, where cross-layer order cannot be observed.
COMMUTATIVE_MODES = ("additive", "max", "luminous")

# Demos authoring >=2 co-visible, spatially overlapping gsplat layers.
OVERLAPPING_LAYER_DEMOS = (
    "demo_gsplats_2d_cmu1_pathology.py",
    "demo_gsplats_2d_codex_pancreas.py",
    "demo_gsplats_3d_acto3d_heart.py",
    "demo_gsplats_3d_cells3d_multichannel.py",
    "demo_gsplats_3d_ct_totalsegmentator.py",
    "demo_gsplats_3d_kidney_multichannel_layers.py",
    "demo_gsplats_3d_kidney_multichannel_toggles.py",
    "demo_gsplats_3d_opencell_map4.py",
    "demo_gsplats_3d_blastocyst_multichannel.py",
    "demo_gsplats_4d_neuromast_2ch.py",
)

_ADD_GSPLATS = {
    "add_gsplats",
    "add_gsplats_from_data",
    "add_gsplats_from_file",
    "add_gsplats_from_volume",
}


def _add_gsplats_calls(source: Path) -> list[ast.Call]:
    tree = ast.parse(source.read_text())
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _ADD_GSPLATS:
            calls.append(node)
    return calls


def _authored_modes(source: Path) -> list[str]:
    """Every literal ``blending_mode`` passed to an ``add_gsplats*`` call."""
    modes: list[str] = []
    for call in _add_gsplats_calls(source):
        for kw in call.keywords:
            if kw.arg == "blending_mode" and isinstance(kw.value, ast.Constant):
                modes.append(kw.value.value)
    return modes


def _calls_passing_depth_level(source: Path) -> int:
    """How many ``add_gsplats*`` calls pass a ``depth_level`` at all.

    Counts the KEYWORD, not a literal: every audited demo drives its channels
    from a config list, so the value is nearly always an expression
    (``ch["depth_level"]``) rather than a constant. Requiring a literal would
    fail every real caller.
    """
    return sum(
        1
        for call in _add_gsplats_calls(source)
        if any(kw.arg == "depth_level" for kw in call.keywords)
    )


@pytest.mark.parametrize("demo", OVERLAPPING_LAYER_DEMOS)
def test_overlapping_layers_declare_their_order(demo: str) -> None:
    """Either commutative blending, or a depth-sorted mode with a stated order."""
    path = _DEMOS / demo
    assert path.exists(), f"audited demo went missing: {demo}"
    modes = _authored_modes(path)
    assert modes, f"{demo}: expected an authored blending_mode to pin"

    depth_sorted = [m for m in modes if m in DEPTH_SORTED_MODES]
    if not depth_sorted:
        # Option 1: every layer commutative. Pin the positive choice too, so an
        # omitted mode is not silently inherited from an ancestor.
        assert set(modes) <= set(COMMUTATIVE_MODES), (
            f"{demo}: authored modes {sorted(set(modes))} — an overlapping-layer "
            "demo must be entirely commutative, or depth-sorted with an explicit "
            "depth_level on every layer"
        )
        return

    # Option 2: depth-sorted, so EVERY add_gsplats* call must state a level.
    # Fewer levels than calls means at least one layer's order is still inferred,
    # which is the case this whole rule exists to prevent.
    calls = len(_add_gsplats_calls(path))
    with_level = _calls_passing_depth_level(path)
    assert with_level == calls, (
        f"{demo} authors depth-sorted blending {sorted(set(depth_sorted))} on "
        f"overlapping gsplat layers, but only {with_level} of {calls} "
        "add_gsplats* calls pass depth_level. A depth-sorted overlapping layer "
        "without a stated level takes whatever order the geometry implies — see "
        "docs/guides/specs/LAYER_DEPTH_LEVEL_SPEC.md"
    )


@pytest.mark.parametrize(
    "demo",
    ("demo_gsplats_3d_acto3d_heart.py", "demo_gsplats_4d_neuromast_2ch.py"),
)
def test_the_demonstrator_demos_state_a_level(demo: str) -> None:
    """The two demos updated to demonstrate the feature must keep stating one.

    They ship ``additive``, so option 1 above would let them drop the levels and
    still pass. That would quietly undo the point: the levels are what make them
    composite correctly when a viewer switches a layer to volumetric by hand,
    which is the exact thing that prompted the feature.
    """
    path = _DEMOS / demo
    calls = len(_add_gsplats_calls(path))
    assert calls, f"{demo}: no add_gsplats* call found"
    assert _calls_passing_depth_level(path) == calls, (
        f"{demo}: every add_gsplats* call must pass depth_level "
        f"({_calls_passing_depth_level(path)} of {calls} do)"
    )
