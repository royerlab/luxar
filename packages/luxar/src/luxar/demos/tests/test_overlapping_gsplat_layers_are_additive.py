"""Overlapping gsplat layers must composite additively.

Splats are depth-sorted WITHIN a gsplat node, but the viewer does not sort
across sibling nodes. So when a demo stacks several gsplat *layers* over the
same specimen, any depth-sorted mode -- ``needsDepthSort`` in the viewer's
``rendering/blending-state.ts`` is exactly ``normal`` and ``volumetric`` --
composites them in an arbitrary order, and the render is not correct. Additive
(like ``max`` and ``luminous``) is order-independent, so it is the only sound
choice for this class of demo.

The demos listed here were audited against their built scenes: each authors two
or more co-visible gsplat layers whose bounding boxes overlap. The test reads
the *source*, not a built scene, so it needs neither the demo data nor a fit.

Demos deliberately NOT listed, and why they are not violations:

- ``demo_gsplats_3d_decimation_study`` -- four variants placed side by side; the
  bounding boxes are disjoint, so nothing overlaps.
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

# Demos authoring >=2 co-visible, spatially overlapping gsplat layers.
OVERLAPPING_LAYER_DEMOS = (
    "demo_gsplats_3d_acto3d_heart.py",
    "demo_gsplats_3d_cells3d_multichannel.py",
    "demo_gsplats_3d_ct_totalsegmentator.py",
    "demo_gsplats_3d_kidney_multichannel_layers.py",
    "demo_gsplats_3d_kidney_multichannel_toggles.py",
    "demo_gsplats_3d_opencell_map4.py",
    "demo_gsplats_4d_neuromast_2ch.py",
)

_ADD_GSPLATS = {
    "add_gsplats",
    "add_gsplats_from_data",
    "add_gsplats_from_file",
    "add_gsplats_from_volume",
}


def _authored_modes(source: Path) -> list[str]:
    """Every literal ``blending_mode`` passed to an ``add_gsplats*`` call."""
    tree = ast.parse(source.read_text())
    modes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _ADD_GSPLATS:
            continue
        for kw in node.keywords:
            if kw.arg == "blending_mode" and isinstance(kw.value, ast.Constant):
                modes.append(kw.value.value)
    return modes


@pytest.mark.parametrize("demo", OVERLAPPING_LAYER_DEMOS)
def test_no_depth_sorted_mode_on_overlapping_layers(demo: str) -> None:
    path = _DEMOS / demo
    assert path.exists(), f"audited demo went missing: {demo}"
    modes = _authored_modes(path)
    assert modes, f"{demo}: expected an authored blending_mode to pin"
    offenders = [m for m in modes if m in DEPTH_SORTED_MODES]
    assert not offenders, (
        f"{demo} authors depth-sorted blending {offenders} on overlapping gsplat "
        "layers; the viewer does not sort across layers, so use 'additive'"
    )


@pytest.mark.parametrize("demo", OVERLAPPING_LAYER_DEMOS)
def test_every_layer_is_explicitly_additive(demo: str) -> None:
    """Pin the positive choice too, so an omitted mode is not silently inherited."""
    modes = _authored_modes(_DEMOS / demo)
    assert set(modes) == {"additive"}, f"{demo}: authored modes {sorted(set(modes))}"
