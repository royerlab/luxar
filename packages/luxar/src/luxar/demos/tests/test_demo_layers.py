"""Static guard: every demo that authors geometry must expose it as a layer.

The viewer's Layers panel lists exactly those scene nodes whose zarr attrs
carry ``layer: true``, and the panel is the ONLY way to toggle a node's
visibility, re-window its display range, change its gamma or switch its
blending mode at view time. ``Node.layer`` defaults to ``False``, so a demo
that never passes ``layer=True`` ships a scene whose panel is inert — often
for the demo's one and only geometry node. That is invisible in Python, in the
zarr store and in a screenshot; only a person opening the panel notices. Hence
a lint rather than a runtime check (see #1362).

Two invariants, weak and strong:

``test_module_exposes_at_least_one_layer``
    Every scanned module that authors geometry marks at least one node — a
    geometry node or a container group — as a layer. This is the invariant that
    matters to a user: an inert panel.
``test_no_geometry_adder_omits_the_layer_kwarg``
    Every individual geometry adder passes ``layer=``, except for the calls
    listed in :data:`EXEMPT`. This is what stops the fix from rotting one call
    at a time: without it, adding a second, layerless node to a demo that
    already has one layer would pass unnoticed.

Two deliberate leniencies, both narrow:

* A geometry adder that splats ``**attrs`` cannot have its kwargs read
  statically, so it counts as satisfied — but ONLY if the module spells the
  word ``layer`` somewhere as a dict key or keyword. The real case is
  ``_interop_common.build_interop_scene``, which assembles
  ``dict(..., layer=True)`` and splats it into ``add_gsplats_from_file``; drop
  that ``layer=True`` and the six ``demo_gsplats_interop_*`` demos it serves go
  inert, so "splats something" alone must not be a free pass.
  The leniency does NOT extend to group adders: a composite group layer is a
  deliberate authoring act that should be spelled out, and treating a
  ``**extra``-built group as a layer would let a whole module pass vacuously.
* A module with no geometry adder at all is skipped — it either builds no scene
  (``registry.py``) or delegates scene building to a shared helper (the
  ``demo_gsplats_interop_*`` family → ``_interop_common``), which is scanned in
  its own right.

Fixing a failure is normally one keyword: add ``layer=True`` to the adder call.
When a demo emits MANY sibling nodes — one per tile, per tree, per repeat — do
not mark each one: the panel would show hundreds of rows and be worse than
inert. Put them under a container group marked ``layer=True`` instead. The
group is a first-class composite: the panel lists it
(``ui/layers/layer-state.ts::walkSceneGraph``) and fans every control it owns
down to the data leaves below it (``ui/layers/layer-apply.ts`` →
``data/attrs-composer.ts::collectDataDescendants``). Then list the sibling call
in :data:`EXEMPT`, naming the group that covers it — the exemption is checked
against that group, so it cannot outlive it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import pytest

from ._scanned_modules import scanned_demo_modules

#: Methods that write a geometry node (Points / Lines / GSplats / Mesh).
#: ``add_text`` / ``add_html`` / ``add_image`` are screen-space overlays, not
#: scene geometry, and are not layers.
GEOMETRY_ADDERS = frozenset(
    {
        "add_points",
        "add_lines",
        "add_gsplats",
        "add_mesh",
        "add_gsplats_from_file",
        "add_gsplats_from_data",
        "add_gsplats_from_volume",
    }
)

#: Methods that write a container node. A group marked ``layer=True`` is a
#: composite layer, so it satisfies the weak invariant for everything below it.
GROUP_ADDERS = frozenset({"add_group", "add_partition_group", "add_lod_group"})


class Exemption(NamedTuple):
    """Geometry adders that intentionally omit ``layer=``.

    Args:
        nodes: The exempt calls, identified by the source text of their node
            name (``ast.unparse`` of the first positional argument or the
            ``name=`` keyword) — never a line number, which every edit above
            the call would invalidate. Listing them individually (rather than
            exempting a whole module) keeps the per-call rule live for whatever
            the demo grows next.
        groups: Source text of the node name of every container group that
            carries the exempt nodes. Each one is asserted to be added with
            ``layer=True``, so the exemption cannot survive the group it
            leans on losing its flag.
        reason: Why the calls are covered without their own ``layer=``.
    """

    nodes: frozenset[str]
    groups: frozenset[str]
    reason: str


#: Keyed on module filename. Each entry names the composite group layer that
#: already covers the exempt nodes — never "this demo has no layer".
EXEMPT: dict[str, Exemption] = {
    "demo_nd_transforms.py": Exemption(
        frozenset(
            {
                "name",  # _add_static_text: labels, notes, channel names
                "'Ruler_Baseline'",
                "'Ruler_Numerals'",
                "'Frame_Cursor_Column'",
                "'Frame_Cursor_Readout'",
                "f'Rail_{name}'",
                "f'Ghosts_{name}'",
                "f'Markers_{name}'",
                "'Channel_Cursor'",
                "f'Chan_Rail_{name}'",
                "f'Chan_Ghosts_{name}'",
                "f'Chan_Marker_{name}_{CH_LETTER[c]}'",
            }
        ),
        frozenset({"'Frame_Section'", "'Channel_Section'"}),
        "The bench's rulers, cursors, rails, ghosts and markers are the two "
        "halves of one measuring instrument; both hang off the `Frame_Section` "
        "/ `Channel_Section` composite group layers.",
    ),
    "demo_biodiversity_planetary_scale.py": Exemption(
        frozenset({"f'part_{i}'"}),
        frozenset({"name"}),  # add_partition_group(name, ..., layer=True)
        "BSP tiles of a kind=partition wrapper that is itself layer=True — the "
        "wrapper is where the compositing attrs live (see the comment there).",
    ),
    "demo_lsystem_forest.py": Exemption(
        frozenset({"f'tree_{i:04d}'"}),
        frozenset({"'trees'"}),
        "Hundreds of per-tree Lines nodes under the layer=True `trees` group.",
    ),
    "demo_gsplats_lod_embryo_line.py": Exemption(
        frozenset({"'lod'"}),
        frozenset({"'embryo_line'"}),
        "100 copies of one ladder under the layer=True `embryo_line` group.",
    ),
}


def _module_ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


MODULES = scanned_demo_modules()


def _node_name(call: ast.Call) -> str:
    """Source text of the node name a geometry adder was given."""
    if call.args:
        return ast.unparse(call.args[0])
    for keyword in call.keywords:
        if keyword.arg == "name":
            return ast.unparse(keyword.value)
    return "<unnamed>"


def _calls(tree: ast.AST, methods: frozenset[str]) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in methods
    ]


def _splats(call: ast.Call) -> bool:
    """True if the call forwards ``**something`` (so ``layer`` may be in it)."""
    return any(keyword.arg is None for keyword in call.keywords)


def _mentions_layer_key(tree: ast.AST) -> bool:
    """True if the module builds a ``layer`` entry it could splat into a call.

    A ``**attrs`` splat hides the kwargs from a static reader, but only a module
    that writes ``layer`` SOMEWHERE can be splatting it. Without this the
    leniency degenerates into "splats anything ⇒ exempt", and dropping
    ``layer=True`` from ``_interop_common.build_interop_scene`` — which takes
    six demos' panels down with it — would go unnoticed.

    A string value does not count: two interop demos carry ``"layer": "<node
    name>"`` in their per-scene spec dicts, which is a name, not the flag.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "layer":
            if _is_flag_value(node.value):
                return True  # dict(..., layer=True)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "layer"
                    and _is_flag_value(value)
                ):
                    return True  # {"layer": True}
        elif isinstance(node, ast.Assign) and _assigns_layer_key(node):
            return True  # attrs["layer"] = True
    return False


def _assigns_layer_key(node: ast.Assign) -> bool:
    """True for ``attrs["layer"] = <flag>`` (any target of the assignment)."""
    return any(
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "layer"
        and _is_flag_value(node.value)
        for target in node.targets
    )


def _is_flag_value(value: ast.expr) -> bool:
    """True unless ``value`` is a literal that cannot ENABLE the ``layer`` flag.

    A non-literal (a variable, a conditional) is unjudgeable and counts; a
    literal has to be ``True``, so neither ``layer="some name"`` nor an explicit
    ``layer=False`` opt-out is mistaken for a splattable flag.
    """
    if isinstance(value, ast.Constant):
        return value.value is True
    return True


def _declares_layer(call: ast.Call) -> bool:
    return any(keyword.arg == "layer" for keyword in call.keywords)


def _enables_layer(call: ast.Call) -> bool:
    """True unless the call spells out a literal ``layer=False``.

    The per-call rule only asks that the author DECIDED (``layer=`` present);
    the weak rule asks that something is actually exposed, so an explicit
    opt-out must not count towards it.
    """
    for keyword in call.keywords:
        if keyword.arg != "layer":
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and not value.value:
            return False
        return True
    return False


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", MODULES, ids=_module_ids(MODULES))
def test_module_exposes_at_least_one_layer(path: Path) -> None:
    """A demo whose geometry is never a layer ships an inert Layers panel."""
    tree = _parse(path)
    geometry = _calls(tree, GEOMETRY_ADDERS)
    if not geometry:
        pytest.skip(f"{path.name} authors no geometry")

    # A splatted geometry adder may well carry `layer` inside the dict — but
    # only if the module writes one at all; group adders get no such benefit of
    # the doubt.
    splat_ok = _mentions_layer_key(tree)
    exposed = [
        call
        for call in geometry
        if _enables_layer(call) or (splat_ok and _splats(call))
    ]
    exposed += [call for call in _calls(tree, GROUP_ADDERS) if _enables_layer(call)]

    assert exposed, (
        f"{path.name} authors {len(geometry)} geometry node(s) but marks none "
        "of them (and no container group) as a Layers-panel layer, so the "
        "panel cannot toggle, range, gamma or re-blend anything in the scene. "
        "Pass `layer=True` to the geometry adder, or — when the demo emits "
        "many sibling nodes — put them under a group added with `layer=True`."
    )


@pytest.mark.parametrize("path", MODULES, ids=_module_ids(MODULES))
def test_no_geometry_adder_omits_the_layer_kwarg(path: Path) -> None:
    """Per-call form of the invariant, so the fix cannot rot one node at a time."""
    exemption = EXEMPT.get(path.name)
    exempt_nodes = exemption.nodes if exemption is not None else frozenset()

    tree = _parse(path)
    splat_ok = _mentions_layer_key(tree)
    missing = [
        f"{_node_name(call)} (line {call.lineno})"
        for call in _calls(tree, GEOMETRY_ADDERS)
        if not _declares_layer(call)
        and not (splat_ok and _splats(call))
        and _node_name(call) not in exempt_nodes
    ]

    assert not missing, (
        f"{path.name} adds geometry without a `layer=` kwarg: "
        f"{', '.join(missing)}. Add `layer=True` so the node reaches the "
        "Layers panel; if it is one of many siblings covered by a "
        "`layer=True` container group, add it to EXEMPT in this module with "
        "the group that covers it."
    )


def _exempt_module(name: str) -> Path:
    """The scanned module an EXEMPT key refers to."""
    for path in MODULES:
        if path.name == name:
            return path
    raise AssertionError(f"EXEMPT names {name}, which is not a scanned module")


class TestTheExemptionsThemselves:
    """An exemption that no longer describes reality is a silent coverage hole."""

    def test_every_exempt_module_exists(self) -> None:
        names = {path.name for path in MODULES}
        assert set(EXEMPT) <= names, (
            f"EXEMPT names modules that are not scanned: {sorted(set(EXEMPT) - names)}"
        )

    def test_every_exemption_is_still_needed(self) -> None:
        """Delete the entry once the call it covers grows a ``layer=`` kwarg."""
        stale = []
        for name, exemption in EXEMPT.items():
            tree = _parse(_exempt_module(name))
            splat_ok = _mentions_layer_key(tree)
            unmarked = {
                _node_name(call)
                for call in _calls(tree, GEOMETRY_ADDERS)
                if not _declares_layer(call) and not (splat_ok and _splats(call))
            }
            for node in sorted(exemption.nodes - unmarked):
                stale.append(f"{name}::{node}")
        assert not stale, f"EXEMPT entries no longer describe any call: {stale}"

    def test_every_covering_group_is_still_a_layer(self) -> None:
        """The group an exemption leans on must still be added with ``layer=True``.

        Without this the exemption is self-certifying: deleting ``layer=True``
        from ``demo_lsystem_forest``'s ``trees`` group takes every tree node out
        of the panel — the exact regression #1362 fixed — while the per-call rule
        stays happy because the trees are exempt.
        """
        for name, exemption in EXEMPT.items():
            tree = _parse(_exempt_module(name))
            layered = {
                _node_name(call)
                for call in _calls(tree, GROUP_ADDERS)
                if _enables_layer(call)
            }
            missing = sorted(exemption.groups - layered)
            assert not missing, (
                f"{name}: the group(s) {missing} that EXEMPT says cover its "
                "unmarked nodes are no longer added with `layer=True`, so those "
                "nodes reach no Layers-panel row at all."
            )


#: Demos whose geometry reaches the panel only through a container group, and
#: that are cheap enough to actually BUILD here (no download, no GPU). The
#: static rules above check that the group is added with ``layer=True``; only
#: writing the store proves the nodes are still PARENTED to it — re-pointing
#: ``_add_frame_ruler(frame_section, …)`` back at ``scene`` would empty the
#: layer while every static check stayed green.
BUILDABLE_COMPOSITE_DEMOS = ("demo_nd_transforms.py", "demo_lsystem_forest.py")


def _build_composite_scene(module_name: str, output: Path) -> None:
    if module_name == "demo_nd_transforms.py":
        from .. import demo_nd_transforms

        demo_nd_transforms.generate_demo(output)
    else:
        from .. import demo_lsystem_forest

        # Smallest forest that still exercises the group: a couple of trees
        # with enough expansion to survive the degenerate-rules skip.
        demo_lsystem_forest.generate_forest(output, iterations=2, n_trees=4)


@pytest.mark.parametrize("module_name", BUILDABLE_COMPOSITE_DEMOS)
def test_written_scene_puts_every_geometry_node_under_a_layer(
    module_name: str, tmp_path: Path
) -> None:
    """Every geometry node in the written store has a ``layer: true`` ancestor."""
    import zarr

    output = tmp_path / "scene.luxar.zarr"
    _build_composite_scene(module_name, output)

    geometry_types = {"points", "lines", "gsplats", "mesh"}
    orphans: list[str] = []

    def walk(group: object, path: str, covered: bool) -> None:
        for name in sorted(group.group_keys()):  # type: ignore[attr-defined]
            child = group[name]  # type: ignore[index]
            child_path = f"{path}/{name}"
            child_covered = covered or bool(child.attrs.get("layer"))
            if child.attrs.get("type") in geometry_types and not child_covered:
                orphans.append(child_path)
            walk(child, child_path, child_covered)

    walk(zarr.open_group(output, mode="r"), "", False)

    assert not orphans, (
        f"{module_name} writes geometry with no `layer: true` node above it: "
        f"{orphans}. Those nodes cannot be toggled, ranged, gamma'd or "
        "re-blended from the Layers panel."
    )
