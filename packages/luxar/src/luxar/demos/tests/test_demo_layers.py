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

* A geometry adder that splats ``**attrs`` does not show its kwargs to a
  static reader, so it counts as satisfied — but ONLY when THAT mapping can be
  shown to carry the flag: a dict literal or ``dict(...)`` written inline, or a
  name bound to one where the call can actually see it — the innermost scope
  that binds the name, above the call (including a later
  ``attrs["layer"] = True``, and honouring a rebinding in between). The
  real case is ``_interop_common.build_interop_scene``, which assembles
  ``dict(..., layer=True)`` and splats it into ``add_gsplats_from_file``; drop
  that ``layer=True`` and the six ``demo_gsplats_interop_*`` demos it serves go
  inert, so "splats something" must not be a free pass.
  The leniency is deliberately per-mapping rather than per-module: a module
  that layers one node correctly must not thereby excuse a second call that
  splats an unrelated ``**style`` dict. A splat this cannot resolve — a
  function parameter, a call result — gets no benefit of the doubt either;
  spell ``layer=`` out, or take an :data:`EXEMPT` entry naming the group that
  covers the node.
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


def _splat_carries_layer(tree: ast.AST, call: ast.Call) -> bool:
    """True if a ``**`` argument of ``call`` can be shown to carry ``layer``.

    A splat hides the kwargs from a static reader, so the mapping ITSELF has to
    be resolvable to something that sets the flag — a dict written inline, or a
    name whose binding the call can actually see (same scope or outwards, and
    written above it). Anything else (a parameter, a call result, a
    ``**a or b``) is not lenient: were it, the rule would degenerate into
    "splats anything ⇒ exempt", and dropping ``layer=True`` from
    ``_interop_common.build_interop_scene`` — which takes six demos' panels down
    with it — would go unnoticed.
    """
    return any(
        keyword.arg is None and _mapping_sets_layer(tree, keyword.value, call)
        for keyword in call.keywords
    )


def _mapping_sets_layer(tree: ast.AST, value: ast.expr, call: ast.Call) -> bool:
    """True if the splatted expression is a mapping that sets ``layer``."""
    if isinstance(value, ast.Name):
        return _binding_sets_layer(tree, value.id, call)
    return _mapping_literal_sets_layer(value)


def _mapping_literal_sets_layer(value: ast.expr) -> bool:
    """True for ``{"layer": <flag>}`` / ``dict(layer=<flag>)`` written inline."""
    if isinstance(value, ast.Dict):
        return any(
            isinstance(key, ast.Constant)
            and key.value == "layer"
            and _is_flag_value(item)
            for key, item in zip(value.keys, value.values)
        )
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "dict"
    ):
        return any(
            keyword.arg == "layer" and _is_flag_value(keyword.value)
            for keyword in value.keywords
        )
    return False


def _binding_sets_layer(tree: ast.AST, name: str, call: ast.Call) -> bool:
    """True if the binding of ``name`` that ``call`` sees sets ``layer``.

    Resolution follows Python's own rules rather than "assigned somewhere in
    the file": the innermost scope that binds the name wins, and only
    statements ABOVE the call count. Without that, a ``**attrs`` splat of a
    function PARAMETER would be excused by an unrelated ``attrs = dict(
    layer=True)`` in a different function, and an assignment BELOW the call
    would validate it — the very benefit of the doubt the module docstring says
    an unresolvable splat does not get.

    Both spellings the demos use are recognised: the mapping literal itself
    (``attrs = dict(..., layer=True)``, annotated or not) and a later
    ``attrs["layer"] = True``. Rebinding is honoured in source order, so a
    mapping that sets the flag and is then replaced by something unresolvable
    is not lenient either.
    """
    for scope in _scope_chain(tree, call):
        rebindings, flag_writes = _bindings_of(scope, name)
        if not rebindings and not flag_writes and not _is_parameter(scope, name):
            continue  # the name is not bound here; look further out
        # The name resolves in THIS scope, so judge it here and do not fall
        # outwards: an outer binding is shadowed.
        sets_layer = False
        # (lineno, then rebinding before flag-write) so `attrs = {}` followed by
        # `attrs["layer"] = True` on ONE line still replays in the right order.
        ordered = sorted(rebindings + flag_writes, key=lambda b: (b[0], b[1]))
        for lineno, is_flag_write, value in ordered:
            if lineno >= call.lineno:
                break
            sets_layer = (
                _is_flag_value(value)
                if is_flag_write
                else _mapping_literal_sets_layer(value)
            )
        return sets_layer
    return False


#: ``(lineno, is_a_layer_flag_write, assigned_value)`` per statement that binds
#: or mutates the name, so the two kinds can be replayed in source order.
_Binding = tuple[int, bool, ast.expr]


def _bindings_of(scope: ast.AST, name: str) -> tuple[list[_Binding], list[_Binding]]:
    """Rebindings of ``name`` and writes to its ``"layer"`` key, in ``scope``."""
    rebindings: list[_Binding] = []
    flag_writes: list[_Binding] = []
    for node in _walk_scope(scope):
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        if node.value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                rebindings.append((node.lineno, False, node.value))
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "layer"
            ):
                flag_writes.append((node.lineno, True, node.value))
    return rebindings, flag_writes


def _scope_chain(tree: ast.AST, call: ast.Call) -> list[ast.AST]:
    """Scopes enclosing ``call``, innermost first, ending at the module.

    Nested ``def``s open later than the ``def``s they sit in, so ordering the
    enclosing functions by descending line number orders them innermost-first.
    """
    enclosing = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(inner is call for inner in ast.walk(node))
    ]
    enclosing.sort(key=lambda node: node.lineno, reverse=True)
    return [*enclosing, tree]


def _walk_scope(scope: ast.AST) -> list[ast.AST]:
    """Every node of ``scope``'s own body, without descending into inner scopes."""
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    found: list[ast.AST] = []
    stack: list[ast.AST] = list(getattr(scope, "body", []))
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, nested):
            stack.extend(ast.iter_child_nodes(node))
    return found


def _is_parameter(scope: ast.AST, name: str) -> bool:
    """True if ``name`` is a parameter of ``scope`` — bound, but to nothing legible."""
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    args = scope.args
    named = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg is not None:
        named.append(args.vararg)
    if args.kwarg is not None:
        named.append(args.kwarg)
    return any(arg.arg == name for arg in named)


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
    # only if THAT dict is resolvable and sets it; group adders get no such
    # benefit of the doubt.
    exposed = [
        call
        for call in geometry
        if _enables_layer(call) or _splat_carries_layer(tree, call)
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
    missing = [
        f"{_node_name(call)} (line {call.lineno})"
        for call in _calls(tree, GEOMETRY_ADDERS)
        if not _declares_layer(call)
        and not _splat_carries_layer(tree, call)
        and _node_name(call) not in exempt_nodes
    ]

    assert not missing, (
        f"{path.name} adds geometry without a `layer=` kwarg: "
        f"{', '.join(missing)}. Add `layer=True` so the node reaches the "
        "Layers panel; if it is one of many siblings covered by a "
        "`layer=True` container group, add it to EXEMPT in this module with "
        "the group that covers it."
    )


#: ``(source, is_lenient)`` cases for the ``**mapping`` benefit of the doubt.
#: Each source's LAST geometry adder is the call under test.
SPLAT_CASES = [
    pytest.param(
        'attrs = {"layer": True}\nscene.add_points("a", **attrs)',
        True,
        id="dict-literal",
    ),
    pytest.param(
        'attrs = dict(layer=True)\nscene.add_points("a", **attrs)', True, id="dict-call"
    ),
    pytest.param(
        'attrs: dict[str, object] = dict(layer=True)\nscene.add_points("a", **attrs)',
        True,
        id="annotated-binding",
    ),
    pytest.param(
        'attrs = {}\nattrs["layer"] = True\nscene.add_points("a", **attrs)',
        True,
        id="subscript-assign",
    ),
    pytest.param('scene.add_points("a", **{"layer": True})', True, id="inline-splat"),
    pytest.param(
        'scene.add_points("a", layer=True)\nstyle = {"opacity": 0.5}\nscene.add_points("b", **style)',
        False,
        id="unrelated-mapping-in-a-layered-module",
    ),
    pytest.param(
        'attrs = {"layer": "some_node"}\nscene.add_points("a", **attrs)',
        False,
        id="layer-is-a-name",
    ),
    pytest.param(
        'attrs = {"layer": False}\nscene.add_points("a", **attrs)',
        False,
        id="explicit-opt-out",
    ),
    pytest.param(
        'scene.add_points("a", **build_attrs())', False, id="unresolvable-call"
    ),
    pytest.param(
        # The `_interop_common` shape: bound and splatted in one function.
        "def build(path):\n"
        "    attrs = dict(layer=True)\n"
        '    scene.add_gsplats_from_file(name="a", path=path, **attrs)',
        True,
        id="binding-in-the-calls-own-scope",
    ),
    pytest.param(
        # A module-level mapping is in scope inside the function.
        'ATTRS = dict(layer=True)\ndef build():\n    scene.add_points("a", **ATTRS)',
        True,
        id="module-level-binding-seen-from-a-function",
    ),
    pytest.param(
        # `attrs` is a PARAMETER here; another function's local of the same
        # name says nothing about it.
        "def build(attrs):\n"
        '    scene.add_points("a", **attrs)\n'
        "def other():\n"
        "    attrs = dict(layer=True)",
        False,
        id="parameter-shadows-another-scopes-binding",
    ),
    pytest.param(
        'scene.add_points("a", **attrs)\nattrs = dict(layer=True)',
        False,
        id="binding-below-the-call",
    ),
    pytest.param(
        'attrs = dict(layer=True)\nattrs = load_attrs()\nscene.add_points("a", **attrs)',
        False,
        id="rebound-to-something-unresolvable",
    ),
]


@pytest.mark.parametrize(("source", "is_lenient"), SPLAT_CASES)
def test_the_splat_leniency_is_per_mapping_not_per_module(
    source: str, is_lenient: bool
) -> None:
    """A layered call elsewhere must not excuse a splat that cannot carry ``layer``.

    The whole point of the per-call rule is that it cannot rot one node at a
    time; a module-wide "mentions layer somewhere" test would hand every later
    ``**kwargs`` call a free pass.
    """
    tree = ast.parse(source)
    call = _calls(tree, GEOMETRY_ADDERS)[-1]
    assert _splat_carries_layer(tree, call) is is_lenient


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
            unmarked = {
                _node_name(call)
                for call in _calls(tree, GEOMETRY_ADDERS)
                if not _declares_layer(call) and not _splat_carries_layer(tree, call)
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
