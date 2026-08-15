"""An explicit ``None`` render attr means ABSENT at the LEAF adders too (#1574).

#1496 settled the rule one level up, on the three ``**attrs``-forwarding gsplats
doors (``add_gsplats_from_data`` / ``add_gsplats_from_file`` / the graft entry —
see ``lod/test_source_validation.py``'s "#1496" section for the four-route
matrix), and deliberately left the leaf adders alone. Two present-but-``None``
render attrs were therefore still mishandled inside the leaves themselves, on
ALL FOUR geometry types, which is why this suite is parametrized over the adder
table rather than living next to the gsplats-pipeline cases.

Measured on the unfixed leaf adders, uncoloured data, a 3-dimension scene:

  * ``colormap=None`` wrote ``colormap='custom'`` — the silent half.
    ``validate_render_attrs`` guards its colormap check on ``is not None`` so the
    key passes, and ``compositing.sync_custom_colormap_attr`` then rewrites the
    None to ``'custom'`` (a None is not a str in ``BUILTIN_COLORMAP_NAMES``)
    WITHOUT writing any ``colormap_lut``. The viewer's ``build-scene-graph.ts``
    answers a ``'custom'`` with no LUT by warning and falling back to VIRIDIS,
    where omitting the key gives ``gray``. Points/Lines/Mesh all wrote
    ``'custom'`` against an omitted-key control of NO key at all; GSplats wrote
    ``'custom'`` against a control of ``'gray'`` (its writer stamps that default
    only when the key is absent).
  * ``coverage_fraction=None`` persisted a literal ``coverage_fraction: null``
    LOD selector threshold into the node's zarr attrs.

Both assertions are made against what LANDED IN THE STORE, not merely against
the returned node object: the complaint is about the file the viewer reads. And
against the WHOLE STORE on the structural routes: the fix is one statement above
every structural branch precisely because the None otherwise reaches each
``part_i`` / ``child_i`` of a partitioned or LOD node — see
:class:`TestTheStripRunsAboveEveryStructuralBranch`, which is what a flat-only
strip fails and the flat cases above do not.

The last two classes are scope controls rather than regression cases. They pass
before and after the fix by construction — their job is to kill an OVER-broad
strip, which would swallow the Nones that are supposed to be loud, and to pin
that the gsplats pipeline's wider set keeps agreeing, key for key, with the leaf
set it is derived from.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Set, Tuple

import numpy as np
import pytest
import zarr

from luxar.core.group.compositing import ABSENT_WHEN_NONE_RENDER_ATTRS
from luxar.core.group.gsplats_pipeline.from_data import (
    ABSENT_WHEN_NONE_ATTRS,
    GATE_FORWARDED_LEAF_PARAMS,
)

from .conftest import cholesky_rows, grid_mesh, open_scene, random_positions

#: Element count for every leaf built here. Small: none of these cases depends
#: on the data, only on which attr keys survive to disk.
_N = 8

#: Sentinel for "the key is not in the node's attrs at all", so an absent key and
#: a key valued ``None`` can never compare equal (``dict.get`` alone returns None
#: for both, which is precisely the distinction under test).
_ABSENT = "<absent>"


def _add_points(scene: Any, name: str, **attrs: Any) -> Any:
    return scene.add_points(name, positions=random_positions(_N, seed=1574), **attrs)


def _add_lines(scene: Any, name: str, **attrs: Any) -> Any:
    return scene.add_lines(
        name,
        vertices=random_positions(_N, seed=1575),
        widths=np.full(_N, 0.5, dtype=np.float32),
        **attrs,
    )


def _add_mesh(scene: Any, name: str, **attrs: Any) -> Any:
    vertices, faces = grid_mesh(4)
    return scene.add_mesh(name, vertices=vertices, faces=faces, **attrs)


def _add_gsplats(scene: Any, name: str, **attrs: Any) -> Any:
    return scene.add_gsplats(
        name,
        centers=random_positions(_N, seed=1576),
        amplitudes=np.ones(_N, dtype=np.float32),
        cholesky_factors=cholesky_rows(_N),
        **attrs,
    )


#: ``(geometry, adder)`` for all four leaf adders. Every case runs against the
#: whole table on purpose: the bug was measured on all four, and each adder owns
#: its own copy of the attrs handling the fix had to reach.
LEAF_ADDERS = [
    ("points", _add_points),
    ("lines", _add_lines),
    ("mesh", _add_mesh),
    ("gsplats", _add_gsplats),
]


def _write_leaf(
    tmp_path: Any, filename: str, adder: Callable[..., Any], **attrs: Any
) -> Tuple[Dict[str, Any], Set[str]]:
    """Write one leaf, finalize, and return its ON-DISK attrs and array names."""
    compiler, scene, path = open_scene(tmp_path, filename)
    adder(scene, "leaf", **attrs)
    compiler.finalize()
    node = zarr.open_group(path, mode="r")["leaf"]
    return dict(node.attrs), set(node.array_keys())


def _subtree(store: Any, node: str) -> Dict[str, Tuple[Dict[str, Any], Set[str]]]:
    """``{path: (attrs, array names)}`` for ``node`` and every descendant.

    The same whole-tree readback shape as ``lod/test_source_validation.py``'s
    ``_node_attrs``, extended with each node's array names because the LUT-less
    ``'custom'`` this suite hunts is a pair of facts (the attr AND the absent
    ``colormap_lut``) rather than one.
    """
    group = store[node]
    found: Dict[str, Tuple[Dict[str, Any], Set[str]]] = {
        node: (dict(group.attrs), set(group.array_keys()))
    }
    for child in sorted(group.group_keys()):
        found.update(_subtree(store, f"{node}/{child}"))
    return found


def _write_tree(
    tmp_path: Any, filename: str, adder: Callable[..., Any], **attrs: Any
) -> Dict[str, Tuple[Dict[str, Any], Set[str]]]:
    """Write one node, finalize, and read back the WHOLE subtree it produced."""
    compiler, scene, path = open_scene(tmp_path, filename)
    adder(scene, "leaf", **attrs)
    compiler.finalize()
    return _subtree(zarr.open_group(path, mode="r"), "leaf")


class TestAColormapNoneIsAbsentNotCustom:
    @pytest.mark.parametrize("geometry,adder", LEAF_ADDERS)
    def test_it_matches_the_omitted_key_control(
        self, tmp_path: Any, geometry: str, adder: Callable[..., Any]
    ) -> None:
        """``colormap=None`` writes whatever omitting the key writes.

        Stated as parity with the control rather than against a literal, because
        the right answer differs by geometry — no key for Points/Lines/Mesh,
        ``'gray'`` for GSplats — and the point of the fix is exactly that the two
        spellings agree, whatever that shared answer is.
        """
        explicit, _ = _write_leaf(
            tmp_path, f"{geometry}_none.luxar.zarr", adder, colormap=None
        )
        omitted, _ = _write_leaf(tmp_path, f"{geometry}_omit.luxar.zarr", adder)

        assert explicit.get("colormap", _ABSENT) == omitted.get("colormap", _ABSENT)

    @pytest.mark.parametrize("geometry,adder", LEAF_ADDERS)
    def test_no_lut_less_custom_node_is_shipped(
        self, tmp_path: Any, geometry: str, adder: Callable[..., Any]
    ) -> None:
        """The specific state the viewer renders as viridis is never written.

        The parity case above would also be satisfied if BOTH spellings started
        writing a LUT-less ``'custom'``, so the failure mode is pinned directly:
        ``colormap='custom'`` is only meaningful alongside the ``colormap_lut``
        dataset the writer resolves a real custom colormap into.
        """
        attrs, arrays = _write_leaf(
            tmp_path, f"{geometry}_custom.luxar.zarr", adder, colormap=None
        )

        assert attrs.get("colormap") != "custom"
        assert "colormap_lut" not in arrays


class TestACoverageFractionNoneIsNotPersisted:
    @pytest.mark.parametrize("geometry,adder", LEAF_ADDERS)
    def test_the_key_is_absent_from_the_written_attrs(
        self, tmp_path: Any, geometry: str, adder: Callable[..., Any]
    ) -> None:
        """No ``coverage_fraction: null`` selector threshold reaches the store.

        ``in`` rather than ``get(...) is None``: a null VALUE is what was written,
        so only key membership tells the two states apart.
        """
        attrs, _ = _write_leaf(
            tmp_path, f"{geometry}_cf.luxar.zarr", adder, coverage_fraction=None
        )

        assert "coverage_fraction" not in attrs


#: ``(id, adder, structural kwargs)`` — every leaf adder through a STRUCTURAL
#: branch, which is where a leaked None actually ships: a partitioned / LOD scene
#: is the one a user builds because it is too big to eyeball, and the strip's
#: placement ABOVE those branches is the whole reason it is one statement at the
#: top of each ``add_*_impl`` rather than one guard per consumer.
#:
#: Both structures each adder supports, minus the combinations that do not exist:
#: ``partition=`` on all four, plus the Points/Lines ``substitutive_lod=`` ladder
#: (whose coarse levels are GSPLAT nodes, written by a different adder than the
#: finest child, so the two halves of one tree can disagree about what the same
#: ``None`` meant). Mesh's ``substitutive_lod=`` and the additive ladders are left to the
#: two structures above: they reach the same ``**attrs`` dict past the same single
#: strip, and each extra route costs a real zarr write.
#:
#: Sizes are the smallest that actually SPLIT — Lines needs ``segments`` to have
#: anything to cut (a single polyline partitions to one node), and the split is
#: asserted rather than assumed by every case below.
STRUCTURAL_ROUTES = [
    ("points_partition", _add_points, {"partition": {"max_elements": 3}}),
    (
        "lines_partition",
        _add_lines,
        {"line_type": "segments", "partition": {"max_elements": 4}},
    ),
    ("mesh_partition", _add_mesh, {"partition": {"max_elements": 8}}),
    ("gsplats_partition", _add_gsplats, {"partition": {"max_elements": 3}}),
    ("points_substitutive_lod", _add_points, {"substitutive_lod": True}),
    ("lines_substitutive_lod", _add_lines, {"substitutive_lod": True}),
]


class TestTheStripRunsAboveEveryStructuralBranch:
    """Placement, not merely the rule: the None must not reach a CHILD either.

    The flat cases above are all satisfiable by a strip scoped to the plain-leaf
    write — measured, by scoping each adder's call to
    ``if partition is None and substitutive_lod is None:``, which leaves every
    flat case green while ``add_points(..., partition={'max_elements': 3},
    colormap=None)`` goes back to stamping ``colormap='custom'`` on all four
    parts. That is the case that actually ships: a partitioned / LOD node is
    exactly the one nobody inspects attr-by-attr, and the viewer renders every
    LUT-less ``'custom'`` part in viridis.

    So this asserts over EVERY descendant of the written tree, not the node the
    adder returned. Both keys travel in ONE write per route — they are
    independent keys of the same dict past the same single strip, and each route
    is a real zarr store.

    Five of the six cases fail against that mutation; ``lines_partition`` is the
    one that does not, and it is kept as a plain regression case rather than
    advertised as a proof. Every partition path writes its parts by RECURSING
    into its own adder, and the recursive call's no-partition sentinel decides
    whether a ``partition is None`` guard still fires down there: Points / Mesh /
    GSplats pass ``partition=False`` (so the guard is False and the mutant skips
    the child's strip too — the raw None reaches all four parts), while Lines
    passes ``partition=None`` (so the child re-strips and absorbs the mutation).
    Lines' own discriminating route is ``substitutive_lod`` below it.
    """

    @pytest.mark.parametrize(
        "route,adder,structure",
        STRUCTURAL_ROUTES,
        ids=[r[0] for r in STRUCTURAL_ROUTES],
    )
    def test_no_descendant_carries_either_wrong_value(
        self,
        tmp_path: Any,
        route: str,
        adder: Callable[..., Any],
        structure: Dict[str, Any],
    ) -> None:
        tree = _write_tree(
            tmp_path,
            f"{route}.luxar.zarr",
            adder,
            colormap=None,
            coverage_fraction=None,
            **structure,
        )

        # The parent plus at least TWO children: a wrapper over a single child
        # is not a split either, and only a real split puts the stripped attrs
        # on more than one node.
        assert len(tree) >= 3, (
            f"{route} wrote {len(tree)} node(s), expected a parent with at least "
            f"two children, so this case would pass without ever exercising a "
            f"structural branch: {sorted(tree)}"
        )

        lutless = {
            node: sorted(arrays)
            for node, (node_attrs, arrays) in tree.items()
            if node_attrs.get("colormap") == "custom" and "colormap_lut" not in arrays
        }
        assert not lutless, (
            f"{route} shipped a LUT-less colormap='custom' on {sorted(lutless)}; "
            "the viewer warns and falls back to VIRIDIS where the omitted key "
            "gives gray"
        )

        nulls = [
            node
            for node, (node_attrs, _) in tree.items()
            if "coverage_fraction" in node_attrs
            and node_attrs["coverage_fraction"] is None
        ]
        assert not nulls, (
            f"{route} persisted a literal coverage_fraction: null on {nulls}"
        )


class TestTheDataDoorAndTheLeafDoorAgree:
    def test_colormap_none_reads_the_same_through_both(self, tmp_path: Any) -> None:
        """``add_gsplats_from_data`` and the leaf ``add_gsplats``, side by side.

        The asymmetry #1574 names: the outer door stripped the None (#1496) and
        wrote ``'gray'``, the leaf it delegates to rewrote it to ``'custom'``, so
        the same argument meant two different things one call apart.

        The outer door is driven through ``additive_lod=`` rather than its flat
        route on purpose. The flat route DELEGATES to the very leaf adder this
        module already parametrizes, so with the outer strip removed its None
        would simply fall through to the leaf strip and the case would still pass
        — a duplicate of the gsplats leaf case wearing a two-door label. The
        laddered route instead lands in ``gsplats_pipeline/lod_dispatch.py``,
        which writes the ladder itself and runs its OWN
        ``sync_custom_colormap_attr`` with no leaf strip anywhere above it:
        measured, with the ``from_data`` strip removed that route writes
        ``colormap='custom'`` on the ladder parent while the leaf beside it still
        writes ``'gray'``. So each door is answering for itself here.

        (The outer door's own five-key × four-route matrix lives with #1496, in
        ``lod/test_source_validation.py``; this case exists only to pin that the
        two doors READ THE SAME, which is the asymmetry #1574 opened with.)
        """
        from luxar.gsplats.gsplat_data import GSplatData

        data = GSplatData(
            centers=random_positions(_N, seed=1577),
            amplitudes=np.ones(_N, dtype=np.float32),
            cholesky_factors=cholesky_rows(_N),
        )

        compiler, scene, path = open_scene(tmp_path, "both_doors.luxar.zarr")
        scene.add_gsplats_from_data(
            "from_data", data, additive_lod={"n_lods": 2}, colormap=None
        )
        _add_gsplats(scene, "leaf", colormap=None)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        leaf = store["leaf"].attrs.get("colormap", _ABSENT)
        # Over whichever nodes of the ladder carry the attr at all, rather than
        # one named node: the ladder puts it on the PARENT and leaves the
        # ``additive_<i>`` rungs without one, and that layout is not this test's
        # business — the VALUE is.
        outer = {
            node: node_attrs["colormap"]
            for node, (node_attrs, _) in _subtree(store, "from_data").items()
            if "colormap" in node_attrs
        }
        assert outer, "no node of the data door's ladder carried a colormap"
        assert set(outer.values()) == {leaf}, (
            f"the leaf wrote colormap={leaf!r} where the data door wrote {outer}"
        )


class TestARenderAttrOutsideTheSetStillRefusesItsNone:
    """Scope control: the strip must not start swallowing the LOUD Nones.

    Every render attr other than the two in the set runs its value validator
    unconditionally, so a None there is already reported rather than persisted —
    and reading it as "absent" would only mask a typo. These cases pass before
    and after the fix; they fail against a strip widened past its set.
    """

    @pytest.mark.parametrize("geometry,adder", LEAF_ADDERS)
    def test_opacity_none_is_reported(
        self, tmp_path: Any, geometry: str, adder: Callable[..., Any]
    ) -> None:
        with pytest.raises(ValueError, match="convertible to float, got NoneType"):
            _write_leaf(tmp_path, f"{geometry}_opacity.luxar.zarr", adder, opacity=None)

    def test_truncation_radius_none_is_reported(self, tmp_path: Any) -> None:
        """The gsplat-only member of the same family, and the near miss.

        It IS in the pipeline's wider set (there an explicit None must decline to
        override the ``GSplatData``'s own radius), so a leaf strip reusing that
        set unchanged would have silently dropped a caller's None here instead of
        answering it.
        """
        with pytest.raises(ValueError, match="convertible to float, got NoneType"):
            _write_leaf(
                tmp_path, "trunc.luxar.zarr", _add_gsplats, truncation_radius=None
            )


class TestTheLeafSetIsTheOnePipelineDerivesFrom:
    """Scope control: the wider pipeline set must keep AGREEING with this one.

    It bounds drift rather than forbidding a second list outright — see the
    measurements in the case below for exactly which change is caught and when.
    """

    def test_the_leaf_set_is_the_two_silently_wrong_render_attrs(self) -> None:
        assert set(ABSENT_WHEN_NONE_RENDER_ATTRS) == {"colormap", "coverage_fraction"}

    def test_the_pipeline_set_is_this_one_plus_its_own_keys(self) -> None:
        """#1496's wider set ADDS to this one; it does not restate it.

        The CONCATENATION, not a subset relation: ``set(RENDER) <= set(ATTRS)``
        holds for any value of either tuple as long as the derivation exists, so
        it can never fail. Comparing the pipeline tuple's VALUE against its two
        component sources is strictly stronger, but it does not catch a
        hand-copied list on the day the copy is written — measured: replacing
        the derivation in ``from_data.py`` with a hand-typed 7-tuple of today's
        keys in today's order still passes, because the value is unchanged.
        What it does catch is the first CHANGE to either component afterwards:
        measured, that same hand-typed literal plus one key added to
        ``GATE_FORWARDED_LEAF_PARAMS`` fails here, so a copy cannot survive its
        first divergence from the sources it was copied from.
        """
        assert ABSENT_WHEN_NONE_ATTRS == (
            GATE_FORWARDED_LEAF_PARAMS
            + ("colors", "truncation_radius")
            + ABSENT_WHEN_NONE_RENDER_ATTRS
        )
