"""The child-enumeration contract of ``finalize.amplitude_window``, pinned.

``_child_nodes`` and ``_lod_children`` are private to that module, but they are
no longer one module's implementation detail: :mod:`luxar.io.lod_restamp`
imports them verbatim so its rewrite of a store's LOD thresholds mirrors the
writers rather than re-stating them. That pass hands the LAST child's element
count to the derivation and writes the finest threshold onto the last child, so
narrowing :data:`_NODE_TYPES` or dropping an ordering fallback would silently
change what ``luxar restamp-lod`` writes onto stores already on disk — with no
test anywhere going red, since ``amplitude_window``'s own coverage is indirect.

What is pinned here is exactly what the reuse depends on: the three ordering
tiers, and which child groups count as ladder levels at all.
"""

from __future__ import annotations

from typing import Any, Dict, List

import zarr

from luxar.io._compiler.finalize.amplitude_window import (
    _NODE_TYPES,
    _child_nodes,
    _lod_children,
)
from luxar.typing_utils._format_contract import GEOMETRY_TYPES


def _group(children: Dict[str, Dict[str, Any]]) -> zarr.Group:
    """An in-memory group whose child groups carry the given attrs verbatim."""
    root = zarr.group()
    for name, attrs in children.items():
        root.create_group(name).attrs.update(attrs)
    return root


def _names(kids: List[Any]) -> List[str]:
    return [name for name, _, _ in kids]


# ────────────────────────────────────────────────────────────────────────
# _lod_children — the three ordering tiers, coarsest → finest
# ────────────────────────────────────────────────────────────────────────


def test_child_index_orders_the_ladder_whatever_the_names_say() -> None:
    """Tier (a). ``child_index`` records on-disk insertion order, and every
    producer inserts coarsest first — so it beats any name signal."""
    root = _group(
        {
            "a": {"type": "points", "child_index": 2},
            "b": {"type": "points", "child_index": 0},
            "c": {"type": "points", "child_index": 1},
        }
    )

    assert _names(_lod_children(root)) == ["b", "c", "a"]


def test_a_partially_stamped_child_index_falls_through_to_the_name_suffix() -> None:
    """Tier (a) needs EVERY candidate to carry one, or the order is a guess.

    ``child_1`` claims index 0 here, so a tier that accepted a partial stamping
    would put it first; the ``child_<i>`` suffix says otherwise and wins.
    """
    root = _group(
        {
            "child_0": {"type": "points"},
            "child_1": {"type": "points", "child_index": 0},
        }
    )

    assert _names(_lod_children(root)) == ["child_0", "child_1"]


def test_the_name_suffix_tier_sorts_numerically_not_alphabetically() -> None:
    """Tier (b). Alphabetically ``child_10`` precedes ``child_2``, which would
    hand the finest threshold — and the finest element count — to the wrong
    child of any ten-plus-level ladder."""
    root = _group(
        {
            "child_10": {"type": "points"},
            "child_2": {"type": "points"},
            "child_9": {"type": "points"},
        }
    )

    assert _names(_lod_children(root)) == ["child_2", "child_9", "child_10"]


def test_sorted_name_is_the_last_resort_for_a_foreign_store() -> None:
    """Tier (c). Unreachable from a Python producer (``Node.__init__`` always
    stamps ``child_index``) and therefore only ever exercised by the hand-edited
    and third-party stores ``restamp-lod`` exists for."""
    root = _group(
        {
            "fine": {"type": "points"},
            "coarse": {"type": "points"},
            "mid": {"type": "points"},
        }
    )

    assert _names(_lod_children(root)) == ["coarse", "fine", "mid"]


# ────────────────────────────────────────────────────────────────────────
# _child_nodes — which child groups are scene nodes at all
# ────────────────────────────────────────────────────────────────────────


def test_every_node_type_counts_as_a_child_node() -> None:
    """The whole vocabulary, enumerated from the FORMAT CONTRACT rather than
    from ``_NODE_TYPES`` itself — building the fixture out of the constant under
    test shrinks both sides together, so a narrowing (the drift this pins) would
    pass. Dropping a type here drops a real ladder level, and the element count
    read off the level that is left is what the threshold derivation is handed.
    """
    expected = sorted({"group", *GEOMETRY_TYPES})
    root = _group({name: {"type": name} for name in expected})

    assert sorted(_names(_child_nodes(root))) == expected
    assert sorted(_NODE_TYPES) == expected


def test_only_typed_scene_nodes_count_as_children() -> None:
    """Untyped and unknown-type children are metadata; a nested node name is authored."""
    root = _group(
        {
            "child_0": {"type": "points", "child_index": 0},
            "child_1": {"child_index": 1},
            "labels": {"type": "not-a-node"},
            "environment": {"type": "group"},
            "pipeline": {"type": "group"},
        }
    )

    assert sorted(_names(_child_nodes(root))) == ["child_0", "environment"]
