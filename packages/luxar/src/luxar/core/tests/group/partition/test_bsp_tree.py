"""Direct tests for the BSP *tree* primitive (``spatial_bsp_tree`` / ``BSPNode``).

The three flat splitters (``median``/``midpoint``/``sah``) each have their own
suite (``test_median`` / ``test_bsp`` / ``test_sah`` / ``test_invariants``); this
is the parallel suite for their shared tree sibling — the split-plane record the
gsplat ``kind=partition`` viewer traverses for exact back-to-front ordering.

Two contracts anchor everything downstream:

1. **Leaves == flat parts.** ``spatial_bsp_tree(...).leaves()`` yields exactly
   the index arrays (and in the same order) the matching ``*_bsp_partition``
   returns. This is what lets the flat splitters delegate to the tree builder
   without changing their output, and what keeps a serialized leaf's ``part``
   index aligned with the on-disk ``part_<i>`` / ``child_index``.
2. **Planes separate their subtrees.** At every internal node, all left-subtree
   coordinates are ``< split`` and all right-subtree coordinates are ``>= split``
   on a spatial ``axis`` in ``{0, 1, 2}`` — the invariant the viewer's painter's
   traversal relies on for a valid order at any camera position.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from luxar.core.group.partition import (
    BSPNode,
    median_bsp_partition,
    midpoint_bsp_partition,
    sah_bsp_partition,
    spatial_bsp_tree,
)

RULES = ["median", "midpoint", "sah"]
FLAT = {
    "median": median_bsp_partition,
    "midpoint": midpoint_bsp_partition,
    "sah": sah_bsp_partition,
}


def _leaves(node: BSPNode) -> list[BSPNode]:
    return [node] if node.is_leaf else _leaves(node.left) + _leaves(node.right)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("seed", range(6))
def test_tree_leaves_match_flat_splitter(rule, seed) -> None:
    """The tree's leaves (left-first DFS) reproduce the flat splitter output
    exactly — same count, same arrays, same order."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-10.0, 10.0, (500, 3)).astype(np.float32)
    tree = spatial_bsp_tree(pos, max_elements=64, rule=rule)
    tree_parts = [leaf.indices for leaf in tree.leaves()]
    flat_parts = FLAT[rule](pos, max_elements=64)
    assert len(tree_parts) == len(flat_parts)
    for t, f in zip(tree_parts, flat_parts):
        np.testing.assert_array_equal(t, f)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("seed", range(6))
def test_tree_disjoint_cover_and_plane_consistency(rule, seed) -> None:
    """Leaves partition ``range(N)``; every split cleanly separates its
    subtrees on a spatial axis."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-10.0, 10.0, (400, 3)).astype(np.float32)
    tree = spatial_bsp_tree(pos, max_elements=50, rule=rule)

    concat = np.concatenate([leaf.indices for leaf in tree.leaves()])
    np.testing.assert_array_equal(np.sort(concat), np.arange(400))

    def check(node: BSPNode) -> None:
        if node.is_leaf:
            return
        assert node.axis in (0, 1, 2)
        left = np.concatenate(
            [pos[leaf.indices][:, node.axis] for leaf in _leaves(node.left)]
        )
        right = np.concatenate(
            [pos[leaf.indices][:, node.axis] for leaf in _leaves(node.right)]
        )
        assert left.max() <= node.split + 1e-4
        assert right.min() >= node.split - 1e-4
        check(node.left)
        check(node.right)

    check(tree)


@pytest.mark.parametrize("rule", RULES)
def test_to_serializable_is_json_safe_and_indexes_parts(rule) -> None:
    """``to_serializable`` emits a JSON-safe nested dict (no numpy scalars)
    whose leaf ``part`` refs are ``0..P-1`` in DFS order — aligned with the
    flat parts / ``child_index``."""
    rng = np.random.RandomState(0)
    pos = rng.uniform(-10.0, 10.0, (300, 3)).astype(np.float32)
    tree = spatial_bsp_tree(pos, max_elements=40, rule=rule)
    ser = tree.to_serializable()
    reloaded = json.loads(json.dumps(ser))  # must not raise (plain int/float only)

    refs: list[int] = []

    def walk(node: dict) -> None:
        if "part" in node:
            refs.append(node["part"])
        else:
            assert isinstance(node["axis"], int) and isinstance(node["split"], float)
            walk(node["left"])
            walk(node["right"])

    walk(reloaded)
    assert refs == list(range(len(list(tree.leaves()))))


def test_under_cap_and_coincident_yield_single_leaf() -> None:
    """A whole input at/under the cap, or fully coincident points, is one leaf
    (no split, no spurious plane)."""
    rng = np.random.RandomState(1)
    under = rng.uniform(0, 1, (20, 3)).astype(np.float32)
    tree = spatial_bsp_tree(under, max_elements=100, rule="median")
    assert tree.is_leaf and tree.axis is None
    np.testing.assert_array_equal(np.sort(tree.indices), np.arange(20))

    coincident = np.ones((30, 3), dtype=np.float32)
    ctree = spatial_bsp_tree(coincident, max_elements=5, rule="median")
    # No split can make spatial progress → one oversized leaf, not a crash.
    assert ctree.is_leaf and ctree.indices.size == 30


def test_validation_errors() -> None:
    good = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        spatial_bsp_tree(np.zeros(10, dtype=np.float32), 5)
    with pytest.raises(ValueError, match="2 spatial"):
        spatial_bsp_tree(np.zeros((10, 1), dtype=np.float32), 5)
    with pytest.raises(ValueError, match="max_elements"):
        spatial_bsp_tree(good, 0)
    with pytest.raises(ValueError, match="non-empty"):
        spatial_bsp_tree(np.zeros((0, 3), dtype=np.float32), 5)
    with pytest.raises(ValueError, match="rule must be"):
        spatial_bsp_tree(good, 5, rule="octree")
    with pytest.raises(ValueError, match="n_candidates"):
        spatial_bsp_tree(good, 5, rule="sah", n_candidates=1)
