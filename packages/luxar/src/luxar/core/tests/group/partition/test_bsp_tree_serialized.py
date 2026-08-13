"""Tests for the algebra on the *serialized* BSP tree (the ``bsp_tree`` attr).

Sibling of ``test_bsp_tree`` (which covers the :class:`BSPNode` primitive). What
is tested here is the lifetime of that tree once it has left the splitter and
become a plain dict: renumbering it when a producer drops empty regions
(:func:`prune_serialized_bsp_tree`), and carrying it through a transform on the
centers (:func:`map_serialized_bsp_tree`).

Both exist because the failure they prevent is SILENT. A tree whose leaf labels
no longer match the written parts, or whose planes sit in a stale coordinate
space, still traverses to a perfectly valid *permutation* of the parts — the
viewer draws confidently in the wrong order rather than falling back to its
documented centroid heuristic. So the assertions here are about labels and
coordinates agreeing with reality, not merely about the tree being well-formed.

The last test is the end-to-end one: that a tree's traversal actually produces a
correct painter's order, checked against separating-plane constraints over many
camera poses rather than against a hand-written expected order.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from luxar.core.group.partition import (
    map_serialized_bsp_tree,
    prune_serialized_bsp_tree,
    spatial_bsp_tree,
)

# A 4-leaf tree: two cuts on axis 0, one on axis 1.
TREE = {
    "axis": 0,
    "split": 10.0,
    "left": {"axis": 1, "split": 5.0, "left": {"part": 0}, "right": {"part": 1}},
    "right": {"axis": 0, "split": 20.0, "left": {"part": 2}, "right": {"part": 3}},
}


def _leaf_labels(node: dict) -> list[int]:
    """Leaf ``part`` labels in left-first order."""
    if "part" in node:
        return [node["part"]]
    return _leaf_labels(node["left"]) + _leaf_labels(node["right"])


# ── prune_serialized_bsp_tree ────────────────────────────────────────────


class TestPrune:
    def test_keeping_everything_is_the_identity(self) -> None:
        assert prune_serialized_bsp_tree(TREE, range(4)) == TREE

    def test_dropping_a_leaf_collapses_its_parent_and_renumbers(self) -> None:
        # Losing part 1 leaves nothing on the far side of the y=5 plane, so that
        # plane carries no ordering information and the node collapses. Parts 2
        # and 3 shift down to 1 and 2 — they are the 2nd and 3rd SURVIVORS, which
        # is how the writers assign child_index.
        assert prune_serialized_bsp_tree(TREE, [0, 2, 3]) == {
            "axis": 0,
            "split": 10.0,
            "left": {"part": 0},
            "right": {
                "axis": 0,
                "split": 20.0,
                "left": {"part": 1},
                "right": {"part": 2},
            },
        }

    def test_renumbering_is_by_ascending_label_not_tree_order(self) -> None:
        # The writers count survivors in ascending ORIGINAL index, so the label
        # -> child_index map must follow that, whatever order the tree visits in.
        pruned = prune_serialized_bsp_tree(TREE, [3, 1])
        assert pruned is not None
        # Original 1 is the first survivor, original 3 the second.
        assert _leaf_labels(pruned) == [0, 1]

    def test_a_single_survivor_collapses_to_a_bare_leaf(self) -> None:
        assert prune_serialized_bsp_tree(TREE, [2]) == {"part": 0}

    def test_no_survivors_and_none_input_yield_none(self) -> None:
        assert prune_serialized_bsp_tree(TREE, []) is None
        assert prune_serialized_bsp_tree(None, [0, 1]) is None

    def test_a_kept_label_absent_from_the_tree_gives_up_entirely(self) -> None:
        # Emitting a partial tree here would leave the viewer mixing ranked and
        # unranked parts. Better to have no tree than a tree covering some parts.
        assert prune_serialized_bsp_tree(TREE, [0, 1, 2, 3, 4]) is None

    def test_pruning_never_invents_or_duplicates_a_label(self) -> None:
        for r in range(1, 5):
            for keep in itertools.combinations(range(4), r):
                pruned = prune_serialized_bsp_tree(TREE, keep)
                assert pruned is not None
                assert sorted(_leaf_labels(pruned)) == list(range(len(keep)))


# ── map_serialized_bsp_tree ──────────────────────────────────────────────


class TestMapThroughAffine:
    def test_no_transform_returns_the_tree_unchanged(self) -> None:
        assert map_serialized_bsp_tree(TREE, None, None) is TREE
        assert map_serialized_bsp_tree(None, np.eye(3), np.zeros(3)) is None

    def test_translation_shifts_each_plane_along_its_own_axis(self) -> None:
        out = map_serialized_bsp_tree(TREE, np.eye(3), np.array([1.0, 2.0, 3.0]))
        assert out is not None
        assert out["split"] == pytest.approx(11.0)  # axis 0
        assert out["left"]["split"] == pytest.approx(7.0)  # axis 1

    def test_per_axis_scale_scales_each_plane_by_its_own_factor(self) -> None:
        out = map_serialized_bsp_tree(TREE, np.diag([4.0, 1.0, 1.0]), np.zeros(3))
        assert out is not None
        assert out["split"] == pytest.approx(40.0)
        assert out["left"]["split"] == pytest.approx(5.0)

    def test_a_reflection_swaps_the_halves_of_every_mirrored_node(self) -> None:
        # `left` means `coord < split`; mirroring inverts that, so a mirrored node
        # must trade its subtrees or the painter's order comes out backwards. Note
        # this applies at EVERY axis-0 node, nested ones included — hence
        # left-first order [3, 2] below — while the axis-1 node, untouched by a
        # mirror on axis 0, keeps its halves.
        out = map_serialized_bsp_tree(TREE, np.diag([-1.0, 1.0, 1.0]), np.zeros(3))
        assert out == {
            "axis": 0,
            "split": -10.0,
            "left": {
                "axis": 0,
                "split": -20.0,
                "left": {"part": 3},
                "right": {"part": 2},
            },
            "right": {
                "axis": 1,
                "split": 5.0,
                "left": {"part": 0},
                "right": {"part": 1},
            },
        }

    def test_a_quarter_turn_remaps_the_split_axis(self) -> None:
        # p -> (-y, x, z): the plane x = 10 becomes the plane y = 10.
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        out = map_serialized_bsp_tree(TREE, rot, np.zeros(3))
        assert out is not None
        assert out["axis"] == 1
        assert out["split"] == pytest.approx(10.0)

    def test_an_arbitrary_rotation_is_refused(self) -> None:
        theta = 0.6
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert map_serialized_bsp_tree(TREE, rot, np.zeros(3)) is None

    def test_an_axis_mapped_beyond_the_third_is_refused(self) -> None:
        # The serialized format admits split axes 0/1/2 only, so a permutation
        # sending axis 0 to axis 3 has no representable image.
        lin = np.zeros((4, 4))
        lin[3, 0] = lin[0, 1] = lin[1, 2] = lin[2, 3] = 1.0
        assert map_serialized_bsp_tree(TREE, lin, np.zeros(4)) is None

    def test_mapping_tracks_the_centers_it_describes(self) -> None:
        # The real invariant: after transforming BOTH the points and the tree,
        # the planes must still separate the same point sets.
        rng = np.random.default_rng(0)
        points = rng.uniform(-50, 50, size=(400, 3))
        tree = spatial_bsp_tree(points, max_elements=60, rule="median")
        serialized = tree.to_serializable()
        parts = [leaf.indices for leaf in tree.leaves()]

        linear = np.diag([2.0, -3.0, 0.5])
        shift = np.array([7.0, -1.0, 4.0])
        moved = points @ linear.T + shift
        mapped = map_serialized_bsp_tree(serialized, linear, shift)
        assert mapped is not None

        def check(node: dict) -> None:
            if "part" in node:
                return
            axis, split = node["axis"], node["split"]
            for label in _leaf_labels(node["left"]):
                assert moved[parts[label]][:, axis].max() < split
            for label in _leaf_labels(node["right"]):
                assert moved[parts[label]][:, axis].min() >= split
            check(node["left"])
            check(node["right"])

        check(mapped)


# ── the ordering guarantee itself ────────────────────────────────────────


def _traverse_back_to_front(node: dict, eye: np.ndarray, out: list[int]) -> None:
    """The viewer's painter traversal (``render-order.ts::traverseBspBackToFront``)."""
    if "part" in node:
        out.append(node["part"])
        return
    if eye[node["axis"]] < node["split"]:
        _traverse_back_to_front(node["right"], eye, out)
        _traverse_back_to_front(node["left"], eye, out)
    else:
        _traverse_back_to_front(node["left"], eye, out)
        _traverse_back_to_front(node["right"], eye, out)


def _required_last(lo_a, hi_a, lo_b, hi_b, eye) -> str | None:
    """Which of two disjoint boxes MUST be drawn last, or ``None`` if either may.

    For a separating axis-aligned plane, every ray from the eye that hits both
    boxes hits the eye's own side first — so that side is nearer and must be
    drawn LAST. When the eye lies strictly INSIDE a separating gap, two planes
    straddle it and give opposite verdicts, which proves neither box can occlude
    the other; that vetoes any verdict from another axis.
    """
    verdicts = set()
    for axis in range(3):
        for name, (lo_x, hi_x, lo_y, hi_y) in (
            ("a", (lo_a, hi_a, lo_b, hi_b)),
            ("b", (lo_b, hi_b, lo_a, hi_a)),
        ):
            if hi_x[axis] <= lo_y[axis]:  # x is the low box on this axis
                if hi_x[axis] < eye[axis] < lo_y[axis]:
                    return None  # eye inside the gap -> mutual non-occlusion
                verdicts.add(
                    name if eye[axis] <= hi_x[axis] else ("b" if name == "a" else "a")
                )
    return verdicts.pop() if len(verdicts) == 1 else None


def test_traversal_is_a_correct_painters_order_from_every_camera_pose() -> None:
    """The end-to-end guarantee, checked against geometry rather than a fixture.

    For every camera pose — including INSIDE the volume, where a centroid sort
    fails worst — the traversal must violate ZERO pairwise ordering constraints.
    A pair is constrained when a separating plane makes one box unambiguously
    nearer; the "either order is fine" pairs are excluded rather than guessed at.

    This is the property the whole ``bsp_tree`` mechanism exists to provide, and
    the one a centroid heuristic does not have.
    """
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 100, size=(2000, 3))
    tree = spatial_bsp_tree(points, max_elements=150, rule="median")
    serialized = tree.to_serializable()
    boxes = [
        (points[leaf.indices].min(axis=0), points[leaf.indices].max(axis=0))
        for leaf in tree.leaves()
    ]
    assert len(boxes) > 4, "need a non-trivial partition for this to mean anything"

    centre = np.array([50.0, 50.0, 50.0])
    violations = 0
    constrained = 0
    for _ in range(120):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        for distance in (150.0, 20.0):  # outside the volume, then inside it
            eye = centre + direction * distance
            order: list[int] = []
            _traverse_back_to_front(serialized, eye, order)
            rank = {part: i for i, part in enumerate(order)}
            assert sorted(rank) == list(range(len(boxes)))
            for a, b in itertools.combinations(range(len(boxes)), 2):
                needs_last = _required_last(*boxes[a], *boxes[b], eye)
                if needs_last is None:
                    continue
                constrained += 1
                drawn_last = "a" if rank[a] > rank[b] else "b"
                violations += drawn_last != needs_last

    assert constrained > 0, "no constrained pairs — the check would be vacuous"
    assert violations == 0
