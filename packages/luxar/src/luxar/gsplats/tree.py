"""Node-tree model for Gaussian splats.

This is the unified in-memory representation behind the v3.0 ``.gsplats.zarr``
format and the scene gsplat-node subtree: a standalone ``.gsplats.zarr`` *is* a
detached node subtree, and embedding it into a scene is a graft of that subtree.

Three node types compose freely (and nest arbitrarily):

* :class:`GSplatLeaf` — a gsplats leaf carrying an **additive ladder** (one or
  more :class:`~luxar.gsplats.gsplat_data.AdditiveSubLOD`, prefix-sum / additive
  LOD). The trivial single-splat-set case is a leaf with a one-entry ladder.
* :class:`GSplatLodGroup` — **substitutive** LOD: children are rendered one at a
  time (the scene ``kind=lod`` Group). Children are ordered **finest → coarsest**
  in memory (matching the historical ``GSplatData.substitutive_levels`` index-0
  = finest convention); the v3.0 serializer is responsible for the on-disk
  ``child_<i>`` coarsest-first convention.
* :class:`GSplatPartition` — **spatial** split: all children are rendered (the
  scene ``kind=partition`` Group), each carrying its own ``position_bounds``.

The classes are intentionally small, pure, and immutable (frozen dataclasses) so
they are trivially unit-testable in isolation. Per-node metadata (LOD provenance
such as ``compression_factor`` / ``parent_method`` / ``level_index``, the
view-driven ``min_pixel_size`` selector threshold, per-node ``stats``) lives in a
free-form ``meta`` dict on each node — mirroring the zarr ``.zattrs`` a node
carries on disk.

The :func:`tree_from_substitutive_levels` / :func:`substitutive_levels_from_tree`
bridge converts to and from the historical 2-D ``substitutive × additive`` matrix
representation (``GSplatData.substitutive_levels``) so the tree can be introduced
without breaking the existing matrix-shaped API. The matrix is exactly one shape
of the tree: a single :class:`GSplatLodGroup` of leaves (or, for a single
substitutive level, a bare :class:`GSplatLeaf`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel


# ────────────────────────────────────────────────────────────────────────
# Node types
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class GSplatLeaf:
    """A gsplats leaf carrying an additive ladder (≥ 1 ``AdditiveSubLOD``).

    Attributes
    ----------
    additive_sublods : list[AdditiveSubLOD]
        The additive (prefix-sum) ladder. Always ≥ 1 entry; a single entry is
        the trivial "no additive sub-ordering" case.
    meta : dict
        Free-form per-node metadata (the node's zarr ``.zattrs``). Recognised
        optional keys include ``compression_factor`` / ``parent_method`` /
        ``level_index`` (LOD provenance when this leaf is a substitutive level),
        ``min_pixel_size`` (selector threshold when a child of a lod group),
        and ``stats``.
    """

    additive_sublods: "List[AdditiveSubLOD]"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.additive_sublods:
            raise ValueError("GSplatLeaf must contain at least one AdditiveSubLOD")

    @property
    def n_additive_sublods(self) -> int:
        """Number of additive sub-LODs in this leaf (≥ 1)."""
        return len(self.additive_sublods)

    @property
    def n_splats(self) -> int:
        """Total splats across this leaf's additive ladder."""
        return int(sum(sub.n_splats for sub in self.additive_sublods))

    @property
    def ndim(self) -> int:
        """Spatial dimensionality (from the first additive sub-LOD)."""
        return int(self.additive_sublods[0].ndim)

    def __repr__(self) -> str:
        ladder = (
            f", {self.n_additive_sublods} additive"
            if self.n_additive_sublods > 1
            else ""
        )
        return f"GSplatLeaf({self.n_splats:,} splats, {self.ndim}D{ladder})"


@dataclass(frozen=True, eq=False)
class GSplatLodGroup:
    """Substitutive LOD group — children rendered one at a time (``kind=lod``).

    Children are ordered **finest → coarsest** in memory. ``default_level`` is
    the index of the child a simple consumer renders by default.
    """

    children: "List[GSplatNode]"
    default_level: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("GSplatLodGroup must contain at least one child")
        if not 0 <= self.default_level < len(self.children):
            raise ValueError(
                f"default_level {self.default_level} out of range "
                f"[0, {len(self.children)})"
            )

    @property
    def n_children(self) -> int:
        return len(self.children)

    @property
    def n_splats(self) -> int:
        """Splats of the default child (a substitutive group renders one child)."""
        return int(self.children[self.default_level].n_splats)

    @property
    def ndim(self) -> int:
        return int(self.children[0].ndim)

    def __repr__(self) -> str:
        return (
            f"GSplatLodGroup({self.n_children} levels, default={self.default_level}, "
            f"{self.ndim}D)"
        )


@dataclass(frozen=True, eq=False)
class GSplatPartition:
    """Spatial partition group — all children rendered (``kind=partition``).

    Each child is a spatial part; ``max_elements`` records the BSP target used
    to build the partition.
    """

    children: "List[GSplatNode]"
    max_elements: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("GSplatPartition must contain at least one child")

    @property
    def n_children(self) -> int:
        return len(self.children)

    @property
    def n_splats(self) -> int:
        """Total splats across all parts (a partition renders every child)."""
        return int(sum(child.n_splats for child in self.children))

    @property
    def ndim(self) -> int:
        return int(self.children[0].ndim)

    def __repr__(self) -> str:
        return f"GSplatPartition({self.n_children} parts, {self.ndim}D)"


#: A node in the gsplat tree — a leaf or one of the two group kinds.
GSplatNode = Union[GSplatLeaf, GSplatLodGroup, GSplatPartition]


# ────────────────────────────────────────────────────────────────────────
# Structural helpers (pure, work on any node)
# ────────────────────────────────────────────────────────────────────────


def iter_leaves(node: GSplatNode) -> Iterator[GSplatLeaf]:
    """Yield every :class:`GSplatLeaf` in ``node`` (depth-first, pre-order)."""
    if isinstance(node, GSplatLeaf):
        yield node
    elif isinstance(node, (GSplatLodGroup, GSplatPartition)):
        for child in node.children:
            yield from iter_leaves(child)
    else:  # pragma: no cover - guards against an unknown node type
        raise TypeError(f"Unknown gsplat node type: {type(node).__name__}")


def total_splats(node: GSplatNode) -> int:
    """Total splats across **every** leaf in the subtree (ignores LOD selection).

    Distinct from ``node.n_splats``, which honours substitutive selection (a lod
    group reports only its default child). This sums all stored splats.
    """
    return int(sum(leaf.n_splats for leaf in iter_leaves(node)))


def node_ndim(node: GSplatNode) -> int:
    """Spatial dimensionality of the subtree (from its first leaf)."""
    for leaf in iter_leaves(node):
        return leaf.ndim
    raise ValueError("empty tree has no dimensionality")  # pragma: no cover


def center_bounds(node: GSplatNode) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned bounds of all splat **centers** in the subtree.

    Returns ``(min, max)`` float arrays of shape ``(d,)``, or ``None`` if the
    subtree holds zero splats. This is a center-only bound; the serializer adds
    the 3σ ellipsoidal extent when it writes ``position_bounds`` / ``chunk_bounds``.
    """
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for leaf in iter_leaves(node):
        for sub in leaf.additive_sublods:
            if sub.n_splats == 0:
                continue
            mins.append(sub.centers.min(axis=0))
            maxs.append(sub.centers.max(axis=0))
    if not mins:
        return None
    return (
        np.min(np.stack(mins, axis=0), axis=0),
        np.max(np.stack(maxs, axis=0), axis=0),
    )


# ────────────────────────────────────────────────────────────────────────
# Bridge: 2-D substitutive × additive matrix  ⇄  node tree
# ────────────────────────────────────────────────────────────────────────

#: meta keys that carry substitutive-level provenance on a leaf node.
_SUBSTITUTIVE_META_KEYS = ("compression_factor", "parent_method", "level_index")


def _leaf_from_substitutive_level(level: "SubstitutiveLevel") -> GSplatLeaf:
    """Wrap one ``SubstitutiveLevel`` as a :class:`GSplatLeaf` (metadata preserved)."""
    return GSplatLeaf(
        additive_sublods=list(level.additive_sublods),
        meta={
            "compression_factor": level.compression_factor,
            "parent_method": level.parent_method,
            "level_index": level.level_index,
            "stats": dict(level.stats),
        },
    )


def tree_from_substitutive_levels(
    levels: "List[SubstitutiveLevel]",
    default_substitutive: int = 0,
) -> GSplatNode:
    """Build a node tree from the historical 2-D matrix representation.

    * A single substitutive level → a bare :class:`GSplatLeaf` (its additive
      ladder), carrying that level's provenance in ``meta``.
    * Multiple substitutive levels → a :class:`GSplatLodGroup` of one leaf per
      level, in the same order as ``levels`` (index 0 = finest), with
      ``default_level = default_substitutive``.

    This is the inverse of :func:`substitutive_levels_from_tree` for any tree
    that is matrix-shaped (a leaf, or a lod group whose children are all leaves).
    """
    if not levels:
        raise ValueError("levels must contain at least one SubstitutiveLevel")
    if len(levels) == 1:
        return _leaf_from_substitutive_level(levels[0])
    return GSplatLodGroup(
        children=[_leaf_from_substitutive_level(lvl) for lvl in levels],
        default_level=default_substitutive,
    )


def substitutive_levels_from_tree(
    node: GSplatNode,
) -> "Tuple[List[SubstitutiveLevel], int]":
    """Project a matrix-shaped tree back to ``(substitutive_levels, default)``.

    Accepts the two matrix shapes produced by
    :func:`tree_from_substitutive_levels`:

    * a bare :class:`GSplatLeaf` → one substitutive level, default 0;
    * a :class:`GSplatLodGroup` whose children are all leaves → one level per
      child, default = ``group.default_level``.

    Raises :class:`ValueError` for genuinely non-matrix trees (partitions, or
    lod groups with non-leaf children) — those have no rectangular-matrix
    equivalent and must be consumed through the tree directly.
    """
    from luxar.gsplats.gsplat_data import SubstitutiveLevel

    def leaf_to_level(leaf: GSplatLeaf, fallback_index: int) -> "SubstitutiveLevel":
        meta = leaf.meta
        return SubstitutiveLevel(
            additive_sublods=list(leaf.additive_sublods),
            compression_factor=int(meta.get("compression_factor", 1)),
            parent_method=meta.get("parent_method", None),
            level_index=int(meta.get("level_index", fallback_index)),
            stats=dict(meta.get("stats", {})),
        )

    if isinstance(node, GSplatLeaf):
        return [leaf_to_level(node, 0)], 0

    if isinstance(node, GSplatLodGroup):
        if not all(isinstance(c, GSplatLeaf) for c in node.children):
            raise ValueError(
                "substitutive_levels_from_tree: lod group has non-leaf children "
                "(a nested tree has no flat substitutive-matrix equivalent)"
            )
        levels = [
            leaf_to_level(child, i)  # type: ignore[arg-type]
            for i, child in enumerate(node.children)
        ]
        return levels, node.default_level

    raise ValueError(
        f"substitutive_levels_from_tree: {type(node).__name__} has no "
        "rectangular-matrix equivalent (consume the tree directly)"
    )


def is_matrix_shaped(node: GSplatNode) -> bool:
    """True if ``node`` maps to a flat substitutive × additive matrix.

    I.e. a bare leaf, or a lod group whose children are all leaves.
    """
    if isinstance(node, GSplatLeaf):
        return True
    if isinstance(node, GSplatLodGroup):
        return all(isinstance(c, GSplatLeaf) for c in node.children)
    return False
