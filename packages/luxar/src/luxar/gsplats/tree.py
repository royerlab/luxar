"""Node-tree model for Gaussian splats.

This is the unified in-memory representation behind the v3.0 ``.gsplats.zarr``
format and the scene gsplat-node subtree: a standalone ``.gsplats.zarr`` *is* a
detached node subtree, and embedding it into a scene is a graft of that subtree.

Three node types compose freely (and nest arbitrarily):

* :class:`GSplatLeaf` — a gsplats leaf carrying an **additive ladder** (one or
  more :class:`~luxar.gsplats.gsplat_data.AdditiveSubLOD`, prefix-sum / additive
  LOD). The trivial single-splat-set case is a leaf with a one-entry ladder.
* :class:`GSplatLodGroup` — **substitutive** LOD: children are rendered one at a
  time (the scene ``kind=lod`` Group). Children are ordered **coarsest → finest**
  in memory — the SAME order as the on-disk ``child_<i>`` layout (child_0 =
  coarsest), so the serializer writes them straight through with no reversal.
  ``default_level`` is a derived property (= the finest, last child).
* :class:`GSplatPartition` — **spatial** split: all children are rendered (the
  scene ``kind=partition`` Group), each carrying its own ``position_bounds``.

The classes are intentionally small, pure, and immutable (frozen dataclasses) so
they are trivially unit-testable in isolation. Per-node metadata (LOD provenance
such as ``compression_factor`` / ``parent_method`` / ``level_index``, the
view-driven ``coverage_fraction`` selector threshold, per-node ``stats``) lives in a
free-form ``meta`` dict on each node — mirroring the zarr ``.zattrs`` a node
carries on disk.

The :func:`tree_from_substitutive_levels` / :func:`substitutive_levels_from_tree`
bridge converts to and from the **derived** 2-D ``substitutive × additive`` matrix
*view* (``GSplatData.substitutive_levels``, finest-first by convention). The tree
is the single in-memory ground truth (``GSplatData`` stores a node and derives the
matrix view on demand); the matrix is exactly one shape of the tree: a single
:class:`GSplatLodGroup` of leaves (or, for a single substitutive level, a bare
:class:`GSplatLeaf`). This bridge is the ONE place the coarsest-first tree order
is reversed to the finest-first matrix-view convention and back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from luxar.gsplats.utils.spatial_axes import (
    SPATIAL_SIGMA_EPS,
    spatial_axes_from_max_sigma,
)

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
        ``coverage_fraction`` (selector threshold when a child of a lod group),
        and ``stats``.
    """

    additive_sublods: "List[AdditiveSubLOD]"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.additive_sublods:
            raise ValueError("GSplatLeaf must contain at least one AdditiveSubLOD")
        ndims = {int(sub.ndim) for sub in self.additive_sublods}
        if len(ndims) > 1:
            raise ValueError(
                f"GSplatLeaf additive sub-LODs must share one dimensionality; "
                f"got mixed ndims {sorted(ndims)}. The leaf's ndim is read from "
                f"the first sub-LOD, so a mix would silently mis-describe the rest."
            )

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

    Children are ordered **coarsest → finest** in memory, matching the on-disk
    ``child_<i>`` layout (child_0 = coarsest) so the serializer needs no reversal.
    ``default_level`` is a derived property (= the finest, last child): the level
    a simple consumer renders by default. It is deliberately distinct from the
    on-disk ``default_level`` (a viewer progressive-load hint = coarsest), which
    the serializer stamps independently.
    """

    children: "List[GSplatNode]"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("GSplatLodGroup must contain at least one child")
        ndims = {int(c.ndim) for c in self.children}
        if len(ndims) > 1:
            raise ValueError(
                f"GSplatLodGroup children must share one dimensionality; "
                f"got mixed ndims {sorted(ndims)}. The group's ndim is read from "
                f"the first child, so a mix would silently mis-describe the rest."
            )

    @property
    def default_level(self) -> int:
        """The finest child's index (last entry, coarsest→finest order)."""
        return len(self.children) - 1

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
        ndims = {int(c.ndim) for c in self.children}
        if len(ndims) > 1:
            raise ValueError(
                f"GSplatPartition children must share one dimensionality; "
                f"got mixed ndims {sorted(ndims)}. The partition's ndim is read "
                f"from the first child, so a mix would silently mis-describe the rest."
            )

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
    subtree holds zero splats. This is a center-only bound. The serializer's
    ``position_bounds`` is the verbatim center bounds (matching the scene path);
    only ``chunk_bounds`` widen each chunk by the 3σ ellipsoidal extent.
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
# Structure-preserving map + default-selection global statistics
# ────────────────────────────────────────────────────────────────────────


def map_leaves(
    node: GSplatNode, fn: "Callable[[GSplatLeaf], GSplatNode]"
) -> GSplatNode:
    """Rebuild the tree with ``fn`` applied to every leaf, preserving its shape.

    Walks the (immutable, frozen) tree depth-first and returns a NEW tree of the
    same shape — same group kinds, ``GSplatPartition.max_elements``, and per-node
    ``meta`` — in which each :class:`GSplatLeaf` is replaced by ``fn(leaf)``
    (``fn`` typically returns a transformed leaf). This is the write-side
    workhorse for tree-aware ops (e.g. ``gsplat transform`` on a
    ``kind=partition``) that the flat :class:`~luxar.gsplats.gsplat_data.GSplatData`
    path — which only handles matrix-shaped trees — cannot express.
    """
    if isinstance(node, GSplatLeaf):
        return fn(node)
    if isinstance(node, GSplatLodGroup):
        return GSplatLodGroup(
            children=[map_leaves(c, fn) for c in node.children],
            meta=dict(node.meta),
        )
    if isinstance(node, GSplatPartition):
        return GSplatPartition(
            children=[map_leaves(c, fn) for c in node.children],
            max_elements=node.max_elements,
            meta=dict(node.meta),
        )
    raise TypeError(  # pragma: no cover - guards against an unknown node type
        f"Unknown gsplat node type: {type(node).__name__}"
    )


def without_meta_key(node: GSplatNode, key: str) -> GSplatNode:
    """Rebuild the tree with ``key`` removed from **every** node's ``meta``.

    Unlike :func:`map_leaves` (which copies group ``meta`` verbatim), this scrubs
    a key from leaves AND group nodes. Its use is dropping the ``coverage_fraction``
    LOD-switch threshold after a geometry transform so the writer re-derives it: a
    stale threshold on a *group* node (a ``multiscale`` partition child, or a
    ``mosaic`` per-part lod group) is otherwise re-applied verbatim by the
    serializer. (Coverage fractions are count-ratios, hence invariant to
    scale/rotate/translate — so this re-derives the same value; it is retained as a
    safety net for transforms that also re-ladder and change per-level counts.)
    """
    new_meta = {k: v for k, v in node.meta.items() if k != key}
    if isinstance(node, GSplatLeaf):
        return replace(node, meta=new_meta)
    if isinstance(node, GSplatLodGroup):
        return GSplatLodGroup(
            children=[without_meta_key(c, key) for c in node.children],
            meta=new_meta,
        )
    if isinstance(node, GSplatPartition):
        return GSplatPartition(
            children=[without_meta_key(c, key) for c in node.children],
            max_elements=node.max_elements,
            meta=new_meta,
        )
    raise TypeError(  # pragma: no cover - guards against an unknown node type
        f"Unknown gsplat node type: {type(node).__name__}"
    )


def iter_default_leaves(node: GSplatNode) -> Iterator[GSplatLeaf]:
    """Yield the leaves of the **default-rendered** selection.

    Mirrors the ``n_splats`` selection semantics: a partition renders all parts,
    but a substitutive lod group renders only its default (finest) child — so
    coarse substitutive levels (downsampled *representations* of the same splats)
    are skipped. Use this for global statistics (centroid, max amplitude) so the
    same splat is not double-counted across levels. (Contrast :func:`iter_leaves`,
    which yields every stored leaf regardless of LOD selection.)
    """
    if isinstance(node, GSplatLeaf):
        yield node
    elif isinstance(node, GSplatLodGroup):
        yield from iter_default_leaves(node.children[node.default_level])
    elif isinstance(node, GSplatPartition):
        for child in node.children:
            yield from iter_default_leaves(child)
    else:  # pragma: no cover - guards against an unknown node type
        raise TypeError(f"Unknown gsplat node type: {type(node).__name__}")


def amplitude_weighted_centroid(node: GSplatNode) -> Optional[np.ndarray]:
    """Global amplitude-weighted centroid over the default-rendered splat set.

    Returns the ``(d,)`` centroid (float64), or ``None`` for an empty tree.
    Falls back to the unweighted center mean when the total amplitude is zero —
    matching :meth:`~luxar.gsplats.gsplat_data.GSplatData.center_at_centroid` on
    a single leaf, so a matrix-shaped tree gives an identical result.
    """
    weighted: Optional[np.ndarray] = None  # Σ aᵢ·cᵢ
    sum_centers: Optional[np.ndarray] = None  # Σ cᵢ (unweighted fallback)
    total_amp = 0.0
    n = 0
    for leaf in iter_default_leaves(node):
        for sub in leaf.additive_sublods:
            if sub.n_splats == 0:
                continue
            c = sub.centers.astype(np.float64)
            a = sub.amplitudes.astype(np.float64)
            wc = c.T @ a
            sc = c.sum(axis=0)
            weighted = wc if weighted is None else weighted + wc
            sum_centers = sc if sum_centers is None else sum_centers + sc
            total_amp += float(a.sum())
            n += int(sub.n_splats)
    if sum_centers is None:
        return None
    if total_amp > 0 and weighted is not None:
        return weighted / total_amp
    return sum_centers / max(1, n)


def global_amplitude_max(node: GSplatNode) -> float:
    """Maximum amplitude over the default-rendered splat set (``0.0`` if empty)."""
    mx = 0.0
    for leaf in iter_default_leaves(node):
        for sub in leaf.additive_sublods:
            if sub.n_splats:
                mx = max(mx, float(sub.amplitudes.max()))
    return mx


def nondegenerate_axes(
    node: GSplatNode, eps: float = SPATIAL_SIGMA_EPS
) -> np.ndarray:
    """Axes with real covariance extent over the default-rendered splat set.

    The node-tree twin of ``GSplatData._nondegenerate_axes``: an axis is spatial
    if its maximum marginal sigma across the splats exceeds ``eps``; a
    zero-variance categorical axis (a stacked-time / channel axis) is excluded.
    Used by ``transform --center`` to re-origin only the spatial axes. Reduces
    to a per-axis max-sigma vector (via each leaf's ``marginal_sigmas``) and
    applies the shared spatial-axis rule. Falls back to all axes when none
    qualify (or the tree is empty).
    """
    max_sigma: Optional[np.ndarray] = None
    ndim = 0
    for leaf in iter_default_leaves(node):
        for sub in leaf.additive_sublods:
            if sub.n_splats == 0:
                continue
            ndim = int(sub.centers.shape[1])
            sig = sub.marginal_sigmas().max(axis=0)  # reuse the shared metric
            max_sigma = sig if max_sigma is None else np.maximum(max_sigma, sig)
    if max_sigma is None:
        return np.arange(ndim)
    return spatial_axes_from_max_sigma(max_sigma, eps)


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


def node_from_substitutive_levels(levels: "List[SubstitutiveLevel]") -> GSplatNode:
    """Build the tree shape from a finest-first matrix view — **no stamping**.

    The lightweight inverse of :func:`substitutive_levels_from_tree`: a single
    level → a bare :class:`GSplatLeaf`; multiple levels → a :class:`GSplatLodGroup`
    reversed to coarsest-first (matching disk). This is what ``GSplatData`` stores
    as its ground-truth node on construction — cheap, with no ``coverage_fraction``
    derivation (the view-driven thresholds are a serialize-time concern, stamped
    by :func:`tree_from_substitutive_levels` / re-derived by the writer).
    """
    if not levels:
        raise ValueError("levels must contain at least one SubstitutiveLevel")
    if len(levels) == 1:
        return _leaf_from_substitutive_level(levels[0])
    # ``levels`` is the finest-first matrix view; the tree stores coarsest-first.
    return GSplatLodGroup(
        children=[_leaf_from_substitutive_level(lvl) for lvl in reversed(levels)]
    )


def tree_from_substitutive_levels(
    levels: "List[SubstitutiveLevel]",
) -> GSplatNode:
    """Build a node tree from the historical 2-D matrix representation.

    * A single substitutive level → a bare :class:`GSplatLeaf` (its additive
      ladder), carrying that level's provenance in ``meta``.
    * Multiple substitutive levels → a :class:`GSplatLodGroup` of one leaf per
      level, **reversed to coarsest-first** (``levels`` is the finest-first matrix
      view; the tree stores coarsest-first to match disk). The in-memory
      ``default_level`` is the derived finest (last) child; the persisted on-disk
      ``default_level`` is the viewer's coarsest-first render hint (stamped by the
      serializer), a separate concept.

    Each child of a multi-level lod group is back-filled with a derived
    ``coverage_fraction`` selector threshold (``sqrt(N_i/N_finest)`` — the
    viewport-relative fraction the viewer multiplies by the viewport diagonal), so
    a standalone substitutive ``.gsplats.zarr`` selects levels correctly in the
    viewer rather than being stuck at the finest level. This is the same
    single-sourced :func:`~luxar.core.group.lod.group.coverage_fractions`
    derivation the scene path uses.

    This is the inverse of :func:`substitutive_levels_from_tree` for any tree
    that is matrix-shaped (a leaf, or a lod group whose children are all leaves).
    """
    node = node_from_substitutive_levels(levels)
    if isinstance(node, GSplatLeaf):
        return node

    # Back-fill per-child coverage_fraction. The derivation is single-sourced in
    # core (coarsest child = 0.0, finest = 1.0, ascending) and uses only per-level
    # splat-count ratios. Children and counts are both coarsest-first — a straight
    # 1:1 mapping.
    from luxar.core.group.lod.group import coverage_fractions

    levels_coarsest_first = list(reversed(levels))
    counts_coarsest_first = [
        sum(sub.n_splats for sub in lvl.additive_sublods)
        for lvl in levels_coarsest_first
    ]
    fractions_coarsest_first = coverage_fractions(counts_coarsest_first)
    for leaf, fraction in zip(node.children, fractions_coarsest_first):
        leaf.meta.setdefault("coverage_fraction", fraction)

    return node


def substitutive_levels_from_tree(
    node: GSplatNode,
) -> "Tuple[List[SubstitutiveLevel], int]":
    """Project a matrix-shaped tree back to ``(substitutive_levels, default)``.

    Accepts the two matrix shapes produced by
    :func:`tree_from_substitutive_levels`:

    * a bare :class:`GSplatLeaf` → one substitutive level, default 0;
    * a :class:`GSplatLodGroup` whose children are all leaves → one level per
      child, **reversed** from the tree's coarsest-first order to the matrix
      view's finest-first convention (index 0 = finest). The returned default is
      always 0 (the matrix view's finest), distinct from the tree's coarsest-first
      on-disk hint.

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
        # Tree children are coarsest-first; the matrix view is finest-first.
        levels = [
            leaf_to_level(child, i)  # type: ignore[arg-type]
            for i, child in enumerate(reversed(node.children))
        ]
        # The matrix-view default is always the finest (index 0); the tree's
        # coarsest-first default_level is a separate (viewer) concept.
        return levels, 0

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
