"""LOD structure / accessor mixin for ``GSplatData``.

The ``substitutive × additive`` matrix views over the ground-truth node tree,
the per-level constructors, and the node-tree bridge (:attr:`tree` /
:meth:`from_tree`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from .base import _GSplatDataOps, _readonly, _readonly_opt, _readonly_sublod
from .filtering import (
    _refresh_reduction_lod_stats_if_needed,
    drop_content_scoped_stats,
)

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )
    from luxar.gsplats.tree import GSplatLeaf, GSplatNode


class LODViewsMixin(_GSplatDataOps):
    """The LOD tree structure/accessor family — matrix views, per-level
    constructors, prefixes/flattening, and the node-tree bridge."""

    def _finest_leaf(self) -> "GSplatLeaf":
        """The finest :class:`GSplatLeaf` of the matrix-shaped ground-truth node.

        The node is coarsest-first, so the finest level is a bare leaf itself or
        the last child of the lod group.
        """
        from luxar.gsplats.tree import GSplatLeaf

        node = self._node
        if isinstance(node, GSplatLeaf):
            return node
        return node.children[-1]  # type: ignore[return-value]

    # ── Derived matrix views over the ground-truth node (finest-first) ──────

    @property
    def substitutive_levels(self) -> List["SubstitutiveLevel"]:
        """Finest-first substitutive × additive matrix view, derived from the node.

        Reconstructed on access from ``self._node`` (the single ground truth). Index
        0 is the finest level — the historical matrix convention — independent of
        the node's coarsest-first storage order.
        """
        from luxar.gsplats.tree import substitutive_levels_from_tree

        return substitutive_levels_from_tree(self._node)[0]

    @property
    def additive_sublods(self) -> List["AdditiveSubLOD"]:
        """The finest level's additive ladder (the "primary" sub-LODs).

        Returns a fresh list (the ``AdditiveSubLOD`` elements are shared) so a
        caller mutating it cannot corrupt the ground-truth node or desync the
        cached ``centers``/``n_splats`` — matching the pre-refactor defensive copy
        and the sibling ``substitutive_levels`` view's semantics.
        """
        return list(self._finest_leaf().additive_sublods)

    @property
    def default_substitutive(self) -> int:
        """Index of the data-model default substitutive level (finest = 0).

        Fixed in the finest-first matrix view; not settable. Distinct from the
        on-disk ``default_level`` (the viewer's coarsest-first progressive-load hint).
        """
        return 0

    # ── LOD-specific methods ───────────────────────────────

    @property
    def n_additive_sublods(self) -> int:
        """Number of LOD levels."""
        return len(self.additive_sublods)

    def additive_sublod(self, level: int) -> "AdditiveSubLOD":
        """Return the AdditiveSubLOD at the given level.

        Args:
            level: LOD level index (0 = coarsest).
        """
        return self.additive_sublods[level]

    def additive_prefix(self, level: int) -> "GSplatData":
        """Return a new GSplatData with LODs 0 through ``level`` (inclusive).

        The returned object's arrays are read-only zero-copy views of this
        one's (the class is conceptually immutable); mutating them raises
        rather than silently corrupting the source.

        A STRICT prefix holds fewer splats than the object the INHERITED top-level
        measured scores were taken on, so the view does not carry them (#1600): a
        view is a reduction like any other. The full prefix (``level == n - 1``) IS
        the input content and keeps everything. Only the view's own top-level dict
        is scrubbed — each rung's ``stats`` (its own ladder PSNR, its e(k)) is a
        statement about that rung, which the prefix still holds unchanged.

        Args:
            level: Maximum LOD level to include (``0 <= level < n_additive_sublods``).

        Returns:
            New GSplatData with ``level + 1`` LODs.

        Raises:
            IndexError: If ``level`` is out of range.
        """
        from luxar.gsplats.gsplat_data import GSplatData

        n = self.n_additive_sublods
        if not 0 <= level < n:
            raise IndexError(f"additive level {level} out of range [0, {n})")
        view = GSplatData(
            additive_sublods=[
                _readonly_sublod(lod) for lod in self.additive_sublods[: level + 1]
            ],
            stats=dict(self.stats),
        )
        if level + 1 < n:
            drop_content_scoped_stats(view.stats)
            view = _refresh_reduction_lod_stats_if_needed(view, self)
        return view

    def flattened(self) -> "GSplatData":
        """Collapse all LODs into a single LOD.

        The returned object's arrays are read-only zero-copy views of this
        one's, honouring the immutability contract: mutating them raises
        rather than silently corrupting the source.

        Returns:
            New GSplatData with ``n_additive_sublods == 1`` containing all splats.
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        single = AdditiveSubLOD(
            centers=_readonly(self.centers),
            amplitudes=_readonly(self.amplitudes),
            cholesky_factors=_readonly(self.cholesky_factors),
            colors=_readonly_opt(self.colors),
            label_ids=_readonly_opt(self.label_ids),
            label_vocabulary=(
                dict(self.label_vocabulary)
                if self.label_vocabulary is not None
                else None
            ),
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )
        return GSplatData(additive_sublods=[single], stats=dict(self.stats))

    def lod_psnrs(self) -> list[float]:
        """Extract cumulative PSNR from each LOD's stats.

        Returns:
            List of PSNR values (one per LOD). NaN if not available.
        """
        return [
            float(lod.stats.get("cumulative_psnr_db", float("nan")))
            for lod in self.additive_sublods
        ]

    @classmethod
    def from_additive_sublods(
        cls,
        additive_sublods: List["AdditiveSubLOD"],
        stats: Optional[Dict[str, Any]] = None,
    ) -> "GSplatData":
        """Construct a GSplatData from a list of additive sub-LODs.

        The result has ``n_substitutive == 1`` (single substitutive level)
        whose additive ladder is the given list.

        Args:
            additive_sublods: List of AdditiveSubLOD (at least one).
            stats: Optional top-level statistics.
        """
        return cast("type[GSplatData]", cls)(
            additive_sublods=additive_sublods, stats=stats
        )

    @classmethod
    def from_substitutive_levels(
        cls,
        substitutive_levels: List["SubstitutiveLevel"],
        stats: Optional[Dict[str, Any]] = None,
    ) -> "GSplatData":
        """Construct a 2-D GSplatData from a list of substitutive levels.

        Each ``SubstitutiveLevel`` carries its own additive ladder (one or
        more :class:`AdditiveSubLOD`). The resulting ``GSplatData`` has
        ``n_substitutive == len(substitutive_levels)`` and represents the
        full ``[N, M_i]`` matrix of splat sets. The accessors
        (``.centers``/``.additive_sublods``/…) always return the FINEST level
        (index 0) — the data-model default is fixed, not settable (see
        ``__init__``).

        Args:
            substitutive_levels: Ordered list, finest at index 0.
            stats: Optional top-level statistics.

        Returns:
            New ``GSplatData`` with the given substitutive × additive matrix.
        """
        return cast("type[GSplatData]", cls)(
            substitutive_levels=substitutive_levels,
            stats=stats,
        )

    # ── 2-D substitutive × additive accessors ──────────────

    @property
    def n_substitutive(self) -> int:
        """Number of substitutive levels (always >= 1)."""
        from luxar.gsplats.tree import GSplatLeaf

        node = self._node
        return 1 if isinstance(node, GSplatLeaf) else node.n_children

    def at_substitutive(self, level: int) -> "GSplatData":
        """Return a single-substitutive-level view as a new ``GSplatData``.

        The returned object has ``n_substitutive == 1`` and its lone
        substitutive level carries the additive ladder of ``self``'s level
        ``level``. Useful for operating one substitutive level at a time
        (e.g., ``data.at_substitutive(s).flattened()``).

        A COARSER level (``level > 0``) is a different, MERGED splat set, so the
        view does not inherit the top-level measured reconstruction scores (#1600)
        — this is the chokepoint through which ``lod --recipe overview`` builds its
        merged coarse cap (as ``at_substitutive(n - 1).flattened()``) and published
        the input fit's ``psnr_db`` on it. Level 0 is the finest content itself and
        keeps them. (``_view_of_level`` itself does not scrub: ``_map_substitutive``
        walks every level through it and discards the view's top-level stats, so
        only the callers that know the index can tell a reduction from a rebuild
        step.)

        The level's OWN stamps are untouched — its ``level_stats`` Q / w and each
        rung's e(k) are measured on this level's content, not inherited from the
        finest, and this method is a plain accessor on the scene-authoring path
        (``lod_dispatch.py`` builds every coarse child of a ``kind=lod`` group with
        ``at_substitutive(s)`` and copies those numbers onto it).

        Args:
            level: Substitutive level index (0 = finest).
        """
        if not 0 <= level < self.n_substitutive:
            raise IndexError(
                f"substitutive level {level} out of range [0, {self.n_substitutive})"
            )
        view = self._view_of_level(self.substitutive_levels[level])
        if level != 0:
            drop_content_scoped_stats(view.stats)
        return view

    def _view_of_level(self, src_level: "SubstitutiveLevel") -> "GSplatData":
        """Wrap an already-fetched ``SubstitutiveLevel`` as a single-level view.

        Extracted from :meth:`at_substitutive` so per-level loops can pass the
        level they already hold (from one ``substitutive_levels`` read) instead of
        re-indexing the property — which would reconstruct the whole matrix each
        call, making an L-level map O(L²).
        """
        from luxar.gsplats.gsplat_data import GSplatData, SubstitutiveLevel

        ro_level = SubstitutiveLevel(
            additive_sublods=[
                _readonly_sublod(lod) for lod in src_level.additive_sublods
            ],
            compression_factor=src_level.compression_factor,
            parent_method=src_level.parent_method,
            level_index=src_level.level_index,
            stats=src_level.stats,
        )
        return GSplatData(
            substitutive_levels=[ro_level],
            stats=dict(self.stats),
        )

    # ── Node-tree bridge (v3.0 unified representation) ──────

    @property
    def tree(self) -> "GSplatNode":
        """This dataset as a :mod:`luxar.gsplats.tree` node subtree.

        The tree is the single in-memory ground truth (this just returns the
        stored node), behind the v3.0 ``.gsplats.zarr`` format and the scene
        gsplat-node subtree. For the matrix shape it is one of: a single
        :class:`~luxar.gsplats.tree.GSplatLeaf` (one substitutive level) or a
        :class:`~luxar.gsplats.tree.GSplatLodGroup` of leaves (multiple levels,
        coarsest first). Per-level provenance rides in each leaf's ``meta``. The
        view-driven ``coverage_fraction`` thresholds are derived at serialize time
        (see :meth:`save` / the writer), not stored here.
        """
        return self._node

    @classmethod
    def from_tree(
        cls,
        node: "GSplatNode",
        stats: Optional[Dict[str, Any]] = None,
    ) -> "GSplatData":
        """Construct a ``GSplatData`` from a matrix-shaped tree node.

        Accepts a bare :class:`~luxar.gsplats.tree.GSplatLeaf` or a
        :class:`~luxar.gsplats.tree.GSplatLodGroup` of leaves (the inverse of
        :attr:`tree`). Genuinely nested trees (partitions, or lod groups with
        non-leaf children) have no flat ``GSplatData`` equivalent and raise —
        they must be consumed through the tree directly.
        """
        from luxar.gsplats.tree import is_matrix_shaped

        if not is_matrix_shaped(node):
            raise ValueError(
                "GSplatData.from_tree: node is not matrix-shaped (a partition, or "
                "a lod group with non-leaf children, has no flat GSplatData "
                "equivalent — consume the tree directly)"
            )
        # Store the node verbatim as the ground truth — preserves its authored
        # meta (e.g. coverage_fraction straight off disk) and avoids a needless
        # matrix round-trip. The matrix views derive finest-first on access.
        return cast("type[GSplatData]", cls)(_node=node, stats=stats)
