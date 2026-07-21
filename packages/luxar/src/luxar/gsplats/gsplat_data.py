"""Gaussian Splat data container."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence

import numpy as np

from luxar.gsplats._data.culling import CullingMixin
from luxar.gsplats._data.filtering import FilteringMixin
from luxar.gsplats._data.io_adapter import IOAdapterMixin
from luxar.gsplats._data.metrics import _SplatArrayMixin
from luxar.gsplats._data.render import RenderMixin

# center_at_centroid (kept on GSplatData) shifts only the spatial axes (#487).
from luxar.gsplats.utils.spatial_axes import spatial_only_shift

if TYPE_CHECKING:
    from luxar.gsplats.tree import GSplatLeaf, GSplatNode, GSplatPartition


def widen_colors_to_rgba(colors: np.ndarray) -> np.ndarray:
    """Widen an (N, 3) RGB colors array to (N, 4) RGBA with opaque alpha.

    No-op (returns the input) when the array already has 4 channels. Integer
    arrays get their dtype's max as "opaque"; floats get 1.0.
    """
    if colors.shape[1] == 4:
        return colors
    opaque = (
        np.iinfo(colors.dtype).max if np.issubdtype(colors.dtype, np.integer) else 1.0
    )
    alpha = np.full((colors.shape[0], 1), opaque, dtype=colors.dtype)
    return np.concatenate([colors, alpha], axis=1)


def _merge_lod_colors(
    lods: "list[AdditiveSubLOD]",
) -> "Optional[np.ndarray]":
    """Merge colors from multiple LODs/datasets using None/all/mixed logic.

    - All have colors → concatenate.
    - All None → return None.
    - Mixed → fill missing with white (1,1,1).
    - Mixed RGB/RGBA channel counts → RGB parts widen to RGBA with alpha=1
      (opaque, the per-element-opacity identity).

    The merge works in **float32**: every part is coerced to float32 and any
    integer part is normalized by its full-scale (÷max) first, so the alpha
    column can never inject an out-of-``[0, 1]`` value (a uint8 opaque widened
    to alpha=255 concatenated with a float [0,1] part would otherwise promote
    to a float ``255.0`` alpha) and colors match the float32 dtype the sibling
    arrays (centers/amplitudes/cholesky) are pinned to in
    :func:`_concat_additive_levels`. Live producers are all float32 already, so
    this is a no-op on every current path and a guard for future integer ones.
    """
    if not lods:
        return None
    has_colors = [lod.colors is not None for lod in lods]
    if not any(has_colors):
        return None
    channels = max(lod.colors.shape[1] for lod in lods if lod.colors is not None)
    parts: list[np.ndarray] = []
    for lod in lods:
        colors = lod.colors
        if colors is None:
            colors = np.ones((lod.n_splats, channels), dtype=np.float32)
        else:
            if np.issubdtype(colors.dtype, np.integer):
                colors = colors.astype(np.float32) / np.iinfo(colors.dtype).max
            else:
                colors = colors.astype(np.float32, copy=False)
            if channels == 4:
                colors = widen_colors_to_rgba(colors)  # float → opaque alpha 1.0
        parts.append(colors)
    merged: np.ndarray = np.concatenate(parts, axis=0)
    return merged


def _readonly(arr: np.ndarray) -> np.ndarray:
    """Return a zero-copy, non-writable view of ``arr``.

    The returned view shares ``arr``'s buffer but cannot be written through,
    so a caller mutating it raises instead of silently corrupting the source.
    Marking the *view* read-only leaves the original array writable.
    """
    view: np.ndarray = arr.view()
    view.flags.writeable = False
    return view


def _readonly_opt(arr: "Optional[np.ndarray]") -> "Optional[np.ndarray]":
    """Read-only view of an optional array (passes ``None`` through)."""
    return None if arr is None else _readonly(arr)


def _readonly_sublod(lod: "AdditiveSubLOD") -> "AdditiveSubLOD":
    """Rebuild ``lod`` as a fully detached, read-only view.

    Arrays become zero-copy read-only views; ``stats`` is shallow-copied so the
    view is immutable through-and-through. (A bare ``stats=lod.stats`` alias would
    let a caller mutate the source node's ``lod_stats`` via a "read-only" view —
    the same aliasing hazard the array views guard against.)
    """
    return AdditiveSubLOD(
        centers=_readonly(lod.centers),
        amplitudes=_readonly(lod.amplitudes),
        cholesky_factors=_readonly(lod.cholesky_factors),
        colors=_readonly_opt(lod.colors),
        stats=dict(lod.stats),
        truncation_radius=lod.truncation_radius,
    )


def _concat_additive_levels(
    views: "list[GSplatData]",
) -> "list[AdditiveSubLOD]":
    """Concatenate the additive ladders of several views into one ladder.

    Each view is treated through its default substitutive level. Ragged
    additive counts are handled (a view that lacks level ``k`` simply does
    not contribute to it). Arrays are kept float32 so a mixed-precision
    input cannot silently promote the merged result to float64.
    """
    max_lods = max(v.n_additive_sublods for v in views)
    merged: "list[AdditiveSubLOD]" = []
    for level in range(max_lods):
        level_lods = [
            v.additive_sublod(level) for v in views if level < v.n_additive_sublods
        ]
        merged.append(
            AdditiveSubLOD(
                centers=np.concatenate(
                    [lod.centers for lod in level_lods], axis=0
                ).astype(np.float32, copy=False),
                amplitudes=np.concatenate(
                    [lod.amplitudes for lod in level_lods]
                ).astype(np.float32, copy=False),
                cholesky_factors=np.concatenate(
                    [lod.cholesky_factors for lod in level_lods], axis=0
                ).astype(np.float32, copy=False),
                colors=_merge_lod_colors(level_lods),
                stats={"lod_level": level, "n_sources": len(level_lods)},
                truncation_radius=level_lods[0].truncation_radius,
            )
        )
    return merged


@dataclass(frozen=True, eq=False)
class AdditiveSubLOD(_SplatArrayMixin):
    """A single Level-of-Detail layer — immutable container for splat arrays.

    Attributes
    ----------
    centers : np.ndarray, shape (N, d)
        Splat center positions.
    amplitudes : np.ndarray, shape (N,)
        Non-negative splat amplitudes.
    cholesky_factors : np.ndarray, shape (N, d*(d+1)//2)
        Packed lower-triangular Cholesky factors.
    colors : Optional[np.ndarray], shape (N, 3) or (N, 4)
        Optional RGB(A) colors per splat. The optional alpha channel is
        per-splat opacity in [0, 1] (consumed by every blending mode; mapped
        into optical depth in volumetric — see VOLUMETRIC_BLENDING_SPEC.md).
    stats : Dict[str, Any]
        Per-LOD statistics (e.g., psnr_db, time_seconds, pass_index).
    truncation_radius : float
        Gaussian truncation radius in standard deviations. Controls the shifted
        Gaussian formula: C = exp(-0.5 * T²), scale = 1/(1-C). Stored in zarr
        metadata and propagated to the viewer for consistent rendering.
    """

    centers: np.ndarray
    amplitudes: np.ndarray
    cholesky_factors: np.ndarray
    colors: Optional[np.ndarray] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    truncation_radius: float = 3.0

    def __post_init__(self) -> None:
        """Validate array shape consistency."""
        n = self.centers.shape[0]
        if self.amplitudes.shape != (n,):
            raise ValueError(
                f"Amplitudes shape {self.amplitudes.shape} doesn't match "
                f"centers count ({n},)"
            )
        if self.colors is not None and self.colors.shape[0] != n:
            raise ValueError(
                f"Colors count {self.colors.shape[0]} doesn't match centers count {n}"
            )
        if self.centers.ndim >= 2:
            from luxar.gsplats.utils.trils import validate_cholesky_shape

            validate_cholesky_shape(
                self.cholesky_factors,
                ndim=self.centers.shape[1],
                n_splats=n,
                allow_uniform=False,
            )

    def __repr__(self) -> str:
        n = self.n_splats
        ndim = self.ndim
        if n > 0:
            amp_range = f"[{float(self.amplitudes.min()):.4g}, {float(self.amplitudes.max()):.4g}]"
        else:
            amp_range = "[]"
        return f"AdditiveSubLOD({n:,} splats, {ndim}D, amplitudes={amp_range})"


@dataclass(frozen=True, eq=False)
class SubstitutiveLevel:
    """One level of a substitutive-LOD ladder — a self-contained splat set.

    Skeleton for the v2.0 2-D LOD model. Each ``SubstitutiveLevel`` carries
    an additive ladder of its own; ``GSplatData`` holds an ordered list of
    these levels (finest at index 0). Substitutive levels operate "in
    parallel" — each is a distinct splat set that *replaces* (not extends)
    finer-resolution levels at render time.

    Attributes
    ----------
    additive_sublods : list[AdditiveSubLOD]
        The additive ladder *within* this substitutive level. Always ≥ 1
        entry; a single entry means "no additive sub-ordering at this level".
    compression_factor : int
        1 for the finest level (= original splats); K, K², … for coarser
        levels (where K is the substitutive compression factor).
    parent_method : str | None
        How this level was constructed from the next-finer one:
        ``"kmeans_lloyd"``, ``"greedy"``, etc. ``None`` for the finest level
        (no parent).
    level_index : int
        Redundant convenience: this level's index inside its parent
        ``GSplatData``. Finest = 0.
    stats : dict
        Per-level metadata (``psnr_estimate``, ``n_splats_total``, etc.).
    """

    additive_sublods: List[AdditiveSubLOD]
    compression_factor: int = 1
    parent_method: Optional[str] = None
    level_index: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.additive_sublods:
            raise ValueError(
                "SubstitutiveLevel must contain at least one AdditiveSubLOD"
            )
        for sub in self.additive_sublods:
            if not isinstance(sub, AdditiveSubLOD):
                raise TypeError(
                    f"Each entry must be an AdditiveSubLOD, got {type(sub).__name__}"
                )
        if self.compression_factor < 1:
            raise ValueError(
                f"compression_factor must be >= 1, got {self.compression_factor}"
            )

    @property
    def n_additive_lods(self) -> int:
        """Number of additive sub-LODs in this substitutive level (≥ 1)."""
        return len(self.additive_sublods)

    @property
    def n_splats_total(self) -> int:
        """Sum of ``n_splats`` across this level's additive sub-LODs."""
        return sum(sub.n_splats for sub in self.additive_sublods)


class GSplatData(RenderMixin, IOAdapterMixin, FilteringMixin, CullingMixin):
    """Container for Gaussian splat data with always-LOD structure.

    Every ``GSplatData`` holds one or more LOD levels (``AdditiveSubLOD`` instances).
    A single-LOD dataset is simply ``additive_sublods=[one_lod]``.

    **Construction styles**::

        # Convenience constructor (wraps into single LOD internally):
        GSplatData(centers=c, amplitudes=a, cholesky_factors=cf)

        # Explicit LOD construction:
        GSplatData.from_additive_sublods([lod0, lod1, lod2])

    Top-level ``centers``, ``amplitudes``, ``cholesky_factors``, and ``colors``
    are the concatenation of all LODs, computed once at construction time.
    The object is conceptually immutable — all operations return new instances.

    Attributes
    ----------
    additive_sublods : List[AdditiveSubLOD]
        Additive sub-LODs of the default substitutive level. Always >= 1.
    centers : np.ndarray, shape (N_total, d)
        Cached concatenation of all LOD centers.
    amplitudes : np.ndarray, shape (N_total,)
        Cached concatenation of all LOD amplitudes.
    cholesky_factors : np.ndarray, shape (N_total, tril)
        Cached concatenation of all LOD Cholesky factors.
    colors : Optional[np.ndarray], shape (N_total, 3) or (N_total, 4)
        Cached concatenation of all LOD colors (None if no LOD has colors).
        The optional 4th column is per-splat opacity alpha in [0, 1].
    stats : Dict[str, Any]
        Top-level statistics (overall quality, timing, etc.).
    """

    def __init__(
        self,
        centers: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
        cholesky_factors: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        stats: Optional[Dict[str, Any]] = None,
        *,
        additive_sublods: Optional[List[AdditiveSubLOD]] = None,
        substitutive_levels: Optional[List[SubstitutiveLevel]] = None,
        truncation_radius: float = 3.0,
        _node: "Optional[GSplatNode]" = None,
    ) -> None:
        """Build the single in-memory ground truth: a matrix-shaped node tree.

        The historical ``substitutive_levels`` / ``additive_sublods`` matrix API is
        preserved as **derived finest-first views** over ``self._node`` (which is
        stored coarsest-first, matching disk). ``_node`` is the internal fast path
        (used by :meth:`from_tree`) that stores a pre-built node verbatim.
        """
        from luxar.gsplats.tree import (
            GSplatLeaf,
            node_from_substitutive_levels,
        )

        if _node is not None:
            # Internal: store a pre-built matrix-shaped node verbatim (preserves
            # authored meta such as coverage_fraction — e.g. straight off disk).
            self._node: "GSplatNode" = _node
        elif substitutive_levels is not None:
            # 2-D construction: full substitutive × additive matrix (finest-first).
            if len(substitutive_levels) == 0:
                raise ValueError(
                    "substitutive_levels must contain at least one SubstitutiveLevel"
                )
            for s in substitutive_levels:
                if not isinstance(s, SubstitutiveLevel):
                    raise TypeError(
                        f"Each entry must be a SubstitutiveLevel, got "
                        f"{type(s).__name__}"
                    )
            self._node = node_from_substitutive_levels(list(substitutive_levels))
        elif additive_sublods is not None:
            # Single-substitutive construction with explicit additive sub-LODs
            if len(additive_sublods) == 0:
                raise ValueError(
                    "additive_sublods must contain at least one AdditiveSubLOD"
                )
            for sub in additive_sublods:
                if not isinstance(sub, AdditiveSubLOD):
                    raise TypeError(
                        f"Each additive sub-LOD must be an AdditiveSubLOD, got "
                        f"{type(sub).__name__}"
                    )
            self._node = GSplatLeaf(additive_sublods=list(additive_sublods))
        elif (
            centers is not None
            and amplitudes is not None
            and cholesky_factors is not None
        ):
            # Convenience constructor — wrap into a single-LOD leaf.
            single_lod = AdditiveSubLOD(
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky_factors,
                colors=colors,
                stats=stats if stats is not None else {},
                truncation_radius=truncation_radius,
            )
            self._node = GSplatLeaf(additive_sublods=[single_lod])
        else:
            raise ValueError(
                "Provide either substitutive_levels=[...], "
                "additive_sublods=[...], or (centers, amplitudes, cholesky_factors)"
            )

        # Cached concatenations from the FINEST leaf's additive ladder (the
        # ``.centers``/etc. accessors return the finest/full-resolution level).
        finest_sublods = self._finest_leaf().additive_sublods
        if len(finest_sublods) == 1:
            # Fast path: single LOD, no copy
            lod0 = finest_sublods[0]
            self.centers = lod0.centers
            self.amplitudes = lod0.amplitudes
            self.cholesky_factors = lod0.cholesky_factors
            self.colors = lod0.colors
        else:
            self.centers = np.concatenate(
                [lod.centers for lod in finest_sublods], axis=0
            )
            self.amplitudes = np.concatenate([lod.amplitudes for lod in finest_sublods])
            self.cholesky_factors = np.concatenate(
                [lod.cholesky_factors for lod in finest_sublods], axis=0
            )
            self.colors = _merge_lod_colors(finest_sublods)

        # Top-level stats (separate from per-LOD stats)
        if stats is not None:
            self.stats: Dict[str, Any] = stats
        elif (
            _node is not None
            or additive_sublods is not None
            or substitutive_levels is not None
        ):
            # When constructed via a node / additive_sublods= / substitutive_levels=,
            # start with empty top-level stats
            self.stats = {}
        else:
            # Convenience constructor already set stats on the LOD; mirror it
            self.stats = dict(finest_sublods[0].stats)

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
    def substitutive_levels(self) -> List[SubstitutiveLevel]:
        """Finest-first substitutive × additive matrix view, derived from the node.

        Reconstructed on access from ``self._node`` (the single ground truth). Index
        0 is the finest level — the historical matrix convention — independent of
        the node's coarsest-first storage order.
        """
        from luxar.gsplats.tree import substitutive_levels_from_tree

        return substitutive_levels_from_tree(self._node)[0]

    @property
    def additive_sublods(self) -> List[AdditiveSubLOD]:
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

    @property
    def truncation_radius(self) -> float:
        """Gaussian truncation radius in standard deviations (from first LOD)."""
        return self.additive_sublods[0].truncation_radius

    def __repr__(self) -> str:
        """Summary representation (avoids dumping full arrays)."""
        n = self.n_splats
        ndim = self.ndim
        if n > 0:
            amp_range = f"[{float(self.amplitudes.min()):.4g}, {float(self.amplitudes.max()):.4g}]"
        else:
            amp_range = "[]"
        colors = "yes" if self.colors is not None else "no"
        lod_str = (
            f", {self.n_additive_sublods} LODs" if self.n_additive_sublods > 1 else ""
        )
        return (
            f"GSplatData({n:,} splats, {ndim}D, "
            f"amplitudes={amp_range}, colors={colors}{lod_str})"
        )

    # ── LOD-specific methods ───────────────────────────────

    @property
    def n_additive_sublods(self) -> int:
        """Number of LOD levels."""
        return len(self.additive_sublods)

    def additive_sublod(self, level: int) -> AdditiveSubLOD:
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

        Args:
            level: Maximum LOD level to include (``0 <= level < n_additive_sublods``).

        Returns:
            New GSplatData with ``level + 1`` LODs.

        Raises:
            IndexError: If ``level`` is out of range.
        """
        n = self.n_additive_sublods
        if not 0 <= level < n:
            raise IndexError(f"additive level {level} out of range [0, {n})")
        return GSplatData(
            additive_sublods=[
                _readonly_sublod(lod) for lod in self.additive_sublods[: level + 1]
            ],
            stats=dict(self.stats),
        )

    def flattened(self) -> "GSplatData":
        """Collapse all LODs into a single LOD.

        The returned object's arrays are read-only zero-copy views of this
        one's, honouring the immutability contract: mutating them raises
        rather than silently corrupting the source.

        Returns:
            New GSplatData with ``n_additive_sublods == 1`` containing all splats.
        """
        single = AdditiveSubLOD(
            centers=_readonly(self.centers),
            amplitudes=_readonly(self.amplitudes),
            cholesky_factors=_readonly(self.cholesky_factors),
            colors=_readonly_opt(self.colors),
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
        additive_sublods: List[AdditiveSubLOD],
        stats: Optional[Dict[str, Any]] = None,
    ) -> "GSplatData":
        """Construct a GSplatData from a list of additive sub-LODs.

        The result has ``n_substitutive == 1`` (single substitutive level)
        whose additive ladder is the given list.

        Args:
            additive_sublods: List of AdditiveSubLOD (at least one).
            stats: Optional top-level statistics.
        """
        return cls(additive_sublods=additive_sublods, stats=stats)

    @classmethod
    def from_substitutive_levels(
        cls,
        substitutive_levels: List[SubstitutiveLevel],
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
        return cls(
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

        Args:
            level: Substitutive level index (0 = finest).
        """
        if not 0 <= level < self.n_substitutive:
            raise IndexError(
                f"substitutive level {level} out of range [0, {self.n_substitutive})"
            )
        return self._view_of_level(self.substitutive_levels[level])

    def _view_of_level(self, src_level: SubstitutiveLevel) -> "GSplatData":
        """Wrap an already-fetched ``SubstitutiveLevel`` as a single-level view.

        Extracted from :meth:`at_substitutive` so per-level loops can pass the
        level they already hold (from one ``substitutive_levels`` read) instead of
        re-indexing the property — which would reconstruct the whole matrix each
        call, making an L-level map O(L²).
        """
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
        return cls(_node=node, stats=stats)

    # ── Combine / Partition / Embed ─────────────────────────────

    @classmethod
    def concatenate(cls, datasets: list["GSplatData"]) -> "GSplatData":
        """Concatenate multiple GSplatData objects into one.

        All datasets must share the same dimensionality, truncation radius,
        and number of substitutive levels. The full 2-D LOD matrix is
        preserved: merging is done per ``(substitutive, additive)`` cell, so
        concatenating pyramids yields a pyramid (no level is silently
        dropped). To merge across a mismatched substitutive hierarchy,
        ``flattened()`` the inputs first.

        Colors: if all have colors, concatenate; if all None, None;
        if mixed, fill missing with white (1,1,1).

        Args:
            datasets: List of GSplatData (same ndim, truncation_radius, and
                n_substitutive required).

        Returns:
            New GSplatData with all splats concatenated per LOD cell.

        Raises:
            ValueError: On empty input list, or mismatched ndim /
                truncation_radius / n_substitutive across datasets.
        """
        if len(datasets) == 0:
            raise ValueError("At least one GSplatData is required")

        # Filter out empty datasets to avoid shape mismatch in np.concatenate
        non_empty = [d for d in datasets if d.n_splats > 0]
        if len(non_empty) == 0:
            # All empty: return a fresh empty instance (never alias an input,
            # per the immutability contract).
            from luxar.gsplats.utils.trils import tril_size

            d0 = datasets[0]
            d = d0.ndim
            return cls(
                centers=np.empty((0, d), dtype=np.float32),
                amplitudes=np.empty(0, dtype=np.float32),
                cholesky_factors=np.empty(
                    (0, tril_size(d) if d > 0 else 0), dtype=np.float32
                ),
                colors=None,
                stats=dict(d0.stats),
                truncation_radius=d0.truncation_radius,
            )

        ndim = non_empty[0].ndim
        tr = non_empty[0].truncation_radius
        n_sub = non_empty[0].n_substitutive
        for i, ds in enumerate(non_empty[1:], start=1):
            if ds.ndim != ndim:
                raise ValueError(
                    f"Dimensionality mismatch: dataset 0 has {ndim}D, "
                    f"dataset {i} has {ds.ndim}D"
                )
            if ds.truncation_radius != tr:
                raise ValueError(
                    f"Truncation radius mismatch: dataset 0 has {tr}, "
                    f"dataset {i} has {ds.truncation_radius}. "
                    f"Cannot concatenate datasets fitted with different truncation radii."
                )
            if ds.n_substitutive != n_sub:
                raise ValueError(
                    f"Substitutive-level count mismatch: dataset 0 has "
                    f"{n_sub}, dataset {i} has {ds.n_substitutive}. "
                    f"concatenate() requires a uniform substitutive hierarchy; "
                    f"flatten() the inputs first to merge mismatched pyramids."
                )

        merged_stats: Dict[str, Any] = {
            "concatenated_from": len(datasets),
            "splats_per_source": [d.n_splats for d in datasets],
        }
        total_time = sum(d.stats.get("time_seconds", 0) for d in non_empty)
        if total_time > 0:
            merged_stats["time_seconds"] = total_time

        # Multi-substitutive path: merge per (substitutive, additive) cell so
        # the full pyramid survives.
        if n_sub > 1:
            # Read each source's finest-first level list ONCE (the property
            # reconstructs the whole matrix per call); index per level below so
            # the merge stays O(L·D) rather than O(L²·D).
            levels_per_source = [d.substitutive_levels for d in non_empty]
            template = levels_per_source[0]
            sub_levels: List[SubstitutiveLevel] = []
            for s in range(n_sub):
                views_s = [
                    d._view_of_level(levels_per_source[di][s])
                    for di, d in enumerate(non_empty)
                ]
                ref = template[s]
                sub_levels.append(
                    SubstitutiveLevel(
                        additive_sublods=_concat_additive_levels(views_s),
                        compression_factor=ref.compression_factor,
                        parent_method=ref.parent_method,
                        level_index=ref.level_index,
                        stats={**ref.stats, "n_sources": len(non_empty)},
                    )
                )
            return cls.from_substitutive_levels(
                sub_levels,
                stats=merged_stats,
            )

        # Single substitutive level: merge its additive ladder.
        merged_lods = _concat_additive_levels(non_empty)
        if len(merged_lods) > 1:
            return cls(additive_sublods=merged_lods, stats=merged_stats)

        only = merged_lods[0]
        return cls(
            centers=only.centers,
            amplitudes=only.amplitudes,
            cholesky_factors=only.cholesky_factors,
            colors=only.colors,
            stats=merged_stats,
            truncation_radius=only.truncation_radius,
        )

    @classmethod
    def combine_as_new_dimension(
        cls,
        datasets: "list[GSplatData]",
        values: "np.ndarray | list[float] | None" = None,
        sigma: float = 0.0,
    ) -> "GSplatData":
        """Combine datasets by embedding each into a new dimension, then concatenating.

        Each dataset is promoted from D-dimensional to (D+1)-dimensional by
        appending a coordinate in the new dimension, then all are concatenated
        into a single dataset.

        This is useful for combining per-timepoint 3D fits into a single 4D
        dataset, per-slice 2D fits into 3D, or any similar stacking operation.

        Args:
            datasets: List of GSplatData, all with the same ndim.
            values: Coordinate for each dataset in the new dimension.
                If None, uses 0.0, 1.0, 2.0, ... (one per dataset).
                If scalar-per-dataset, all splats in that dataset get the same
                coordinate.  Can also be a list of per-splat arrays if different
                splats within a dataset need different coordinates.
            sigma: Standard deviation in the new dimension.
                Use 0.0 for discrete dimensions (e.g., time frames) where
                splats should not extend across the new axis.
                Use a positive value for continuous dimensions where splats
                should have Gaussian extent.

        Returns:
            Single GSplatData with ndim+1 dimensions containing all splats.

        Raises:
            ValueError: If datasets is empty, lengths mismatch, or ndims differ.

        Example:
            >>> # Combine 3D timepoints into 4D
            >>> combined = GSplatData.combine_as_new_dimension(
            ...     [t0_3d, t1_3d, t2_3d], sigma=0.0
            ... )
            >>> combined.ndim  # 4
            >>> combined.n_splats  # sum of all timepoints
        """
        if not datasets:
            raise ValueError("At least one GSplatData is required")

        if values is None:
            values = [float(i) for i in range(len(datasets))]
        elif hasattr(values, "__len__"):
            values = list(values)
        else:
            raise TypeError(
                f"values must be a list/array or None, got {type(values).__name__}"
            )

        if len(values) != len(datasets):
            raise ValueError(
                f"Number of values ({len(values)}) must match "
                f"number of datasets ({len(datasets)})"
            )

        embedded = [
            ds.embed_dimension(val, sigma=sigma) for ds, val in zip(datasets, values)
        ]
        return cls.concatenate(embedded)

    def to_spatial_partition(
        self,
        *,
        max_elements: int,
        rule: Literal["median", "midpoint", "sah"] = "median",
    ) -> "GSplatPartition":
        """Spatially partition the splats into a ``kind=partition`` tree node.

        Recursively BSP-splits the splat **centers** so each part holds at most
        ``max_elements`` splats, using the shared splitters in
        :mod:`luxar.core.group.partition` (the same machinery the scene uses).
        Returns a :class:`~luxar.gsplats.tree.GSplatPartition` (a tree node, not
        a ``GSplatData`` — a partition has no flat-matrix equivalent); write it
        with ``write_gsplats_tree`` (one self-contained ``kind=partition`` file)
        or embed it in a scene. Each part gets its own ``position_bounds`` at
        write time so the viewer can frustum-cull per part.

        A multi-LOD input is flattened to its default substitutive level first
        (BSP partitions a single splat set), matching :meth:`partition`.
        """
        from luxar.core.group.partition import spatial_bsp_tree

        from .tree import GSplatLeaf, GSplatPartition

        if max_elements < 1:
            raise ValueError(f"max_elements must be >= 1, got {max_elements}")
        if rule not in ("median", "midpoint", "sah"):
            raise ValueError(
                f"rule must be 'median', 'midpoint', or 'sah'; got {rule!r}"
            )

        src: GSplatData = self
        if self.n_substitutive > 1 or self.n_additive_sublods > 1:
            warnings.warn(
                "to_spatial_partition() flattens LOD structure: input has "
                f"n_substitutive={self.n_substitutive}, "
                f"n_additive_sublods={self.n_additive_sublods}; coarser "
                "substitutive levels and the additive ladder are collapsed "
                "into a single level before partitioning.",
                UserWarning,
                stacklevel=2,
            )
            src = self.flattened()

        centers = np.asarray(src.centers)
        # Build the BSP TREE (retains split planes), then read its leaves in
        # left-first DFS order as the parts. The serialized tree rides along on
        # the partition so the viewer can order parts back-to-front exactly
        # (see GSplatPartition.bsp_tree); leaf part index k == child index k.
        tree = spatial_bsp_tree(centers, max_elements, rule=rule)
        parts = [leaf.indices for leaf in tree.leaves()]
        children: List["GSplatNode"] = []
        for idx in parts:
            assert idx is not None
            children.append(
                GSplatLeaf(
                    additive_sublods=[
                        AdditiveSubLOD(
                            centers=src.centers[idx],
                            amplitudes=src.amplitudes[idx],
                            cholesky_factors=src.cholesky_factors[idx],
                            colors=src.colors[idx] if src.colors is not None else None,
                            truncation_radius=src.truncation_radius,
                        )
                    ]
                )
            )
        return GSplatPartition(
            children=children,
            max_elements=max_elements,
            bsp_tree=tree.to_serializable(),
        )

    @staticmethod
    def partition_from_regions(
        regions: "List[GSplatData]",
        *,
        recipe: "Optional[str]" = None,
        recipe_params: "Optional[Any]" = None,
    ) -> "GSplatNode":
        """Assemble a ``kind=partition`` tree from pre-decomposed spatial regions.

        Unlike :meth:`to_spatial_partition` (which BSP-splits a flat splat set),
        this keeps the **given** spatial decomposition: each region becomes one
        partition part, preserving the exact tile/box boundaries the fitter
        already produced. Used by tiled / content-aware fitting, where the
        regions are the per-tile (apodized) or per-box (core-kept) splats — both
        sum correctly as additive partition parts, so the partitioned render
        equals the flat concatenation with no double-count.

        With ``recipe`` (one of
        :data:`~luxar.gsplats.lod.recipes.PER_PART_RECIPES`: ``additive`` →
        ``partitioned`` topology, ``substitutive`` → ``mosaic``) each part is
        given its OWN LOD via :func:`~luxar.gsplats.lod.recipes.build_part_lod`
        (clamped to the part's splat count), so the output is a partition whose
        every child carries a ladder/lod-group — the fit-time equivalent of a
        per-part ``gsplat lod`` pass (which cannot run on a partition). Without a
        recipe each part is a bare leaf (the historical behaviour).

        Empty regions (0 splats) are dropped. With a single non-empty region the
        bare part node is returned (no 1-part partition wrapper); with none,
        raises. Returns a tree node (write with ``write_gsplats_tree`` or embed
        in a scene) — a partition has no flat-matrix ``GSplatData`` equivalent.
        """
        from .tree import GSplatPartition

        nonempty = [r for r in regions if r.n_splats > 0]
        if not nonempty:
            raise ValueError("partition_from_regions: all regions are empty")

        def _part_node(region: "GSplatData") -> "GSplatNode":
            if recipe is None:
                return region.tree
            from luxar.gsplats.lod.recipes import RecipeParams, build_part_lod

            params = recipe_params if recipe_params is not None else RecipeParams()
            return build_part_lod(region.tree, recipe, params)

        if len(nonempty) == 1:
            return _part_node(nonempty[0])  # single part -> bare part node
        return GSplatPartition(children=[_part_node(r) for r in nonempty])

    def embed_dimension(
        self,
        values: "np.ndarray | float",
        sigma: float = 0.0,
    ) -> "GSplatData":
        """Add a new dimension to the splat data.

        Appends a column to centers and embeds Cholesky factors into
        the higher-dimensional space.

        Args:
            values: Coordinate for the new dimension. Scalar (same for all)
                or (N,) array (per-splat).
            sigma: Standard deviation in the new dimension (default 0.0
                for discrete dimensions like time).

        Returns:
            New GSplatData with ndim+1 dimensions.

        Example:
            >>> data_4d = data_3d.embed_dimension(5.0, sigma=0.0)
            >>> data_4d = data_3d.embed_dimension(time_values, sigma=0.5)
        """
        from luxar.gsplats.utils.trils import embed_cholesky_packed

        # A 0-d numpy array is semantically a scalar; unwrap it so the
        # np.isscalar() branches below treat it as the broadcast coordinate it
        # represents (rather than a malformed per-splat array of shape ()).
        if isinstance(values, np.ndarray) and values.ndim == 0:
            values = values.item()

        n = self.n_splats
        d = self.ndim

        # Multi-substitutive: embed every level and rebuild the pyramid. A scalar
        # coordinate broadcasts cleanly to all levels; a per-splat array is sized
        # to the finest level only and cannot map to coarser levels, so reject it
        # (mirrors with_colors) rather than silently collapsing the ladder. The
        # scalar path is the one the merge pipeline (combine_as_new_dimension)
        # exercises on pyramid inputs.
        if self.n_substitutive > 1:
            if not np.isscalar(values):
                raise ValueError(
                    "embed_dimension with a per-splat values array is not "
                    "supported on a multi-substitutive pyramid (each level has a "
                    "different splat count). Pass a scalar coordinate to broadcast "
                    "across all levels, or operate per level via at_substitutive()."
                )
            return self._map_substitutive(
                lambda lvl: lvl.embed_dimension(values, sigma)
            )

        # Multi-LOD path: embed each LOD independently
        if self.n_additive_sublods > 1:
            is_scalar = np.isscalar(values)
            values_arr: Optional[np.ndarray] = None
            if not is_scalar:
                values_arr = np.asarray(values, dtype=self.centers.dtype)
                if values_arr.shape != (n,):
                    raise ValueError(
                        f"values shape {values_arr.shape} doesn't match splat count ({n},)"
                    )
            dim_mapping = list(range(d))
            fill_sigma = {d: sigma}

            def _embed_lod(lod: AdditiveSubLOD, offset: int, nl: int) -> AdditiveSubLOD:
                if is_scalar:
                    lod_col = np.full((nl, 1), values, dtype=lod.centers.dtype)
                else:
                    assert values_arr is not None
                    lod_col = values_arr[offset : offset + nl].reshape(nl, 1)
                lod_centers = np.concatenate([lod.centers, lod_col], axis=1)
                lod_cholesky = embed_cholesky_packed(
                    lod.cholesky_factors, d, d + 1, dim_mapping, fill_sigma
                )
                return AdditiveSubLOD(
                    centers=lod_centers,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod_cholesky,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )

            return self._map_additive(_embed_lod)

        # Single-LOD fast path (unchanged)
        if np.isscalar(values):
            new_col = np.full((n, 1), values, dtype=self.centers.dtype)
        else:
            values = np.asarray(values, dtype=self.centers.dtype)
            if values.shape != (n,):
                raise ValueError(
                    f"values shape {values.shape} doesn't match splat count ({n},)"
                )
            new_col = values.reshape(n, 1)

        new_centers = np.concatenate([self.centers, new_col], axis=1)
        new_cholesky = embed_cholesky_packed(
            self.cholesky_factors,
            d_src=d,
            d_dst=d + 1,
            dim_mapping=list(range(d)),
            fill_sigma={d: sigma},
        )

        return GSplatData(
            centers=new_centers,
            amplitudes=self.amplitudes,
            cholesky_factors=new_cholesky,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    # ── Geometric transforms ────────────────────────────────

    def _map_substitutive(
        self, fn: "Callable[[GSplatData], GSplatData]"
    ) -> "GSplatData":
        """Apply a single-level transform to EVERY substitutive level, rebuild.

        ``fn`` maps a single-substitutive-level view (``n_substitutive == 1``)
        to a transformed single-level ``GSplatData``; per-level metadata
        (compression_factor / parent_method / level_index / stats) is preserved.
        Mirrors :meth:`filter_by`'s per-level rebuild (decision 6) so spatial
        and intensity ops never silently collapse the substitutive LOD ladder
        to the finest level. Callers guard with ``if self.n_substitutive > 1``.
        """
        new_levels: List[SubstitutiveLevel] = []
        for s, src in enumerate(self.substitutive_levels):
            out = fn(self._view_of_level(src))
            new_levels.append(
                SubstitutiveLevel(
                    additive_sublods=out.substitutive_levels[0].additive_sublods,
                    compression_factor=src.compression_factor,
                    parent_method=src.parent_method,
                    level_index=src.level_index,
                    stats=dict(src.stats),
                )
            )
        return GSplatData.from_substitutive_levels(new_levels, stats=dict(self.stats))

    def _map_additive(
        self, fn: "Callable[[AdditiveSubLOD, int, int], AdditiveSubLOD]"
    ) -> "GSplatData":
        """Apply a per-sub-LOD transform to EVERY additive sub-LOD, rebuild.

        ``fn`` receives ``(lod, offset, n)`` — the sub-LOD, its start offset
        into the flattened finest-leaf arrays, and its splat count — and returns
        a replacement :class:`AdditiveSubLOD` (which may change N, ndim, or
        array widths). The additive-dimension sibling of :meth:`_map_substitutive`;
        callers guard the multi-sub-LOD branch with
        ``if self.n_additive_sublods > 1``.
        """
        new_lods: List[AdditiveSubLOD] = []
        offset = 0
        for lod in self.additive_sublods:
            n = lod.n_splats
            new_lods.append(fn(lod, offset, n))
            offset += n
        return GSplatData.from_additive_sublods(new_lods, stats=dict(self.stats))

    def transform(self, matrix: np.ndarray) -> "GSplatData":
        """Apply affine transformation to all splats.

        Transforms centers and covariance matrices. Amplitudes and colors
        are unchanged.

        Args:
            matrix: Either (d, d) for linear-only transform or
                (d+1, d+1) for full affine (last row must be [0..0, 1]).

        Returns:
            New GSplatData with transformed geometry.

        Raises:
            ValueError: If matrix shape is invalid.
            np.linalg.LinAlgError: If transform produces non-positive-definite covariance.

        Example:
            >>> scaled = data.transform(np.eye(3) * 2.0)
            >>> M = np.eye(4); M[:3, 3] = [10, 20, 30]
            >>> transformed = data.transform(M)
        """
        from luxar.gsplats.utils.trils import pack_tril, unpack_tril

        # Multi-substitutive: transform every level and rebuild the pyramid
        # (mirrors filter_by/cull) rather than collapsing to the finest level.
        if self.n_substitutive > 1:
            return self._map_substitutive(lambda lvl: lvl.transform(matrix))

        matrix = np.asarray(matrix, dtype=np.float64)
        d = self.ndim

        if matrix.shape == (d, d):
            A = matrix
            t = np.zeros(d, dtype=np.float64)
        elif matrix.shape == (d + 1, d + 1):
            A = matrix[:d, :d]
            t = matrix[:d, d]
            expected = np.zeros(d + 1, dtype=np.float64)
            expected[-1] = 1.0
            if not np.allclose(matrix[d, :], expected):
                raise ValueError(
                    f"Last row of (d+1)x(d+1) matrix must be [0...0, 1], "
                    f"got {matrix[d, :]}"
                )
        else:
            raise ValueError(
                f"Matrix shape must be ({d},{d}) or ({d + 1},{d + 1}), got {matrix.shape}"
            )

        if self.n_splats == 0:
            if self.n_additive_sublods > 1:
                return GSplatData.from_additive_sublods(
                    [
                        AdditiveSubLOD(
                            centers=lod.centers.copy(),
                            amplitudes=lod.amplitudes,
                            cholesky_factors=lod.cholesky_factors.copy(),
                            colors=lod.colors,
                            stats=dict(lod.stats),
                            truncation_radius=lod.truncation_radius,
                        )
                        for lod in self.additive_sublods
                    ],
                    stats=dict(self.stats),
                )
            return GSplatData(
                centers=self.centers.copy(),
                amplitudes=self.amplitudes,
                cholesky_factors=self.cholesky_factors.copy(),
                colors=self.colors,
                stats=dict(self.stats),
                truncation_radius=self.truncation_radius,
            )

        # Precompute cholesky transform (shared between single/multi-LOD paths)
        is_diagonal = np.count_nonzero(A - np.diag(np.diagonal(A))) == 0
        if is_diagonal:
            diag = np.diagonal(A)
            if np.any(diag <= 0):
                raise ValueError(f"Diagonal scale factors must be positive, got {diag}")
            tril_scales = np.concatenate([[diag[i]] * (i + 1) for i in range(d)])

        def _transform_cholesky(chol: np.ndarray) -> np.ndarray:
            if is_diagonal:
                return np.asarray(chol * tril_scales.astype(chol.dtype))
            L = unpack_tril(chol.astype(np.float64), d)
            Sigma = L @ np.swapaxes(L, -2, -1)
            Sigma_new = A @ Sigma @ A.T
            L_new = np.linalg.cholesky(Sigma_new)
            return pack_tril(L_new).astype(chol.dtype)

        # Multi-LOD path: transform each LOD independently
        if self.n_additive_sublods > 1:

            def _transform_lod(
                lod: AdditiveSubLOD, offset: int, n: int
            ) -> AdditiveSubLOD:
                lod_centers = (lod.centers.astype(np.float64) @ A.T + t).astype(
                    lod.centers.dtype
                )
                return AdditiveSubLOD(
                    centers=lod_centers,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=_transform_cholesky(lod.cholesky_factors),
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )

            return self._map_additive(_transform_lod)

        # Single-LOD fast path
        new_centers = (self.centers.astype(np.float64) @ A.T + t).astype(
            self.centers.dtype
        )
        new_cholesky = _transform_cholesky(self.cholesky_factors)

        return GSplatData(
            centers=new_centers,
            amplitudes=self.amplitudes,
            cholesky_factors=new_cholesky,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    # ── Intensity transforms ────────────────────────────────

    def _with_new_amplitudes(self, new_amplitudes: np.ndarray) -> "GSplatData":
        """Return a new GSplatData with replaced amplitudes, preserving LODs."""
        if self.n_additive_sublods > 1:
            return self._map_additive(
                lambda lod, offset, n: AdditiveSubLOD(
                    centers=lod.centers,
                    amplitudes=new_amplitudes[offset : offset + n],
                    cholesky_factors=lod.cholesky_factors,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )
        return GSplatData(
            centers=self.centers,
            amplitudes=new_amplitudes,
            cholesky_factors=self.cholesky_factors,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    def with_colors(self, colors: "np.ndarray | tuple[float, ...]") -> "GSplatData":
        """Return a new GSplatData with replaced colors, preserving LODs.

        Args:
            colors: Either an (N, 3) RGB / (N, 4) RGBA array of per-splat
                colors, or a single (r, g, b) / (r, g, b, a) tuple/array to
                broadcast to all splats. The alpha channel is per-splat
                opacity in [0, 1].

        Returns:
            New GSplatData with the specified colors.
        """
        arr = (
            colors
            if isinstance(colors, np.ndarray)
            else np.asarray(colors, dtype=np.float32)
        )
        is_broadcast = arr.ndim == 1 and arr.shape in ((3,), (4,))

        # Multi-substitutive: rebuild the pyramid. A single (r, g, b) broadcasts
        # cleanly to every level; an explicit per-splat array cannot (each level
        # has a different splat count), so reject it rather than silently
        # collapse the ladder to the finest level.
        if self.n_substitutive > 1:
            if not is_broadcast:
                raise ValueError(
                    "with_colors with an explicit per-splat color array is not "
                    "supported on a multi-substitutive pyramid (each level has a "
                    "different splat count). Pass a single (r, g, b) to broadcast "
                    "across all levels, or operate per level via at_substitutive()."
                )
            rgb = arr.astype(np.float32)
            return self._map_substitutive(lambda lvl: lvl.with_colors(rgb))

        if is_broadcast:
            # Broadcast single color to all splats
            colors = np.tile(arr.astype(np.float32), (self.n_splats, 1))
        else:
            colors = arr
        if colors.shape not in ((self.n_splats, 3), (self.n_splats, 4)):
            raise ValueError(
                f"colors shape {colors.shape} doesn't match "
                f"({self.n_splats}, 3) or ({self.n_splats}, 4)"
            )
        if self.n_additive_sublods > 1:
            return self._map_additive(
                lambda lod, offset, n: AdditiveSubLOD(
                    centers=lod.centers,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod.cholesky_factors,
                    colors=colors[offset : offset + n],
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )
        return GSplatData(
            centers=self.centers,
            amplitudes=self.amplitudes,
            cholesky_factors=self.cholesky_factors,
            colors=colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    def affine_intensity(self, scale: float = 1.0, offset: float = 0.0) -> "GSplatData":
        """Apply affine transform to amplitudes: new_amp = scale * amp + offset.

        Args:
            scale: Multiplicative factor.
            offset: Additive offset.

        Returns:
            New GSplatData with transformed amplitudes.
        """
        if self.n_substitutive > 1:
            return self._map_substitutive(
                lambda lvl: lvl.affine_intensity(scale, offset)
            )
        return self._with_new_amplitudes(self.amplitudes * scale + offset)

    def normalize_intensity(self, target_max: float = 1.0) -> "GSplatData":
        """Normalize amplitudes so the maximum equals target_max.

        Args:
            target_max: Desired maximum amplitude (default 1.0).

        Returns:
            New GSplatData. Returns copy if all amplitudes are zero.
        """
        # A single global factor (from the finest level's max) is applied
        # uniformly to all substitutive levels via scale_intensity — which is
        # itself pyramid-preserving — so the ladder is kept and levels stay
        # consistently scaled (a per-level normalization would shift them apart).
        current_max = float(self.amplitudes.max()) if self.n_splats > 0 else 0.0
        if current_max == 0:
            return self.scale_intensity(1.0)  # no-op, but preserves the pyramid
        return self.scale_intensity(target_max / current_max)

    def clamp_intensity(
        self,
        min: "float | None" = None,
        max: "float | None" = None,
    ) -> "GSplatData":
        """Clamp amplitudes to a range.

        Args:
            min: Lower bound (None = no lower bound).
            max: Upper bound (None = no upper bound).

        Returns:
            New GSplatData with clamped amplitudes.
        """
        if self.n_substitutive > 1:
            return self._map_substitutive(lambda lvl: lvl.clamp_intensity(min, max))
        new_amps = self.amplitudes.copy()
        if min is not None:
            new_amps = np.maximum(new_amps, min)
        if max is not None:
            new_amps = np.minimum(new_amps, max)
        return self._with_new_amplitudes(new_amps)

    def translate(self, offset: np.ndarray) -> "GSplatData":
        """Translate all splat centers by an offset vector.

        Args:
            offset: Translation vector (shape: (d,) where d is spatial dimensions)

        Returns:
            New GSplatData with translated centers (all other data unchanged)

        Example:
            >>> # Shift all splats by [10, 20, 30]
            >>> translated = data.translate(np.array([10, 20, 30]))
        """
        # Multi-substitutive: translate every level and rebuild the pyramid.
        if self.n_substitutive > 1:
            return self._map_substitutive(lambda lvl: lvl.translate(offset))

        # Multi-LOD path: translate each LOD independently
        if self.n_additive_sublods > 1:
            return self._map_additive(
                lambda lod, offset_, n: AdditiveSubLOD(
                    centers=lod.centers + offset,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod.cholesky_factors,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )

        return GSplatData(
            centers=self.centers + offset,
            amplitudes=self.amplitudes,
            cholesky_factors=self.cholesky_factors,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    def center_at_centroid(self) -> "GSplatData":
        """Center the splats at their center of mass (amplitude-weighted centroid).

        The centroid is the amplitude-weighted average of splat centers (the
        center of mass of the represented density). Only the **spatial**
        (non-degenerate) axes are re-origined: a zero-variance categorical axis
        (a per-timepoint time axis, a channel axis) keeps its original
        coordinates, because centering it would push integer timepoints to
        fractional offsets and misalign the viewer's slice navigator. For pure
        spatial data (no degenerate axis) every axis is centered, as before.

        Returns:
            New GSplatData with its spatial centroid at the origin.

        Example:
            >>> # Center splats at origin for easier viewing
            >>> centered = data.center_at_centroid()
        """
        # Empty data: nothing to center. Return a structure-preserving copy
        # (translate by zero) rather than computing mean() of an empty array,
        # which would emit a spurious "Mean of empty slice" RuntimeWarning.
        if self.n_splats == 0:
            return self.translate(np.zeros(self.ndim, dtype=np.float64))

        # Compute amplitude-weighted centroid
        total_amplitude = self.amplitudes.sum()
        if total_amplitude > 0:
            centroid = (self.centers.T @ self.amplitudes) / total_amplitude
        else:
            centroid = self.centers.mean(axis=0)

        # Shift only the spatial (non-degenerate) axes; leave categorical axes
        # (zero covariance extent — e.g. a stacked-time axis) at their
        # coordinates. Mirrors scale()/eccentricities()/isolation grouping.
        shift = spatial_only_shift(centroid, self._nondegenerate_axes())
        return self.translate(-shift)

    def scale_intensity(self, factor: float) -> "GSplatData":
        """Scale all splat amplitudes by a multiplicative factor.

        This effectively brightens (factor > 1) or dims (factor < 1) the
        entire representation.

        Args:
            factor: Multiplicative scaling factor for amplitudes

        Returns:
            New GSplatData with scaled amplitudes

        Example:
            >>> # Reduce brightness by 10x
            >>> dimmed = data.scale_intensity(0.1)
            >>> # Brighten by 2x
            >>> brightened = data.scale_intensity(2.0)
        """
        if self.n_substitutive > 1:
            return self._map_substitutive(lambda lvl: lvl.scale_intensity(factor))
        return self._with_new_amplitudes(self.amplitudes * factor)

    def reweight_amplitude(self, multiplier: np.ndarray) -> "GSplatData":
        """Return a copy with per-splat amplitudes multiplied by ``multiplier``.

        The per-splat counterpart of ``scale_intensity`` (which is scalar-only).
        ``multiplier`` must be shape ``(n_splats,)`` and operates on this
        (matrix / default-level) view; it preserves the additive ladder. A
        global multiplier is not meaningful across substitutive levels — callers
        with a pyramid should reweight per-level (see ``soft_scale_filter``).
        """
        multiplier = np.asarray(multiplier, dtype=np.float64)
        if multiplier.shape != (self.n_splats,):
            raise ValueError(
                f"multiplier shape {multiplier.shape} != ({self.n_splats},)"
            )
        return self._with_new_amplitudes(self.amplitudes * multiplier)

    def soft_scale_filter(
        self,
        *,
        highpass: float | None = None,
        lowpass: float | None = None,
        width: float = 1.0,
        spatial_dims: Sequence[int] | None = None,
    ) -> "GSplatData":
        """Soft "frequency" filter: attenuate amplitude by a smooth function of
        each splat's characteristic ``scale()`` — a gentler alternative to a hard
        scale cut (no popping, splat count unchanged).

        - ``highpass``: suppress splats with scale ABOVE the cutoff (removes
          large diffuse / low-frequency background). Multiplier → 0 for very
          large scales, → 1 for small.
        - ``lowpass``: suppress splats with scale BELOW the cutoff (removes fine
          detail / high-frequency). Multiplier → 0 for very small scales, → 1
          for large.

        Both may be combined (a band-pass). ``width`` is the transition softness
        in octaves (log2 scale); larger = gentler roll-off.

        The cutoff is in the same world units as ``scale()``.
        """
        if self.n_splats == 0 or (highpass is None and lowpass is None):
            return self
        if self.n_substitutive > 1:
            # Reweight each substitutive level against its OWN scale distribution.
            return self._map_substitutive(
                lambda lvl: lvl.soft_scale_filter(
                    highpass=highpass,
                    lowpass=lowpass,
                    width=width,
                    spatial_dims=spatial_dims,
                )
            )
        scl = np.clip(self.scale(axes=spatial_dims), 1e-12, None)
        w = max(float(width), 1e-6)
        mult = np.ones(self.n_splats, dtype=np.float64)
        # Smoothstep in log2(scale) space, spanning ±width octaves about cutoff.

        def _smoothstep(t: np.ndarray) -> np.ndarray:
            t = np.clip(t, 0.0, 1.0)
            out: np.ndarray = t * t * (3.0 - 2.0 * t)
            return out

        if highpass is not None:
            # 1 (keep) for scale <= cutoff, ramping to 0 above.
            t = (np.log2(scl) - np.log2(float(highpass))) / w + 0.5
            mult *= 1.0 - _smoothstep(t)
        if lowpass is not None:
            # 1 (keep) for scale >= cutoff, ramping to 0 below.
            t = (np.log2(scl) - np.log2(float(lowpass))) / w + 0.5
            mult *= _smoothstep(t)
        return self.reweight_amplitude(mult)

    @classmethod
    def merge_with_channel_colors(
        cls,
        gsplats_per_channel: list["GSplatData"],
        channel_colors: list[tuple[float, float, float]],
    ) -> "GSplatData":
        """Merge multiple GSplatData objects, assigning a fixed color per channel.

        This is useful for multi-channel visualization where each channel was
        fitted separately and should be displayed with a distinct color.

        Args:
            gsplats_per_channel: List of GSplatData objects, one per channel.
                All must have the same dimensionality.
            channel_colors: List of RGB color tuples (one per channel).
                Each tuple should have values in [0, 1] range, e.g., (1.0, 0.0, 0.5).

        Returns:
            New GSplatData with all splats merged and colors assigned.

        Raises:
            ValueError: If lists have different lengths or dimensionalities don't match.

        Example:
            >>> # Fit each channel separately
            >>> gsplats_ch0 = fit_gaussian_splats(volume_ch0, ...)
            >>> gsplats_ch1 = fit_gaussian_splats(volume_ch1, ...)
            >>>
            >>> # Merge with magenta for ch0, cyan for ch1
            >>> merged = GSplatData.merge_with_channel_colors(
            ...     [gsplats_ch0, gsplats_ch1],
            ...     channel_colors=[(1.0, 0.0, 0.5), (0.0, 1.0, 0.5)],
            ... )
            >>>
            >>> # Add to scene
            >>> scene.add_gsplats_from_data("multichannel", merged)
        """
        if len(gsplats_per_channel) != len(channel_colors):
            raise ValueError(
                f"Number of GSplatData objects ({len(gsplats_per_channel)}) must match "
                f"number of colors ({len(channel_colors)})"
            )

        if len(gsplats_per_channel) == 0:
            raise ValueError("At least one GSplatData object is required")

        # Validate all have same dimensionality and truncation radius
        ndim = gsplats_per_channel[0].ndim
        tr = gsplats_per_channel[0].truncation_radius
        for i, gsplat in enumerate(gsplats_per_channel[1:], start=1):
            if gsplat.ndim != ndim:
                raise ValueError(
                    f"Dimensionality mismatch: channel 0 has {ndim}D, "
                    f"channel {i} has {gsplat.ndim}D"
                )
            if gsplat.truncation_radius != tr:
                raise ValueError(
                    f"Truncation radius mismatch: channel 0 has {tr}, "
                    f"channel {i} has {gsplat.truncation_radius}. "
                    f"Cannot merge datasets fitted with different truncation radii."
                )

        # Merge stats (basic aggregation)
        merged_stats: Dict[str, Any] = {
            "merged_from_channels": len(gsplats_per_channel),
            "splats_per_channel": [len(g.amplitudes) for g in gsplats_per_channel],
        }
        total_time = sum(g.stats.get("time_seconds", 0) for g in gsplats_per_channel)
        if total_time > 0:
            merged_stats["time_seconds"] = total_time

        # Multi-substitutive: merge per substitutive level and rebuild the
        # pyramid (mirrors concatenate / the transform ops), never silently
        # collapsing to the finest level. Reachable via `luxar gsplat merge
        # --channel-colors` on kind=lod inputs. Each level merges the channels
        # that HAVE that level (parallel to the additive max_lods path below).
        max_sub = max(g.n_substitutive for g in gsplats_per_channel)
        if max_sub > 1:
            new_levels: List[SubstitutiveLevel] = []
            for s in range(max_sub):
                parts = [
                    (g.at_substitutive(s), color)
                    for g, color in zip(gsplats_per_channel, channel_colors)
                    if s < g.n_substitutive
                ]
                merged_level = cls.merge_with_channel_colors(
                    [view for view, _ in parts], [color for _, color in parts]
                )
                template = parts[0][0].substitutive_levels[0]
                new_levels.append(
                    SubstitutiveLevel(
                        additive_sublods=merged_level.substitutive_levels[
                            0
                        ].additive_sublods,
                        compression_factor=template.compression_factor,
                        parent_method=template.parent_method,
                        level_index=template.level_index,
                        stats=dict(template.stats),
                    )
                )
            return cls.from_substitutive_levels(new_levels, stats=merged_stats)

        # Multi-LOD path: per-LOD channel color assignment
        max_lods = max(g.n_additive_sublods for g in gsplats_per_channel)
        if max_lods > 1:
            merged_lods = []
            for level in range(max_lods):
                level_parts = [
                    (g.additive_sublod(level), color)
                    for g, color in zip(gsplats_per_channel, channel_colors)
                    if level < g.n_additive_sublods
                ]
                centers = np.concatenate(
                    [lod.centers for lod, _ in level_parts], axis=0
                )
                amplitudes = np.concatenate([lod.amplitudes for lod, _ in level_parts])
                cholesky = np.concatenate(
                    [lod.cholesky_factors for lod, _ in level_parts], axis=0
                )
                colors = np.concatenate(
                    [
                        np.tile(np.array(c, dtype=np.float32), (lod.n_splats, 1))
                        for lod, c in level_parts
                    ],
                    axis=0,
                )
                merged_lods.append(
                    AdditiveSubLOD(
                        centers=centers,
                        amplitudes=amplitudes,
                        cholesky_factors=cholesky,
                        colors=colors,
                        stats={"lod_level": level, "n_channels": len(level_parts)},
                        truncation_radius=level_parts[0][0].truncation_radius,
                    )
                )
            return cls(additive_sublods=merged_lods, stats=merged_stats)

        # Single-LOD fast path (unchanged)
        all_centers = np.concatenate([g.centers for g in gsplats_per_channel], axis=0)
        all_amplitudes = np.concatenate(
            [g.amplitudes for g in gsplats_per_channel], axis=0
        )
        all_cholesky = np.concatenate(
            [g.cholesky_factors for g in gsplats_per_channel], axis=0
        )
        color_arrays = []
        for gsplat, color in zip(gsplats_per_channel, channel_colors):
            channel_color_array = np.tile(
                np.array(color, dtype=np.float32), (gsplat.n_splats, 1)
            )
            color_arrays.append(channel_color_array)
        all_colors = np.concatenate(color_arrays, axis=0)

        return cls(
            centers=all_centers,
            amplitudes=all_amplitudes,
            cholesky_factors=all_cholesky,
            colors=all_colors,
            stats=merged_stats,
            truncation_radius=gsplats_per_channel[0].truncation_radius,
        )
