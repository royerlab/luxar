"""Gaussian Splat data container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import numpy as np

# Used by __init__ below; the ``as`` alias marks it an EXPLICIT re-export so the
# historical ``luxar.gsplats.gsplat_data._merge_lod_colors`` import path (used by
# luxar.gsplats.lod.annotate and its tests) keeps resolving under mypy strict.
from luxar.gsplats._data.base import _merge_lod_colors as _merge_lod_colors
from luxar.gsplats._data.composition import CompositionMixin
from luxar.gsplats._data.culling import CullingMixin
from luxar.gsplats._data.filtering import FilteringMixin

# The PUBLIC face of the `_data` package's stats-hygiene rules (#1600): a caller
# outside `luxar.gsplats` that rewrites a store without ever building a
# `GSplatData` — the CLI's node-tree `transform` path — needs the same predicate
# and the same scrub as the dataset methods, and must not reach into the private
# subpackage for them. `as` aliases mark them explicit re-exports for mypy strict.
from luxar.gsplats._data.filtering import (
    drop_content_scoped_stats as drop_content_scoped_stats,
)
from luxar.gsplats._data.filtering import (
    scrub_measured_stats as scrub_measured_stats,
)
from luxar.gsplats._data.filtering import (
    scrub_region_scoped_stats as scrub_region_scoped_stats,
)
from luxar.gsplats._data.filtering import (
    stamp_region_scoped_stats as stamp_region_scoped_stats,
)
from luxar.gsplats._data.filtering import (
    stats_after_structure_change as stats_after_structure_change,
)
from luxar.gsplats._data.intensity import IntensityMixin
from luxar.gsplats._data.intensity import amplitudes_changed as amplitudes_changed
from luxar.gsplats._data.io_adapter import IOAdapterMixin
from luxar.gsplats._data.lod_views import LODViewsMixin
from luxar.gsplats._data.metrics import _SplatArrayMixin
from luxar.gsplats._data.render import RenderMixin
from luxar.gsplats._data.transforms import TransformsMixin
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from luxar.validation.types import validate_truncation_radius

if TYPE_CHECKING:
    from luxar.gsplats.tree import GSplatNode


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


def normalize_label_vocabulary(
    vocabulary: Optional[Mapping[int, str]],
) -> Optional[Dict[int, str]]:
    """Validate and normalize an id-to-name vocabulary."""
    if vocabulary is None:
        return None
    normalized: Dict[int, str] = {}
    for raw_id, name in vocabulary.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, np.integer)):
            raise TypeError("label_vocabulary keys must be non-negative integers")
        label_id = int(raw_id)
        if label_id < 0:
            raise ValueError("label_vocabulary keys must be non-negative integers")
        if not isinstance(name, str):
            raise TypeError("label_vocabulary values must be strings")
        normalized[label_id] = name
    return normalized


def validate_label_channel(
    label_ids: Optional[np.ndarray],
    label_vocabulary: Optional[Mapping[int, str]],
    n_splats: int,
) -> Optional[Dict[int, str]]:
    """Validate one exact categorical per-splat channel."""
    vocabulary = normalize_label_vocabulary(label_vocabulary)
    if label_ids is None:
        if vocabulary is not None:
            raise ValueError("label_vocabulary requires label_ids")
        return None
    if vocabulary is None:
        raise ValueError("label_ids requires an explicit label_vocabulary")
    if label_ids.shape != (n_splats,):
        raise ValueError(
            f"Label IDs shape {label_ids.shape} doesn't match centers count ({n_splats},)"
        )
    if not np.issubdtype(label_ids.dtype, np.integer) or (
        np.issubdtype(label_ids.dtype, np.signedinteger) and np.any(label_ids < 0)
    ):
        raise ValueError("label_ids must contain non-negative integers")
    missing = sorted(
        set(int(value) for value in np.unique(label_ids)) - vocabulary.keys()
    )
    if missing:
        raise ValueError(
            f"label_vocabulary is missing ids present in label_ids: {missing}"
        )
    return vocabulary


def _merge_additive_label_channel(
    sublods: List["AdditiveSubLOD"],
) -> tuple[Optional[np.ndarray], Optional[Dict[int, str]]]:
    label_presence = {lod.label_ids is not None for lod in sublods}
    if len(label_presence) > 1:
        raise ValueError(
            "additive ladder must carry label_ids on every sub-LOD or none"
        )
    if label_presence == {False}:
        return None, None
    label_ids = np.concatenate(
        [lod.label_ids for lod in sublods if lod.label_ids is not None]
    )
    vocabularies = [lod.label_vocabulary for lod in sublods]
    if any(vocab != vocabularies[0] for vocab in vocabularies[1:]):
        raise ValueError("additive ladder label_vocabulary values must be identical")
    return label_ids, vocabularies[0]


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
        Packed lower-triangular factor L of the covariance (Σ = L·Lᵀ). The
        diagonal is scale-like: isotropic std σ uses [σ, 0, σ, 0, 0, σ], not
        1/sigma.
    colors : Optional[np.ndarray], shape (N, 3) or (N, 4)
        Optional RGB(A) colors per splat. The optional alpha channel is
        per-splat opacity in [0, 1] (consumed by every blending mode; mapped
        into optical depth in volumetric — see VOLUMETRIC_BLENDING_SPEC.md).
    label_ids : Optional[np.ndarray], shape (N,)
        Exact non-negative categorical id per splat.
    label_vocabulary : Optional[Dict[int, str]]
        Explicit name for every id present in ``label_ids``.
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
    label_ids: Optional[np.ndarray] = None
    label_vocabulary: Optional[Dict[int, str]] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    truncation_radius: float = DEFAULT_TRUNCATION_RADIUS

    def __post_init__(self) -> None:
        """Validate array shape consistency."""
        # The standalone .gsplats.zarr path never reaches the scene compiler's
        # `validate_render_attrs`, so the truncation radius is checked here too.
        # Validating per sub-LOD (not just via GSplatData's first-LOD property)
        # is deliberate: the tree writer reads each sub-LOD's own value, so a
        # bad radius on a later rung would otherwise reach disk unchecked.
        validate_truncation_radius(self.truncation_radius)
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
        vocabulary = validate_label_channel(self.label_ids, self.label_vocabulary, n)
        object.__setattr__(self, "label_vocabulary", vocabulary)
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


class GSplatData(
    RenderMixin,
    IOAdapterMixin,
    FilteringMixin,
    CullingMixin,
    LODViewsMixin,
    CompositionMixin,
    TransformsMixin,
    IntensityMixin,
):
    """Container for Gaussian splat data with always-LOD structure.

    Every ``GSplatData`` holds one or more LOD levels (``AdditiveSubLOD`` instances).
    A single-LOD dataset is simply ``additive_sublods=[one_lod]``.

    **Construction styles**::

        # Convenience constructor (wraps into single LOD internally):
        GSplatData(centers=c, amplitudes=a, cholesky_factors=cf)

        # Explicit LOD construction:
        GSplatData.from_additive_sublods([lod0, lod1, lod2])

    Top-level ``centers``, ``amplitudes``, ``cholesky_factors``, ``colors``, and
    ``label_ids`` are the concatenation of all additive LODs, computed once at
    construction time.
    The object is conceptually immutable — all operations return new instances.

    The behaviour is split across the domain mixins in
    :mod:`luxar.gsplats._data` (render / io / filtering / culling / lod views /
    composition / transforms / intensity); this class holds construction, the
    truncation radius and the repr.

    Attributes
    ----------
    additive_sublods : List[AdditiveSubLOD]
        Additive sub-LODs of the default substitutive level. Always >= 1.
    centers : np.ndarray, shape (N_total, d)
        Cached concatenation of all LOD centers.
    amplitudes : np.ndarray, shape (N_total,)
        Cached concatenation of all LOD amplitudes.
    cholesky_factors : np.ndarray, shape (N_total, tril)
        Cached concatenation of all LOD packed lower-triangular factors L of the
        covariance (Σ = L·Lᵀ), using the convenience-constructor convention:
        isotropic std σ uses [σ, 0, σ, 0, 0, σ], not 1/sigma.
    colors : Optional[np.ndarray], shape (N_total, 3) or (N_total, 4)
        Cached concatenation of all LOD colors (None if no LOD has colors).
        The optional 4th column is per-splat opacity alpha in [0, 1].
    label_ids : Optional[np.ndarray], shape (N_total,)
        Cached concatenation of exact categorical ids.
    label_vocabulary : Optional[Dict[int, str]]
        Shared id-to-name vocabulary for the categorical channel.
    stats : Dict[str, Any]
        Top-level statistics (overall quality, timing, etc.).
    """

    def __init__(
        self,
        centers: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
        cholesky_factors: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        label_ids: Optional[np.ndarray] = None,
        label_vocabulary: Optional[Mapping[int, str]] = None,
        stats: Optional[Dict[str, Any]] = None,
        *,
        additive_sublods: Optional[List[AdditiveSubLOD]] = None,
        substitutive_levels: Optional[List[SubstitutiveLevel]] = None,
        truncation_radius: float = DEFAULT_TRUNCATION_RADIUS,
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
                label_ids=label_ids,
                label_vocabulary=normalize_label_vocabulary(label_vocabulary),
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
            self.label_ids = lod0.label_ids
            self.label_vocabulary = lod0.label_vocabulary
        else:
            self.centers = np.concatenate(
                [lod.centers for lod in finest_sublods], axis=0
            )
            self.amplitudes = np.concatenate([lod.amplitudes for lod in finest_sublods])
            self.cholesky_factors = np.concatenate(
                [lod.cholesky_factors for lod in finest_sublods], axis=0
            )
            self.colors = _merge_lod_colors(finest_sublods)
            self.label_ids, self.label_vocabulary = _merge_additive_label_channel(
                finest_sublods
            )

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
