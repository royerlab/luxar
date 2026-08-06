"""Shared foundation for the ``GSplatData`` domain mixins.

``_GSplatDataOps`` gives the domain mixins (render / io / filtering / culling /
lod views / composition / transforms / intensity) a single base so their
cross-mixin ``self.`` calls resolve (and type-check). It declares the extra
instance attributes ``GSplatData.__init__`` assigns plus typing STUBS for the
``GSplatData`` methods/properties/classmethods those moved mixins call on
``self`` — the real implementations live on ``GSplatData`` (or the sibling
mixins). The stubs mirror the concrete signatures exactly and only ever raise.

The module also holds the shared module-level helpers the mixins and
``GSplatData.__init__`` need (read-only array views and the additive-ladder
color merge).
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)

import numpy as np

# Runtime (not TYPE_CHECKING) import: ``_node``'s annotation below is a CLASS
# annotation, so ``typing.get_type_hints(GSplatData)`` must be able to resolve
# it. ``tree`` only pulls numpy and ``utils.spatial_axes`` at module level and
# reaches back into ``gsplat_data`` lazily, so this direction is cycle-free.
from luxar.gsplats.tree import GSplatNode

from .metrics import _SplatArrayMixin

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )


class _GSplatDataOps(_SplatArrayMixin):
    """Base the ``GSplatData`` domain mixins inherit so cross-mixin ``self.``
    calls resolve through a single class.

    ``GSplatData.__init__`` assigns the instance attributes annotated below; the
    stubs are declared here but implemented on ``GSplatData`` / the sibling
    mixins, so a mixin calling e.g. ``self.filter`` or ``self._map_substitutive``
    type-checks.
    """

    # Instance attributes assigned by GSplatData.__init__ (beyond the
    # centers/amplitudes/cholesky_factors declared on _SplatArrayMixin).
    colors: Optional[np.ndarray]
    stats: Dict[str, Any]
    _node: GSplatNode

    # ── Derived matrix views (real on GSplatData) ──────────────────────────
    @property
    def substitutive_levels(self) -> List["SubstitutiveLevel"]:
        raise NotImplementedError

    @property
    def additive_sublods(self) -> List["AdditiveSubLOD"]:
        raise NotImplementedError

    @property
    def n_substitutive(self) -> int:
        raise NotImplementedError

    @property
    def n_additive_sublods(self) -> int:
        raise NotImplementedError

    @property
    def truncation_radius(self) -> float:
        raise NotImplementedError

    # ── Cross-mixin methods (real on GSplatData / sibling mixins) ──────────
    def filter(self, mask: np.ndarray) -> "GSplatData":
        raise NotImplementedError

    def filter_by(
        self,
        *,
        bbox: list[tuple[float, float]] | None = None,
        volume_min: float | None = None,
        volume_max: float | None = None,
        volume_normalized: bool = False,
        volume_percentile: bool = False,
        scale_min: float | None = None,
        scale_max: float | None = None,
        scale_normalized: bool = False,
        scale_percentile: bool = False,
        amplitude_min: float | None = None,
        amplitude_max: float | None = None,
        amplitude_normalized: bool = False,
        amplitude_percentile: bool = False,
        eccentricity_min: float | None = None,
        eccentricity_max: float | None = None,
        eccentricity_percentile: bool = False,
        mass_min: float | None = None,
        mass_max: float | None = None,
        mass_normalized: bool = False,
        mass_percentile: bool = False,
        sigma_axis: int | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        sigma_percentile: bool = False,
        isolation_max: float | None = None,
        isolation_percentile: bool = False,
        min_neighbors: int | None = None,
        neighbor_radius: float | None = None,
        spatial_dims: Sequence[int] | None = None,
        truncate: float | None = None,
    ) -> "GSplatData":
        raise NotImplementedError

    def _map_substitutive(
        self, fn: "Callable[[GSplatData], GSplatData]"
    ) -> "GSplatData":
        raise NotImplementedError

    def _map_additive(
        self, fn: "Callable[[AdditiveSubLOD, int, int], AdditiveSubLOD]"
    ) -> "GSplatData":
        raise NotImplementedError

    def _view_of_level(self, src_level: "SubstitutiveLevel") -> "GSplatData":
        raise NotImplementedError

    def flattened(self) -> "GSplatData":
        raise NotImplementedError

    # ── Classmethod constructors (real on GSplatData) ──────────────────────
    @classmethod
    def from_additive_sublods(
        cls,
        additive_sublods: List["AdditiveSubLOD"],
        stats: Optional[Dict[str, Any]] = None,
    ) -> "GSplatData":
        raise NotImplementedError

    @classmethod
    def from_substitutive_levels(
        cls,
        substitutive_levels: List["SubstitutiveLevel"],
        stats: Optional[Dict[str, Any]] = None,
    ) -> "GSplatData":
        raise NotImplementedError


# ── Shared module-level helpers ─────────────────────────────────────────────


def _merge_lod_colors(
    lods: "list[AdditiveSubLOD]",
    *,
    channels: "Optional[int]" = None,
    allow_integer: bool = True,
) -> "Optional[np.ndarray]":
    """Merge colors from multiple LODs/datasets using None/all/mixed logic.

    - All have colors → concatenate.
    - All None → return None.
    - Mixed → fill missing with white (1,1,1).
    - Mixed RGB/RGBA channel counts → RGB parts widen to RGBA with alpha=1
      (opaque, the per-element-opacity identity).

    ``channels``/``allow_integer`` let a caller merging one LEVEL of a larger
    ladder impose the ladder-wide layout (channel count) and dtype policy —
    the viewer fail-fasts on ladders whose levels disagree on either, so
    per-level decisions must never diverge across levels (see
    ``_concat_additive_levels``). Defaults preserve the standalone behavior:
    layout/dtype decided from the given ``lods`` alone.

    Dtype: when every present part shares ONE integer dtype and no white-fill
    is needed (and ``allow_integer``), the native integer dtype is
    **preserved** (full-scale = opaque), matching the single-LOD path
    (``self.colors = lod0.colors``) — uint8 colors are a valid SDR storage
    form, unlike the always-float centers/amps/cholesky.
    Otherwise the merge is float32 in ``[0, 1]``: integer parts are normalized
    by their full-scale (÷max) first, so a widened integer opaque (``iinfo.max``)
    can never ride into a promoted float array as an out-of-``[0, 1]`` value
    (a uint8 alpha=255 beside a float ``[0, 1]`` part would otherwise become a
    float ``255.0`` alpha). Live producers are all float32, so this is a no-op
    on every current path.
    """
    # widen_colors_to_rgba is part of gsplat_data's PUBLIC surface, so it stays
    # there; imported lazily here (as the mixins do for GSplatData) to keep this
    # module importable before gsplat_data finishes loading.
    from luxar.gsplats.gsplat_data import widen_colors_to_rgba

    if not lods:
        return None
    present = [lod.colors for lod in lods if lod.colors is not None]
    if not present:
        return None
    if channels is None:
        channels = max(c.shape[1] for c in present)

    # Preserve a uniform integer dtype (matches the single-LOD path); promote to
    # float32 only when a white-fill or a dtype mismatch would otherwise force a
    # lossy/scale-inconsistent concat.
    any_fill = any(lod.colors is None for lod in lods)
    dtypes = {c.dtype for c in present}
    all_same_integer = (
        allow_integer
        and not any_fill
        and len(dtypes) == 1
        and np.issubdtype(present[0].dtype, np.integer)
    )
    if all_same_integer:
        parts = [widen_colors_to_rgba(c) if channels == 4 else c for c in present]
        return np.concatenate(parts, axis=0)

    parts = []
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
    from luxar.gsplats.gsplat_data import AdditiveSubLOD

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

    The color layout (RGB vs RGBA) and dtype policy are decided ONCE for the
    whole ladder, not per level: with ragged inputs, later levels see a
    different source subset than earlier ones (e.g. merging an RGBA 2-level
    view with an RGB uint8 3-level view leaves level 2 RGB-uint8-only), and a
    per-level decision would emit a mixed-layout/mixed-dtype ladder — exactly
    what the viewer's fail-fast load validation rejects as malformed.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD

    max_lods = max(v.n_additive_sublods for v in views)
    levels: "list[list[AdditiveSubLOD]]" = [
        [v.additive_sublod(level) for v in views if level < v.n_additive_sublods]
        for level in range(max_lods)
    ]

    # Ladder-wide color policy (see docstring). A white-fill only happens on a
    # WITHIN-level presence mix (a wholly colorless level stays None, which the
    # viewer legally white-fills), so only those force the float32 promotion.
    all_present = [
        lod.colors for lods in levels for lod in lods if lod.colors is not None
    ]
    ladder_channels = max((c.shape[1] for c in all_present), default=3)
    ladder_dtypes = {c.dtype for c in all_present}
    ladder_any_fill = any(
        any(lod.colors is None for lod in lods)
        and any(lod.colors is not None for lod in lods)
        for lods in levels
    )
    ladder_allow_integer = (
        not ladder_any_fill
        and len(ladder_dtypes) == 1
        and np.issubdtype(next(iter(ladder_dtypes)), np.integer)
    )

    merged: "list[AdditiveSubLOD]" = []
    for level in range(max_lods):
        level_lods = levels[level]
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
                colors=_merge_lod_colors(
                    level_lods,
                    channels=ladder_channels if all_present else None,
                    allow_integer=ladder_allow_integer,
                ),
                stats={"lod_level": level, "n_sources": len(level_lods)},
                truncation_radius=level_lods[0].truncation_radius,
            )
        )
    return merged
