"""Shared foundation for the ``GSplatData`` domain mixins.

``_GSplatDataOps`` gives the render / io / filtering / culling mixins a single
base so their cross-mixin ``self.`` calls resolve (and type-check). It declares
the extra instance attributes ``GSplatData.__init__`` assigns plus typing STUBS
for the ``GSplatData`` methods/properties/classmethods those moved mixins call
on ``self`` — the real implementations live on ``GSplatData`` (or the sibling
mixins). The stubs mirror the concrete signatures exactly and only ever raise.
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
    _node: Any

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
