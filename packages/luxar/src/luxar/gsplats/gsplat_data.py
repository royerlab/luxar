"""Gaussian Splat data container."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from luxar.encoding import EncodingMode
    from luxar.gsplats.tree import GSplatNode, GSplatPartition


#: Sentinel for ``GSplatData.save(compressor=...)`` distinguishing "not specified"
#: (→ default Blosc) from an explicit ``compressor=None`` (→ no compression). A
#: plain ``None`` default would conflate the two and make uncompressed output
#: impossible (the bug that produced blosc-bitshuffle fixtures zarrita can't read).
_USE_DEFAULT_COMPRESSOR = object()


def _merge_lod_colors(
    lods: "list[AdditiveSubLOD]",
) -> "Optional[np.ndarray]":
    """Merge colors from multiple LODs/datasets using None/all/mixed logic.

    - All have colors → concatenate.
    - All None → return None.
    - Mixed → fill missing with white (1,1,1).
    """
    if not lods:
        return None
    has_colors = [lod.colors is not None for lod in lods]
    if all(has_colors):
        result: np.ndarray = np.concatenate([lod.colors for lod in lods], axis=0)
        return result
    elif not any(has_colors):
        return None
    else:
        parts: list[np.ndarray] = []
        for lod in lods:
            if lod.colors is not None:
                parts.append(lod.colors)
            else:
                parts.append(np.ones((lod.n_splats, 3), dtype=np.float32))
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
    """Rebuild ``lod`` with read-only (zero-copy) array views."""
    return AdditiveSubLOD(
        centers=_readonly(lod.centers),
        amplitudes=_readonly(lod.amplitudes),
        cholesky_factors=_readonly(lod.cholesky_factors),
        colors=_readonly_opt(lod.colors),
        stats=lod.stats,
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


class _SplatArrayMixin:
    """Shared computed properties for splat array containers.

    Requires the implementing class to have:
    - ``centers``: np.ndarray of shape (N, d)
    - ``amplitudes``: np.ndarray of shape (N,)
    - ``cholesky_factors``: np.ndarray of shape (N, d*(d+1)//2)
    """

    centers: np.ndarray
    amplitudes: np.ndarray
    cholesky_factors: np.ndarray

    @property
    def n_splats(self) -> int:
        """Number of splats."""
        return int(self.centers.shape[0])

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return int(self.centers.shape[1]) if self.centers.ndim >= 2 else 0

    def __len__(self) -> int:
        """Return number of splats."""
        return self.n_splats

    def _cholesky_diag_elements(self) -> np.ndarray:
        """Extract diagonal elements from packed Cholesky factors.

        Returns shape (N, d) where result[i, j] = L_i[j, j].
        """
        ndim = self.ndim
        diag_indices = np.cumsum(np.arange(1, ndim + 1)) - 1
        return self.cholesky_factors[:, diag_indices]

    def volumes(self) -> np.ndarray:
        """Per-splat characteristic length: det(Σ)^(1/d).

        This is the geometric mean of the eigenvalues (not a true volume).
        For lower-triangular L: det(L) = product of diagonal elements,
        det(Sigma) = det(L)^2.

        Returns:
            shape (N,) float array.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        diag = self._cholesky_diag_elements()
        det_L = np.prod(diag, axis=1)
        result: np.ndarray = np.abs(det_L**2) ** (1.0 / self.ndim)
        return result

    def masses(self) -> np.ndarray:
        """Per-splat mass: amplitude * volume.

        Returns:
            shape (N,) float array.
        """
        result: np.ndarray = self.amplitudes * self.volumes()
        return result

    def marginal_sigmas(self) -> np.ndarray:
        """Per-dimension standard deviation: sqrt(Sigma_ii).

        For lower-triangular L: Sigma[i,i] = sum_j L[i,j]^2.

        Returns:
            shape (N, d) float array.
        """
        if self.n_splats == 0:
            return np.empty((0, self.ndim), dtype=np.float64)
        from luxar.gsplats.utils.trils import unpack_tril

        L = unpack_tril(self.cholesky_factors.astype(np.float64), self.ndim)
        result: np.ndarray = np.sqrt(np.sum(L**2, axis=2))
        return result

    def eccentricities(self) -> np.ndarray:
        """Per-splat eccentricity: max marginal sigma / min marginal sigma.

        1.0 = isotropic. Higher values = more elongated.

        Returns:
            shape (N,) float array. Returns 1.0 for degenerate splats.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        sigmas = self.marginal_sigmas()
        min_s = sigmas.min(axis=1)
        max_s = sigmas.max(axis=1)
        result = np.ones(self.n_splats, dtype=np.float64)
        nonzero = min_s > 0
        result[nonzero] = max_s[nonzero] / min_s[nonzero]
        return result

    def principal_radii(self, anisotropy: bool = True) -> np.ndarray:
        """Per-splat element radius (world units) at the truncation boundary.

        This is the extent that governs on-screen resolvability for LOD switching
        (see ``core.group.lod.group.extent_min_pixel_sizes``): the Gaussian is
        truncated at ``truncation_radius`` sigmas, so the radius is
        ``truncation_radius * semi_axis``.

        - ``anisotropy=True`` → the largest principal semi-axis
          ``sqrt(lambda_max(Sigma))`` (worst-case projected radius;
          orientation-independent — the splat's biggest reach in any direction).
        - ``anisotropy=False`` → the isotropic-equivalent geometric-mean semi-axis
          ``det(Sigma)^(1/2d)`` (== ``sqrt(volumes())``).

        Returns:
            shape (N,) float array.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        trunc = float(getattr(self, "truncation_radius", 3.0))
        if anisotropy:
            from luxar.gsplats.utils.trils import unpack_tril

            chol = self.cholesky_factors.astype(np.float64)
            ell = unpack_tril(chol, self.ndim)
            sigma = ell @ np.swapaxes(ell, -2, -1)
            # eigvalsh returns ascending eigenvalues; the last is lambda_max.
            lam_max = np.linalg.eigvalsh(sigma)[:, -1]
            semi = np.sqrt(np.clip(lam_max, 0.0, None))
        else:
            semi = np.sqrt(self.volumes())
        result: np.ndarray = trunc * semi
        return result


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
    colors : Optional[np.ndarray], shape (N, 3)
        Optional RGB colors per splat.
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


class GSplatData(_SplatArrayMixin):
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
    colors : Optional[np.ndarray], shape (N_total, 3)
        Cached concatenation of all LOD colors (None if no LOD has colors).
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
    ) -> None:
        if substitutive_levels is not None:
            # New 2-D construction: full substitutive × additive matrix
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
            self.substitutive_levels: List[SubstitutiveLevel] = list(
                substitutive_levels
            )
            # The data-model default is fixed at the FINEST level (index 0) — it
            # is not a settable, persistable concept (the on-disk default_level
            # is the viewer's separate coarsest-first render hint). Kept as a
            # constant attribute so accessors document "return the finest".
            self.default_substitutive: int = 0
            # Derived: the "primary" additive ladder is the finest level's.
            self.additive_sublods: List[AdditiveSubLOD] = list(
                self.substitutive_levels[0].additive_sublods
            )
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
            self.additive_sublods = list(additive_sublods)
            # Single substitutive level wrapping the additive ladder
            self.substitutive_levels = [
                SubstitutiveLevel(
                    additive_sublods=list(additive_sublods),
                    compression_factor=1,
                    parent_method=None,
                    level_index=0,
                )
            ]
            self.default_substitutive = 0
        elif (
            centers is not None
            and amplitudes is not None
            and cholesky_factors is not None
        ):
            # Convenience constructor — wrap into single LOD, single substitutive level
            single_lod = AdditiveSubLOD(
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky_factors,
                colors=colors,
                stats=stats if stats is not None else {},
                truncation_radius=truncation_radius,
            )
            self.additive_sublods = [single_lod]
            self.substitutive_levels = [
                SubstitutiveLevel(
                    additive_sublods=[single_lod],
                    compression_factor=1,
                    parent_method=None,
                    level_index=0,
                )
            ]
            self.default_substitutive = 0
        else:
            raise ValueError(
                "Provide either substitutive_levels=[...], "
                "additive_sublods=[...], or (centers, amplitudes, cholesky_factors)"
            )

        # Compute cached concatenations from LODs
        if len(self.additive_sublods) == 1:
            # Fast path: single LOD, no copy
            lod0 = self.additive_sublods[0]
            self.centers = lod0.centers
            self.amplitudes = lod0.amplitudes
            self.cholesky_factors = lod0.cholesky_factors
            self.colors = lod0.colors
        else:
            self.centers = np.concatenate(
                [lod.centers for lod in self.additive_sublods], axis=0
            )
            self.amplitudes = np.concatenate(
                [lod.amplitudes for lod in self.additive_sublods]
            )
            self.cholesky_factors = np.concatenate(
                [lod.cholesky_factors for lod in self.additive_sublods], axis=0
            )
            self.colors = _merge_lod_colors(self.additive_sublods)

        # Top-level stats (separate from per-LOD stats)
        if stats is not None:
            self.stats: Dict[str, Any] = stats
        elif additive_sublods is not None or substitutive_levels is not None:
            # When constructed via additive_sublods=/substitutive_levels=,
            # start with empty top-level stats
            self.stats = {}
        else:
            # Convenience constructor already set stats on the LOD; mirror it
            self.stats = dict(self.additive_sublods[0].stats)

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
        return len(self.substitutive_levels)

    @property
    def default_substitutive_level(self) -> SubstitutiveLevel:
        """The substitutive level pointed to by ``default_substitutive``."""
        return self.substitutive_levels[self.default_substitutive]

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
        src_level = self.substitutive_levels[level]
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

    def cell(self, substitutive: int, additive: int) -> AdditiveSubLOD:
        """Direct 2-D matrix access: cell at ``(substitutive, additive)``.

        Args:
            substitutive: Substitutive level index.
            additive: Additive sub-LOD index within that substitutive level.

        Returns:
            The :class:`AdditiveSubLOD` at the requested matrix cell.
        """
        if not 0 <= substitutive < self.n_substitutive:
            raise IndexError(
                f"substitutive level {substitutive} out of range "
                f"[0, {self.n_substitutive})"
            )
        level = self.substitutive_levels[substitutive]
        if not 0 <= additive < level.n_additive_lods:
            raise IndexError(
                f"additive sub-LOD {additive} out of range "
                f"[0, {level.n_additive_lods}) at substitutive level {substitutive}"
            )
        return level.additive_sublods[additive]

    # ── Node-tree bridge (v3.0 unified representation) ──────

    @property
    def tree(self) -> "GSplatNode":
        """This dataset as a :mod:`luxar.gsplats.tree` node subtree.

        The tree is the unified representation behind the v3.0 ``.gsplats.zarr``
        format and the scene gsplat-node subtree. For the historical
        ``substitutive × additive`` matrix this is exactly one shape: a single
        :class:`~luxar.gsplats.tree.GSplatLeaf` (one substitutive level) or a
        :class:`~luxar.gsplats.tree.GSplatLodGroup` of leaves (multiple levels,
        finest first). Per-level provenance rides in each leaf's ``meta``.
        """
        from luxar.gsplats.tree import tree_from_substitutive_levels

        return tree_from_substitutive_levels(self.substitutive_levels)

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
        from luxar.gsplats.tree import substitutive_levels_from_tree

        # The on-disk default_level is the viewer's coarsest-first render hint,
        # not a data-model default — the accessors always return the finest
        # level (index 0), so it is intentionally ignored here.
        levels, _default = substitutive_levels_from_tree(node)
        return cls(
            substitutive_levels=levels,
            stats=stats,
        )

    # ── Filtering ───────────────────────────────────────────

    def filter(self, mask: np.ndarray) -> "GSplatData":
        """Return new GSplatData with only the splats where mask is True.

        Args:
            mask: Boolean array of shape (N,).

        Returns:
            New GSplatData with filtered arrays.

        Example:
            >>> filtered = data.filter(data.volumes() < 100)
            >>> filtered = data.filter((data.amplitudes > 0.1) & (data.eccentricities() < 5))
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_splats,):
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match splat count ({self.n_splats},)"
            )

        # A raw boolean mask is sized to the default substitutive level, so it
        # cannot be applied per-level — coarser substitutive levels are dropped.
        # Warn loudly (never silent) and point at the criteria-based ops, which
        # DO preserve the full pyramid (see filter_by / cull).
        if self.n_substitutive > 1:
            warnings.warn(
                "filter(mask) keeps only the default substitutive level "
                f"(n_substitutive={self.n_substitutive}); coarser levels are "
                "dropped. Use filter_by(...) / cull(...) to filter every "
                "substitutive level and preserve the pyramid.",
                UserWarning,
                stacklevel=2,
            )

        # Multi-LOD path: split mask across LODs
        if self.n_additive_sublods > 1:
            new_lods = []
            offset = 0
            for lod in self.additive_sublods:
                n = lod.n_splats
                lod_mask = mask[offset : offset + n]
                new_lods.append(
                    AdditiveSubLOD(
                        centers=lod.centers[lod_mask],
                        amplitudes=lod.amplitudes[lod_mask],
                        cholesky_factors=lod.cholesky_factors[lod_mask],
                        colors=lod.colors[lod_mask] if lod.colors is not None else None,
                        stats=dict(lod.stats),
                        truncation_radius=lod.truncation_radius,
                    )
                )
                offset += n
            return GSplatData.from_additive_sublods(new_lods, stats=dict(self.stats))

        return GSplatData(
            centers=self.centers[mask],
            amplitudes=self.amplitudes[mask],
            cholesky_factors=self.cholesky_factors[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    @staticmethod
    def _resolve_threshold(
        val: float | None,
        normalized: bool,
        dataset_values: np.ndarray,
    ) -> float | None:
        """Map a threshold from [0,1] normalized range to absolute if needed."""
        if val is None:
            return None
        if normalized:
            dmin, dmax = float(dataset_values.min()), float(dataset_values.max())
            return dmin + val * (dmax - dmin)
        return val

    def filter_by(
        self,
        *,
        bbox: list[tuple[float, float]] | None = None,
        volume_min: float | None = None,
        volume_max: float | None = None,
        volume_normalized: bool = False,
        amplitude_min: float | None = None,
        amplitude_max: float | None = None,
        amplitude_normalized: bool = False,
        eccentricity_min: float | None = None,
        eccentricity_max: float | None = None,
        mass_min: float | None = None,
        mass_max: float | None = None,
        mass_normalized: bool = False,
        sigma_axis: int | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        truncate: float | None = None,
    ) -> "GSplatData":
        """Filter splats by multiple criteria (AND logic).

        All criteria are optional. Only specified criteria are applied.
        Multiple criteria combine with AND — a splat must satisfy all
        active criteria to be kept.

        Args:
            bbox: Bounding box per dimension as [(min0, max0), (min1, max1), ...].
                  Length must equal ndim. Filters by center position.
            volume_min: Minimum volume (characteristic length * truncate).
            volume_max: Maximum volume.
            volume_normalized: If True, interpret volume thresholds as 0-1
                mapped to the dataset's [min, max] volume range.
            amplitude_min: Minimum amplitude.
            amplitude_max: Maximum amplitude.
            amplitude_normalized: If True, interpret amplitude thresholds as 0-1
                mapped to the dataset's [min, max] amplitude range.
            eccentricity_min: Minimum eccentricity (1.0 = isotropic).
            eccentricity_max: Maximum eccentricity.
            mass_min: Minimum mass (amplitude * volume).
            mass_max: Maximum mass.
            mass_normalized: If True, interpret mass thresholds as 0-1
                mapped to the dataset's [min, max] mass range.
            sigma_axis: Axis index for per-axis sigma filtering.
            sigma_min: Minimum marginal sigma on sigma_axis.
            sigma_max: Maximum marginal sigma on sigma_axis.
            truncate: Sigma truncation factor for volume computation.
                Defaults to ``self.truncation_radius``.

        Returns:
            New GSplatData with only splats that pass all criteria.

        Raises:
            ValueError: If bbox length doesn't match ndim, sigma_axis is out
                of range, or sigma_min/sigma_max given without sigma_axis.

        Examples:
            >>> # Keep splats with amplitude >= 0.1 and eccentricity <= 5
            >>> filtered = data.filter_by(amplitude_min=0.1, eccentricity_max=5.0)
            >>>
            >>> # Spatial crop to a bounding box (3D)
            >>> filtered = data.filter_by(bbox=[(0, 50), (0, 50), (0, 50)])
            >>>
            >>> # Remove top 10% largest volumes (normalized)
            >>> filtered = data.filter_by(volume_max=0.9, volume_normalized=True)
        """
        if truncate is None:
            truncate = self.truncation_radius

        # Short-circuit for empty data
        if self.n_splats == 0:
            result = self.filter(np.ones(0, dtype=bool))
            result.stats.update(
                {
                    "filtered": True,
                    "filter_criteria": {},
                    "n_original": 0,
                    "n_removed": 0,
                    "truncate": truncate,
                }
            )
            return result

        # Validate sigma_axis usage
        if (sigma_min is not None or sigma_max is not None) and sigma_axis is None:
            raise ValueError("sigma_min/sigma_max require sigma_axis to be specified")
        if sigma_axis is not None and not (0 <= sigma_axis < self.ndim):
            raise ValueError(
                f"sigma_axis={sigma_axis} out of range for {self.ndim}D data"
            )

        # Multi-substitutive: apply the SAME criteria to every substitutive
        # level and rebuild the pyramid (decision 6) rather than silently
        # collapsing to the default level. Each level is filtered through the
        # single-substitutive path below (a per-level view); thresholds with
        # *_normalized resolve per-level (each level to its own range).
        if self.n_substitutive > 1:
            new_levels: List[SubstitutiveLevel] = []
            for s, src in enumerate(self.substitutive_levels):
                filtered = self.at_substitutive(s).filter_by(
                    bbox=bbox,
                    volume_min=volume_min,
                    volume_max=volume_max,
                    volume_normalized=volume_normalized,
                    amplitude_min=amplitude_min,
                    amplitude_max=amplitude_max,
                    amplitude_normalized=amplitude_normalized,
                    eccentricity_min=eccentricity_min,
                    eccentricity_max=eccentricity_max,
                    mass_min=mass_min,
                    mass_max=mass_max,
                    mass_normalized=mass_normalized,
                    sigma_axis=sigma_axis,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    truncate=truncate,
                )
                new_levels.append(
                    SubstitutiveLevel(
                        additive_sublods=filtered.substitutive_levels[
                            0
                        ].additive_sublods,
                        compression_factor=src.compression_factor,
                        parent_method=src.parent_method,
                        level_index=src.level_index,
                        stats=dict(src.stats),
                    )
                )
            out = GSplatData.from_substitutive_levels(
                new_levels,
                stats=dict(self.stats),
            )
            out.stats.update(
                {
                    "filtered": True,
                    "n_original": self.n_splats,
                    "n_removed": self.n_splats - out.n_splats,
                    "truncate": truncate,
                }
            )
            return out

        mask = np.ones(self.n_splats, dtype=bool)
        criteria: dict[str, object] = {}

        # -- Bounding box (center position)
        if bbox is not None:
            if len(bbox) != self.ndim:
                raise ValueError(
                    f"bbox has {len(bbox)} dimensions, expected {self.ndim}"
                )
            criteria["bbox"] = bbox
            for i, (lo, hi) in enumerate(bbox):
                mask &= (self.centers[:, i] >= lo) & (self.centers[:, i] <= hi)

        # -- Volume (characteristic length * truncate)
        if volume_min is not None or volume_max is not None:
            vols = self.volumes() * truncate
            vmin = self._resolve_threshold(volume_min, volume_normalized, vols)
            vmax = self._resolve_threshold(volume_max, volume_normalized, vols)
            if vmin is not None:
                mask &= vols >= vmin
                criteria["volume_min"] = vmin
            if vmax is not None:
                mask &= vols <= vmax
                criteria["volume_max"] = vmax
            if volume_normalized:
                criteria["volume_normalized"] = True

        # -- Amplitude
        if amplitude_min is not None or amplitude_max is not None:
            amps = self.amplitudes
            amin = self._resolve_threshold(amplitude_min, amplitude_normalized, amps)
            amax = self._resolve_threshold(amplitude_max, amplitude_normalized, amps)
            if amin is not None:
                mask &= amps >= amin
                criteria["amplitude_min"] = amin
            if amax is not None:
                mask &= amps <= amax
                criteria["amplitude_max"] = amax
            if amplitude_normalized:
                criteria["amplitude_normalized"] = True

        # -- Eccentricity
        if eccentricity_min is not None or eccentricity_max is not None:
            ecc = self.eccentricities()
            if eccentricity_min is not None:
                mask &= ecc >= eccentricity_min
                criteria["eccentricity_min"] = eccentricity_min
            if eccentricity_max is not None:
                mask &= ecc <= eccentricity_max
                criteria["eccentricity_max"] = eccentricity_max

        # -- Mass (amplitude * volume)
        if mass_min is not None or mass_max is not None:
            m = self.masses()
            mmin = self._resolve_threshold(mass_min, mass_normalized, m)
            mmax = self._resolve_threshold(mass_max, mass_normalized, m)
            if mmin is not None:
                mask &= m >= mmin
                criteria["mass_min"] = mmin
            if mmax is not None:
                mask &= m <= mmax
                criteria["mass_max"] = mmax
            if mass_normalized:
                criteria["mass_normalized"] = True

        # -- Per-axis sigma
        if sigma_axis is not None and (sigma_min is not None or sigma_max is not None):
            sigmas = self.marginal_sigmas()[:, sigma_axis]
            criteria["sigma_axis"] = sigma_axis
            if sigma_min is not None:
                mask &= sigmas >= sigma_min
                criteria["sigma_min"] = sigma_min
            if sigma_max is not None:
                mask &= sigmas <= sigma_max
                criteria["sigma_max"] = sigma_max

        # Apply mask
        result = self.filter(mask)
        result.stats.update(
            {
                "filtered": True,
                "filter_criteria": criteria,
                "n_original": self.n_splats,
                "n_removed": self.n_splats - result.n_splats,
                "truncate": truncate,
            }
        )
        return result

    def slice_by(self, slices: list[slice]) -> "GSplatData":
        """Slice splats by coordinate ranges per dimension (numpy-style).

        Each slice specifies a [start, stop] range for that dimension's center
        coordinate. ``None`` in start/stop means unbounded.

        Args:
            slices: One slice per dimension. ``slice(lo, hi)`` keeps splats
                with center in [lo, hi]. ``slice(None, None)`` keeps all.

        Returns:
            New GSplatData with only splats inside all ranges.

        Raises:
            ValueError: If number of slices doesn't match ndim.

        Examples:
            >>> # Keep x in [0,50], all y, z in [10,90]
            >>> sliced = data.slice_by([slice(0, 50), slice(None, None), slice(10, 90)])
            >>>
            >>> # Open-ended: x >= 50
            >>> sliced = data.slice_by([slice(50, None), slice(None, None), slice(None, None)])
        """
        if len(slices) != self.ndim:
            raise ValueError(f"Expected {self.ndim} slices, got {len(slices)}")
        bbox = []
        for s in slices:
            lo = float(s.start) if s.start is not None else float("-inf")
            hi = float(s.stop) if s.stop is not None else float("inf")
            bbox.append((lo, hi))
        return self.filter_by(bbox=bbox)

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
            template = non_empty[0].substitutive_levels
            sub_levels: List[SubstitutiveLevel] = []
            for s in range(n_sub):
                views_s = [d.at_substitutive(s) for d in non_empty]
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
        from luxar.core.group.partition import (
            median_bsp_partition,
            midpoint_bsp_partition,
            sah_bsp_partition,
        )

        from .tree import GSplatLeaf, GSplatPartition

        if max_elements < 1:
            raise ValueError(f"max_elements must be >= 1, got {max_elements}")

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
        if rule == "median":
            parts = median_bsp_partition(centers, max_elements)
        elif rule == "midpoint":
            parts = midpoint_bsp_partition(centers, max_elements)
        elif rule == "sah":
            parts = sah_bsp_partition(centers, max_elements)
        else:
            raise ValueError(
                f"rule must be 'median', 'midpoint', or 'sah'; got {rule!r}"
            )
        children: List["GSplatNode"] = []
        for idx in parts:
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
        return GSplatPartition(children=children, max_elements=max_elements)

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
            new_lods = []
            offset = 0
            dim_mapping = list(range(d))
            fill_sigma = {d: sigma}
            for lod in self.additive_sublods:
                nl = lod.n_splats
                if is_scalar:
                    lod_col = np.full((nl, 1), values, dtype=lod.centers.dtype)
                else:
                    assert values_arr is not None
                    lod_col = values_arr[offset : offset + nl].reshape(nl, 1)
                lod_centers = np.concatenate([lod.centers, lod_col], axis=1)
                lod_cholesky = embed_cholesky_packed(
                    lod.cholesky_factors, d, d + 1, dim_mapping, fill_sigma
                )
                new_lods.append(
                    AdditiveSubLOD(
                        centers=lod_centers,
                        amplitudes=lod.amplitudes,
                        cholesky_factors=lod_cholesky,
                        colors=lod.colors,
                        stats=dict(lod.stats),
                        truncation_radius=lod.truncation_radius,
                    )
                )
                offset += nl
            return GSplatData.from_additive_sublods(new_lods, stats=dict(self.stats))

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
            out = fn(self.at_substitutive(s))
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
            new_lods = []
            for lod in self.additive_sublods:
                lod_centers = (lod.centers.astype(np.float64) @ A.T + t).astype(
                    lod.centers.dtype
                )
                lod_cholesky = _transform_cholesky(lod.cholesky_factors)
                new_lods.append(
                    AdditiveSubLOD(
                        centers=lod_centers,
                        amplitudes=lod.amplitudes,
                        cholesky_factors=lod_cholesky,
                        colors=lod.colors,
                        stats=dict(lod.stats),
                        truncation_radius=lod.truncation_radius,
                    )
                )
            return GSplatData.from_additive_sublods(new_lods, stats=dict(self.stats))

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
            new_lods = []
            offset = 0
            for lod in self.additive_sublods:
                n = lod.n_splats
                new_lods.append(
                    AdditiveSubLOD(
                        centers=lod.centers,
                        amplitudes=new_amplitudes[offset : offset + n],
                        cholesky_factors=lod.cholesky_factors,
                        colors=lod.colors,
                        stats=dict(lod.stats),
                        truncation_radius=lod.truncation_radius,
                    )
                )
                offset += n
            return GSplatData.from_additive_sublods(new_lods, stats=dict(self.stats))
        return GSplatData(
            centers=self.centers,
            amplitudes=new_amplitudes,
            cholesky_factors=self.cholesky_factors,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    def with_colors(
        self, colors: "np.ndarray | tuple[float, float, float]"
    ) -> "GSplatData":
        """Return a new GSplatData with replaced colors, preserving LODs.

        Args:
            colors: Either an (N, 3) array of per-splat colors, or a single
                (r, g, b) tuple/array to broadcast to all splats.

        Returns:
            New GSplatData with the specified colors.
        """
        arr = (
            colors
            if isinstance(colors, np.ndarray)
            else np.asarray(colors, dtype=np.float32)
        )
        is_broadcast = arr.ndim == 1 and arr.shape == (3,)

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
        if colors.shape != (self.n_splats, 3):
            raise ValueError(
                f"colors shape {colors.shape} doesn't match ({self.n_splats}, 3)"
            )
        if self.n_additive_sublods > 1:
            new_lods = []
            offset = 0
            for lod in self.additive_sublods:
                n = lod.n_splats
                new_lods.append(
                    AdditiveSubLOD(
                        centers=lod.centers,
                        amplitudes=lod.amplitudes,
                        cholesky_factors=lod.cholesky_factors,
                        colors=colors[offset : offset + n],
                        stats=dict(lod.stats),
                        truncation_radius=lod.truncation_radius,
                    )
                )
                offset += n
            return GSplatData.from_additive_sublods(new_lods, stats=dict(self.stats))
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

    # ── I/O ─────────────────────────────────────────────────

    def save(
        self,
        path: str | Path,
        ordering: Literal["morton", "hilbert", "none"] = "hilbert",
        encoding_mode: Optional["EncodingMode"] = None,
        include_fitting_info: bool = True,
        include_provenance: bool = False,
        description: Optional[str] = None,
        compress: Optional[Literal["zip", "tar.gz"]] = None,
        compressor: Any = _USE_DEFAULT_COMPRESSOR,
        zip_deflate: bool = False,
    ) -> None:
        """Save splats to .gsplats.zarr format.

        Args:
            path: Output path (should end with .gsplats.zarr or .gsplats.zarr.zip/.tar.gz if compress is used)
            ordering: Spatial ordering method ("morton", "hilbert", or "none")
            encoding_mode: Encoding mode (AUTO, PRECISION, or MEMORY), defaults to AUTO
            include_fitting_info: Whether to include fitting statistics
            include_provenance: Whether to include provenance info from stats
            description: Optional user description
            compress: Optional compression format ("zip" or "tar.gz"). Creates compressed archive.
            zip_deflate: Use DEFLATE compression for the outer zip (default: STORED).
                Useful when metadata overhead matters, e.g. for Git LFS storage.

        Colors are written via the shared COLOR helper, which auto-detects SDR vs
        HDR (values > 1) — there is no explicit ``color_mode`` knob.

        Example:
            >>> result = fit_gaussian_splats(image, n_iters=1000)
            >>> result.save("fitted.gsplats.zarr", encoding_mode=EncodingMode.MEMORY)
            >>> # With compression for storage/git-lfs
            >>> result.save("fitted.gsplats.zarr.zip", compress="zip")
        """
        from luxar.encoding import EncodingMode
        from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
        from luxar.io.reader import DEFAULT_COMP

        # Use AUTO as default
        if encoding_mode is None:
            encoding_mode = EncodingMode.AUTO

        # Use Blosc(zstd) by default; an EXPLICIT compressor=None disables
        # compression (e.g. for raw, zarrita-readable cross-language fixtures).
        # Only the sentinel "not specified" coerces to the default.
        if compressor is _USE_DEFAULT_COMPRESSOR:
            compressor = DEFAULT_COMP

        # Extract fitting/provenance groups from stats (single-sourced helper).
        fitting_info, fitting_config, provenance_info = split_fitting_info(
            self.stats,
            include_fitting_info=include_fitting_info,
            include_provenance=include_provenance,
        )

        # One authoring path: serialize this dataset's node tree to v3.0 via the
        # shared walker (the same machinery the scene compiler uses for leaves).
        write_gsplats_tree(
            path,
            self.tree,
            ordering=ordering,
            encoding_mode=encoding_mode,
            fitting_info=fitting_info,
            fitting_config=fitting_config,
            provenance_info=provenance_info,
            description=description,
            compress=compress,
            compressor=compressor,
            zip_deflate=zip_deflate,
        )

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
            new_lods = [
                AdditiveSubLOD(
                    centers=lod.centers + offset,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod.cholesky_factors,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
                for lod in self.additive_sublods
            ]
            return GSplatData.from_additive_sublods(new_lods, stats=dict(self.stats))

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

        The centroid is computed as the amplitude-weighted average of splat centers,
        which corresponds to the center of mass of the represented density.

        Returns:
            New GSplatData centered at origin (amplitude-weighted centroid at [0, 0, ...])

        Example:
            >>> # Center splats at origin for easier viewing
            >>> centered = data.center_at_centroid()
            >>> # Amplitude-weighted centroid is now at origin
            >>> centroid = (centered.centers.T @ centered.amplitudes) / centered.amplitudes.sum()
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

        # Translate to center at origin
        return self.translate(-centroid)

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

    def cull(
        self,
        target: np.ndarray | None = None,
        *,
        method: str = "auto",
        shape: tuple[int, ...] | None = None,
        truncate: float | None = None,
        # --- error_budget / redundancy params ---
        error_percentile: float = 99.0,
        error_tolerance: float = 1.0,
        redundancy_threshold: float = 0.01,
        max_binary_search_iters: int = 8,
        device: str | None = None,
        intensity_floor: float = 1e-5,
        # --- heuristic params ---
        retention: float = 0.95,
        amplitude_percentile: float = 5.0,
        volume_percentile: float = 95.0,
        verbose: bool = False,
    ) -> "GSplatData":
        """Cull splats that contribute negligibly to the reconstruction.

        This is the unified entry point for all splat removal strategies,
        from fast heuristics to principled contribution-based methods.
        The ``method`` parameter selects which strategy to use.

        Methods (ordered from cheapest to most principled)
        --------------------------------------------------

        **"cumulative"** — Keep the top splats that account for a target
        fraction of the total amplitude.  Fast (no rendering), but blind
        to spatial overlap: a low-amplitude splat covering a unique region
        will be removed even though it is the sole contributor there.

            >>> data.cull(method="cumulative", retention=0.95)

        **"amplitude_percentile"** — Remove splats in the bottom X
        percentile of amplitude.  Same limitation as cumulative: ignores
        spatial context.

            >>> data.cull(method="amplitude_percentile", amplitude_percentile=10)

        **"combined"** — Remove splats that have low amplitude OR unusually
        large volume (artifacts).  Useful as a quick cleanup pass.

            >>> data.cull(method="combined", amplitude_percentile=5, volume_percentile=95)

        **"redundancy"** — Render the full reconstruction and measure each
        splat's maximum *fractional contribution* ``g_j(x) / V_pred(x)``.
        If a splat never contributes more than ``redundancy_threshold`` of
        the local signal, it is redundant.  Does not need the target volume
        but requires GPU rendering.

            >>> data.cull(method="redundancy", shape=(128,128,128), redundancy_threshold=0.02)

        **"error_budget"** — The most principled mode.  Requires the
        original target volume.  Computes the residual ``R = target - V_pred``
        and derives an error budget from it.  A splat is safe to remove when
        the worst-case error *increase* from its removal is below the budget.
        Robust to pre-existing noise and accounts for spatial redundancy.

            >>> data.cull(target_volume, method="error_budget", error_percentile=99)

        **"auto"** (default) — Selects automatically:
        ``"error_budget"`` if *target* is provided, ``"redundancy"`` if
        *shape* is provided, ``"cumulative"`` otherwise.

        Joint compounding check (error_budget and redundancy only)
        ----------------------------------------------------------
        After identifying individual candidates, verifies that their
        *joint* removal does not exceed the budget.  If it does, a binary
        search tightens the per-splat threshold until the joint constraint
        holds, guaranteeing that the combined removal is safe.

        Args:
            target: Original target volume.  If provided and ``method="auto"``,
                selects error-budget mode.
            method: Culling strategy.  One of ``"auto"``, ``"error_budget"``,
                ``"redundancy"``, ``"cumulative"``, ``"amplitude_percentile"``,
                ``"combined"``.
            shape: Volume shape for rendering (error_budget / redundancy).
                Defaults to ``target.shape`` when target is provided.
            truncate: Truncation radius in standard deviations.
                Defaults to ``self.truncation_radius``.
            error_percentile: *error_budget only.*  Percentile of ``|residual|``
                for the budget (0--100).
            error_tolerance: *error_budget only.*  Multiplier on the budget.
            redundancy_threshold: *redundancy only.*  Max fractional
                contribution (0--1) below which a splat is redundant.
            max_binary_search_iters: *error_budget / redundancy only.*
                Max iterations for the joint compounding binary search.
            device: Device for GPU computation.  Auto-detected if None.
            intensity_floor: Min intensity threshold for AABB computation.
            retention: *cumulative only.*  Fraction of total amplitude to
                retain (0--1).
            amplitude_percentile: *amplitude_percentile / combined only.*
                Bottom percentile to remove (0--100).
            volume_percentile: *combined only.*  Remove splats above this
                volume percentile (0--100).
            verbose: Print progress information.

        Returns:
            New GSplatData with culled splats removed.  Stats include
            ``culled``, ``culling_method``, ``n_original``, ``n_culled``.
        """
        if truncate is None:
            truncate = self.truncation_radius

        # --- Resolve "auto" method ---
        if method == "auto":
            if target is not None:
                method = "error_budget"
            elif shape is not None:
                method = "redundancy"
            else:
                method = "cumulative"

        # Multi-substitutive: cull EVERY substitutive level and rebuild the
        # pyramid (decision 6) rather than collapsing to the default level via
        # the single-level mask that the strategies below feed to self.filter().
        # Each level is culled through the single-substitutive path (the same
        # target volume reconstructs every level). Mirrors filter_by().
        if self.n_substitutive > 1:
            culled_levels: List[SubstitutiveLevel] = []
            for s, src in enumerate(self.substitutive_levels):
                culled_level = self.at_substitutive(s).cull(
                    target,
                    method=method,
                    shape=shape,
                    truncate=truncate,
                    error_percentile=error_percentile,
                    error_tolerance=error_tolerance,
                    redundancy_threshold=redundancy_threshold,
                    max_binary_search_iters=max_binary_search_iters,
                    device=device,
                    intensity_floor=intensity_floor,
                    retention=retention,
                    amplitude_percentile=amplitude_percentile,
                    volume_percentile=volume_percentile,
                    verbose=verbose,
                )
                culled_levels.append(
                    SubstitutiveLevel(
                        additive_sublods=culled_level.substitutive_levels[
                            0
                        ].additive_sublods,
                        compression_factor=src.compression_factor,
                        parent_method=src.parent_method,
                        level_index=src.level_index,
                        stats=dict(src.stats),
                    )
                )
            out = GSplatData.from_substitutive_levels(
                culled_levels,
                stats=dict(self.stats),
            )
            out.stats.update(
                {
                    "culled": True,
                    "culling_method": method,
                    "n_original": self.n_splats,
                    "n_culled": self.n_splats - out.n_splats,
                }
            )
            return out

        # =================================================================
        # Heuristic methods (no rendering, CPU-only, fast)
        # =================================================================
        if method in ("cumulative", "amplitude_percentile", "combined"):
            return self._cull_heuristic(
                method=method,
                retention=retention,
                amplitude_percentile=amplitude_percentile,
                volume_percentile=volume_percentile,
            )

        # =================================================================
        # Rendering-based methods (GPU, contribution-aware)
        # =================================================================
        if method not in ("error_budget", "redundancy"):
            raise ValueError(
                f"Unknown culling method: {method!r}. "
                "Choose from: 'auto', 'error_budget', 'redundancy', "
                "'cumulative', 'amplitude_percentile', 'combined'."
            )

        import torch

        from luxar.gsplats.culling import cull_by_contribution
        from luxar.gsplats.rendering.volume_rendering import auto_detect_device

        if target is not None and shape is None:
            shape = target.shape
        if shape is None:
            raise ValueError(
                "shape is required for error_budget/redundancy modes. "
                "Pass the volume shape, e.g. shape=(128, 128, 128), "
                "or provide a target volume."
            )

        if device is None:
            device = auto_detect_device()

        # Convert to GPU tensors
        centers_t = torch.from_numpy(self.centers.astype(np.float32)).to(device)
        amps_t = torch.from_numpy(self.amplitudes.astype(np.float32)).to(device)
        target_t = (
            torch.from_numpy(target.astype(np.float32)).to(device)
            if target is not None
            else None
        )

        # Unpack Cholesky factors: (N, d*(d+1)/2) -> (N, d, d) lower-triangular
        chol = self.cholesky_factors
        ndim = self.ndim
        chol_t = torch.from_numpy(chol).to(device)
        Ls_t = torch.zeros((len(chol), ndim, ndim), device=device, dtype=torch.float32)
        if ndim == 2:
            Ls_t[:, 0, 0] = chol_t[:, 0]
            Ls_t[:, 1, 0] = chol_t[:, 1]
            Ls_t[:, 1, 1] = chol_t[:, 2]
        elif ndim == 3:
            Ls_t[:, 0, 0] = chol_t[:, 0]
            Ls_t[:, 1, 0] = chol_t[:, 1]
            Ls_t[:, 1, 1] = chol_t[:, 2]
            Ls_t[:, 2, 0] = chol_t[:, 3]
            Ls_t[:, 2, 1] = chol_t[:, 4]
            Ls_t[:, 2, 2] = chol_t[:, 5]
        else:
            idx = 0
            for i in range(ndim):
                for j in range(i + 1):
                    Ls_t[:, i, j] = chol_t[:, idx]
                    idx += 1
        del chol_t

        result = cull_by_contribution(
            centers_t,
            Ls_t,
            amps_t,
            target_t,
            shape,
            truncate=truncate,
            error_percentile=error_percentile,
            error_tolerance=error_tolerance,
            redundancy_threshold=redundancy_threshold,
            max_binary_search_iters=max_binary_search_iters,
            intensity_floor=intensity_floor,
            verbose=verbose,
        )

        # Free GPU tensors used for culling
        del centers_t, amps_t, Ls_t
        if target_t is not None:
            del target_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        culled = self.filter(result.keep_mask)
        culled.stats.update(
            {
                "culled": True,
                "culling_method": result.mode,
                "n_original": self.n_splats,
                "n_culled": result.n_culled,
                "error_budget": result.error_budget,
                "phase1_candidates": result.phase1_candidates,
                "phase2_iterations": result.phase2_iterations,
                "max_joint_error": result.max_joint_error,
            }
        )
        return culled

    def _cull_heuristic(
        self,
        method: str,
        retention: float = 0.95,
        amplitude_percentile: float = 5.0,
        volume_percentile: float = 95.0,
    ) -> "GSplatData":
        """Heuristic culling methods (no rendering needed)."""
        N_original = self.n_splats

        if N_original == 0:
            result = self.filter(np.ones(0, dtype=bool))
            result.stats.update(
                {
                    "culled": True,
                    "culling_method": method,
                    "n_original": 0,
                    "n_culled": 0,
                }
            )
            return result

        mask = np.ones(N_original, dtype=bool)

        if method == "cumulative":
            sorted_indices = np.argsort(self.amplitudes)[::-1]
            sorted_amps = self.amplitudes[sorted_indices]
            cumsum_amps = np.cumsum(sorted_amps)
            total_amp = cumsum_amps[-1]
            if total_amp == 0:
                mask = (
                    np.ones(N_original, dtype=bool)
                    if retention > 0
                    else np.zeros(N_original, dtype=bool)
                )
            else:
                cumsum_norm = cumsum_amps / total_amp
                n_keep = np.searchsorted(cumsum_norm, retention) + 1
                n_keep = min(n_keep, N_original)
                keep_indices = sorted_indices[:n_keep]
                mask = np.zeros(N_original, dtype=bool)
                mask[keep_indices] = True

        elif method == "amplitude_percentile":
            threshold = np.percentile(self.amplitudes, amplitude_percentile)
            mask = self.amplitudes >= threshold

        elif method == "combined":
            vols = self.volumes()
            amp_threshold = np.percentile(self.amplitudes, amplitude_percentile)
            vol_threshold = np.percentile(vols, volume_percentile)
            mask = (self.amplitudes >= amp_threshold) & (vols <= vol_threshold)

        else:
            raise ValueError(f"Unknown heuristic method: {method!r}")

        result = self.filter(mask)

        total_amp = np.sum(self.amplitudes)
        result.stats.update(
            {
                "culled": True,
                "culling_method": method,
                "n_original": N_original,
                "n_culled": N_original - result.n_splats,
                "amplitude_retention": (
                    float(np.sum(result.amplitudes) / total_amp)
                    if total_amp > 0
                    else 1.0
                ),
            }
        )
        return result

    def render_to_volume(
        self,
        shape: tuple[int, ...],
        device: str | None = None,
        truncate: float | None = None,
        intensity_floor: float = 1e-5,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """Render Gaussian splats to a volume using GPU-accelerated rendering.

        This is a convenience method that automatically selects the fastest available
        backend (CUDA, MPS, or CPU) and uses the optimized PyTorch renderer.

        Parameters
        ----------
        shape : tuple[int, ...]
            Output volume shape (e.g., (128, 128, 128) for 3D).
        device : str, optional
            Device to use for rendering. If None, auto-detects the best device.
            Options: "cuda", "mps", "cpu".
        truncate : float, optional
            Truncation radius in standard deviations. Gaussians are evaluated within
            this radius from their centers. Defaults to ``self.truncation_radius``.
        intensity_floor : float, default=1e-5
            Minimum intensity threshold for amplitude-aware culling. Splats with
            contributions below this threshold are culled early for performance.
        chunk_size : int, optional
            Chunk size for memory management when processing large volumes. If None,
            automatically calculated based on available memory.

        Returns
        -------
        np.ndarray
            Rendered volume with the specified shape.

        Examples
        --------
        >>> # Render to 128³ volume
        >>> volume = gsplat_data.render_to_volume(shape=(128, 128, 128))
        >>>
        >>> # Force CPU rendering
        >>> volume = gsplat_data.render_to_volume(shape=(128, 128, 128), device="cpu")
        >>>
        >>> # Use larger truncation radius
        >>> volume = gsplat_data.render_to_volume(shape=(128, 128, 128), truncate=4.0)

        Notes
        -----
        - For 8K splats on 128³ volume: substantially faster than NumPy
          implementation (often orders of magnitude on GPU; varies by hardware)
        - Automatically chunks large volumes to prevent out-of-memory errors
        - Uses specialized fast paths for 2D/3D rendering
        """
        if truncate is None:
            truncate = self.truncation_radius

        from luxar.gsplats.rendering.volume_rendering import render_to_volume

        return render_to_volume(
            self,
            shape=tuple(shape),
            device=device,
            truncate=truncate,
            intensity_floor=intensity_floor,
            chunk_size=chunk_size,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        include_stats: bool = False,
    ) -> "GSplatData":
        """Load splats from .gsplats.zarr format.

        Args:
            path: Path to .gsplats.zarr directory
            include_stats: Whether to include fitting/provenance metadata

        Returns:
            GSplatData with decoded arrays

        Example:
            >>> data = GSplatData.load("fitted.gsplats.zarr")
            >>> aprint(data.centers.shape)
        """
        from luxar.gsplats.io.load_gsplats import load_gsplats

        return load_gsplats(path, include_stats=include_stats)

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
