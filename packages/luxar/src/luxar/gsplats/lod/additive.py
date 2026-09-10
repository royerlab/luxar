"""Additive Levels-of-Detail for Gaussian splats.

Implements the additive-LOD algorithms from the supplementary document
``additive_lod`` (luxar-paper/supp_doc/additive_lod): given a fitted
:class:`GSplatData`, compute a permutation that orders the splats so the
prefix sum approximates the full scene at every intermediate $k$, and
split the ordered set into ``n_lods`` levels.

Algorithms (additive_lod §3-4)
------------------------------

- ``random``       — uniform permutation; baseline.
- ``amplitude``    — sort by peak amplitude $a_i$, descending.
- ``mass``         — sort by integral mass $m_i \\propto a_i\\,|\\Sigma_i|^{1/2}$.
- ``self_energy``  — sort by $L^2$ self-energy $\\|\\phi_i\\|^2 \\propto a_i^2\\,|\\Sigma_i|^{1/2}$.
- ``spectral``     — sort by $|u_1[i]|$, leading eigenvector of the Gram matrix.
- ``greedy``       — submodular greedy / matching pursuit.  $(1-1/e)$ optimal
  at every prefix simultaneously (Nemhauser--Wolsey--Fisher 1978); empirically
  $\\geq 99.9\\%$ of the exhaustive optimum on dense-overlap instances.

Modifiers (compose with EVERY method above)
--------------------------------------------

- ``slice_dims`` — :func:`interleave_order_across_slices` re-emits the chosen
  ordering round-robin across the distinct coordinates of the named centre
  columns, so every PREFIX is slice-even. On a node the viewer SLICES (a hidden
  time/channel dimension) a rung is sized against the whole node but only one
  coordinate is ever on screen, so a global contribution-ordered prefix starves
  the sparse coordinates; this gives each of them an equal ABSOLUTE budget
  instead. Deterministic (no seed) and idempotent. Not an ordering method — it
  is applied *after* one, and the within-coordinate order stays whatever the
  method produced.

The greedy path uses a sparse Gram matrix built via Mahalanobis truncation at
the dataset's own ``truncation_radius`` (the support it was fitted and is
rendered at) + k-d-tree pruning (Algorithm 4.4 in the supp doc), keeping
memory at $O(\\mathrm{nnz}(\\mathbf{G}))$.  At $N \\leq 2000$ a dense Gram
+ scan-greedy is faster than the heap-based lazy greedy due to Python
overhead (supp doc §4.3); we switch automatically.
"""

from __future__ import annotations

import heapq
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from luxar.gsplats.gsplat_data import (
    AdditiveSubLOD,
    GSplatData,
    SubstitutiveLevel,
)
from luxar.gsplats.lod._kernels import (
    gaussian_pair_inner_product_numpy,
    gaussian_self_energy_numpy,
    truncation_radii_numpy,
)
from luxar.gsplats.spatial_hash import BatchedSpatialHashGrid
from luxar.gsplats.utils.alpha import effective_amplitudes
from luxar.gsplats.utils.trils import unpack_tril
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from luxar.utils.lod_breakpoints import DEFAULT_BANDWIDTH_MBPS as DEFAULT_BANDWIDTH_MBPS
from luxar.utils.lod_breakpoints import (
    DEFAULT_STREAM_MAX_LEVELS as DEFAULT_STREAM_MAX_LEVELS,
)
from luxar.utils.lod_breakpoints import (
    EQUI_ENERGY_PREFIX,
    equi_energy_cuts,
    parse_equi_energy_rungs,
    parse_stream_chunk,
    stream_cuts,
)
from luxar.utils.lod_breakpoints import BreakpointSpec as _BreakpointSpec
from luxar.utils.lod_breakpoints import (
    sibling_aware_stream_breakpoints as sibling_aware_stream_breakpoints,
)
from luxar.utils.lod_breakpoints import (
    streaming_chunk_splats as streaming_chunk_splats,
)
from luxar.utils.lod_methods import GSPLAT_ADDITIVE_CHOICES, GSPLAT_ADDITIVE_METHODS
from luxar.utils.lod_methods import AutoOrMethod as AutoOrMethod
from luxar.utils.lod_methods import MethodName as MethodName
from luxar.utils.lod_methods import is_reveal_method as _is_reveal_method
from luxar.validation.types import (
    validate_finite_reveal_coords,
    validate_integral_axis_indices,
)

# The method registry lives in `luxar.utils.lod_methods` so the CLI can share it
# without importing this package (`luxar/gsplats/__init__.py` adds ~600 ms on top
# of the CLI's own ~250 ms import — a 3.4x multiplier on `luxar --help`).
# Re-exported under the historical names — `MethodName` / `AutoOrMethod` are
# imported from here by `pyramid` and `recipes`.
_VALID_METHODS: tuple[str, ...] = GSPLAT_ADDITIVE_METHODS
_VALID_CHOICES: tuple[str, ...] = GSPLAT_ADDITIVE_CHOICES

#: ``method="auto"`` resolves to ``greedy`` (the Minoux 1978 lazy-greedy
#: submodular selection in :func:`_lazy_greedy` — provably (1-1/e)-optimal at
#: every prefix) at or below this count, else ``self_energy`` (a cheap
#: O(N log N) vectorized ranking). The dominant cost of the greedy/spectral path
#: is NOT the lazy-heap pass (sub-second even at 20K) but :func:`_build_sparse_gram`,
#: whose pure-Python per-pair loop is O(nnz) and scales with *overlap density*
#: (avg neighbours per splat), not N alone — so a dense scientific volume can
#: make even a modest part slow (this is what made a 1.5 M-splat part hang). N is
#: only a crude proxy for nnz, so the threshold is set conservatively (mirrors
#: substitutive LOD's ``_AUTO_GREEDY_MAX_N=5000``); above it ``self_energy`` never
#: builds the Gram. Tunable. (NB: "Runnalls" is the *substitutive* Gaussian-
#: mixture-reduction greedy — distinct from this additive Minoux lazy-greedy.)
_AUTO_ADDITIVE_MAX_N = 5_000


def resolve_additive_method(method: AutoOrMethod, n: int) -> MethodName:
    """Resolve ``method`` for ``n`` splats, handling the ``"auto"`` sentinel.

    ``auto`` → ``greedy`` when ``n <= _AUTO_ADDITIVE_MAX_N`` (high quality and
    affordable at small N), else ``self_energy`` (avoids the O(nnz) sparse-Gram
    build that greedy/spectral need, which blows up with overlap density on large
    inputs). A concrete method passes through unchanged.
    """
    if method != "auto":
        return method
    return "greedy" if n <= _AUTO_ADDITIVE_MAX_N else "self_energy"


def resolve_truncation_sigmas(
    truncation_sigmas: float | None, data: GSplatData
) -> float:
    """Resolve the σ multiplier used for Gaussian truncation, honouring the data.

    ``None`` (the default everywhere on the LOD path) means "use the dataset's
    own ``truncation_radius``" — the support the splats were *fitted* at and are
    *rendered* at. The ladder used to hard-code 3.0, so a dataset fitted at the
    canonical :data:`~luxar.typing_utils.constants.DEFAULT_TRUNCATION_RADIUS`
    (2.75) was pruned at a support it never had. ``getattr`` with that same
    constant as fallback mirrors the defensive read in
    ``GSplatData.principal_radii`` (``gsplats/_data/metrics.py``): it covers the
    (structural) case of a data-like object that exposes no radius at all, and —
    since ``getattr`` swallows any ``AttributeError``, including one raised
    *inside* the property (``truncation_radius`` → ``additive_sublods[0]``) — a
    mis-wired object too, which prunes at the constant rather than failing.

    An explicit value is checked here, locally: this σ is a *CPU pruning* cutoff
    (which pairs enter the sparse Gram), not a render uniform, so it carries no
    float32/shader bounds — any finite positive value is meaningful. Only a
    *degenerate* cutoff is rejected, because it poisons the per-splat truncation
    radii that feed the k-d-tree pair search and the Gram entries (``0`` → all
    zero, negative → negative, NaN/inf → NaN/inf radii). A legitimately tiny σ
    is accepted on purpose: it merely yields a diagonal-only Gram, degrading the
    greedy ordering toward the score-only one rather than being an error.
    """
    if truncation_sigmas is None:
        return float(getattr(data, "truncation_radius", DEFAULT_TRUNCATION_RADIUS))
    sigmas = float(truncation_sigmas)
    if not math.isfinite(sigmas) or sigmas <= 0.0:
        raise ValueError(
            "truncation_sigmas must be a finite value > 0 (it is the Mahalanobis "
            "cutoff for sparse-Gram pruning; a non-positive or non-finite cutoff "
            "poisons every per-splat truncation radius, which is what the pair "
            f"search and the Gram entries are built from), got {truncation_sigmas!r}"
        )
    return sigmas


#: Breakpoint specification for the additive ladder. String forms:
#: ``"equal-count"`` (n_lods equal levels) and ``"stream:<c>"`` (geometric
#: cumulative cuts ``[c, 2c, 4c, …, N]`` — a bandwidth-derived first chunk that
#: doubles; resolved per-N inside :func:`_resolve_breakpoints`, so the same spec
#: adapts to every part/level size). List forms: ``list[int]`` explicit
#: cumulative counts; ``list[float]`` cumulative energy fractions in (0, 1].
#:
#: The cut geometry itself lives in :mod:`luxar.utils.lod_breakpoints` so all
#: three geometries derive identical cuts from an identical spec; the names below
#: are re-exported here because they are part of this module's public surface.
BreakpointSpec = _BreakpointSpec


def clamp_counts_breakpoints(breakpoints: BreakpointSpec, n: int) -> BreakpointSpec:
    """Clamp explicit ``counts:`` breakpoints to a part/level of ``n`` splats.

    Per-part and per-level ladders (BSP parts, pyramid levels) have differing
    N; a fixed ``counts:`` list whose largest cut exceeds a small part would
    otherwise abort the whole build via ``_resolve_breakpoints``'s strict
    "largest breakpoint exceeds N" check (which is the RIGHT behavior for a
    direct whole-dataset build, where the user knows N). This helper keeps the
    cuts below ``n`` and lets ``_resolve_breakpoints`` append the final ``n``;
    non-count specs (strings, energy fractions) pass through unchanged — they
    are already size-adaptive.
    """
    if isinstance(breakpoints, str) or n <= 0:
        return breakpoints
    if not isinstance(breakpoints, (list, tuple)) or len(breakpoints) == 0:
        return breakpoints
    if not all(
        isinstance(x, (int, np.integer)) and not isinstance(x, bool)
        for x in breakpoints
    ):
        return breakpoints  # energy fractions (or invalid — let validation raise)
    kept = [int(c) for c in breakpoints if int(c) < n]
    return kept if kept else [int(n)]


def validate_counts_breakpoints(breakpoints: BreakpointSpec, n: int) -> None:
    """Strictly validate explicit ``counts:`` breakpoints against the FULL ``n``.

    The whole-dataset companion of :func:`clamp_counts_breakpoints`: clamping
    is right for an individual part/level whose N the user cannot know, but
    the spec itself must still be sane for the dataset as a whole — a largest
    count exceeding the full N is a typo (e.g. ``counts:1000000`` on a 50 k
    dataset) and must abort loudly, exactly like a direct whole-dataset
    :func:`make_additive_lod` build does via ``_resolve_breakpoints``. Callers
    that clamp per part/level call this ONCE up front with the union /
    finest-level size. Non-count specs pass through (validated downstream).
    """
    if isinstance(breakpoints, str) or n <= 0:
        return
    if not isinstance(breakpoints, (list, tuple)) or len(breakpoints) == 0:
        return
    if not all(
        isinstance(x, (int, np.integer)) and not isinstance(x, bool)
        for x in breakpoints
    ):
        return  # energy fractions (or invalid — let downstream validation raise)
    largest = max(int(c) for c in breakpoints)
    if largest > n:
        raise ValueError(
            f"largest breakpoint {largest} exceeds N={n} (the full dataset); "
            "explicit counts: breakpoints must fit the dataset "
            "(smaller parts/levels clamp automatically)"
        )


# ─────────────────────────────────────────────────────────────────────
# Score-based orderings
# ─────────────────────────────────────────────────────────────────────


def _det_L(data: GSplatData) -> np.ndarray:
    """Per-splat product of Cholesky diagonals: $|L_i| = |\\Sigma_i|^{1/2}$."""
    diag = data._cholesky_diag_elements()  # (N, d)
    out: np.ndarray = np.abs(np.prod(diag.astype(np.float64), axis=1))
    return out


def _self_energy_score(data: GSplatData) -> np.ndarray:
    """Self-energy ranking score: $a_i^2 \\, |\\Sigma_i|^{1/2}$.

    The $\\pi^{D/2}$ constant is shared across all splats and drops out
    of the ordering. Amplitudes are alpha-effective ($A_i·a_i$): every
    blending mode scales a splat's contribution by its color alpha, so
    ordering by raw $A$ would misrank imported classical splats (amplitude 1,
    weight in alpha).
    """
    amps = np.asarray(effective_amplitudes(data), dtype=np.float64)
    out: np.ndarray = amps**2 * _det_L(data)
    return out


def _mass_score(data: GSplatData) -> np.ndarray:
    """Integral-mass ranking score: $a_i \\, |\\Sigma_i|^{1/2}$.

    The $(2\\pi)^{D/2}$ constant is shared across all splats and drops
    out of the ordering. Amplitudes are alpha-effective (see
    ``_self_energy_score``).
    """
    amps = np.asarray(effective_amplitudes(data), dtype=np.float64)
    out: np.ndarray = amps * _det_L(data)
    return out


def _truncation_radii(data: GSplatData, *, sigmas: float) -> np.ndarray:
    """Per-splat truncation radius $r_i = \\sigma\\,\\sqrt{\\lambda_{\\max}(\\Sigma_i)}$.

    ``sigmas`` is REQUIRED (no default): a default here is what let the ladder
    prune at 3.0 while the dataset was fitted at its own ``truncation_radius``.
    Callers resolve it with :func:`resolve_truncation_sigmas`.
    """
    if data.n_splats == 0:
        return np.empty(0, dtype=np.float64)
    L = unpack_tril(np.asarray(data.cholesky_factors, dtype=np.float64), data.ndim)
    return truncation_radii_numpy(L, sigmas=sigmas)


# ─────────────────────────────────────────────────────────────────────
# Sparse Gram (σ-truncation Mahalanobis pruning + k-d tree, supp doc Algo 4.4)
# ─────────────────────────────────────────────────────────────────────


def _build_sparse_gram(data: GSplatData, *, sigmas: float) -> sparse.csr_matrix:
    """Build a sparse symmetric Gram matrix via $\\sigma$-truncation pruning.

    ``sigmas`` is REQUIRED (see :func:`_truncation_radii`); resolve it from the
    dataset with :func:`resolve_truncation_sigmas`.

    Two splats whose $\\sigma$-truncation ellipsoids do not overlap have an
    inner product $K_{ij}$ negligibly small (and exactly zero under the
    truncation convention used at render time); we treat those entries
    as structural zeros.  The remaining entries are the closed-form
    Gaussian inner products from ``additive_lod`` Eq. (2.1).
    """
    N = data.n_splats
    D = data.ndim
    if N == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64)

    centers = np.asarray(data.centers, dtype=np.float64)
    # Alpha-effective amplitudes: greedy ranks by RENDERED energy, and every
    # blending mode scales contribution by the color alpha.
    amps = np.asarray(effective_amplitudes(data), dtype=np.float64)
    L = unpack_tril(np.asarray(data.cholesky_factors, dtype=np.float64), D)
    Sigma = L @ L.transpose(0, 2, 1)
    sqrt_det_Sigma = _det_L(data)  # |Σ_i|^{1/2}
    radii = _truncation_radii(data, sigmas=sigmas)
    r_max = float(radii.max()) if N > 0 else 0.0

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # Diagonal: K_ii = a_i^2 * π^(D/2) * |Σ_i|^{1/2}
    diag_vals = gaussian_self_energy_numpy(amps, sqrt_det_Sigma, ndim=D)
    for i in range(N):
        rows.append(i)
        cols.append(i)
        vals.append(float(diag_vals[i]))

    if N > 1 and r_max > 0.0:
        # All per-query radii ``radii[i] + r_max`` are <= ``2 * r_max``;
        # build the spatial hash with that as cell size and run a single
        # batched radius query. Per-pair tightening to
        # ``radii[i] + radii[j]`` happens in the inner filter below.
        coarse_radius = 2.0 * r_max
        grid = BatchedSpatialHashGrid.from_points(
            centers, cell_size=coarse_radius, device="auto"
        )
        candidates_per_i = grid.query_radius(centers, radius=coarse_radius)
        for i in range(N):
            for j in candidates_per_i[i]:
                j = int(j)
                if j <= i:
                    continue
                d = centers[i] - centers[j]
                d_norm = float(np.linalg.norm(d))
                if d_norm > radii[i] + radii[j]:
                    continue
                K_ij = gaussian_pair_inner_product_numpy(
                    centers[i],
                    Sigma[i],
                    float(amps[i]),
                    centers[j],
                    Sigma[j],
                    float(amps[j]),
                    sqrt_det_Sigma_i=float(sqrt_det_Sigma[i]),
                    sqrt_det_Sigma_j=float(sqrt_det_Sigma[j]),
                )
                if K_ij <= 0.0:
                    continue
                rows.append(i)
                cols.append(j)
                vals.append(K_ij)
                rows.append(j)
                cols.append(i)
                vals.append(K_ij)

    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(N, N),
    )


# ─────────────────────────────────────────────────────────────────────
# Greedy / matching pursuit (supp doc Algos 4.1-4.2)
# ─────────────────────────────────────────────────────────────────────


def _dense_greedy(gram: np.ndarray) -> np.ndarray:
    """Naive $O(N^2)$ scan-greedy.  Faster than heap at small $N$ in pure Python."""
    N = gram.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    sigma = gram.sum(axis=1).astype(np.float64)
    G_diag = np.diag(gram).astype(np.float64).copy()
    chosen = np.zeros(N, dtype=bool)
    order = np.empty(N, dtype=np.int64)

    for k in range(N):
        delta = 2.0 * sigma - G_diag
        delta[chosen] = -np.inf
        i_star = int(np.argmax(delta))
        order[k] = i_star
        chosen[i_star] = True
        sigma -= gram[:, i_star]

    return order


def _lazy_greedy(gram_csr: sparse.csr_matrix) -> np.ndarray:
    """Lazy greedy on sparse Gram (Minoux 1978; supp doc Algorithm 4.2)."""
    N = gram_csr.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    sigma = np.asarray(gram_csr.sum(axis=1)).ravel().astype(np.float64)
    G_diag = gram_csr.diagonal().astype(np.float64)
    gram_csc = gram_csr.tocsc()

    chosen = np.zeros(N, dtype=bool)
    order = np.empty(N, dtype=np.int64)

    # Heap entries: (-marginal_ub, i, stale_step).  Negated for max-heap.
    heap: list[tuple[float, int, int]] = []
    for i in range(N):
        delta_i = float(2.0 * sigma[i] - G_diag[i])
        heapq.heappush(heap, (-delta_i, i, 0))

    for k in range(N):
        i_star = -1
        while heap:
            neg_delta, i, stale = heap[0]
            if chosen[i]:
                heapq.heappop(heap)
                continue
            if stale == k:
                heapq.heappop(heap)
                i_star = i
                break
            heapq.heappop(heap)
            delta_i = float(2.0 * sigma[i] - G_diag[i])
            heapq.heappush(heap, (-delta_i, i, k))
        if i_star < 0:
            # Heap empty — pick any remaining (defensive; shouldn't happen)
            remaining = np.where(~chosen)[0]
            if remaining.size == 0:
                break
            i_star = int(remaining[0])

        order[k] = i_star
        chosen[i_star] = True
        col = gram_csc.getcol(int(i_star)).tocoo()
        sigma[col.row] -= col.data

    return order


def _spectral_order(gram_csr: sparse.csr_matrix) -> np.ndarray:
    """Sort by $|u_1[i]|$ descending, where $u_1$ is the leading eigenvector."""
    N = gram_csr.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)
    if N == 1:
        return np.array([0], dtype=np.int64)
    eigvals, eigvecs = eigsh(gram_csr.astype(np.float64), k=1, which="LA")
    u1 = eigvecs[:, 0]
    return np.argsort(-np.abs(u1)).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────
# Residual-energy curve (used by energy-fraction breakpoints)
# ─────────────────────────────────────────────────────────────────────


def _residual_energy_curve(
    gram_csr: sparse.csr_matrix, order: np.ndarray
) -> np.ndarray:
    """Compute $E_k = \\|f - g_k\\|^2$ for $k = 0,\\dots,N$ along ``order``.

    $E_0 = \\|f\\|^2 = \\mathbf{1}^\\top \\mathbf{G}\\,\\mathbf{1}$;
    $E_N = 0$.  Recurrence:
    $\\Delta_i(S_{k-1}) = 2\\sigma_i^{(k-1)} - \\mathbf{G}_{ii}$,
    $E_k = E_{k-1} - \\Delta_{i_k}$.
    """
    N = gram_csr.shape[0]
    if N == 0:
        return np.zeros(1, dtype=np.float64)

    sigma = np.asarray(gram_csr.sum(axis=1)).ravel().astype(np.float64)
    G_diag = gram_csr.diagonal().astype(np.float64)
    gram_csc = gram_csr.tocsc()

    E = np.empty(N + 1, dtype=np.float64)
    E[0] = float(gram_csr.sum())
    for k in range(N):
        i = int(order[k])
        delta = 2.0 * sigma[i] - G_diag[i]
        E[k + 1] = max(E[k] - delta, 0.0)
        col = gram_csc.getcol(i).tocoo()
        sigma[col.row] -= col.data

    return E


# ─────────────────────────────────────────────────────────────────────
# Public API: ordering
# ─────────────────────────────────────────────────────────────────────


def _radial_score(
    data: GSplatData,
    centre: Sequence[float] | None = None,
    spatial_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Distance of each splat from ``centre``, over the SPATIAL axes only.

    The ordering key for ``method="radial"`` — the concentric-shell reveal. This
    is a presentation choice, not an error metric: it exists so a streaming
    additive ladder visibly grows outward from the middle of the object rather
    than filling in by contribution (which reads as confetti).

    ``centre`` defaults to the **bounding-box centre of the spatial axes**, not
    the scene origin: a dataset sitting far from the origin would otherwise
    reveal from one corner instead of growing from its own middle.

    ``spatial_dims`` defaults to :meth:`GSplatData._nondegenerate_axes` — the
    shared "axis with real covariance extent" rule. That matters here: a stacked
    time or channel axis has zero variance, and including it would make the
    shells expand through TIME as well as space (every timepoint of the innermost
    shell before any of the next), which is not a reveal.

    An explicit ``spatial_dims`` is validated rather than trusted, mirroring the
    element-side :func:`~luxar.core.group.lod.reveal.radial_element_score`: every
    rejected case silently produced a WRONG ordering instead of an error — a
    negative index ALIASES to another column under numpy indexing, a repeat
    DOUBLE-COUNTS that axis in the distance, an empty selection scores every
    splat 0.0, degrading the ladder to input order with nothing to show it, and a
    FRACTIONAL index is truncated to a different column than the one named. A
    non-finite ``centre`` coordinate is rejected for the same reason: it makes
    every distance NaN/inf, which a stable argsort leaves in input order.
    """
    if int(np.asarray(data.centers).shape[0]) == 0:
        # Read the reach precisely: `compute_additive_order` ALREADY short-circuits
        # an empty dataset before dispatching here, so no public path was broken —
        # measured, it returns an empty permutation either way. What this fixes is
        # the HELPER's own contract: called directly it raised "zero-size array to
        # reduction operation minimum" out of the default-centre bbox below, while
        # its public element-side twin `radial_element_score` has always returned
        # an empty score. Same ordering, two implementations, and only one of them
        # could be called with an empty input — so this is symmetry insurance for a
        # future caller, not a live-bug fix.
        return np.empty(0, dtype=np.float64)
    if spatial_dims is None:
        dims = data._nondegenerate_axes()
    else:
        validate_integral_axis_indices(spatial_dims)
        dims = np.asarray(spatial_dims, dtype=np.intp)
        if dims.ndim != 1 or dims.size == 0:
            raise ValueError("spatial_dims must be a non-empty sequence of indices")
        if int(dims.min()) < 0:
            raise ValueError(
                "spatial_dims must be non-negative (a negative index would alias "
                f"to another column); got {list(spatial_dims)}"
            )
        if np.unique(dims).size != dims.size:
            raise ValueError(
                "spatial_dims must not repeat an axis (a repeat would count it "
                f"twice in the distance); got {list(spatial_dims)}"
            )
        if int(dims.max()) >= data.ndim:
            raise ValueError(
                f"spatial_dims {list(spatial_dims)} out of range for centers with "
                f"{data.ndim} columns"
            )
    pts = np.asarray(data.centers, dtype=np.float64)[:, dims]
    # Same data-side guard as the element scorer, from the one shared validator:
    # measured, a NaN centre coordinate made this return INPUT order silently.
    # Only the shell columns — a NaN on an axis the distance does not span is
    # irrelevant to the ordering.
    validate_finite_reveal_coords(pts, "centers")
    if centre is None:
        origin = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    else:
        origin = np.asarray(centre, dtype=np.float64)
        if origin.shape != (len(dims),):
            raise ValueError(
                f"reveal_centre must have one coordinate per spatial axis "
                f"{tuple(int(d) for d in dims)}; got shape {origin.shape}"
            )
        if not bool(np.all(np.isfinite(origin))):
            # Same class as an empty `spatial_dims`: every distance comes back
            # non-finite, they all compare equal under a stable argsort, and the
            # ladder silently degrades to input order instead of revealing.
            raise ValueError(
                "reveal_centre must be finite (a NaN/inf coordinate makes every "
                "distance non-finite, degrading the ladder to input order); got "
                f"{[float(c) for c in origin]}"
            )
    return np.asarray(np.linalg.norm(pts - origin, axis=1), dtype=np.float64)


def _validate_slice_dims(slice_dims: Sequence[int], ndim: int) -> np.ndarray:
    """Validate ``slice_dims`` and return it as a ``(k,)`` index array.

    Modelled on the ``spatial_dims`` block in :func:`_radial_score`, and rejects
    the same family for the same reason — each malformed case would produce a
    WRONG grouping rather than an error. A negative index ALIASES to another
    column (grouping by, say, the last spatial axis instead of time), a repeat
    contributes the same column twice to the key tuple (harmless to the grouping
    but a sign the caller miscounted, so it is refused rather than silently
    tolerated), an empty selection puts every splat in ONE group, which is a
    silent no-op interleave, and a fractional index truncates to a different
    column than the one named.

    The indices are RAW, PRE-``dim_order`` centre columns — see
    :func:`interleave_order_across_slices` for why that distinction is the
    likeliest way to get this wrong.

    There is deliberately no default, and not because the columns are
    undiscoverable: the WRITER auto-detects them, and
    ``gsplats/io/save_gsplats.py`` stamps the result as the ``slice_dims`` attr of
    each rung. But that detection runs at SAVE time, after the ladder has been
    ordered and cut into rungs, so it is not available to the ordering that has to
    consume it. Note the stamped attr is also in a DIFFERENT frame: it derives
    from barrier dims resolved POST-``dim_order``, so it and this kwarg coincide
    only when ``dim_order`` does not permute — which is why the built NEXRAD store
    happens to stamp the same ``[3]``. Hence the caller names them, exactly as the
    CLI does for ``--coarsen-dims``.
    """
    validate_integral_axis_indices(slice_dims, "slice_dims")
    dims = np.asarray(slice_dims, dtype=np.intp)
    if dims.ndim != 1 or dims.size == 0:
        raise ValueError(
            "slice_dims must be a non-empty 1-D sequence of centre-column "
            "indices (an empty selection groups every splat together, which "
            f"is a silent no-op interleave); got {slice_dims!r}"
        )
    if int(dims.min()) < 0:
        raise ValueError(
            "slice_dims must be non-negative (a negative index would alias to "
            f"another column); got {list(slice_dims)}"
        )
    if np.unique(dims).size != dims.size:
        raise ValueError(
            "slice_dims must not repeat an axis (a repeat would contribute the "
            f"same column twice to the slice key); got {list(slice_dims)}"
        )
    if int(dims.max()) >= ndim:
        raise ValueError(
            f"slice_dims {list(slice_dims)} out of range for centers with "
            f"{ndim} columns"
        )
    return dims


def _validate_interleave_order(order: Any, n_splats: int) -> np.ndarray:
    """Validate a caller-supplied ``order`` and return it as ``int64``.

    :func:`interleave_order_across_slices` is public, so ``order`` is untrusted
    input and every malformed shape of it was measured to produce a plausible
    wrong ANSWER rather than an error: a float array is silently truncated by the
    ``int64`` cast (``[2.9, 0.1, 1.5]`` orders as ``[2, 0, 1]``), a negative entry
    ALIASES to another element under numpy indexing, and a SHORT array returns a
    partial ordering that a caller would write out as a complete ladder. Only an
    out-of-range positive index raised anything, and only by accident (the
    ``centers`` gather).

    DUPLICATE entries stay the caller's responsibility: ``[0, 0, 1]`` passes here
    and comes back a non-permutation. Detecting it needs a sort or a bincount over
    the whole array — a second O(N log N) pass on the hot path, to catch something
    no in-tree caller can produce (every producer is an ``argsort`` or an
    ``rng.permutation``). The four guards below are all O(N) with no allocation of
    consequence.
    """
    arr = np.asarray(order)
    if arr.shape != (n_splats,):
        raise ValueError(
            f"order must have shape ({n_splats},), one entry per splat (a SHORT "
            "array silently yields a partial ordering a caller would write out "
            f"as a complete ladder); got shape {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            "order must be an integer array (a float order is truncated by the "
            "int64 cast, so [2.9, 0.1, 1.5] silently orders as [2, 0, 1]); got "
            f"dtype {arr.dtype}"
        )
    if arr.size and int(arr.min()) < 0:
        raise ValueError(
            "order must be non-negative (a negative index would alias to another "
            f"element); got minimum {int(arr.min())}"
        )
    if arr.size and int(arr.max()) >= n_splats:
        raise ValueError(
            f"order index {int(arr.max())} out of range for {n_splats} splats"
        )
    return arr.astype(np.int64, copy=False)


def interleave_order_across_slices(
    data: GSplatData,
    order: np.ndarray,
    slice_dims: Sequence[int],
) -> np.ndarray:
    """Re-emit ``order`` round-robin across slices, so every prefix is slice-even.

    A node the viewer SLICES (any non-displayed dimension) shows one hidden
    coordinate at a time, but an additive rung is sized against the WHOLE node.
    A contribution-ordered prefix therefore concentrates wherever the signal is
    and the sparse coordinates get almost nothing: the NEXRAD supercell's
    82-scan stack (817,989 splats; per-scan min 562, p05 774, median 10,499, max
    19,237) shipped an absolute ``breakpoints="stream:20000"`` first rung whose
    5th-percentile scan held **4 splats** — an empty screen during playback, and
    two failures in ``scripts/check_demo_ladders.py`` (#2485).

    Grouping the elements of ``order`` by their distinct combination of the
    ``slice_dims`` centre columns and emitting one per group per pass turns that
    global budget into an EQUAL ABSOLUTE per-slice budget. Precisely: after $R$
    completed passes the prefix holds $\\min(n_i, R)$ elements of every slice
    $i$, so a slice SMALLER than the budget is carried WHOLE and a large one is
    capped — which is exactly the shape ``check_demo_ladders.py``'s absolute
    first-paint arm asks for. Ties inside a pass are broken by position in
    ``order``, so the pass order is stable.

    That guarantee has a PRECONDITION worth stating: a rung must be at least as
    large as the slice count, or it cannot reach every slice at all. A rung
    smaller than $S$ does not even complete its first pass, and since
    within-pass ties break by position in ``order`` the coordinates left with
    NOTHING are the faintest ones — measured on 500 slices of 40 splats with
    ``breakpoints=[200]``, 300 coordinates got zero. Nowhere near a hazard for the
    demo this was built for (204,497 against 82 slices), but a ladder whose first
    rung is smaller than its hidden-axis cardinality is not made even by this.

    Two properties worth relying on:

    * **Deterministic** — no seed, and no distributional argument. Sizing the
      ladder by hand cannot get here: on the stack above the 250-element floor
      needs ~32% of the sparsest scan, and a uniform ``method="random"``
      permutation only makes the per-slice share proportional IN EXPECTATION —
      measured over seeds 0-7, ``n_lods=3`` cleared the floor 5 times in 8
      (p05 243-277 against a floor of 250) and ``n_lods=4`` never did (186-205).
    * **Idempotent** — re-interleaving an already-interleaved order returns it
      unchanged, because the within-group order (and hence every within-group
      rank) is untouched. Both the authoring door and
      :func:`make_additive_lod`'s Gram branch could in principle apply it.

    The within-slice order is still whatever the base method produced, so under
    the default ``method="auto"`` each coordinate keeps painting bright-core-
    first rather than evenly thin.

    ``slice_dims`` indexes RAW, PRE-``dim_order`` centre columns, and that is the
    likeliest way to get this wrong. The ladder is built in
    ``core/group/gsplats_pipeline/from_data.py`` ABOVE the ``apply_dim_order_*``
    pass that ``lod_dispatch`` runs, so these are the columns of the array the
    caller handed in — NOT the scene's post-``dim_order`` dimension positions.
    They coincide for the NEXRAD supercell only because its ``dim_order`` leaves
    time last in both frames. An author who reads off the scene's ``Dimensions``
    list instead gets a different column, and per the next paragraph that is a
    near-no-op. A high-cardinality diagnostic below warns about this likely
    pre-/post-``dim_order`` mixup without rejecting legitimate small slices.

    The columns must also be genuinely DISCRETE — a stacked time/channel axis,
    where coordinates repeat. Pointed at a continuous one, nearly every key is
    distinct, nearly every rank is 0, and the ``lexsort`` reproduces ``order``:
    functionally a no-op. NOT a bit-identical one, though, so this must not be
    used as an equality assertion — real centres do collide, and each collision
    demotes one element a pass later, which shifts the whole tail behind it.
    Measured on 8 cached NEXRAD frames (5,937 splats, 5,907 distinct values in
    column 0), ``slice_dims=[0]`` left the first 20 positions untouched and moved
    5,901 of 5,937 overall; one deliberate collision among 2,000 float32 samples
    moved 48. Reachable by composition and not only by typo: a
    ``lod_group=dict(coarsen_dims=[0, 1, 2, 3])`` — coarsening OVER the stacked
    axis — turned 3 exact time coordinates into 35 fractional ones on the coarse
    level, and :func:`~luxar.core.group.lod.gsplats.resolve_additive_axis_gsplats`
    applies one ``slice_dims`` to every substitutive level, giving a slice-even
    finest level and silently uneven coarse ones. The default ``Auto``
    coarsening (hidden axis as a hard barrier) is safe.

    Parameters
    ----------
    data : GSplatData
        The dataset ``order`` indexes; only its ``centers`` are read.
    order : np.ndarray
        A length-N integer permutation, as returned by
        :func:`compute_additive_order`. Validated for shape, dtype and range;
        duplicate entries are the caller's responsibility (see
        :func:`_validate_interleave_order`).
    slice_dims : sequence of int
        RAW, pre-``dim_order`` centre columns whose distinct combinations define
        a slice. No default — see :func:`_validate_slice_dims`.

    Returns
    -------
    np.ndarray of shape (N,), dtype int64
        A permutation of the same elements, slice-even at every prefix.
    """
    dims = _validate_slice_dims(slice_dims, data.ndim)
    order = _validate_interleave_order(order, data.n_splats)
    n = int(order.size)

    keys = np.asarray(data.centers, dtype=np.float64)[:, dims][order]
    if not bool(np.isfinite(keys).all()):
        # NaN is the one that CORRUPTS the grouping: `np.unique` compares NaN
        # unequal to itself, so every NaN-keyed splat becomes its own singleton
        # slice — and a singleton is smaller than any budget, i.e. "carried
        # whole", i.e. inside pass 0. Measured on 3 real slices of 10 splats plus
        # 6 NaN-keyed ones, ALL SIX landed in pass 0 and diluted every real
        # slice's budget, with nothing to say why. (±0.0 and ±inf group correctly
        # — checked — but a non-finite slice coordinate is not a coordinate the
        # viewer can slice AT, so both go out together, which also matches the
        # `spatial_dims` guard in `_radial_score`. That one routes through the
        # shared `validate_finite_reveal_coords`; this cannot, because that
        # validator's message describes a radial reveal degrading to input order,
        # which is a different failure from the one here.)
        bad = np.flatnonzero(~np.isfinite(keys).all(axis=1))
        raise ValueError(
            "the slice_dims columns of centers must be finite: a NaN key "
            "compares unequal to itself, so each NaN-keyed splat becomes its "
            "own singleton slice and is carried WHOLE in the first pass, "
            "silently diluting every real slice's budget; an infinite key does "
            "group correctly but is not a coordinate the viewer can slice AT. "
            f"{bad.size} of {n} splats are non-finite on columns "
            f"{[int(d) for d in dims]} (first at order position {int(bad[0])})."
        )
    if n <= 1:
        # Nothing to interleave, and reached only AFTER the finiteness check, so
        # a trivial leaf gives the same verdict as a populated one. Same reason
        # the column validation is unconditional: the fan-out callers hand ONE
        # spec and one dataset to many leaves.
        return order

    # Vectorized, O(N log N): the corpus builds this on 818k splats, so a Python
    # loop over slice groups is not an option. `group` is a dense slice label per
    # POSITION IN `order`; a stable argsort of it makes each group's positions
    # contiguous and ascending in base order, so subtracting the group's first
    # index gives that element's rank WITHIN its slice. Sorting by (rank,
    # base position) is then exactly the round-robin.
    #
    # A single hidden axis is the common case (every sliced demo today), and the
    # 1-D `unique` is ~19x cheaper than the structured-row sort `axis=0` takes:
    # measured 0.854 s -> 0.045 s at 818k x 1 column. Equivalent because only the
    # PARTITION matters here, never the group LABELS — ranks are computed within a
    # group off a stable argsort, so relabelling cannot move an element. Both
    # forms partition by exact float equality and both collapse ±0.0; NaN/±inf
    # are refused above. Verified identical on 400 adversarial random cases and
    # on the real 818k stack for slice_dims=[3] and [0].
    flat = keys[:, 0] if dims.size == 1 else keys
    group = np.asarray(
        np.unique(flat, return_inverse=True)[1]
        if dims.size == 1
        else np.unique(flat, axis=0, return_inverse=True)[1]
    ).ravel()
    n_groups = int(group.max()) + 1
    if n >= 32 and 2 * n_groups >= n:
        warnings.warn(
            f"slice_dims {[int(d) for d in dims]} produced {n_groups} distinct "
            f"slice keys for {n} splats; this usually means a continuous center "
            "column was selected. slice_dims indexes raw pre-`dim_order` center "
            "columns, not post-`dim_order` scene positions.",
            UserWarning,
            stacklevel=2,
        )
    sorter = np.argsort(group, kind="stable")
    grouped = group[sorter]
    rank = np.empty(n, dtype=np.int64)
    rank[sorter] = np.arange(n, dtype=np.int64) - np.searchsorted(
        grouped, grouped, side="left"
    )
    return order[np.lexsort((np.arange(n), rank))].astype(np.int64)


def compute_additive_order(
    data: GSplatData,
    method: AutoOrMethod = "auto",
    *,
    truncation_sigmas: float | None = None,
    max_n_dense: int = 2_000,
    seed: int | None = None,
    reveal_centre: Sequence[float] | None = None,
    spatial_dims: Sequence[int] | None = None,
    slice_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Compute an additive ordering permutation for the splats in ``data``.

    Parameters
    ----------
    data : GSplatData
        Fitted (single- or multi-LOD) gsplat dataset.  Operates on the
        flattened concatenation across LODs.
    method : str
        One of ``auto``, ``greedy``, ``self_energy``, ``mass``,
        ``amplitude``, ``spectral``, ``random``, ``radial``.  ``auto`` (the default)
        resolves to ``greedy`` at small N and ``self_energy`` above
        :data:`_AUTO_ADDITIVE_MAX_N` — see :func:`resolve_additive_method`.
        See module docstring for details.
    truncation_sigmas : float, optional
        Mahalanobis cutoff used for sparse-Gram pruning. Defaults to the
        dataset's own ``truncation_radius`` — the support the splats were
        fitted at and are rendered at. Only relevant for ``greedy`` and
        ``spectral``.
    max_n_dense : int
        For ``greedy``, build a dense Gram and use scan-greedy when
        $N \\leq$ this threshold.  Above it, build a sparse Gram and
        use lazy-greedy.  Default 2000 (per supp doc §4.3).
    seed : int, optional
        Random seed for ``method='random'``.
    reveal_centre : sequence of float, optional
        Centre of the shells for ``method='radial'``. Defaults to the spatial
        bounding-box centre — NOT the scene origin, so a dataset far from the
        origin still grows from its own middle. One coordinate per spatial axis.
    spatial_dims : sequence of int, optional
        Centre columns the radial distance is measured over. Defaults to the
        non-degenerate (real-extent) axes, which excludes a stacked time or
        channel axis. Ignored by every other method.
    slice_dims : sequence of int, optional
        RAW, pre-``dim_order`` centre columns whose distinct combinations the
        viewer SLICES (a hidden time / channel axis) — NOT the scene's
        post-``dim_order`` dimension positions, which are a different frame of
        reference. When given, the ordering chosen by ``method`` is
        re-emitted round-robin across those slices by
        :func:`interleave_order_across_slices`, so every prefix carries an equal
        ABSOLUTE budget per slice instead of a global contribution-ordered
        prefix that starves the sparse ones. Composes with EVERY method — it is
        a modifier, not a method. ``None`` (the default) leaves the order
        untouched; there is no default column set, since a standalone
        ``GSplatData`` has no display information to derive one from.

    Returns
    -------
    np.ndarray of shape (N,), dtype int64
        ``order[k]`` is the original index of the splat at rank $k$.
    """
    if method not in _VALID_CHOICES:
        raise ValueError(f"method must be one of {_VALID_CHOICES}, got {method!r}")

    sigmas = resolve_truncation_sigmas(truncation_sigmas, data)

    if slice_dims is not None:
        # ABOVE the N <= 1 short-circuits, deliberately: a caller with a typo'd
        # column must hear about it whatever this leaf happens to hold. Both
        # fan-out callers hand this function many leaves of ONE spec —
        # `resolve_additive_axis_gsplats` walks substitutive levels, and
        # `adders/gsplats.py::add_gsplats_partition_wrapper_impl` puts
        # `additive_lod` in `leaf_attrs` and forwards it to every `part_i` — so
        # validating only where the interleave RUNS means the same spec is
        # rejected by the populated leaves and waved through by any empty or
        # single-splat one.
        _validate_slice_dims(slice_dims, data.ndim)

    N = data.n_splats
    if N <= 1:
        # Nothing to rank, and `arange` spells both trivial answers at once: the
        # empty permutation and the single-element one.
        return np.arange(N, dtype=np.int64)

    # Resolve the size-adaptive sentinel ONCE, before any (expensive) Gram
    # build, so every downstream branch sees a concrete method.
    method = resolve_additive_method(method, N)

    # One assignment per branch and a SINGLE return, so the `slice_dims`
    # modifier below is applied to whichever ordering was chosen and cannot be
    # skipped by a method that returns early.
    if method == "radial":
        # ASCENDING, unlike every other method here: the score is a DISTANCE, so
        # the smallest is revealed first and the prefixes grow outward as
        # concentric shells. Every other branch sorts `-score` because its score
        # is a contribution to maximize.
        order = np.argsort(
            _radial_score(data, centre=reveal_centre, spatial_dims=spatial_dims),
            kind="stable",
        ).astype(np.int64)
    elif method == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(N).astype(np.int64)
    elif method == "amplitude":
        score = np.asarray(effective_amplitudes(data), dtype=np.float64)
        order = np.argsort(-score, kind="stable").astype(np.int64)
    elif method == "mass":
        score = _mass_score(data)
        order = np.argsort(-score, kind="stable").astype(np.int64)
    elif method == "self_energy":
        score = _self_energy_score(data)
        order = np.argsort(-score, kind="stable").astype(np.int64)
    else:
        # Spectral and greedy need the Gram matrix.
        gram_csr = _build_sparse_gram(data, sigmas=sigmas)
        order = _order_from_gram(gram_csr, method, max_n_dense)

    if slice_dims is None:
        return order
    return interleave_order_across_slices(data, order, slice_dims)


def _order_from_gram(
    gram_csr: sparse.csr_matrix, method: MethodName, max_n_dense: int
) -> np.ndarray:
    """Spectral / greedy ordering from a prebuilt sparse Gram.

    Split out so callers that already hold the Gram (e.g.
    :func:`make_additive_lod` resolving energy-fraction breakpoints) can
    reuse it instead of rebuilding the most expensive structure twice.
    """
    if method == "spectral":
        return _spectral_order(gram_csr)
    if method == "greedy":
        if gram_csr.shape[0] <= max_n_dense:
            return _dense_greedy(gram_csr.toarray())
        return _lazy_greedy(gram_csr)
    raise AssertionError(f"unhandled gram method {method!r}")  # pragma: no cover


# ─────────────────────────────────────────────────────────────────────
# Public API: LOD-ladder construction
# ─────────────────────────────────────────────────────────────────────


def _prefixed_string_breakpoints(
    n: int, breakpoints: str
) -> tuple[list[int], str] | None:
    """The two size-adaptive string specs, or ``None`` for any other string.

    ``stream:<c>``: bandwidth-derived streaming ladder — geometric cumulative
    cuts ``[c, 2c, 4c, …]`` (increments ``[c, c, 2c, …]``: first paint = c
    splats, then doubling), resolved against THIS ``n`` so the same spec adapts
    to every part/level size. Silently clamped, never raises on small ``n``
    (unlike explicit counts — deliberate: per-part N is unknowable to the user).
    Capped at ``DEFAULT_STREAM_MAX_LEVELS``; a final increment smaller than
    ``c/2`` folds into the previous cut (no sliver levels).

    ``equi-energy:<n>``: equal shares of cumulative energy — returned as the
    FRACTIONS ``k/n``; :func:`make_additive_lod` resolves them against the
    ladder's own self-energy cumulative and commit-caps the result.
    """
    if breakpoints.startswith(EQUI_ENERGY_PREFIX):
        n_rungs = parse_equi_energy_rungs(breakpoints)
        shares = [k / n_rungs for k in range(1, n_rungs + 1)]
        return shares, "equi-energy"  # type: ignore[return-value]
    if breakpoints.startswith("stream:"):
        return stream_cuts(n, parse_stream_chunk(breakpoints)), "stream"
    return None


def _resolve_breakpoints(
    n: int,
    n_lods: int,
    breakpoints: BreakpointSpec,
) -> tuple[list[int], str]:
    """Resolve ``breakpoints`` to a list of cumulative cutpoints ending at $N$.

    Returns ``(cumulative_counts, kind)`` where ``kind`` is one of
    ``equal-count``, ``stream``, ``explicit-counts``, ``energy-fractions``,
    ``equi-energy``. The two energy kinds return FRACTIONS, not counts: they
    need the ordering and the energy curve, which only :func:`make_additive_lod`
    holds.
    """
    if isinstance(breakpoints, str):
        prefixed = _prefixed_string_breakpoints(n, breakpoints)
        if prefixed is not None:
            return prefixed
        if breakpoints != "equal-count":
            raise ValueError(
                f"unknown breakpoints string {breakpoints!r}; "
                "expected 'equal-count', 'stream:<c>', 'equi-energy:<n>', or a list."
            )
        if n_lods <= 0:
            raise ValueError("n_lods must be positive")
        if n_lods > n:
            n_lods = n
        cuts = sorted({round((k + 1) * n / n_lods) for k in range(n_lods)})
        if cuts[0] <= 0:
            cuts = [c for c in cuts if c > 0]
        if not cuts or cuts[-1] != n:
            cuts.append(n)
        return cuts, "equal-count"

    if not isinstance(breakpoints, (list, tuple)):
        raise TypeError(
            "breakpoints must be 'equal-count', 'stream:<c>', 'equi-energy:<n>', "
            "a list of ints (cumulative counts), or a list of floats in (0, 1] "
            f"(cumulative energy fractions); got {type(breakpoints).__name__}"
        )
    if len(breakpoints) == 0:
        raise ValueError("breakpoints list must not be empty")

    # Distinguish int vs float by inspecting elements. Excludes bool explicitly
    # (bool is an int subclass in Python) — [True, False] must not be accepted
    # as cumulative counts.
    all_int = all(
        isinstance(x, (int, np.integer)) and not isinstance(x, bool)
        for x in breakpoints
    )
    all_float = all(
        isinstance(x, float) or (isinstance(x, np.floating)) for x in breakpoints
    )
    if all_int:
        cuts = [int(c) for c in breakpoints]
        if any(c <= 0 for c in cuts):
            raise ValueError("explicit count breakpoints must be positive")
        if not all(cuts[i] < cuts[i + 1] for i in range(len(cuts) - 1)):
            raise ValueError("explicit count breakpoints must be strictly increasing")
        if cuts[-1] > n:
            raise ValueError(f"largest breakpoint {cuts[-1]} exceeds N={n}")
        if cuts[-1] < n:
            cuts.append(n)
        return cuts, "explicit-counts"

    if all_float:
        # Resolved by caller against the residual-energy curve (needs Gram).
        # Validate range/monotonicity here; resolution happens in
        # ``make_additive_lod`` where the curve is available.
        fracs = [float(x) for x in breakpoints]
        if any(f <= 0.0 or f > 1.0 for f in fracs):
            raise ValueError("energy-fraction breakpoints must lie in (0, 1]")
        if not all(fracs[i] < fracs[i + 1] for i in range(len(fracs) - 1)):
            raise ValueError("energy-fraction breakpoints must be strictly increasing")
        return fracs, "energy-fractions"  # type: ignore[return-value]

    raise TypeError(
        "breakpoints list must be either all ints (cumulative counts) "
        "or all floats (cumulative energy fractions); mixed types not allowed"
    )


def additive_rung_count(
    n: int,
    n_lods: int = 4,
    breakpoints: BreakpointSpec = "equal-count",
) -> int | None:
    """How many rungs would this spec leave on a leaf of ``n`` splats? (#1632)

    A cheap, ordering-free query. The exactness comes from SHARING the cut
    resolver :func:`make_additive_lod` uses (:func:`_resolve_breakpoints`): the
    answer is counted off the very cuts that build would consume, so for
    equal-count / ``stream:`` / explicit counts it is the number of
    :class:`AdditiveSubLOD` objects that build would EMIT, not a re-derivation
    free to drift. The loop below also MIRRORS that build loop's ``if end <=
    prev: continue`` de-duplication — but as a mirror only, so the two cannot
    diverge if a future cut resolver ever emits a duplicate. It is not a live
    filter and is not what makes the count exact: every non-energy path of
    :func:`_resolve_breakpoints` returns strictly-increasing positive cuts, so
    neither loop can skip one today. Pinned against
    ``make_additive_lod(...).n_additive_sublods`` per breakpoint kind in
    ``tests/test_additive.py``.

    It exists because the callers that must decide *whether a ladder will exist*
    cannot afford to build one. The live one is the file/graft door's
    partition-vs-ladder gate
    (:func:`~luxar.core.group.gsplats_pipeline.from_io._reject_a_partition_beside_a_stored_ladder`,
    via the spec-level wrapper
    :func:`~luxar.core.group.lod.gsplats.resolve_additive_rungs`): ``partition=``
    and a multi-rung ladder are mutually exclusive, and that gate runs before
    ``graft_gsplat_node`` builds a ``kind=partition`` wrapper it would otherwise
    strand. Presence of the ``additive_lod=`` kwarg is not the question —
    ``{"n_lods": 1}`` resolves to one rung and partitions perfectly well, while
    ``{"method": "radial"}`` carries no ``n_lods`` to read and falls to the
    ``n_lods=4`` default — four rungs on any leaf of >= 4 splats, and ``n`` on a
    smaller one, since equal-count cuts clamp to the leaf's own size.

    Returns ``None`` for UNKNOWN, never raises:

    * ``kind == "energy-fractions"`` or ``"equi-energy"`` — those cuts need the
      ordering and the energy curve, i.e. exactly the expensive half this query
      exists to avoid.
    * anything :func:`_resolve_breakpoints` would reject (a non-positive
      ``n_lods``, an unknown breakpoints string, a mixed list, a counts list
      exceeding ``n``, …). Swallowing the fault is deliberate: this is a QUERY,
      and the real build must stay the thing that reports it, at its own site,
      with its own message. ``None`` says nothing about the INPUT, only that this
      spec is unreadable here, so a caller that cannot act on it should fall back
      to what it already knows rather than assume "no ladder".

    Note the converse, for the gate: a fault this COUNT cannot see — a bad
    ``method``, a stray ``substitutive_level`` key, anything past the cut
    resolver — makes no difference to the number, so a caller refusing on the
    count MASKS it rather than letting the builder report it. Same trade in the
    other direction, and an acceptable one where the caller's own conflict is the
    more fundamental fault and nothing is written either way.

    ``n <= 0`` returns ``1``, mirroring :func:`make_additive_lod`'s empty-leaf
    branch, which emits exactly one sub-LOD labelled ``lod_method="none"``.
    """
    if n <= 0:
        return 1
    try:
        cuts_or_fracs, kind = _resolve_breakpoints(n, n_lods, breakpoints)
    except (ValueError, TypeError):
        return None
    if kind in ("energy-fractions", "equi-energy"):
        return None
    rungs = 0
    prev = 0
    for end in cuts_or_fracs:
        end = int(end)
        if end <= prev:
            continue
        rungs += 1
        prev = end
    return rungs


def _energy_fraction_cuts(
    fracs: list[float],
    gram_csr: sparse.csr_matrix,
    order: np.ndarray,
) -> list[int]:
    """Resolve cumulative energy fractions to cumulative splat counts."""
    N = gram_csr.shape[0]
    E_curve = _residual_energy_curve(gram_csr, order)
    E_total = float(E_curve[0])
    if E_total <= 0.0 or not np.isfinite(E_total):
        # Degenerate; place cuts uniformly.
        return sorted({round((i + 1) * N / len(fracs)) for i in range(len(fracs))})
    U_frac = (E_total - E_curve) / E_total  # length N+1
    cuts: list[int] = []
    for f in fracs:
        idxs = np.where(U_frac >= f)[0]
        k = int(idxs[0]) if idxs.size else N
        cuts.append(max(k, 1))
    cuts = sorted(set(cuts))
    if cuts[-1] < N:
        cuts.append(N)
    return cuts


def _energy_fraction_cuts_from_cumulative(
    fracs: list[float], energy_cum: np.ndarray
) -> list[int]:
    """Resolve cumulative energy fractions to counts against a precomputed
    cumulative self-energy curve (O(N), no Gram). For score-ordered ladders
    the ordering AND the viewer's e(k) quality stamp are both self-energy
    based, so cuts land where e(k) ≈ the target fraction (self-consistent),
    and the maximally-overlapping coarse children never trigger the O(N²)
    sparse-Gram residual-curve build."""
    N = int(energy_cum.size)
    total = float(energy_cum[-1]) if N else 0.0
    if total <= 0.0 or not np.isfinite(total):
        return sorted({round((i + 1) * N / len(fracs)) for i in range(len(fracs))})
    U = energy_cum / total  # U[k] = fraction captured after k+1 splats
    cuts: list[int] = []
    for f in fracs:
        idxs = np.where(U >= f)[0]
        k = (int(idxs[0]) + 1) if idxs.size else N
        cuts.append(max(k, 1))
    cuts = sorted(set(cuts))
    if not cuts or cuts[-1] < N:
        cuts.append(N)
    return cuts


def _cuts_for_kind(
    kind: str,
    cuts_or_fracs: Sequence[float],
    *,
    breakpoints: BreakpointSpec,
    gram_csr: sparse.csr_matrix | None,
    order: np.ndarray,
    energy_ordered: np.ndarray,
    energy_cum: np.ndarray,
) -> list[int]:
    """Turn a resolved breakpoint spec into cumulative cuts for THIS ordering.

    Count kinds pass through. The two energy kinds need the curve:

    * ``equi-energy`` — equal shares of the SELF-energy cumulative for every
      ordering (the same curve the ``e(k)`` stamps read, so the stamps land at
      ~``k/n`` by construction), then split at the shared commit cap. The Gram
      residual curve is deliberately not used even when greedy built one:
      equi-energy is a first-paint sizing rule, and the stamp-consistent curve is
      the one the viewer will act on.
    * ``energy-fractions`` — when greedy/spectral already built the Gram for the
      ORDERING, reuse its residual-energy curve (behaviour unchanged); for score
      orderings (self_energy/mass/amplitude/random) resolve against the O(N)
      self-energy cumulative and never build the Gram — coarse lifted children
      are maximally-overlapping merged blobs, the worst case for the sparse
      residual-curve build.
    """
    if kind == "equi-energy":
        return equi_energy_cuts(
            energy_ordered, parse_equi_energy_rungs(str(breakpoints))
        )
    if kind == "energy-fractions":
        fracs = [float(x) for x in cuts_or_fracs]
        if gram_csr is not None:
            return _energy_fraction_cuts(fracs, gram_csr, order)
        return _energy_fraction_cuts_from_cumulative(fracs, energy_cum)
    return [int(x) for x in cuts_or_fracs]


def _sublod_stats(
    *,
    method: str,
    level: int,
    kind: str,
    prev: int,
    end: int,
    energy_cum: np.ndarray,
    energy_total: float,
    breakpoints: BreakpointSpec,
) -> dict[str, Any]:
    """Per-sub-LOD ``lod_stats`` for one rung of an additive ladder.

    Extracted from :func:`make_additive_lod`'s build loop, which the energy-stamp
    guard pushed past the C901 ratchet — and a 30-line stats block nested in a
    loop inside an already-long function reads better named anyway.

    ``radial`` is a REVEAL, so it carries **no** energy stamps. The viewer
    multiplies a leaf's brightness by ``1/e(k)`` while a ladder is incomplete,
    gated on the BLENDING MODE and not on geometry type (``scene/lod-blend.ts``).
    That is right for a contribution-ordered prefix, which genuinely is a dimmer
    version of the whole, and backwards for a radial one, which is a PARTIAL
    OBJECT AT FULL BRIGHTNESS: an inner shell would be brightened (up to 10x —
    ``ENERGY_FLOOR`` caps it), blazing and then dimming as the object completes —
    the exact inverse of growing outward. Omitting the stamp is the honest
    encoding, and ``energyCompensation(undefined)`` returns exactly 1, so the leaf
    is byte-identical. Measured: the compensation reaches a leaf only through the
    viewer's ``kind=lod`` group registry, so it bites for a ladder inside a lod
    group (1.87x brighter in mean luma while incomplete) and is inert on a bare
    leaf. The rule stays unconditional because ``method`` already covers both —
    ``lod --recipe levels|adaptive|overview -m radial`` lands reveal ladders inside
    a lod group, ``--recipe stream -m radial`` does not. See MESH_NODE_SPEC §9.1,
    which states the rule for mesh; the reasoning is geometry-agnostic.
    """
    stats: dict[str, Any] = {
        "lod_method": method,
        "lod_level": level,
        "lod_breakpoints_kind": kind,
        "lod_n_splats": int(end - prev),
        "lod_cumulative_n": end,
    }
    if energy_total > 0.0 and not _is_reveal_method(method):
        e_frac = float(energy_cum[end - 1] / energy_total)
        if np.isfinite(e_frac):
            stats["energy_fraction_cum"] = min(1.0, max(0.0, e_frac))
    if kind == "stream":
        # Provenance: the bandwidth-derived first-chunk size, otherwise only
        # recoverable by re-parsing the breakpoints string.
        stats["lod_stream_chunk_splats"] = int(str(breakpoints)[len("stream:") :])
    if kind == "equi-energy":
        # Provenance: the requested equal-energy rung count; the emitted rung
        # count can be higher (commit-cap splits) or lower (dedup on tiny N).
        stats["lod_equi_energy_rungs"] = parse_equi_energy_rungs(str(breakpoints))
    return stats


def _ladder_order(
    target_view: GSplatData,
    *,
    method: MethodName,
    gram_csr: sparse.csr_matrix | None,
    sigmas: float,
    max_n_dense: int,
    seed: int | None,
    reveal_centre: Sequence[float] | None,
    spatial_dims: Sequence[int] | None,
    slice_dims: Sequence[int] | None,
) -> np.ndarray:
    """The ladder's ordering permutation, from either of the TWO order paths.

    Extracted from :func:`make_additive_lod`'s build (which the ``slice_dims``
    modifier pushed past the C901 ratchet — same reason as :func:`_sublod_stats`)
    and worth a name of its own, because the two-path split is the trap here:
    when ``gram_csr`` is already built (greedy / spectral) this NEVER calls
    :func:`compute_additive_order` — it reuses that matrix rather than paying the
    O(nnz) build twice — so every modifier that function applies has to be
    applied on this branch as well. Both branches converge here, so a caller of
    :func:`make_additive_lod` cannot get a slice-even ladder out of
    ``self_energy`` and a starved one out of ``greedy``.
    """
    if gram_csr is not None:
        order = _order_from_gram(gram_csr, method, max_n_dense)
        if slice_dims is not None:
            order = interleave_order_across_slices(target_view, order, slice_dims)
        return order
    return compute_additive_order(
        target_view,
        method=method,
        truncation_sigmas=sigmas,
        max_n_dense=max_n_dense,
        seed=seed,
        reveal_centre=reveal_centre,
        spatial_dims=spatial_dims,
        slice_dims=slice_dims,
    )


def make_additive_lod(
    data: GSplatData,
    n_lods: int = 4,
    *,
    method: AutoOrMethod = "auto",
    breakpoints: BreakpointSpec = "equal-count",
    truncation_sigmas: float | None = None,
    max_n_dense: int = 2_000,
    seed: int | None = None,
    substitutive_level: int | None = None,
    reveal_centre: Sequence[float] | None = None,
    spatial_dims: Sequence[int] | None = None,
    slice_dims: Sequence[int] | None = None,
) -> GSplatData:
    """Permute and split a fitted gsplat dataset into a multi-LOD ladder.

    The result is a ``GSplatData`` with ``n_lods`` (or as resolved by
    ``breakpoints``) ``AdditiveSubLOD`` levels on the selected
    substitutive level.  ``additive_prefix(k)`` returns the valid
    additive prefix of size :math:`\\sum_{\\ell \\leq k} N_\\ell` for
    that level.

    Parameters
    ----------
    data : GSplatData
        Fitted gsplat dataset. May be multi-substitutive: the
        ``substitutive_level`` argument (default = default substitutive
        level) selects which level receives the new additive ladder.
        Other substitutive levels are carried over verbatim.
    n_lods : int
        Number of LOD levels when ``breakpoints='equal-count'``.  Ignored
        when ``breakpoints`` is a list or ``'stream:<c>'`` (those determine
        the level count themselves).
    method : str
        Ordering method (see :func:`compute_additive_order`).
    breakpoints : ``'equal-count'``, ``'stream:<c>'``, or list of int / float
        - ``'equal-count'``: ``n_lods`` levels of (nearly-)equal size.
        - ``'stream:<c>'``: geometric streaming ladder — cumulative cuts
          ``[c, 2c, 4c, …, N]`` sized so the first chunk is ``c`` splats
          (bandwidth-derived via :func:`streaming_chunk_splats`), then
          doubling. Resolved against each call's own N (per part / per
          substitutive level), silently clamped for small N (never raises,
          unlike explicit counts), capped at
          :data:`DEFAULT_STREAM_MAX_LEVELS` levels.
        - ``'equi-energy:<n>'``: ``n`` rungs at EQUAL shares of cumulative
          self-energy along the ordering — the first rung is the few heaviest
          splats, later rungs are fatter in count for the same light — with
          any increment above :data:`DEFAULT_MAX_ADDITIVE_COMMIT` split into
          capped steps (:func:`luxar.utils.lod_breakpoints.equi_energy_cuts`).
          Pair with a contribution-first ``method`` (``self_energy``, the
          large-N default) — under ``random`` the rungs are still equal in
          energy but there is no ordering to front-load.
        - list[int]: explicit cumulative splat counts per level.
        - list[float] in $(0, 1]$: cumulative energy fractions; the
          smallest $k$ at which the cumulative-utility curve crosses
          each fraction is used as the cutpoint. For ``greedy`` / spectral
          orderings that build a Gram matrix the curve is the residual-energy
          curve; for score-ordered methods (``self_energy`` / ``mass`` /
          ``amplitude`` / ``random``) it is the O(N) self-energy cumulative,
          so cuts land where the viewer's own $e(k)$ quality stamp reads the
          requested fraction.
    truncation_sigmas : float, optional
        $\\sigma$ multiplier for sparse-Gram pruning. Defaults to the dataset's
        own ``truncation_radius``.
    max_n_dense : int
        Threshold below which ``greedy`` uses a dense Gram + scan-greedy.
    seed : int, optional
        Random seed for ``method='random'``.
    substitutive_level : int, optional
        Index of the substitutive level to build the ladder for. Defaults
        to ``data.default_substitutive``.
    reveal_centre : sequence of float, optional
        ``method='radial'`` only — centre of the concentric shells. Defaults to
        the spatial bounding-box centre (NOT the scene origin, so a dataset far
        from the origin still reveals from its own middle).
    spatial_dims : sequence of int, optional
        ``method='radial'`` only — the centre columns the shell distance is
        measured over. Defaults to the non-degenerate axes, so a stacked
        time/channel axis cannot become a shell dimension.
    slice_dims : sequence of int, optional
        RAW, pre-``dim_order`` centre columns the viewer SLICES (a hidden time /
        channel axis) — NOT the scene's post-``dim_order`` dimension positions.
        When given, the ordering is re-emitted round-robin across those slices (see
        :func:`interleave_order_across_slices`), so every rung carries an equal
        ABSOLUTE budget per slice rather than a global prefix that starves the
        sparse ones (#2485). A modifier: it composes with every ``method``, and
        the ``energy_fraction_cum`` stamps are computed from the interleaved
        order, so the viewer's committed e(k) describes the prefix actually
        written — NODE-GLOBALLY, which is the only granularity that stamp has:
        e(k) is one number per rung and the viewer's ``1/max(e, 0.1)``
        compensation is applied uniformly across hidden coordinates, so a small
        slice that slice-evenness has already loaded COMPLETELY is still
        brightened (measured, rung 0 stamps e = 0.408 interleaved against 0.684
        plain — a 2.45x boost on an already-complete slice). This is NOT confined
        to a ``kind=lod`` group: ``applyLodFade`` has a second caller in the
        viewer's ``scene/density-guard.ts``, driven by the projected-density
        tracker over the whole scene graph for any blendable data mesh, and both
        the guard and the compensation default ON — so a BARE multi-additive leaf
        is in scope too once the guard steps it. Small in practice; on the NEXRAD
        node it is a 1.108x brightening (that store stamps e(0) = 0.9024),
        applied to the 12 of 82 scans this already loads whole as much as to the
        rest. Note also
        that interleaving FLATTENS the cumulative-energy curve, so
        energy-fraction ``breakpoints`` resolve to materially larger first rungs
        — measured on 1,580 splats over 4 slices, ``[0.5, 0.9, 0.99, 1.0]`` cuts
        at ``[222, 749, 1116, 1580]`` plain and ``[521, 1128, 1492, 1580]``
        interleaved. The requested fractions are still delivered and the rungs
        are still slice-even; it is the first-paint COST that moves.
        ``None`` (the default) leaves the order untouched.

    Returns
    -------
    GSplatData
        A matrix-shaped ``GSplatData`` with the same ``n_substitutive`` as
        ``data``; the selected level's additive sub-LODs form the new
        ladder, other substitutive levels are passed through unchanged.
    """
    s_target = (
        data.default_substitutive
        if substitutive_level is None
        else int(substitutive_level)
    )
    if not (0 <= s_target < data.n_substitutive):
        raise ValueError(
            f"substitutive_level={s_target} is out of bounds for "
            f"n_substitutive={data.n_substitutive}"
        )

    # Build the new additive ladder on the chosen substitutive level
    target_view = data.at_substitutive(s_target).flattened()
    n = target_view.n_splats

    # Resolve the pruning σ from the view we actually prune, NOT from `data`:
    # per-sub-LOD truncation radii round-trip independently, so the selected
    # substitutive level may claim a different support than the finest leaf.
    sigmas = resolve_truncation_sigmas(truncation_sigmas, target_view)

    # Resolve the size-adaptive sentinel ONCE, before the (expensive) Gram
    # build below, so `needs_gram` and the recorded `lod_method` stat both
    # see the concrete method. (compute_additive_order resolves it again
    # harmlessly for the score-method path — it is idempotent.)
    method = resolve_additive_method(method, n)

    if slice_dims is not None:
        # ABOVE the empty-leaf branch, for the same reason as in
        # `compute_additive_order`: a per-part / per-level loop must not accept a
        # typo'd column on the leaves that happen to be empty and reject it on
        # the rest.
        _validate_slice_dims(slice_dims, target_view.ndim)

    if n == 0:
        new_sublods: list[AdditiveSubLOD] = [
            AdditiveSubLOD(
                centers=np.asarray(target_view.centers, dtype=np.float32),
                amplitudes=np.asarray(target_view.amplitudes, dtype=np.float32),
                cholesky_factors=np.asarray(
                    target_view.cholesky_factors, dtype=np.float32
                ),
                colors=(
                    np.asarray(target_view.colors)
                    if target_view.colors is not None
                    else None
                ),
                label_ids=(
                    np.asarray(target_view.label_ids)
                    if target_view.label_ids is not None
                    else None
                ),
                label_vocabulary=target_view.label_vocabulary,
                stats={
                    "lod_method": "none",
                    "lod_level": 0,
                    # Trivially complete: nothing to stream.
                    "energy_fraction_cum": 1.0,
                },
                truncation_radius=target_view.truncation_radius,
            )
        ]
        cuts = [0]
        # An empty leaf has no ladder to speak of: label the kind "none" (like
        # lod_method above) rather than mislabeling whatever spec was requested
        # as "equal-count".
        kind = "none"
        energy_total = 0.0
    else:
        # Build the (expensive) sparse Gram at most once: greedy/spectral need
        # it for the ordering, and energy-fraction breakpoints need it again
        # for the residual-energy curve. Reuse the same matrix across both.
        needs_gram = method in ("greedy", "spectral")
        gram_csr = (
            _build_sparse_gram(target_view, sigmas=sigmas) if needs_gram else None
        )
        order = _ladder_order(
            target_view,
            method=method,
            gram_csr=gram_csr,
            sigmas=sigmas,
            max_n_dense=max_n_dense,
            seed=seed,
            reveal_centre=reveal_centre,
            spatial_dims=spatial_dims,
            slice_dims=slice_dims,
        )

        # Cumulative self-energy over the ladder ordering — the e(k) of the
        # viewer's committed quality Q·e(k). Fractions, so the shared π^{D/2}
        # constant of the true self-energy cancels and the (cheap, O(N))
        # ordering score suffices; π^{D/2} is multiplied back only for the
        # absolute reference_energy weight w (partition aggregation). Computed
        # ONCE here (before cut resolution) so score-ordered energy-fraction
        # breakpoints resolve against it in O(N) — never the O(N²) sparse Gram.
        energy_ordered = _self_energy_score(target_view)[order]
        energy_cum = np.cumsum(energy_ordered)
        energy_total = float(energy_cum[-1]) if energy_cum.size else 0.0

        cuts_or_fracs, kind = _resolve_breakpoints(n, n_lods, breakpoints)
        cuts = _cuts_for_kind(
            kind,
            cuts_or_fracs,
            breakpoints=breakpoints,
            gram_csr=gram_csr,
            order=order,
            energy_ordered=energy_ordered,
            energy_cum=energy_cum,
        )

        centers_full = np.asarray(target_view.centers)[order]
        amps_full = np.asarray(target_view.amplitudes)[order]
        chol_full = np.asarray(target_view.cholesky_factors)[order]
        colors_full = (
            np.asarray(target_view.colors)[order]
            if target_view.colors is not None
            else None
        )
        label_ids_full = (
            np.asarray(target_view.label_ids)[order]
            if target_view.label_ids is not None
            else None
        )

        new_sublods = []
        prev = 0
        for level, end in enumerate(cuts):
            end = int(end)
            if end <= prev:
                continue
            lod_stats = _sublod_stats(
                method=method,
                level=level,
                kind=kind,
                prev=prev,
                end=end,
                energy_cum=energy_cum,
                energy_total=energy_total,
                breakpoints=breakpoints,
            )
            new_sublods.append(
                AdditiveSubLOD(
                    centers=centers_full[prev:end].astype(np.float32, copy=False),
                    amplitudes=amps_full[prev:end].astype(np.float32, copy=False),
                    cholesky_factors=chol_full[prev:end].astype(np.float32, copy=False),
                    colors=(colors_full[prev:end] if colors_full is not None else None),
                    label_ids=(
                        label_ids_full[prev:end] if label_ids_full is not None else None
                    ),
                    label_vocabulary=target_view.label_vocabulary,
                    stats=lod_stats,
                    truncation_radius=target_view.truncation_radius,
                )
            )
            prev = end

    # The leaf's absolute reference energy w = Σ aᵢ²·π^{D/2}·|Σᵢ|^{1/2} — the
    # weight of this leaf in partition-level quality aggregation (disjoint
    # regions ⇒ L² decomposes additively; see lod/quality.py). The ordering
    # score already carries a²·|Σ|^{1/2}; multiply the shared constant back.
    reference_energy = energy_total * math.pi ** (target_view.ndim / 2.0)
    if not np.isfinite(reference_energy):
        reference_energy = 0.0

    # Build new substitutive_levels: replace target index with the new
    # ladder; carry the rest through verbatim.
    merged_level_stats = {
        **data.substitutive_levels[s_target].stats,
        "lod_method": method,
        "lod_n_lods": len(new_sublods),
        "lod_breakpoints_kind": kind,
        "lod_cutpoints": [int(c) for c in cuts],
    }
    # The ladder's own total is only a FALLBACK weight: a substitutive build
    # stamps the group-consistent finest-content energy first (see
    # make_substitutive_lod) and that must win for coarser levels — self-
    # energy is quadratic in amplitude, so per-level totals differ and would
    # skew partition-of-lod aggregation.
    # Both-or-neither: a reveal ladder omits `energy_fraction_cum` per sub-LOD
    # (above), so it must omit the leaf's `reference_energy` too. The pair is a
    # contract — the viewer's display gate uses `reference_energy` as the
    # aggregation weight for the per-level fractions, and a weight with nothing
    # to weight is a half-written stamp.
    #
    # POP, not merely skip: `merged_level_stats` inherits the input level's stats,
    # so a weight is usually already there — a substitutive build stamps one per
    # level (so `--recipe levels/adaptive -m radial` hits this on every level),
    # and so do `gsplat additive` over an annotated tree and any re-ladder of a
    # previously energy-ordered leaf. Skipping the `setdefault` would leave those
    # untouched and the ladder half-stamped anyway.
    #
    # The n == 0 branch is the one exception: it labels itself `lod_method="none"`
    # and stamps a trivially-complete e(k)=1.0 (nothing to stream, so 1/e(k) is
    # exactly 1), so it keeps its weight — and `annotate-quality` mirrors that
    # empty-leaf case, which it could not do if the pair were split here.
    if _is_reveal_method(method) and n > 0:
        merged_level_stats.pop("reference_energy", None)
    else:
        merged_level_stats.setdefault("reference_energy", float(reference_energy))
    new_sub_levels = list(data.substitutive_levels)
    new_sub_levels[s_target] = SubstitutiveLevel(
        additive_sublods=new_sublods,
        compression_factor=data.substitutive_levels[s_target].compression_factor,
        parent_method=data.substitutive_levels[s_target].parent_method,
        level_index=data.substitutive_levels[s_target].level_index,
        stats=merged_level_stats,
    )

    out_stats = dict(data.stats)
    out_stats.update(
        {
            "lod_method": method,
            "lod_n_lods": len(new_sublods),
            "lod_breakpoints_kind": kind,
            "lod_cutpoints": [int(c) for c in cuts],
            "lod_substitutive_level": s_target,
        }
    )
    return GSplatData(
        substitutive_levels=new_sub_levels,
        stats=out_stats,
    )
