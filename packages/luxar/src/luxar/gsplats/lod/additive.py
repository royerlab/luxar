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

The greedy path uses a sparse Gram matrix built via $3\\sigma$ Mahalanobis
truncation + k-d-tree pruning (Algorithm 4.4 in the supp doc), keeping
memory at $O(\\mathrm{nnz}(\\mathbf{G}))$.  At $N \\leq 2000$ a dense Gram
+ scan-greedy is faster than the heap-based lazy greedy due to Python
overhead (supp doc §4.3); we switch automatically.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Literal, Sequence, Union

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
from luxar.gsplats.utils.trils import unpack_tril
from luxar.utils.spatial_hash import BatchedSpatialHashGrid

MethodName = Literal["greedy", "self_energy", "mass", "amplitude", "spectral", "random"]
_VALID_METHODS = (
    "greedy",
    "self_energy",
    "mass",
    "amplitude",
    "spectral",
    "random",
)

#: ``method`` accepted at the API/CLI boundary, including the size-adaptive
#: ``"auto"`` sentinel resolved by :func:`resolve_additive_method`.
AutoOrMethod = Literal[
    "auto", "greedy", "self_energy", "mass", "amplitude", "spectral", "random"
]
_VALID_CHOICES: tuple[str, ...] = ("auto", *_VALID_METHODS)

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


#: Assumed downlink for streaming-breakpoint sizing when the caller gives none —
#: a conservative "typical broadband" figure that also covers good 4G.
DEFAULT_BANDWIDTH_MBPS = 25.0

#: Hard cap on the number of levels a ``stream:<c>`` ladder may produce. The
#: geometric doubling schedule gives ~log2(N/c) levels, so 16 covers c·2^15
#: splats (≈ 460 M at c=14 k) — far beyond realistic leaves. On hitting the cap
#: the last cut jumps straight to N.
DEFAULT_STREAM_MAX_LEVELS = 16


def streaming_chunk_splats(
    target_ms: float,
    bandwidth_mbps: float,
    bytes_per_splat: float,
) -> int:
    """Splat count whose download takes ``target_ms`` at ``bandwidth_mbps``.

    Pure sizing math for the ``stream:<c>`` breakpoint spec:
    ``bandwidth_mbps × 125_000 B/s/Mbps × target_ms/1000 ÷ bytes_per_splat``.
    E.g. 200 ms @ 25 Mbps @ 45 B/splat → ~13.9 k splats.
    """
    if target_ms <= 0:
        raise ValueError(f"target_ms must be positive; got {target_ms}")
    if bandwidth_mbps <= 0:
        raise ValueError(f"bandwidth_mbps must be positive; got {bandwidth_mbps}")
    if bytes_per_splat <= 0:
        raise ValueError(f"bytes_per_splat must be positive; got {bytes_per_splat}")
    return max(
        1, round(bandwidth_mbps * 125_000.0 * (target_ms / 1000.0) / bytes_per_splat)
    )


#: Breakpoint specification for the additive ladder. String forms:
#: ``"equal-count"`` (n_lods equal levels) and ``"stream:<c>"`` (geometric
#: cumulative cuts ``[c, 2c, 4c, …, N]`` — a bandwidth-derived first chunk that
#: doubles; resolved per-N inside :func:`_resolve_breakpoints`, so the same spec
#: adapts to every part/level size). List forms: ``list[int]`` explicit
#: cumulative counts; ``list[float]`` cumulative energy fractions in (0, 1].
BreakpointSpec = Union[str, Sequence[int], Sequence[float]]


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


def sibling_aware_stream_breakpoints(
    breakpoints: BreakpointSpec,
    leaf_n: int,
    compression_factor: int,
) -> BreakpointSpec:
    """Raise a ``stream:C`` ladder's first chunk for a leaf that has a
    COARSER SIBLING in its lod group.

    Measured pathology (h2afva vrefit, 23.4M splats): with every level's
    geometric ladder starting at the SAME small base chunk, the point where a
    finer level's committed content catches up with its coarser sibling —
    whether by count, energy, or measured L² quality — structurally lands
    ``log2(sibling_total / base)`` sequential network passes into the ladder,
    i.e. always 2-3 chunks from the END. Upgrades therefore feel like
    "waits until fully loaded".

    Fix the geometry instead of the currency: a leaf whose group contains a
    coarser sibling starts its ladder at ``ceil(leaf_n / (2·K))`` — half the
    sibling's expected size — so the catch-up fires at chunk 1-2 by
    construction (energy-ordered first chunks of that size carry ~70%+ of the
    leaf's energy on real data, comfortably past the viewer's committed-energy
    switch threshold). The user's ``stream:C`` base still applies wherever it
    is LARGER, and — crucially — the group's COARSEST leaf must NOT go through
    this helper: it is the eager default level whose small first chunk is the
    fast-first-paint path.

    Non-``stream:`` specs (equal-count, explicit counts, energy fractions)
    pass through untouched — their chunk structure has no shared-base
    pathology (e.g. equal-count crosses the sibling at chunk 1 already).
    """
    if not (isinstance(breakpoints, str) and breakpoints.startswith("stream:")):
        return breakpoints
    try:
        user_base = int(breakpoints[len("stream:") :])
    except ValueError:
        return breakpoints
    sibling_base = math.ceil(leaf_n / (2.0 * max(2, compression_factor)))
    return f"stream:{max(user_base, sibling_base)}"


def _self_energy_score(data: GSplatData) -> np.ndarray:
    """Self-energy ranking score: $a_i^2 \\, |\\Sigma_i|^{1/2}$.

    The $\\pi^{D/2}$ constant is shared across all splats and drops out
    of the ordering.
    """
    out: np.ndarray = np.asarray(data.amplitudes, dtype=np.float64) ** 2 * _det_L(data)
    return out


def _mass_score(data: GSplatData) -> np.ndarray:
    """Integral-mass ranking score: $a_i \\, |\\Sigma_i|^{1/2}$.

    The $(2\\pi)^{D/2}$ constant is shared across all splats and drops
    out of the ordering.
    """
    out: np.ndarray = np.asarray(data.amplitudes, dtype=np.float64) * _det_L(data)
    return out


def _truncation_radii(data: GSplatData, sigmas: float = 3.0) -> np.ndarray:
    """Per-splat truncation radius $r_i = \\sigma\\,\\sqrt{\\lambda_{\\max}(\\Sigma_i)}$."""
    if data.n_splats == 0:
        return np.empty(0, dtype=np.float64)
    L = unpack_tril(np.asarray(data.cholesky_factors, dtype=np.float64), data.ndim)
    return truncation_radii_numpy(L, sigmas=sigmas)


# ─────────────────────────────────────────────────────────────────────
# Sparse Gram (3σ Mahalanobis pruning + k-d tree, supp doc Algo 4.4)
# ─────────────────────────────────────────────────────────────────────


def _build_sparse_gram(data: GSplatData, sigmas: float = 3.0) -> sparse.csr_matrix:
    """Build a sparse symmetric Gram matrix via $3\\sigma$ pruning.

    Two splats whose $3\\sigma$ ellipsoids do not overlap have an
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
    amps = np.asarray(data.amplitudes, dtype=np.float64)
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


def compute_additive_order(
    data: GSplatData,
    method: AutoOrMethod = "auto",
    *,
    truncation_sigmas: float = 3.0,
    max_n_dense: int = 2_000,
    seed: int | None = None,
) -> np.ndarray:
    """Compute an additive ordering permutation for the splats in ``data``.

    Parameters
    ----------
    data : GSplatData
        Fitted (single- or multi-LOD) gsplat dataset.  Operates on the
        flattened concatenation across LODs.
    method : str
        One of ``auto``, ``greedy``, ``self_energy``, ``mass``,
        ``amplitude``, ``spectral``, ``random``.  ``auto`` (the default)
        resolves to ``greedy`` at small N and ``self_energy`` above
        :data:`_AUTO_ADDITIVE_MAX_N` — see :func:`resolve_additive_method`.
        See module docstring for details.
    truncation_sigmas : float
        Mahalanobis cutoff used for sparse-Gram pruning (default 3.0).
        Only relevant for ``greedy`` and ``spectral``.
    max_n_dense : int
        For ``greedy``, build a dense Gram and use scan-greedy when
        $N \\leq$ this threshold.  Above it, build a sparse Gram and
        use lazy-greedy.  Default 2000 (per supp doc §4.3).
    seed : int, optional
        Random seed for ``method='random'``.

    Returns
    -------
    np.ndarray of shape (N,), dtype int64
        ``order[k]`` is the original index of the splat at rank $k$.
    """
    if method not in _VALID_CHOICES:
        raise ValueError(f"method must be one of {_VALID_CHOICES}, got {method!r}")

    N = data.n_splats
    if N == 0:
        return np.empty(0, dtype=np.int64)
    if N == 1:
        return np.array([0], dtype=np.int64)

    # Resolve the size-adaptive sentinel ONCE, before any (expensive) Gram
    # build, so every downstream branch sees a concrete method.
    method = resolve_additive_method(method, N)

    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(N).astype(np.int64)

    if method == "amplitude":
        score = np.asarray(data.amplitudes, dtype=np.float64)
        return np.argsort(-score, kind="stable").astype(np.int64)

    if method == "mass":
        score = _mass_score(data)
        return np.argsort(-score, kind="stable").astype(np.int64)

    if method == "self_energy":
        score = _self_energy_score(data)
        return np.argsort(-score, kind="stable").astype(np.int64)

    # Spectral and greedy need the Gram matrix.
    gram_csr = _build_sparse_gram(data, sigmas=truncation_sigmas)
    return _order_from_gram(gram_csr, method, max_n_dense)


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


def _resolve_breakpoints(
    n: int,
    n_lods: int,
    breakpoints: BreakpointSpec,
) -> tuple[list[int], str]:
    """Resolve ``breakpoints`` to a list of cumulative cutpoints ending at $N$.

    Returns ``(cumulative_counts, kind)`` where ``kind`` is one of
    ``equal-count``, ``stream``, ``explicit-counts``, ``energy-fractions``.
    """
    if isinstance(breakpoints, str):
        if breakpoints.startswith("stream:"):
            # Bandwidth-derived streaming ladder: geometric cumulative cuts
            # [c, 2c, 4c, …] (increments [c, c, 2c, …] — first paint = c splats,
            # then doubling), resolved against THIS n so the same spec adapts to
            # every part/level size. Silently clamped, never raises on small n
            # (unlike explicit counts — deliberate: per-part N is unknowable to
            # the user). Capped at DEFAULT_STREAM_MAX_LEVELS; a final increment
            # smaller than c/2 folds into the previous cut (no sliver levels).
            body = breakpoints[len("stream:") :]
            try:
                c = int(body)
            except ValueError as e:
                raise ValueError(
                    "stream breakpoints must be 'stream:<c>' with integer "
                    f"c >= 1; got {breakpoints!r}"
                ) from e
            if c < 1:
                raise ValueError(f"stream first-chunk size must be >= 1; got {c}")
            if n <= c:
                return [n], "stream"
            cuts: list[int] = []
            cum = c
            while cum < n and len(cuts) < DEFAULT_STREAM_MAX_LEVELS - 1:
                cuts.append(cum)
                cum *= 2
            # Fold a sliver tail (< c/2 remaining) into the previous cut.
            if cuts and (n - cuts[-1]) < c / 2:
                cuts.pop()
            cuts.append(n)
            return cuts, "stream"
        if breakpoints != "equal-count":
            raise ValueError(
                f"unknown breakpoints string {breakpoints!r}; "
                "expected 'equal-count', 'stream:<c>', or a list."
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
            "breakpoints must be 'equal-count', 'stream:<c>', a list of ints "
            "(cumulative counts), or a list of floats in (0, 1] "
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


def make_additive_lod(
    data: GSplatData,
    n_lods: int = 4,
    *,
    method: AutoOrMethod = "auto",
    breakpoints: BreakpointSpec = "equal-count",
    truncation_sigmas: float = 3.0,
    max_n_dense: int = 2_000,
    seed: int | None = None,
    substitutive_level: int | None = None,
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
        - list[int]: explicit cumulative splat counts per level.
        - list[float] in $(0, 1]$: cumulative energy fractions; the
          smallest $k$ at which the cumulative-utility curve crosses
          each fraction is used as the cutpoint.
    truncation_sigmas : float
        $\\sigma$ multiplier for sparse-Gram pruning.  Default 3.0.
    max_n_dense : int
        Threshold below which ``greedy`` uses a dense Gram + scan-greedy.
    seed : int, optional
        Random seed for ``method='random'``.
    substitutive_level : int, optional
        Index of the substitutive level to build the ladder for. Defaults
        to ``data.default_substitutive``.

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

    # Resolve the size-adaptive sentinel ONCE, before the (expensive) Gram
    # build below, so `needs_gram` and the recorded `lod_method` stat both
    # see the concrete method. (compute_additive_order resolves it again
    # harmlessly for the score-method path — it is idempotent.)
    method = resolve_additive_method(method, n)

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
            _build_sparse_gram(target_view, sigmas=truncation_sigmas)
            if needs_gram
            else None
        )
        if gram_csr is not None:
            order = _order_from_gram(gram_csr, method, max_n_dense)
        else:
            order = compute_additive_order(
                target_view,
                method=method,
                truncation_sigmas=truncation_sigmas,
                max_n_dense=max_n_dense,
                seed=seed,
            )

        cuts_or_fracs, kind = _resolve_breakpoints(n, n_lods, breakpoints)

        if kind == "energy-fractions":
            if gram_csr is None:  # score method + energy breakpoints
                gram_csr = _build_sparse_gram(target_view, sigmas=truncation_sigmas)
            cuts = _energy_fraction_cuts(
                [float(x) for x in cuts_or_fracs], gram_csr, order
            )
        else:
            cuts = [int(x) for x in cuts_or_fracs]

        centers_full = np.asarray(target_view.centers)[order]
        amps_full = np.asarray(target_view.amplitudes)[order]
        chol_full = np.asarray(target_view.cholesky_factors)[order]
        colors_full = (
            np.asarray(target_view.colors)[order]
            if target_view.colors is not None
            else None
        )

        # Cumulative self-energy over the ladder ordering — the e(k) of the
        # viewer's committed quality Q·e(k). Fractions, so the shared π^{D/2}
        # constant of the true self-energy cancels and the (cheap, O(N))
        # ordering score suffices; π^{D/2} is multiplied back only for the
        # absolute reference_energy weight w (partition aggregation).
        energy_cum = np.cumsum(_self_energy_score(target_view)[order])
        energy_total = float(energy_cum[-1]) if energy_cum.size else 0.0

        new_sublods = []
        prev = 0
        for level, end in enumerate(cuts):
            end = int(end)
            if end <= prev:
                continue
            lod_stats: dict[str, Any] = {
                "lod_method": method,
                "lod_level": level,
                "lod_breakpoints_kind": kind,
                "lod_n_splats": int(end - prev),
                "lod_cumulative_n": end,
            }
            if energy_total > 0.0:
                e_frac = float(energy_cum[end - 1] / energy_total)
                if np.isfinite(e_frac):
                    lod_stats["energy_fraction_cum"] = min(1.0, max(0.0, e_frac))
            if kind == "stream":
                # Provenance: the bandwidth-derived first-chunk size, otherwise
                # only recoverable by re-parsing the breakpoints string.
                lod_stats["lod_stream_chunk_splats"] = int(
                    str(breakpoints)[len("stream:") :]
                )
            new_sublods.append(
                AdditiveSubLOD(
                    centers=centers_full[prev:end].astype(np.float32, copy=False),
                    amplitudes=amps_full[prev:end].astype(np.float32, copy=False),
                    cholesky_factors=chol_full[prev:end].astype(np.float32, copy=False),
                    colors=(colors_full[prev:end] if colors_full is not None else None),
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
