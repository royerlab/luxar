"""Measured approximation quality for gsplat mixtures (the Q·e scheme).

The LOD viewer needs to answer "is the committed part of level A at least as
good an approximation as what level B shows?" — and neither splat counts nor
mass can answer it (counts compare merged coarse blobs against fine splats;
mass conservation pins every complete level to the same total). The honest
currency is a measured fidelity: the closed-form mixture L² against a common
reference, normalized to a unitless quality

    Q(approx, ref) = 1 − ‖approx − ref‖² / ‖ref‖²   ∈ [0, 1]

with ``Q(ref, ref) = 1`` by construction. ``Q`` is computed ONCE at build /
annotate time per substitutive level (against the group's finest content) and
stamped into ``level_stats``; the viewer combines it with the additive
ladder's cumulative energy fraction ``e(k)`` (``lod_stats``) into the
committed quality ``Q·e(k)`` — see ``docs/specs/GSPLATS_ZARR_FORMAT.md``.

Estimator design — ``‖A−B‖² = ‖A‖² − 2⟨A,B⟩ + ‖B‖²`` where every term
expands into pairwise Gaussian inner products:

- The per-splat SELF diagonals of ``‖A‖²``/``‖B‖²`` are O(N) closed form and
  always computed EXACTLY on the full mixtures.
- Every off-diagonal / cross sum is a **directed row-sum estimate**: a
  fixed-seed uniform sample of query splats, each queried against the other
  side's spatial-hash grid (kNN + radius prune), scaled by ``n/|queries|``.
  Query sampling is the load-bearing scalability lever — the hash grid
  gathers candidates in a per-query Python loop (~100 µs/query on both its
  backends), so querying every splat of a multi-million mixture is
  intractable; a bounded query sample estimates the same sums unbiasedly at
  constant cost. The cross term averages the A-side and B-side row-sum
  estimates for symmetric coverage.
- TRUNCATION CONSISTENCY IS LOAD-BEARING: all directed sums of one
  comparison share per-side radii (shrunk to a density budget derived from
  FULL-population counts, so exact and sampled runs truncate identically)
  and adaptive k sized so the radius is the binding prune. For ``A == B``
  the four directed queries then see identical grids/k/radii and the three
  terms cancel: ``mixture_quality(x, x) ≈ 1`` by construction.
- Row subsampling for the pair views uses a fixed-seed UNIFORM draw
  (deterministic ⇒ reproducible stamps; uniform ⇒ the inverse-inclusion
  rescale is unbiased). Evenly-spaced strides are unsafe: both mixtures are
  Hilbert/ladder-ordered and two regular strides alias, over-including
  near-duplicate cross partners.

Kernel sums run in float64 (MPS lacks float64 → CPU, mirroring
``make_substitutive_lod``); the spatial grids run on the fast device when
one is available (positions are float32 there).

No optimizer, no gradients — this module only *measures*.

@module luxar.gsplats.lod.quality
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod._kernels import gaussian_self_energy_numpy
from luxar.gsplats.lod._substitutive.refine import (
    L2RefineConfig,
    _hash_cell_size,
    _pair_K_sum_chunked,
)
from luxar.gsplats.utils.alpha import effective_amplitudes
from luxar.gsplats.utils.device import resolve_torch_device
from luxar.gsplats.utils.trils import unpack_tril
from luxar.utils.spatial_hash import BatchedSpatialHashGrid

__all__ = ["QualityResult", "mixture_quality", "total_self_energy"]

#: Pair-view subsampling threshold (rows entering the kernel tensors).
DEFAULT_MAX_PAIR_SPLATS = 2_000_000

#: Query budget per directed row-sum. The hash grid's candidate gathering is
#: a per-query Python loop, so queries — not kernel evals — dominate; this
#: bounds each of the four directed sums to a constant number of queries.
_QUERY_BUDGET = 8_192

#: Target expected neighbours inside the pair radius (at FULL-population
#: density). Denser mixtures get their radius shrunk to meet this budget —
#: bounding per-query candidate counts at any scale. ~1.5× the refiner's cc_k.
_PAIR_EXPECTED_BUDGET = 96

#: Hard cap on the adaptive per-query neighbour count (safety on top of the
#: radius shrink, e.g. pathologically clustered mixtures).
_PAIR_K_MAX = 1024


def _sqrt_det(data: GSplatData) -> np.ndarray:
    """Per-splat ``|Σᵢ|^{1/2}`` = |product of Cholesky diagonals| (float64)."""
    diag = data._cholesky_diag_elements()  # (N, d)
    out: np.ndarray = np.abs(np.prod(diag.astype(np.float64), axis=1))
    return out


def total_self_energy(data: GSplatData) -> float:
    """Exact total self-energy ``Σᵢ aᵢ²·π^{D/2}·|Σᵢ|^{1/2}`` (float64, O(N)).

    ``aᵢ`` is the ALPHA-EFFECTIVE amplitude ``A·α`` (raw amplitude times the
    per-splat color-alpha opacity for RGBA splats; equal to the raw amplitude
    when there is no RGBA alpha) — the same rendered mass the additive ladder
    scores by, so build- and annotate-time stamps stay on one convention.

    This is the ``reference_energy`` weight ``w`` stamped per leaf: partition
    aggregates combine child qualities as the ``w``-weighted mean (disjoint
    regions ⇒ L² decomposes additively over parts).
    """
    if data.n_splats == 0:
        return 0.0
    amps = np.asarray(effective_amplitudes(data), dtype=np.float64)
    return float(gaussian_self_energy_numpy(amps, _sqrt_det(data), data.ndim).sum())


@dataclass(frozen=True)
class QualityResult:
    """Result of :func:`mixture_quality` (all plain Python scalars)."""

    #: ``clamp(1 − l2_sq/ref_norm_sq, 0, 1)`` — the stampable quality.
    quality: float
    #: ``‖approx − ref‖²`` (closed-form mixture L², estimated — see module doc).
    l2_sq: float
    approx_norm_sq: float
    ref_norm_sq: float
    #: Row-inclusion fractions of the pair views (1.0 = full mixture).
    approx_pair_fraction: float
    ref_pair_fraction: float
    #: Directed pairs evaluated for the cross / self sums (post-sampling).
    n_cross_pairs: int
    n_approx_pairs: int
    n_ref_pairs: int


def _pair_view(
    data: GSplatData, max_pair_splats: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(centers, L, amps, inclusion_fraction) for the pair terms, float64.

    Fixed-seed uniform row subsample above ``max_pair_splats`` (see the
    module docstring on why evenly-spaced strides are unsafe). Amplitudes are
    ALPHA-EFFECTIVE (``A·α``, matching ``total_self_energy``'s exact diagonals)
    but NOT rescaled for the subsample here — sums are rescaled by the inverse
    inclusion fractions at the call sites (the exact diagonals never go through
    this view).
    """
    n = data.n_splats
    if n > max_pair_splats:
        rng = np.random.default_rng(0)  # fixed seed: reproducible stamps
        idx = np.sort(rng.choice(n, size=max_pair_splats, replace=False))
    else:
        idx = np.arange(n, dtype=np.intp)
    centers = np.array(data.centers, dtype=np.float64)[idx]
    tri = np.asarray(data.cholesky_factors, dtype=np.float64)[idx]
    L = unpack_tril(tri, data.ndim)
    amps = np.asarray(effective_amplitudes(data), dtype=np.float64)[idx]
    return centers, L, amps, float(idx.size) / float(n)


def _median_sigma(L: np.ndarray) -> float:
    """Median absolute Cholesky diagonal — the pair-radius scale."""
    d = L.shape[-1]
    return float(np.median(np.abs(L[:, np.arange(d), np.arange(d)])))


def _live_density_volume(mu: np.ndarray) -> tuple[int, float]:
    """(live dims, bbox volume over live dims) — degenerate dims excluded,
    matching ``_hash_cell_size``'s convention."""
    ext = (mu.max(0) - mu.min(0)).astype(np.float64)
    live = ext > 1e-3 * float(ext.max())
    d_live = int(live.sum())
    vol = float(np.prod(ext[live])) if d_live else 0.0
    return d_live, vol


def _ball_volume(d: int, radius: float) -> float:
    return float(math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0) * radius**d)


def _effective_radius(mu: np.ndarray, n_full: int, nominal: float) -> float:
    """Shrink the pair radius so the expected FULL-density neighbour count
    inside it stays ≤ ``_PAIR_EXPECTED_BUDGET``.

    Derived from the FULL population count (``n_full``), never the possibly
    subsampled view, so the exact and sampled evaluations of one mixture share
    IDENTICAL truncation — the property the estimator's cancellation rests on
    (extents come from the view; ≈ the full extents for a uniform sample).
    The truncation bias is shared by all terms of one comparison; across
    levels it varies with density, an accepted approximation of the stamp.
    """
    d_live, vol = _live_density_volume(mu)
    if d_live == 0 or vol <= 0.0 or n_full <= 1:
        return nominal
    expected = n_full * min(_ball_volume(d_live, nominal) / vol, 1.0)
    if expected <= _PAIR_EXPECTED_BUDGET:
        return nominal
    return float(nominal * (_PAIR_EXPECTED_BUDGET / expected) ** (1.0 / d_live))


def _adaptive_k(mu: np.ndarray, radius: float, base_k: int) -> int:
    """Per-query neighbour count sized so the RADIUS is the binding prune.

    A fixed k truncates density-dependently — the same k that covers a sparse
    mixture's radius-ball caps a dense one's — so k is sized from the expected
    neighbours inside the radius ball (×1.5 safety), clamped to
    ``[base_k, _PAIR_K_MAX]``. With the radius already shrunk to the density
    budget this rarely exceeds ``base_k`` by much.
    """
    n = mu.shape[0]
    if n <= base_k:
        return base_k
    d_live, vol = _live_density_volume(mu)
    if d_live == 0:
        return int(min(_PAIR_K_MAX, n))  # all points coincide: every pair is near
    if vol <= 0.0:
        return base_k
    expected = n * min(_ball_volume(d_live, radius) / vol, 1.0)
    return int(min(max(base_k, math.ceil(1.5 * expected)), _PAIR_K_MAX, n))


def _query_sample(n: int, seed: int) -> np.ndarray:
    """Fixed-seed uniform query sample (all rows when within budget)."""
    if n <= _QUERY_BUDGET:
        return np.arange(n, dtype=np.intp)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=_QUERY_BUDGET, replace=False))


def _directed_row_sum(
    X: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Y: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    x_mu_np: np.ndarray,
    q_idx: np.ndarray,
    grid_y: BatchedSpatialHashGrid,
    *,
    k: int,
    radius: float,
    drop_self: bool,
    device: torch.device,
) -> tuple[float, int]:
    """Estimate ``Σ_{i∈X} Σ_{j∈Y near i} K(xᵢ, yⱼ)`` from sampled queries.

    Queries ``x_mu_np[q_idx]`` against ``grid_y`` (kNN + radius prune),
    evaluates the pair kernels chunked, and rescales by ``n_x/|q_idx|`` —
    unbiased for a uniform query sample. ``drop_self`` removes the ``i == j``
    hits when X IS Y (the exact diagonal is added separately); cross sums
    keep them (a matched splat in the other mixture is a genuine cross pair).
    Returns ``(scaled_sum, n_pairs_evaluated)``.
    """
    n_x = x_mu_np.shape[0]
    if n_x == 0 or len(grid_y) == 0 or q_idx.size == 0:
        return 0.0, 0
    dist, idx = grid_y.query_knn(x_mu_np[q_idx], k=min(k, len(grid_y)))
    keep = (idx >= 0) & (dist < radius)
    if drop_self:
        keep &= idx != q_idx[:, None]
    rows, cols = np.nonzero(keep)
    if rows.size == 0:
        return 0.0, 0
    ii = torch.from_numpy(q_idx[rows].astype(np.int64)).to(device)
    jj = torch.from_numpy(idx[rows, cols].astype(np.int64)).to(device)
    with torch.no_grad():
        total = float(_pair_K_sum_chunked(*X, *Y, ii, jj))
    return total * (n_x / float(q_idx.size)), int(rows.size)


def mixture_quality(
    approx: GSplatData,
    reference: GSplatData,
    *,
    max_pair_splats: int = DEFAULT_MAX_PAIR_SPLATS,
    config: L2RefineConfig | None = None,
    device: str = "auto",
) -> QualityResult:
    """Measure how well ``approx`` approximates ``reference``.

    Returns :class:`QualityResult` with ``quality = 1 − ‖A−B‖²/‖B‖²`` clamped
    to [0, 1]. Degenerate inputs: an empty ``approx`` scores 0 (it explains
    none of the reference); an empty ``reference`` raises (quality against
    nothing is undefined).

    Estimation caveats (see the module docstring for the design): the kNN +
    radius truncation is the same approximation the L² refiner trusts; query
    and row sampling make the off-diagonal terms estimates (exact diagonals
    dominate); when the two mixtures share literally identical splats the
    cross estimate concentrates on few matched pairs and gets noisy — the
    intended use (a MERGED coarse level vs the finest content) never shares
    rows; a mixture's own row-prefix is what the cheap ``e(k)`` cumulative
    energy fraction is for.
    """
    if reference.n_splats == 0:
        raise ValueError("mixture_quality: reference mixture is empty")
    if approx.ndim != reference.ndim:
        raise ValueError(
            f"mixture_quality: ndim mismatch (approx {approx.ndim} vs "
            f"reference {reference.ndim})"
        )
    cfg = config or L2RefineConfig()

    # Exact diagonals on the FULL mixtures (O(N) closed form, float64).
    approx_diag = total_self_energy(approx)
    ref_diag = total_self_energy(reference)

    if approx.n_splats == 0:
        # Nothing committed: ‖A−B‖² = ‖B‖² → quality 0. The off-diagonal part
        # of ‖B‖² cancels in 1 − ‖B‖²/‖B‖² regardless, so skip the pair work.
        return QualityResult(
            quality=0.0,
            l2_sq=ref_diag,
            approx_norm_sq=0.0,
            ref_norm_sq=ref_diag,
            approx_pair_fraction=1.0,
            ref_pair_fraction=1.0,
            n_cross_pairs=0,
            n_approx_pairs=0,
            n_ref_pairs=0,
        )

    # Two devices, deliberately split: the KERNEL sums run float64 (MPS lacks
    # float64 → CPU, the same fallback make_substitutive_lod applies); the
    # spatial grids take the fast device when available (float32 positions).
    pair_dev = resolve_torch_device(None if device == "auto" else device)
    dev = torch.device("cpu") if pair_dev.type == "mps" else pair_dev

    a_mu, a_L, a_amps, p_a = _pair_view(approx, max_pair_splats)
    b_mu, b_L, b_amps, p_b = _pair_view(reference, max_pair_splats)
    sigma_a = _median_sigma(a_L)
    sigma_b = _median_sigma(b_L)

    def to(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(device=dev)

    A = (to(a_mu), to(a_L), to(a_amps))
    B = (to(b_mu), to(b_L), to(b_amps))

    # Per-side radii shrunk to the density budget from FULL-population counts
    # (identical truncation for exact and sampled runs); the cross radius
    # takes the larger one so its coverage dominates both self sums.
    r_a = _effective_radius(a_mu, approx.n_splats, cfg.cc_radius_sigmas * sigma_a)
    r_b = _effective_radius(b_mu, reference.n_splats, cfg.cc_radius_sigmas * sigma_b)
    r_x = max(r_a, r_b)
    k_a = _adaptive_k(a_mu, r_x, cfg.cc_k)
    k_b = _adaptive_k(b_mu, r_x, cfg.cc_k)

    grid_a = BatchedSpatialHashGrid.from_points(
        a_mu, cell_size=_hash_cell_size(a_mu), device=str(pair_dev)
    )
    grid_b = BatchedSpatialHashGrid.from_points(
        b_mu, cell_size=_hash_cell_size(b_mu), device=str(pair_dev)
    )
    qa = _query_sample(a_mu.shape[0], seed=1)
    qb = _query_sample(b_mu.shape[0], seed=2)

    # Directed row sums (module docstring): self off-diagonals from each
    # mixture's own queries (self-hits dropped, exact diagonal added below);
    # the cross term as the symmetric average of both sides' row sums.
    a_off, n_aa = _directed_row_sum(
        A, A, a_mu, qa, grid_a, k=k_a + 1, radius=r_a, drop_self=True, device=dev
    )
    b_off, n_bb = _directed_row_sum(
        B, B, b_mu, qb, grid_b, k=k_b + 1, radius=r_b, drop_self=True, device=dev
    )
    x_ab, n_ab = _directed_row_sum(
        A, B, a_mu, qa, grid_b, k=k_b + 1, radius=r_x, drop_self=False, device=dev
    )
    x_ba, n_ba = _directed_row_sum(
        B, A, b_mu, qb, grid_a, k=k_a + 1, radius=r_x, drop_self=False, device=dev
    )

    approx_norm_sq = approx_diag + a_off / (p_a * p_a)
    ref_norm_sq = ref_diag + b_off / (p_b * p_b)
    cross = 0.5 * (x_ab + x_ba) / (p_a * p_b)
    l2_sq = approx_norm_sq - 2.0 * cross + ref_norm_sq
    # Estimation noise / truncation can push the ratio marginally outside
    # [0, 1]; the stamp is defined clamped (and must stay finite for the
    # JSON-attrs contract — the viewer's JSON.parse rejects NaN/Inf).
    quality = 1.0 - l2_sq / ref_norm_sq if ref_norm_sq > 0.0 else 0.0
    if not np.isfinite(quality):
        quality = 0.0
    quality = min(1.0, max(0.0, quality))

    return QualityResult(
        quality=float(quality),
        l2_sq=float(l2_sq),
        approx_norm_sq=float(approx_norm_sq),
        ref_norm_sq=float(ref_norm_sq),
        approx_pair_fraction=p_a,
        ref_pair_fraction=p_b,
        n_cross_pairs=int(n_ab + n_ba),
        n_approx_pairs=int(n_aa),
        n_ref_pairs=int(n_bb),
    )
