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

Implementation: ``‖A−B‖² = ‖A‖² − 2⟨A,B⟩ + ‖B‖²`` where every term expands
into pairwise Gaussian inner products. The exact per-splat SELF terms (the
diagonals of ``‖A‖²``/``‖B‖²``) are O(N) closed form and always computed on
the FULL mixtures; only the pairwise (off-diagonal / cross) sums use the
sparse kNN pair lists shared with the L² refiner
(:mod:`._substitutive.refine`), evaluated no-grad and chunked. Mixtures above
``max_pair_splats`` are subsampled for the pair terms with a fixed-seed
uniform draw (deterministic ⇒ reproducible stamps; uniform ⇒ the
inverse-inclusion-fraction rescale of every pair sum is unbiased — see
``_pair_view`` on why evenly-spaced strides are unsafe here). The exact
diagonals always use the full mixtures. Everything runs in float64 (MPS is
routed to CPU, mirroring ``make_substitutive_lod``).

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
    _knn_pairs,
    _pair_K_sum_chunked,
)
from luxar.gsplats.utils.device import resolve_torch_device
from luxar.gsplats.utils.trils import unpack_tril
from luxar.utils.spatial_hash import BatchedSpatialHashGrid

__all__ = ["QualityResult", "mixture_quality", "total_self_energy"]

#: Pair-term subsampling threshold. Mixtures at or below this size use their
#: full pair lists; larger ones are subsampled (fixed-seed uniform draw — see
#: ``_pair_view``) to this many splats for the pairwise sums (the exact
#: diagonals always use the full arrays). 2M matches the refiner's pair
#: working-set scale.
DEFAULT_MAX_PAIR_SPLATS = 2_000_000


def _sqrt_det(data: GSplatData) -> np.ndarray:
    """Per-splat ``|Σᵢ|^{1/2}`` = |product of Cholesky diagonals| (float64)."""
    diag = data._cholesky_diag_elements()  # (N, d)
    out: np.ndarray = np.abs(np.prod(diag.astype(np.float64), axis=1))
    return out


def total_self_energy(data: GSplatData) -> float:
    """Exact total self-energy ``Σᵢ aᵢ²·π^{D/2}·|Σᵢ|^{1/2}`` (float64, O(N)).

    This is the ``reference_energy`` weight ``w`` stamped per leaf: partition
    aggregates combine child qualities as the ``w``-weighted mean (disjoint
    regions ⇒ L² decomposes additively over parts).
    """
    if data.n_splats == 0:
        return 0.0
    amps = np.asarray(data.amplitudes, dtype=np.float64)
    return float(gaussian_self_energy_numpy(amps, _sqrt_det(data), data.ndim).sum())


@dataclass(frozen=True)
class QualityResult:
    """Result of :func:`mixture_quality` (all plain Python scalars)."""

    #: ``clamp(1 − l2_sq/ref_norm_sq, 0, 1)`` — the stampable quality.
    quality: float
    #: ``‖approx − ref‖²`` (closed-form mixture L², possibly sampled).
    l2_sq: float
    approx_norm_sq: float
    ref_norm_sq: float
    #: Inclusion fractions used for the pair terms (1.0 = exact/full).
    approx_pair_fraction: float
    ref_pair_fraction: float
    n_cross_pairs: int
    n_approx_pairs: int
    n_ref_pairs: int


def _pair_view(
    data: GSplatData, max_pair_splats: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(centers, L, amps, inclusion_fraction) for the pair terms, float64.

    Above ``max_pair_splats``, rows are subsampled with a FIXED-SEED uniform
    draw (deterministic, so stamps are reproducible) rather than the encoder's
    evenly-spaced linspace: both mixtures are Hilbert/ladder-ordered, and two
    regular strides can ALIAS — near-duplicate cross partners then get
    included at far above the ``p_a·p_b`` rate and the inverse-fraction
    rescale systematically overestimates the cross term. Independent uniform
    inclusion keeps every pair sum unbiased. Amplitudes are NOT rescaled here
    — pair sums are rescaled by the inverse inclusion fractions at the call
    sites (the exact diagonals never go through this view).
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
    amps = np.array(data.amplitudes, dtype=np.float64)[idx]
    return centers, L, amps, float(idx.size) / float(n)


def _median_sigma(L: np.ndarray) -> float:
    """Median absolute Cholesky diagonal — the pair-radius scale."""
    d = L.shape[-1]
    return float(np.median(np.abs(L[:, np.arange(d), np.arange(d)])))


#: Hard cap on the adaptive per-query neighbour count — bounds pair-list cost
#: on pathologically dense clusters (documented truncation bias there).
_PAIR_K_MAX = 1024


def _adaptive_k(mu: np.ndarray, radius: float, base_k: int) -> int:
    """Per-query neighbour count sized so the RADIUS is the binding prune.

    A fixed k truncates density-dependently: the same k that covers a sparse
    mixture's radius-ball caps a dense one's, and — critically for the sampled
    estimator — a subsample is less dense than its full mixture, so fixed-k
    pair lists are truncated differently on the two and the ``1/p²`` rescale
    inflates the sampled sums. Sizing k from the expected number of neighbours
    inside the radius ball (uniform-density estimate over the live-extent
    dims, ×1.5 safety) makes the radius govern in both regimes. Clamped to
    ``[base_k, _PAIR_K_MAX]``; degenerate dims are excluded from the volume
    exactly as in ``_hash_cell_size``.
    """
    n = mu.shape[0]
    if n <= base_k:
        return base_k
    ext = (mu.max(0) - mu.min(0)).astype(np.float64)
    live = ext > 1e-3 * float(ext.max())
    d_live = int(live.sum())
    if d_live == 0:
        return int(min(_PAIR_K_MAX, n))  # all points coincide: every pair is near
    vol = float(np.prod(ext[live]))
    if vol <= 0.0:
        return base_k
    ball = math.pi ** (d_live / 2.0) / math.gamma(d_live / 2.0 + 1.0) * radius**d_live
    expected = n * min(ball / vol, 1.0)
    return int(min(max(base_k, math.ceil(1.5 * expected)), _PAIR_K_MAX, n))


def _self_pair_list(
    mu: np.ndarray, *, k: int, radius: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unordered within-mixture pairs ``(i < j)`` via OR-symmetrized kNN.

    Same construction as the refiner's coarse–coarse list: generous-k kNN with
    a radius prune, then canonicalize each directed edge to ``(min, max)`` and
    keep every unordered pair once (a pair survives when EITHER endpoint saw
    the other — per-query kNN truncation is asymmetric in dense regions).
    """
    n = mu.shape[0]
    grid = BatchedSpatialHashGrid.from_points(
        mu, cell_size=_hash_cell_size(mu), device=str(device)
    )
    k = _adaptive_k(mu, radius, k)
    qi, pj = _knn_pairs(mu, grid, k + 1, radius, device)  # +1: self hit dropped below
    lo = torch.minimum(qi, pj)
    hi = torch.maximum(qi, pj)
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    if lo.numel() == 0:
        return lo, hi
    key = torch.unique(lo * n + hi)
    return key // n, key % n


def _cross_pair_list(
    a_mu: np.ndarray, b_mu: np.ndarray, *, k: int, radius: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordered cross pairs ``(i∈A, j∈B)`` via the OR-closure of kNN both ways.

    TRUNCATION CONSISTENCY IS LOAD-BEARING: the three terms of
    ``‖A−B‖² = ‖A‖² − 2⟨A,B⟩ + ‖B‖²`` only cancel correctly when their pair
    sets agree. With the same ``k``/``radius`` as :func:`_self_pair_list`, the
    OR-closure over A==B is exactly the self pairs (both orders) plus the
    diagonal — so ``mixture_quality(x, x)`` cancels to ~0 by construction. A
    one-directional or smaller-k cross list (the refiner's ``cross_k=16``
    default) systematically over/under-counts the cross term relative to the
    norms and biases the quality by several percent even at identity.
    """
    nb = b_mu.shape[0]
    grid_b = BatchedSpatialHashGrid.from_points(
        b_mu, cell_size=_hash_cell_size(b_mu), device=str(device)
    )
    ai1, bj1 = _knn_pairs(
        a_mu, grid_b, _adaptive_k(b_mu, radius, k) + 1, radius, device
    )
    grid_a = BatchedSpatialHashGrid.from_points(
        a_mu, cell_size=_hash_cell_size(a_mu), device=str(device)
    )
    bi2, aj2 = _knn_pairs(
        b_mu, grid_a, _adaptive_k(a_mu, radius, k) + 1, radius, device
    )
    ai = torch.cat([ai1, aj2])
    bj = torch.cat([bj1, bi2])
    if ai.numel() == 0:
        return ai, bj
    key = torch.unique(ai * nb + bj)
    return key // nb, key % nb


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

    The kNN pair truncation (radii from :class:`L2RefineConfig`) is the same
    approximation the L² refiner trusts for its objective — far pairs are
    negligible by construction. Above ``max_pair_splats`` the pair terms are
    evenly-spaced-sampled estimates; the exact self-energy diagonals always
    use the full mixtures, so the dominant term of each norm is exact.

    Sampling caveat: when the two mixtures share literally identical splats,
    the cross term's mass concentrates on the few matched pairs and the
    sampled estimate becomes high-variance (unbiased, but noisy). The
    intended use (a MERGED coarse level vs the finest content) never shares
    rows; do not use the sampled path to compare a mixture against its own
    row-prefix (that is what the cheap ``e(k)`` cumulative energy fraction is
    for).
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

    # float64 throughout — MPS lacks float64, so route it to CPU (the same
    # fallback make_substitutive_lod applies for its float64 stages).
    dev = resolve_torch_device(None if device == "auto" else device)
    if dev.type == "mps":
        dev = torch.device("cpu")

    a_mu, a_L, a_amps, p_a = _pair_view(approx, max_pair_splats)
    b_mu, b_L, b_amps, p_b = _pair_view(reference, max_pair_splats)
    sigma_a = _median_sigma(a_L)
    sigma_b = _median_sigma(b_L)

    def to(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(device=dev)

    A = (to(a_mu), to(a_L), to(a_amps))
    B = (to(b_mu), to(b_L), to(b_amps))

    # One k for all three lists; radii scale with each list's own σ, and the
    # cross radius takes the larger σ so its coverage dominates both self
    # lists — see _cross_pair_list on why truncation consistency matters.
    k = cfg.cc_k
    r_a = cfg.cc_radius_sigmas * sigma_a
    r_b = cfg.cc_radius_sigmas * sigma_b
    r_x = max(r_a, r_b)
    aai, aaj = _self_pair_list(a_mu, k=k, radius=r_a, device=dev)
    bbi, bbj = _self_pair_list(b_mu, k=k, radius=r_b, device=dev)
    ci, cb = _cross_pair_list(a_mu, b_mu, k=k, radius=r_x, device=dev)

    with torch.no_grad():
        cross = float(_pair_K_sum_chunked(*A, *B, ci, cb)) / (p_a * p_b)
        a_off = 2.0 * float(_pair_K_sum_chunked(*A, *A, aai, aaj)) / (p_a * p_a)
        b_off = 2.0 * float(_pair_K_sum_chunked(*B, *B, bbi, bbj)) / (p_b * p_b)

    approx_norm_sq = approx_diag + a_off
    ref_norm_sq = ref_diag + b_off
    l2_sq = approx_norm_sq - 2.0 * cross + ref_norm_sq
    # Sampling noise / kNN truncation can push the ratio marginally outside
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
        n_cross_pairs=int(ci.numel()),
        n_approx_pairs=int(aai.numel()),
        n_ref_pairs=int(bbi.numel()),
    )
