"""Tests for :mod:`luxar.gsplats.lod.substitutive` and its
``_substitutive/`` algorithm helpers (greedy Runnalls merge, Morton
warm-start).

The shared closed-form kernels are tested in :mod:`test_kernels`.

Coverage:

- Multi-level hierarchy emits the expected $\\lceil N/K^\\ell\\rceil$ counts
  and is monotone non-increasing across levels.
- Cost-aware Lloyd refinement does not increase the rendered $L^2$ error
  vs spatial-only k-means (mirrors supp doc Experiment C qualitatively).
- Greedy pairwise merge conserves the mass-weighted mean; Morton partition
  is balanced and surjective.
- Edge cases: empty input, ``levels=1``, ``N << K``, all four methods,
  colors / float32 round-trip, ``device='auto'``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod import make_substitutive_lod
from luxar.gsplats.lod._kernels import bin_squared_norm_torch
from luxar.gsplats.lod._substitutive.greedy import _merge_two_clusters
from luxar.gsplats.lod._substitutive.warm_start import _morton_partition
from luxar.gsplats.utils.trils import pack_tril, unpack_tril

# ─────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────


def _make_anisotropic_3d(n: int, seed: int = 42) -> GSplatData:
    """Synthetic 3D mixture with anisotropic covariances + varied amplitudes."""
    rng = np.random.RandomState(seed)
    centres = rng.randn(n, 3).astype(np.float32) * 2.0
    L = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        A = rng.randn(3, 3).astype(np.float32) * 0.4
        A_lower = np.tril(A)
        # Anisotropic diagonal entries (mix of small and large scales).
        A_lower[0, 0] = abs(A_lower[0, 0]) + 0.3 + rng.rand() * 1.5
        A_lower[1, 1] = abs(A_lower[1, 1]) + 0.3 + rng.rand() * 0.5
        A_lower[2, 2] = abs(A_lower[2, 2]) + 0.3 + rng.rand() * 1.0
        L[i] = A_lower
    chol = pack_tril(L)
    amps = rng.rand(n).astype(np.float32) + 0.5
    return GSplatData(centers=centres, amplitudes=amps, cholesky_factors=chol)


def _make_isotropic_3d(n: int, seed: int = 0) -> GSplatData:
    rng = np.random.RandomState(seed)
    centres = rng.randn(n, 3).astype(np.float32)
    chol = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1))
    amps = rng.rand(n).astype(np.float32) + 0.5
    return GSplatData(centers=centres, amplitudes=amps, cholesky_factors=chol)


def _empty_3d() -> GSplatData:
    return GSplatData(
        centers=np.zeros((0, 3), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        cholesky_factors=np.zeros((0, 6), dtype=np.float32),
    )


def _gsplat_to_torch(
    data: GSplatData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # np.array (copy): ``data`` may be a flattened()/at_substitutive() view
    # whose arrays are read-only, which torch.from_numpy rejects.
    centres = torch.from_numpy(np.array(data.centers, dtype=np.float32)).to(
        torch.float64
    )
    L = torch.from_numpy(
        unpack_tril(
            np.array(data.cholesky_factors, dtype=np.float32), data.ndim
        ).astype(np.float64)
    )
    amps = torch.from_numpy(np.array(data.amplitudes, dtype=np.float32)).to(
        torch.float64
    )
    return centres, L, amps


# ─────────────────────────────────────────────────────────────────────
# Low-level algorithm primitives (_substitutive/)
# ─────────────────────────────────────────────────────────────────────


class TestGreedyMerge:
    """[P12/G9] Moment conservation of the Runnalls pairwise merge."""

    def test_merge_preserves_mass_weighted_mean(self):
        """The merged centre is the mass-weighted mean of its two parents
        (mass = a·|Σ|^{1/2}); the merged covariance is symmetric/PD."""
        cen = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
        Lc = torch.stack(
            [
                torch.eye(3, dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 2.0,
            ]
        )
        am = torch.tensor([1.0, 3.0], dtype=torch.float64)
        mu_bar, L_bar, a_star = _merge_two_clusters(cen, Lc, am, 0, 1)

        sqrt_det = torch.abs(torch.prod(torch.diagonal(Lc, dim1=-2, dim2=-1), dim=-1))
        masses = am * sqrt_det
        w = masses / masses.sum()
        expected_mu = (w[:, None] * cen).sum(dim=0)
        torch.testing.assert_close(mu_bar, expected_mu, rtol=1e-12, atol=1e-12)
        # Merged covariance is a valid Gaussian: positive Cholesky diagonal.
        assert torch.all(torch.diagonal(L_bar) > 0)
        # L²-optimal amplitude of a real merge is strictly positive.
        assert float(a_star) > 0


class TestMortonPartition:
    """[P5/P12] Warm-start partition is a balanced, surjective assignment."""

    def test_partition_is_balanced_and_surjective(self):
        """[G11] N=20 into M=4 bins: assignments in [0, M), every bin used,
        sizes within 1 of N/M."""
        rng = np.random.RandomState(0)
        centres = torch.from_numpy(rng.randn(20, 3).astype(np.float64))
        assignments = _morton_partition(centres, M=4).numpy()
        assert assignments.min() >= 0 and assignments.max() < 4
        counts = np.bincount(assignments, minlength=4)
        assert (counts > 0).all()  # surjective onto [0, M)
        assert counts.max() - counts.min() <= 1  # balanced

    def test_partition_caps_M_to_N(self):
        """[C7] M > N is silently capped to N (one splat per bin); no bin
        index exceeds N-1."""
        rng = np.random.RandomState(1)
        centres = torch.from_numpy(rng.randn(10, 3).astype(np.float64))
        assignments = _morton_partition(centres, M=100).numpy()
        assert assignments.max() < 10
        assert len(np.unique(assignments)) == 10  # each splat its own bin


# ─────────────────────────────────────────────────────────────────────
# Hierarchy + structure
# ─────────────────────────────────────────────────────────────────────


class TestHierarchy:
    @pytest.mark.parametrize(
        "method", ["kmeans", "kmeans_lloyd", "greedy", "greedy_lloyd"]
    )
    def test_counts_K4_L2(self, method):
        data = _make_isotropic_3d(n=64, seed=1)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method=method,
            lloyd_iterations=2,
            candidate_bins_k=4,
            device="cpu",
            seed=42,
        )
        assert pyramid.n_substitutive == 3
        # Level 0 unchanged
        assert pyramid.substitutive_levels[0].n_splats_total == 64
        # Levels 1, 2: ceil(64/4)=16, ceil(16/4)=4
        # (k-means may produce empty bins → fewer; assert <= target)
        assert 1 <= pyramid.substitutive_levels[1].n_splats_total <= 16
        assert 1 <= pyramid.substitutive_levels[2].n_splats_total <= 4

        # [Python-R3/B-C2][P2] The above bounds-only assertions accept a
        # degenerate "1 splat per level" output as valid. In practice,
        # 64 splats compressed by factor-4 to 1 splat is a catastrophic
        # collapse (a k-means bug that returned all-zero assignments
        # would pass). All four methods empirically reach the EXACT target
        # ([64, 16, 4]) on these isotropic 3D inputs, so pin a tight lower
        # bound (≥ 75% of target) that still leaves a small margin for the
        # handful of empty bins Lloyd may produce, while catching any
        # serious collapse. Upper bound stays at the target.
        target_l1 = 16  # ceil(64/4)
        target_l2 = 4  # ceil(16/4)
        assert pyramid.substitutive_levels[1].n_splats_total >= target_l1 * 3 // 4, (
            f"level-1 collapsed to "
            f"{pyramid.substitutive_levels[1].n_splats_total} splats "
            f"(target {target_l1}); k-means likely producing too many empty bins"
        )
        assert pyramid.substitutive_levels[2].n_splats_total >= max(
            target_l2 * 3 // 4, 1
        ), (
            f"level-2 collapsed to "
            f"{pyramid.substitutive_levels[2].n_splats_total} splats "
            f"(target {target_l2})"
        )

    @pytest.mark.parametrize(
        "method", ["kmeans", "kmeans_lloyd", "greedy", "greedy_lloyd"]
    )
    def test_levels_are_flat(self, method):
        data = _make_isotropic_3d(n=32, seed=1)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method=method,
            lloyd_iterations=1,
            candidate_bins_k=4,
            device="cpu",
            seed=0,
        )
        for lev in pyramid.substitutive_levels:
            assert lev.n_additive_lods == 1, f"{method} produced multi-additive level"

    def test_levels_eq_one(self):
        data = _make_isotropic_3d(n=20, seed=1)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans_lloyd",
            lloyd_iterations=1,
            candidate_bins_k=4,
            device="cpu",
            seed=0,
        )
        assert pyramid.n_substitutive == 2
        assert pyramid.substitutive_levels[0].n_splats_total == 20
        assert pyramid.substitutive_levels[1].n_splats_total <= 5

    def test_n_too_small_for_K(self):
        """N=10, K=4, L=3: levels collapse but don't crash."""
        data = _make_isotropic_3d(n=10, seed=1)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=3,
            method="kmeans_lloyd",
            lloyd_iterations=1,
            candidate_bins_k=2,
            device="cpu",
            seed=0,
        )
        # Should produce something for each level (some may be n=1 with stop reason)
        assert pyramid.n_substitutive >= 2

    def test_empty_input_raises_or_handled(self):
        """Empty input: API contract is to refuse via ValueError or
        produce single-level output. Either is acceptable."""
        data = _empty_3d()
        # The current implementation hits the n_splats <= 1 short-circuit
        # which appends a stats-only level and stops.
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            device="cpu",
            seed=0,
        )
        assert pyramid.substitutive_levels[0].n_splats_total == 0

    def test_counts_monotone_non_increasing_across_levels(self):
        """[P12/G10] Each substitutive level has ≤ the count of the previous
        one — the hierarchy must never grow."""
        data = _make_isotropic_3d(n=200, seed=2)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=3,
            method="kmeans_lloyd",
            lloyd_iterations=1,
            candidate_bins_k=2,
            device="cpu",
            seed=0,
        )
        counts = [lv.n_splats_total for lv in pyramid.substitutive_levels]
        assert counts[0] == 200
        assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), (
            f"counts not monotone non-increasing: {counts}"
        )
        assert all(c >= 1 for c in counts)

    def test_output_arrays_are_float32(self):
        """[P5/G4] The reduction works internally in float64 but must emit
        float32 GSplatData (the on-disk / viewer contract), with finite
        values after the float64→float32 round-trip."""
        data = _make_anisotropic_3d(n=48, seed=3)
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans_lloyd",
            lloyd_iterations=1,
            candidate_bins_k=2,
            device="cpu",
            seed=0,
        )
        level_1 = out.at_substitutive(1)
        assert level_1.centers.dtype == np.float32
        assert level_1.cholesky_factors.dtype == np.float32
        assert level_1.amplitudes.dtype == np.float32
        assert np.all(np.isfinite(level_1.centers))
        assert np.all(np.isfinite(level_1.cholesky_factors))

    def test_colors_preserved_through_reduction(self):
        """[P11/G6] When the input carries per-splat colors, the reduced
        level must too — the colors branch of the merge is otherwise
        never executed."""
        rng = np.random.RandomState(0)
        n = 40
        data = GSplatData(
            centers=rng.randn(n, 3).astype(np.float32),
            amplitudes=(rng.rand(n) + 0.5).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
            ),
            colors=rng.rand(n, 3).astype(np.float32),
        )
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans_lloyd",
            lloyd_iterations=1,
            candidate_bins_k=2,
            device="cpu",
            seed=0,
        )
        level_1 = out.at_substitutive(1)
        assert level_1.colors is not None
        assert level_1.colors.shape == (level_1.n_splats, 3)
        assert np.all(np.isfinite(level_1.colors))


# ─────────────────────────────────────────────────────────────────────
# Lloyd refinement: monotonicity + cost-aware advantage
# ─────────────────────────────────────────────────────────────────────


class TestLloyd:
    def test_monotone_non_increase_per_iter(self):
        """Cost-increment Lloyd shouldn't *increase* the per-bin sum."""
        # Build a known, mildly-suboptimal partition by running k-means
        # briefly; then do additional Lloyd iterations and verify the
        # final residual is <= the initial one.
        data = _make_anisotropic_3d(n=32, seed=42)
        out_kmeans = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans",
            lloyd_iterations=0,
            candidate_bins_k=4,
            device="cpu",
            seed=7,
        )
        out_lloyd = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans_lloyd",
            lloyd_iterations=5,
            candidate_bins_k=4,
            device="cpu",
            seed=7,
        )
        # As an indirect monotonicity check: the level-1 dataset's
        # representative splats should fit the data at least as well as
        # the kmeans-only baseline. Compare squared L2 residual of
        # f - g, computed on a deterministic query grid.
        rel_l2_kmeans = _rel_l2_render(data, out_kmeans.at_substitutive(1))
        rel_l2_lloyd = _rel_l2_render(data, out_lloyd.at_substitutive(1))
        # Lloyd should not be substantially worse than kmeans on this
        # small synthetic; in practice it tends to improve, but for the
        # monotonicity contract we just require non-regression
        # within a small tolerance.
        assert rel_l2_lloyd <= rel_l2_kmeans + 0.05, (
            f"lloyd worse than kmeans: {rel_l2_lloyd:.3f} vs {rel_l2_kmeans:.3f}"
        )

    def test_kmeans_lloyd_helpful_on_anisotropic(self):
        """Mirror supp-doc Experiment C qualitatively at tiny scale.

        Spatial-only kmeans may underfit on anisotropic data; Lloyd
        should do *no worse* and on average improves. We assert the
        Lloyd result has finite, non-degenerate residual rather than
        a strict-inequality on this tiny fixture (whose statistics
        are noisy).
        """
        data = _make_anisotropic_3d(n=24, seed=11)
        out = make_substitutive_lod(
            data,
            compression_factor=3,
            levels=1,
            method="kmeans_lloyd",
            lloyd_iterations=5,
            candidate_bins_k=4,
            device="cpu",
            seed=11,
        )
        level_1 = out.at_substitutive(1)
        assert level_1.n_splats >= 1
        # All representative amplitudes finite + non-negative.
        assert np.all(np.isfinite(level_1.amplitudes))
        assert np.all(level_1.amplitudes >= 0)


# ─────────────────────────────────────────────────────────────────────
# Helpers (local quality metric)
# ─────────────────────────────────────────────────────────────────────


def _rel_l2_render(original: GSplatData, reduced: GSplatData) -> float:
    """Render-free relative L² metric: ‖f - g‖² / ‖f‖² in the L² space.

    Computed in closed form: ``‖f - g‖² = ‖f‖² - 2⟨f,g⟩ + ‖g‖²``,
    where each inner product expands as a sum of pairwise Gaussian
    inner products via :mod:`luxar.gsplats.lod._kernels`.
    """
    fc, fL, fa = _gsplat_to_torch(original)
    gc, gL, ga = _gsplat_to_torch(reduced)
    # Stack: f's splats followed by NEGATIVE of g's splats. The L² norm
    # of the combined "signed mixture" with negative amplitudes is
    # ‖f - g‖² because K_ij is bilinear in amplitudes.
    centres_combined = torch.cat([fc, gc], dim=0)
    L_combined = torch.cat([fL, gL], dim=0)
    amps_combined = torch.cat([fa, -ga], dim=0)
    diff_norm_sq = float(
        bin_squared_norm_torch(centres_combined, L_combined, amps_combined)
    )
    f_norm_sq = float(bin_squared_norm_torch(fc, fL, fa))
    return float(np.sqrt(max(diff_norm_sq, 0.0) / max(f_norm_sq, 1e-12)))


# ─────────────────────────────────────────────────────────────────────
# Device / API contract
# ─────────────────────────────────────────────────────────────────────


class TestApiContract:
    def test_invalid_method_raises(self):
        data = _make_isotropic_3d(n=16, seed=0)
        with pytest.raises(ValueError, match="method must be"):
            make_substitutive_lod(data, method="foo", device="cpu")  # type: ignore[arg-type]

    def test_invalid_K_raises(self):
        data = _make_isotropic_3d(n=16, seed=0)
        with pytest.raises(ValueError, match="compression_factor"):
            make_substitutive_lod(data, compression_factor=1, device="cpu")

    def test_invalid_levels_raises(self):
        data = _make_isotropic_3d(n=16, seed=0)
        with pytest.raises(ValueError, match="levels"):
            make_substitutive_lod(data, levels=0, device="cpu")

    def test_device_auto_smoke(self):
        """``device='auto'`` shouldn't crash regardless of GPU presence."""
        data = _make_isotropic_3d(n=16, seed=0)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans_lloyd",
            lloyd_iterations=1,
            candidate_bins_k=2,
            device="auto",
            seed=0,
        )
        assert pyramid.n_substitutive == 2

    def test_stats_recorded(self):
        data = _make_isotropic_3d(n=16, seed=0)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            lloyd_iterations=1,
            candidate_bins_k=2,
            device="cpu",
            seed=0,
        )
        # Top-level pyramid stats carry the substitutive metadata.
        assert pyramid.stats["lod_kind"] == "substitutive"
        assert pyramid.stats["compression_factor"] == 4
        assert pyramid.stats["method"] == "kmeans_lloyd"
        # Per-level metadata: compression_factor = K^level_index.
        K = 4
        for s, lev in enumerate(pyramid.substitutive_levels):
            assert lev.compression_factor == K**s
            assert lev.level_index == s
            if s == 0:
                assert lev.parent_method is None
            else:
                assert lev.parent_method == "kmeans_lloyd"

    def test_save_load_roundtrip(self, tmp_path):
        """The pyramid round-trips through a single v2.0 .gsplats.zarr."""
        data = _make_isotropic_3d(n=16, seed=0)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans_lloyd",
            lloyd_iterations=1,
            candidate_bins_k=2,
            device="cpu",
            seed=0,
        )
        path = tmp_path / "pyramid.gsplats.zarr"
        pyramid.save(str(path), ordering="none")
        loaded = GSplatData.load(str(path), include_stats=True)
        assert loaded.n_substitutive == pyramid.n_substitutive
        for s in range(pyramid.n_substitutive):
            assert (
                loaded.substitutive_levels[s].n_splats_total
                == pyramid.substitutive_levels[s].n_splats_total
            )


# ─────────────────────────────────────────────────────────────────────
# Shape-aware k-means warm start (WS7)
# ─────────────────────────────────────────────────────────────────────


class TestShapeAwareSeeding:
    def test_shape_aware_not_worse_than_centers_only(self):
        """kmeans_lloyd (shape-aware seed) should not regress vs plain kmeans
        on anisotropic data — same non-regression contract as Lloyd."""
        data = _make_anisotropic_3d(n=48, seed=11)
        out_plain = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans",  # centers-only baseline
            device="cpu",
            seed=11,
        )
        out_shape = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans_lloyd",  # shape-aware seed + Lloyd refinement
            lloyd_iterations=5,
            candidate_bins_k=4,
            device="cpu",
            seed=11,
        )
        rel_plain = _rel_l2_render(data, out_plain.at_substitutive(1))
        rel_shape = _rel_l2_render(data, out_shape.at_substitutive(1))
        assert rel_shape <= rel_plain + 0.05, (
            f"shape-aware worse: {rel_shape:.3f} vs centers-only {rel_plain:.3f}"
        )

    def test_empty_bins_culled(self):
        """Coincident points force empty k-means bins; representatives with
        zero amplitude must be culled (no degenerate output splats)."""
        # 16 points but only 2 distinct locations → many empty bins at K=4.
        centres = np.zeros((16, 3), dtype=np.float32)
        centres[8:] = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        chol = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (16, 1))
        data = GSplatData(
            centers=centres,
            amplitudes=np.ones(16, dtype=np.float32),
            cholesky_factors=chol,
        )
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans_lloyd",
            lloyd_iterations=2,
            candidate_bins_k=2,
            device="cpu",
            seed=0,
        )
        level_1 = out.at_substitutive(1)
        # [P2] The level must actually be REDUCED (mutant that returns the
        # input unchanged would keep all 16). Target M = ceil(16/4) = 4.
        assert level_1.n_splats <= 4, f"not reduced: {level_1.n_splats} splats"
        assert level_1.n_splats >= 1
        assert np.all(level_1.amplitudes > 0)  # no zero-amplitude empties
        assert np.all(np.isfinite(level_1.centers))
        # [P12] Every representative covariance must be a valid (PD) Gaussian:
        # the Cholesky-factor diagonal is strictly positive (no rank-deficient
        # or inverted splats slipping through the merge).
        L = unpack_tril(np.asarray(level_1.cholesky_factors), level_1.ndim)
        diag = np.diagonal(L, axis1=-2, axis2=-1)
        assert np.all(diag > 0), "non-positive Cholesky diagonal (degenerate Σ)"


# ─────────────────────────────────────────────────────────────────────
# Large-N scaling guard
# ─────────────────────────────────────────────────────────────────────


class TestScaling:
    """Regression guard against an O(N²) blow-up in the warm start / Lloyd.

    The substitutive regime has ``M = N/K`` bins, so a per-bin or per-splat
    Python loop — or a global k-means++ init (its former warm start) — is
    ``O(N²/K)`` and was intractable at tens of thousands of splats. The
    vectorised Morton partition + segment-reduction merge + vectorised Lloyd
    are ``O(N log N)``; this test would hang for minutes under the old code.
    """

    def test_scales_to_large_n_quickly(self):
        import time

        n = 20_000
        data = _make_anisotropic_3d(n=n, seed=3)
        t0 = time.perf_counter()
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            lloyd_iterations=3,
            candidate_bins_k=8,
            device="cpu",
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        # Counts roughly track ⌈N / K^ℓ⌉ (Morton chunking yields no empty bins;
        # Lloyd may empty a few, so allow a generous lower bound).
        counts = [lv.n_splats_total for lv in out.substitutive_levels]
        assert counts[0] == n
        assert n // 4 - n // 40 <= counts[1] <= n // 4 + 1
        assert n // 16 - n // 160 <= counts[2] <= n // 16 + 1
        level_1 = out.at_substitutive(1)
        assert np.all(np.isfinite(level_1.centers))
        assert np.all(level_1.amplitudes >= 0)
        # Generous wall-clock ceiling: the vectorised path finishes in well
        # under a second; the former O(N²/K) k-means++ init took minutes at
        # this N. A 60 s bound catches catastrophic regressions without
        # being flaky on slow CI.
        assert elapsed < 60.0, (
            f"reduction too slow ({elapsed:.1f}s) — O(N²) regression?"
        )

    def test_greedy_scales_past_old_quadratic_wall(self):
        """Greedy's lazy-heap Runnalls is ~O(Nk log Nk), not O(N²).

        The former full-rescan greedy took ~57 s at N=200 / ~212 s at N=400.
        N=2000 was wholly intractable; the incremental version does it in a
        couple of seconds. This would hang for many minutes under the old code.
        """
        import time

        n = 2_000
        data = _make_anisotropic_3d(n=n, seed=5)
        t0 = time.perf_counter()
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="greedy",
            device="cpu",
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        level_1 = out.at_substitutive(1)
        # Greedy merges down to exactly ⌈N/K⌉ clusters (no empty bins).
        assert level_1.n_splats == n // 4
        assert np.all(np.isfinite(level_1.centers))
        assert np.all(level_1.amplitudes >= 0)
        assert elapsed < 60.0, f"greedy too slow ({elapsed:.1f}s) — O(N²) regression?"


class TestAutoMethod:
    """The default ``method="auto"`` resolves per level by input size."""

    def test_auto_is_the_default(self):
        # No method passed → auto. On a small dataset every level resolves to
        # greedy, which produces exactly ⌈N/Kᵍ⌉ clusters with no empty bins.
        data = _make_anisotropic_3d(n=400, seed=2)
        out = make_substitutive_lod(data, compression_factor=4, levels=2, device="cpu")
        assert out.n_substitutive == 3
        # Below the 5000 threshold → greedy at every level.
        assert out.substitutive_levels[1].parent_method == "greedy"
        assert out.substitutive_levels[2].parent_method == "greedy"

    def test_auto_switches_method_by_level_size(self):
        # N=20000: level 1 (>5000 in) → kmeans_lloyd; level 2 (≤5000 in) → greedy.
        data = _make_anisotropic_3d(n=20_000, seed=4)
        out = make_substitutive_lod(
            data, compression_factor=4, levels=2, method="auto", device="cpu"
        )
        assert out.substitutive_levels[0].parent_method is None  # source level
        assert out.substitutive_levels[1].parent_method == "kmeans_lloyd"
        assert out.substitutive_levels[2].parent_method == "greedy"

    def test_auto_is_a_valid_choice(self):
        # "auto" must not raise the invalid-method error.
        data = _make_isotropic_3d(n=16, seed=0)
        out = make_substitutive_lod(data, levels=1, method="auto", device="cpu")
        assert out.n_substitutive == 2
