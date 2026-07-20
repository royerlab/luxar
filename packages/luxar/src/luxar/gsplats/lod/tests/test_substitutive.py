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

    def test_rgba_alpha_preserved_and_bounded(self):
        """RGBA colors survive a substitutive reduction: the merged level
        keeps 4 channels and every merged alpha stays a valid opacity in
        [0, 1] (the w-space aggregation maps back through 1 − e^(−w))."""
        rng = np.random.RandomState(1)
        n = 48
        colors = rng.rand(n, 4).astype(np.float32)  # RGB + opacity, all in [0,1]
        data = GSplatData(
            centers=rng.randn(n, 3).astype(np.float32),
            amplitudes=(rng.rand(n) + 0.5).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
            ),
            colors=colors,
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
        assert level_1.colors.shape == (level_1.n_splats, 4)
        alpha = level_1.colors[:, 3]
        assert np.all(np.isfinite(alpha))
        assert np.all(alpha >= 0.0) and np.all(alpha <= 1.0)

    def test_uniform_alpha_is_a_fixed_point_of_the_merge(self):
        """When every splat shares one opacity a₀, w = −ln(1−a₀) is constant,
        so the mass-weighted w-mean is w and the merged alpha maps back to
        exactly a₀ — no drift from the optical-depth round-trip."""
        rng = np.random.RandomState(2)
        n = 40
        a0 = 0.6
        colors = np.empty((n, 4), dtype=np.float32)
        colors[:, :3] = rng.rand(n, 3).astype(np.float32)
        colors[:, 3] = a0
        data = GSplatData(
            centers=rng.randn(n, 3).astype(np.float32),
            amplitudes=(rng.rand(n) + 0.5).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
            ),
            colors=colors,
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
        merged_alpha = out.at_substitutive(1).colors[:, 3]
        assert np.allclose(merged_alpha, a0, atol=1e-4)


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

    def test_quality_stamps_opt_in(self):
        """quality_stamps=True stamps Q + reference_energy on every level.

        The finest level is the reference (Q == 1.0); coarser levels score
        strictly below it; reference_energy is the FINEST content's total
        self-energy, identical across the group (self-energy is quadratic in
        amplitude, so per-level totals differ — the shared value is what
        keeps partition-of-lod weighting level-independent).
        """
        from luxar.gsplats.lod.quality import total_self_energy

        data = _make_anisotropic_3d(n=512, seed=7)
        pyramid = make_substitutive_lod(
            data,
            compression_factor=8,
            levels=2,
            method="kmeans_lloyd",
            lloyd_iterations=2,
            candidate_bins_k=4,
            device="cpu",
            seed=0,
            quality_stamps=True,
        )
        levels = pyramid.substitutive_levels
        assert levels[0].stats["quality"] == 1.0
        w = total_self_energy(data.flattened())
        for lv in levels:
            assert lv.stats["reference_energy"] == pytest.approx(w, rel=1e-6)
        qualities = [lv.stats["quality"] for lv in levels]
        assert all(0.0 <= q <= 1.0 for q in qualities)
        # Quality is non-increasing toward coarser levels, and the coarsest
        # (8 splats for 512) is measurably below the reference. A near-perfect
        # intermediate merge may saturate at 1.0 within estimator noise — the
        # measured-and-clamped semantics, not a bug.
        assert qualities[0] >= qualities[1] >= qualities[2]
        assert qualities[2] < 1.0

    def test_quality_stamps_off_by_default(self):
        data = _make_isotropic_3d(n=32, seed=1)
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
        for lv in pyramid.substitutive_levels:
            assert "quality" not in lv.stats
            assert "reference_energy" not in lv.stats

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
        """The pyramid round-trips through a single .gsplats.zarr node tree."""
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

    def test_save_load_preserves_pipeline_stats(self, tmp_path):
        """Reduction/topology stats survive the disk round-trip via the
        ``pipeline/`` group — historically ALL of them (lod_kind, method,
        compression_factor, coverage_inflation, refine, ...) were silently
        dropped by the fitting-keys whitelist, and the per-level
        ``refine_stats`` dict was dropped by the scalar-only ``level_stats``
        filter. Asserts absolute values, not just key presence."""
        data = _make_isotropic_3d(n=64, seed=0)
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans",
            device="cpu",
            refine="l2",
            refine_iters=4,
            seed=0,
        )
        path = tmp_path / "pipe.gsplats.zarr"
        out.save(str(path), ordering="none")
        back = GSplatData.load(str(path), include_stats=True)
        assert back.stats["lod_kind"] == "substitutive"
        assert back.stats["method"] == "kmeans"
        assert back.stats["compression_factor"] == 4
        assert back.stats["coverage_inflation"] == 3.0
        assert back.stats["refine"] == "l2"
        assert back.stats["refine_iters"] == 4
        assert back.stats["n_substitutive_levels"] == 2
        # Per-level refine_stats dict (nested) round-trips through level_stats.
        rs = back.substitutive_levels[1].stats.get("refine_stats")
        assert isinstance(rs, dict)
        assert "improvement_frac" in rs and "mass_vs_fine" in rs
        assert (
            rs["iters_run"]
            == out.substitutive_levels[1].stats["refine_stats"]["iters_run"]
        )
        # coarsen_dims=None must round-trip AS None (not "None"/dropped).
        assert back.stats["coarsen_dims"] is None

    def test_plain_fit_writes_no_pipeline_group(self, tmp_path):
        """A plain fit (no reduction/topology stats, just fit-runtime scratch
        like ``movie_frames=None``) must NOT emit a ``pipeline/`` group — the
        blacklist excludes fit-scratch keys, so the round-trip is unchanged
        from before the pipeline/ feature existed. Regression for the workflow
        finding that ``movie_frames`` leaked a spurious group into every fit."""
        import zarr

        from luxar.gsplats.io.save_gsplats import split_fitting_info

        # Fit-like stats: only whitelisted fitting keys + the scratch key.
        _, _, _, pipe = split_fitting_info(
            {"fitter_name": "luxar", "n_splats": 5, "movie_frames": None}
        )
        assert pipe is None, f"plain fit produced pipeline_info={pipe}"

        data = _make_isotropic_3d(n=32, seed=0)
        flat = data  # a bare leaf, stats carry a scratch key
        flat = GSplatData(
            centers=np.asarray(data.centers),
            amplitudes=np.asarray(data.amplitudes),
            cholesky_factors=np.asarray(data.cholesky_factors),
            stats={"movie_frames": None, "fitter_name": "luxar"},
        )
        path = tmp_path / "plain.gsplats.zarr"
        flat.save(str(path), ordering="none")
        root = zarr.open_group(str(path), mode="r")
        assert "pipeline" not in root, "plain fit wrote a spurious pipeline/ group"

    def test_json_safe_value_hardening(self):
        """The shared sanitizer: numpy floats coerced to Python float (np.float64
        is a float subclass, so it must be caught before the scalar branch), and
        non-finite floats rejected (NaN/±Inf are invalid JSON / rejected by the
        viewer's JSON.parse)."""
        from luxar.io._compiler.gsplat_tree import json_safe_value as J

        ok, c = J(np.float64(1.5))
        assert ok and type(c) is float and c == 1.5
        ok, c = J(np.int32(7))
        assert ok and type(c) is int and c == 7
        assert J(True) == (True, True)  # bool stays bool, not 1
        assert J(np.bool_(True)) == (True, True)
        for bad in (float("nan"), float("inf"), -float("inf"), np.float64("nan")):
            assert J(bad) == (False, None)
        # A nested dict drops only the bad entry, keeps the rest.
        ok, c = J({"good": 1.0, "bad": float("inf"), "s": "x"})
        assert ok and c == {"good": 1.0, "s": "x"}
        # Non-JSON values excluded, not crashing.
        assert J(np.array([1, 2])) == (False, None)


# ─────────────────────────────────────────────────────────────────────
# Mass conservation + barrier-width numerics (LOD brightness-pop fixes)
# ─────────────────────────────────────────────────────────────────────


def _timelapse_4d(per: int = 2000, n_t: int = 3, sigma_t: float = 3.57e-9):
    """Real-shaped 4D fixture: 3 spatial dims (voxel scale) + a near-delta
    time axis — the geometry that exposed both brightness-pop bugs on the
    h2afva datasets (a* mass drift per group; absolute ridge inflating the
    tiny barrier width x~300,000)."""
    rng = np.random.default_rng(0)
    parts = [
        np.column_stack([rng.random((per, 3)) * 2000, np.full(per, float(t))])
        for t in range(n_t)
    ]
    pts = np.vstack(parts).astype(np.float32)
    n = len(pts)
    L = np.zeros((n, 4, 4), np.float32)
    for i in range(3):
        L[:, i, i] = 2.3
    L[:, 3, 3] = sigma_t
    return GSplatData(
        centers=pts,
        amplitudes=np.ones(n, np.float32),
        cholesky_factors=pack_tril(L),
    )


def _per_slice_spatial_mass(lev, t: int) -> float:
    a = np.asarray(lev.amplitudes, np.float64)
    L = unpack_tril(np.asarray(lev.cholesky_factors), lev.ndim)
    diag = np.abs(np.diagonal(L, axis1=-2, axis2=-1))
    m = np.abs(np.asarray(lev.centers)[:, 3] - t) < 0.4
    return float((a[m] * diag[m, 0] * diag[m, 1] * diag[m, 2]).sum())


class TestMassConservation:
    def test_per_time_slice_mass_constant_across_levels(self):
        """The brightness-pop fix: every time-slice's spatial mass (the DC an
        additive render integrates at that slice) is IDENTICAL at every LOD
        level. Pre-fix the coarsest level drifted up to ~27 % on this fixture."""
        data = _timelapse_4d()
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans",
            device="cpu",
            coarsen_dims=[0, 1, 2],
        )
        base = [
            _per_slice_spatial_mass(out.at_substitutive(0).flattened(), t)
            for t in range(3)
        ]
        for s in range(1, out.n_substitutive):
            lev = out.at_substitutive(s).flattened()
            for t in range(3):
                np.testing.assert_allclose(
                    _per_slice_spatial_mass(lev, t),
                    base[t],
                    rtol=1e-3,
                    err_msg=f"level {s} slice t={t} mass drifted",
                )

    def test_conserve_mass_off_restores_raw_amplitudes(self):
        """conserve_mass=False keeps the raw per-bin a* (which drifts) — the
        discriminating contrast proving the flag is wired and the default is
        doing real work."""
        data = _timelapse_4d()
        kw = dict(
            compression_factor=4,
            levels=2,
            method="kmeans",
            device="cpu",
            coarsen_dims=[0, 1, 2],
        )
        on = make_substitutive_lod(data, conserve_mass=True, **kw)
        off = make_substitutive_lod(data, conserve_mass=False, **kw)
        m_on = _per_slice_spatial_mass(on.at_substitutive(2).flattened(), 0)
        m_off = _per_slice_spatial_mass(off.at_substitutive(2).flattened(), 0)
        base = _per_slice_spatial_mass(on.at_substitutive(0).flattened(), 0)
        assert abs(m_on / base - 1) < 1e-3
        assert abs(m_off / base - 1) > 0.02  # raw a* really does drift
        assert on.stats["conserve_mass"] is True
        assert off.stats["conserve_mass"] is False

    def test_full_mass_conserved_without_barriers(self):
        """Ungrouped (all dims coarsened): the FULL nD mass is conserved."""
        data = _make_isotropic_3d(n=256, seed=1)
        out = make_substitutive_lod(
            data, compression_factor=4, levels=2, method="kmeans", device="cpu"
        )

        def full_mass(lev):
            a = np.asarray(lev.amplitudes, np.float64)
            L = unpack_tril(np.asarray(lev.cholesky_factors), lev.ndim)
            det = np.abs(np.prod(np.diagonal(L, axis1=-2, axis2=-1), axis=-1))
            return float((a * det).sum())

        base = full_mass(out.at_substitutive(0).flattened())
        for s in range(1, out.n_substitutive):
            np.testing.assert_allclose(
                full_mass(out.at_substitutive(s).flattened()), base, rtol=1e-3
            )

    def test_refine_mass_manifold_safe_with_tiny_barrier_sigma(self):
        """refine="l2" pins each level's FULL-det mass to its fine input's.
        With the old absolute ridge the merged sigma_t inflated x~300,000, so
        the pinning would have collapsed amplitudes by the inverse factor
        (black output). With the proportional ridge + frozen barrier dims the
        pinning is consistent: per-slice brightness stays ~1."""
        data = _timelapse_4d(per=800, n_t=2)
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=1,
            method="kmeans",
            device="cpu",
            coarsen_dims=[0, 1, 2],
            refine="l2",
            refine_iters=6,
            seed=0,
        )
        base = _per_slice_spatial_mass(out.at_substitutive(0).flattened(), 0)
        got = _per_slice_spatial_mass(out.at_substitutive(1).flattened(), 0)
        assert 0.9 < got / base < 1.1, f"slice mass ratio {got / base:.3g}"

    def test_tiny_barrier_sigma_survives_merge_ridge(self):
        """The Cholesky ridge is proportional per-dim, so a near-delta barrier
        width (sigma_t ~ 1e-9) passes through the merge unchanged. Pre-fix the
        absolute 1e-6*I ridge inflated it to sqrt(1e-6)=1e-3 (x~300,000),
        which poisoned every downstream mass/DC accounting."""
        data = _timelapse_4d(per=1200, n_t=2)
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans",
            device="cpu",
            coarsen_dims=[0, 1, 2],
        )
        for s in range(out.n_substitutive):
            lev = out.at_substitutive(s).flattened()
            L = unpack_tril(np.asarray(lev.cholesky_factors), 4)
            st = np.abs(L[:, 3, 3])
            assert st.max() < 1e-8, (
                f"level {s}: barrier sigma_t inflated to {st.max():.3g}"
            )

    def test_conserve_mass_guard_skips_degenerate_rescale(self, monkeypatch):
        """A tiny-but-positive ``mass_out`` (e.g. most representatives'
        coarsened-dim submatrices numerically degenerate → ~zero determinant)
        passed the old ``mass_out > 0.0`` gate and produced an unbounded
        amplitude blow-up under the conserve_mass=True default. The rescale
        must be skipped when the factor leaves the [0.1, 10] band — the output
        then equals the conserve_mass=False result exactly."""
        import luxar.gsplats.lod.substitutive as sub_mod

        n = 256
        data = _make_isotropic_3d(n=n, seed=0)
        kw = dict(compression_factor=4, levels=1, method="kmeans", device="cpu")
        baseline = make_substitutive_lod(data, conserve_mass=False, **kw)

        real_subset_mass = sub_mod._subset_mass

        def degenerate(L, amps, dims, chunk=2_000_000):
            m = real_subset_mass(L, amps, dims, chunk)
            # The coarse (merged, < n rows) call reports a numerically
            # degenerate tiny-but-positive mass; the fine call is untouched.
            return m * 1e-9 if L.shape[0] < n else m

        monkeypatch.setattr(sub_mod, "_subset_mass", degenerate)
        guarded = make_substitutive_lod(data, conserve_mass=True, **kw)
        # Guard held: no 1e9x white-out; amplitudes identical to the raw path.
        np.testing.assert_allclose(
            np.asarray(guarded.at_substitutive(1).amplitudes),
            np.asarray(baseline.at_substitutive(1).amplitudes),
            rtol=1e-6,
        )


def test_merge_refine_stats_zero_seed_energy():
    """An exactly-0.0 summed seed objective must still yield an
    ``improvement_frac`` when ``trusted_E_best`` improved (the old truthiness
    gate silently dropped the key), and the division must stay guarded."""
    from luxar.gsplats.lod.substitutive import _merge_refine_stats

    sink: dict = {}
    _merge_refine_stats(
        sink,
        {
            "trusted_E_seed": 0.0,
            "trusted_E_best": -0.5,
            "iters_run": 3,
            "rebuilds": 1,
            "wall_s": 0.1,
        },
    )
    # Zero seed: normalized by |best| -> a finite, meaningful 100 %.
    assert sink["improvement_frac"] == pytest.approx(1.0)

    # Both exactly zero: defined and 0.0, no ZeroDivisionError.
    sink2: dict = {}
    _merge_refine_stats(sink2, {"trusted_E_seed": 0.0, "trusted_E_best": 0.0})
    assert sink2["improvement_frac"] == 0.0

    # Normal (nonzero-seed) semantics unchanged: (seed - best) / |seed|.
    sink3: dict = {}
    _merge_refine_stats(sink3, {"trusted_E_seed": -2.0, "trusted_E_best": -2.5})
    assert sink3["improvement_frac"] == pytest.approx(0.25)


# ─────────────────────────────────────────────────────────────────────
# Coverage inflation (anti-grid inter-spread widening)
# ─────────────────────────────────────────────────────────────────────


def _mixture_at(points: np.ndarray, data: GSplatData) -> np.ndarray:
    """Evaluate the Gaussian mixture at query points (dense, test-sized only)."""
    centres = np.asarray(data.centers, dtype=np.float64)
    L = unpack_tril(np.asarray(data.cholesky_factors), data.ndim).astype(np.float64)
    Sigma = L @ L.transpose(0, 2, 1)
    Sinv = np.linalg.inv(Sigma)
    amps = np.asarray(data.amplitudes, dtype=np.float64)
    diff = points[:, None, :] - centres[None, :, :]  # (P, N, D)
    quad = np.einsum("pnd,nde,pne->pn", diff, Sinv, diff)
    return np.asarray((amps[None, :] * np.exp(-0.5 * quad)).sum(axis=1))


class TestCoverageInflation:
    """The β·inter widening that suppresses the coarse-level grid ripple."""

    _KW = dict(
        compression_factor=4,
        levels=1,
        method="kmeans",  # warm start only → deterministic identical partitions
        lloyd_iterations=0,
        candidate_bins_k=2,
        device="cpu",
        seed=0,
    )

    def test_widens_covariance_and_preserves_mass(self):
        """Same partition, inflated output: identical centers, wider Σ, and
        per-splat mass a·|Σ|^{1/2} (the X-ray integral) exactly preserved."""
        data = _make_isotropic_3d(n=64, seed=3)
        base = make_substitutive_lod(
            data, coverage_inflation=1.0, **self._KW
        ).at_substitutive(1)
        infl = make_substitutive_lod(
            data, coverage_inflation=3.0, **self._KW
        ).at_substitutive(1)
        np.testing.assert_allclose(
            np.asarray(infl.centers), np.asarray(base.centers), atol=1e-5
        )
        Lb = unpack_tril(np.asarray(base.cholesky_factors), 3)
        Li = unpack_tril(np.asarray(infl.cholesky_factors), 3)
        det_b = np.abs(np.prod(np.diagonal(Lb, axis1=-2, axis2=-1), axis=-1))
        det_i = np.abs(np.prod(np.diagonal(Li, axis1=-2, axis2=-1), axis=-1))
        assert np.all(det_i >= det_b * (1.0 - 1e-5))  # never narrower
        # Genuinely widened. (On this fixture the unit-σ intra term dominates
        # the moment match, so the det ratio is well below the tiny-splat
        # asymptote of 3^{D/2}; ~1.45 measured.)
        assert np.median(det_i / det_b) > 1.2
        mass_b = np.asarray(base.amplitudes) * det_b
        mass_i = np.asarray(infl.amplitudes) * det_i
        np.testing.assert_allclose(mass_i, mass_b, rtol=1e-3)

    def test_single_member_bins_untouched(self):
        """inter = 0 for a one-splat bin → inflation is a no-op there."""
        import torch as _torch

        from luxar.gsplats.lod._substitutive.kmeans_lloyd import (
            _build_representatives_vectorized,
        )

        rng = np.random.RandomState(0)
        n = 8
        centres = _torch.tensor(rng.randn(n, 3), dtype=_torch.float64)
        L = _torch.eye(3, dtype=_torch.float64).expand(n, 3, 3).contiguous()
        amps = _torch.ones(n, dtype=_torch.float64)
        assign = _torch.arange(n, dtype=_torch.int64)  # every splat its own bin
        out1 = _build_representatives_vectorized(
            centres, L, amps, None, assign, M=n, coverage_inflation=1.0
        )
        out3 = _build_representatives_vectorized(
            centres, L, amps, None, assign, M=n, coverage_inflation=3.0
        )
        for a, b in zip(out1[:3], out3[:3]):
            _torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_ripple_suppressed_on_uniform_lattice(self):
        """The regression the feature exists for: coarsening a uniform lattice
        of tiny splats must not render as a deeply rippled (grid) field. The
        inflated level's interior intensity variation is a fraction of the
        un-inflated one's."""
        g = np.arange(12, dtype=np.float32)
        zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
        centres = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])
        n = len(centres)  # 1728
        sig = 0.5
        chol = np.tile(np.array([sig, 0, sig, 0, 0, sig], dtype=np.float32), (n, 1))
        data = GSplatData(
            centers=centres,
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        )
        kw = dict(self._KW, compression_factor=8)
        base = make_substitutive_lod(
            data, coverage_inflation=1.0, **kw
        ).at_substitutive(1)
        infl = make_substitutive_lod(
            data, coverage_inflation=3.0, **kw
        ).at_substitutive(1)
        # Interior query line (away from the lattice boundary falloff).
        t = np.linspace(3.0, 9.0, 121)
        pts = np.column_stack([t, np.full_like(t, 5.7), np.full_like(t, 5.3)])
        f_base = _mixture_at(pts, base)
        f_infl = _mixture_at(pts, infl)
        cv_base = f_base.std() / f_base.mean()
        cv_infl = f_infl.std() / f_infl.mean()
        assert cv_infl < 0.6 * cv_base, (
            f"inflation did not suppress the lattice ripple: "
            f"cv_infl={cv_infl:.3f} vs cv_base={cv_base:.3f}"
        )

    def test_invalid_inflation_raises(self):
        data = _make_isotropic_3d(n=16, seed=0)
        with pytest.raises(ValueError, match="coverage_inflation"):
            make_substitutive_lod(data, coverage_inflation=0.5, device="cpu")

    def test_inflation_recorded_in_stats(self):
        data = _make_isotropic_3d(n=16, seed=0)
        out = make_substitutive_lod(data, levels=1, device="cpu")
        assert out.stats["coverage_inflation"] == 3.0  # default ON


# ─────────────────────────────────────────────────────────────────────
# Per-bin amplitude mode (amplitude="l2" | "mass")
# ─────────────────────────────────────────────────────────────────────


class TestAmplitudeMode:
    """amplitude="mass": every bin's merged splat carries exactly its members'
    summed a·|det L| mass — the per-channel colored-light invariant the lifted
    points/lines LOD path relies on for hue coherence across levels."""

    _KW = dict(
        compression_factor=4,
        levels=1,
        method="kmeans",  # warm start only → deterministic identical partitions
        lloyd_iterations=0,
        candidate_bins_k=2,
        device="cpu",
        seed=0,
        conserve_mass=False,  # isolate the PER-BIN rule from the global net
    )

    @staticmethod
    def _colored_mass(lev) -> np.ndarray:
        L = unpack_tril(np.asarray(lev.cholesky_factors, np.float64), 3)
        det = np.abs(np.prod(np.diagonal(L, axis1=-2, axis2=-1), axis=-1))
        a = np.asarray(lev.amplitudes, np.float64)
        c = np.asarray(lev.colors, np.float64)
        return (a * det) @ c  # (3,) per-channel colored mass

    def _data_with_colors(self, n=256, seed=5):
        base = _make_anisotropic_3d(n, seed=seed)
        rng = np.random.RandomState(seed + 1)
        return GSplatData(
            centers=np.asarray(base.centers),
            amplitudes=np.asarray(base.amplitudes),
            cholesky_factors=np.asarray(base.cholesky_factors),
            colors=rng.rand(n, 3).astype(np.float32) * 0.9 + 0.05,
        )

    def test_mass_mode_conserves_colored_mass_exactly(self):
        data = self._data_with_colors()
        out = make_substitutive_lod(data, amplitude="mass", **self._KW)
        fine = self._colored_mass(out.at_substitutive(0).flattened())
        coarse = self._colored_mass(out.at_substitutive(1).flattened())
        np.testing.assert_allclose(coarse, fine, rtol=1e-6)

    def test_l2_default_differs_from_mass(self):
        # Guard that the knob is live: with the global conserve_mass net off,
        # the L2-optimal amplitudes drift (3-17% measured, content-dependent).
        data = self._data_with_colors()
        out_l2 = make_substitutive_lod(data, **self._KW)  # default amplitude="l2"
        fine = self._colored_mass(out_l2.at_substitutive(0).flattened())
        coarse = self._colored_mass(out_l2.at_substitutive(1).flattened())
        assert float(np.abs(coarse / fine - 1.0).max()) > 1e-3

    def test_invalid_amplitude_rejected(self):
        with pytest.raises(ValueError, match="amplitude"):
            make_substitutive_lod(
                self._data_with_colors(), amplitude="peak", **self._KW
            )


# ─────────────────────────────────────────────────────────────────────
# L2 refinement (refine="l2")
# ─────────────────────────────────────────────────────────────────────


class TestRefine:
    """Integration of the L2 refit (engine unit tests live in test_refine.py)."""

    _KW = dict(compression_factor=8, levels=1, method="kmeans", device="cpu")

    def _blob_data(self, seed: int = 1) -> GSplatData:
        """Isolated clusters — the regime where the refit shines."""
        rng = np.random.default_rng(seed)
        centers = rng.random((6, 3)) * 0.8 + 0.1
        pts = np.concatenate(
            [c + rng.normal(0, 0.02, (80, 3)) for c in centers], axis=0
        ).astype(np.float32)
        n = len(pts)
        chol = np.tile(
            np.array([0.008, 0, 0.008, 0, 0, 0.008], dtype=np.float32), (n, 1)
        )
        return GSplatData(
            centers=pts,
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        )

    def test_refine_none_matches_omitted_kwarg(self):
        data = _make_isotropic_3d(n=64, seed=3)
        a = make_substitutive_lod(data, refine="none", **self._KW)
        b = make_substitutive_lod(data, **self._KW)
        for s in range(a.n_substitutive):
            la, lb = a.at_substitutive(s), b.at_substitutive(s)
            assert np.array_equal(np.asarray(la.centers), np.asarray(lb.centers))
            assert np.array_equal(
                np.asarray(la.cholesky_factors), np.asarray(lb.cholesky_factors)
            )
            assert np.array_equal(np.asarray(la.amplitudes), np.asarray(lb.amplitudes))
        assert a.stats["refine"] == "none"

    def test_refine_l2_improves_level_fidelity(self):
        data = self._blob_data()
        plain = make_substitutive_lod(data, refine="none", **self._KW)
        refined = make_substitutive_lod(
            data, refine="l2", refine_iters=60, seed=7, **self._KW
        )
        rel_plain = _rel_l2_render(data, plain.at_substitutive(1))
        rel_refined = _rel_l2_render(data, refined.at_substitutive(1))
        assert rel_refined < 0.9 * rel_plain, (
            f"refine=l2 did not improve: {rel_refined:.4f} vs {rel_plain:.4f}"
        )

    def test_refine_chains_levels_and_counts_unchanged(self):
        data = self._blob_data()
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans",
            refine="l2",
            refine_iters=15,
            seed=0,
            device="cpu",
        )
        counts = [lv.n_splats_total for lv in out.substitutive_levels]
        assert counts[0] == 480
        assert counts[1] <= 120 and counts[1] >= 90
        assert counts[2] <= 30 and counts[2] >= 20
        # Every refined level records its refine stats block.
        for lv in out.substitutive_levels[1:]:
            assert lv.stats.get("refine") == "l2"
            assert lv.stats["refine_stats"]["iters_run"] >= 0

    def test_refine_with_coarsen_dims_keeps_barrier_pure(self):
        data = _stacked_categorical(n_per=800, n_groups=3)
        kw = dict(
            compression_factor=4, levels=2, seed=0, device="cpu", coarsen_dims=[1, 2, 3]
        )
        merged = make_substitutive_lod(data, refine="none", **kw)
        out = make_substitutive_lod(data, refine="l2", refine_iters=12, **kw)
        for s in range(out.n_substitutive):
            lev = out.at_substitutive(s).flattened()
            c0 = np.asarray(lev.centers)[:, 0]
            assert np.abs(c0 - np.round(c0)).max() < 1e-4
            # The refit FROZE the barrier dim (0): the barrier marginal
            # variance Σ[0,0] must be bit-preserved from the pre-refit merge,
            # not merely "small". Compare the sorted per-splat barrier widths
            # (refit may permute splat order but must not change the multiset
            # of Σ[0,0] values — every group's reps keep their merge width).
            L_ref = unpack_tril(np.asarray(lev.cholesky_factors), lev.ndim)
            L_mrg = unpack_tril(
                np.asarray(merged.at_substitutive(s).flattened().cholesky_factors),
                lev.ndim,
            )
            s00_ref = np.sort(L_ref[:, 0, 0] ** 2)
            s00_mrg = np.sort(L_mrg[:, 0, 0] ** 2)
            np.testing.assert_allclose(s00_ref, s00_mrg, rtol=1e-4, atol=1e-8)

    def test_refine_deterministic_given_seed(self):
        """Same seed → same result up to CPU-threading noise (torch reductions
        are not bitwise run-to-run reproducible; contract is ~1-ulp closeness)."""
        data = self._blob_data()
        kw = dict(self._KW, refine="l2", refine_iters=20)
        a = make_substitutive_lod(data, seed=42, **kw)
        b = make_substitutive_lod(data, seed=42, **kw)
        np.testing.assert_allclose(
            np.asarray(a.at_substitutive(1).centers),
            np.asarray(b.at_substitutive(1).centers),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_refine_recorded_in_stats(self):
        data = self._blob_data()
        out = make_substitutive_lod(
            data, refine="l2", refine_iters=8, seed=0, **self._KW
        )
        assert out.stats["refine"] == "l2"
        assert out.stats["refine_iters"] == 8
        lev1 = out.substitutive_levels[1]
        rs = lev1.stats["refine_stats"]
        assert rs["rebuilds"] >= 2  # seed eval + at least one checkpoint
        assert "improvement_frac" in rs
        assert "_mass_n" not in rs  # aggregation scratch key stripped

    def test_refine_invalid_choice_raises(self):
        data = _make_isotropic_3d(n=16, seed=0)
        with pytest.raises(ValueError, match="refine must be"):
            make_substitutive_lod(data, refine="banana", device="cpu")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="refine_iters"):
            make_substitutive_lod(data, refine="l2", refine_iters=0, device="cpu")


class TestVolumeRefit:
    """Integration of ``refine="volume"`` (engine unit tests live in
    test_volume_refit.py)."""

    @staticmethod
    def _volume_and_fit() -> "tuple[np.ndarray, GSplatData]":
        from luxar.gsplats.fit_gsplats import fit_gaussian_splats

        rng = np.random.default_rng(0)
        grid = np.mgrid[0:20, 0:20, 0:20].astype(np.float32)
        vol = np.zeros((20,) * 3, dtype=np.float32)
        for _ in range(4):
            c = rng.uniform(4, 16, 3)
            s = rng.uniform(1.5, 2.5)
            r2 = sum((grid[d] - c[d]) ** 2 for d in range(3))
            vol += rng.uniform(0.4, 1.0) * np.exp(-r2 / (2 * s * s))
        fine = fit_gaussian_splats(
            vol, seeds=80, n_iters=200, device="cpu", verbose=False
        )
        return vol, fine

    @staticmethod
    def _mse(data: GSplatData, vol: np.ndarray) -> float:
        rendered = data.render_to_volume(shape=vol.shape, device="cpu")
        return float(np.mean((rendered.astype(np.float32) - vol) ** 2))

    def test_refine_volume_improves_level_mse(self):
        vol, fine = self._volume_and_fit()
        kw = dict(compression_factor=4, levels=1, device="cpu")
        plain = make_substitutive_lod(fine, refine="none", **kw)
        refit = make_substitutive_lod(
            fine, refine="volume", refine_iters=60, volume=vol, **kw
        )
        # Identical structure; only the coarse level's fidelity changes.
        assert refit.n_substitutive == plain.n_substitutive
        assert refit.at_substitutive(1).n_splats == plain.at_substitutive(1).n_splats
        mse_plain = self._mse(plain.at_substitutive(1), vol)
        mse_refit = self._mse(refit.at_substitutive(1), vol)
        assert mse_refit < 0.9 * mse_plain, (
            f"refine=volume did not improve: {mse_refit:.3e} vs {mse_plain:.3e}"
        )
        # Stats recorded on the ladder and the level.
        assert refit.stats["refine"] == "volume"
        assert refit.stats["refine_iters"] == 60
        lev = refit.substitutive_levels[1]
        assert lev.stats["refine"] == "volume"
        assert lev.stats["refine_stats"]["improved"] is True

    def test_refine_volume_requires_volume(self):
        data = _make_isotropic_3d(n=16, seed=0)
        with pytest.raises(ValueError, match="requires the `volume`"):
            make_substitutive_lod(data, refine="volume", device="cpu")

    def test_volume_without_refine_volume_raises(self):
        data = _make_isotropic_3d(n=16, seed=0)
        with pytest.raises(ValueError, match="only consumed by refine='volume'"):
            make_substitutive_lod(
                data, volume=np.zeros((4, 4, 4), np.float32), device="cpu"
            )

    def test_refine_volume_rejects_barrier_dims(self):
        data = _make_isotropic_3d(n=16, seed=0)
        with pytest.raises(ValueError, match="barrier dims"):
            make_substitutive_lod(
                data,
                refine="volume",
                volume=np.zeros((4, 4, 4), np.float32),
                coarsen_dims=[1, 2],
                device="cpu",
            )

    def test_refine_volume_round_trips_stats(self, tmp_path):
        vol, fine = self._volume_and_fit()
        refit = make_substitutive_lod(
            fine,
            refine="volume",
            refine_iters=20,
            volume=vol,
            compression_factor=4,
            levels=1,
            device="cpu",
        )
        out = tmp_path / "vr.gsplats.zarr"
        refit.save(out)
        back = GSplatData.load(out, include_stats=True)
        assert back.stats["refine"] == "volume"
        assert back.stats["refine_iters"] == 20
        lev = back.substitutive_levels[1]
        assert lev.stats["refine"] == "volume"
        assert set(lev.stats["refine_stats"]) >= {"mse_seed", "mse_refit", "improved"}

    def test_default_refine_iters_is_300(self):
        """The library default for refine="volume" resolves to VolumeRefitConfig's
        300 (not l2's 120) when refine_iters is omitted."""
        vol, fine = self._volume_and_fit()
        lad = make_substitutive_lod(
            fine,
            refine="volume",
            volume=vol,
            compression_factor=4,
            levels=1,
            device="cpu",
        )
        assert lad.stats["refine_iters"] == 300

    def test_chain_continues_from_unrefined_merge(self, monkeypatch):
        """levels>=2 invariant: the coarsening chain continues from the
        UNREFINED merge, so level L's re-fit seed is the pure merge chain's
        level L — NOT the (re-fit) previous level. Uses a deterministic
        monkeypatched re-fit (scale centers ×0.9) so a mutation feeding the
        re-fit forward (`current = stored`) is caught."""
        import luxar.gsplats.lod.volume_refit as vr

        vol, fine = self._volume_and_fit()

        def _scale_refit(seed, volume, *, config, device=None):
            scaled = GSplatData(
                centers=(np.asarray(seed.centers) * 0.9).astype(np.float32),
                amplitudes=seed.amplitudes,
                cholesky_factors=seed.cholesky_factors,
            )
            return scaled, {"improved": True, "seed_won": False}

        monkeypatch.setattr(vr, "volume_refine_splats", _scale_refit)
        kw = dict(compression_factor=4, levels=2, device="cpu")
        got = make_substitutive_lod(fine, refine="volume", volume=vol, **kw)
        # Pure merge chain (no refit): level 2 built from the unrefined chain.
        merge = make_substitutive_lod(fine, refine="none", **kw)
        # If the chain is correct, level 2 == 0.9 * (pure merge level 2).
        expected = np.asarray(merge.at_substitutive(2).centers) * 0.9
        np.testing.assert_allclose(
            np.asarray(got.at_substitutive(2).centers), expected, rtol=1e-5, atol=1e-5
        )

    def test_conserve_mass_no_cross_level_brightness_pop(self):
        """refine="volume" must not reintroduce the brightness pop conserve_mass
        prevents. The re-fit tracks the volume's true DC, which the finest
        (fine-fit) level under-explains — stored unpinned that is a large pop.
        With conserve_mass the volume ladder's per-level DC ratios track the
        pure-merge ladder's (both pinned to the fine chain), NOT the volume."""

        def dc_ratios(lad):
            dcs = [
                float(
                    np.asarray(
                        lad.at_substitutive(s).render_to_volume(
                            shape=vol.shape, device="cpu"
                        )
                    ).sum()
                )
                for s in range(lad.n_substitutive)
            ]
            return np.array([d / dcs[0] for d in dcs])

        vol, fine = self._volume_and_fit()
        kw = dict(compression_factor=4, levels=2, device="cpu")
        merge_ratios = dc_ratios(make_substitutive_lod(fine, refine="none", **kw))
        vol_ratios = dc_ratios(
            make_substitutive_lod(
                fine, refine="volume", refine_iters=60, volume=vol, **kw
            )
        )
        # The volume ladder brightness-tracks the merge ladder (pinned), rather
        # than popping toward the volume's higher DC (the pre-fix +14% bug).
        np.testing.assert_allclose(vol_ratios, merge_ratios, atol=0.03)

    def test_conserve_mass_false_lets_refit_track_volume_dc(self):
        """--no-conserve-mass opts into the raw volume-accurate DC: the re-fit's
        brightness then tracks the VOLUME (may pop vs the finest) — the escape
        hatch, and proof the pinning in the default path is load-bearing."""
        vol, fine = self._volume_and_fit()
        kw = dict(compression_factor=4, levels=1, device="cpu")
        pinned = make_substitutive_lod(
            fine,
            refine="volume",
            refine_iters=60,
            volume=vol,
            conserve_mass=True,
            **kw,
        )
        raw = make_substitutive_lod(
            fine,
            refine="volume",
            refine_iters=60,
            volume=vol,
            conserve_mass=False,
            **kw,
        )

        def r(lad):
            return float(
                np.asarray(
                    lad.at_substitutive(1).render_to_volume(
                        shape=vol.shape, device="cpu"
                    )
                ).sum()
            )

        finest = float(
            np.asarray(
                pinned.at_substitutive(0).render_to_volume(
                    shape=vol.shape, device="cpu"
                )
            ).sum()
        )
        # Pinned coarse DC hugs the finest; raw coarse DC sits meaningfully higher.
        assert abs(r(pinned) - finest) < abs(r(raw) - finest)


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


# ─────────────────────────────────────────────────────────────────────
# coarsen_dims (barrier-dim grouping)
# ─────────────────────────────────────────────────────────────────────


def _stacked_categorical(n_per: int = 1500, n_groups: int = 3, seed: int = 0):
    """A 4D lifted gsplat set: dim 0 is a categorical barrier (0..G-1), dims
    1-3 are xyz shared across the groups (spatially coincident copies)."""
    from luxar.gsplats.lift import lift_points_to_gsplats

    rng = np.random.default_rng(seed)
    xyz = rng.normal(0, 5, (n_per, 3)).astype(np.float32)
    parts = [
        np.column_stack([np.full(n_per, g, np.float32), xyz]) for g in range(n_groups)
    ]
    pos = np.vstack(parts).astype(np.float32)
    return lift_points_to_gsplats(pos, np.full(len(pos), 0.5, np.float32))


class TestCoarsenDims:
    def test_barrier_invariant_keeps_groups_pure(self):
        # Coarsening over xyz (dims 1-3), grouping by the categorical dim 0:
        # every coarse splat must sit exactly on an integer group value.
        data = _stacked_categorical(n_groups=3)
        out = make_substitutive_lod(
            data, compression_factor=4, levels=3, device="cpu", coarsen_dims=[1, 2, 3]
        )
        for s in range(out.n_substitutive):
            c0 = np.asarray(out.at_substitutive(s).flattened().centers)[:, 0]
            assert np.abs(c0 - np.round(c0)).max() < 1e-4

    def test_baseline_merges_across_barrier(self):
        # Without coarsen_dims the coarse levels DO blend across the barrier
        # (this is the bug coarsen_dims fixes) — assert it actually happens so
        # the invariant test above is meaningful.
        data = _stacked_categorical(n_per=2000, n_groups=4)
        out = make_substitutive_lod(
            data, compression_factor=4, levels=3, method="kmeans_lloyd", device="cpu"
        )
        c0 = np.asarray(
            out.at_substitutive(out.n_substitutive - 1).flattened().centers
        )[:, 0]
        assert np.abs(c0 - np.round(c0)).max() > 0.1

    def test_all_dims_is_noop_equivalent(self):
        # coarsen_dims == every dim normalises to None (the all-dims path).
        data = _stacked_categorical(n_groups=2)
        a = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            device="cpu",
            coarsen_dims=[0, 1, 2, 3],
        )
        b = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            device="cpu",
        )
        for s in range(a.n_substitutive):
            assert a.at_substitutive(s).n_splats == b.at_substitutive(s).n_splats

    def test_coarsest_level_has_at_least_n_groups(self):
        # Each barrier group keeps >= 1 representative, so the coarsest level
        # cannot compress below the number of groups.
        data = _stacked_categorical(n_per=2000, n_groups=4)
        out = make_substitutive_lod(
            data, compression_factor=4, levels=5, device="cpu", coarsen_dims=[1, 2, 3]
        )
        coarsest = out.at_substitutive(out.n_substitutive - 1).n_splats
        assert coarsest >= 4

    def test_invalid_coarsen_dims_raise(self):
        data = _stacked_categorical(n_groups=2)
        with pytest.raises(ValueError):
            make_substitutive_lod(data, levels=1, device="cpu", coarsen_dims=[])
        with pytest.raises(ValueError):
            make_substitutive_lod(data, levels=1, device="cpu", coarsen_dims=[7])

    def test_single_barrier_value_is_noop(self):
        # All splats share barrier value 0 → one group → identical to all-dims.
        data = _stacked_categorical(n_per=1200, n_groups=1)
        a = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            device="cpu",
            coarsen_dims=[1, 2, 3],
        )
        b = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=2,
            method="kmeans_lloyd",
            device="cpu",
        )
        for s in range(a.n_substitutive):
            assert a.at_substitutive(s).n_splats == b.at_substitutive(s).n_splats


class TestAllocateGroupM:
    """Direct unit coverage of the per-group M allocator (its clamp + water-fill
    branches are never reached by the balanced integration fixtures)."""

    import pytest as _pytest

    @_pytest.mark.parametrize(
        "sizes,M_target,expected",
        [
            ([1, 1, 1000], 2, [1, 1, 1]),  # M_target < G -> >=1 each, bumped to G
            ([2, 2], 10, [2, 2]),  # over-target: size-clamp + water-fill
            ([1, 1, 1], 5, [1, 1, 1]),  # all size-1: clamp, water-fill breaks
            ([5, 5, 5], 2, [1, 1, 1]),  # M_target < G, balanced
            ([100, 1, 1], 50, [48, 1, 1]),  # skewed proportional, small clamped
        ],
    )
    def test_branches(self, sizes, M_target, expected):
        from luxar.gsplats.lod.substitutive import _allocate_group_M

        s = np.asarray(sizes)
        alloc = _allocate_group_M(s, M_target)
        g = len(s)
        assert (alloc >= 1).all()
        assert (alloc <= s).all()  # never asks for more reps than splats
        assert int(alloc.sum()) == min(max(M_target, g), int(s.sum()))
        assert alloc.tolist() == expected


def test_continuous_barrier_warns():
    """A barrier dim with ~all-distinct values (continuous) should warn that
    coarsening will be negligible."""
    from luxar.gsplats.lift import lift_points_to_gsplats

    rng = np.random.default_rng(0)
    # dim 0 is continuous (unique per splat) -> grouping by it = N singletons.
    pos = rng.normal(0, 5, (400, 4)).astype(np.float32)
    g = lift_points_to_gsplats(pos, np.full(len(pos), 0.5, np.float32))
    with pytest.warns(RuntimeWarning, match="look continuous"):
        make_substitutive_lod(g, levels=1, device="cpu", coarsen_dims=[1, 2, 3])


def test_make_lod_pyramid_respects_coarsen_dims():
    """coarsen_dims threads through the pyramid (substitutive × additive) builder."""
    from luxar.gsplats.lift import lift_points_to_gsplats
    from luxar.gsplats.lod.pyramid import make_lod_pyramid

    rng = np.random.default_rng(0)
    xyz = rng.normal(0, 5, (700, 3)).astype(np.float32)
    pos = np.vstack(
        [np.column_stack([np.full(700, g, np.float32), xyz]) for g in range(3)]
    ).astype(np.float32)
    data = lift_points_to_gsplats(pos, np.full(len(pos), 0.5, np.float32))
    out = make_lod_pyramid(
        data,
        compression_factor=4,
        levels=2,
        n_additive_lods=1,
        device="cpu",
        coarsen_dims=[1, 2, 3],
    )
    for s in range(out.n_substitutive):
        c0 = np.asarray(out.at_substitutive(s).flattened().centers)[:, 0]
        assert np.abs(c0 - np.round(c0)).max() < 1e-4


def test_drop_nonpositive_culls_only_nonpositive():
    """_drop_nonpositive keeps positive-amplitude splats, drops <=0, and is an
    identity (returns the same object) when all amplitudes are positive."""
    from luxar.gsplats.lod.substitutive import _drop_nonpositive

    centers = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], np.float32)
    chol = np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (4, 1))
    amps = np.array([0.5, 0.0, -1.0, 2.0], np.float32)
    data = GSplatData(centers=centers, amplitudes=amps, cholesky_factors=chol)
    out = _drop_nonpositive(data)
    assert out.n_splats == 2  # only the 0.5 and 2.0 rows survive
    assert np.array_equal(np.asarray(out.amplitudes), np.array([0.5, 2.0], np.float32))

    allpos = GSplatData(
        centers=centers, amplitudes=np.full(4, 1.0, np.float32), cholesky_factors=chol
    )
    assert _drop_nonpositive(allpos) is allpos  # no-op shortcut, no copy


def test_allocate_group_M_stable_tie_break():
    """Equal-size groups with a fractional tie give the extra rep(s) to the
    LOWEST-index groups (stable), so allocation is reproducible."""
    from luxar.gsplats.lod.substitutive import _allocate_group_M

    # 4 equal groups, total=6 -> base 1 each, rem=2 split over equal weights:
    # ideal=0.5 each, floor=0, leftover=2 -> first two groups get +1.
    alloc = _allocate_group_M(np.array([10, 10, 10, 10]), 6)
    assert alloc.tolist() == [2, 2, 1, 1]
