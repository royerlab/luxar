"""Tests for :mod:`luxar.gsplats.lod._substitutive.refine` (the L2 refit engine).

Engine-level coverage (the ``make_substitutive_lod(refine="l2")`` integration is
tested in :mod:`test_substitutive`):

- The refit strictly improves the closed-form relative L² vs the β=3 merge seed.
- Never-worse-than-seed: the trusted best is at most the seed's trusted E, and
  the raw seed genuinely wins when mass-pinning would worsen L² (the safety net
  is load-bearing, not decorative).
- Output mass equals the fine mixture's when the refit wins (the DC invariant);
  the old +41 % pair-list mass-inflation exploit is now structurally impossible.
- Barrier (frozen) dims: centers and Σ rows/cols stay at the seed values.
- Determinism given a generator; different seeds diverge.
- Minibatch path smoke; NaN-grad guard; standardization invariance.
- Spatial-hash cell sizing ignores degenerate (constant barrier) dims, so
  barrier-grouped refine never falls into the brute-force kNN path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from luxar.gsplats.lod._kernels import bin_squared_norm_torch
from luxar.gsplats.lod._substitutive.kmeans_lloyd import (
    _build_representatives_vectorized,
)
from luxar.gsplats.lod._substitutive.refine import (
    L2RefineConfig,
    l2_refine_mixture,
)
from luxar.gsplats.lod._substitutive.warm_start import _morton_partition

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


def _blobs_fine(
    n_clusters: int = 6, per: int = 80, d: int = 3, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isolated compact clusters — the regime where the β=3 seed over-blurs."""
    rng = np.random.default_rng(seed)
    centers = rng.random((n_clusters, d)) * 0.8 + 0.1
    pts = np.concatenate([c + rng.normal(0, 0.02, (per, d)) for c in centers], axis=0)
    n = len(pts)
    sig = 0.008
    L = np.zeros((n, d, d))
    for i in range(d):
        L[:, i, i] = sig
    return (
        torch.tensor(pts, dtype=torch.float64),
        torch.tensor(L, dtype=torch.float64),
        torch.ones(n, dtype=torch.float64),
    )


def _seed_from(
    fmu: torch.Tensor, fL: torch.Tensor, fa: torch.Tensor, K: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production-faithful seed: Morton partition + β=3 inflated merge + cull."""
    M = max(1, fmu.shape[0] // K)
    assign = _morton_partition(fmu, M=M)
    mu, L, a, _ = _build_representatives_vectorized(
        fmu, fL, fa, None, assign, M=M, coverage_inflation=3.0
    )
    keep = a > 0
    return mu[keep], L[keep], a[keep]


def _rel_l2(
    fmu: torch.Tensor,
    fL: torch.Tensor,
    fa: torch.Tensor,
    gmu: torch.Tensor,
    gL: torch.Tensor,
    ga: torch.Tensor,
) -> float:
    """Closed-form ‖f−g‖/‖f‖ via the signed-mixture dense Gram (test-sized)."""
    c = torch.cat([fmu, gmu.to(torch.float64)], dim=0)
    L = torch.cat([fL, gL.to(torch.float64)], dim=0)
    a = torch.cat([fa, -ga.to(torch.float64)], dim=0)
    diff = float(bin_squared_norm_torch(c, L, a))
    f2 = float(bin_squared_norm_torch(fmu, fL, fa))
    return float(np.sqrt(max(diff, 0.0) / max(f2, 1e-30)))


_FAST = L2RefineConfig(iters=60, rebuild_every=15, early_stop_patience=4)


def _gen(seed: int = 7) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


class TestL2RefineQuality:
    def test_improves_rel_l2_over_seed_on_blobs(self):
        fmu, fL, fa = _blobs_fine()
        smu, sL, sa = _seed_from(fmu, fL, fa)
        rmu, rL, ra, stats = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=_FAST, generator=_gen()
        )
        rel_seed = _rel_l2(fmu, fL, fa, smu, sL, sa)
        rel_refit = _rel_l2(fmu, fL, fa, rmu, rL, ra)
        # Prototype achieved 0.141 vs 0.233 on this regime; require >= 10 %
        # residual reduction with margin to spare.
        assert rel_refit < 0.9 * rel_seed, (
            f"refit did not improve: {rel_refit:.4f} vs seed {rel_seed:.4f}"
        )
        assert stats["improvement_frac"] > 0
        assert stats["iters_run"] > 0

    def test_never_worse_than_seed_trusted_metric(self):
        """Adversarially large lr: the trusted best must still be <= seed E
        (the seed is the first trusted candidate)."""
        fmu, fL, fa = _blobs_fine(seed=3)
        smu, sL, sa = _seed_from(fmu, fL, fa)
        bad = L2RefineConfig(
            iters=20, rebuild_every=5, lr_shape=5.0, lr_center_scale=5.0
        )
        rmu, rL, ra, stats = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=bad, generator=_gen()
        )
        assert stats["trusted_E_best"] <= stats["trusted_E_seed"] + 1e-9
        assert torch.isfinite(rmu).all()
        assert torch.isfinite(rL).all()
        assert (ra > 0).all()

    def test_seed_wins_when_mass_pinning_hurts(self):
        """The seed-as-first-candidate is load-bearing, not decorative. Build a
        case where the fine mixture's total mass is far from the coarse splat's
        L²-optimal mass, so pinning the coarse amplitude to the fine mass (what
        every *optimized* iterate does) strictly WORSENS L², and give the
        optimizer no room to recover (iters=1, huge lr). The engine must fall
        back to the raw seed exactly. Removing the pre-loop seed candidate makes
        this return the worse mass-pinned iterate — i.e. this test kills that
        mutant, unlike the generic never-worse case above."""
        d, n = 3, 200
        rng = np.random.default_rng(0)
        fmu = torch.tensor(rng.normal(0, 0.05, (n, d)))  # tight overlap: huge mass
        fL = torch.zeros(n, d, d, dtype=torch.float64)
        for i in range(d):
            fL[:, i, i] = 0.1
        fa = torch.ones(n, dtype=torch.float64)
        smu = fmu.mean(0, keepdim=True)
        sL = (torch.eye(d, dtype=torch.float64) * 0.15).unsqueeze(0)
        # L²-optimal single-splat amplitude for this field.
        from luxar.gsplats.lod._kernels import (
            gaussian_pair_inner_product_torch,
            gaussian_self_energy_torch,
            sqrt_det_from_cholesky,
        )

        Sf = fL @ fL.transpose(-1, -2)
        Sg = sL @ sL.transpose(-1, -2)
        fg = gaussian_pair_inner_product_torch(
            fmu, Sf, fa, smu.expand(n, d), Sg.expand(n, d, d), torch.ones(n)
        ).sum()
        gg_unit = gaussian_self_energy_torch(
            torch.ones(1, dtype=torch.float64), sqrt_det_from_cholesky(sL), d
        )
        sa = (fg / gg_unit).reshape(1)
        cfg = L2RefineConfig(
            iters=1, rebuild_every=1, lr_shape=80.0, lr_center_scale=80.0
        )
        rmu, rL, ra, stats = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=cfg, generator=_gen(0)
        )
        # Seed strictly beats fine-mass pinning here, so it must win outright.
        assert stats["trusted_E_best"] == stats["trusted_E_seed"]
        assert stats["seed_won"] is True
        torch.testing.assert_close(rmu, smu.to(torch.float32), rtol=0, atol=1e-6)
        torch.testing.assert_close(ra, sa.to(torch.float32), rtol=1e-5, atol=1e-6)

    def test_output_mass_equals_fine_when_refit_wins(self):
        """The DC invariant that mass pinning exists for: whenever the refit
        beats the merge (the normal case), the winning iterate's total mass
        equals the FINE mixture's exactly — so additive/X-ray brightness does
        not pop across LOD levels. (The old +41 % pair-list mass-inflation
        exploit is now structurally impossible: amplitudes live on the mass
        manifold.)"""
        fmu, fL, fa = _blobs_fine()
        smu, sL, sa = _seed_from(fmu, fL, fa)
        _, _, _, stats = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=_FAST, generator=_gen()
        )
        assert stats["seed_won"] is False  # refit beat the merge
        assert abs(stats["mass_vs_fine"] - 1.0) < 1e-4
        assert stats["mass_drift_warning"] is False


class TestFrozenDims:
    def test_frozen_center_and_sigma_rows_stay_at_seed(self):
        """4D fixture with dim 0 a barrier: centers keep the barrier value and
        Σ's row/col 0 stay at the seed (within the float32 standardization
        round-trip, ~1 ulp)."""
        rng = np.random.default_rng(0)
        n = 320
        xyz = rng.random((n, 3)) * 0.9
        pts = np.column_stack([np.full(n, 2.0), xyz])
        L = np.zeros((n, 4, 4))
        for i in range(4):
            L[:, i, i] = 0.02 if i else 0.001
        fmu = torch.tensor(pts, dtype=torch.float64)
        fL = torch.tensor(L, dtype=torch.float64)
        fa = torch.ones(n, dtype=torch.float64)
        smu, sL, sa = _seed_from(fmu, fL, fa)
        rmu, rL, _, _ = l2_refine_mixture(
            fmu,
            fL,
            fa,
            smu,
            sL,
            sa,
            config=_FAST,
            frozen_dims=(0,),
            generator=_gen(),
        )
        np.testing.assert_allclose(
            rmu[:, 0].numpy(), smu[:, 0].to(torch.float32).numpy(), rtol=1e-6
        )
        # Row 0 and column 0 of L frozen -> Sigma's barrier row/col preserved.
        np.testing.assert_allclose(
            rL[:, 0, :].numpy(),
            sL[:, 0, :].to(torch.float32).numpy(),
            rtol=1e-5,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            rL[:, :, 0].numpy(),
            sL[:, :, 0].to(torch.float32).numpy(),
            rtol=1e-5,
            atol=1e-9,
        )
        # And the free block actually moved (the freeze isn't a global no-op).
        assert not np.allclose(
            rL[:, 1:, 1:].numpy(), sL[:, 1:, 1:].to(torch.float32).numpy()
        )

    def test_nonzero_barrier_dim_freezes_full_row(self):
        """Barrier dim 0 is a weak test: row 0 of a lower-triangular L has only
        the [0,0] diagonal (in column 0), so freezing columns alone already
        freezes the row. A barrier dim >= 1 with off-diagonal seed entries in
        FREE columns is the discriminating case — it distinguishes a row+col
        freeze from a column-only freeze. Here barrier = dim 2, and the seed has
        nonzero L[:, 2, 0] / L[:, 2, 1] (barrier row coupled to free columns);
        the whole barrier row AND column must stay bit-exact. A column-only
        mask would let L[:, 2, 0] / L[:, 2, 1] drift."""
        rng = np.random.default_rng(0)
        n = 240
        xyz = rng.random((n, 3)) * 0.8
        pts = np.column_stack([xyz[:, 0], xyz[:, 1], np.full(n, 3.0), xyz[:, 2]])
        fL = np.zeros((n, 4, 4))
        for i in range(4):
            fL[:, i, i] = 0.03
        fmu = torch.tensor(pts, dtype=torch.float64)
        fLt = torch.tensor(fL, dtype=torch.float64)
        fa = torch.ones(n, dtype=torch.float64)
        smu, sL, sa = _seed_from(fmu, fLt, fa)
        # Inject a NON-axis-aligned barrier row: couple barrier dim 2 to free
        # columns 0 and 1 (valid lower-triangular, diagonal untouched -> PD).
        sL = sL.clone()
        sL[:, 2, 0] = 0.011
        sL[:, 2, 1] = -0.007
        rmu, rL, _, _ = l2_refine_mixture(
            fmu,
            fLt,
            fa,
            smu,
            sL,
            sa,
            config=_FAST,
            frozen_dims=(2,),
            generator=_gen(),
        )
        s32 = sL.to(torch.float32).numpy()
        np.testing.assert_allclose(
            rmu[:, 2].numpy(), smu[:, 2].to(torch.float32).numpy(), rtol=1e-6
        )
        # Full barrier ROW (incl. the injected free-column couplings) frozen.
        np.testing.assert_allclose(
            rL[:, 2, :].numpy(), s32[:, 2, :], rtol=1e-5, atol=1e-9
        )
        np.testing.assert_allclose(
            rL[:, :, 2].numpy(), s32[:, :, 2], rtol=1e-5, atol=1e-9
        )
        # Free block moved.
        free = [0, 1, 3]
        assert not np.allclose(
            rL[:, free][:, :, free].numpy(), s32[:, free][:, :, free]
        )


class TestCoarseCoarsePairs:
    def test_cc_pairs_or_symmetric(self):
        """The coarse-coarse pair list must be OR-symmetric: an unordered pair
        {a,b} is present iff a is in b's kNN OR b is in a's. The naive bi<bj on
        directed truncated-kNN edges would drop pairs where the lower-index
        endpoint is in a denser region. Verify against a brute-force radius set
        on a heterogeneous-density coarse cloud."""
        from luxar.gsplats.lod._substitutive.refine import _build_pair_lists

        rng = np.random.default_rng(0)
        # Dense cluster + sparse halo -> asymmetric kNN truncation.
        dense = rng.normal(0, 0.05, (200, 3))
        sparse = rng.normal(0, 1.0, (40, 3))
        coarse = np.concatenate([dense, sparse], axis=0).astype(np.float32)
        fine = rng.normal(0, 0.5, (300, 3)).astype(np.float32)
        cfg = L2RefineConfig(cc_k=8)  # small k -> aggressive truncation
        sc = 0.05
        _, _, bi, bj = _build_pair_lists(
            fine,
            coarse,
            config=cfg,
            sigma_c=sc,
            sigma_f=0.1,
            device=torch.device("cpu"),
        )
        got = {
            (int(min(a, b)), int(max(a, b))) for a, b in zip(bi.tolist(), bj.tolist())
        }
        # No self-pairs, no duplicates.
        assert all(a != b for a, b in got)
        assert len(got) == bi.numel()
        # Every returned pair is genuinely within the cc radius.
        r_cc = cfg.cc_radius_sigmas * sc
        d = np.linalg.norm(coarse[bi.numpy()] - coarse[bj.numpy()], axis=1)
        assert (d < r_cc + 1e-6).all()
        # OR-symmetry: build the directed kNN both ways; every unordered pair
        # seen from EITHER side and within radius must be present.
        from scipy.spatial import cKDTree

        tree = cKDTree(coarse)
        expect = set()
        for i in range(len(coarse)):
            dd, idx = tree.query(coarse[i], k=min(cfg.cc_k + 1, len(coarse)))
            for dist, j in zip(np.atleast_1d(dd), np.atleast_1d(idx)):
                if j != i and dist < r_cc:
                    expect.add((min(i, int(j)), max(i, int(j))))
        assert got == expect, f"missing {expect - got}, extra {got - expect}"


class TestDegenerateDimCellSizing:
    """Regression for the constant-barrier-dim spatial-hash collapse: sizing
    the hash cell from the FULL-D bbox product let a zero-extent dim (clamped
    to 1e-9) shrink the cell ~100x below the true neighbour spacing, so the
    ``_KNN_MAX_SHELL`` shells found nothing and every kNN query degenerated to
    a per-query brute-force scan of the whole set — barrier-grouped refine
    (constant barrier coords per group, the flagship ``--coarsen-dims`` 4D
    timelapse case) effectively hung at realistic sizes. The cell must now be
    computed from the EFFECTIVE dims only."""

    @staticmethod
    def _clouds(
        seed: int = 0, n_fine: int = 600, n_coarse: int = 150
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """3D clouds plus the same clouds lifted to 4D with a constant dim 0
        (the barrier-group geometry: every splat shares the barrier value)."""
        rng = np.random.default_rng(seed)
        fine3 = rng.random((n_fine, 3)).astype(np.float32)
        coarse3 = rng.random((n_coarse, 3)).astype(np.float32)
        fine4 = np.column_stack([np.full(n_fine, 2.0, np.float32), fine3])
        coarse4 = np.column_stack([np.full(n_coarse, 2.0, np.float32), coarse3])
        return fine3, coarse3, fine4, coarse4

    def test_cell_size_ignores_degenerate_dims(self):
        """The computed cell with a constant dim present equals the cell with
        that dim dropped (the load-bearing assertion: brute-force fallback
        also returns *correct* pairs, so only the cell size proves the shell
        search stays effective)."""
        from luxar.gsplats.lod._substitutive.refine import _hash_cell_size

        _, coarse3, _, coarse4 = self._clouds()
        cell3 = _hash_cell_size(coarse3)
        cell4 = _hash_cell_size(coarse4)
        assert cell4 == pytest.approx(cell3, rel=1e-6)
        # Sanity: matches the documented 2x-typical-NN-spacing convention over
        # the three live dims (not collapsed by the 1e-9-clamped fourth).
        ext = (coarse3.max(0) - coarse3.min(0)).astype(np.float64)
        expected = 2.0 * float((np.prod(ext) / len(coarse3)) ** (1.0 / 3.0))
        assert cell4 == pytest.approx(expected, rel=1e-6)

    def test_cell_size_all_coincident_fallback(self):
        from luxar.gsplats.lod._substitutive.refine import _hash_cell_size

        cell = _hash_cell_size(np.full((7, 3), 1.5, dtype=np.float32))
        assert cell == pytest.approx(1e-6)

    def test_pair_lists_match_dropped_dim(self):
        """A constant dim contributes exactly 0 to every neighbour distance,
        so the pair lists (cross AND cc) must be identical to those of the
        same data with the dim dropped."""
        from luxar.gsplats.lod._substitutive.refine import _build_pair_lists

        fine3, coarse3, fine4, coarse4 = self._clouds()
        kw = dict(
            config=L2RefineConfig(),
            sigma_c=0.05,
            sigma_f=0.02,
            device=torch.device("cpu"),
        )
        p3 = _build_pair_lists(fine3, coarse3, **kw)
        p4 = _build_pair_lists(fine4, coarse4, **kw)

        def pairset(a: torch.Tensor, b: torch.Tensor) -> set:
            return set(zip(a.tolist(), b.tolist()))

        assert pairset(p4[0], p4[1]) == pairset(p3[0], p3[1])  # fine→coarse
        assert pairset(p4[2], p4[3]) == pairset(p3[2], p3[3])  # coarse-coarse
        assert p4[0].numel() > 0 and p4[2].numel() > 0  # non-trivial fixture


class TestDeterminismAndMinibatch:
    # Tiny pair budget forces the minibatch (per-step pair sampling) path.
    _MB = L2RefineConfig(iters=40, rebuild_every=10, step_pair_budget=512)

    def test_deterministic_given_generator(self):
        """Same generator seed → same result up to CPU-threading noise (torch
        multithreaded reductions are not bitwise run-to-run reproducible, so
        the contract is tight closeness, ~1 ulp); different seeds → genuinely
        different minibatch trajectories."""
        fmu, fL, fa = _blobs_fine()
        smu, sL, sa = _seed_from(fmu, fL, fa)
        out1 = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=self._MB, generator=_gen(7)
        )
        out2 = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=self._MB, generator=_gen(7)
        )
        for a, b in zip(out1[:3], out2[:3]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        out3 = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=self._MB, generator=_gen(8)
        )
        assert float((out1[0] - out3[0]).abs().max()) > 1e-4

    def test_minibatch_path_improves(self):
        fmu, fL, fa = _blobs_fine()
        smu, sL, sa = _seed_from(fmu, fL, fa)
        _, _, _, stats = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=self._MB, generator=_gen()
        )
        assert stats["minibatched"] is True
        assert stats["improvement_frac"] > 0
        assert abs(stats["mass_vs_fine"] - 1.0) < 1e-4

    def test_trusted_gg_chunking_equals_full(self):
        """E1 memory-bound: the no-grad trusted ‖g‖² uses a chunked coarse-
        coarse sum. It must equal a single full-tensor evaluation exactly (the
        chunking only bounds the working set — it must not change the value that
        gates best-iterate acceptance). Force a chunk boundary via a tiny cap."""
        import luxar.gsplats.lod._substitutive.refine as refine_mod

        fmu, fL, fa = _blobs_fine()
        smu, sL, sa = _seed_from(fmu, fL, fa)
        smu, sL, sa = smu.to(torch.float32), sL.to(torch.float32), sa.to(torch.float32)
        _, _, bi, bj = refine_mod._build_pair_lists(
            fmu.to(torch.float32).numpy(),
            smu.numpy(),
            config=L2RefineConfig(),
            sigma_c=0.02,
            sigma_f=0.01,
            device=torch.device("cpu"),
        )
        assert bi.numel() > 4  # need several pairs to span chunks
        full = (
            refine_mod._g_self_energy(sL, sa)
            + 2.0
            * refine_mod._pair_K(smu[bi], sL[bi], sa[bi], smu[bj], sL[bj], sa[bj]).sum()
        )
        orig = refine_mod._EVAL_CHUNK_PAIRS
        try:
            refine_mod._EVAL_CHUNK_PAIRS = 3  # force multiple chunks
            chunked = refine_mod._g_self_energy(
                sL, sa
            ) + 2.0 * refine_mod._pair_K_sum_chunked(smu, sL, sa, smu, sL, sa, bi, bj)
        finally:
            refine_mod._EVAL_CHUNK_PAIRS = orig
        torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-8)

    def test_cc_minibatch_cross_is_unbiased(self):
        """The per-step minibatched cc cross term (sampled |batch|=budget of n_cc
        pairs, scaled n_cc/budget) is an unbiased estimator of the full cc cross
        sum — the property that makes E1's memory bound safe."""
        import luxar.gsplats.lod._substitutive.refine as refine_mod

        fmu, fL, fa = _blobs_fine()
        smu, sL, sa = _seed_from(fmu, fL, fa)
        smu, sL, sa = smu.to(torch.float32), sL.to(torch.float32), sa.to(torch.float32)
        _, _, bi, bj = refine_mod._build_pair_lists(
            fmu.to(torch.float32).numpy(),
            smu.numpy(),
            config=L2RefineConfig(),
            sigma_c=0.02,
            sigma_f=0.01,
            device=torch.device("cpu"),
        )
        full = float(
            refine_mod._pair_K(smu[bi], sL[bi], sa[bi], smu[bj], sL[bj], sa[bj]).sum()
        )
        budget = max(4, bi.numel() // 3)
        mb = refine_mod._PairMinibatch(bi, bj, budget, torch.device("cpu"))
        assert mb.minibatched
        g = _gen(0)
        est = []
        for _ in range(500):
            ii, jj, scale = mb.sample(g)
            est.append(
                float(
                    refine_mod._pair_K(
                        smu[ii], sL[ii], sa[ii], smu[jj], sL[jj], sa[jj]
                    ).sum()
                )
                * scale
            )
        mc = float(np.mean(est))
        assert abs(mc - full) < 0.05 * abs(full) + 1e-9, (
            f"biased: MC {mc} vs full {full}"
        )


class TestRobustness:
    def test_nan_grad_guard_skips_and_stays_finite(self, monkeypatch):
        """Inject NaN into the step objective's ‖g‖² term for the first few
        gradient steps (coarse Σ is PD by construction, so no *data* can reach
        the kernels' non-PD branch — the guard is a defensive layer and must be
        exercised by fault injection): the guard skips those steps and the
        output stays finite."""
        import luxar.gsplats.lod._substitutive.refine as refine_mod

        fmu, fL, fa = _blobs_fine(n_clusters=3, per=60)
        smu, sL, sa = _seed_from(fmu, fL, fa)
        original = refine_mod._g_norm_sq
        calls = {"n": 0}

        def poisoned(mu, L, a, bi, bj, cc_scale=1.0):
            out = original(mu, L, a, bi, bj, cc_scale=cc_scale)
            if a.requires_grad:  # only poison grad-carrying (step) evaluations
                calls["n"] += 1
                if calls["n"] <= 3:
                    # sqrt of a negative grad-carrying value -> NaN value+grads
                    out = out + torch.sqrt(-(a.sum().abs() + 1.0))
            return out

        monkeypatch.setattr(refine_mod, "_g_norm_sq", poisoned)
        rmu, rL, ra, stats = l2_refine_mixture(
            fmu,
            fL,
            fa,
            smu,
            sL,
            sa,
            config=L2RefineConfig(iters=12, rebuild_every=6),
            generator=_gen(),
        )
        assert stats["nan_grad_skips"] > 0
        assert torch.isfinite(rmu).all()
        assert torch.isfinite(rL).all()
        assert torch.isfinite(ra).all()

    def test_empty_and_trivial_inputs_pass_through(self):
        d = 3
        empty = torch.zeros((0, d), dtype=torch.float64)
        empty_L = torch.zeros((0, d, d), dtype=torch.float64)
        empty_a = torch.zeros(0, dtype=torch.float64)
        fmu, fL, fa = _blobs_fine(n_clusters=2, per=20)
        mu, L, a, stats = l2_refine_mixture(
            fmu, fL, fa, empty, empty_L, empty_a, config=_FAST
        )
        assert mu.shape[0] == 0 and stats["iters_run"] == 0

    def test_standardization_invariance(self):
        """refit(s*x + t) == s*refit(x) + t under a global similarity."""
        fmu, fL, fa = _blobs_fine()
        smu, sL, sa = _seed_from(fmu, fL, fa)
        base = l2_refine_mixture(
            fmu, fL, fa, smu, sL, sa, config=_FAST, generator=_gen(5)
        )
        s, t = 3.7, 1.234
        moved = l2_refine_mixture(
            fmu * s + t,
            fL * s,
            fa,
            smu * s + t,
            sL * s,
            sa,
            config=_FAST,
            generator=_gen(5),
        )
        np.testing.assert_allclose(
            moved[0].numpy(), base[0].numpy() * s + t, rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            moved[1].numpy(), base[1].numpy() * s, rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(moved[2].numpy(), base[2].numpy(), rtol=1e-3)


def test_config_defaults_documented_invariants():
    cfg = L2RefineConfig()
    assert cfg.iters == 120
    assert cfg.rebuild_every == 20
    assert cfg.early_stop_patience >= 1
