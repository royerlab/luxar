"""Tests for :mod:`luxar.gsplats.lod._kernels`.

The closed-form Gaussian-mixture primitives shared by the additive and
substitutive LOD axes: mass-weighted moment matching (law of total
variance), the $L^2$-optimal bin amplitude, per-bin squared norms, and
their degenerate-input guards and algebraic invariants (symmetry / PSD /
positive self-energy).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod._kernels import (
    bin_inner_product_with_template_torch,
    bin_squared_norm_torch,
    gaussian_pair_inner_product_numpy,
    gaussian_self_energy_numpy,
    kwise_moment_match_torch,
    template_squared_norm_torch,
)
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
# Bin-merge primitives
# ─────────────────────────────────────────────────────────────────────


class TestKWiseMomentMatch:
    def test_law_of_total_variance(self):
        """Σ̄ = Σ_i w_i Σ_i + Σ_i w_i (μ_i - μ̄)(μ_i - μ̄)^T."""
        data = _make_anisotropic_3d(n=4, seed=10)
        centres, L, amps = _gsplat_to_torch(data)
        mu_bar, Sigma_bar, weights = kwise_moment_match_torch(centres, L, amps)

        # Compute expected via direct mass weighting.
        sqrt_det = torch.abs(torch.prod(torch.diagonal(L, dim1=-2, dim2=-1), dim=-1))
        masses = amps * sqrt_det
        expected_w = masses / masses.sum()
        # [P7] These are exact float64 closed-form identities (no iteration),
        # so the tolerance should reflect float64 round-off (~1e-12), far
        # tighter than assert_close's 1e-5 default which would mask a real
        # systematic bias in the moment aggregation.
        torch.testing.assert_close(weights, expected_w, rtol=1e-12, atol=1e-12)
        expected_mu = (expected_w[:, None] * centres).sum(dim=0)
        torch.testing.assert_close(mu_bar, expected_mu, rtol=1e-10, atol=1e-12)

        # Direct intra + inter computation
        Sigma = L @ L.transpose(-1, -2)
        intra = (expected_w[:, None, None] * Sigma).sum(dim=0)
        delta = centres - expected_mu[None, :]
        inter = (
            expected_w[:, None, None] * (delta.unsqueeze(2) @ delta.unsqueeze(1))
        ).sum(dim=0)
        expected_Sigma = intra + inter
        torch.testing.assert_close(Sigma_bar, expected_Sigma, rtol=1e-10, atol=1e-12)
        # [P12] Σ̄ is a covariance: it must be symmetric and positive
        # semi-definite by construction (intra + inter are both PSD). These
        # invariants are never asserted otherwise, yet they are exactly what
        # a downstream Cholesky relies on.
        torch.testing.assert_close(
            Sigma_bar, Sigma_bar.transpose(-1, -2), rtol=0, atol=1e-12
        )
        eigvals = torch.linalg.eigvalsh(Sigma_bar)
        assert float(eigvals[0]) > -1e-10, f"Σ̄ not PSD; min eig {float(eigvals[0])}"

    def test_pairwise_reduces_to_companion_doc(self):
        """K=2: inter-bin spread = w_1 w_2 (μ_1 - μ_2)(μ_1 - μ_2)^T."""
        data = _make_anisotropic_3d(n=2, seed=20)
        centres, L, amps = _gsplat_to_torch(data)
        mu_bar, Sigma_bar, weights = kwise_moment_match_torch(centres, L, amps)
        Sigma = L @ L.transpose(-1, -2)
        # Manual pairwise formula
        w1, w2 = float(weights[0]), float(weights[1])
        diff = centres[0] - centres[1]
        inter_expected = w1 * w2 * torch.outer(diff, diff)
        intra_expected = w1 * Sigma[0] + w2 * Sigma[1]
        torch.testing.assert_close(Sigma_bar, intra_expected + inter_expected)

    def test_l2_optimal_amplitude_zero_gradient(self):
        """∂/∂a ‖f - a Ḡ‖² = 0 at a = a* (Prop. 2.2)."""
        data = _make_anisotropic_3d(n=4, seed=30)
        centres, L, amps = _gsplat_to_torch(data)
        mu_bar, Sigma_bar, _ = kwise_moment_match_torch(centres, L, amps)
        template_inner = bin_inner_product_with_template_torch(
            centres, L, amps, mu_bar, Sigma_bar
        )
        template_norm_sq = template_squared_norm_torch(Sigma_bar)
        a_star = float(template_inner / template_norm_sq)

        # Numerical gradient: grad = -2 ⟨f, Ḡ⟩ + 2a ‖Ḡ‖²
        def cost(a: float) -> float:
            return (
                float(bin_squared_norm_torch(centres, L, amps))
                - 2 * a * float(template_inner)
                + a * a * float(template_norm_sq)
            )

        # [P7] The cost is an exact quadratic in ``a``, so the central
        # finite-difference gradient is exact up to float64 round-off for any
        # eps. The previous (eps=1e-3, tol=1e-3) accepted a gradient up to
        # ~10x the noise floor. Tie the tolerance to the curvature
        # (2·‖Ḡ‖²): the optimum gradient must be a negligible fraction of it.
        eps = 1e-4
        grad = (cost(a_star + eps) - cost(a_star - eps)) / (2 * eps)
        curvature = 2.0 * float(template_norm_sq)
        assert abs(grad) < 1e-6 * max(curvature, 1.0), (
            f"gradient at optimum should be ~0; got {grad} (curvature {curvature})"
        )


class TestKernelDegenerate:
    """[P5] Degenerate / boundary inputs to the load-bearing kernels.

    These guards (empty bin, zero-mass fallback, single-point bin,
    coincident points) are reachable in real reductions but were never
    exercised directly.
    """

    def test_empty_bin_returns_sentinel(self):
        """K=0 → (zeros(D), I_D, zeros(0)) per the documented contract."""
        D = 3
        centres = torch.zeros((0, D), dtype=torch.float64)
        L = torch.zeros((0, D, D), dtype=torch.float64)
        amps = torch.zeros(0, dtype=torch.float64)
        mu_bar, Sigma_bar, weights = kwise_moment_match_torch(centres, L, amps)
        torch.testing.assert_close(mu_bar, torch.zeros(D, dtype=torch.float64))
        torch.testing.assert_close(Sigma_bar, torch.eye(D, dtype=torch.float64))
        assert weights.shape == (0,)
        # An empty bin has zero L² energy.
        assert float(bin_squared_norm_torch(centres, L, amps)) == 0.0

    def test_zero_mass_bin_falls_back_to_uniform_weights(self):
        """All-zero amplitudes → mass total ≤ 0 → uniform weights 1/K
        (so downstream code never divides by zero)."""
        K, D = 4, 3
        # [P6] Seed torch's RNG (the conftest only seeds numpy's global RNG).
        # The uniform-weights result is input-independent, but seed anyway
        # for full determinism per the codebase convention.
        torch.manual_seed(0)
        centres = torch.randn(K, D, dtype=torch.float64)
        L = torch.eye(D, dtype=torch.float64).expand(K, D, D).contiguous()
        amps = torch.zeros(K, dtype=torch.float64)
        _, _, weights = kwise_moment_match_torch(centres, L, amps)
        torch.testing.assert_close(
            weights, torch.full((K,), 1.0 / K, dtype=torch.float64)
        )

    def test_single_point_bin_norm_equals_self_energy(self):
        """[P5/P12] K=1 short-circuit: ‖f‖² of a one-splat bin equals the
        closed-form self-energy a²·π^{D/2}·|Σ|^{1/2}."""
        D = 3
        centres = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
        L = (torch.eye(D, dtype=torch.float64) * 1.5).unsqueeze(0)
        amps = torch.tensor([2.0], dtype=torch.float64)
        norm_sq = float(bin_squared_norm_torch(centres, L, amps))
        sqrt_det = float(torch.abs(torch.prod(torch.diagonal(L[0]))))
        expected = float(
            gaussian_self_energy_numpy(np.array([2.0]), np.array([sqrt_det]), D)[0]
        )
        assert norm_sq == pytest.approx(expected, rel=1e-12)

    def test_coincident_points_no_nan(self):
        """[P5] Identical centres → zero inter-bin spread; Σ̄ reduces to the
        shared intra covariance with no NaN/Inf."""
        K, D = 3, 3
        centre = torch.tensor([0.7, -0.3, 1.1], dtype=torch.float64)
        centres = centre.unsqueeze(0).expand(K, D).contiguous()
        L = (torch.eye(D, dtype=torch.float64) * 0.9).expand(K, D, D).contiguous()
        amps = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
        mu_bar, Sigma_bar, _ = kwise_moment_match_torch(centres, L, amps)
        assert torch.all(torch.isfinite(Sigma_bar))
        torch.testing.assert_close(mu_bar, centre, rtol=1e-12, atol=1e-12)
        # Inter-bin spread is zero → Σ̄ equals the (shared) intra covariance.
        torch.testing.assert_close(Sigma_bar, (L[0] @ L[0].T), rtol=1e-12, atol=1e-12)


class TestKernelInvariants:
    """[P12] Algebraic invariants of the Gaussian inner-product kernel."""

    def test_self_energy_strictly_positive(self):
        """[P12/G13] K_ii = a²·π^{D/2}·|Σ|^{1/2} > 0 for every non-degenerate
        splat."""
        data = _make_anisotropic_3d(n=12, seed=7)
        _, L, amps = _gsplat_to_torch(data)
        sqrt_det = np.abs(np.prod(np.diagonal(L.numpy(), axis1=-2, axis2=-1), axis=-1))
        e = gaussian_self_energy_numpy(amps.numpy(), sqrt_det, data.ndim)
        assert np.all(e > 0)

    def test_bin_gram_symmetric_and_psd(self):
        """[P12/G12] The K×K Gram block of a bin is symmetric and PSD.

        Built explicitly via the closed-form pair kernel; ``bin_squared_norm``
        sums it, so its symmetry/PSD are what make ‖f‖² ≥ 0 well-defined.
        """
        data = _make_anisotropic_3d(n=5, seed=13)
        centres, L, amps = _gsplat_to_torch(data)
        Sigma = (L @ L.transpose(-1, -2)).numpy()
        mu = centres.numpy()
        a = amps.numpy()
        n = mu.shape[0]

        gram = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                gram[i, j] = gaussian_pair_inner_product_numpy(
                    mu[i], Sigma[i], float(a[i]), mu[j], Sigma[j], float(a[j])
                )
        np.testing.assert_allclose(gram, gram.T, rtol=0, atol=1e-10)
        eig = np.linalg.eigvalsh(gram)
        assert eig.min() > -1e-8, f"Gram not PSD; min eig {eig.min()}"
