"""Tests for the measured mixture quality (`lod/quality.py`, the Q of Q·e).

Pure-measurement tests: no optimizer, no recipes — synthetic mixtures with
known relationships (identity, prefix nesting, disjoint content) pin the
quality semantics the LOD stamps rely on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from luxar.gsplats import GSplatData
from luxar.gsplats.lod.quality import (
    QualityResult,
    mixture_quality,
    total_self_energy,
)


def _make_mixture(
    n: int, *, seed: int = 0, offset: float = 0.0, sigma: float = 0.4
) -> GSplatData:
    """Random isotropic 3D mixture in a unit-ish box (+ optional offset)."""
    rng = np.random.default_rng(seed)
    centers = (rng.random((n, 3)) * 10.0 + offset).astype(np.float32)
    amps = rng.uniform(0.5, 1.5, n).astype(np.float32)
    tri = np.zeros((n, 6), dtype=np.float32)
    tri[:, 0] = sigma  # L00
    tri[:, 2] = sigma  # L11
    tri[:, 5] = sigma  # L22
    return GSplatData(centers=centers, amplitudes=amps, cholesky_factors=tri)


def _prefix(data: GSplatData, k: int) -> GSplatData:
    return GSplatData(
        centers=np.asarray(data.centers)[:k].copy(),
        amplitudes=np.asarray(data.amplitudes)[:k].copy(),
        cholesky_factors=np.asarray(data.cholesky_factors)[:k].copy(),
    )


class TestTotalSelfEnergy:
    def test_single_isotropic_gaussian_closed_form(self) -> None:
        # One splat: energy = a² · π^{D/2} · |Σ|^{1/2}, |Σ|^{1/2} = σ^D.
        a, sigma = 2.0, 0.5
        data = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.array([a], dtype=np.float32),
            cholesky_factors=np.array(
                [[sigma, 0, sigma, 0, 0, sigma]], dtype=np.float32
            ),
        )
        expected = a**2 * math.pi ** (3 / 2) * sigma**3
        assert total_self_energy(data) == pytest.approx(expected, rel=1e-6)

    def test_alpha_effective_amplitude_for_rgba(self) -> None:
        # Classical-import convention: amplitude ≡ 1, per-splat weight carried
        # in the color alpha. The self-energy must use the alpha-effective
        # amplitude A·α, so it equals Σ α² · π^{D/2} · σ^D (NOT the raw Σ 1).
        sigma = 0.4
        alpha = np.array([0.25, 1.0, 0.5, 1.0, 0.25, 0.75, 1.0, 0.5], dtype=np.float32)
        n = alpha.size
        colors = np.ones((n, 4), dtype=np.float32)
        colors[:, 3] = alpha
        tri = np.zeros((n, 6), dtype=np.float32)
        tri[:, 0] = sigma  # L00
        tri[:, 2] = sigma  # L11
        tri[:, 5] = sigma  # L22
        rng = np.random.default_rng(42)
        data = GSplatData(
            centers=rng.random((n, 3)).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=tri,
            colors=colors,
        )
        expected = (
            float(np.sum(alpha.astype(np.float64) ** 2)) * math.pi ** (3 / 2) * sigma**3
        )
        assert total_self_energy(data) == pytest.approx(expected, rel=1e-6)
        # And it genuinely differs from the raw-amplitude value (Σ 1²·…).
        raw = float(n) * math.pi ** (3 / 2) * sigma**3
        assert total_self_energy(data) != pytest.approx(raw, rel=1e-3)

    def test_non_rgba_matches_raw_amplitude(self) -> None:
        # No RGBA alpha ⇒ effective_amplitudes is a no-op ⇒ the value is the
        # raw-amplitude self-energy, both with no colors and with 3-col RGB.
        data = _make_mixture(64, seed=13)  # constant σ = 0.4
        amps = np.asarray(data.amplitudes, dtype=np.float64)
        expected = float(np.sum(amps**2)) * math.pi ** (3 / 2) * 0.4**3
        assert total_self_energy(data) == pytest.approx(expected, rel=1e-6)
        rgb = GSplatData(
            centers=np.asarray(data.centers).copy(),
            amplitudes=np.asarray(data.amplitudes).copy(),
            cholesky_factors=np.asarray(data.cholesky_factors).copy(),
            colors=np.full((data.n_splats, 3), 0.5, dtype=np.float32),
        )
        assert total_self_energy(rgb) == pytest.approx(expected, rel=1e-6)

    def test_additivity_over_splats(self) -> None:
        data = _make_mixture(64, seed=1)
        half_a = total_self_energy(_prefix(data, 32))
        # Second half via a fresh GSplatData over the tail rows.
        tail = GSplatData(
            centers=np.asarray(data.centers)[32:].copy(),
            amplitudes=np.asarray(data.amplitudes)[32:].copy(),
            cholesky_factors=np.asarray(data.cholesky_factors)[32:].copy(),
        )
        assert half_a + total_self_energy(tail) == pytest.approx(
            total_self_energy(data), rel=1e-9
        )


class TestMixtureQuality:
    def test_identity_scores_one(self) -> None:
        data = _make_mixture(300, seed=2)
        result = mixture_quality(data, data)
        assert isinstance(result, QualityResult)
        assert result.quality == pytest.approx(1.0, abs=1e-4)
        assert result.l2_sq == pytest.approx(0.0, abs=result.ref_norm_sq * 1e-4)

    def test_empty_approx_scores_zero(self) -> None:
        ref = _make_mixture(50, seed=3)
        empty = _prefix(ref, 0)
        result = mixture_quality(empty, ref)
        assert result.quality == 0.0
        assert result.n_cross_pairs == 0

    def test_empty_reference_raises(self) -> None:
        ref = _make_mixture(10, seed=4)
        with pytest.raises(ValueError, match="reference"):
            mixture_quality(ref, _prefix(ref, 0))

    def test_ndim_mismatch_raises(self) -> None:
        a = _make_mixture(10, seed=5)
        rng = np.random.default_rng(6)
        b = GSplatData(
            centers=rng.random((10, 2)).astype(np.float32),
            amplitudes=np.ones(10, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([0.3, 0.0, 0.3], dtype=np.float32), (10, 1)
            ),
        )
        with pytest.raises(ValueError, match="ndim"):
            mixture_quality(a, b)

    def test_disjoint_content_scores_zero(self) -> None:
        # An approximation with all its energy FAR from the reference explains
        # nothing: ‖A−B‖² = ‖A‖² + ‖B‖² ≥ ‖B‖² → quality clamps to 0.
        ref = _make_mixture(200, seed=7)
        far = _make_mixture(200, seed=8, offset=1e4)
        result = mixture_quality(far, ref)
        assert result.quality == 0.0

    def test_prefix_quality_is_monotone_and_interior(self) -> None:
        # Energy-ordered-ish prefixes of the reference itself: bigger prefix →
        # strictly better approximation; both strictly inside (0, 1).
        ref = _make_mixture(400, seed=9)
        q_small = mixture_quality(_prefix(ref, 40), ref).quality
        q_large = mixture_quality(_prefix(ref, 300), ref).quality
        assert 0.0 < q_small < q_large < 1.0

    def test_sampled_matches_exact_within_tolerance(self) -> None:
        # The approximation is a jittered, amplitude-rescaled subset — merged-
        # level-like content that shares NO identical splats with the
        # reference (the documented domain of the sampled estimator).
        ref = _make_mixture(4_000, seed=10)
        rng = np.random.default_rng(20)
        approx = GSplatData(
            centers=(
                np.asarray(ref.centers)[::3] + rng.normal(0, 0.05, (1334, 3))
            ).astype(np.float32),
            amplitudes=(np.asarray(ref.amplitudes)[::3] * 3.0).astype(np.float32),
            cholesky_factors=np.asarray(ref.cholesky_factors)[::3].copy(),
        )
        exact = mixture_quality(approx, ref)
        sampled = mixture_quality(approx, ref, max_pair_splats=800)
        assert sampled.approx_pair_fraction < 1.0
        assert sampled.ref_pair_fraction < 1.0
        # Pair terms are sampled estimates; the exact diagonals dominate, so
        # the quality agrees to a few percent.
        assert sampled.quality == pytest.approx(exact.quality, abs=0.05)

    def test_quality_always_finite_and_clamped(self) -> None:
        ref = _make_mixture(60, seed=11)
        # Same content duplicated with doubled amplitudes: a deliberately BAD
        # approximation whose raw ratio would go negative → clamps to [0, 1].
        bad = GSplatData(
            centers=np.asarray(ref.centers).copy(),
            amplitudes=np.asarray(ref.amplitudes) * 3.0,
            cholesky_factors=np.asarray(ref.cholesky_factors).copy(),
        )
        result = mixture_quality(bad, ref)
        assert 0.0 <= result.quality <= 1.0
        assert np.isfinite(result.l2_sq)
