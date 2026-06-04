"""Tests for the additive-LOD post-processing operator.

Covers ``compute_additive_order`` and ``make_additive_lod`` from
``luxar.gsplats.lod.additive``.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod import compute_additive_order, make_additive_lod
from luxar.gsplats.lod.additive import (
    _build_sparse_gram,
    _residual_energy_curve,
    _self_energy_score,
)

METHODS = ("random", "amplitude", "mass", "self_energy", "spectral", "greedy")


def _make_random_gsplat(n: int = 64, ndim: int = 3, seed: int = 0) -> GSplatData:
    """Random anisotropic Gaussian splats with overlap."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n, ndim)).astype(np.float32) * 1.5
    amplitudes = (np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32)
    tril = ndim * (ndim + 1) // 2
    chol = (rng.standard_normal((n, tril)) * 0.1).astype(np.float32)
    diag_idx = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag_idx] = np.abs(chol[:, diag_idx]) + 0.4
    return GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=chol,
    )


def _make_empty_gsplat(ndim: int = 3) -> GSplatData:
    tril = ndim * (ndim + 1) // 2
    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        cholesky_factors=np.zeros((0, tril), dtype=np.float32),
    )


# ── compute_additive_order ─────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_order_is_permutation(method: str) -> None:
    data = _make_random_gsplat(n=20, ndim=3, seed=1)
    order = compute_additive_order(data, method=method, seed=0)
    assert order.shape == (data.n_splats,)
    assert order.dtype == np.int64
    assert sorted(order.tolist()) == list(range(data.n_splats))


@pytest.mark.parametrize("method", METHODS)
def test_residual_energy_monotone(method: str) -> None:
    """E_k = ||f - g_k||^2 should be non-increasing along any returned order."""
    data = _make_random_gsplat(n=40, ndim=2, seed=2)
    order = compute_additive_order(data, method=method, seed=0)
    gram = _build_sparse_gram(data, sigmas=3.0)
    E = _residual_energy_curve(gram, order)
    diffs = np.diff(E)
    # [Python-R3/B-C1] Residual energy is computed in float32 by the
    # sparse Gram accumulation. The previous `<= 1e-9` tolerance was
    # tighter than float32's natural ULP at the magnitudes involved
    # (typical residual ~ O(1), ULP ~ 1e-7) — a stochastic seed could
    # produce a 1e-8 positive fluctuation and the test would fail
    # spuriously. Loosen to `<= 1e-5` which is two orders of magnitude
    # ABOVE float32 ULP and still kills any meaningful monotonicity
    # regression (a real violation would push diffs into O(0.01)).
    assert (diffs <= 1e-5).all(), (
        f"{method!r}: residual energy not monotone non-increasing; "
        f"max positive diff = {diffs.max():.3e}"
    )


def test_greedy_beats_random_on_auc() -> None:
    data = _make_random_gsplat(n=64, ndim=2, seed=3)
    gram = _build_sparse_gram(data, sigmas=3.0)

    def auc(order: np.ndarray) -> float:
        E = _residual_energy_curve(gram, order)
        if E[0] <= 0.0:
            return 0.0
        return float(E.sum() / (len(E) * E[0]))

    a_random = auc(compute_additive_order(data, method="random", seed=0))
    a_greedy = auc(compute_additive_order(data, method="greedy"))
    assert a_greedy < a_random, (
        f"greedy AUC ({a_greedy:.4f}) should be < random AUC ({a_random:.4f})"
    )


def test_greedy_competitive_with_amplitude() -> None:
    """Greedy should not be worse than amplitude on AUC."""
    data = _make_random_gsplat(n=64, ndim=2, seed=4)
    gram = _build_sparse_gram(data, sigmas=3.0)

    def auc(order: np.ndarray) -> float:
        E = _residual_energy_curve(gram, order)
        return float(E.sum() / (len(E) * E[0])) if E[0] > 0 else 0.0

    a_amp = auc(compute_additive_order(data, method="amplitude"))
    a_greedy = auc(compute_additive_order(data, method="greedy"))
    # Allow a tiny float slack.
    assert a_greedy <= a_amp + 1e-9


def test_invalid_method_raises() -> None:
    data = _make_random_gsplat(n=10)
    with pytest.raises(ValueError):
        compute_additive_order(data, method="not_a_method")  # type: ignore[arg-type]


def test_self_energy_score_matches_closed_form() -> None:
    """The self-energy ranking score should equal a_i^2 * |Σ_i|^{1/2}
    (up to the shared π^{D/2} constant which cancels in any ordering).
    The Gram diagonal includes π^{D/2}, so we reintroduce it for the
    cross-check.
    """
    data = _make_random_gsplat(n=8, ndim=3, seed=5)
    gram = _build_sparse_gram(data, sigmas=3.0)
    diag = gram.diagonal()
    D = data.ndim
    score = _self_energy_score(data) * (np.pi ** (D / 2.0))
    np.testing.assert_allclose(diag, score, rtol=1e-5, atol=1e-9)


# ── make_additive_lod ──────────────────────────────────────────────────


def test_make_additive_lod_equal_count() -> None:
    data = _make_random_gsplat(n=20, ndim=3, seed=6)
    ladder = make_additive_lod(data, n_lods=4)
    assert ladder.n_additive_sublods == 4
    assert ladder.n_splats == data.n_splats
    sizes = [
        ladder.additive_sublod(i).n_splats for i in range(ladder.n_additive_sublods)
    ]
    assert sum(sizes) == data.n_splats
    # Equal-count: at most a 1-splat spread across levels for divisible N.
    assert max(sizes) - min(sizes) <= 1
    # up_to_lod prefix should grow monotonically.
    counts = [
        ladder.additive_prefix(k).n_splats for k in range(ladder.n_additive_sublods)
    ]
    assert all(counts[i] < counts[i + 1] for i in range(len(counts) - 1))


def test_make_additive_lod_explicit_counts() -> None:
    data = _make_random_gsplat(n=20, ndim=3, seed=7)
    ladder = make_additive_lod(data, breakpoints=[5, 10, 15, 20])
    assert ladder.n_additive_sublods == 4
    sizes = [
        ladder.additive_sublod(i).n_splats for i in range(ladder.n_additive_sublods)
    ]
    assert sizes == [5, 5, 5, 5]
    assert ladder.stats["lod_breakpoints_kind"] == "explicit-counts"


def test_make_additive_lod_explicit_counts_partial() -> None:
    """If the largest cumulative count is < N, the resolver appends a final
    cut at N so the ladder always covers all splats."""
    data = _make_random_gsplat(n=20, ndim=3, seed=8)
    ladder = make_additive_lod(data, breakpoints=[5, 10])
    assert ladder.n_additive_sublods == 3
    sizes = [
        ladder.additive_sublod(i).n_splats for i in range(ladder.n_additive_sublods)
    ]
    assert sizes == [5, 5, 10]


def test_make_additive_lod_builds_gram_once(monkeypatch) -> None:
    """M1: greedy + energy breakpoints must build the sparse Gram once, not
    twice (it is the dominant cost)."""
    import luxar.gsplats.lod.additive as additive_mod

    calls = {"n": 0}
    real = additive_mod._build_sparse_gram

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(additive_mod, "_build_sparse_gram", _counting)
    data = _make_random_gsplat(n=40, ndim=2, seed=9)
    make_additive_lod(data, breakpoints=[0.5, 0.9, 1.0], method="greedy")
    assert calls["n"] == 1


def test_make_additive_lod_energy_fractions() -> None:
    data = _make_random_gsplat(n=40, ndim=2, seed=9)
    fracs = [0.5, 0.9, 1.0]
    ladder = make_additive_lod(data, breakpoints=fracs, method="greedy")
    assert ladder.stats["lod_breakpoints_kind"] == "energy-fractions"
    assert ladder.n_additive_sublods >= 1

    # Re-derive the residual-energy curve and confirm each cumulative cut
    # achieves the target fraction.
    order = compute_additive_order(data, method="greedy")
    gram = _build_sparse_gram(data, sigmas=3.0)
    E = _residual_energy_curve(gram, order)
    E_total = E[0]
    cuts = ladder.stats["lod_cutpoints"]
    # First cumulative cut should hit ≥ 50% utility, last should hit ≥ 100%.
    U_at_cut = (E_total - E[cuts[0]]) / E_total
    assert U_at_cut >= 0.5 - 1e-6
    assert cuts[-1] == data.n_splats


def test_make_additive_lod_invalid_breakpoints() -> None:
    data = _make_random_gsplat(n=20)
    with pytest.raises(ValueError):
        make_additive_lod(data, breakpoints=[10, 5, 15])  # not increasing
    with pytest.raises(ValueError):
        make_additive_lod(data, breakpoints=[1.5])  # > 1
    with pytest.raises(TypeError):
        make_additive_lod(data, breakpoints=[1, 2.0])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        make_additive_lod(data, breakpoints=[100])  # exceeds N


def test_make_additive_lod_empty() -> None:
    data = _make_empty_gsplat(ndim=3)
    ladder = make_additive_lod(data, n_lods=4)
    # Empty input collapses to a single (empty) LOD.
    assert ladder.n_splats == 0
    assert ladder.n_additive_sublods == 1


def test_make_additive_lod_n_lods_exceeds_n() -> None:
    """When n_lods > N, we clamp to N (one splat per LOD)."""
    data = _make_random_gsplat(n=3, ndim=2, seed=10)
    ladder = make_additive_lod(data, n_lods=10)
    assert ladder.n_additive_sublods == 3
    assert ladder.n_splats == 3


def test_lod_stats_recorded() -> None:
    data = _make_random_gsplat(n=20, ndim=3, seed=11)
    ladder = make_additive_lod(data, n_lods=4, method="self_energy")
    assert ladder.stats["lod_method"] == "self_energy"
    assert ladder.stats["lod_n_lods"] == 4
    assert ladder.stats["lod_breakpoints_kind"] == "equal-count"
    assert "lod_cutpoints" in ladder.stats
    # Per-LOD stats:
    for level in range(ladder.n_additive_sublods):
        lod_stats = ladder.additive_sublod(level).stats
        assert lod_stats["lod_method"] == "self_energy"
        assert lod_stats["lod_level"] == level


def test_make_additive_lod_substitutive_level_arg() -> None:
    """`substitutive_level` selects which substitutive level receives the new ladder."""
    from luxar.gsplats.lod import make_substitutive_lod

    data = _make_random_gsplat(n=32, ndim=3, seed=12)
    pyr = make_substitutive_lod(
        data,
        compression_factor=4,
        levels=2,
        method="kmeans_lloyd",
        device="cpu",
        seed=0,
    )
    # Pyramid has 3 substitutive levels, each with M=1 additive sub-LOD.
    assert pyr.n_substitutive == 3
    for s in range(3):
        assert pyr.substitutive_levels[s].n_additive_lods == 1

    # Build an additive ladder on substitutive level 1 only.
    ladded = make_additive_lod(
        pyr,
        n_lods=3,
        method="self_energy",
        substitutive_level=1,
    )
    # Same n_substitutive, level 1 has 3 sub-LODs, others unchanged.
    assert ladded.n_substitutive == 3
    assert ladded.substitutive_levels[0].n_additive_lods == 1
    assert ladded.substitutive_levels[1].n_additive_lods <= 3
    assert ladded.substitutive_levels[2].n_additive_lods == 1
    # Total splats per level preserved
    for s in range(3):
        assert (
            ladded.substitutive_levels[s].n_splats_total
            == pyr.substitutive_levels[s].n_splats_total
        )


def test_make_additive_lod_substitutive_level_out_of_bounds() -> None:
    data = _make_random_gsplat(n=8, ndim=3, seed=13)
    with pytest.raises(ValueError, match="out of bounds"):
        make_additive_lod(data, n_lods=2, substitutive_level=2)


def test_make_lod_pyramid_full_matrix() -> None:
    """`make_lod_pyramid` produces a [levels+1, n_additive_lods] matrix."""
    from luxar.gsplats.lod import make_lod_pyramid

    data = _make_random_gsplat(n=64, ndim=3, seed=14)
    pyr = make_lod_pyramid(
        data,
        compression_factor=4,
        levels=2,
        substitutive_method="kmeans_lloyd",
        n_additive_lods=3,
        additive_method="self_energy",
        device="cpu",
        seed=0,
    )
    assert pyr.n_substitutive == 3
    # Each substitutive level has its own additive ladder (subject to clamping).
    for s in range(3):
        lev = pyr.substitutive_levels[s]
        assert lev.n_additive_lods >= 1
        # Stats propagated:
        assert lev.compression_factor == 4**s
        assert lev.level_index == s
        if s == 0:
            assert lev.parent_method is None
        else:
            assert lev.parent_method == "kmeans_lloyd"


def test_make_lod_pyramid_defaults_to_auto_substitutive_method() -> None:
    """The library default for `substitutive_method` is `auto` (matches
    `make_substitutive_lod`), not the legacy `kmeans_lloyd`."""
    import inspect

    from luxar.gsplats.lod import make_lod_pyramid

    assert inspect.signature(make_lod_pyramid).parameters["substitutive_method"].default == "auto"

    # And it runs end-to-end without an explicit method (resolves via `auto`).
    data = _make_random_gsplat(n=64, ndim=3, seed=21)
    pyr = make_lod_pyramid(data, compression_factor=4, levels=1, n_additive_lods=2, device="cpu")
    assert pyr.n_substitutive == 2
