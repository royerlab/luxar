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
    _radial_score,
    _residual_energy_curve,
    _self_energy_score,
)
from luxar.utils.lod_methods import GSPLAT_ADDITIVE_METHODS

#: DERIVED from the registry, not hand-listed. This was an opt-in allowlist, and
#: a method missing from it got zero coverage from the four parametrized property
#: tests below — silently. Deriving it means a new ordering method is covered the
#: moment it is registered. ``GSPLAT_ADDITIVE_METHODS`` excludes the ``auto`` sentinel,
#: which is exactly what these tests want (they pass a concrete method).
METHODS = GSPLAT_ADDITIVE_METHODS


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
    # [P2] A permutation check alone is satisfied by a no-op that returns
    # ``arange(N)`` unchanged. On this non-degenerate random data every
    # method genuinely reorders, so require the result to differ from the
    # trivial identity — this kills a mutant that skips the ordering.
    assert order.tolist() != list(range(data.n_splats)), (
        f"{method!r} returned the identity order; ranking is a no-op"
    )


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


# ── method='auto' size-adaptive resolution (perf guard) ────────────────────


def test_resolve_additive_method_threshold() -> None:
    """``auto`` picks greedy at/below the threshold and self_energy above it.

    Mirrors substitutive LOD's ``_resolve_method``. Guards the perf fix: a
    large additive part must NOT silently fall into greedy's O(N·nnz·logN)
    lazy-heap path (which hung ~2 h on a 1.5 M-splat part).
    """
    from luxar.gsplats.lod.additive import (
        _AUTO_ADDITIVE_MAX_N,
        resolve_additive_method,
    )

    assert resolve_additive_method("auto", 1) == "greedy"
    assert resolve_additive_method("auto", _AUTO_ADDITIVE_MAX_N) == "greedy"
    assert resolve_additive_method("auto", _AUTO_ADDITIVE_MAX_N + 1) == "self_energy"
    assert resolve_additive_method("auto", 10_000_000) == "self_energy"
    # A concrete method always passes through unchanged (no override).
    assert resolve_additive_method("greedy", 10_000_000) == "greedy"
    assert resolve_additive_method("self_energy", 1) == "self_energy"


# ── stream:<c> breakpoints (bandwidth-derived streaming ladder) ─────────────


def test_streaming_chunk_splats_math() -> None:
    """200 ms @ 25 Mbps @ 45 B/splat → ~13.9 k splats; validation raises."""
    from luxar.gsplats.lod.additive import streaming_chunk_splats

    assert streaming_chunk_splats(200, 25, 45.0) == 13889
    assert streaming_chunk_splats(1000, 8, 45.0) == round(8 * 125_000 / 45.0)
    assert streaming_chunk_splats(1, 0.1, 1e9) == 1  # floor at 1
    for bad in [(0, 25, 45), (200, 0, 45), (200, 25, 0)]:
        with pytest.raises(ValueError):
            streaming_chunk_splats(*bad)


def test_stream_breakpoints_geometric_cuts() -> None:
    """Geometric cumulative cuts [c, 2c, 4c, …, N]; kind == 'stream'."""
    from luxar.gsplats.lod.additive import _resolve_breakpoints

    cuts, kind = _resolve_breakpoints(23_368_376, 4, "stream:14000")
    assert kind == "stream"
    assert cuts[0] == 14000
    assert cuts[-1] == 23_368_376
    # Doubling schedule: each interior cut is 2x the previous.
    assert all(cuts[i + 1] == 2 * cuts[i] for i in range(len(cuts) - 2))


def test_stream_breakpoints_small_n_clamps_silently() -> None:
    """n <= c → single level [n] — never raises (unlike explicit counts)."""
    from luxar.gsplats.lod.additive import _resolve_breakpoints

    assert _resolve_breakpoints(500, 4, "stream:14000") == ([500], "stream")
    assert _resolve_breakpoints(14000, 4, "stream:14000") == ([14000], "stream")
    assert _resolve_breakpoints(1, 4, "stream:14000") == ([1], "stream")


def test_stream_breakpoints_sliver_tail_folds() -> None:
    """A final increment < c/2 folds into the previous cut (no 1-splat levels)."""
    from luxar.gsplats.lod.additive import _resolve_breakpoints

    # tail of 1 (< 7000) folds → single level
    assert _resolve_breakpoints(14001, 4, "stream:14000") == ([14001], "stream")
    # healthy tail (13000 >= 7000) kept
    assert _resolve_breakpoints(27000, 4, "stream:14000") == (
        [14000, 27000],
        "stream",
    )


def test_stream_breakpoints_level_cap() -> None:
    """The doubling schedule is capped; the last cut jumps straight to N."""
    from luxar.gsplats.lod.additive import (
        DEFAULT_STREAM_MAX_LEVELS,
        _resolve_breakpoints,
    )

    cuts, _ = _resolve_breakpoints(10**9, 4, "stream:1")
    assert len(cuts) <= DEFAULT_STREAM_MAX_LEVELS
    assert cuts[-1] == 10**9


@pytest.mark.parametrize(
    "bad", ["stream:", "stream:0", "stream:-5", "stream:14000.5", "stream:abc"]
)
def test_stream_breakpoints_invalid_payloads_raise(bad: str) -> None:
    from luxar.gsplats.lod.additive import _resolve_breakpoints

    with pytest.raises(ValueError):
        _resolve_breakpoints(100, 4, bad)


def test_make_additive_lod_stream_no_gram_for_score_methods(monkeypatch) -> None:
    """stream + a score method must build ZERO Gram matrices (unlike energy
    fractions, which need the residual-energy curve)."""
    import luxar.gsplats.lod.additive as additive_mod

    called = {"gram": 0}
    real = additive_mod._build_sparse_gram

    def _spy(*args, **kwargs):
        called["gram"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(additive_mod, "_build_sparse_gram", _spy)
    data = _make_random_gsplat(n=64, seed=7)
    out = additive_mod.make_additive_lod(
        data, method="self_energy", breakpoints="stream:20"
    )
    assert called["gram"] == 0
    incs = [s.n_splats for s in out.additive_sublods]
    assert sum(incs) == 64 and incs[0] == 20


def test_make_additive_lod_stream_provenance_stat() -> None:
    """Each stream sub-LOD records kind='stream' + the first-chunk size."""
    data = _make_random_gsplat(n=64, seed=8)
    out = make_additive_lod(data, method="self_energy", breakpoints="stream:20")
    for sub in out.additive_sublods:
        assert sub.stats["lod_breakpoints_kind"] == "stream"
        assert sub.stats["lod_stream_chunk_splats"] == 20


# ── pre-existing-issue regression tests ─────────────────────────────────────


def test_clamp_counts_breakpoints_helper() -> None:
    """Explicit counts clamp to a small part's N; other specs pass through."""
    from luxar.gsplats.lod.additive import clamp_counts_breakpoints

    assert clamp_counts_breakpoints([500, 2000, 10000], 800) == [500]
    assert clamp_counts_breakpoints([500, 2000], 100) == [100]
    assert clamp_counts_breakpoints([500, 2000], 10_000) == [500, 2000]
    assert clamp_counts_breakpoints("stream:14000", 100) == "stream:14000"
    assert clamp_counts_breakpoints("equal-count", 100) == "equal-count"
    assert clamp_counts_breakpoints([0.5, 0.9], 100) == [0.5, 0.9]


def test_validate_counts_breakpoints_helper() -> None:
    """Typo-scale counts abort loudly against the FULL dataset N (the strict
    companion of the per-part/per-level clamp); non-count specs pass through."""
    from luxar.gsplats.lod.additive import validate_counts_breakpoints

    with pytest.raises(ValueError, match="exceeds N=800"):
        validate_counts_breakpoints([500, 2000, 10000], 800)
    # <= N is fine (== N allowed; _resolve_breakpoints handles the final cut).
    validate_counts_breakpoints([500, 2000], 2000)
    validate_counts_breakpoints([500, 2000], 10_000)
    # Non-count specs are size-adaptive / validated downstream — never raise.
    validate_counts_breakpoints("stream:14000", 100)
    validate_counts_breakpoints("equal-count", 100)
    validate_counts_breakpoints([0.5, 0.9], 100)
    validate_counts_breakpoints([500], 0)  # degenerate n: defer to downstream


def test_stream_breakpoints_malformed_payload_message() -> None:
    """The malformed-payload error reads clearly (was the garbled
    "must be 'stream:<int>=1>'")."""
    from luxar.gsplats.lod.additive import _resolve_breakpoints

    with pytest.raises(ValueError, match=r"must be 'stream:<c>' with integer c >= 1"):
        _resolve_breakpoints(100, 4, "stream:abc")


def test_bool_breakpoints_rejected() -> None:
    """bool is an int subclass — [True, False] must not pass as counts."""
    from luxar.gsplats.lod.additive import _resolve_breakpoints

    with pytest.raises((TypeError, ValueError)):
        _resolve_breakpoints(100, 4, [True, False])


def test_empty_leaf_kind_labeled_none() -> None:
    """The n==0 fast path labels the kind 'none' (was mislabeled
    'equal-count' regardless of the requested spec)."""
    data = _make_empty_gsplat(ndim=3)
    out = make_additive_lod(data, breakpoints="stream:100")
    assert out.stats["lod_breakpoints_kind"] == "none"


def test_compute_additive_order_auto_resolves_to_greedy_at_small_n() -> None:
    """At small N, ``auto`` produces the SAME ordering as explicit greedy."""
    data = _make_random_gsplat(n=64, seed=3)
    auto = compute_additive_order(data, method="auto")
    greedy = compute_additive_order(data, method="greedy")
    assert auto.tolist() == greedy.tolist()


def test_compute_additive_order_auto_avoids_greedy_above_threshold(monkeypatch) -> None:
    """Above the threshold, ``auto`` must route to self_energy — NOT greedy.

    Proven by patching the threshold below N and asserting the Gram (built only
    by greedy/spectral) is never touched. Pre-fix (no resolver) this raised on
    an unknown method or ran greedy; either way this test fails without the fix.
    """
    import luxar.gsplats.lod.additive as additive_mod

    monkeypatch.setattr(additive_mod, "_AUTO_ADDITIVE_MAX_N", 8)

    called = {"gram": 0}
    real_gram = additive_mod._build_sparse_gram

    def _spy(*args, **kwargs):
        called["gram"] += 1
        return real_gram(*args, **kwargs)

    monkeypatch.setattr(additive_mod, "_build_sparse_gram", _spy)

    data = _make_random_gsplat(n=64, seed=5)  # 64 > patched threshold of 8
    auto = additive_mod.compute_additive_order(data, method="auto")
    self_energy = additive_mod.compute_additive_order(data, method="self_energy")
    # Routed to the cheap O(N log N) score path — Gram never built.
    assert called["gram"] == 0
    assert auto.tolist() == self_energy.tolist()


@pytest.mark.parametrize("method", METHODS)
def test_order_empty_returns_empty(method: str) -> None:
    """[P5] N=0 boundary: every method returns an empty int64 array."""
    data = _make_empty_gsplat(ndim=3)
    order = compute_additive_order(data, method=method, seed=0)
    assert order.shape == (0,)
    assert order.dtype == np.int64


@pytest.mark.parametrize("method", METHODS)
def test_order_single_splat_returns_zero(method: str) -> None:
    """[P5] N=1 boundary: the only valid ordering is ``[0]``."""
    data = _make_random_gsplat(n=1, ndim=3, seed=99)
    order = compute_additive_order(data, method=method, seed=0)
    assert order.tolist() == [0]
    assert order.dtype == np.int64


def test_greedy_lazy_path_matches_dense() -> None:
    """[P11] The lazy/sparse greedy branch (``N > max_n_dense``) is never
    exercised by the other tests, which all sit below the 2000 default.
    Force it with a tiny ``max_n_dense`` and verify it returns the same
    valid permutation as the dense scan-greedy path."""
    data = _make_random_gsplat(n=20, ndim=3, seed=17)
    dense = compute_additive_order(data, method="greedy", max_n_dense=2_000)
    lazy = compute_additive_order(data, method="greedy", max_n_dense=4)
    assert sorted(lazy.tolist()) == list(range(data.n_splats))
    # Both branches implement the same greedy criterion → identical order.
    assert lazy.tolist() == dense.tolist()


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
    # [P2] The prefix counts must form an exact partition of all N splats:
    # the final prefix covers everything, and each prefix equals the
    # running cumulative of the per-LOD sizes (no gaps, no double-counting).
    # A bare ``sum(sizes) == N`` would still pass if an intermediate LOD
    # were duplicated or dropped; the cumulative-equality check would not.
    assert counts[-1] == data.n_splats
    expected_cumulative = np.cumsum(sizes).tolist()
    assert counts == expected_cumulative


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


def test_make_additive_lod_score_energy_fractions_builds_no_gram(monkeypatch) -> None:
    """Score-ordered (self_energy) + energy breakpoints must resolve cuts against
    the O(N) self-energy cumulative — never the O(N²) sparse Gram. Coarse lifted
    children are maximally-overlapping blobs, the worst case for the Gram, so
    ``gsplat_additive_lod_from`` pins ``method='self_energy'`` and this path must
    not build a Gram at all."""
    import luxar.gsplats.lod.additive as additive_mod

    calls = {"n": 0}
    real = additive_mod._build_sparse_gram

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(additive_mod, "_build_sparse_gram", _counting)
    data = _make_random_gsplat(n=40, ndim=2, seed=9)
    ladder = make_additive_lod(data, breakpoints=[0.5, 0.9, 1.0], method="self_energy")
    assert calls["n"] == 0
    assert ladder.stats["lod_breakpoints_kind"] == "energy-fractions"
    assert ladder.stats["lod_cutpoints"][-1] == data.n_splats
    assert ladder.n_additive_sublods >= 1


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
    # [P2/P12] The residual energy at successive cut indices must be
    # monotonically non-increasing (each LOD captures strictly more
    # energy). A lower-bound-only check on the first cut would not catch
    # a resolver that places later cuts at a *higher* residual.
    residual_at_cuts = [E[c] for c in cuts]
    assert all(
        residual_at_cuts[i] >= residual_at_cuts[i + 1] - 1e-5
        for i in range(len(residual_at_cuts) - 1)
    ), f"residual energy not non-increasing at cuts: {residual_at_cuts}"


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


def test_energy_fraction_cum_stamped_monotone_and_complete() -> None:
    """The Q·e quality stamps: e(k) per sub-LOD, w on the level stats.

    e(k) must be monotone increasing over the ladder with e(last) == 1.0
    (the fraction of the leaf's total self-energy committed by the prefix),
    and the level carries the absolute reference_energy w matching the
    closed-form total self-energy of the whole leaf.
    """
    from luxar.gsplats.lod.quality import total_self_energy

    data = _make_random_gsplat(n=64, ndim=3, seed=13)
    for breakpoints in ("equal-count", "stream:8"):
        ladder = make_additive_lod(
            data, n_lods=4, method="self_energy", breakpoints=breakpoints
        )
        fracs = [
            ladder.additive_sublod(k).stats["energy_fraction_cum"]
            for k in range(ladder.n_additive_sublods)
        ]
        assert all(0.0 < f <= 1.0 for f in fracs)
        assert fracs == sorted(fracs)  # cumulative → monotone
        assert fracs[-1] == pytest.approx(1.0)
        # Self-energy ordering front-loads energy: the first sub-LOD holds
        # MORE than its count share.
        first = ladder.additive_sublod(0)
        assert fracs[0] > first.stats["lod_cumulative_n"] / ladder.n_splats
        # The level's absolute weight matches the closed-form total.
        w = ladder.substitutive_levels[0].stats["reference_energy"]
        assert w == pytest.approx(total_self_energy(data.flattened()), rel=1e-6)


def test_energy_fraction_cum_on_empty_leaf() -> None:
    data = _make_empty_gsplat(ndim=3)
    ladder = make_additive_lod(data, n_lods=4)
    assert ladder.additive_sublod(0).stats["energy_fraction_cum"] == 1.0
    assert ladder.substitutive_levels[0].stats["reference_energy"] == 0.0


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


# ── sibling-aware stream ladders (upgrade catch-up geometry) ────────────────


def test_sibling_aware_stream_breakpoints_helper() -> None:
    """`stream:C` bases are raised to ceil(n / 2K) for leaves that have a
    coarser sibling; every other spec passes through untouched."""
    from luxar.gsplats.lod.additive import sibling_aware_stream_breakpoints as saw

    # Raised to ceil(leaf_n / (2·K)) when that exceeds the user base.
    assert saw("stream:8", 512, 4) == "stream:64"
    # The user's base wins when it is already larger.
    assert saw("stream:100", 512, 4) == "stream:100"
    # Compression factor floors at 2 (a degenerate K=1 group must not force
    # the whole leaf into the first chunk).
    assert saw("stream:1", 512, 1) == "stream:128"
    # Non-stream specs pass through untouched (no shared-base pathology).
    assert saw("equal-count", 512, 4) == "equal-count"
    assert saw([10, 20], 512, 4) == [10, 20]
    assert saw("energy:0.5,1.0", 512, 4) == "energy:0.5,1.0"
    # Malformed stream payloads pass through for the validator to reject.
    assert saw("stream:abc", 512, 4) == "stream:abc"


# ── radial (concentric-shell reveal) ordering ──────────────────────────


def _ray_gsplat(radii: np.ndarray, ndim: int = 3, offset: float = 0.0) -> GSplatData:
    """Splats along one axis at the given radii, optionally re-origined.

    A ray rather than a sphere keeps the expected order unambiguous: distance
    from the set's own bbox centre is monotone in ``radii``.
    """
    n = radii.size
    centers = np.zeros((n, ndim), dtype=np.float32)
    centers[:, 0] = radii
    centers += np.float32(offset)
    tril = ndim * (ndim + 1) // 2
    chol = np.zeros((n, tril), dtype=np.float32)
    diag_idx = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag_idx] = 0.4
    return GSplatData(
        centers=centers,
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )


def test_radial_orders_innermost_first():
    """Prefixes grow OUTWARD: the ordering is ascending in distance."""
    # Shuffled input, so a pass cannot come from input order alone.
    radii = np.array([5.0, 1.0, 4.0, 2.0, 3.0], dtype=np.float32)
    data = _ray_gsplat(radii)

    order = compute_additive_order(data, method="radial")

    # Centred on the ray's own midpoint (r=3), so distance is |r - 3|.
    assert radii[order[0]] == pytest.approx(3.0)
    assert sorted(radii[order[1:3]]) == pytest.approx([2.0, 4.0])
    assert sorted(radii[order[3:]]) == pytest.approx([1.0, 5.0])


def test_radial_centre_defaults_to_bbox_not_scene_origin():
    """A far-from-origin set reveals from ITS OWN middle, not from a corner."""
    radii = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    here = _ray_gsplat(radii)
    far = _ray_gsplat(radii, offset=1000.0)

    far_order = compute_additive_order(far, method="radial")

    # Pin the far set's ORDER, not just its agreement with the near one: both
    # rays are monotone in r, so a scene-origin default reveals each of them
    # strictly left-to-right and the two still agree. Only the absolute order
    # separates the two defaults — from the middle out, r=3 first.
    assert list(radii[far_order]) == pytest.approx([3.0, 2.0, 4.0, 1.0, 5.0])
    # And the 1000-unit translation is irrelevant: the centre travels with the
    # data, so the near set ranks identically.
    assert np.array_equal(compute_additive_order(here, method="radial"), far_order)


def test_radial_explicit_centre_overrides_the_bbox():
    """An explicit centre re-aims the shells."""
    radii = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    data = _ray_gsplat(radii)

    order = compute_additive_order(data, method="radial", reveal_centre=[1.0, 0.0, 0.0])

    # Aimed at the near end, so it reveals strictly outward from r=1.
    assert list(radii[order]) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_radial_ignores_a_zero_variance_time_axis():
    """A stacked-time axis must not become a shell dimension.

    Two timepoints of the same 3D ray. The centre is r=2, so BOTH timepoints of
    r=2 sit at distance 0 and must come first; had the degenerate 4th axis
    entered the distance, the t=1 copy would be at distance 1 and be displaced
    by the t=0 copies of r=1 / r=3.
    """
    radii = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    n = radii.size
    centers = np.zeros((2 * n, 4), dtype=np.float32)
    centers[:n, 0] = radii
    centers[n:, 0] = radii
    centers[n:, 3] = 1.0  # timepoint 1
    chol = np.zeros((2 * n, 10), dtype=np.float32)
    diag_idx = np.cumsum(np.arange(1, 5)) - 1
    chol[:, diag_idx] = 0.4
    chol[:, diag_idx[3]] = 0.0  # degenerate time axis
    data = GSplatData(
        centers=centers,
        amplitudes=np.ones(2 * n, dtype=np.float32),
        cholesky_factors=chol,
    )

    order = compute_additive_order(data, method="radial")
    ranked_r = centers[order, 0]
    ranked_t = centers[order, 3]

    assert list(ranked_r[:2]) == pytest.approx([2.0, 2.0])
    assert sorted(ranked_t[:2]) == pytest.approx([0.0, 1.0])
    # The rest is the distance-1 shell: both timepoints of r=1 and r=3. They
    # TIE, so their relative order is input order — deliberately not asserted.
    assert sorted(ranked_r[2:]) == pytest.approx([1.0, 1.0, 3.0, 3.0])


def test_radial_still_reveals_when_every_axis_is_degenerate():
    """All-zero covariance must not leave the shell axes EMPTY.

    The default shell axes come from ``_nondegenerate_axes``, and an empty
    selection would score every splat 0.0 — a ladder silently emitted in input
    order. It cannot happen because that helper falls back to ALL axes when
    nothing clears the sigma threshold, but the ordering depends on a default
    two modules away, so pin it here.
    """
    radii = np.array([5.0, 1.0, 4.0, 2.0, 3.0], dtype=np.float32)
    centers = np.zeros((radii.size, 3), dtype=np.float32)
    centers[:, 0] = radii
    data = GSplatData(
        centers=centers,
        amplitudes=np.ones(radii.size, dtype=np.float32),
        cholesky_factors=np.zeros((radii.size, 6), dtype=np.float32),
    )

    order = compute_additive_order(data, method="radial")

    # Same expectation as the non-degenerate ray: centred on r=3, |r - 3|.
    assert radii[order[0]] == pytest.approx(3.0)
    assert sorted(radii[order[1:3]]) == pytest.approx([2.0, 4.0])
    assert sorted(radii[order[3:]]) == pytest.approx([1.0, 5.0])


def test_radial_rejects_a_mis_shaped_centre():
    data = _ray_gsplat(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="one coordinate per spatial axis"):
        compute_additive_order(data, method="radial", reveal_centre=[0.0, 0.0])


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
def test_radial_rejects_a_non_finite_centre(bad: float) -> None:
    """Same class as an empty `spatial_dims`: a silent no-op, not a reveal.

    Every distance comes back non-finite, they all compare equal under the stable
    argsort, and the ladder is emitted in INPUT order with nothing to indicate the
    centre was junk.
    """
    data = _make_random_gsplat(n=8, ndim=3, seed=4)
    with pytest.raises(ValueError, match="must be finite"):
        compute_additive_order(data, method="radial", reveal_centre=[bad, 0.0, 0.0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_radial_refuses_non_finite_centers(bad: float) -> None:
    """A non-finite CENTER coordinate is the same silent no-op as a bad knob.

    Measured before the guard: `compute_additive_order(..., method="radial")`
    returned the identity permutation — every distance non-finite, all equal under
    the stable argsort, ladder emitted in INPUT order. The element side behaved the
    same way and Lines raised a misleading `reveal_centre` error, so the three
    implementations of one ordering disagreed about malformed data. Now they share
    `validate_finite_reveal_coords` and each names its own array.
    """
    data = _ray_gsplat(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    centers = np.array(data.centers, dtype=np.float32, copy=True)
    centers[2, 0] = bad
    data.centers = centers

    with pytest.raises(ValueError, match="centers must be finite"):
        compute_additive_order(data, method="radial")


def test_radial_scorer_handles_an_empty_dataset_like_its_element_twin() -> None:
    """`_radial_score` must not raise on an empty input, for symmetry.

    Scope, stated honestly because it is narrower than it looks: the public
    `compute_additive_order` ALREADY short-circuits an empty dataset before it
    dispatches here, so it returns an empty permutation with or without this
    guard (asserted below as the boundary of the claim). The defect was in the
    HELPER: called directly it raised out of the default-centre bbox reduction,
    while the element-side `radial_element_score` — the other implementation of the
    same ordering — has always returned an empty score. This pins the two
    together, so the mutation that fails is removing the helper's guard, NOT
    breaking a user-visible path.
    """
    from luxar.core.group.lod.reveal import radial_element_score

    empty = _make_empty_gsplat(ndim=3)

    # The helper itself — the surface that was actually broken.
    assert _radial_score(empty).shape == (0,)
    # Its element-side twin, which already agreed.
    assert radial_element_score(np.empty((0, 3), dtype=np.float64)).shape == (0,)
    # Boundary of the claim: the public entry point was never affected.
    assert compute_additive_order(empty, method="radial").shape == (0,)


def _sublod_stats(laddered: GSplatData) -> list[dict]:
    """`lod_stats` of every additive sub-LOD of the (single) substitutive level."""
    return [dict(s.stats) for s in laddered.substitutive_levels[0].additive_sublods]


def test_radial_ladder_carries_no_energy_stamps():
    """A reveal must not be brightened by the viewer's 1/e(k) compensation.

    `energyCompensation` is gated on the BLENDING MODE, not on geometry type, so a
    stamped radial ladder would blow out the inner shell (~20x for a 5% first
    shell) and dim as the object completes — the inverse of growing outward.
    Omitting the stamp makes `energyCompensation(undefined)` return exactly 1.
    """
    data = _ray_gsplat(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32))

    laddered = make_additive_lod(data, n_lods=3, method="radial")

    stats = _sublod_stats(laddered)
    assert len(stats) == 3
    for i, s in enumerate(stats):
        assert "energy_fraction_cum" not in s, f"sub-LOD {i} carries an energy stamp"
        # The non-energy provenance must survive — this is not a blanket wipe.
        assert s["lod_method"] == "radial"
        assert s["lod_n_splats"] > 0
    # Both-or-neither: the leaf weight goes too, or the pair is half-written.
    assert "reference_energy" not in laddered.substitutive_levels[0].stats


def test_non_reveal_ladders_still_carry_energy_stamps():
    """The sensitivity control: suppression is scoped to reveal methods only.

    Without this, a bug that dropped stamps for EVERY method would pass the test
    above while silently disabling cross-fade and energy compensation everywhere.
    """
    data = _ray_gsplat(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32))

    laddered = make_additive_lod(data, n_lods=3, method="mass")

    stats = _sublod_stats(laddered)
    assert [("energy_fraction_cum" in s) for s in stats] == [True, True, True]
    assert "reference_energy" in laddered.substitutive_levels[0].stats


def test_radial_erases_an_inherited_reference_energy():
    """The weight must GO, not merely not be re-added.

    `merged_level_stats` inherits the input level's stats, so a weight is usually
    already there: a substitutive build stamps one per level (so
    `--recipe levels/adaptive -m radial` hits this on every level), and so does
    `gsplat additive` over an annotated tree. Skipping the `setdefault` alone left
    the ladder half-stamped — no per-level `energy_fraction_cum`, but the leaf
    weight the viewer pairs it with still on disk.
    """
    from luxar.gsplats.gsplat_data import SubstitutiveLevel

    data = _ray_gsplat(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32))
    level = data.substitutive_levels[0]
    pre_stamped = GSplatData(
        substitutive_levels=[
            SubstitutiveLevel(
                additive_sublods=level.additive_sublods,
                compression_factor=level.compression_factor,
                parent_method=level.parent_method,
                level_index=level.level_index,
                stats={**level.stats, "reference_energy": 1234.5},
            )
        ],
        stats=dict(data.stats),
    )

    laddered = make_additive_lod(pre_stamped, n_lods=3, method="radial")
    assert "reference_energy" not in laddered.substitutive_levels[0].stats

    # Control: a non-reveal rebuild keeps the inherited weight (setdefault wins,
    # so the group-consistent value from a substitutive build is not clobbered).
    kept = make_additive_lod(pre_stamped, n_lods=3, method="mass")
    assert kept.substitutive_levels[0].stats["reference_energy"] == 1234.5


def test_empty_radial_ladder_keeps_both_halves_of_the_pair():
    """An empty leaf labels itself `lod_method="none"` and stays fully stamped.

    Nothing streams, so e(k)=1.0 makes the 1/e(k) compensation exactly 1 — there
    is no reveal to protect. Dropping only the leaf weight there would half-write
    the pair in the other direction, and `annotate-quality` (which mirrors the
    build for an empty leaf) would then disagree with it.
    """
    laddered = make_additive_lod(_make_empty_gsplat(), n_lods=3, method="radial")

    stats = _sublod_stats(laddered)
    assert [s["energy_fraction_cum"] for s in stats] == [1.0]
    assert "reference_energy" in laddered.substitutive_levels[0].stats


@pytest.mark.parametrize(
    ("dims", "match"),
    [
        ([], "non-empty"),
        ([-1], "non-negative"),
        ([0, 0], "must not repeat"),
        ([0, 7], "out of range"),
        ([1.9], "integer column indices"),
    ],
    ids=["empty", "negative", "duplicate", "out-of-range", "fractional"],
)
def test_radial_rejects_malformed_spatial_dims(dims: list[int], match: str) -> None:
    """Each of these silently produced a WRONG ordering before being rejected.

    A negative index ALIASES to another column under numpy indexing, a repeat
    DOUBLE-COUNTS that axis in the distance, an empty selection scores every
    splat 0.0 — degrading the ladder to input order with nothing to show it — and
    `np.asarray([1.9], dtype=np.intp)` TRUNCATES to axis 1, measuring a different
    column than the caller named. The element-side scorer rejects the same five;
    the CLI bounds-checks too, but the Python API reaches here directly.
    """
    data = _make_random_gsplat(n=8, ndim=3, seed=3)
    with pytest.raises(ValueError, match=match):
        compute_additive_order(data, method="radial", spatial_dims=dims)


def test_radial_excludes_an_asymmetric_degenerate_time_axis() -> None:
    """Default shell axes are the non-degenerate (real-sigma) ones, so a stacked
    time axis is excluded EVEN when it is asymmetric enough to reorder — the
    discriminating case #1452 flagged (the symmetric two-timepoint fixture
    cancels and cannot catch a `dims = arange(ndim)` regression).

    Three timepoints t in {0, 1, 5} at the SAME spatial point (real spatial
    sigma, zero time sigma). Spatial-only, all three tie at spatial distance 0
    and the stable sort keeps input order [0, 1, 2]. Had the degenerate time
    axis entered the distance, its bbox centre would sit at t=2.5 and reorder
    them (t=1 nearest -> [1, 0, 2]).
    """
    t = np.array([0.0, 1.0, 5.0], dtype=np.float32)
    n = t.size
    centers = np.zeros((n, 4), dtype=np.float32)
    centers[:, 0] = 2.0  # identical spatial coords (a single point)
    centers[:, 3] = t
    diag_idx = np.cumsum(np.arange(1, 5)) - 1
    chol = np.zeros((n, 10), dtype=np.float32)
    chol[:, diag_idx] = 0.4  # non-degenerate spatial sigma
    chol[:, diag_idx[3]] = 0.0  # degenerate (stacked) time axis
    data = GSplatData(
        centers=centers,
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )

    order = compute_additive_order(data, method="radial")

    assert list(order) == [0, 1, 2]


def test_radial_spatial_dims_override_selects_the_shell_axes() -> None:
    """`spatial_dims=` overrides the default selection: a ray that varies only
    along axis 3 is ordered by axis 3 when it is named explicitly, and
    `reveal_centre` carries one coordinate PER SELECTED axis (not per ndim)."""
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    n = r.size
    centers = np.zeros((n, 4), dtype=np.float32)
    centers[:, 3] = r  # variation only along axis 3
    diag_idx = np.cumsum(np.arange(1, 5)) - 1
    chol = np.zeros((n, 10), dtype=np.float32)
    chol[:, diag_idx] = 0.4
    data = GSplatData(
        centers=centers,
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )

    order = compute_additive_order(
        data, method="radial", spatial_dims=[3], reveal_centre=[1.0]
    )

    # Aimed at r=1 along the selected axis -> reveals strictly outward.
    assert list(r[order]) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_radial_reveal_centre_length_checked_against_selected_dims() -> None:
    """`reveal_centre` is validated against the SELECTED axes, not ndim: one
    selected axis but a three-vector centre is a mismatch and must be rejected."""
    data = _ray_gsplat(np.array([1.0, 2.0, 3.0], dtype=np.float32))  # 3D
    with pytest.raises(ValueError, match="one coordinate per spatial axis"):
        compute_additive_order(
            data, method="radial", spatial_dims=[0], reveal_centre=[0.0, 0.0, 0.0]
        )
