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
    interleave_order_across_slices,
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


# ── truncation support comes from the DATA, not a hardcoded 3.0 (#1180) ──────


def _gsplat_at_radius(radius: float, n: int = 40, seed: int = 11) -> GSplatData:
    """A random overlapping dataset fitted/rendered at ``radius`` sigmas."""
    base = _make_random_gsplat(n=n, ndim=3, seed=seed)
    return GSplatData(
        centers=base.centers,
        amplitudes=base.amplitudes,
        cholesky_factors=base.cholesky_factors,
        truncation_radius=radius,
    )


def _sigmas_seen_by_gram(monkeypatch, call) -> list[float]:
    """Record the ``sigmas`` every ``_build_sparse_gram`` call receives."""
    import luxar.gsplats.lod.additive as additive_mod

    seen: list[float] = []
    real = additive_mod._build_sparse_gram

    def _spy(data, *, sigmas):
        seen.append(float(sigmas))
        return real(data, sigmas=sigmas)

    monkeypatch.setattr(additive_mod, "_build_sparse_gram", _spy)
    call()
    return seen


def test_resolve_truncation_sigmas_reads_the_dataset() -> None:
    """``None`` resolves to the dataset's own radius; an explicit value wins.

    The radius here is deliberately NOT ``DEFAULT_TRUNCATION_RADIUS``: asserting
    on a 2.75 dataset would also pass for a resolver that never looked at ``data``
    and just returned the constant, so it would not test the "reads the dataset"
    claim at all.
    """
    from luxar.gsplats.lod.additive import resolve_truncation_sigmas
    from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

    data = _gsplat_at_radius(1.5, n=4)
    assert data.truncation_radius != pytest.approx(DEFAULT_TRUNCATION_RADIUS)
    assert resolve_truncation_sigmas(None, data) == pytest.approx(1.5)
    assert resolve_truncation_sigmas(3.0, data) == pytest.approx(3.0)


def test_resolve_truncation_sigmas_falls_back_without_a_radius() -> None:
    """A data-like object exposing no ``truncation_radius`` falls back to the
    canonical default rather than crashing (mirrors ``principal_radii``)."""
    from luxar.gsplats.lod.additive import resolve_truncation_sigmas
    from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

    class _NoRadius:
        pass

    assert resolve_truncation_sigmas(None, _NoRadius()) == pytest.approx(  # type: ignore[arg-type]
        DEFAULT_TRUNCATION_RADIUS
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_resolve_truncation_sigmas_rejects_a_degenerate_value(bad: float) -> None:
    """An explicit non-positive/non-finite σ empties the Gram silently; reject."""
    from luxar.gsplats.lod.additive import resolve_truncation_sigmas

    data = _gsplat_at_radius(2.75, n=4)
    # The message must be about the PRUNING σ (not the render pipeline's GPU
    # uniforms) and must name the offending value.
    with pytest.raises(ValueError, match="pruning"):
        resolve_truncation_sigmas(bad, data)


@pytest.mark.parametrize("radius", [1.0, 2.75])
def test_compute_additive_order_prunes_at_the_dataset_radius(
    monkeypatch, radius: float
) -> None:
    """#1180: the greedy Gram was pruned at a hardcoded 3.0 regardless of the
    radius the data was fitted at. It must use the dataset's own radius."""
    data = _gsplat_at_radius(radius)
    seen = _sigmas_seen_by_gram(
        monkeypatch, lambda: compute_additive_order(data, method="greedy")
    )
    assert seen == [pytest.approx(radius)]


def test_compute_additive_order_explicit_sigmas_still_override(monkeypatch) -> None:
    """An explicit ``truncation_sigmas=`` beats the dataset's radius."""
    data = _gsplat_at_radius(1.0)
    seen = _sigmas_seen_by_gram(
        monkeypatch,
        lambda: compute_additive_order(data, method="greedy", truncation_sigmas=3.0),
    )
    assert seen == [pytest.approx(3.0)]


@pytest.mark.parametrize("radius", [1.0, 2.75])
def test_make_additive_lod_prunes_at_the_dataset_radius(
    monkeypatch, radius: float
) -> None:
    """Same contract through the ladder builder (and its internal reuse of the
    single Gram): every build sees the dataset's radius."""
    data = _gsplat_at_radius(radius)
    seen = _sigmas_seen_by_gram(
        monkeypatch, lambda: make_additive_lod(data, n_lods=2, method="greedy")
    )
    assert seen and all(s == pytest.approx(radius) for s in seen)


def test_make_additive_lod_explicit_sigmas_still_override(monkeypatch) -> None:
    data = _gsplat_at_radius(1.0)
    seen = _sigmas_seen_by_gram(
        monkeypatch,
        lambda: make_additive_lod(
            data, n_lods=2, method="greedy", truncation_sigmas=3.0
        ),
    )
    assert seen and all(s == pytest.approx(3.0) for s in seen)


def test_smaller_radius_prunes_more_pairs() -> None:
    """Grounding the spy assertions: the σ really is the pruning knob, so a
    tighter support keeps strictly fewer off-diagonal Gram entries."""
    data = _gsplat_at_radius(2.75)
    tight = _build_sparse_gram(data, sigmas=1.0)
    wide = _build_sparse_gram(data, sigmas=3.0)
    assert tight.nnz < wide.nnz


def test_make_lod_pyramid_resolves_the_radius_once(monkeypatch) -> None:
    """``make_lod_pyramid`` resolves ``None`` from the INPUT dataset and hands
    every per-level ladder the concrete value (never a hardcoded 3.0)."""
    import luxar.gsplats.lod.pyramid as pyramid_mod

    seen: list[float | None] = []
    real = pyramid_mod.make_additive_lod

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("truncation_sigmas"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pyramid_mod, "make_additive_lod", _spy)
    data = _gsplat_at_radius(1.0, n=32, seed=12)
    pyramid_mod.make_lod_pyramid(
        data, compression_factor=4, levels=1, n_additive_lods=2, device="cpu"
    )
    assert seen and all(s == pytest.approx(1.0) for s in seen)


def test_make_additive_lod_prunes_at_the_TARGET_levels_radius(monkeypatch) -> None:
    """#1180 follow-up: the σ must come from the substitutive level being pruned.

    Per-sub-LOD truncation radii are independent and really round-trip (see
    ``TestTruncationRadiusRoundtrip::test_per_level_truncation_radius_roundtrip``
    in ``gsplats/io/tests/test_save_load.py``),
    so resolving from ``data`` — whose radius is the FINEST leaf's — prunes the
    selected level at a support it does not claim. Here the finest level is 1.0
    and level 1 is 4.0; ``substitutive_level=1`` must prune at 4.0.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

    def _sublod(n: int, seed: int, radius: float) -> AdditiveSubLOD:
        base = _make_random_gsplat(n=n, ndim=3, seed=seed)
        return AdditiveSubLOD(
            centers=base.centers,
            amplitudes=base.amplitudes,
            cholesky_factors=base.cholesky_factors,
            truncation_radius=radius,
        )

    data = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[_sublod(30, 1, 1.0)],
                compression_factor=1,
                level_index=0,
            ),
            SubstitutiveLevel(
                additive_sublods=[_sublod(12, 2, 4.0)],
                compression_factor=4,
                level_index=1,
            ),
        ]
    )
    # Precondition: the radii really are heterogeneous, and `data`'s own radius
    # (what the buggy resolution reads) is the finest level's, not the target's.
    assert data.truncation_radius == pytest.approx(1.0)
    assert data.at_substitutive(1).flattened().truncation_radius == pytest.approx(4.0)

    seen = _sigmas_seen_by_gram(
        monkeypatch,
        lambda: make_additive_lod(
            data, n_lods=2, method="greedy", substitutive_level=1
        ),
    )
    assert seen and all(s == pytest.approx(4.0) for s in seen)


def test_resolve_truncation_sigmas_accepts_a_tiny_positive_value() -> None:
    """A pruning σ is a CPU knob, not a render uniform: it carries no float32
    shader bounds, so a very small positive cutoff (which the render-domain
    ``validate_truncation_radius`` rejects) must still be accepted here."""
    from luxar.gsplats.lod.additive import resolve_truncation_sigmas

    data = _gsplat_at_radius(2.75, n=4)
    assert resolve_truncation_sigmas(1e-5, data) == pytest.approx(1e-5)
    # Numpy scalars and ints round-trip through the same coercion.
    assert resolve_truncation_sigmas(np.float32(2.5), data) == pytest.approx(2.5)
    assert resolve_truncation_sigmas(4, data) == pytest.approx(4.0)


# ── additive_rung_count (#1632) ────────────────────────────────────────
#
# The whole value of this query is that it agrees with the BUILDER, so every
# case below is scored against ``make_additive_lod(...).n_additive_sublods``
# rather than against a hand-written expectation — a re-derivation is exactly
# the drift the shared ``_resolve_breakpoints`` call exists to rule out.

_RUNG_SPECS: list[tuple[int, object]] = [
    # equal-count, including n_lods > n (which clamps to n) and n_lods == 1.
    (1, "equal-count"),
    (2, "equal-count"),
    (3, "equal-count"),
    (4, "equal-count"),
    (5, "equal-count"),
    (32, "equal-count"),
    (40, "equal-count"),
    # stream:<c> — a chunk that yields several rungs, and one >= n (single rung).
    (4, "stream:4"),
    (4, "stream:8"),
    (4, "stream:32"),
    (4, "stream:64"),
    # explicit cumulative counts — a full list ending at N and a partial one
    # (``_resolve_breakpoints`` appends the final N, which is one more rung).
    (4, [8, 16, 32]),
    (4, [5, 11]),
    (4, [32]),
]


@pytest.mark.parametrize("n_lods,breakpoints", _RUNG_SPECS)
def test_additive_rung_count_matches_the_builder(
    n_lods: int, breakpoints: object
) -> None:
    """Parity with ``make_additive_lod`` itself, per breakpoint kind (#1632).

    The gate in ``from_io._reject_a_partition_beside_a_stored_ladder`` refuses a
    call on the strength of this number, so an answer that merely looks
    plausible is not good enough: it must be the count of sub-LODs the build
    loop would EMIT, ``if end <= prev: continue`` de-duplication included.
    """
    from luxar.gsplats.lod.additive import additive_rung_count

    data = _make_random_gsplat(n=32, ndim=3, seed=11)
    built = make_additive_lod(
        data, n_lods, method="self_energy", breakpoints=breakpoints
    )
    assert additive_rung_count(32, n_lods, breakpoints) == built.n_additive_sublods


def test_additive_rung_count_on_an_empty_leaf_matches_the_builder() -> None:
    """``n == 0`` is 1, not 0: the empty branch emits one ``lod_method="none"``."""
    from luxar.gsplats.lod.additive import additive_rung_count

    empty = _make_empty_gsplat(ndim=3)
    built = make_additive_lod(empty, 4, method="self_energy")
    assert built.n_additive_sublods == 1
    assert additive_rung_count(0, 4) == 1
    assert additive_rung_count(-1, 4) == 1


def test_energy_fraction_breakpoints_are_unknown_not_guessed() -> None:
    """Those cuts need the ordering AND the energy curve — the expensive half.

    Answering ``len(fracs)`` would be a guess: the resolver de-duplicates cuts
    that land on the same k and appends a final N — measured on a 12-splat leaf,
    ``[1.0]`` builds one rung and ``[0.5]`` builds two, so neither the list
    length nor "one fraction means one rung" is a safe stand-in. UNKNOWN is the
    honest answer, and callers fall back to what they already know rather than
    skipping (the gate falls back to the leaf's STORED rung count — skipping was
    the round-1 bug that re-stranded the wrapper this query exists to prevent).
    """
    from luxar.gsplats.lod.additive import additive_rung_count

    assert additive_rung_count(32, 4, [0.5, 0.9, 1.0]) is None
    assert additive_rung_count(32, 4, [1.0]) is None


@pytest.mark.parametrize(
    "n_lods,breakpoints",
    [
        (0, "equal-count"),  # n_lods must be positive
        (-3, "equal-count"),
        (4, "nonsense"),  # unknown breakpoints string
        (4, []),  # empty list
        (4, [1, 0.5]),  # mixed int/float
        (4, [8, 8]),  # not strictly increasing
        (4, [-1, 4]),  # non-positive count
        (4, [100_000]),  # largest breakpoint exceeds N
        (4, 42),  # not a string and not a list
        (None, "equal-count"),  # unusable n_lods
    ],
)
def test_a_malformed_spec_is_unknown_and_does_not_raise(
    n_lods: object, breakpoints: object
) -> None:
    """A query must not pre-empt the builder's own fault report (#1632).

    Every one of these aborts a real ``make_additive_lod`` build, at its own
    site with its own message. Raising here too would move that verdict into a
    gate that has no business owning it — and, at the gate, would turn a
    diagnosable build failure into a partition refusal.
    """
    from luxar.gsplats.lod.additive import additive_rung_count

    assert additive_rung_count(32, n_lods, breakpoints) is None  # type: ignore[arg-type]


# ── slice_dims: the slice-even interleave (#2485) ─────────────────────────


def _make_sliced_gsplat(
    sizes: tuple[int, ...], ndim: int = 4, seed: int = 0
) -> GSplatData:
    """A stack whose LAST centre column is a discrete slice coordinate.

    Deliberately UNBALANCED: the whole point of the interleave is what happens
    to the small coordinates, and an even stack cannot tell a round-robin apart
    from a global prefix. Mirrors the shape of the sliced 4D demos — three
    spatial columns plus a stacked time column at index ``ndim - 1``.

    Amplitudes are scaled by ``100 ** slice_index``, so slice 0 is both the
    SMALLEST and the FAINTEST and a contribution ordering ranks all of it last.
    That is the real situation this fixes — a sparse early radar scan is small
    and weak at once — and it makes the "starved without the interleave" half
    structural rather than a property of one lucky seed: the self-energy score
    is $a^2|\\Sigma|^{1/2}$, so a 100x amplitude gap is a 10^4 score gap, which
    no plausible covariance spread can close.
    """
    rng = np.random.default_rng(seed)
    n = int(sum(sizes))
    centers = (rng.standard_normal((n, ndim)) * 1.5).astype(np.float32)
    slice_index = np.concatenate(
        [np.full(size, float(i)) for i, size in enumerate(sizes)]
    )
    centers[:, ndim - 1] = slice_index.astype(np.float32)
    amplitudes = ((np.abs(rng.standard_normal(n)) + 0.5) * 100.0**slice_index).astype(
        np.float32
    )
    tril = ndim * (ndim + 1) // 2
    chol = (rng.standard_normal((n, tril)) * 0.1).astype(np.float32)
    diag_idx = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag_idx] = np.abs(chol[:, diag_idx]) + 0.4
    return GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=chol,
    )


def _slice_histogram(
    data: GSplatData, indices: np.ndarray, n_slices: int, col: int
) -> list[int]:
    """How many of ``indices`` fall in each slice coordinate ``0..n_slices-1``."""
    coords = np.asarray(data.centers)[:, col][indices]
    return [int((coords == float(i)).sum()) for i in range(n_slices)]


#: Slice sizes 2 / 5 / 40 over 3 coordinates — 47 splats, the smallest slice
#: 4% of the node. At the demo's ``n_lods=4`` that puts rung 0 at 12 splats,
#: which is more than enough budget to carry the 2-splat slice WHOLE while a
#: global energy-ordered prefix gives it nothing.
_UNBALANCED_SIZES = (2, 5, 40)


def test_interleave_returns_a_permutation() -> None:
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=21)
    order = compute_additive_order(data, method="self_energy", slice_dims=[3])
    assert order.shape == (data.n_splats,)
    assert order.dtype == np.int64
    assert sorted(order.tolist()) == list(range(data.n_splats))


def test_interleave_gives_every_slice_an_equal_absolute_budget() -> None:
    """Every prefix holds ``min(slice size, completed passes)`` of each slice.

    The concrete numbers below are all pass boundaries, so they are fixed by the
    round-robin alone and not by which base order fed it: 3 = one element each,
    6 = two each (slice 0 now exhausted), 12 = 6 + three passes of the two
    surviving slices, 15 = 12 + three more from the only slice still holding
    anything. That is the equal-ABSOLUTE-budget shape
    ``scripts/check_demo_ladders.py``'s first-paint arm asks for.
    """
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=22)
    order = compute_additive_order(data, method="self_energy", slice_dims=[3])
    for prefix, expected in (
        (3, [1, 1, 1]),
        (6, [2, 2, 2]),
        (12, [2, 5, 5]),
        (15, [2, 5, 8]),
    ):
        assert _slice_histogram(data, order[:prefix], 3, 3) == expected, (
            f"prefix {prefix} is not slice-even"
        )
    assert _slice_histogram(data, order, 3, 3) == list(_UNBALANCED_SIZES)


def test_the_small_slice_is_starved_without_the_interleave() -> None:
    """The regression half of #2485: the base order is what shipped.

    Same data, same method, same rung-0 size (``n_lods=4`` over 47 splats = 12).
    Interleaved, the 2-splat slice arrives COMPLETE; ordered globally by
    contribution it gets nothing at all, which is the 4-splats-per-scan failure
    the NEXRAD supercell shipped with in miniature.
    """
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=23)
    rung0 = -(-data.n_splats // 4)
    base = compute_additive_order(data, method="self_energy")
    even = compute_additive_order(data, method="self_energy", slice_dims=[3])
    assert _slice_histogram(data, base[:rung0], 3, 3)[0] == 0
    assert _slice_histogram(data, even[:rung0], 3, 3)[0] == _UNBALANCED_SIZES[0]
    assert min(_slice_histogram(data, even[:rung0], 3, 3)) > 0


def test_interleave_preserves_the_within_slice_order() -> None:
    """Within a slice the base method still ranks — bright core first.

    This is why the interleave is a MODIFIER and not an ordering method: it
    redistributes across slices and reorders nothing inside one, so each
    coordinate keeps painting by contribution rather than evenly thin.
    """
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=24)
    base = compute_additive_order(data, method="self_energy")
    even = compute_additive_order(data, method="self_energy", slice_dims=[3])
    coords = np.asarray(data.centers)[:, 3]
    for i in range(len(_UNBALANCED_SIZES)):
        assert [int(k) for k in base if coords[k] == float(i)] == [
            int(k) for k in even if coords[k] == float(i)
        ]


def test_interleave_is_deterministic_and_idempotent() -> None:
    """No seed, and re-applying it is a no-op.

    Idempotency is what lets both the authoring door and ``make_additive_lod``'s
    Gram branch apply the modifier without a "has this already been done?" flag:
    the within-group order — and hence every within-group rank — is unchanged by
    a second pass.
    """
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=25)
    once = compute_additive_order(data, method="self_energy", slice_dims=[3])
    again = compute_additive_order(data, method="self_energy", slice_dims=[3])
    assert np.array_equal(once, again)
    assert np.array_equal(once, interleave_order_across_slices(data, once, [3]))


def test_slice_dims_none_leaves_the_order_untouched() -> None:
    """The modifier is strictly opt-in: no `slice_dims`, byte-identical order."""
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=26)
    for method in ("self_energy", "mass", "amplitude", "greedy", "radial"):
        assert np.array_equal(
            compute_additive_order(data, method=method, seed=0),
            compute_additive_order(data, method=method, seed=0, slice_dims=None),
        ), method


@pytest.mark.parametrize("method", ["self_energy", "greedy"])
def test_make_additive_lod_builds_a_slice_even_ladder(method: str) -> None:
    """End to end, on BOTH order paths.

    ``self_energy`` runs through ``compute_additive_order``; ``greedy`` takes
    ``make_additive_lod``'s own ``_order_from_gram`` shortcut (it reuses the Gram
    it built) and never reaches that function, so the modifier has to be applied
    on that branch too — this parametrization is what kills a fix applied to
    only one of them.

    Counts in == counts out is asserted alongside: an interleave that dropped or
    duplicated a splat would still look slice-even on rung 0.
    """
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=27)
    built = make_additive_lod(data, 4, method=method, slice_dims=[3])
    sublods = built.substitutive_levels[0].additive_sublods
    per_rung = [
        [
            int((np.asarray(sub.centers)[:, 3] == float(i)).sum())
            for i in range(len(_UNBALANCED_SIZES))
        ]
        for sub in sublods
    ]
    assert per_rung[0] == [2, 5, 5], f"{method}: rung 0 is not slice-even: {per_rung}"
    assert [sum(a) for a in zip(*per_rung, strict=True)] == list(_UNBALANCED_SIZES)
    assert sum(sum(rung) for rung in per_rung) == data.n_splats


@pytest.mark.parametrize(
    "slice_dims,match",
    [
        ([], "non-empty"),
        ([-1], "non-negative"),
        ([3, 3], "must not repeat"),
        ([9], "out of range"),
        ([1.5], "integer column indices"),
    ],
)
def test_malformed_slice_dims_raise(slice_dims: list[object], match: str) -> None:
    """Each of these would GROUP WRONG rather than fail — see `_validate_slice_dims`."""
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=28)
    with pytest.raises(ValueError, match=match):
        compute_additive_order(data, method="self_energy", slice_dims=slice_dims)  # type: ignore[arg-type]


def test_interleave_on_degenerate_inputs() -> None:
    """N=0, N=1, one slice, and one slice per splat all reduce to the base order.

    A single slice coordinate is the important one of the four: the interleave
    must be a no-op when there is nothing to interleave across, so a demo that
    names `slice_dims` on an axis it turns out not to be sliced on loses nothing.
    """
    empty = _make_empty_gsplat(ndim=4)
    assert compute_additive_order(
        empty, method="self_energy", slice_dims=[3]
    ).shape == (0,)

    single_splat = _make_sliced_gsplat((1,), seed=29)
    assert compute_additive_order(
        single_splat, method="self_energy", slice_dims=[3]
    ).tolist() == [0]

    one_slice = _make_sliced_gsplat((16,), seed=30)
    assert np.array_equal(
        compute_additive_order(one_slice, method="self_energy"),
        compute_additive_order(one_slice, method="self_energy", slice_dims=[3]),
    )

    all_distinct = _make_sliced_gsplat((1,) * 16, seed=31)
    assert np.array_equal(
        compute_additive_order(all_distinct, method="self_energy"),
        compute_additive_order(all_distinct, method="self_energy", slice_dims=[3]),
    )


def test_nan_slice_keys_are_refused_not_silently_split() -> None:
    """A NaN slice coordinate would become its own singleton slice.

    ``np.unique`` compares NaN unequal to itself, so every NaN-keyed splat gets
    its own group — and a singleton is smaller than any budget, i.e. "carried
    whole", i.e. inside pass 0. Measured before the guard, 6 NaN-keyed splats
    beside 3 real slices of 10 ALL landed in the first pass and diluted every
    real slice's budget silently. ``_radial_score`` guards the same class on
    ``spatial_dims``; this is the ``slice_dims`` peer.
    """
    data = _make_sliced_gsplat(_UNBALANCED_SIZES, seed=32)
    centers = np.asarray(data.centers).copy()
    centers[:6, 3] = np.nan
    poisoned = GSplatData(
        centers=centers,
        amplitudes=data.amplitudes,
        cholesky_factors=data.cholesky_factors,
    )
    with pytest.raises(ValueError, match="slice_dims columns of centers must be"):
        compute_additive_order(poisoned, method="self_energy", slice_dims=[3])


def test_negative_zero_slice_keys_group_together() -> None:
    """``-0.0`` and ``0.0`` are ONE slice — the grouping is value-based.

    Worth pinning because the guard above rejects non-finite keys and could
    plausibly have been written to reject signed-zero pathologies too. It must
    not: a stacked axis written through float32 can legitimately carry ``-0.0``,
    and splitting that coordinate in two would halve its budget.
    """
    data = _make_sliced_gsplat((5, 5, 5), seed=33)
    centers = np.asarray(data.centers).copy()
    centers[:2, 3] = -0.0
    signed = GSplatData(
        centers=centers,
        amplitudes=data.amplitudes,
        cholesky_factors=data.cholesky_factors,
    )
    order = compute_additive_order(signed, method="self_energy", slice_dims=[3])
    coords = centers[:, 3]
    assert [int((coords[order[:3]] == value).sum()) for value in np.unique(coords)] == [
        1,
        1,
        1,
    ]


@pytest.mark.parametrize("n_splats", [0, 1])
@pytest.mark.parametrize("slice_dims", [[], [-5], [1.5], [999]])
def test_malformed_slice_dims_raise_even_on_a_trivial_leaf(
    n_splats: int, slice_dims: list[object]
) -> None:
    """Validation must not depend on how many splats the leaf happens to hold.

    Both loop-callers hand the SAME spec to many leaves —
    ``resolve_additive_axis_gsplats`` walks substitutive levels and
    ``recipes._ladder_for_part`` walks BSP parts — so validating only where the
    interleave actually runs means one build both rejects and accepts a typo,
    depending on which leaf is empty. Measured before the fix: every entry below
    passed silently at N in {0, 1} and raised at N = 3.
    """
    data = (
        _make_empty_gsplat(ndim=4)
        if n_splats == 0
        else _make_sliced_gsplat((1,), seed=34)
    )
    assert data.n_splats == n_splats
    with pytest.raises(ValueError):
        compute_additive_order(data, method="self_energy", slice_dims=slice_dims)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_additive_lod(data, 4, method="self_energy", slice_dims=slice_dims)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "order,match",
    [
        (np.arange(9, dtype=np.float64), "must be an integer array"),
        (np.array([-1, *range(1, 9)]), "must be non-negative"),
        (np.arange(7), "must have shape"),
        (np.arange(10), "must have shape"),
        (np.array([*range(8), 9]), "out of range"),
    ],
)
def test_a_malformed_order_is_refused(order: np.ndarray, match: str) -> None:
    """``interleave_order_across_slices`` is public, so ``order`` is untrusted.

    Every case below was measured to give a plausible WRONG answer rather than an
    error: the float array is truncated by the int64 cast, the negative entry
    aliases to another element, and a short/long array silently yields a partial
    ordering. Only the out-of-range index raised, and only via the gather.
    """
    data = _make_sliced_gsplat((3, 3, 3), seed=35)
    assert data.n_splats == 9
    with pytest.raises(ValueError, match=match):
        interleave_order_across_slices(data, order, [3])


def test_duplicate_order_entries_are_the_callers_problem() -> None:
    """Documented non-guarantee: detecting them costs a second O(N log N) pass.

    Pinned so the contract is deliberate rather than an oversight — no in-tree
    producer can emit a duplicate (they are all ``argsort`` or
    ``rng.permutation``), and the four cheap O(N) guards above catch everything
    that has actually been hit.
    """
    data = _make_sliced_gsplat((3, 3, 3), seed=36)
    duplicated = np.array([0, 0, *range(2, 9)])
    result = interleave_order_across_slices(data, duplicated, [3])
    assert result.size == 9
    assert sorted(result.tolist()) != list(range(9))
