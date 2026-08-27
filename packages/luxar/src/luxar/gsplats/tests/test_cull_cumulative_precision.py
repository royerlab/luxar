"""``cull(method="cumulative")`` must not accumulate in float32.

The bug this guards against shipped a 5x over-culled 500-timepoint archive and
raised nothing. ``np.cumsum`` on a float32 array returns float32, and the
amplitudes are sorted DESCENDING, so the running sum grows while the values
being added shrink. Once the sum passes roughly 1e9 a single float32 ULP (~64)
exceeds the amplitudes still arriving and the sum simply stops growing. The
total is then understated as well, so ``retention`` of a too-small total is
reached far too early and the cull silently keeps far fewer splats than asked.

Measured on a real Drosophila fit at ``retention=0.960`` — the correct answer is
55.3% kept at every size:

      5.12M splats  ->  float32 kept 53.5%   total 1.90e9  vs 1.92e9
     25.6M splats   ->  float32 kept 37.7%   total 8.59e9  vs 9.60e9
    128.0M splats   ->  float32 kept 11.6%   total 1.96e10 vs 4.80e10

**Why the ordinary fixtures cannot catch this.** Saturation needs the ratio of a
typical value to the running total to fall below 2^-24. At a few thousand splats
the error is under a hundredth of a percent, so every small test passes. The
test below therefore uses a few million splats AND a wide dynamic range (the
dim tail is what saturates first), and asserts against a float64 ground truth
computed independently in the test rather than against a hard-coded number.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData

#: Big enough that a float32 accumulator is off by ~0.8 percentage points, which
#: is 8x the tolerance below. Larger would be a starker gap and a slower test;
#: this is the smallest size that fails float32 unambiguously.
N_SPLATS = 2_000_000

#: The dim tail is what saturates first, so the spread matters as much as N.
#: Roughly matches a real light-sheet fit (min ~0.002, p99.9 ~480).
DYNAMIC_RANGE = 750_000

RETENTION = 0.960


def _heavy_tailed_amplitudes(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.random(n).astype(np.float32) ** 3) * 1500.0 + 0.002).astype(np.float32)


def _float64_kept_fraction(amps: np.ndarray, retention: float) -> float:
    """What the documented rule says, computed without the precision fault."""
    ordered = np.sort(amps)[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    n_keep = int(np.searchsorted(cumulative / cumulative[-1], retention)) + 1
    return n_keep / amps.size


def _data(amps: np.ndarray) -> GSplatData:
    n = amps.size
    return GSplatData(
        # Positions and covariance are irrelevant to a cumulative cull; keep
        # them as small and cheap as the constructor allows.
        centers=np.zeros((n, 3), np.float32),
        amplitudes=amps,
        cholesky_factors=np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (n, 1)),
    )


@pytest.mark.slow
def test_cumulative_cull_matches_the_float64_rule_at_scale() -> None:
    """The kept fraction must be what the amplitude CDF says, not less."""
    amps = _heavy_tailed_amplitudes(N_SPLATS)
    assert amps.max() / amps.min() > DYNAMIC_RANGE / 2, "fixture lost its dim tail"

    expected = _float64_kept_fraction(amps, RETENTION)
    culled = _data(amps).cull(method="cumulative", retention=RETENTION)
    actual = culled.n_splats / N_SPLATS

    # 0.1 percentage points. A float32 accumulator misses by ~0.8pp here, so
    # this fails decisively while leaving room for tie-breaking at the
    # searchsorted boundary.
    assert actual == pytest.approx(expected, abs=0.001), (
        f"cumulative cull kept {100 * actual:.2f}% but the amplitude CDF says "
        f"{100 * expected:.2f}% — the accumulator is losing the dim tail. This "
        f"error GROWS with splat count: it reached 5x on a 128M-splat store."
    )


@pytest.mark.slow
def test_cumulative_cull_fraction_is_invariant_under_tiling() -> None:
    """Pooling more of the SAME distribution must not change the fraction.

    This is the scale-invariance the float32 accumulator breaks, stated
    directly: tiling a splat set k times leaves the amplitude CDF identical, so
    the kept fraction has to be identical too. It is the property that made the
    real bug visible — a 20-frame subset and the full 500-frame merge had
    provably identical distributions yet culled to 65% and 12%.
    """
    base = _heavy_tailed_amplitudes(250_000, seed=1)
    small = _data(base).cull(method="cumulative", retention=RETENTION)
    large = _data(np.tile(base, 8)).cull(method="cumulative", retention=RETENTION)

    f_small = small.n_splats / base.size
    f_large = large.n_splats / (base.size * 8)
    assert f_large == pytest.approx(f_small, abs=0.001), (
        f"tiling the same distribution 8x moved the kept fraction "
        f"{100 * f_small:.2f}% -> {100 * f_large:.2f}%; the cull is "
        f"size-dependent, which it must not be"
    )


def test_reported_amplitude_retention_is_honest() -> None:
    """The stat the caller reads back must also survive the reduction.

    Cheap (small N) because this guards the stats sums rather than the
    accumulator, and a wrong dtype there would misreport at any size.
    """
    amps = _heavy_tailed_amplitudes(50_000, seed=2)
    culled = _data(amps).cull(method="cumulative", retention=RETENTION)
    reported = culled.stats.get("amplitude_retention")
    assert reported is not None
    truth = float(
        np.sum(np.asarray(culled.amplitudes), dtype=np.float64)
        / np.sum(amps, dtype=np.float64)
    )
    assert reported == pytest.approx(truth, rel=1e-6)
    assert reported >= RETENTION - 1e-6, (
        f"kept only {reported:.9f} of the amplitude but was asked for "
        f"{RETENTION:.9f} (short by {RETENTION - reported:.2e})"
    )
