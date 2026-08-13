"""``decimate`` — reduce a dataset to a target splat count.

Covers the two things a caller relies on and cannot check cheaply themselves:
the target is honoured (never exceeded, in either the count or the fraction
spelling), and the family choice follows the measured crossover rather than a
coin flip. The merge family's own quality is covered by the substitutive tests;
what is asserted here is that ``decimate`` routes to it and preserves the
dataset's physical content (mass) rather than silently dropping it.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.decimate import (
    PREFIX_ABOVE_FRACTION,
    decimate,
    resolve_method,
    resolve_target_count,
)


def _cloud(n: int, seed: int = 0) -> GSplatData:
    """A compact isotropic cloud; large enough that clustering has work to do."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 50, size=(n, 3)).astype(np.float32)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    return GSplatData(
        centers=centers,
        amplitudes=rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        cholesky_factors=chol,
    )


# --------------------------------------------------------------------------
# target resolution
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("target", "n_in", "expected"),
    [
        (100, 1000, 100),  # int = absolute count
        (0.1, 1000, 100),  # float = fraction
        (1.0, 1000, 1000),  # 1.0 keeps everything ...
        (1, 1000, 1),  # ... while 1 is a single splat
        (5000, 1000, 1000),  # clamped to the input
    ],
)
def test_resolve_target_count(target, n_in, expected) -> None:
    assert resolve_target_count(target, n_in) == expected


@pytest.mark.parametrize("bad", [0, -5, 0.0, 1.5, -0.2])
def test_resolve_target_count_rejects_out_of_range(bad) -> None:
    with pytest.raises(ValueError):
        resolve_target_count(bad, 1000)


def test_resolve_target_count_rejects_bool() -> None:
    """``True`` is an int subclass and would silently mean "one splat"."""
    with pytest.raises(TypeError):
        resolve_target_count(True, 1000)


# --------------------------------------------------------------------------
# family choice
# --------------------------------------------------------------------------
def test_auto_picks_prefix_above_the_crossover_and_merge_below() -> None:
    n_in = 1000
    above = int(PREFIX_ABOVE_FRACTION * n_in) + 10
    below = int(PREFIX_ABOVE_FRACTION * n_in) - 10
    assert resolve_method("auto", above, n_in) == "prefix"
    assert resolve_method("auto", below, n_in) == "merge"


def test_explicit_method_overrides_auto() -> None:
    """An explicit family is honoured on the side auto would not have chosen."""
    n_in = 1000
    assert resolve_method("merge", 900, n_in) == "merge"  # auto would say prefix
    assert resolve_method("prefix", 10, n_in) == "prefix"  # auto would say merge


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown decimation method"):
        resolve_method("cull", 100, 1000)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["prefix", "merge"])
@pytest.mark.parametrize("target", [250, 137, 999])
def test_target_count_is_delivered(method, target) -> None:
    """The requested count is delivered, not merely respected as a ceiling.

    ``merge`` reduces by an INTEGER factor, so its natural counts are quantised
    (N/2, N/3, ...) and most targets fall between two of them. Landing on the
    nearest quantised value below the request returned 9.1% for a 10% ask on
    real data, which reads as the tool ignoring the number you gave it. The
    awkward targets here (137, 999 out of 2000) fall between achievable factors.

    The contract is "at most ``target``, and close to it" rather than exact
    equality: the clustering drops degenerate (empty / non-positive-mass)
    clusters, so a merge can legitimately land a hair under — on the real
    1.65M-splat dataset a 165,340 request yields 165,276. Asserting equality
    would encode a promise the reduction cannot always keep.
    """
    data = _cloud(2000)
    out = decimate(data, target=target, method=method)
    assert out.n_splats <= target
    assert out.n_splats >= 0.98 * target


def test_fraction_and_count_spellings_agree() -> None:
    data = _cloud(2000)
    by_count = decimate(data, target=500, method="prefix")
    by_fraction = decimate(data, target=0.25, method="prefix")
    assert by_count.n_splats == by_fraction.n_splats


def test_full_target_returns_input_unchanged() -> None:
    data = _cloud(500)
    assert decimate(data, target=1.0) is data
    assert decimate(data, target=500) is data


def test_prefix_is_a_subset_of_the_input() -> None:
    """The prefix family must SELECT splats, never synthesise new ones."""
    data = _cloud(1000)
    out = decimate(data, target=100, method="prefix")
    src = {tuple(c) for c in np.asarray(data.centers).tolist()}
    assert all(tuple(c) in src for c in np.asarray(out.centers).tolist())


def test_merge_synthesises_representatives_and_keeps_mass() -> None:
    """The merge family is NOT a subset: it summarises, conserving total mass.

    This is the property that makes merge win under aggressive reduction, so it
    is the one worth pinning: a merge that merely dropped splats would lose mass
    proportionally and would show up here.
    """
    data = _cloud(2000)
    out = decimate(data, target=200, method="merge")

    src = {tuple(c) for c in np.asarray(data.centers).tolist()}
    synthesised = [c for c in np.asarray(out.centers).tolist() if tuple(c) not in src]
    assert synthesised, "merge should produce representatives, not a subset"

    def mass(d: GSplatData) -> float:
        a = np.asarray(d.amplitudes, dtype=np.float64)
        chol = np.asarray(d.cholesky_factors, dtype=np.float64)
        return float(np.sum(a * np.prod(chol[:, [0, 2, 5]], axis=1)))

    ratio = mass(out) / mass(data)
    assert 0.5 < ratio < 2.0, f"merge lost/gained mass: ratio={ratio:.3f}"


def test_decimate_rejects_a_bad_target() -> None:
    data = _cloud(100)
    with pytest.raises(ValueError):
        decimate(data, target=0)
