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


def _mass(d: GSplatData) -> float:
    a = np.asarray(d.amplitudes, dtype=np.float64)
    chol = np.asarray(d.cholesky_factors, dtype=np.float64)
    return float(np.sum(a * np.prod(chol[:, [0, 2, 5]], axis=1)))


# A merge conserves mass by construction (each representative carries its
# cluster's combined mass, and the reduction rescales to pin the total), so the
# tolerance is tight on purpose: a loose band would pass a merge that quietly
# threw a third of the object away.
_MASS_TOL = 0.01


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

    ratio = _mass(out) / _mass(data)
    assert abs(ratio - 1.0) < _MASS_TOL, f"merge lost/gained mass: ratio={ratio:.3f}"


@pytest.mark.parametrize("target", [340, 700, 1234])
def test_merge_conserves_mass_at_an_awkward_target(target) -> None:
    """Mass survives targets that are NOT a whole fraction of the input.

    The merge used to reduce by an integer FACTOR — the only counts it could
    land on were N/2, N/3, ... — and trimmed the surplus away to reach anything
    in between. Each discarded representative carries its whole cluster's mass,
    so a 340-of-1000 request (factor 2 -> 500 representatives -> drop 160) came
    back 27% dimmer: the one property merge exists for, lost on most targets.
    """
    data = _cloud(1000)
    out = decimate(data, target=target, method="merge")
    assert out.n_splats <= target
    ratio = _mass(out) / _mass(data)
    assert abs(ratio - 1.0) < _MASS_TOL, f"merge lost/gained mass: ratio={ratio:.3f}"


@pytest.mark.parametrize("target", [700, 900, 999])
def test_explicit_merge_honours_a_target_above_the_crossover(target) -> None:
    """`-m merge` above 50% is a legal request, and must not silently halve.

    ``auto`` picks prefix up there, but the family is user-selectable and the
    contract is "at most target, and close to it". An integer compression factor
    is at least 2, so the old merge path answered every one of these with N/2 —
    500 splats for a 900 request.
    """
    data = _cloud(1000)
    out = decimate(data, target=target, method="merge")
    assert out.n_splats <= target
    assert out.n_splats >= 0.98 * target


def test_decimate_rejects_a_bad_target() -> None:
    data = _cloud(100)
    with pytest.raises(ValueError):
        decimate(data, target=0)


# --------------------------------------------------------------------------
# coarsen_dims: a merge-only knob, validated and reported for BOTH families
# --------------------------------------------------------------------------
def _cloud4d(n: int = 400, seed: int = 0) -> GSplatData:
    """A 4D cloud whose last axis is three integer timepoints."""
    rng = np.random.default_rng(seed)
    centers = np.empty((n, 4), dtype=np.float32)
    centers[:, :3] = rng.uniform(0, 50, size=(n, 3))
    centers[:, 3] = rng.integers(0, 3, size=n).astype(np.float32)
    chol = np.zeros((n, 10), dtype=np.float32)
    chol[:, [d * (d + 1) // 2 + d for d in range(4)]] = 1.0
    return GSplatData(
        centers=centers,
        amplitudes=rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        cholesky_factors=chol,
    )


@pytest.mark.parametrize("method", ["merge", "prefix"])
def test_an_out_of_range_coarsen_dim_is_rejected_by_both_families(method) -> None:
    """The knob is merge-only; its VALIDATION must not be.

    Only ``merge`` reached ``_normalise_coarsen_dims`` (inside
    ``merge_to_count``), so ``coarsen_dims=[9, 17]`` was a hard ValueError on
    one family and silently accepted on the other — and under the default
    ``method="auto"`` which family runs depends on the kept fraction, so the
    same call could raise or not depending on the target (#1600 review).
    """
    data = _cloud4d()
    with pytest.raises(ValueError, match=r"out of range for ndim=4"):
        decimate(data, target=50, method=method, coarsen_dims=[9, 17])


def test_an_in_range_coarsen_dims_request_is_accepted_by_both_families() -> None:
    """The validator must not start rejecting the requests that are fine."""
    data = _cloud4d()
    for method in ("merge", "prefix"):
        assert (
            decimate(data, target=50, method=method, coarsen_dims=[0, 1, 2]).n_splats
            == 50
        )


def test_a_prefix_says_it_is_ignoring_coarsen_dims(recwarn) -> None:
    """The family decides whether the knob means anything — silently, before.

    With the default ``method="auto"`` the family (and therefore whether
    ``--coarsen-dims`` is honoured at all, and therefore the output's chunk
    layout) flips at the 50%-kept crossover with nothing said.
    """
    data = _cloud4d(400)
    decimate(data, target=0.4, coarsen_dims=[0, 1, 2])  # -> merge, honoured
    assert not [w for w in recwarn.list if "IGNORED" in str(w.message)]

    with pytest.warns(UserWarning, match=r"coarsen_dims=\[0, 1, 2\] is IGNORED") as rec:
        decimate(data, target=0.5, coarsen_dims=[0, 1, 2])  # -> prefix, ignored
    message = str(rec[0].message)
    assert "'prefix' family" in message
    # ...and it says WHY this run became a prefix, since nothing was asked for.
    assert "method='auto' resolved to prefix" in message

    # An EXPLICIT prefix still says the knob was dropped, without the crossover
    # explanation (nothing resolved — the caller named the family).
    with pytest.warns(UserWarning, match=r"coarsen_dims=\[0, 1, 2\] is IGNORED") as rec:
        decimate(data, target=0.4, method="prefix", coarsen_dims=[0, 1, 2])
    assert "resolved to prefix" not in str(rec[0].message)


def test_the_ignored_coarsen_dims_notice_is_a_warning_not_a_print(capsys) -> None:
    """A library function does not write to stdout behind ``verbose=False``.

    Every other line ``decimate`` emits is gated on ``verbose``; this one was an
    unconditional ``aprint``, so a programmatic
    ``decimate(..., verbose=False, coarsen_dims=[...])`` printed unbidden. The
    house convention for "your argument had a surprising effect" is
    ``warnings.warn`` (``GSplatData.filter``'s pyramid notice), which a caller
    can filter, record, or turn into an error — and which the CLI still shows
    (:func:`test_the_ignored_coarsen_dims_notice_reaches_a_cli_user`).
    """
    data = _cloud4d(400)
    with pytest.warns(UserWarning, match="is IGNORED"):
        decimate(data, target=0.5, coarsen_dims=[0, 1, 2], verbose=False)
    captured = capsys.readouterr()
    assert "IGNORED" not in captured.out and "IGNORED" not in captured.err
