"""Barrier-coverage invariant: every barrier group is represented at every level.

Regression guard for the tiled-batch merge scenario (``batch-fit merge
--recipe`` with per-level substitutive reduction): timepoints are stacked via
:meth:`GSplatData.embed_dimension` with ``sigma=0.0`` — which the Cholesky
embed regularises to a *near-delta* axis (sigma = 1e-7, variance 1e-14) — and
the stacked axis is a hard ``coarsen_dims`` barrier. The invariant pinned here
is stronger than group purity (splats sit ON their barrier value) and stronger
than the coarsest-count floor (>= 1 splat per group): **no barrier group may
vanish at any substitutive level**, under the production defaults
(``conserve_mass=True``, ``coverage_inflation=3.0``) and the proportional
per-dim Cholesky ridge, which all interact with the near-delta axis.
"""

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.substitutive import make_substitutive_lod


def _stacked_timepoints(
    n_per: int = 40, n_t: int = 2, seed: int = 0, spread: float = 50.0
) -> GSplatData:
    """3D random splats per timepoint, lifted to 4D exactly like the batch
    merge does (``embed_dimension(t, sigma=0.0)`` -> sigma_t = 1e-7), with the
    stacked-timepoint axis appended LAST."""
    rng = np.random.default_rng(seed)
    parts = []
    for t in range(n_t):
        n = n_per
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        d3 = GSplatData(
            centers=rng.uniform(0, spread, (n, 3)).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        )
        parts.append(d3.embed_dimension(float(t), sigma=0.0))
    return GSplatData(
        centers=np.concatenate([p.centers for p in parts]),
        amplitudes=np.concatenate([p.amplitudes for p in parts]),
        cholesky_factors=np.concatenate([p.cholesky_factors for p in parts]),
    )


def _assert_all_groups_at_every_level(out: GSplatData, n_t: int) -> None:
    """Every substitutive level contains every timepoint value, with finite
    strictly-positive amplitudes (a NaN/zero amplitude is how a group would
    silently vanish through the zero-amplitude cull)."""
    expected = np.arange(n_t)
    for s in range(out.n_substitutive):
        lvl = out.at_substitutive(s).flattened()
        amps = np.asarray(lvl.amplitudes)
        assert np.all(np.isfinite(amps)), f"level {s}: non-finite amplitudes"
        assert np.all(amps > 0), f"level {s}: non-positive amplitudes"
        tvals = np.unique(np.round(np.asarray(lvl.centers)[:, -1]).astype(int))
        np.testing.assert_array_equal(
            tvals,
            expected,
            err_msg=f"level {s}: barrier groups {tvals.tolist()} != "
            f"{expected.tolist()} — a timepoint vanished from this level",
        )


class TestBarrierCoverage:
    def test_every_group_at_every_level_default_knobs(self):
        """Production defaults (conserve_mass=True, coverage_inflation=3.0)."""
        data = _stacked_timepoints(n_per=40, n_t=2)
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=3,
            device="cpu",
            coarsen_dims=[0, 1, 2],
        )
        _assert_all_groups_at_every_level(out, n_t=2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"conserve_mass": False},
            {"coverage_inflation": 1.0},
            {"conserve_mass": False, "coverage_inflation": 1.0},
        ],
        ids=["no-conserve-mass", "no-inflation", "both-off"],
    )
    def test_every_group_at_every_level_knob_matrix(self, kwargs):
        """The invariant must hold regardless of the mass/inflation knobs."""
        data = _stacked_timepoints(n_per=40, n_t=2)
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=3,
            device="cpu",
            coarsen_dims=[0, 1, 2],
            **kwargs,
        )
        _assert_all_groups_at_every_level(out, n_t=2)

    def test_many_groups_deep_levels(self):
        """More groups than the naive coarsest count (N/K^L < n_t): the >=1
        per-group allocation floor must keep all groups alive."""
        n_t = 5
        data = _stacked_timepoints(n_per=30, n_t=n_t, seed=3)
        # 150 splats, K=4, L=4 -> naive coarsest ceil(150/256) = 1 < 5 groups.
        out = make_substitutive_lod(
            data,
            compression_factor=4,
            levels=4,
            device="cpu",
            coarsen_dims=[0, 1, 2],
        )
        _assert_all_groups_at_every_level(out, n_t=n_t)
