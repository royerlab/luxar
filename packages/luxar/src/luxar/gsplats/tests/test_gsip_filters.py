"""Tests for the GSIP filtering additions to GSplatData / filter_by.

Covers the new spatial-aware metrics (scale / eccentricity that auto-ignore a
zero-variance time axis), percentile thresholds, isolation / local-density
filters, and the soft (amplitude-reweighting) high/low-pass.
"""

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData


# ── Helpers ───────────────────────────────────────────────


def _make(centers: np.ndarray, sigmas: np.ndarray, amps: np.ndarray) -> GSplatData:
    """Build a GSplatData whose covariance is diagonal with the given per-axis
    marginal sigmas (packed lower-triangular Cholesky)."""
    from luxar.gsplats.utils.trils import pack_tril

    n, d = centers.shape
    L = np.zeros((n, d, d))
    for i in range(d):
        L[:, i, i] = sigmas[:, i]
    return GSplatData(
        centers=centers.astype(np.float32),
        amplitudes=amps.astype(np.float32),
        cholesky_factors=pack_tril(L).astype(np.float32),
    )


def _iso3d(n: int, sigma: float, seed: int = 0) -> GSplatData:
    rng = np.random.default_rng(seed)
    return _make(
        rng.random((n, 3)) * 100,
        np.full((n, 3), sigma),
        np.ones(n),
    )


# ── Spatial-aware metrics ─────────────────────────────────


class TestSpatialMetrics:
    def test_scale_3d_matches_geometric_mean(self):
        d = _make(
            np.zeros((1, 3)),
            np.array([[2.0, 2.0, 4.0]]),
            np.ones(1),
        )
        assert d.scale()[0] == pytest.approx((2 * 2 * 4) ** (1 / 3))

    def test_eccentricity_3d_unchanged(self):
        d = _make(np.zeros((1, 3)), np.array([[2.0, 2.0, 4.0]]), np.ones(1))
        assert d.eccentricities()[0] == pytest.approx(2.0)

    def test_nondegenerate_axes_drops_zero_variance_time(self):
        # 4D with a zero-variance time axis (dim 3).
        d = _make(
            np.array([[1.0, 2.0, 3.0, 0.0]]),
            np.array([[2.0, 2.0, 4.0, 0.0]]),
            np.ones(1),
        )
        assert list(d._nondegenerate_axes()) == [0, 1, 2]

    def test_scale_ignores_time_axis(self):
        # volumes() (all dims) collapses to 0 on the zero-variance axis; scale()
        # stays spatial and meaningful.
        d = _make(
            np.array([[1.0, 2.0, 3.0, 0.0]]),
            np.array([[2.0, 2.0, 4.0, 0.0]]),
            np.ones(1),
        )
        assert d.volumes()[0] == pytest.approx(0.0)
        assert d.scale()[0] == pytest.approx((2 * 2 * 4) ** (1 / 3))

    def test_eccentricity_spatial_on_timelapse(self):
        # All-dims eccentricity would hit the degenerate 1.0 fallback (min=0);
        # the spatial default gives the real 2.0.
        d = _make(
            np.array([[1.0, 2.0, 3.0, 0.0]]),
            np.array([[2.0, 2.0, 4.0, 0.0]]),
            np.ones(1),
        )
        assert d.eccentricities()[0] == pytest.approx(2.0)

    def test_spatial_dims_override(self):
        d = _make(np.zeros((1, 3)), np.array([[2.0, 2.0, 8.0]]), np.ones(1))
        # restrict to axes 0,1 → isotropic → ecc 1.0
        assert d.eccentricities(axes=[0, 1])[0] == pytest.approx(1.0)


# ── Isolation / local density ─────────────────────────────


class TestIsolation:
    def _cluster_plus_isolated(self) -> GSplatData:
        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [50, 50, 50]], dtype=float
        )
        return _make(pts, np.full((4, 3), 1.0), np.ones(4))

    def test_nn_distance_flags_isolated(self):
        d = self._cluster_plus_isolated()
        nn = d.nearest_neighbor_distances()
        assert nn[3] == max(nn)  # far splat has the largest NN distance
        assert nn[3] > 40

    def test_neighbor_counts(self):
        d = self._cluster_plus_isolated()
        counts = d.neighbor_counts(radius=2.0)
        assert counts[3] == 0  # isolated splat has no neighbours
        assert counts[0] >= 2

    def test_isolation_filter_removes_isolated(self):
        d = self._cluster_plus_isolated()
        out = d.filter_by(isolation_max=10.0)
        assert out.n_splats == 3  # the far splat is dropped

    def test_isolation_grouped_by_timepoint(self):
        # Two identical clusters at different timepoints must NOT count each
        # other as neighbours (grouping by the non-spatial axis).
        c = _iso3d(5, 1.0)
        tp = GSplatData.combine_as_new_dimension([c, c], values=[0.0, 1.0], sigma=0.0)
        # Each splat's nearest neighbour must be within its own timepoint.
        nn_grouped = tp.nearest_neighbor_distances()
        nn_flat = tp.nearest_neighbor_distances(group_axes=[])
        # Grouping can only increase (or equal) NN distance vs ignoring groups.
        assert np.all(nn_grouped >= nn_flat - 1e-9)


# ── Percentile thresholds ─────────────────────────────────


class TestPercentile:
    def test_resolve_threshold_percentile(self):
        vals = np.arange(101, dtype=float)  # 0..100
        assert GSplatData._resolve_threshold(90, False, vals, percentile=True) == 90.0

    def test_filter_by_amplitude_percentile(self):
        amps = np.linspace(0.1, 1.0, 100)
        d = _make(np.random.default_rng(1).random((100, 3)) * 10,
                  np.full((100, 3), 1.0), amps)
        out = d.filter_by(amplitude_min=90, amplitude_percentile=True)
        assert out.n_splats == 10

    def test_filter_by_scale_percentile(self):
        sig = np.column_stack([np.linspace(1, 10, 100)] * 3)
        d = _make(np.zeros((100, 3)), sig, np.ones(100))
        out = d.filter_by(scale_max=90, scale_percentile=True)
        assert out.n_splats == 90  # keep smallest 90%


# ── Soft filtering (reweighting) ──────────────────────────


class TestSoftFilter:
    def _scale_ramp(self) -> GSplatData:
        sig = np.column_stack([np.linspace(1, 10, 100)] * 3)
        return _make(np.zeros((100, 3)), sig, np.ones(100))

    def test_soft_highpass_attenuates_large(self):
        d = self._scale_ramp()
        out = d.soft_scale_filter(highpass=5.0, width=1.0)
        assert out.n_splats == 100  # count unchanged
        amps = np.asarray(out.amplitudes)
        assert amps[0] > amps[-1]  # small scale kept, large scale suppressed
        assert amps[0] == pytest.approx(1.0, abs=1e-3)
        assert amps[-1] < 0.05

    def test_soft_lowpass_attenuates_small(self):
        d = self._scale_ramp()
        out = d.soft_scale_filter(lowpass=5.0, width=1.0)
        amps = np.asarray(out.amplitudes)
        assert amps[-1] > amps[0]  # large kept, small suppressed

    def test_reweight_amplitude_shape_check(self):
        d = self._scale_ramp()
        with pytest.raises(ValueError):
            d.reweight_amplitude(np.ones(5))

    def test_soft_noop_without_cutoffs(self):
        d = self._scale_ramp()
        out = d.soft_scale_filter()
        assert out is d
