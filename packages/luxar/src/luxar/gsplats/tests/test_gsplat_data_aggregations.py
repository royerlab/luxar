"""Tests for GSplatData computed properties, filtering, and reshape ops.

Partition out from ``test_gsplat_data.py`` to keep that file focused on
the core data API. This file covers:

- Computed properties: ``TestVolumes``, ``TestMasses``,
  ``TestMarginalSigmas``, ``TestEccentricities``
- Filtering / slicing: ``TestFilter``, ``TestFilterBy``, ``TestSliceBy``
- Reshape ops: ``TestConcatenate``, ``TestPartition``,
  ``TestEmbedDimension``, ``TestCombineAsNewDimension``
"""

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData

from ._gsplat_data_helpers import _make_2d_gsplat, _make_3d_gsplat, _make_empty_gsplat


class TestVolumes:
    def test_isotropic_identity_3d(self):
        """Identity Cholesky [1,0,1,0,0,1] has det(L)=1, volume=1."""
        gs = _make_3d_gsplat(n=3)
        vols = gs.volumes()
        assert vols.shape == (3,)
        # Identity diagonal -> det(L)=1, det(Sigma)=1, vol=1^(1/3)=1
        assert np.allclose(vols, 1.0)

    def test_scaled_diagonal(self):
        """Diagonal [2,0,3,0,0,4] -> det(L)=24, vol=(24^2)^(1/3)."""
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32),
        )
        vols = gs.volumes()
        expected = abs(24**2) ** (1.0 / 3)
        assert np.allclose(vols[0], expected)

    def test_2d(self):
        gs = _make_2d_gsplat(n=3)
        vols = gs.volumes()
        assert vols.shape == (3,)

    def test_empty(self):
        gs = _make_empty_gsplat()
        assert gs.volumes().shape == (0,)


class TestPrincipalRadii:
    """Per-splat element radius (truncation_radius × semi-axis) for LOD switching."""

    def test_anisotropic_largest_semi_axis(self):
        # Axis-aligned diagonal Cholesky [2,0,3,0,0,4] → Σ diag (4, 9, 16),
        # eigenvalues = variances; largest semi-axis = sqrt(16) = 4, × truncation.
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32),
            truncation_radius=3.0,
        )
        r = gs.principal_radii(anisotropy=True)
        assert r.shape == (1,)
        assert np.allclose(r[0], 3.0 * 4.0)

    def test_isotropic_geometric_mean(self):
        # Same splat, isotropic mode → geometric-mean semi-axis
        # (2·3·4)^(1/3) = 24^(1/3), × truncation.
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32),
            truncation_radius=3.0,
        )
        r = gs.principal_radii(anisotropy=False)
        assert np.allclose(r[0], 3.0 * (24.0 ** (1.0 / 3.0)))

    def test_anisotropic_ge_isotropic(self):
        gs = _make_3d_gsplat(n=8)
        assert np.all(
            gs.principal_radii(anisotropy=True)
            >= gs.principal_radii(anisotropy=False) - 1e-6
        )

    def test_scales_with_truncation(self):
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32),
            truncation_radius=5.0,
        )
        assert np.allclose(gs.principal_radii(anisotropy=True)[0], 5.0 * 4.0)

    def test_rotation_invariant_recovers_true_semi_axes(self):
        """A ROTATED (off-diagonal Cholesky) covariance with principal semi-axes
        (5,2,1): anisotropy=True must recover the largest semi-axis (5)·truncation
        and anisotropy=False the geometric mean — both rotation-invariant. Guards
        against using axis-aligned marginal sigmas (which an axis-aligned-only test
        would not catch — marginal σ ≠ principal semi-axis once rotated)."""
        from luxar.gsplats.utils.trils import pack_tril

        rng = np.random.default_rng(1)
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))  # random rotation
        axes = np.array([5.0, 2.0, 1.0])
        sigma = q @ np.diag(axes**2) @ q.T
        chol = pack_tril(np.linalg.cholesky(sigma)[None, :, :])
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=chol.astype(np.float32),
            truncation_radius=3.0,
        )
        assert np.allclose(gs.principal_radii(anisotropy=True)[0], 3.0 * 5.0, atol=1e-2)
        assert np.allclose(
            gs.principal_radii(anisotropy=False)[0],
            3.0 * (5 * 2 * 1) ** (1 / 3),
            atol=1e-2,
        )

    def test_empty(self):
        assert _make_empty_gsplat().principal_radii().shape == (0,)


class TestMasses:
    def test_mass_is_amplitude_times_det_sigma_root_d(self):
        # Independent hand-computed expectation (does NOT call volumes(), so a
        # mutation of either masses() or volumes() is caught). Cholesky packing
        # is lower-triangular [L00,L10,L11,L20,L21,L22]; the diagonal sets the
        # characteristic length volume = det(Sigma)^(1/d) = (prod(diag)^2)^(1/d).
        gs = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.array([0.5, 2.0], dtype=np.float32),
            cholesky_factors=np.array(
                [[2, 0, 3, 0, 0, 4], [1, 0, 1, 0, 0, 1]], dtype=np.float32
            ),
        )
        d = 3
        vol0 = ((2.0 * 3.0 * 4.0) ** 2) ** (1.0 / d)  # = 576 ** (1/3)
        vol1 = ((1.0 * 1.0 * 1.0) ** 2) ** (1.0 / d)  # = 1.0
        expected = np.array([0.5 * vol0, 2.0 * vol1])
        np.testing.assert_allclose(gs.masses(), expected, rtol=1e-6)

    def test_zero_amplitudes(self):
        gs = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.zeros(2, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
        )
        assert np.allclose(gs.masses(), 0.0)


class TestMarginalSigmas:
    def test_identity(self):
        """Identity Cholesky gives sigma=[1,1,1]."""
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
        )
        sigmas = gs.marginal_sigmas()
        assert sigmas.shape == (1, 3)
        assert np.allclose(sigmas[0], [1, 1, 1])

    def test_diagonal(self):
        """Diagonal [2,0,3,0,0,4] gives sigma=[2,3,4]."""
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32),
        )
        assert np.allclose(gs.marginal_sigmas()[0], [2, 3, 4])

    def test_off_diagonal(self):
        """L=[[2,0,0],[1,3,0],[0,0,4]] -> sigma_0=2, sigma_1=sqrt(1+9)=sqrt(10)."""
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2, 1, 3, 0, 0, 4]], dtype=np.float32),
        )
        sigmas = gs.marginal_sigmas()[0]
        assert np.allclose(sigmas[0], 2.0)
        assert np.allclose(sigmas[1], np.sqrt(1 + 9))
        assert np.allclose(sigmas[2], 4.0)

    def test_empty(self):
        gs = _make_empty_gsplat()
        assert gs.marginal_sigmas().shape == (0, 3)

    def test_2d(self):
        gs = _make_2d_gsplat(n=2)
        assert gs.marginal_sigmas().shape == (2, 2)


class TestEccentricities:
    def test_isotropic_is_one(self):
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
        )
        assert np.allclose(gs.eccentricities(), 1.0)

    def test_anisotropic(self):
        gs = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 4]], dtype=np.float32),
        )
        ecc = gs.eccentricities()
        assert ecc[0] == pytest.approx(4.0)  # max_sigma=4, min_sigma=1

    def test_empty(self):
        gs = _make_empty_gsplat()
        assert gs.eccentricities().shape == (0,)


# ── Filter ──────────────────────────────────────────────


class TestFilter:
    def test_filter_keeps_exactly_the_masked_splats(self):
        gs = _make_3d_gsplat(n=5)
        mask = np.array([True, True, False, True, False])
        filtered = gs.filter(mask)
        # Assert IDENTITY, not just the count — a mutant that kept the wrong
        # rows (or reordered) would pass a bare `n_splats == 3` check.
        assert filtered.n_splats == 3
        np.testing.assert_array_equal(filtered.centers, gs.centers[mask])
        np.testing.assert_array_equal(filtered.amplitudes, gs.amplitudes[mask])
        np.testing.assert_array_equal(
            filtered.cholesky_factors, gs.cholesky_factors[mask]
        )

    def test_all_true(self):
        gs = _make_3d_gsplat(n=3)
        filtered = gs.filter(np.ones(3, dtype=bool))
        assert filtered.n_splats == 3

    def test_all_false(self):
        gs = _make_3d_gsplat(n=3)
        filtered = gs.filter(np.zeros(3, dtype=bool))
        assert filtered.n_splats == 0

    def test_colors_preserved(self):
        gs = GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
            colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        )
        filtered = gs.filter(np.array([True, False, True]))
        assert np.allclose(filtered.colors[0], [1, 0, 0])
        assert np.allclose(filtered.colors[1], [0, 0, 1])

    def test_none_colors(self):
        gs = _make_3d_gsplat(n=3)
        filtered = gs.filter(np.array([True, False, True]))
        assert filtered.colors is None

    def test_mask_shape_validation(self):
        gs = _make_3d_gsplat(n=5)
        with pytest.raises(ValueError, match="Mask shape"):
            gs.filter(np.ones(3, dtype=bool))

    def test_with_computed_property(self):
        """Filter by volume end-to-end."""
        gs = _make_3d_gsplat(n=10)
        filtered = gs.filter(gs.volumes() > 0)
        assert filtered.n_splats <= gs.n_splats

    def test_empty(self):
        gs = _make_empty_gsplat()
        filtered = gs.filter(np.ones(0, dtype=bool))
        assert filtered.n_splats == 0

    # Audit G8 (python-gsplats-data-io-lod): non-boolean dtype mask
    # behaviour was untested. The implementation accepts int8 [1, 0, 1]
    # and applies bool-cast semantics (NOT numpy's fancy-indexing
    # default which would treat them as positional indices). Pin this
    # surprising-but-useful contract — a mutant that switched to
    # `.astype(np.int)` would change the meaning.
    def test_mask_dtype_int8_treated_as_boolean(self):
        """int8 [1, 0, 1] mask filters as boolean — selects 2 of 3 splats."""
        gs = _make_3d_gsplat(n=3)
        filtered = gs.filter(np.array([1, 0, 1], dtype=np.int8))
        # Boolean semantics: 1→True, 0→False → 2 splats survive.
        assert filtered.n_splats == 2


# ── FilterBy (multi-criteria) ──────────────────────────


class TestFilterBy:
    """Tests for filter_by() multi-criteria filtering."""

    def test_bbox(self):
        """Filter by bounding box keeps only splats inside."""
        gs = GSplatData(
            centers=np.array(
                [[10, 10, 10], [50, 50, 50], [90, 90, 90]], dtype=np.float32
            ),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
        )
        filtered = gs.filter_by(bbox=[(0, 60), (0, 60), (0, 60)])
        assert filtered.n_splats == 2
        assert np.all(filtered.centers[:, 0] <= 60)

    def test_volume_absolute(self):
        """Filter by absolute volume range."""
        # Create splats with different Cholesky diagonals → different volumes
        chol_small = np.array([0.5, 0, 0.5, 0, 0, 0.5], dtype=np.float32)
        chol_big = np.array([5.0, 0, 5.0, 0, 0, 5.0], dtype=np.float32)
        gs = GSplatData(
            centers=np.zeros((4, 3), dtype=np.float32),
            amplitudes=np.ones(4, dtype=np.float32),
            cholesky_factors=np.array(
                [chol_small, chol_small, chol_big, chol_big], dtype=np.float32
            ),
        )
        # volume = det(Sigma)^(1/d) * truncate
        # small: det(L)=0.125, det(Sigma)=0.015625, vol^(1/3)=0.25, *3=0.75
        # big: det(L)=125, det(Sigma)=15625, vol^(1/3)=25.0, *3=75.0
        filtered = gs.filter_by(volume_max=10.0)
        assert filtered.n_splats == 2  # Only the small ones

    def test_volume_normalized(self):
        """Normalized volume threshold maps to dataset range."""
        chol_small = np.array([1, 0, 1, 0, 0, 1], dtype=np.float32)
        chol_big = np.array([10, 0, 10, 0, 0, 10], dtype=np.float32)
        gs = GSplatData(
            centers=np.zeros((4, 3), dtype=np.float32),
            amplitudes=np.ones(4, dtype=np.float32),
            cholesky_factors=np.array(
                [chol_small, chol_small, chol_big, chol_big], dtype=np.float32
            ),
        )
        # With normalized=True, 0.5 should be the midpoint of volume range
        filtered = gs.filter_by(volume_max=0.5, volume_normalized=True)
        assert filtered.n_splats == 2  # Only the small ones

    def test_amplitude_absolute(self):
        """Filter by amplitude range."""
        gs = GSplatData(
            centers=np.zeros((5, 3), dtype=np.float32),
            amplitudes=np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (5, 1)
            ),
        )
        filtered = gs.filter_by(amplitude_min=0.4, amplitude_max=0.8)
        assert filtered.n_splats == 2  # 0.5 and 0.7

    def test_amplitude_normalized(self):
        """Normalized amplitude: 0.5 maps to midpoint of dataset range."""
        gs = GSplatData(
            centers=np.zeros((4, 3), dtype=np.float32),
            amplitudes=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (4, 1)
            ),
        )
        # normalized 0.5 → 0.0 + 0.5*(3.0-0.0) = 1.5
        filtered = gs.filter_by(amplitude_min=0.5, amplitude_normalized=True)
        assert filtered.n_splats == 2  # 2.0 and 3.0

    def test_eccentricity(self):
        """Filter by eccentricity."""
        # Isotropic: diag = [1,1,1], eccentricity ≈ 1
        chol_iso = np.array([1, 0, 1, 0, 0, 1], dtype=np.float32)
        # Elongated: diag = [10, 0, 1, 0, 0, 1] → sigma_0 >> sigma_1,2
        chol_elong = np.array([10, 0, 1, 0, 0, 1], dtype=np.float32)
        gs = GSplatData(
            centers=np.zeros((4, 3), dtype=np.float32),
            amplitudes=np.ones(4, dtype=np.float32),
            cholesky_factors=np.array(
                [chol_iso, chol_iso, chol_elong, chol_elong], dtype=np.float32
            ),
        )
        filtered = gs.filter_by(eccentricity_max=2.0)
        assert filtered.n_splats == 2  # Only isotropic

    def test_mass(self):
        """Filter by mass (amplitude * volume)."""
        gs = _make_3d_gsplat(n=10)
        masses = gs.masses()
        median_mass = float(np.median(masses))
        filtered = gs.filter_by(mass_min=median_mass)
        assert filtered.n_splats <= 10
        assert filtered.n_splats > 0

    def test_sigma_axis(self):
        """Filter by marginal sigma on a specific axis."""
        # axis 0 has large sigma, axes 1,2 have small sigma
        chol = np.array([10, 0, 1, 0, 0, 1], dtype=np.float32)
        gs = GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(chol, (3, 1)),
        )
        # sigma on axis 0 should be large (10), sigma on axis 1 should be ~1
        filtered = gs.filter_by(sigma_axis=0, sigma_max=5.0)
        assert filtered.n_splats == 0  # All have sigma_0 = 10 > 5

        filtered2 = gs.filter_by(sigma_axis=1, sigma_max=5.0)
        assert filtered2.n_splats == 3  # All pass

    def test_combined_criteria(self):
        """Multiple criteria combine with AND logic."""
        gs = GSplatData(
            centers=np.array(
                [[10, 10, 10], [50, 50, 50], [90, 90, 90]], dtype=np.float32
            ),
            amplitudes=np.array([0.1, 0.5, 0.9], dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
        )
        # bbox keeps first two, amplitude keeps last two → AND keeps only [50,50,50]
        filtered = gs.filter_by(
            bbox=[(0, 60), (0, 60), (0, 60)],
            amplitude_min=0.3,
        )
        assert filtered.n_splats == 1
        assert np.allclose(filtered.centers[0], [50, 50, 50])

    def test_empty_result(self):
        """Restrictive criteria can produce 0 splats."""
        gs = _make_3d_gsplat(n=5)
        filtered = gs.filter_by(amplitude_min=999.0)
        assert filtered.n_splats == 0

    def test_all_pass(self):
        """No criteria → all splats pass."""
        gs = _make_3d_gsplat(n=5)
        filtered = gs.filter_by()
        assert filtered.n_splats == 5

    def test_empty_input(self):
        """Empty input returns empty output."""
        gs = _make_empty_gsplat()
        filtered = gs.filter_by(amplitude_min=0.1)
        assert filtered.n_splats == 0

    def test_stats_updated(self):
        """Stats contain filtering metadata."""
        gs = _make_3d_gsplat(n=5)
        filtered = gs.filter_by(amplitude_min=0.3)
        assert filtered.stats["filtered"] is True
        assert filtered.stats["n_original"] == 5
        assert "n_removed" in filtered.stats
        assert "filter_criteria" in filtered.stats

    def test_preserves_colors(self):
        """Colors array is filtered when present."""
        gs = GSplatData(
            centers=np.array(
                [[0, 0, 0], [50, 50, 50], [100, 100, 100]], dtype=np.float32
            ),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        )
        filtered = gs.filter_by(bbox=[(0, 60), (0, 60), (0, 60)])
        assert filtered.n_splats == 2
        assert filtered.colors is not None
        assert np.array_equal(filtered.colors[0], [255, 0, 0])
        assert np.array_equal(filtered.colors[1], [0, 255, 0])

    def test_invalid_bbox_length(self):
        """Wrong bbox length raises ValueError."""
        gs = _make_3d_gsplat(n=3)
        with pytest.raises(ValueError, match="bbox has 2 dimensions"):
            gs.filter_by(bbox=[(0, 10), (0, 10)])  # 2 dims for 3D data

    def test_sigma_without_axis(self):
        """sigma_min without sigma_axis raises ValueError."""
        gs = _make_3d_gsplat(n=3)
        with pytest.raises(ValueError, match="sigma_min/sigma_max require sigma_axis"):
            gs.filter_by(sigma_min=1.0)


# ── SliceBy ────────────────────────────────────────────


class TestSliceBy:
    """Tests for slice_by() coordinate-based slicing."""

    def test_basic_slice(self):
        gs = GSplatData(
            centers=np.array(
                [[10, 10, 10], [50, 50, 50], [90, 90, 90]], dtype=np.float32
            ),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
        )
        sliced = gs.slice_by([slice(0, 60), slice(0, 60), slice(0, 60)])
        assert sliced.n_splats == 2

    def test_open_start(self):
        """slice(None, 50) keeps centers <= 50."""
        gs = GSplatData(
            centers=np.array([[10, 0, 0], [50, 0, 0], [90, 0, 0]], dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
        )
        sliced = gs.slice_by([slice(None, 50), slice(None, None), slice(None, None)])
        assert sliced.n_splats == 2  # 10 and 50

    def test_open_end(self):
        """slice(50, None) keeps centers >= 50."""
        gs = GSplatData(
            centers=np.array([[10, 0, 0], [50, 0, 0], [90, 0, 0]], dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
        )
        sliced = gs.slice_by([slice(50, None), slice(None, None), slice(None, None)])
        assert sliced.n_splats == 2  # 50 and 90

    def test_float_values(self):
        """Float coordinate ranges work."""
        gs = GSplatData(
            centers=np.array([[1.5, 0, 0], [2.7, 0, 0], [4.2, 0, 0]], dtype=np.float32),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
        )
        sliced = gs.slice_by([slice(2.0, 3.0), slice(None, None), slice(None, None)])
        assert sliced.n_splats == 1
        assert np.isclose(sliced.centers[0, 0], 2.7)

    def test_all_pass(self):
        gs = _make_3d_gsplat(n=5)
        sliced = gs.slice_by([slice(None, None)] * 3)
        assert sliced.n_splats == 5

    def test_empty_result(self):
        gs = _make_3d_gsplat(n=5)
        sliced = gs.slice_by([slice(9999, 10000), slice(None, None), slice(None, None)])
        assert sliced.n_splats == 0

    def test_wrong_ndim(self):
        gs = _make_3d_gsplat(n=3)
        with pytest.raises(ValueError, match="Expected 3 slices"):
            gs.slice_by([slice(0, 10), slice(0, 10)])  # 2 slices for 3D

    def test_preserves_colors(self):
        gs = GSplatData(
            centers=np.array(
                [[0, 0, 0], [50, 50, 50], [100, 100, 100]], dtype=np.float32
            ),
            amplitudes=np.ones(3, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (3, 1)
            ),
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
        )
        sliced = gs.slice_by([slice(0, 60), slice(0, 60), slice(0, 60)])
        assert sliced.n_splats == 2
        assert sliced.colors is not None
        assert np.array_equal(sliced.colors[1], [0, 255, 0])


# ── Concatenate ─────────────────────────────────────────


class TestConcatenate:
    def test_two_datasets(self):
        gs1 = _make_3d_gsplat(n=3, seed=1)
        gs2 = _make_3d_gsplat(n=5, seed=2)
        result = GSplatData.concatenate([gs1, gs2])
        assert result.n_splats == 8
        assert result.ndim == 3

    def test_preserves_data(self):
        gs1 = _make_3d_gsplat(n=2, seed=1)
        gs2 = _make_3d_gsplat(n=2, seed=2)
        result = GSplatData.concatenate([gs1, gs2])
        assert np.allclose(result.centers[:2], gs1.centers)
        assert np.allclose(result.centers[2:], gs2.centers)

    def test_all_colors(self):
        gs1 = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            colors=np.array([[1, 0, 0]], dtype=np.float32),
        )
        gs2 = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            colors=np.array([[0, 0, 1]], dtype=np.float32),
        )
        result = GSplatData.concatenate([gs1, gs2])
        assert result.colors is not None
        assert np.allclose(result.colors[0], [1, 0, 0])
        assert np.allclose(result.colors[1], [0, 0, 1])

    def test_no_colors(self):
        gs1 = _make_3d_gsplat(n=2, seed=1)
        gs2 = _make_3d_gsplat(n=2, seed=2)
        result = GSplatData.concatenate([gs1, gs2])
        assert result.colors is None

    def test_mixed_colors_fills_white(self):
        gs_with = GSplatData(
            centers=np.zeros((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            colors=np.array([[1, 0, 0]], dtype=np.float32),
        )
        gs_without = _make_3d_gsplat(n=1, seed=1)
        result = GSplatData.concatenate([gs_with, gs_without])
        assert result.colors is not None
        assert np.allclose(result.colors[1], [1, 1, 1])  # white fill

    def test_ndim_mismatch(self):
        gs_3d = _make_3d_gsplat(n=1)
        gs_2d = _make_2d_gsplat(n=1)
        with pytest.raises(ValueError, match="Dimensionality mismatch"):
            GSplatData.concatenate([gs_3d, gs_2d])

    def test_empty_list(self):
        with pytest.raises(ValueError, match="At least one"):
            GSplatData.concatenate([])

    def test_single_dataset(self):
        gs = _make_3d_gsplat(n=5)
        result = GSplatData.concatenate([gs])
        assert result.n_splats == 5

    @staticmethod
    def _make_2sub_pyramid(seed: int = 0) -> GSplatData:
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

        rng = np.random.RandomState(seed)

        def _lod(n: int) -> AdditiveSubLOD:
            return AdditiveSubLOD(
                centers=rng.rand(n, 3).astype(np.float32),
                amplitudes=rng.rand(n).astype(np.float32),
                cholesky_factors=np.tile(
                    np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
                ),
            )

        fine = SubstitutiveLevel(additive_sublods=[_lod(8)], compression_factor=1)
        coarse = SubstitutiveLevel(
            additive_sublods=[_lod(2)],
            compression_factor=4,
            parent_method="greedy",
            level_index=1,
        )
        return GSplatData.from_substitutive_levels([fine, coarse])

    def test_preserves_substitutive_levels(self):
        """H3: concatenating pyramids preserves all substitutive levels."""
        p = self._make_2sub_pyramid(seed=1)
        assert p.n_substitutive == 2
        result = GSplatData.concatenate([p, self._make_2sub_pyramid(seed=2)])
        assert result.n_substitutive == 2
        assert result.at_substitutive(0).n_splats == 16  # 8 + 8 fine
        assert result.at_substitutive(1).n_splats == 4  # 2 + 2 coarse

    def test_substitutive_count_mismatch_raises(self):
        """H3: mismatched n_substitutive is rejected, not silently dropped."""
        pyramid = self._make_2sub_pyramid()
        flat = _make_3d_gsplat(n=5)  # n_substitutive == 1
        with pytest.raises(ValueError, match="Substitutive-level count mismatch"):
            GSplatData.concatenate([pyramid, flat])

    def test_float32_dtype_guard(self):
        """H3: mixed float precision must not promote the result to float64."""
        gs32 = _make_3d_gsplat(n=2, seed=1)
        gs64 = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float64),
            amplitudes=np.ones(2, dtype=np.float64),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float64), (2, 1)
            ),
        )
        result = GSplatData.concatenate([gs32, gs64])
        assert result.centers.dtype == np.float32
        assert result.amplitudes.dtype == np.float32
        assert result.cholesky_factors.dtype == np.float32

    def test_all_empty_returns_fresh_instance(self):
        """H3: all-empty input returns a new object, not an aliased input."""
        e1 = _make_empty_gsplat(ndim=3)
        e2 = _make_empty_gsplat(ndim=3)
        result = GSplatData.concatenate([e1, e2])
        assert result is not e1 and result is not e2
        assert result.n_splats == 0
        assert result.ndim == 3


class TestEmbedDimension:
    def test_scalar_value(self):
        gs = _make_3d_gsplat(n=3)
        result = gs.embed_dimension(5.0)
        assert result.ndim == 4
        assert result.n_splats == 3
        assert np.allclose(result.centers[:, 3], 5.0)

    def test_per_splat_values(self):
        gs = _make_3d_gsplat(n=3)
        vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = gs.embed_dimension(vals)
        assert np.allclose(result.centers[:, 3], vals)

    def test_ndim_increases(self):
        gs = _make_2d_gsplat(n=2)
        result = gs.embed_dimension(0.0)
        assert result.ndim == 3
        # 2D chol has k=3, 3D chol has k=6
        assert result.cholesky_factors.shape[1] == 6

    def test_original_dims_preserved(self):
        gs = _make_3d_gsplat(n=2)
        result = gs.embed_dimension(0.0)
        assert np.allclose(result.centers[:, :3], gs.centers)

    def test_amplitudes_unchanged(self):
        gs = _make_3d_gsplat(n=2)
        result = gs.embed_dimension(0.0)
        assert result.amplitudes is gs.amplitudes

    def test_values_shape_validation(self):
        gs = _make_3d_gsplat(n=3)
        with pytest.raises(ValueError, match="values shape"):
            gs.embed_dimension(np.array([1.0, 2.0]))  # wrong size

    def test_empty(self):
        gs = _make_empty_gsplat(ndim=3)
        result = gs.embed_dimension(0.0)
        assert result.ndim == 4
        assert result.n_splats == 0


# ── Combine as new dimension ─────────────────────────────


class TestCombineAsNewDimension:
    """Tests for GSplatData.combine_as_new_dimension()."""

    def test_basic_3d_to_4d(self):
        """Combine three 3D datasets into one 4D dataset."""
        datasets = [_make_3d_gsplat(n=4, seed=i) for i in range(3)]
        result = GSplatData.combine_as_new_dimension(datasets)
        assert result.ndim == 4
        assert result.n_splats == 12
        # Cholesky factors for 4D: k = 4*5/2 = 10
        assert result.cholesky_factors.shape[1] == 10

    def test_default_values_are_sequential(self):
        """When values=None, the new dim gets 0.0, 1.0, 2.0, ..."""
        datasets = [_make_3d_gsplat(n=2, seed=i) for i in range(3)]
        result = GSplatData.combine_as_new_dimension(datasets)
        # New dim is the last column (index 3)
        time_col = result.centers[:, 3]
        assert np.allclose(time_col[:2], 0.0)
        assert np.allclose(time_col[2:4], 1.0)
        assert np.allclose(time_col[4:6], 2.0)

    def test_explicit_values(self):
        """Custom coordinate values for the new dimension."""
        datasets = [_make_3d_gsplat(n=2, seed=i) for i in range(2)]
        result = GSplatData.combine_as_new_dimension(datasets, values=[10.0, 20.0])
        time_col = result.centers[:, 3]
        assert np.allclose(time_col[:2], 10.0)
        assert np.allclose(time_col[2:4], 20.0)

    def test_sigma_zero_discrete(self):
        """sigma=0 gives near-zero variance in the new dimension."""
        gs = _make_3d_gsplat(n=3)
        result = GSplatData.combine_as_new_dimension([gs], sigma=0.0)
        assert result.ndim == 4
        # The time dimension should have essentially zero variance.
        # Cholesky for 4D: last diagonal element at index 9 (d=3: start=6, diag at 6+3=9)
        # L[3,3] should be ~1e-7 (the epsilon used for positive-definiteness)
        l_33 = result.cholesky_factors[:, 9]
        assert np.all(np.abs(l_33) < 1e-4)

    def test_sigma_positive(self):
        """sigma>0 gives nonzero extent in the new dimension."""
        gs = _make_3d_gsplat(n=3)
        result = GSplatData.combine_as_new_dimension([gs], sigma=2.0)
        # L[3,3] should be ~2.0
        l_33 = result.cholesky_factors[:, 9]
        assert np.allclose(l_33, 2.0, atol=0.1)

    def test_preserves_spatial_centers(self):
        """Original spatial coordinates are preserved."""
        gs = _make_3d_gsplat(n=5, seed=42)
        result = GSplatData.combine_as_new_dimension([gs], values=[7.0])
        assert np.allclose(result.centers[:, :3], gs.centers)

    def test_preserves_colors(self):
        """Colors are correctly concatenated."""
        gs1 = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.ones(2, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
            colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        )
        gs2 = GSplatData(
            centers=np.ones((1, 3), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
            colors=np.array([[0, 0, 1]], dtype=np.float32),
        )
        result = GSplatData.combine_as_new_dimension([gs1, gs2])
        assert result.colors is not None
        assert np.allclose(result.colors[0], [1, 0, 0])
        assert np.allclose(result.colors[2], [0, 0, 1])

    def test_2d_to_3d(self):
        """Also works for 2D → 3D."""
        datasets = [_make_2d_gsplat(n=3, seed=i) for i in range(2)]
        result = GSplatData.combine_as_new_dimension(datasets)
        assert result.ndim == 3
        assert result.n_splats == 6

    def test_empty_datasets_handled(self):
        """Empty datasets in the list don't break concatenation."""
        gs_real = _make_3d_gsplat(n=5, seed=1)
        gs_empty = _make_empty_gsplat(ndim=3)
        result = GSplatData.combine_as_new_dimension([gs_real, gs_empty])
        assert result.ndim == 4
        assert result.n_splats == 5

    def test_single_dataset(self):
        """Works with a single dataset (adds the dimension)."""
        gs = _make_3d_gsplat(n=5)
        result = GSplatData.combine_as_new_dimension([gs])
        assert result.ndim == 4
        assert result.n_splats == 5

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            GSplatData.combine_as_new_dimension([])

    def test_values_length_mismatch_raises(self):
        datasets = [_make_3d_gsplat(n=2, seed=i) for i in range(3)]
        with pytest.raises(ValueError, match="must match"):
            GSplatData.combine_as_new_dimension(datasets, values=[0.0, 1.0])

    def test_values_not_list_raises(self):
        datasets = [_make_3d_gsplat(n=2)]
        with pytest.raises(TypeError, match="must be a list"):
            GSplatData.combine_as_new_dimension(datasets, values=5.0)
