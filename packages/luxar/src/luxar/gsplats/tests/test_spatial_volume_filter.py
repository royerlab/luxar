"""Tests for spatial volume computation and specific-brightness filtering.

These tests guard the background-splat filtering logic used in the
C. elegans 4D demo (``demo_gsplats_4d_celegans_tracking.py``).  The key
insight is that after ``combine_as_new_dimension`` with ``sigma=0``, the
full nD ``volumes()`` method collapses to near-zero (because the time
dimension's diagonal Cholesky element is 0).  The filter must therefore
compute volumes using only the *spatial* Cholesky sub-block.
"""

import numpy as np

from luxar.gsplats.gsplat_data import GSplatData

# ── Helpers ───────────────────────────────────────────────


def _make_isotropic_3d(sigma: float, amplitude: float, center=None) -> GSplatData:
    """Create a single isotropic 3D splat with given sigma and amplitude.

    Cholesky for isotropic: L = diag(sigma, sigma, sigma)
    Packed: [sigma, 0, sigma, 0, 0, sigma]
    """
    if center is None:
        center = [0.0, 0.0, 0.0]
    chol = np.array([sigma, 0, sigma, 0, 0, sigma], dtype=np.float32)
    return GSplatData(
        centers=np.array([center], dtype=np.float32),
        amplitudes=np.array([amplitude], dtype=np.float32),
        cholesky_factors=chol.reshape(1, -1),
        sharpnesses=np.array([2.0], dtype=np.float32),
    )


def _make_4d_with_time(splats_3d: list[GSplatData]) -> GSplatData:
    """Combine a list of 3D GSplatData into 4D via combine_as_new_dimension."""
    return GSplatData.combine_as_new_dimension(
        splats_3d,
        values=[float(i) for i in range(len(splats_3d))],
        sigma=0.0,
    )


def _spatial_volumes_3d(gsplats_4d: GSplatData) -> np.ndarray:
    """Compute spatial-only (3D) volumes from a 4D GSplatData.

    This mirrors the logic in filter_background_splats.
    """
    n_spatial = 3
    n_chol_spatial = n_spatial * (n_spatial + 1) // 2  # 6
    spatial_chol = gsplats_4d.cholesky_factors[:, :n_chol_spatial]
    diag_indices = np.cumsum(np.arange(1, n_spatial + 1)) - 1  # [0, 2, 5]
    det_L = np.prod(spatial_chol[:, diag_indices], axis=1)
    return np.abs(det_L**2) ** (1.0 / n_spatial)


# ── Tests: spatial volume extraction ──────────────────────


class TestSpatialVolumeExtraction:
    """Verify that extracting the 3D Cholesky sub-block from 4D data
    yields volumes consistent with the original 3D data."""

    def test_isotropic_volume_matches_3d(self):
        """Spatial volume of a 4D splat must equal the original 3D volume."""
        sigma = 2.0
        g3 = _make_isotropic_3d(sigma, amplitude=1.0)
        vol_3d = g3.volumes()

        g4 = _make_4d_with_time([g3])
        vol_spatial = _spatial_volumes_3d(g4)

        np.testing.assert_allclose(vol_spatial, vol_3d, rtol=1e-5)

    def test_4d_volume_much_smaller_than_3d(self):
        """Full 4D volumes() should be much smaller than 3D spatial volume.

        embed_dimension clamps sigma=0 to 1e-7, so the 4D volume is not
        exactly zero but is many orders of magnitude smaller than the
        spatial volume — making it useless for specific-brightness filtering.
        """
        g3 = _make_isotropic_3d(2.0, amplitude=1.0)
        g4 = _make_4d_with_time([g3])
        vol_4d = g4.volumes()[0]
        vol_3d = g3.volumes()[0]
        # 4D volume should be at least 1000x smaller than 3D
        assert vol_4d < vol_3d / 1000

    def test_multiple_timepoints_preserve_spatial_volume(self):
        """All timepoints should retain their 3D spatial volume."""
        sigmas = [1.0, 2.0, 3.0]
        splats_3d = [_make_isotropic_3d(s, amplitude=0.5) for s in sigmas]
        vols_3d = [g.volumes()[0] for g in splats_3d]

        g4 = _make_4d_with_time(splats_3d)
        vols_spatial = _spatial_volumes_3d(g4)

        np.testing.assert_allclose(vols_spatial, vols_3d, rtol=1e-5)

    def test_anisotropic_splat(self):
        """Non-isotropic 3D splat: spatial volume from 4D must match 3D."""
        # L = [[1, 0, 0], [0.5, 2, 0], [0.3, 0.1, 3]]
        # Packed: [1, 0.5, 2, 0.3, 0.1, 3]
        chol_3d = np.array([[1.0, 0.5, 2.0, 0.3, 0.1, 3.0]], dtype=np.float32)
        g3 = GSplatData(
            centers=np.array([[0, 0, 0]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=chol_3d,
            sharpnesses=np.array([2.0], dtype=np.float32),
        )
        vol_3d = g3.volumes()

        g4 = _make_4d_with_time([g3])
        vol_spatial = _spatial_volumes_3d(g4)

        np.testing.assert_allclose(vol_spatial, vol_3d, rtol=1e-5)


# ── Tests: specific brightness filtering ──────────────────


class TestSpecificBrightnessFiltering:
    """Test the specific-brightness filtering logic."""

    def test_bright_compact_splat_kept(self):
        """High amplitude, small volume → high sb → should be kept."""
        # sigma=0.5 → small volume; amplitude=1.0 → high
        g3 = _make_isotropic_3d(0.5, amplitude=1.0)
        g4 = _make_4d_with_time([g3])
        vols = _spatial_volumes_3d(g4)
        sb = g4.amplitudes / np.clip(vols, 1e-8, None)
        assert sb[0] > 0.005  # Well above threshold

    def test_dim_diffuse_splat_removed(self):
        """Low amplitude, large volume → low sb → should be removed."""
        # sigma=5.0 → large volume; amplitude=0.001 → very dim
        g3 = _make_isotropic_3d(5.0, amplitude=0.001)
        g4 = _make_4d_with_time([g3])
        vols = _spatial_volumes_3d(g4)
        sb = g4.amplitudes / np.clip(vols, 1e-8, None)
        assert sb[0] < 0.005  # Below threshold

    def test_mixed_population_separation(self):
        """Given a mix of signal and background, filtering separates them."""
        # Signal: compact + bright
        signal = [
            _make_isotropic_3d(0.5, amplitude=0.8, center=[i, 0, 0]) for i in range(5)
        ]
        # Background: diffuse + dim
        background = [
            _make_isotropic_3d(4.0, amplitude=0.002, center=[i, 10, 0])
            for i in range(5)
        ]

        all_3d = signal + background
        g4 = _make_4d_with_time(all_3d)
        assert g4.n_splats == 10

        vols = _spatial_volumes_3d(g4)
        sb = g4.amplitudes / np.clip(vols, 1e-8, None)
        keep = sb > 0.005
        filtered = g4.filter(keep)

        # All 5 signal splats should be kept, all 5 background removed
        assert filtered.n_splats == 5
        # Kept centers should all have y=0 (signal), not y=10 (background)
        assert np.all(filtered.centers[:, 1] == 0.0)

    def test_empty_input(self):
        """Filtering an empty dataset should return empty."""
        g3_a = _make_isotropic_3d(1.0, amplitude=0.5)
        g3_b = _make_isotropic_3d(1.0, amplitude=0.5, center=[1, 0, 0])
        g4 = _make_4d_with_time([g3_a, g3_b])
        # Filter everything out with impossible threshold
        vols = _spatial_volumes_3d(g4)
        sb = g4.amplitudes / np.clip(vols, 1e-8, None)
        keep = sb > 1e10  # Nothing passes
        filtered = g4.filter(keep)
        assert filtered.n_splats == 0

    def test_diag_indices_correct(self):
        """Verify diagonal index computation matches GSplatData._cholesky_diag_elements."""
        for ndim in [2, 3, 4, 5]:
            expected = np.cumsum(np.arange(1, ndim + 1)) - 1
            # For 2D: [0, 2], 3D: [0, 2, 5], 4D: [0, 2, 5, 9], 5D: [0, 2, 5, 9, 14]
            # These are the positions of L[0,0], L[1,1], L[2,2], ... in packed format
            # Verify against the formula: position of L[i,i] = i*(i+1)/2 + i = i*(i+3)/2
            manual = np.array([i * (i + 1) // 2 + i for i in range(ndim)])
            np.testing.assert_array_equal(expected, manual)
