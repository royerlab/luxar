"""Tests for GSplatData class methods."""

import numpy as np
import pytest

from luxar.gsplats._data.base import _GSplatDataOps
from luxar.gsplats.gsplat_data import GSplatData

from ._gsplat_data_helpers import _make_3d_gsplat, _make_empty_gsplat

# ── Mixin composition ───────────────────────────────────────


class TestMixinComposition:
    """Guard the invariant the ``_data`` mixin split rests on."""

    def test_no_ops_stub_wins_the_mro(self):
        """Every ``_GSplatDataOps`` stub must resolve to a real implementation.

        The stubs exist so each domain mixin's cross-mixin ``self.`` calls
        type-check; they only ever raise. They are safe solely because C3 puts
        ``_GSplatDataOps`` after every mixin that inherits it. Reorder the base
        list (or drop the inheritance from the final base) and a stub silently
        wins — ``flattened()`` and friends start raising ``NotImplementedError``
        with nothing else in the suite to notice.
        """
        stubs = [name for name in vars(_GSplatDataOps) if not name.startswith("__")]
        assert stubs, "expected _GSplatDataOps to declare cross-mixin stubs"
        owners = {
            name: next(c for c in GSplatData.__mro__ if name in vars(c))
            for name in stubs
        }
        leaked = sorted(n for n, owner in owners.items() if owner is _GSplatDataOps)
        assert leaked == [], (
            f"_GSplatDataOps stubs shadow their real implementations: {leaked}. "
            f"MRO: {[c.__name__ for c in GSplatData.__mro__]}"
        )


# ── Properties and basic interface ──────────────────────────


class TestProperties:
    """Tests for n_splats, ndim, __len__, __repr__."""

    def test_n_splats(self):
        gs = _make_3d_gsplat(n=7)
        assert gs.n_splats == 7

    def test_ndim(self):
        gs = _make_3d_gsplat()
        assert gs.ndim == 3

    def test_ndim_2d(self):
        gs = GSplatData(
            centers=np.zeros((2, 2), dtype=np.float32),
            amplitudes=np.zeros(2, dtype=np.float32),
            cholesky_factors=np.zeros((2, 3), dtype=np.float32),
        )
        assert gs.ndim == 2

    def test_ndim_empty(self):
        """Empty GSplatData with 2D centers still knows its dimensionality."""
        gs = _make_empty_gsplat(ndim=3)
        assert gs.ndim == 3

    def test_len(self):
        gs = _make_3d_gsplat(n=10)
        assert len(gs) == 10

    def test_len_empty(self):
        gs = _make_empty_gsplat()
        assert len(gs) == 0

    def test_repr_nonempty(self):
        gs = _make_3d_gsplat(n=3)
        r = repr(gs)
        assert "3 splats" in r
        assert "3D" in r

    def test_repr_empty(self):
        gs = _make_empty_gsplat()
        r = repr(gs)
        assert "0 splats" in r

    def test_default_stats_is_empty_dict(self):
        gs = _make_3d_gsplat()
        assert gs.stats == {}
        assert isinstance(gs.stats, dict)

    def test_stats_not_shared_between_instances(self):
        gs1 = _make_3d_gsplat(n=1)
        gs2 = _make_3d_gsplat(n=1)
        gs1.stats["foo"] = "bar"
        assert "foo" not in gs2.stats


# ── Validation ──────────────────────────────────────────────


class TestValidation:
    """Tests for __post_init__ shape validation."""

    def test_valid_construction(self):
        """Valid shapes should not raise."""
        _make_3d_gsplat(n=5)  # Should not raise

    def test_valid_empty(self):
        """Empty GSplatData (0 splats) should pass validation."""
        _make_empty_gsplat()  # Should not raise

    def test_mismatched_amplitudes(self):
        with pytest.raises(ValueError, match="Amplitudes shape"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),
                amplitudes=np.zeros(5, dtype=np.float32),  # wrong count
                cholesky_factors=np.zeros((3, 6), dtype=np.float32),
            )

    def test_mismatched_colors(self):
        with pytest.raises(ValueError, match="Colors count"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),
                amplitudes=np.zeros(3, dtype=np.float32),
                cholesky_factors=np.zeros((3, 6), dtype=np.float32),
                colors=np.zeros((5, 3), dtype=np.float32),  # wrong count
            )

    def test_wrong_cholesky_packed_size(self):
        """Cholesky factors with wrong packed size for dimensionality."""
        with pytest.raises(ValueError, match="wrong packed size"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),  # 3D → expect k=6
                amplitudes=np.zeros(3, dtype=np.float32),
                cholesky_factors=np.zeros((3, 3), dtype=np.float32),  # k=3 (2D size)
            )

    def test_wrong_cholesky_splat_count(self):
        """Cholesky factors with correct packed size but wrong N."""
        with pytest.raises(ValueError, match="count mismatch"):
            GSplatData(
                centers=np.zeros((3, 3), dtype=np.float32),
                amplitudes=np.zeros(3, dtype=np.float32),
                cholesky_factors=np.zeros((5, 6), dtype=np.float32),  # N=5, not 3
            )

    def test_valid_with_colors(self):
        """Valid construction with colors should not raise."""
        GSplatData(
            centers=np.zeros((3, 3), dtype=np.float32),
            amplitudes=np.zeros(3, dtype=np.float32),
            cholesky_factors=np.zeros((3, 6), dtype=np.float32),
            colors=np.zeros((3, 3), dtype=np.float32),
        )

    def test_valid_2d(self):
        """2D data with correct Cholesky size (k=3) should not raise."""
        GSplatData(
            centers=np.zeros((4, 2), dtype=np.float32),
            amplitudes=np.zeros(4, dtype=np.float32),
            cholesky_factors=np.zeros((4, 3), dtype=np.float32),
        )


# ── Translate ───────────────────────────────────────────────


class TestTranslate:
    def test_basic_translate(self):
        gs = _make_3d_gsplat(n=3)
        offset = np.array([10, 20, 30], dtype=np.float32)
        translated = gs.translate(offset)
        assert np.allclose(translated.centers, gs.centers + offset)

    def test_translate_preserves_other_fields(self):
        gs = _make_3d_gsplat(n=3)
        gs.stats["key"] = "value"
        translated = gs.translate(np.zeros(3))
        # Amplitudes, cholesky should be same objects (references)
        assert translated.amplitudes is gs.amplitudes
        assert translated.cholesky_factors is gs.cholesky_factors
        # Stats should be a copy
        assert translated.stats == gs.stats
        assert translated.stats is not gs.stats

    def test_translate_with_colors(self):
        gs = _make_3d_gsplat(n=2)
        gs_with_colors = GSplatData(
            centers=gs.centers,
            amplitudes=gs.amplitudes,
            cholesky_factors=gs.cholesky_factors,
            colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        )
        translated = gs_with_colors.translate(np.array([1, 2, 3]))
        assert translated.colors is gs_with_colors.colors


# ── Center at centroid ──────────────────────────────────────


class TestCenterAtCentroid:
    def test_uniform_amplitudes(self):
        """With uniform amplitudes, centroid = unweighted mean."""
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32),
            amplitudes=np.array([1.0, 1.0], dtype=np.float32),
            cholesky_factors=np.zeros((2, 6), dtype=np.float32),
        )
        centered = gs.center_at_centroid()
        centroid = centered.centers.mean(axis=0)
        assert np.allclose(centroid, 0.0, atol=1e-5)

    def test_weighted_centroid(self):
        """Amplitude-weighted centroid should be at origin."""
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
            amplitudes=np.array([3.0, 1.0], dtype=np.float32),  # weighted toward first
            cholesky_factors=np.zeros((2, 6), dtype=np.float32),
        )
        centered = gs.center_at_centroid()
        total = centered.amplitudes.sum()
        weighted_centroid = (centered.centers.T @ centered.amplitudes) / total
        assert np.allclose(weighted_centroid, 0.0, atol=1e-5)

    def test_zero_amplitudes_uses_mean(self):
        """With all-zero amplitudes, falls back to unweighted mean."""
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32),
            amplitudes=np.array([0.0, 0.0], dtype=np.float32),
            cholesky_factors=np.zeros((2, 6), dtype=np.float32),
        )
        centered = gs.center_at_centroid()
        assert np.allclose(centered.centers.mean(axis=0), 0.0, atol=1e-5)

    @staticmethod
    def _iso3d(center, sigma=2.0):
        from luxar.gsplats.utils.trils import pack_tril

        L = np.zeros((1, 3, 3))
        for i in range(3):
            L[0, i, i] = sigma
        return GSplatData(
            centers=np.array([center], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=pack_tril(L).astype(np.float32),
        )

    def test_pure_3d_all_axes_centered(self):
        """With real (non-degenerate) covariance on every axis, all axes center
        — the historical behavior is preserved for spatial data."""
        from luxar.gsplats.utils.trils import pack_tril

        L = np.zeros((2, 3, 3))
        for i in range(3):
            L[:, i, i] = 2.0
        gs = GSplatData(
            centers=np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32),
            amplitudes=np.array([1.0, 1.0], dtype=np.float32),
            cholesky_factors=pack_tril(L).astype(np.float32),
        )
        centered = gs.center_at_centroid()
        assert np.allclose(centered.centers.mean(axis=0), 0.0, atol=1e-5)

    def test_leaves_categorical_time_axis_uncentered(self):
        """A zero-variance time axis (combine_as_new_dimension, sigma=0) keeps
        its integer coordinates; only the spatial centroid is re-origined.

        Regression guard for the timelapse-centering bug that pushed integer
        timepoints to fractional offsets and misaligned the slice navigator.
        """
        t0 = self._iso3d([10.0, 20.0, 30.0])
        t1 = self._iso3d([10.0, 20.0, 30.0])
        gs = GSplatData.combine_as_new_dimension(
            [t0, t1], values=[0.0, 1.0], sigma=0.0
        )
        centered = gs.center_at_centroid()
        # Time axis (dim3) must be untouched — still exactly {0, 1}.
        assert set(np.round(centered.centers[:, 3], 5).tolist()) == {0.0, 1.0}
        # Spatial centroid (dims 0..2) at the origin.
        amp = centered.amplitudes
        sp_centroid = (centered.centers[:, :3].T @ amp) / amp.sum()
        assert np.allclose(sp_centroid, 0.0, atol=1e-4)


# ── Scale intensity ─────────────────────────────────────────


class TestScaleIntensity:
    def test_double(self):
        gs = _make_3d_gsplat()
        scaled = gs.scale_intensity(2.0)
        assert np.allclose(scaled.amplitudes, gs.amplitudes * 2.0)

    def test_preserves_other_fields(self):
        gs = _make_3d_gsplat()
        scaled = gs.scale_intensity(0.5)
        assert scaled.centers is gs.centers
        assert scaled.cholesky_factors is gs.cholesky_factors

    def test_zero_factor(self):
        gs = _make_3d_gsplat()
        scaled = gs.scale_intensity(0.0)
        assert np.allclose(scaled.amplitudes, 0.0)


class TestTransform:
    def test_identity(self):
        gs = _make_3d_gsplat(n=3)
        result = gs.transform(np.eye(3))
        assert np.allclose(result.centers, gs.centers, atol=1e-5)

    def test_uniform_scale(self):
        gs = _make_3d_gsplat(n=3)
        result = gs.transform(np.eye(3) * 2.0)
        assert np.allclose(result.centers, gs.centers * 2.0, atol=1e-5)

    def test_translation_affine(self):
        gs = _make_3d_gsplat(n=3)
        M = np.eye(4)
        M[:3, 3] = [10, 20, 30]
        result = gs.transform(M)
        assert np.allclose(result.centers, gs.centers + [10, 20, 30], atol=1e-5)

    def test_rotation_2d(self):
        """90-degree rotation in 2D."""
        gs = GSplatData(
            centers=np.array([[1.0, 0.0]], dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1]], dtype=np.float32),
        )
        theta = np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        result = gs.transform(R)
        assert np.allclose(result.centers[0], [0, 1], atol=1e-5)

    def test_covariance_correctness(self):
        """Verify Sigma_new = A @ Sigma @ A.T for known transform."""
        from luxar.gsplats.utils.trils import unpack_tril

        gs = GSplatData(
            centers=np.zeros((1, 2), dtype=np.float32),
            amplitudes=np.ones(1, dtype=np.float32),
            cholesky_factors=np.array([[2.0, 0.5, 1.5]], dtype=np.float32),
        )
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        result = gs.transform(A)
        # Compute expected Sigma
        L_orig = unpack_tril(gs.cholesky_factors.astype(np.float64), 2)
        Sigma_orig = L_orig @ np.swapaxes(L_orig, -2, -1)
        Sigma_expected = A @ Sigma_orig[0] @ A.T
        # Verify result
        L_result = unpack_tril(result.cholesky_factors.astype(np.float64), 2)
        Sigma_result = (L_result @ np.swapaxes(L_result, -2, -1))[0]
        assert np.allclose(Sigma_result, Sigma_expected, atol=1e-6)

    def test_amplitudes_unchanged(self):
        gs = _make_3d_gsplat(n=3)
        result = gs.transform(np.eye(3) * 2.0)
        assert result.amplitudes is gs.amplitudes

    def test_invalid_shape(self):
        gs = _make_3d_gsplat(n=2)
        with pytest.raises(ValueError, match="Matrix shape"):
            gs.transform(np.eye(2))

    def test_invalid_last_row(self):
        gs = _make_3d_gsplat(n=2)
        M = np.eye(4)
        M[3, 0] = 1.0
        with pytest.raises(ValueError, match="Last row"):
            gs.transform(M)

    def test_empty(self):
        gs = _make_empty_gsplat()
        result = gs.transform(np.eye(3))
        assert result.n_splats == 0


# ── Intensity transforms ────────────────────────────────


class TestAffineIntensity:
    def test_scale_only(self):
        gs = _make_3d_gsplat()
        result = gs.affine_intensity(scale=2.0)
        assert np.allclose(result.amplitudes, gs.amplitudes * 2.0)

    def test_offset_only(self):
        gs = _make_3d_gsplat()
        result = gs.affine_intensity(offset=0.5)
        assert np.allclose(result.amplitudes, gs.amplitudes + 0.5)

    def test_both(self):
        gs = _make_3d_gsplat()
        result = gs.affine_intensity(scale=2.0, offset=0.1)
        assert np.allclose(result.amplitudes, gs.amplitudes * 2.0 + 0.1)

    def test_preserves_centers(self):
        gs = _make_3d_gsplat()
        result = gs.affine_intensity(scale=3.0)
        assert result.centers is gs.centers


class TestNormalizeIntensity:
    def test_basic(self):
        gs = _make_3d_gsplat()
        result = gs.normalize_intensity()
        assert np.allclose(result.amplitudes.max(), 1.0)

    def test_custom_target(self):
        gs = _make_3d_gsplat()
        result = gs.normalize_intensity(target_max=0.5)
        assert np.allclose(result.amplitudes.max(), 0.5)

    def test_zero_amplitudes(self):
        gs = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.zeros(2, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
        )
        result = gs.normalize_intensity()
        assert np.allclose(result.amplitudes, 0.0)


class TestClampIntensity:
    def test_min(self):
        gs = _make_3d_gsplat()
        result = gs.clamp_intensity(min=0.5)
        assert result.amplitudes.min() >= 0.5

    def test_max(self):
        gs = _make_3d_gsplat()
        result = gs.clamp_intensity(max=0.1)
        assert result.amplitudes.max() <= 0.1

    def test_both(self):
        gs = _make_3d_gsplat()
        result = gs.clamp_intensity(min=0.2, max=0.8)
        assert result.amplitudes.min() >= 0.2
        assert result.amplitudes.max() <= 0.8

    def test_preserves_centers(self):
        gs = _make_3d_gsplat()
        result = gs.clamp_intensity(min=0.0, max=1.0)
        assert result.centers is gs.centers


# ── LOD Tests ──────────────────────────────────────────────
