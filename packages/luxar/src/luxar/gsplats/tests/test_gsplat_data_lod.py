"""Tests for GSplatData LOD functionality.

Split out from ``test_gsplat_data.py`` to keep that file focused on the
core (single-LOD) data API. This file covers:

- ``AdditiveSubLOD`` (frozen dataclass)
- ``GSplatData`` multi-additive-sublod construction & accessors
- LOD preservation through merge / transform operations
- Sharpness-removal regression (gsplats vs points/lines)
- ``truncation_radius`` propagation
- ``SubstitutiveLevel`` and 2-D substitutive × additive accessors
"""

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel

from ._gsplat_data_helpers import _make_3d_gsplat


class TestAdditiveSubLOD:
    """Tests for the AdditiveSubLOD frozen dataclass."""

    def test_creation(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        lod = AdditiveSubLOD(
            centers=np.array([[1, 2, 3]], dtype=np.float32),
            amplitudes=np.array([0.5], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
        )
        assert lod.n_splats == 1
        assert lod.ndim == 3

    def test_frozen(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        lod = AdditiveSubLOD(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.ones(2, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
        )
        with pytest.raises(AttributeError):
            lod.centers = np.zeros((2, 3))  # type: ignore[misc]

    def test_mixin_properties(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        lod = AdditiveSubLOD(
            centers=np.array([[10, 20, 30]], dtype=np.float32),
            amplitudes=np.array([2.0], dtype=np.float32),
            cholesky_factors=np.array([[2, 0, 3, 0, 0, 4]], dtype=np.float32),
        )
        assert lod.n_splats == 1
        assert lod.ndim == 3
        assert len(lod.volumes()) == 1
        assert len(lod.masses()) == 1
        assert lod.marginal_sigmas().shape == (1, 3)
        assert len(lod.eccentricities()) == 1

    def test_validation(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        with pytest.raises(ValueError, match="Amplitudes shape"):
            AdditiveSubLOD(
                centers=np.zeros((3, 2), dtype=np.float32),
                amplitudes=np.ones(2, dtype=np.float32),  # wrong count
                cholesky_factors=np.zeros((3, 3), dtype=np.float32),
            )


class TestGSplatDataLOD:
    """Tests for LOD functionality in GSplatData."""

    def _make_lods(self, n_lods=3, splats_per_lod=10):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        rng = np.random.RandomState(42)
        lods = []
        for i in range(n_lods):
            lods.append(
                AdditiveSubLOD(
                    centers=rng.rand(splats_per_lod, 3).astype(np.float32) * 100,
                    amplitudes=rng.rand(splats_per_lod).astype(np.float32),
                    cholesky_factors=np.tile(
                        np.array([1, 0, 1, 0, 0, 1], dtype=np.float32),
                        (splats_per_lod, 1),
                    ),
                    stats={"pass_index": i, "cumulative_psnr_db": 20.0 + i * 5.0},
                )
            )
        return lods

    def test_from_lods(self):
        lods = self._make_lods()
        data = GSplatData.from_additive_sublods(lods)
        assert data.n_additive_sublods == 3
        assert data.n_splats == 30
        assert data.centers.shape == (30, 3)

    def test_convenience_constructor_is_single_lod(self):
        gs = _make_3d_gsplat(n=5)
        assert gs.n_additive_sublods == 1
        assert gs.additive_sublods[0].n_splats == 5

    def test_n_lods(self):
        lods = self._make_lods(n_lods=4)
        data = GSplatData.from_additive_sublods(lods)
        assert data.n_additive_sublods == 4

    def test_at_lod(self):
        lods = self._make_lods()
        data = GSplatData.from_additive_sublods(lods)
        lod0 = data.additive_sublod(0)
        assert lod0.n_splats == 10
        assert lod0.stats["pass_index"] == 0

    def test_up_to_lod(self):
        lods = self._make_lods(n_lods=4)
        data = GSplatData.from_additive_sublods(lods)
        trimmed = data.additive_prefix(1)
        assert trimmed.n_additive_sublods == 2
        assert trimmed.n_splats == 20

    def test_trim_lods(self):
        lods = self._make_lods()
        data = GSplatData.from_additive_sublods(lods)
        trimmed = data.additive_prefix(0)
        assert trimmed.n_additive_sublods == 1
        assert trimmed.n_splats == 10

    def test_flattened(self):
        lods = self._make_lods()
        data = GSplatData.from_additive_sublods(lods)
        flat = data.flattened()
        assert flat.n_additive_sublods == 1
        assert flat.n_splats == data.n_splats
        np.testing.assert_array_equal(flat.centers, data.centers)

    def test_lod_psnrs(self):
        lods = self._make_lods()
        data = GSplatData.from_additive_sublods(lods)
        psnrs = data.lod_psnrs()
        assert psnrs == [20.0, 25.0, 30.0]

    def test_lod_psnrs_missing(self):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        lods = [
            AdditiveSubLOD(
                centers=np.zeros((1, 2), dtype=np.float32),
                amplitudes=np.ones(1, dtype=np.float32),
                cholesky_factors=np.array([[1, 0, 1]], dtype=np.float32),
                stats={},  # no psnr
            )
        ]
        data = GSplatData.from_additive_sublods(lods)
        psnrs = data.lod_psnrs()
        assert np.isnan(psnrs[0])

    def test_cached_concat_matches_lods(self):
        lods = self._make_lods()
        data = GSplatData.from_additive_sublods(lods)
        expected_centers = np.concatenate([lod.centers for lod in lods], axis=0)
        np.testing.assert_array_equal(data.centers, expected_centers)

    def test_single_lod_no_copy(self):
        """Single-LOD fast path should share array references."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        c = np.zeros((5, 3), dtype=np.float32)
        a = np.ones(5, dtype=np.float32)
        cf = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (5, 1))
        lod = AdditiveSubLOD(centers=c, amplitudes=a, cholesky_factors=cf)
        data = GSplatData.from_additive_sublods([lod])
        # Should share memory, not copy
        assert data.centers is c
        assert data.amplitudes is a

    def test_repr_multi_lod(self):
        lods = self._make_lods()
        data = GSplatData.from_additive_sublods(lods)
        r = repr(data)
        assert "3 LODs" in r
        assert "30 splats" in r

    def test_repr_single_lod(self):
        gs = _make_3d_gsplat()
        r = repr(gs)
        assert "LOD" not in r  # single LOD should not mention LODs

    def test_mixin_on_multi_lod(self):
        """Computed properties work on the concatenated view."""
        lods = self._make_lods(n_lods=2, splats_per_lod=5)
        data = GSplatData.from_additive_sublods(lods)
        assert data.volumes().shape == (10,)
        assert data.masses().shape == (10,)
        assert data.eccentricities().shape == (10,)

    def test_filter_preserves_lods(self):
        """filter() preserves LOD structure on multi-LOD data."""
        lods = self._make_lods()
        data = GSplatData.from_additive_sublods(lods)
        mask = data.amplitudes > 0.5
        filtered = data.filter(mask)
        assert filtered.n_additive_sublods == 3  # LODs preserved
        assert filtered.n_splats == int(mask.sum())

    def test_empty_lods_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            GSplatData(additive_sublods=[])

    def test_invalid_lod_type_raises(self):
        with pytest.raises(TypeError, match="AdditiveSubLOD"):
            GSplatData(additive_sublods=["not a lod"])  # type: ignore[list-item]

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            GSplatData()


class TestLODPreservation:
    """Tests that merge/transform operations preserve multi-LOD structure."""

    def _make_multi_lod(self, n_lods=3, splats_per_lod=10, ndim=3, seed=42):
        """Create a multi-LOD GSplatData for testing."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        rng = np.random.RandomState(seed)
        tril_size = ndim * (ndim + 1) // 2
        lods = []
        for i in range(n_lods):
            # Identity-like cholesky: diagonal = 1, off-diagonal = 0
            chol = np.zeros((splats_per_lod, tril_size), dtype=np.float32)
            k = 0
            for row in range(ndim):
                for col in range(row + 1):
                    if row == col:
                        chol[:, k] = 1.0
                    k += 1
            lods.append(
                AdditiveSubLOD(
                    centers=rng.rand(splats_per_lod, ndim).astype(np.float32) * 100,
                    amplitudes=rng.rand(splats_per_lod).astype(np.float32) + 0.1,
                    cholesky_factors=chol,
                    stats={"pass_index": i, "cumulative_psnr_db": 20.0 + i * 5.0},
                )
            )
        return GSplatData.from_additive_sublods(lods)

    def test_concatenate_preserves_lods(self):
        d1 = self._make_multi_lod(n_lods=3, splats_per_lod=10, seed=1)
        d2 = self._make_multi_lod(n_lods=3, splats_per_lod=15, seed=2)
        result = GSplatData.concatenate([d1, d2])
        assert result.n_additive_sublods == 3
        for level in range(3):
            lod = result.additive_sublod(level)
            assert lod.n_splats == 25  # 10 + 15

    def test_concatenate_mixed_lod_counts(self):
        d1 = self._make_multi_lod(n_lods=2, splats_per_lod=10, seed=1)
        d2 = self._make_multi_lod(n_lods=4, splats_per_lod=8, seed=2)
        result = GSplatData.concatenate([d1, d2])
        assert result.n_additive_sublods == 4
        # Level 0 and 1: both contribute
        assert result.additive_sublod(0).n_splats == 18  # 10 + 8
        assert result.additive_sublod(1).n_splats == 18
        # Level 2 and 3: only d2 contributes
        assert result.additive_sublod(2).n_splats == 8
        assert result.additive_sublod(3).n_splats == 8

    def test_concatenate_single_lod_unchanged(self):
        d1 = GSplatData(
            centers=np.random.rand(5, 3).astype(np.float32),
            amplitudes=np.ones(5, dtype=np.float32),
            cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (5, 1)).astype(np.float32),
        )
        d2 = GSplatData(
            centers=np.random.rand(7, 3).astype(np.float32),
            amplitudes=np.ones(7, dtype=np.float32),
            cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (7, 1)).astype(np.float32),
        )
        result = GSplatData.concatenate([d1, d2])
        assert result.n_additive_sublods == 1
        assert result.n_splats == 12

    def test_embed_dimension_preserves_lods(self):
        data = self._make_multi_lod(n_lods=3, splats_per_lod=10, ndim=3)
        assert data.ndim == 3
        embedded = data.embed_dimension(5.0, sigma=0.0)
        assert embedded.ndim == 4
        assert embedded.n_additive_sublods == 3
        for level in range(3):
            lod = embedded.additive_sublod(level)
            assert lod.n_splats == 10
            assert lod.centers.shape == (10, 4)
            # Check the new dimension has value 5.0
            np.testing.assert_allclose(lod.centers[:, 3], 5.0)

    def test_embed_dimension_per_splat_values_multi_lod(self):
        data = self._make_multi_lod(n_lods=2, splats_per_lod=10, ndim=3)
        # Per-splat values: 20 total splats
        values = np.arange(20, dtype=np.float32)
        embedded = data.embed_dimension(values, sigma=0.0)
        assert embedded.n_additive_sublods == 2
        # LOD 0 should get values 0-9, LOD 1 should get values 10-19
        np.testing.assert_allclose(
            embedded.additive_sublod(0).centers[:, 3], np.arange(10)
        )
        np.testing.assert_allclose(
            embedded.additive_sublod(1).centers[:, 3], np.arange(10, 20)
        )

    def test_combine_as_new_dimension_preserves_lods(self):
        d1 = self._make_multi_lod(n_lods=2, splats_per_lod=10, ndim=3, seed=1)
        d2 = self._make_multi_lod(n_lods=2, splats_per_lod=10, ndim=3, seed=2)
        combined = GSplatData.combine_as_new_dimension([d1, d2], sigma=0.0)
        assert combined.ndim == 4
        assert combined.n_additive_sublods == 2
        # Each LOD should have 20 splats (10 from each dataset)
        assert combined.additive_sublod(0).n_splats == 20
        assert combined.additive_sublod(1).n_splats == 20

    def test_merge_with_channel_colors_preserves_lods(self):
        d1 = self._make_multi_lod(n_lods=2, splats_per_lod=10, seed=1)
        d2 = self._make_multi_lod(n_lods=2, splats_per_lod=8, seed=2)
        merged = GSplatData.merge_with_channel_colors(
            [d1, d2],
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )
        assert merged.n_additive_sublods == 2
        # Each LOD: 10 + 8 = 18 splats
        for level in range(2):
            lod = merged.additive_sublod(level)
            assert lod.n_splats == 18
            assert lod.colors is not None
            # First 10 splats should be red, next 8 green
            np.testing.assert_allclose(lod.colors[:10], [[1, 0, 0]] * 10)
            np.testing.assert_allclose(lod.colors[10:], [[0, 1, 0]] * 8)

    def test_translate_preserves_lods(self):
        data = self._make_multi_lod(n_lods=3, splats_per_lod=10)
        original_centers = [data.additive_sublod(i).centers.copy() for i in range(3)]
        offset = np.array([10, 20, 30], dtype=np.float32)
        translated = data.translate(offset)
        assert translated.n_additive_sublods == 3
        for level in range(3):
            lod = translated.additive_sublod(level)
            assert lod.n_splats == 10
            np.testing.assert_allclose(
                lod.centers, original_centers[level] + offset, atol=1e-5
            )

    def test_transform_preserves_lods(self):
        data = self._make_multi_lod(n_lods=2, splats_per_lod=10)
        # Uniform 2x scaling
        scale = np.eye(3) * 2.0
        transformed = data.transform(scale)
        assert transformed.n_additive_sublods == 2
        for level in range(2):
            orig = data.additive_sublod(level)
            new = transformed.additive_sublod(level)
            assert new.n_splats == orig.n_splats
            np.testing.assert_allclose(new.centers, orig.centers * 2.0, atol=1e-4)

    def test_full_merge_pipeline_preserves_lods(self):
        """End-to-end: tile concat → timepoint stacking → channel merge."""

        # Simulate 2 tiles per timepoint, 2 timepoints, 2 channels
        # Each tile has 2 LODs
        def make_tiles(seed):
            return [
                self._make_multi_lod(n_lods=2, splats_per_lod=5, seed=seed + i)
                for i in range(2)
            ]

        # Level 1: tile merge (concatenate)
        tp0_ch0 = GSplatData.concatenate(make_tiles(0))
        tp1_ch0 = GSplatData.concatenate(make_tiles(10))
        tp0_ch1 = GSplatData.concatenate(make_tiles(20))
        tp1_ch1 = GSplatData.concatenate(make_tiles(30))
        assert tp0_ch0.n_additive_sublods == 2  # LODs preserved through tile concat

        # Level 2: timepoint stacking (combine_as_new_dimension)
        ch0_4d = GSplatData.combine_as_new_dimension([tp0_ch0, tp1_ch0], sigma=0.0)
        ch1_4d = GSplatData.combine_as_new_dimension([tp0_ch1, tp1_ch1], sigma=0.0)
        assert ch0_4d.n_additive_sublods == 2  # LODs preserved through stacking
        assert ch0_4d.ndim == 4

        # Level 3: channel merge
        final = GSplatData.merge_with_channel_colors(
            [ch0_4d, ch1_4d],
            channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )
        assert final.n_additive_sublods == 2  # LODs preserved through channel merge
        assert final.ndim == 4
        # 2 tiles × 5 splats × 2 timepoints × 2 channels = 40 splats per LOD
        assert final.additive_sublod(0).n_splats == 40
        assert final.additive_sublod(1).n_splats == 40


# ── Sharpness removal regression tests ───────────────────────


class TestGSplatsWithoutSharpness:
    """Verify that GSplats work without sharpness while Points/Lines retain it."""

    def test_gsplats_roundtrip_no_sharpness(self, tmp_path):
        """GSplats should round-trip through zarr without a sharpness attribute."""
        from luxar import Dimensions, LuxarScene, LuxarZarrCompiler

        path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "test",
                centers=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                amplitudes=np.array([1.0, 2.0], dtype=np.float32),
                cholesky_factors=np.array(
                    [[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32
                ),
                colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            )

        s = LuxarScene.load(path)
        data = s.get_gsplats("test")
        assert len(data["centers"]) == 2
        assert data.get("sharpness") is None, "GSplats should NOT have sharpness"

    def test_gsplat_data_no_sharpnesses_field(self):
        """GSplatData should not require a sharpnesses field."""
        result = GSplatData(
            centers=np.array([[0, 0, 0]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
        )
        assert not hasattr(result, "sharpnesses") or not hasattr(
            result.__dataclass_fields__, "sharpnesses"
        )

    def test_add_gsplats_signature_no_sharpness(self):
        """add_gsplats() signature should not include sharpness."""
        import inspect

        from luxar.core.group import Group

        sig = inspect.signature(Group.add_gsplats)
        params = list(sig.parameters.keys())
        assert "sharpness" not in params, (
            f"sharpness should not be in add_gsplats params: {params}"
        )

    def test_points_still_have_sharpness(self, tmp_path):
        """Points should still support the sharpness parameter."""
        from luxar import Dimensions, LuxarScene, LuxarZarrCompiler

        path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                positions=np.array([[1, 2, 3]], dtype=np.float32),
                radii=np.array([0.5], dtype=np.float32),
                sharpness=np.array([3.0], dtype=np.float32),
            )

        s = LuxarScene.load(path)
        data = s.get_points("pts")
        assert data["sharpness"] is not None
        np.testing.assert_allclose(data["sharpness"], [3.0])

    def test_lines_still_have_sharpness(self, tmp_path):
        """Lines should still support the sharpness parameter."""
        from luxar import Dimensions, LuxarScene, LuxarZarrCompiler

        path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                widths=np.array([0.1, 0.1], dtype=np.float32),
                sharpness=np.array([1.5, 1.5], dtype=np.float32),
            )

        s = LuxarScene.load(path)
        data = s.get_lines("ln")
        assert data["sharpness"] is not None

    def test_fitting_returns_no_sharpness(self):
        """fit_gaussian_splats should work and not return sharpness."""
        from luxar.gsplats import fit_gaussian_splats

        img = np.random.rand(16, 16).astype(np.float32) * 0.5
        img[5:10, 5:10] = 1.0

        result = fit_gaussian_splats(img, seeds=5, max_iterations=10, verbose=False)
        assert isinstance(result, GSplatData)
        assert not hasattr(result, "sharpnesses")
        assert result.centers.shape[1] == 2


class TestTruncationRadius:
    """Tests for the truncation_radius field on GSplatData and AdditiveSubLOD."""

    def _make_gsplat(self, n: int = 10, truncation_radius: float = 3.0) -> GSplatData:
        rng = np.random.RandomState(42)
        return GSplatData(
            centers=rng.rand(n, 3).astype(np.float32),
            amplitudes=rng.rand(n).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
            ),
            truncation_radius=truncation_radius,
        )

    def test_default_truncation_radius(self):
        """Default truncation_radius is 3.0."""
        g = self._make_gsplat()
        assert g.truncation_radius == 3.0
        assert g.additive_sublods[0].truncation_radius == 3.0

    def test_custom_truncation_radius(self):
        """Custom truncation_radius is propagated to LOD."""
        g = self._make_gsplat(truncation_radius=2.5)
        assert g.truncation_radius == 2.5
        assert g.additive_sublods[0].truncation_radius == 2.5

    def test_truncation_preserved_by_filter(self):
        """filter() preserves truncation_radius."""
        g = self._make_gsplat(truncation_radius=2.75)
        mask = np.ones(g.n_splats, dtype=bool)
        mask[0] = False
        filtered = g.filter(mask)
        assert filtered.truncation_radius == 2.75

    def test_truncation_preserved_by_flattened(self):
        """flattened() preserves truncation_radius."""
        g = self._make_gsplat(truncation_radius=2.5)
        flat = g.flattened()
        assert flat.truncation_radius == 2.5

    def test_concatenate_same_truncation(self):
        """concatenate works when all datasets have same truncation_radius."""
        g1 = self._make_gsplat(5, truncation_radius=2.75)
        g2 = self._make_gsplat(8, truncation_radius=2.75)
        merged = GSplatData.concatenate([g1, g2])
        assert merged.truncation_radius == 2.75
        assert merged.n_splats == 13

    def test_concatenate_rejects_mismatched_truncation(self):
        """concatenate raises ValueError on mismatched truncation_radius."""
        g1 = self._make_gsplat(5, truncation_radius=3.0)
        g2 = self._make_gsplat(5, truncation_radius=2.5)
        with pytest.raises(ValueError, match="Truncation radius mismatch"):
            GSplatData.concatenate([g1, g2])

    def test_filter_by_uses_stored_truncation(self):
        """filter_by uses self.truncation_radius when truncate=None."""
        g = self._make_gsplat(truncation_radius=2.5)
        # filter_by with no filtering criteria returns same data
        filtered = g.filter_by()
        assert filtered.truncation_radius == 2.5


# ---------------------------------------------------------------------------
# SubstitutiveLevel — v2.0 skeleton class (full 2-D wiring in commit 2)
# ---------------------------------------------------------------------------


class TestSubstitutiveLevel:
    """Skeleton-level tests for the new 2-D LOD wrapper.

    Full ``GSplatData.substitutive_levels`` semantics arrive in the next
    commit; this class just covers the dataclass itself (construction,
    validation, ``n_additive_lods`` / ``n_splats_total`` properties).
    """

    @staticmethod
    def _make_additive(n: int = 5) -> AdditiveSubLOD:
        return AdditiveSubLOD(
            centers=np.zeros((n, 3), dtype=np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
            ),
        )

    def test_construct_minimal(self):
        from luxar.gsplats.gsplat_data import SubstitutiveLevel

        sub = self._make_additive(3)
        level = SubstitutiveLevel(additive_sublods=[sub])
        assert level.n_additive_lods == 1
        assert level.n_splats_total == 3
        assert level.compression_factor == 1
        assert level.parent_method is None
        assert level.level_index == 0
        assert level.stats == {}

    def test_construct_with_metadata(self):
        from luxar.gsplats.gsplat_data import SubstitutiveLevel

        level = SubstitutiveLevel(
            additive_sublods=[self._make_additive(4), self._make_additive(2)],
            compression_factor=4,
            parent_method="kmeans_lloyd",
            level_index=1,
            stats={"psnr_estimate": 33.5},
        )
        assert level.n_additive_lods == 2
        assert level.n_splats_total == 6
        assert level.compression_factor == 4
        assert level.parent_method == "kmeans_lloyd"
        assert level.level_index == 1
        assert level.stats == {"psnr_estimate": 33.5}

    def test_empty_additive_sublods_rejected(self):
        from luxar.gsplats.gsplat_data import SubstitutiveLevel

        with pytest.raises(ValueError, match="at least one AdditiveSubLOD"):
            SubstitutiveLevel(additive_sublods=[])

    def test_non_additive_entry_rejected(self):
        from luxar.gsplats.gsplat_data import SubstitutiveLevel

        with pytest.raises(TypeError, match="AdditiveSubLOD"):
            SubstitutiveLevel(additive_sublods=["not a splat set"])  # type: ignore[list-item]

    def test_invalid_compression_factor_rejected(self):
        from luxar.gsplats.gsplat_data import SubstitutiveLevel

        with pytest.raises(ValueError, match="compression_factor"):
            SubstitutiveLevel(
                additive_sublods=[self._make_additive(2)], compression_factor=0
            )

    def test_frozen(self):
        from luxar.gsplats.gsplat_data import SubstitutiveLevel

        level = SubstitutiveLevel(additive_sublods=[self._make_additive(2)])
        # frozen=True dataclass: setting an attribute should raise
        with pytest.raises((AttributeError, Exception)):
            level.compression_factor = 4  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GSplatData 2-D substitutive × additive accessors
# ---------------------------------------------------------------------------


class TestGSplatData2DAccessors:
    """Tests for the new v2.0 2-D accessors: n_substitutive, at_substitutive,
    cell, default_substitutive, from_substitutive_levels.

    Construction paths (centers/amplitudes/cholesky_factors, lods=..., and
    substitutive_levels=...) all yield a valid 2-D shape; verify the new
    accessors return the right cells and that existing accessors
    (``lods``, ``n_lods``) remain consistent with the default substitutive
    level.
    """

    @staticmethod
    def _make_additive(n: int = 5, seed: int = 0) -> AdditiveSubLOD:
        rng = np.random.default_rng(seed)
        return AdditiveSubLOD(
            centers=(rng.random((n, 3)) * 10).astype(np.float32),
            amplitudes=rng.random(n).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (n, 1)
            ),
        )

    def _make_substitutive_level(
        self, n_additive: int = 2, n_per: int = 4, compression: int = 1
    ) -> "SubstitutiveLevel":
        from luxar.gsplats.gsplat_data import SubstitutiveLevel

        sublods = [self._make_additive(n_per, seed=i) for i in range(n_additive)]
        return SubstitutiveLevel(
            additive_sublods=sublods,
            compression_factor=compression,
        )

    # ── Construction paths ────────────────────────────────────

    def test_array_form_yields_single_substitutive(self):
        """Convenience constructor produces ``[1, 1]`` matrix shape."""
        rng = np.random.default_rng(0)
        data = GSplatData(
            centers=(rng.random((5, 3)) * 10).astype(np.float32),
            amplitudes=rng.random(5).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (5, 1)
            ),
        )
        assert data.n_substitutive == 1
        assert data.default_substitutive == 0
        # The single substitutive level wraps a single additive sub-LOD
        level = data.default_substitutive_level
        assert level.n_additive_lods == 1
        assert level.n_splats_total == 5
        assert level.compression_factor == 1

    def test_lods_form_yields_single_substitutive(self):
        """``lods=...`` constructor produces ``[1, M]`` matrix shape."""
        sublods = [self._make_additive(3, seed=0), self._make_additive(2, seed=1)]
        data = GSplatData(additive_sublods=sublods)
        assert data.n_substitutive == 1
        # Existing lods accessor still works and matches default substitutive level
        assert data.n_additive_sublods == 2
        assert list(data.additive_sublods) == list(
            data.default_substitutive_level.additive_sublods
        )

    def test_substitutive_levels_form_yields_multi(self):
        """``substitutive_levels=...`` constructor produces ``[N, M_i]`` shape."""
        levels = [
            self._make_substitutive_level(n_additive=3, n_per=4, compression=1),
            self._make_substitutive_level(n_additive=1, n_per=2, compression=4),
        ]
        data = GSplatData(substitutive_levels=levels)
        assert data.n_substitutive == 2
        assert data.default_substitutive == 0
        # Default substitutive level's additive ladder is the shape of lods
        assert data.n_additive_sublods == 3
        # The coarser level has its own count
        assert data.substitutive_levels[1].n_additive_lods == 1
        assert data.substitutive_levels[1].compression_factor == 4

    def test_from_substitutive_levels_classmethod(self):
        levels = [
            self._make_substitutive_level(n_additive=2, n_per=4, compression=1),
            self._make_substitutive_level(n_additive=1, n_per=3, compression=2),
        ]
        data = GSplatData.from_substitutive_levels(levels, stats={"hello": "world"})
        assert data.n_substitutive == 2
        assert data.stats == {"hello": "world"}

    def test_default_substitutive_out_of_range_rejected(self):
        levels = [self._make_substitutive_level()]
        with pytest.raises(ValueError, match="default_substitutive"):
            GSplatData(substitutive_levels=levels, default_substitutive=5)

    def test_empty_substitutive_levels_rejected(self):
        with pytest.raises(ValueError, match="at least one SubstitutiveLevel"):
            GSplatData(substitutive_levels=[])

    def test_non_substitutive_level_rejected(self):
        with pytest.raises(TypeError, match="SubstitutiveLevel"):
            GSplatData(substitutive_levels=["not a level"])  # type: ignore[list-item]

    # ── Accessors ────────────────────────────────────────────

    def test_at_substitutive_returns_view(self):
        levels = [
            self._make_substitutive_level(n_additive=2, n_per=4, compression=1),
            self._make_substitutive_level(n_additive=1, n_per=3, compression=4),
        ]
        data = GSplatData.from_substitutive_levels(levels)
        coarse = data.at_substitutive(1)
        assert isinstance(coarse, GSplatData)
        assert coarse.n_substitutive == 1
        assert coarse.n_additive_sublods == 1
        # The coarse level's data: 3 splats in one additive sub-LOD
        assert coarse.n_splats == 3

    def test_at_substitutive_out_of_range_raises(self):
        data = GSplatData(additive_sublods=[self._make_additive(3)])
        with pytest.raises(IndexError, match="substitutive"):
            data.at_substitutive(5)
        with pytest.raises(IndexError, match="substitutive"):
            data.at_substitutive(-1)

    def test_cell_direct_2d_access(self):
        levels = [
            self._make_substitutive_level(n_additive=3, n_per=4),
            self._make_substitutive_level(n_additive=2, n_per=2, compression=4),
        ]
        data = GSplatData.from_substitutive_levels(levels)
        c00 = data.cell(0, 0)
        c11 = data.cell(1, 1)
        assert isinstance(c00, AdditiveSubLOD)
        assert isinstance(c11, AdditiveSubLOD)
        assert c00.n_splats == 4
        assert c11.n_splats == 2

    def test_cell_out_of_range_raises(self):
        data = GSplatData(additive_sublods=[self._make_additive(3)])
        with pytest.raises(IndexError, match="substitutive"):
            data.cell(5, 0)
        with pytest.raises(IndexError, match="additive"):
            data.cell(0, 5)

    def test_array_caches_reflect_default_substitutive(self):
        """``data.centers/amplitudes/cholesky_factors`` reflect default level."""
        levels = [
            self._make_substitutive_level(n_additive=2, n_per=4),
            self._make_substitutive_level(n_additive=1, n_per=2, compression=4),
        ]
        data = GSplatData.from_substitutive_levels(levels)
        # n_splats = concatenation of default substitutive level's additive ladder
        # = 4 + 4 = 8 (not 8 + 2 = 10 — the coarse level is alternative)
        assert data.n_splats == 8


class TestImmutableViews:
    """M5/L3: view methods return read-only zero-copy instances and
    additive_prefix bounds-checks its argument."""

    def test_flattened_is_readonly_and_does_not_corrupt_source(self):
        d = _make_3d_gsplat(n=6)
        original = d.amplitudes.copy()
        f = d.flattened()
        with pytest.raises(ValueError, match="read-only|read only"):
            f.amplitudes[0] = 999.0
        # Source remains intact and writable.
        assert np.array_equal(d.amplitudes, original)
        d.amplitudes[0] = 1.0  # source still mutable

    def test_additive_prefix_is_readonly(self):
        lods = [
            AdditiveSubLOD(
                centers=np.zeros((n, 3), dtype=np.float32),
                amplitudes=np.ones(n, dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
                ),
            )
            for n in (3, 4)
        ]
        d = GSplatData.from_additive_sublods(lods)
        prefix = d.additive_prefix(0)
        with pytest.raises(ValueError, match="read-only|read only"):
            prefix.cell(0, 0).centers[0, 0] = 5.0

    def test_additive_prefix_out_of_range_raises_clearly(self):
        d = _make_3d_gsplat(n=4)  # single additive level
        with pytest.raises(IndexError, match="additive level"):
            d.additive_prefix(-1)
        with pytest.raises(IndexError, match="additive level"):
            d.additive_prefix(5)

    def test_at_substitutive_is_readonly(self):
        d = _make_3d_gsplat(n=5)
        view = d.at_substitutive(0)
        with pytest.raises(ValueError, match="read-only|read only"):
            view.centers[0, 0] = 7.0

    def test_lod_builders_tolerate_readonly_views(self):
        """make_substitutive_lod consumes flattened() internally; it must not
        emit a torch non-writable warning nor fail on the read-only arrays."""
        import warnings

        from luxar.gsplats.lod.substitutive import make_substitutive_lod

        d = _make_3d_gsplat(n=24)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = make_substitutive_lod(d, compression_factor=2, levels=1, device="cpu")
        assert out.n_substitutive == 2
        assert not any("writable" in str(w.message).lower() for w in caught)
        # The stored level arrays are normal writable arrays (cell() returns
        # the backing AdditiveSubLOD, not a read-only view).
        assert out.cell(0, 0).centers.flags.writeable
