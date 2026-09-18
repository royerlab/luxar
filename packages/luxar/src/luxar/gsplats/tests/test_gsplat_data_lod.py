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

from dataclasses import replace

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
from luxar.gsplats.lod.additive import make_additive_lod
from luxar.gsplats.lod.decimate import decimate
from luxar.gsplats.lod.substitutive import make_substitutive_lod
from luxar.gsplats.tree import iter_leaves
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

from ._gsplat_data_helpers import _make_3d_gsplat

_LABEL_COUNT = 3
_LABEL_VOCABULARY = {i: f"class-{i}" for i in range(_LABEL_COUNT)}


def _label_operation(
    operation, *, id: str, refuses: bool = False, labels_follow_centers: bool = True
):
    return pytest.param(operation, refuses, labels_follow_centers, id=id)


_LABEL_CHANNEL_OPERATIONS = (
    _label_operation(
        lambda data: data.filter(np.arange(data.n_splats) % 2 == 0), id="filter"
    ),
    _label_operation(
        lambda data: data.filter_by(bbox=[(2.0, 10.0), (-1.0, 1.0), (-1.0, 1.0)]),
        id="filter-by",
    ),
    _label_operation(
        lambda data: data.slice_by([slice(2.0, 10.0), slice(None), slice(None)]),
        id="slice-by",
    ),
    _label_operation(
        lambda data: data.cull(method="amplitude_percentile", amplitude_percentile=25),
        id="cull",
    ),
    _label_operation(
        lambda data: data.transform(np.eye(data.ndim, dtype=np.float32)),
        id="transform",
    ),
    _label_operation(
        lambda data: data.translate(np.array([3.0, 0.0, 0.0], dtype=np.float32)),
        id="translate",
    ),
    _label_operation(lambda data: data.center_at_centroid(), id="center-at-centroid"),
    _label_operation(lambda data: data.with_colors((0.2, 0.4, 0.6)), id="with-colors"),
    _label_operation(
        lambda data: data.affine_intensity(scale=2.0, offset=0.1),
        id="affine-intensity",
    ),
    _label_operation(lambda data: data.normalize_intensity(), id="normalize-intensity"),
    _label_operation(
        lambda data: data.clamp_intensity(min=0.5, max=2.0),
        id="clamp-intensity",
    ),
    _label_operation(lambda data: data.scale_intensity(2.0), id="scale-intensity"),
    _label_operation(
        lambda data: data.reweight_amplitude(np.linspace(0.5, 1.5, data.n_splats)),
        id="reweight-amplitude",
    ),
    _label_operation(
        lambda data: data.soft_scale_filter(highpass=2.0),
        id="soft-scale-filter",
    ),
    _label_operation(lambda data: data.flattened(), id="flattened"),
    _label_operation(
        lambda data: data.additive_prefix(min(1, data.n_additive_sublods - 1)),
        id="additive-prefix",
    ),
    _label_operation(lambda data: data.at_substitutive(0), id="at-substitutive"),
    _label_operation(lambda data: data.embed_dimension(2.0), id="embed-dimension"),
    _label_operation(
        lambda data: GSplatData.concatenate([data, data]), id="concatenate"
    ),
    _label_operation(
        lambda data: GSplatData.combine_as_new_dimension([data, data]),
        id="combine-as-new-dimension",
    ),
    _label_operation(
        lambda data: GSplatData.merge_with_channel_colors(
            [data, data], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        ),
        id="merge-with-channel-colors",
    ),
    _label_operation(
        lambda data: data.to_spatial_partition(max_elements=4),
        id="to-spatial-partition",
    ),
    _label_operation(
        lambda data: GSplatData.partition_from_regions(
            [
                data.filter(np.arange(data.n_splats) % 2 == 0),
                data.filter(np.arange(data.n_splats) % 2 == 1),
            ]
        ),
        id="partition-from-regions",
    ),
    _label_operation(
        lambda data: make_additive_lod(data, n_lods=3, method="radial"),
        id="make-additive-lod",
    ),
    _label_operation(
        lambda data: decimate(data, target=0.5, method="prefix", verbose=False),
        id="decimate-prefix",
    ),
    _label_operation(
        lambda data: decimate(data, target=0.4, method="auto", verbose=False),
        id="decimate-auto",
    ),
    _label_operation(
        lambda data: decimate(
            data,
            target=0.5,
            method="merge",
            device="cpu",
            lloyd_iterations=1,
            verbose=False,
        ),
        id="decimate-merge",
        refuses=True,
    ),
    _label_operation(
        lambda data: make_substitutive_lod(
            data,
            levels=1,
            compression_factor=2,
            method="greedy",
            device="cpu",
            verbose=False,
        ),
        id="make-substitutive-lod",
        labels_follow_centers=False,
    ),
)


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

    def test_label_ids_survive_subset_reorder_and_partition(self):
        centers = np.column_stack(
            [np.arange(12, dtype=np.float32), np.zeros((12, 2), dtype=np.float32)]
        )
        label_ids = np.arange(12, dtype=np.uint8)
        vocabulary = {i: f"class-{i}" for i in range(12)}
        data = GSplatData(
            centers=centers,
            amplitudes=np.linspace(0.1, 1.2, 12, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (12, 1)
            ),
            label_ids=label_ids,
            label_vocabulary=vocabulary,
        )

        filtered = data.filter(np.arange(12) % 2 == 0)
        np.testing.assert_array_equal(filtered.label_ids, label_ids[::2])
        assert filtered.label_vocabulary == vocabulary

        translated = data.translate(np.array([3.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(translated.label_ids, label_ids)

        ladder = make_additive_lod(data, n_lods=3, method="radial")
        assert ladder.label_ids is not None
        for center, label_id in zip(ladder.centers, ladder.label_ids):
            assert int(label_id) == int(center[0])

        partition = data.to_spatial_partition(max_elements=4)
        for child in partition.children:
            for sublod in child.additive_sublods:
                assert sublod.label_ids is not None
                for center, label_id in zip(sublod.centers, sublod.label_ids):
                    assert int(label_id) == int(center[0])
                assert sublod.label_vocabulary == vocabulary

    @pytest.mark.parametrize(
        "operation,refuses,labels_follow_centers", _LABEL_CHANNEL_OPERATIONS
    )
    @pytest.mark.parametrize("laddered", [False, True], ids=["flat", "laddered"])
    @pytest.mark.parametrize("labeled", [False, True], ids=["unlabeled", "labeled"])
    def test_operations_carry_or_refuse_categorical_channel(
        self,
        operation,
        refuses: bool,
        labels_follow_centers: bool,
        laddered: bool,
        labeled: bool,
    ) -> None:
        centers = np.column_stack(
            [np.arange(12, dtype=np.float32), np.zeros((12, 2), dtype=np.float32)]
        )
        amplitudes = np.ones(12, dtype=np.float32)
        amplitudes[9] = 3.0
        data = GSplatData(
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (12, 1)
            ),
        )
        if labeled:
            data = data.with_label_ids(
                np.arange(12, dtype=np.uint8) % _LABEL_COUNT,
                _LABEL_VOCABULARY,
            )
        if laddered:
            data = make_additive_lod(data, n_lods=3, method="radial")

        if labeled and refuses:
            with pytest.raises(
                ValueError,
                match="cannot coarsen: input carries categorical channel 'label_ids'",
            ):
                operation(data)
            return

        result = operation(data)

        node = result.tree if isinstance(result, GSplatData) else result
        sublods = [
            sublod for leaf in iter_leaves(node) for sublod in leaf.additive_sublods
        ]
        assert sublods
        if not labeled:
            assert all(sublod.label_ids is None for sublod in sublods)
            assert all(sublod.label_vocabulary is None for sublod in sublods)
            return

        for sublod in sublods:
            assert sublod.label_ids is not None
            assert sublod.label_vocabulary == _LABEL_VOCABULARY
            if labels_follow_centers:
                expected = np.rint(sublod.centers[:, 0]).astype(np.int64) % _LABEL_COUNT
                np.testing.assert_array_equal(sublod.label_ids, expected)
            else:
                assert set(np.asarray(sublod.label_ids).tolist()) == set(
                    _LABEL_VOCABULARY
                )

    def test_with_label_ids_accepts_integer_sequence(self) -> None:
        source = _make_3d_gsplat(6)
        label_ids = [0, 1, 0, 2, 1, 2]
        vocabulary = {0: "background", 1: "left", 2: "right"}

        labeled = source.with_label_ids(label_ids, vocabulary)

        np.testing.assert_array_equal(labeled.label_ids, label_ids)
        assert labeled.label_vocabulary == vocabulary
        assert source.label_ids is None
        assert source.label_vocabulary is None

    def test_without_label_ids_preserves_additive_ladder_and_stats(self) -> None:
        source = _make_3d_gsplat(12).with_label_ids(
            np.arange(12, dtype=np.uint8) % 3,
            {0: "background", 1: "left", 2: "right"},
        )
        ladder = make_additive_lod(source, n_lods=3, method="radial")
        ladder.stats["psnr_db"] = 42.0
        for index, sublod in enumerate(ladder.additive_sublods):
            sublod.stats["rung"] = index

        stripped = ladder.without_label_ids()

        assert stripped.n_additive_sublods == ladder.n_additive_sublods
        assert stripped.stats == ladder.stats
        for original, result in zip(ladder.additive_sublods, stripped.additive_sublods):
            np.testing.assert_array_equal(result.centers, original.centers)
            np.testing.assert_array_equal(result.amplitudes, original.amplitudes)
            np.testing.assert_array_equal(
                result.cholesky_factors, original.cholesky_factors
            )
            assert result.stats == original.stats
            assert result.label_ids is None
            assert result.label_vocabulary is None
        assert ladder.label_ids is not None
        assert ladder.label_vocabulary is not None

    def test_derived_operations_do_not_rescan_label_membership(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _make_3d_gsplat(12).with_label_ids(
            np.arange(12, dtype=np.uint8),
            {i: f"class-{i}" for i in range(12)},
        )

        def fail_unique(*args: object, **kwargs: object) -> None:
            raise AssertionError("derived data re-scanned categorical membership")

        monkeypatch.setattr(np, "unique", fail_unique)
        filtered = data.filter(np.arange(12) % 2 == 0)

        np.testing.assert_array_equal(filtered.label_ids, np.arange(0, 12, 2))
        assert filtered.label_vocabulary == data.label_vocabulary

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

    def test_concatenate_requires_identical_label_vocabularies(self):
        left = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.ones(2, dtype=np.float32),
            cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (2, 1)).astype(np.float32),
            label_ids=np.array([0, 1], dtype=np.uint8),
            label_vocabulary={0: "zero", 1: "one"},
        )
        right = GSplatData(
            centers=np.ones((2, 3), dtype=np.float32),
            amplitudes=np.ones(2, dtype=np.float32),
            cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (2, 1)).astype(np.float32),
            label_ids=np.array([0, 1], dtype=np.uint8),
            label_vocabulary={0: "background", 1: "foreground"},
        )

        with pytest.raises(ValueError, match="different label_vocabulary"):
            GSplatData.concatenate([left, right])

    def test_all_empty_concatenate_preserves_compatible_labels(self):
        vocabulary = {0: "zero"}

        def empty() -> GSplatData:
            return GSplatData(
                centers=np.empty((0, 3), dtype=np.float32),
                amplitudes=np.empty(0, dtype=np.float32),
                cholesky_factors=np.empty((0, 6), dtype=np.float32),
                label_ids=np.empty(0, dtype=np.uint8),
                label_vocabulary=vocabulary,
            )

        merged = GSplatData.concatenate([empty(), empty()])
        assert merged.label_ids is not None
        assert merged.label_ids.dtype == np.uint8
        assert merged.label_ids.size == 0
        assert merged.label_vocabulary == vocabulary

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


class TestSubstitutivePreservation:
    """Spatial & intensity transforms must rebuild EVERY substitutive level,
    not silently collapse the pyramid to the finest (mirrors filter_by/cull).
    """

    def _make_pyramid(self, counts=(100, 30, 10), ndim=3, seed=7):
        rng = np.random.RandomState(seed)
        tril = ndim * (ndim + 1) // 2
        levels = []
        for i, n in enumerate(counts):
            chol = np.zeros((n, tril), dtype=np.float32)
            k = 0
            for row in range(ndim):
                for col in range(row + 1):
                    if row == col:
                        chol[:, k] = 1.0
                    k += 1
            sub = AdditiveSubLOD(
                centers=(rng.rand(n, ndim).astype(np.float32) * 100),
                amplitudes=(rng.rand(n).astype(np.float32) + 0.1),
                cholesky_factors=chol,
            )
            levels.append(
                SubstitutiveLevel(
                    additive_sublods=[sub],
                    compression_factor=4**i,
                    parent_method=None if i == 0 else "kmeans_lloyd",
                    level_index=i,
                )
            )
        return GSplatData.from_substitutive_levels(levels)

    def test_transform_preserves_pyramid(self):
        data = self._make_pyramid()
        data = GSplatData.from_substitutive_levels(
            [
                replace(
                    level,
                    stats={
                        **level.stats,
                        "median_footprint": float(index + 1),
                        "footprint_dims": [0, 1, 2],
                    },
                )
                for index, level in enumerate(data.substitutive_levels)
            ]
        )
        out = data.transform(np.eye(3) * 2.0)
        assert out.n_substitutive == 3
        for s in range(3):
            np.testing.assert_allclose(
                out.at_substitutive(s).centers,
                data.at_substitutive(s).centers * 2.0,
                atol=1e-4,
            )
            assert out.substitutive_levels[s].stats[
                "median_footprint"
            ] == pytest.approx(2.0 * (s + 1))

    def test_without_label_ids_preserves_pyramid(self) -> None:
        data = self._make_pyramid(counts=(12, 5))
        vocabulary = {0: "background", 1: "foreground"}
        labeled = GSplatData.from_substitutive_levels(
            [
                replace(
                    level,
                    additive_sublods=[
                        replace(
                            sublod,
                            label_ids=np.arange(sublod.n_splats, dtype=np.uint8) % 2,
                            label_vocabulary=vocabulary,
                        )
                        for sublod in level.additive_sublods
                    ],
                )
                for level in data.substitutive_levels
            ],
            stats={"psnr_db": 42.0},
        )

        stripped = labeled.without_label_ids()

        assert stripped.n_substitutive == labeled.n_substitutive
        assert stripped.stats == labeled.stats
        for original_level, result_level in zip(
            labeled.substitutive_levels, stripped.substitutive_levels
        ):
            assert result_level.compression_factor == original_level.compression_factor
            assert result_level.parent_method == original_level.parent_method
            assert result_level.level_index == original_level.level_index
            for original, result in zip(
                original_level.additive_sublods, result_level.additive_sublods
            ):
                np.testing.assert_array_equal(result.centers, original.centers)
                assert result.label_ids is None
                assert result.label_vocabulary is None

    def test_translate_preserves_pyramid(self):
        data = self._make_pyramid()
        offset = np.array([10, 20, 30], dtype=np.float32)
        out = data.translate(offset)
        assert out.n_substitutive == 3
        for s in range(3):
            np.testing.assert_allclose(
                out.at_substitutive(s).centers,
                data.at_substitutive(s).centers + offset,
                atol=1e-4,
            )

    def test_scale_intensity_preserves_pyramid(self):
        data = self._make_pyramid()
        out = data.scale_intensity(3.0)
        assert out.n_substitutive == 3
        for s in range(3):
            np.testing.assert_allclose(
                out.at_substitutive(s).amplitudes,
                data.at_substitutive(s).amplitudes * 3.0,
                atol=1e-5,
            )

    def test_affine_and_clamp_intensity_preserve_pyramid(self):
        data = self._make_pyramid()
        assert data.affine_intensity(2.0, 0.5).n_substitutive == 3
        assert data.clamp_intensity(min=0.2, max=0.8).n_substitutive == 3

    def test_normalize_intensity_preserves_pyramid_and_global_scale(self):
        data = self._make_pyramid()
        out = data.normalize_intensity(target_max=1.0)
        assert out.n_substitutive == 3
        # A single global factor (finest max) is applied uniformly to all levels.
        factor = 1.0 / float(data.at_substitutive(0).amplitudes.max())
        for s in range(3):
            np.testing.assert_allclose(
                out.at_substitutive(s).amplitudes,
                data.at_substitutive(s).amplitudes * factor,
                atol=1e-5,
            )

    def test_center_at_centroid_preserves_pyramid_with_uniform_shift(self):
        data = self._make_pyramid()
        out = data.center_at_centroid()
        assert out.n_substitutive == 3
        # The centroid is computed once from the finest level and the SAME shift
        # is applied to every level (per-level centroids would shift levels apart).
        finest = data.at_substitutive(0)
        centroid = (finest.centers.T @ finest.amplitudes) / finest.amplitudes.sum()
        for s in range(3):
            np.testing.assert_allclose(
                out.at_substitutive(s).centers,
                data.at_substitutive(s).centers - centroid,
                atol=1e-4,
            )

    def test_with_colors_broadcast_preserves_pyramid(self):
        data = self._make_pyramid()
        out = data.with_colors((1.0, 0.0, 0.0))
        assert out.n_substitutive == 3
        for s in range(3):
            colors = out.at_substitutive(s).colors
            assert colors is not None
            np.testing.assert_allclose(colors, [[1.0, 0.0, 0.0]] * colors.shape[0])

    def test_with_colors_explicit_array_rejected_on_pyramid(self):
        data = self._make_pyramid()
        explicit = np.ones((data.n_splats, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="multi-substitutive"):
            data.with_colors(explicit)

    def test_embed_dimension_scalar_preserves_pyramid(self):
        data = self._make_pyramid()
        out = data.embed_dimension(5.0, sigma=0.0)
        assert out.n_substitutive == 3
        assert out.ndim == 4
        for s in range(3):
            lvl = out.at_substitutive(s)
            assert lvl.n_splats == data.at_substitutive(s).n_splats
            np.testing.assert_allclose(lvl.centers[:, 3], 5.0)

    def test_embed_dimension_per_splat_array_rejected_on_pyramid(self):
        data = self._make_pyramid()
        with pytest.raises(ValueError, match="multi-substitutive"):
            data.embed_dimension(np.arange(data.n_splats, dtype=np.float32))

    def test_combine_as_new_dimension_preserves_pyramid(self):
        # `luxar gsplat merge --as-dimension` on pyramid inputs (calls
        # embed_dimension per dataset) must keep every substitutive level.
        d1 = self._make_pyramid(seed=1)
        d2 = self._make_pyramid(seed=2)
        out = GSplatData.combine_as_new_dimension([d1, d2], sigma=0.0)
        assert out.n_substitutive == 3
        assert out.ndim == 4
        for s in range(3):
            # each level gets both datasets' splats at that level
            expected = d1.at_substitutive(s).n_splats + d2.at_substitutive(s).n_splats
            assert out.at_substitutive(s).n_splats == expected

    def test_merge_with_channel_colors_preserves_pyramid(self):
        d1 = self._make_pyramid(seed=1)
        d2 = self._make_pyramid(seed=2)
        out = GSplatData.merge_with_channel_colors(
            [d1, d2], channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        )
        assert out.n_substitutive == 3
        for s in range(3):
            lvl = out.at_substitutive(s)
            n1 = d1.at_substitutive(s).n_splats
            assert lvl.n_splats == n1 + d2.at_substitutive(s).n_splats
            assert lvl.colors is not None
            np.testing.assert_allclose(lvl.colors[:n1], [[1.0, 0.0, 0.0]] * n1)
            np.testing.assert_allclose(
                lvl.colors[n1:], [[0.0, 1.0, 0.0]] * (lvl.n_splats - n1)
            )

    @pytest.mark.parametrize("merge", ["concatenate", "channel_colors"])
    def test_pyramid_merge_drops_stale_footprint_stats(self, merge):
        def stamped(data, footprint):
            return GSplatData.from_substitutive_levels(
                [
                    replace(
                        level,
                        stats={
                            **level.stats,
                            "median_footprint": footprint,
                            "footprint_dims": [0, 1, 2],
                        },
                    )
                    for level in data.substitutive_levels
                ]
            )

        left = stamped(self._make_pyramid(seed=1), 1.0)
        right = stamped(self._make_pyramid(seed=2), 8.0)
        if merge == "concatenate":
            out = GSplatData.concatenate([left, right])
        else:
            out = GSplatData.merge_with_channel_colors(
                [left, right],
                channel_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            )

        for level in out.substitutive_levels:
            assert "median_footprint" not in level.stats
            assert "footprint_dims" not in level.stats


class TestDegenerateInputs:
    """Edge-case hardening surfaced by the deep double-check."""

    def _g(self, n=3):
        rng = np.random.RandomState(0)
        return GSplatData(
            centers=rng.rand(n, 3).astype(np.float32) * 10,
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (n, 1)),
        )

    def test_embed_dimension_accepts_0d_array_as_scalar(self):
        # A 0-d numpy array is semantically a scalar — it must broadcast, not be
        # mistaken for a malformed per-splat array. (Pre-fix: np.isscalar(0-d) is
        # False → ValueError "values shape () doesn't match splat count".)
        g = self._g(4)
        out = g.embed_dimension(np.array(7.0))
        assert out.ndim == 4
        assert out.n_splats == 4
        np.testing.assert_allclose(out.centers[:, 3], 7.0)

    def test_center_at_centroid_on_empty_emits_no_warning(self):
        import warnings

        empty = GSplatData(
            centers=np.zeros((0, 3), np.float32),
            amplitudes=np.zeros((0,), np.float32),
            cholesky_factors=np.zeros((0, 6), np.float32),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = empty.center_at_centroid()
        assert out.n_splats == 0
        assert not any("empty slice" in str(w.message).lower() for w in caught), [
            str(w.message) for w in caught
        ]


# ── Sharpness removal regression tests ───────────────────────


class TestGSplatsWithoutSharpness:
    """Verify that GSplats work without sharpness while Points/Lines retain it."""

    def test_gsplats_roundtrip_no_sharpness(self, tmp_path):
        """GSplats should round-trip through zarr without a sharpness attribute."""
        from luxar import Dimensions, LuxarScene, LuxarZarrCompiler

        path = tmp_path / "test.luxar.zarr"
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

        path = tmp_path / "test.luxar.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                positions=np.array([[1, 2, 3]], dtype=np.float32),
                radii=np.array([0.5], dtype=np.float32),
                sharpness=np.array([0.55], dtype=np.float32),
            )

        s = LuxarScene.load(path)
        data = s.get_points("pts")
        assert data["sharpness"] is not None
        np.testing.assert_allclose(data["sharpness"], [0.55], atol=1.0 / 255)

    def test_lines_still_have_sharpness(self, tmp_path):
        """Lines should still support the sharpness parameter."""
        from luxar import Dimensions, LuxarScene, LuxarZarrCompiler

        path = tmp_path / "test.luxar.zarr"
        with LuxarZarrCompiler(path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
                widths=np.array([0.1, 0.1], dtype=np.float32),
                sharpness=np.array([0.5, 0.5], dtype=np.float32),
            )

        s = LuxarScene.load(path)
        data = s.get_lines("ln")
        # Assert the VALUE round-trips (mirrors the points twin above) — a bare
        # `is not None` would pass even if the encoder garbled every value.
        assert data["sharpness"] is not None
        np.testing.assert_allclose(data["sharpness"], [0.5, 0.5], atol=1.0 / 255)

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

    def _make_gsplat(
        self, n: int = 10, truncation_radius: float | None = None
    ) -> GSplatData:
        rng = np.random.RandomState(42)
        return GSplatData(
            centers=rng.rand(n, 3).astype(np.float32),
            amplitudes=rng.rand(n).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
            ),
            **(
                {}
                if truncation_radius is None
                else {"truncation_radius": truncation_radius}
            ),
        )

    def test_default_truncation_radius(self):
        """Default truncation_radius is DEFAULT_TRUNCATION_RADIUS."""
        g = self._make_gsplat()
        assert g.truncation_radius == DEFAULT_TRUNCATION_RADIUS
        assert g.additive_sublods[0].truncation_radius == DEFAULT_TRUNCATION_RADIUS

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
        level = data.substitutive_levels[0]
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
            data.substitutive_levels[0].additive_sublods
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

    def test_default_substitutive_is_not_settable(self):
        # The data-model default is fixed at the finest level (index 0) — it is
        # not a settable, persistable concept, so the constructor no longer
        # accepts the kwarg and the accessors always report the finest level.
        levels = [self._make_substitutive_level(), self._make_substitutive_level()]
        with pytest.raises(TypeError):
            GSplatData(substitutive_levels=levels, default_substitutive=1)  # type: ignore[call-arg]
        data = GSplatData(substitutive_levels=levels)
        assert data.default_substitutive == 0

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
            prefix.additive_sublod(0).centers[0, 0] = 5.0

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
        # The stored level arrays are normal writable arrays (additive_sublod()
        # returns the backing AdditiveSubLOD, not a read-only view).
        assert out.additive_sublod(0).centers.flags.writeable


class TestMergeLodColors:
    """`_merge_lod_colors` — dtype preservation + integer-alpha [0,1] guard."""

    @staticmethod
    def _lod(colors, n):
        return AdditiveSubLOD(
            centers=np.zeros((n, 3), dtype=np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
            ),
            colors=colors,
        )

    def test_mixed_int_rgb_and_float_rgba_keeps_alpha_in_unit_range(self) -> None:
        # A uint8 RGB LOD (opaque = full-scale) merged with a float RGBA LOD:
        # widening the uint8 part must NOT inject a 255-valued alpha into the
        # promoted float result. Pre-fix, widen filled iinfo.max (255) and the
        # concat promoted it to 255.0 — out of the [0, 1] opacity contract.
        from luxar.gsplats.gsplat_data import _merge_lod_colors

        rgb_u8 = self._lod(np.array([[255, 0, 0]], dtype=np.uint8), 1)
        rgba_f = self._lod(np.array([[0.2, 0.4, 0.6, 0.5]], dtype=np.float32), 1)
        merged = _merge_lod_colors([rgb_u8, rgba_f])
        assert merged is not None
        assert merged.dtype == np.float32  # mixed int+float → float32 [0,1]
        assert merged.shape == (2, 4)
        assert merged[:, 3].max() <= 1.0 and merged[:, 3].min() >= 0.0
        # uint8 RGB normalized to [0,1]; its widened alpha is opaque 1.0.
        np.testing.assert_allclose(merged[0], [1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(merged[1], [0.2, 0.4, 0.6, 0.5])

    def test_all_float_rgba_is_unchanged(self) -> None:
        # The live (all-float32) path must be byte-identical to the input.
        from luxar.gsplats.gsplat_data import _merge_lod_colors

        a = np.array([[0.1, 0.2, 0.3, 0.9]], dtype=np.float32)
        b = np.array([[0.4, 0.5, 0.6, 0.2]], dtype=np.float32)
        merged = _merge_lod_colors([self._lod(a, 1), self._lod(b, 1)])
        assert merged is not None and merged.dtype == np.float32
        np.testing.assert_array_equal(merged, np.concatenate([a, b], axis=0))

    def test_float64_colors_are_pinned_to_float32(self) -> None:
        from luxar.gsplats.gsplat_data import _merge_lod_colors

        a = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
        merged = _merge_lod_colors([self._lod(a, 1), self._lod(a, 1)])
        assert merged is not None and merged.dtype == np.float32

    def test_uniform_uint8_preserves_dtype(self) -> None:
        # A uniform-dtype integer merge must PRESERVE that dtype (full-scale =
        # opaque) — NOT normalize to float. uint8 colors are a valid SDR storage
        # form; forcing float32 here diverged multi-LOD from the single-LOD path
        # (self.colors = lod0.colors, which keeps uint8) and silently changed the
        # stored encoding. Regression guard for that divergence.
        from luxar.gsplats.gsplat_data import _merge_lod_colors

        a = np.array([[255, 0, 128]], dtype=np.uint8)
        b = np.array([[0, 64, 255]], dtype=np.uint8)
        merged = _merge_lod_colors([self._lod(a, 1), self._lod(b, 1)])
        assert merged is not None and merged.dtype == np.uint8
        np.testing.assert_array_equal(merged, np.concatenate([a, b], axis=0))

    def test_multi_lod_uint8_matches_single_lod(self) -> None:
        # End-to-end: a uint8-color dataset must expose the SAME .colors dtype
        # whether it has one sub-LOD or several (the divergence this fixes).
        from luxar.gsplats.gsplat_data import GSplatData

        c0 = np.array([[255, 0, 128], [0, 255, 64]], dtype=np.uint8)
        single = GSplatData(
            centers=np.zeros((2, 3), dtype=np.float32),
            amplitudes=np.ones(2, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
            colors=c0,
        )
        multi = GSplatData.from_additive_sublods([self._lod(c0, 2), self._lod(c0, 2)])
        assert single.colors is not None and multi.colors is not None
        assert multi.colors.dtype == single.colors.dtype == np.uint8

    def test_mixed_integer_dtypes_normalize_to_float(self) -> None:
        # uint8 + uint16 (mismatched integer dtypes) can't share a native dtype
        # → normalize each by its own full-scale into float32 [0, 1].
        from luxar.gsplats.gsplat_data import _merge_lod_colors

        u8 = np.array([[255, 0, 128]], dtype=np.uint8)  # → [1, 0, 0.502]
        u16 = np.array([[65535, 0, 32768]], dtype=np.uint16)  # → [1, 0, 0.5]
        merged = _merge_lod_colors([self._lod(u8, 1), self._lod(u16, 1)])
        assert merged is not None and merged.dtype == np.float32
        assert merged.max() <= 1.0 and merged.min() >= 0.0
        np.testing.assert_allclose(merged[0], [1.0, 0.0, 128 / 255], atol=1e-6)
        np.testing.assert_allclose(merged[1], [1.0, 0.0, 32768 / 65535], atol=1e-6)

    def test_uint8_with_white_fill_promotes_to_float(self) -> None:
        # A None (white-filled) part forces the float path even for uniform uint8
        # inputs — the fill is float ones, so the concat must be float [0,1].
        from luxar.gsplats.gsplat_data import _merge_lod_colors

        u8 = np.array([[255, 0, 128]], dtype=np.uint8)
        merged = _merge_lod_colors([self._lod(u8, 1), self._lod(None, 1)])
        assert merged is not None and merged.dtype == np.float32
        np.testing.assert_allclose(merged[0], [1.0, 0.0, 128 / 255], atol=1e-6)
        np.testing.assert_allclose(merged[1], [1.0, 1.0, 1.0])  # white fill


class TestLadderWideColorPolicy:
    """`GSplatData.concatenate` must emit LAYOUT/DTYPE-uniform ladders.

    The viewer fail-fasts on ladders whose levels disagree on color layout
    (RGB vs RGBA) or dtype. With ragged inputs, later levels see a different
    source subset than earlier ones, so the color policy must be decided once
    per ladder, not per level (pre-fix, RGBA 2-level + RGB 3-level emitted an
    RGB tail level beside RGBA ones — an unloadable store from a supported
    `gsplat merge` invocation).
    """

    @staticmethod
    def _view(n_levels: int, colors_per_level):
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        sublods = []
        for k in range(n_levels):
            n = 4
            sublods.append(
                AdditiveSubLOD(
                    centers=np.zeros((n, 3), dtype=np.float32),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=np.tile(
                        np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
                    ),
                    colors=colors_per_level(k, n),
                )
            )
        return GSplatData.from_additive_sublods(sublods)

    def test_ragged_rgba_plus_rgb_yields_uniform_rgba_ladder(self) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        rgba = self._view(2, lambda k, n: np.full((n, 4), 0.5, dtype=np.float32))
        rgb_u8 = self._view(3, lambda k, n: np.full((n, 3), 128, dtype=np.uint8))
        out = GSplatData.concatenate([rgba, rgb_u8])
        layouts = set()
        dtypes = set()
        for k in range(out.n_additive_sublods):
            colors = out.additive_sublod(k).colors
            assert colors is not None
            layouts.add(colors.shape[1])
            dtypes.add(colors.dtype)
        # Pre-fix: level 2 (RGB-source-only) stayed (n,3) uint8 while levels
        # 0-1 were (n,4) float32 — a mixed ladder the viewer rejects.
        assert layouts == {4}, f"mixed layouts survived: {layouts}"
        assert dtypes == {np.dtype(np.float32)}, f"mixed dtypes survived: {dtypes}"
        # uint8 tail level was normalized semantically (÷255), alpha opaque.
        tail = out.additive_sublod(2).colors
        assert tail is not None
        np.testing.assert_allclose(tail[0], [128 / 255] * 3 + [1.0], atol=1e-6)

    def test_uniform_integer_ragged_layout_stays_integer_rgba(self) -> None:
        from luxar.gsplats.gsplat_data import GSplatData

        rgba_u8 = self._view(2, lambda k, n: np.full((n, 4), 200, dtype=np.uint8))
        rgb_u8 = self._view(3, lambda k, n: np.full((n, 3), 100, dtype=np.uint8))
        out = GSplatData.concatenate([rgba_u8, rgb_u8])
        for k in range(out.n_additive_sublods):
            colors = out.additive_sublod(k).colors
            assert colors is not None
            assert colors.shape[1] == 4
            assert colors.dtype == np.uint8
        # Widened integer alpha is full-scale opaque.
        assert int(out.additive_sublod(2).colors[0, 3]) == 255


class TestWriterRejectsMixedLadder:
    """`write_gsplats_tree` refuses to write viewer-unloadable mixed ladders."""

    @staticmethod
    def _sublod(colors, n=4, ndim=3):
        chol = np.zeros((n, ndim * (ndim + 1) // 2), dtype=np.float32)
        chol[:, [0, 2, 5][:ndim] if ndim == 3 else range(chol.shape[1])] = 1.0
        return AdditiveSubLOD(
            centers=np.random.default_rng(0).random((n, ndim)).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
            )
            if ndim == 3
            else chol,
            colors=colors,
        )

    def _write(self, tmp_path, sublods):
        import pytest

        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatLeaf

        leaf = GSplatLeaf(additive_sublods=sublods)
        with pytest.raises(ValueError, match="additive ladder has mixed"):
            write_gsplats_tree(tmp_path / "bad.gsplats.zarr", leaf)

    def test_mixed_color_layout_rejected(self, tmp_path) -> None:
        self._write(
            tmp_path,
            [
                self._sublod(np.full((4, 4), 0.5, dtype=np.float32)),
                self._sublod(np.full((4, 3), 0.5, dtype=np.float32)),
            ],
        )

    def test_mixed_color_dtype_rejected(self, tmp_path) -> None:
        self._write(
            tmp_path,
            [
                self._sublod(np.full((4, 3), 0.5, dtype=np.float32)),
                self._sublod(np.full((4, 3), 128, dtype=np.uint8)),
            ],
        )
