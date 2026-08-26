"""Tests for voxel-sampled payload agreement."""

import inspect
import warnings

import numpy as np
import pytest

from luxar.demos._support.datasets.payload_agreement import (
    voxel_sampled_payload_agreement,
)


class TestVoxelSampledPayloadAgreement:
    """Tests for the per-splat sidecar / fit correspondence check.

    The invariant: a payload sampled nearest-voxel at the splat centers must give
    two splats that round to the same voxel the same row. A sidecar written in a
    different splat order than the fit violates it almost everywhere.
    """

    @staticmethod
    def _colliding_centers(n: int = 4000, extent: int = 8, seed: int = 0) -> np.ndarray:
        """Centers over a small grid, so many splats share a voxel."""
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, float(extent) - 0.51, (n, 3)).astype(np.float32)

    @staticmethod
    def _sample(vol: np.ndarray, centers: np.ndarray) -> np.ndarray:
        idx = tuple(
            np.clip(np.rint(centers[:, d]).astype(int), 0, vol.shape[d] - 1)
            for d in range(3)
        )
        return vol[idx]

    def test_aligned_labels_score_exactly_one(self) -> None:
        rng = np.random.default_rng(1)
        vol = rng.integers(0, 118, (8, 8, 8)).astype(np.int32)
        centers = self._colliding_centers()
        labels = self._sample(vol, centers)
        assert voxel_sampled_payload_agreement(centers, labels) == 1.0

    def test_aligned_rgb_payload_scores_exactly_one(self) -> None:
        rng = np.random.default_rng(2)
        vol = rng.uniform(0.0, 1.0, (8, 8, 8, 3)).astype(np.float32)
        centers = self._colliding_centers()
        colors = self._sample(vol, centers)
        assert colors.shape == (len(centers), 3)
        assert voxel_sampled_payload_agreement(centers, colors) == 1.0

    def test_permuted_payload_scores_near_zero(self) -> None:
        rng = np.random.default_rng(3)
        vol = rng.uniform(0.0, 1.0, (8, 8, 8, 3)).astype(np.float32)
        centers = self._colliding_centers()
        colors = self._sample(vol, centers)
        shuffled = colors[rng.permutation(len(colors))]
        score = voxel_sampled_payload_agreement(centers, shuffled)
        assert score is not None
        assert score < 0.05

    def test_too_few_colliding_pairs_is_unverifiable(self) -> None:
        # One splat per voxel on a coarse lattice → no same-voxel pair at all.
        centers = (
            np.stack(np.meshgrid(*[np.arange(6.0)] * 3, indexing="ij"), axis=-1)
            .reshape(-1, 3)
            .astype(np.float32)
        )
        labels = np.arange(len(centers), dtype=np.int32)
        assert voxel_sampled_payload_agreement(centers, labels) is None

    def test_min_pairs_threshold_is_respected(self) -> None:
        rng = np.random.default_rng(4)
        vol = rng.integers(0, 118, (8, 8, 8)).astype(np.int32)
        centers = self._colliding_centers(n=4000)
        labels = self._sample(vol, centers)
        # Plenty of collisions for the default, none for an absurd requirement.
        assert voxel_sampled_payload_agreement(centers, labels) == 1.0
        assert voxel_sampled_payload_agreement(centers, labels, min_pairs=10**6) is None

    def test_default_min_pairs_has_statistical_power(self) -> None:
        """The default floor must make a ">= 0.99 agreement" verdict mean something.

        At 64 pairs, ">= 0.99" means "all 64 agree", which a FULLY SHUFFLED
        sidecar reaches with probability p**64 in the payload's own chance level p
        — 52% at p=0.99, 3.7% at p=0.95. Both real datasets offer >= 70,000
        pairs, so the floor costs nothing where it matters.
        """
        default = inspect.signature(voxel_sampled_payload_agreement).parameters[
            "min_pairs"
        ]
        assert default.default >= 1024

    def test_min_pairs_zero_on_a_pair_free_input_is_unverifiable_not_a_crash(
        self,
    ) -> None:
        """`min_pairs=0` must not divide by zero.

        With no colliding pair the final ``count / n_pairs`` is ``0 / 0``, so an
        unclamped floor of 0 raised ``ZeroDivisionError`` on exactly the input the
        helper is supposed to answer ``None`` for.
        """
        centers = (
            np.stack(np.meshgrid(*[np.arange(6.0)] * 3, indexing="ij"), axis=-1)
            .reshape(-1, 3)
            .astype(np.float32)
        )
        labels = np.arange(len(centers), dtype=np.int32)
        assert voxel_sampled_payload_agreement(centers, labels, min_pairs=0) is None
        assert voxel_sampled_payload_agreement(centers, labels, min_pairs=-5) is None

    def test_min_pairs_one_still_judges_a_single_pair(self) -> None:
        """The clamp lowers the floor to 1, it does not disable the verdict."""
        centers = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], dtype=np.float32)
        assert (
            voxel_sampled_payload_agreement(
                centers, np.array([7, 7], dtype=np.int32), min_pairs=0
            )
            == 1.0
        )
        assert (
            voxel_sampled_payload_agreement(
                centers, np.array([7, 8], dtype=np.int32), min_pairs=0
            )
            == 0.0
        )

    def test_zero_column_centers_raise_valueerror(self) -> None:
        """An ``(N, 0)`` array is 2-D but carries no voxel key.

        It used to reach ``np.lexsort`` and raise a bare ``TypeError: need
        sequence of keys with len > 0`` — an internal-looking crash where the
        documented contract is ``ValueError``.
        """
        with pytest.raises(ValueError, match="at least one column"):
            voxel_sampled_payload_agreement(
                np.zeros((10, 0), dtype=np.float32), np.zeros(10, dtype=np.int32)
            )

    def test_length_mismatch_raises(self) -> None:
        centers = self._colliding_centers(n=100)
        with pytest.raises(ValueError, match="length mismatch"):
            voxel_sampled_payload_agreement(centers, np.zeros(99, dtype=np.int32))

    def test_non_2d_centers_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            voxel_sampled_payload_agreement(
                np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.int32)
            )

    def test_two_column_centers_are_judged_not_rejected(self) -> None:
        """A 2D fit is a first-class authoring path — it must not raise.

        The helper keys on EVERY center column, so fewer than three is fine.
        """
        rng = np.random.default_rng(11)
        vol = rng.integers(0, 118, (8, 8)).astype(np.int32)
        centers = rng.uniform(0.0, 7.49, (4000, 2)).astype(np.float32)
        idx = tuple(
            np.clip(np.rint(centers[:, d]).astype(int), 0, vol.shape[d] - 1)
            for d in range(2)
        )
        assert voxel_sampled_payload_agreement(centers, vol[idx]) == 1.0

    def test_stacked_axis_column_is_part_of_the_voxel_key(self) -> None:
        """An nD (stacked) fit keeps its stacked axis in the key.

        Spatial dims come first and the stacked axis LAST, so keying on the first
        three columns alone folds every timepoint of a voxel together and an
        ALIGNED sidecar is rejected (measured 0.646 on this 8-timepoint case).
        """
        rng = np.random.default_rng(12)
        vol = rng.integers(0, 118, (8, 8, 8, 8)).astype(np.int32)  # (z, y, x, t)
        spatial = rng.uniform(0.0, 7.49, (6000, 3)).astype(np.float32)
        t = rng.integers(0, 8, 6000).astype(np.float32)
        centers = np.column_stack([spatial, t]).astype(np.float32)
        idx = tuple(
            np.clip(np.rint(centers[:, d]).astype(int), 0, vol.shape[d] - 1)
            for d in range(4)
        )
        labels = vol[idx]
        assert voxel_sampled_payload_agreement(centers, labels) == 1.0

    def test_whole_payload_row_is_compared_not_just_column_zero(self) -> None:
        """A wide payload must agree in EVERY column, not only the first.

        Comparing `payload[:, 0]` alone would call this aligned: column 0 is
        constant, so it agrees for any permutation whatsoever.
        """
        rng = np.random.default_rng(13)
        centers = self._colliding_centers(n=4000)
        vol = rng.uniform(0.0, 1.0, (8, 8, 8, 3)).astype(np.float32)
        payload = self._sample(vol, centers)
        payload[:, 0] = 0.5  # constant first column
        assert voxel_sampled_payload_agreement(centers, payload) == 1.0
        shuffled = payload[rng.permutation(len(payload))]
        score = voxel_sampled_payload_agreement(centers, shuffled)
        assert score is not None and score < 0.05

    def test_non_finite_centers_are_excluded_without_warning(self) -> None:
        """NaN/inf centers have no voxel — they must not warn or invent collisions.

        `np.rint(...).astype(np.int64)` emits a bare RuntimeWarning (fatal under
        `-W error`, which several tests in this repo use) and maps every
        non-finite row onto ONE sentinel voxel, so a shuffled payload would score
        against pairs that share nothing.
        """
        rng = np.random.default_rng(14)
        vol = rng.integers(0, 118, (8, 8, 8)).astype(np.int32)
        centers = self._colliding_centers(n=4000)
        labels = self._sample(vol, centers)
        # The verdict for the finite rows, for comparison.
        clean = voxel_sampled_payload_agreement(centers, labels)
        assert clean == 1.0

        polluted = centers.copy()
        polluted[:80:4] = np.nan
        polluted[1:80:4] = np.inf
        polluted[2:80:4] = -np.inf
        polluted[3:80:4] = 1e30  # finite, but overflows the int64 voxel cast
        bad_labels = labels.copy()
        bad_labels[:80] = 99  # a value that disagrees with its voxel's neighbours
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score = voxel_sampled_payload_agreement(polluted, bad_labels)
        assert score == 1.0, "non-finite rows must be excluded, not sentinel-voxeled"
