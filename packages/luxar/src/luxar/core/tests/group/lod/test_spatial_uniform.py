"""Tests for ``stratified_grid_order`` — the spatial-uniform LOD sampler.

B13/[P6]: the sampler is deterministic (stable scan, no RNG) but its
*contract* is statistical — each coarse-to-fine level should pick at most
one element per occupied grid cell, and the union of all levels must be a
complete, duplicate-free permutation. Prior coverage exercised this only
indirectly through ``test_points.py`` (a single ``counts[0] <= 8`` check).
These tests assert the structural invariants directly, plus the parameter
validation guards (B9-M3/[P5]) that had no dedicated test.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core.group.lod.spatial_uniform import stratified_grid_order


class TestStratifiedGridInvariants:
    def test_permutation_is_complete_bijection(self) -> None:
        """Every element appears exactly once; counts sum to N."""
        pos = np.random.RandomState(0).rand(500, 3).astype(np.float32)
        perm, counts = stratified_grid_order(pos, n_lods=5)
        np.testing.assert_array_equal(np.sort(perm), np.arange(500))
        assert sum(counts) == 500
        assert len(counts) == 5

    def test_each_level_has_at_most_one_element_per_cell(self) -> None:
        """The defining invariant: within a single LOD level, no two picked
        elements share the same grid cell at that level's resolution
        (``2^(level+1)`` per axis). This is what makes any cumulative prefix
        approximately spatially uniform.

        The FINAL level is excluded: it also absorbs the unassigned
        spillover tail (elements that share the finest cell), which
        deliberately violates per-cell uniqueness.
        """
        pos = np.random.RandomState(1).uniform(-5, 5, (400, 3)).astype(np.float64)
        n_lods = 4
        perm, counts = stratified_grid_order(pos, n_lods=n_lods)

        mins = pos[:, :3].min(axis=0)
        maxs = pos[:, :3].max(axis=0)
        extents = maxs - mins
        extents[extents == 0.0] = 1.0

        cursor = 0
        for level in range(n_lods - 1):  # exclude final (tail-absorbing) level
            level_idx = perm[cursor : cursor + counts[level]]
            cursor += counts[level]
            if level_idx.size == 0:
                continue
            res = 1 << min(level + 1, 31)
            normalized = (pos[level_idx, :3] - mins) / extents
            bucket = np.minimum((normalized * res).astype(np.int64), res - 1)
            bucket_id = bucket[:, 0] * (res * res) + bucket[:, 1] * res + bucket[:, 2]
            # No duplicate cell within this level.
            assert len(set(bucket_id.tolist())) == bucket_id.size, (
                f"level {level} picked >1 element in some cell"
            )

    def test_coarsest_level_count_bounded_by_grid_cells(self) -> None:
        """LOD 0 uses a 2×2×2 grid → at most 8 cells → at most 8 picks."""
        pos = np.random.RandomState(2).rand(1000, 3).astype(np.float32)
        _, counts = stratified_grid_order(pos, n_lods=4)
        assert counts[0] <= 8

    def test_deterministic_for_same_input(self) -> None:
        """No RNG: identical input yields identical permutation and counts."""
        pos = np.random.RandomState(3).rand(200, 3).astype(np.float32)
        perm_a, counts_a = stratified_grid_order(pos, n_lods=4)
        perm_b, counts_b = stratified_grid_order(pos, n_lods=4)
        np.testing.assert_array_equal(perm_a, perm_b)
        assert counts_a == counts_b

    def test_coincident_points_all_land_in_last_level(self) -> None:
        """All-coincident points share every grid cell → one pick at LOD 0,
        the rest spill into the final level (the unassigned-tail path)."""
        pos = np.zeros((10, 3), dtype=np.float32)
        perm, counts = stratified_grid_order(pos, n_lods=4)
        np.testing.assert_array_equal(np.sort(perm), np.arange(10))
        # One element picked per level from the single shared cell (3 levels),
        # then the remaining 7 spill into the final level's tail.
        assert counts == [1, 1, 1, 7]

    def test_empty_input_returns_zero_counts(self) -> None:
        pos = np.zeros((0, 3), dtype=np.float32)
        perm, counts = stratified_grid_order(pos, n_lods=3)
        assert perm.size == 0
        assert counts == [0, 0, 0]


class TestStratifiedGridValidation:
    """B9-M3/[P5]: parameter-validation guards (previously untested)."""

    def test_n_lods_below_one_raises(self) -> None:
        pos = np.random.RandomState(0).rand(10, 3).astype(np.float32)
        with pytest.raises(ValueError, match="n_lods must be >= 1"):
            stratified_grid_order(pos, n_lods=0)

    def test_positions_too_few_dims_raises(self) -> None:
        pos = np.random.RandomState(0).rand(10, 2).astype(np.float32)
        with pytest.raises(ValueError, match="d >= 3"):
            stratified_grid_order(pos, n_lods=3)

    def test_positions_not_2d_raises(self) -> None:
        pos = np.zeros((10,), dtype=np.float32)
        with pytest.raises(ValueError, match="d >= 3"):
            stratified_grid_order(pos, n_lods=3)
