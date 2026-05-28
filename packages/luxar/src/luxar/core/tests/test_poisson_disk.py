"""Tests for the Bridson-style Poisson-disk sampler.

Covers:

- :func:`luxar.core._poisson_disk.poisson_disk_order` invariants:
  full permutation coverage, per-level counts sum to N, blue-noise
  minimum-distance respected within a single level.
- ``method='poisson-disk'`` branch in
  :func:`luxar.core.lod_points.compute_additive_order_points` and the
  Lines counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core._poisson_disk import poisson_disk_order
from luxar.core.lod_lines import (
    compute_additive_order_lines,
    identify_polylines,
    make_additive_lod_lines,
)
from luxar.core.lod_points import (
    compute_additive_order_points,
    make_additive_lod_points,
    resolve_additive_axis_points,
)

# ────────────────────────────────────────────────────────────────────────
# Pure-algorithm invariants
# ────────────────────────────────────────────────────────────────────────


class TestPoissonDiskOrder:
    def test_permutation_covers_every_input(self):
        rng = np.random.RandomState(0)
        pos = rng.uniform(-10, 10, (500, 3)).astype(np.float32)
        perm, counts = poisson_disk_order(pos, n_lods=4, seed=42)
        assert perm.size == 500
        assert sorted(perm.tolist()) == list(range(500))
        assert sum(counts) == 500

    def test_n_lods_levels_in_counts(self):
        rng = np.random.RandomState(1)
        pos = rng.uniform(-5, 5, (200, 3)).astype(np.float32)
        _, counts = poisson_disk_order(pos, n_lods=5, seed=0)
        assert len(counts) == 5

    def test_blue_noise_within_a_level(self):
        """Within a single (non-final) level, samples should respect
        the level's radius — no two finer than ``r``."""
        rng = np.random.RandomState(2)
        # Dense input so the rejection radius actually rejects.
        pos = rng.uniform(-1, 1, (300, 3)).astype(np.float32)
        perm, counts = poisson_disk_order(pos, n_lods=4, seed=7)
        # Reconstruct level 0's indices.
        level_0 = perm[: counts[0]]
        # Diagonal is sqrt(12) ≈ 3.464; base_r = diag/2 ≈ 1.732.
        diag = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
        r = diag / 2.0
        sel = pos[level_0]
        # All pairwise distances ≥ r among accepted samples.
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                d = float(np.linalg.norm(sel[i] - sel[j]))
                assert d >= r * 0.99, (
                    f"level-0 samples {i},{j} closer than r ({d} < {r})"
                )

    def test_deterministic_for_same_seed(self):
        rng = np.random.RandomState(3)
        pos = rng.uniform(-3, 3, (100, 3)).astype(np.float32)
        p1, _ = poisson_disk_order(pos, n_lods=3, seed=11)
        p2, _ = poisson_disk_order(pos, n_lods=3, seed=11)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_changes_order(self):
        rng = np.random.RandomState(4)
        pos = rng.uniform(-3, 3, (50, 3)).astype(np.float32)
        p1, _ = poisson_disk_order(pos, n_lods=3, seed=1)
        p2, _ = poisson_disk_order(pos, n_lods=3, seed=2)
        assert not np.array_equal(p1, p2)

    def test_empty_input(self):
        pos = np.empty((0, 3), dtype=np.float32)
        perm, counts = poisson_disk_order(pos, n_lods=3)
        assert perm.size == 0
        assert counts == [0, 0, 0]

    def test_degenerate_all_coincident(self):
        """All-coincident points → everything into LOD 0, rest empty."""
        pos = np.zeros((20, 3), dtype=np.float32)
        perm, counts = poisson_disk_order(pos, n_lods=4)
        assert perm.size == 20
        assert counts == [20, 0, 0, 0]

    def test_rejects_invalid_n_lods(self):
        pos = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="n_lods"):
            poisson_disk_order(pos, n_lods=0)

    def test_rejects_invalid_shape(self):
        with pytest.raises(ValueError, match="d >= 3"):
            poisson_disk_order(np.zeros((5, 2)), n_lods=3)


# ────────────────────────────────────────────────────────────────────────
# Points: method='poisson-disk' integration
# ────────────────────────────────────────────────────────────────────────


class TestPointsPoissonDiskIntegration:
    def test_compute_additive_order_routes_to_poisson(self):
        rng = np.random.RandomState(0)
        pos = rng.uniform(-5, 5, (100, 3)).astype(np.float32)
        perm, counts = compute_additive_order_points(
            pos, method="poisson-disk", n_lods=4
        )
        assert perm.size == 100
        assert sum(counts) == 100

    def test_make_additive_lod_respects_natural_partition(self):
        rng = np.random.RandomState(1)
        pos = rng.uniform(-5, 5, (200, 3)).astype(np.float32)
        levels = make_additive_lod_points(
            pos, method="poisson-disk", n_lods=4
        )
        total = sum(L.size for L in levels)
        assert total == 200

    def test_resolver_accepts_poisson_disk(self):
        spec = resolve_additive_axis_points(
            {"method": "poisson-disk", "n_lods": 3}
        )
        assert spec["method"] == "poisson-disk"
        assert spec["n_lods"] == 3

    def test_resolver_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="poisson-disk"):
            resolve_additive_axis_points({"method": "blue-noise"})


# ────────────────────────────────────────────────────────────────────────
# Lines: method='poisson-disk' over polyline-centroid representatives
# ────────────────────────────────────────────────────────────────────────


class TestLinesPoissonDiskIntegration:
    @staticmethod
    def _segments_data(n_segments: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        vertices = rng.uniform(-5, 5, (2 * n_segments, 3)).astype(np.float32)
        widths = np.full(2 * n_segments, 0.05, dtype=np.float32)
        return vertices, widths

    def test_compute_additive_order_routes_to_poisson(self):
        v, w = self._segments_data(50)
        polys = identify_polylines(v.shape[0], "segments")
        perm, counts = compute_additive_order_lines(
            v, polys, widths=w, method="poisson-disk", n_lods=4
        )
        # One slot per polyline.
        assert perm.size == len(polys)
        assert sum(counts) == len(polys)

    def test_make_additive_lod_lines_with_poisson(self):
        v, w = self._segments_data(60, seed=2)
        levels = make_additive_lod_lines(
            v,
            line_type="segments",
            widths=w,
            method="poisson-disk",
            n_lods=4,
        )
        total_polylines = sum(len(L) for L in levels)
        # Every polyline appears in exactly one level.
        assert total_polylines == 60
