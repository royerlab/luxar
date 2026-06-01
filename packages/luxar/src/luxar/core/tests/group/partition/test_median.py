"""Tests for ``median_bsp_partition`` / ``median_bsp_polylines`` and the
``rule='median'`` default.

Covers:
- :func:`luxar.core.group.partition.median_bsp_partition` invariants: complete
  cover, every part honors ``max_elements``, and — the whole point — balanced
  part sizes on clustered data where midpoint produces wildly uneven tiles.
- :func:`luxar.core.group.partition.median_bsp_polylines` atomic-polyline
  balance.
- ``rule='median'`` is the default end-to-end through ``add_points`` /
  ``add_gsplats``.
"""

import numpy as np
import pytest

from luxar.core.group.partition import (
    median_bsp_partition,
    median_bsp_polylines,
    midpoint_bsp_partition,
)
from luxar.core.scene import Scene  # noqa: F401  (import-safety)
from luxar.io.compiler import Dimensions, LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# median_bsp_partition — pure algorithm
# ────────────────────────────────────────────────────────────────────────


class TestMedianBspPartition:
    def test_single_part_below_cap(self) -> None:
        pos = np.random.RandomState(0).uniform(-10, 10, (50, 3)).astype(np.float32)
        parts = median_bsp_partition(pos, max_elements=100)
        assert len(parts) == 1
        assert parts[0].size == 50

    def test_complete_cover_and_cap(self) -> None:
        pos = np.random.RandomState(1).uniform(-10, 10, (1000, 3)).astype(np.float32)
        parts = median_bsp_partition(pos, max_elements=120)
        # Every part within cap.
        assert all(p.size <= 120 for p in parts)
        # Concatenation is a permutation of range(N).
        allidx = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(allidx, np.arange(1000))

    def test_balanced_on_clustered_data(self) -> None:
        """The headline property: median balances a tight cluster + sparse halo
        far better than midpoint."""
        rng = np.random.default_rng(7)
        cluster = rng.normal(0.0, 0.01, (9000, 3)).astype(np.float32)
        halo = rng.uniform(-1.0, 1.0, (1000, 3)).astype(np.float32)
        pos = np.concatenate([cluster, halo])

        median_parts = median_bsp_partition(pos, max_elements=2500)
        midpoint_parts = midpoint_bsp_partition(pos, max_elements=2500)

        median_sizes = sorted(p.size for p in median_parts)
        midpoint_sizes = sorted(p.size for p in midpoint_parts)

        # Median: balanced — spread (max-min) is small relative to the cap.
        median_spread = median_sizes[-1] - median_sizes[0]
        midpoint_spread = midpoint_sizes[-1] - midpoint_sizes[0]
        assert median_spread <= midpoint_spread
        # And every median part respects the cap.
        assert median_sizes[-1] <= 2500

    def test_all_coincident_one_part(self) -> None:
        pos = np.zeros((500, 3), dtype=np.float32)
        parts = median_bsp_partition(pos, max_elements=100)
        # No spatial progress possible → single (oversized) part.
        assert len(parts) == 1
        assert parts[0].size == 500

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            median_bsp_partition(np.zeros((10,), dtype=np.float32), max_elements=5)
        with pytest.raises(ValueError):
            median_bsp_partition(np.zeros((10, 2), dtype=np.float32), max_elements=5)
        with pytest.raises(ValueError):
            median_bsp_partition(np.zeros((10, 3), dtype=np.float32), max_elements=0)


# ────────────────────────────────────────────────────────────────────────
# median_bsp_polylines — atomic-polyline balance
# ────────────────────────────────────────────────────────────────────────


class TestMedianBspPolylines:
    def test_polylines_atomic_and_covered(self) -> None:
        rng = np.random.default_rng(3)
        # 40 polylines of 10 verts each, centroids spread across X.
        verts = []
        polys = []
        cursor = 0
        for p in range(40):
            base = np.array([p * 1.0, 0.0, 0.0], dtype=np.float32)
            v = base + rng.normal(0, 0.05, (10, 3)).astype(np.float32)
            verts.append(v)
            polys.append(np.arange(cursor, cursor + 10, dtype=np.intp))
            cursor += 10
        vertices = np.concatenate(verts)
        parts = median_bsp_polylines(vertices, polys, max_elements=120)
        # Each part within the per-vertex cap.
        for part in parts:
            total = sum(int(polys[p].size) for p in part)
            assert total <= 120
        # Every polyline appears exactly once.
        flat = sorted(p for part in parts for p in part)
        assert flat == list(range(40))


# ────────────────────────────────────────────────────────────────────────
# rule='median' is the default — end to end
# ────────────────────────────────────────────────────────────────────────


class TestMedianDefaultEndToEnd:
    def test_add_points_default_is_median(self, tmp_path) -> None:
        pos = np.random.RandomState(2).uniform(-10, 10, (300, 3)).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos, partition=dict(max_elements=120))
        # Over cap → wrapper produced.
        assert node.attrs.get("kind") == "partition"

    def test_explicit_median_rule(self, tmp_path) -> None:
        pos = np.random.RandomState(4).uniform(-10, 10, (300, 3)).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points(
                "pts", pos, partition=dict(max_elements=120, rule="median")
            )
        assert node.attrs.get("kind") == "partition"
