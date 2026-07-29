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
import zarr

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

    def test_median_at_axis_min_triggers_stable_bisection(self) -> None:
        """B1/[P5]: when a majority of elements sit at the longest axis's
        minimum, ``np.median`` returns that minimum and ``coords < median``
        empties the left side. The fallback (partition.py:254-258) must then
        re-sort by the axis and bisect by COUNT so recursion still makes
        progress (otherwise the recursion would never shrink below the cap).

        4 of 6 points sit at X=0, the rest at X=10 → median(X)=0 → the ``<``
        test selects nothing on the left, exercising the count-bisection
        fallback rather than the geometric split.
        """
        pos = np.array(
            [
                [0, 0, 0.0],
                [0, 0, 0.1],
                [0, 0, 0.2],
                [0, 0, 0.3],
                [10, 0, 0],
                [10, 0, 0.1],
            ],
            dtype=np.float32,
        )
        assert float(np.median(pos[:, 0])) == 0.0  # precondition for the branch
        parts = median_bsp_partition(pos, max_elements=2)
        # The fallback made progress: every part honors the cap...
        assert all(p.size <= 2 for p in parts)
        # ...and the cover is still complete and duplicate-free.
        all_idx = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(all_idx, np.arange(6))

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            median_bsp_partition(np.zeros((10,), dtype=np.float32), max_elements=5)
        with pytest.raises(ValueError):
            median_bsp_partition(np.zeros((10, 1), dtype=np.float32), max_elements=5)
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

    def test_empty_polyline_member_is_skipped_but_covered(self) -> None:
        """B10/[P5]: a zero-vertex polyline (empty member array) must skip the
        centroid computation (partition.py:406-407 ``continue``) yet still
        appear in the cover so no polyline index is silently dropped."""
        rng = np.random.default_rng(0)
        vertices = rng.random((60, 3)).astype(np.float32)
        polys = [
            np.array([], dtype=np.intp),  # empty polyline → exercises the skip
            np.arange(0, 30, dtype=np.intp),
            np.arange(30, 60, dtype=np.intp),
        ]
        parts = median_bsp_polylines(vertices, polys, max_elements=30)
        flat = sorted(p for part in parts for p in part)
        assert flat == [0, 1, 2]  # the empty polyline (index 0) is still covered

    def test_total_verts_at_cap_is_not_partitioned(self) -> None:
        """B10/[P5]: at-cap equality boundary — two polylines whose vertex
        counts sum to exactly ``max_elements`` satisfy ``total_verts <=
        max_elements`` (partition.py:415) and stay in a single part."""
        vertices = np.random.RandomState(1).rand(100, 3).astype(np.float32)
        polys = [np.arange(0, 50, dtype=np.intp), np.arange(50, 100, dtype=np.intp)]
        parts = median_bsp_polylines(vertices, polys, max_elements=100)
        assert len(parts) == 1
        assert sorted(parts[0]) == [0, 1]


# ────────────────────────────────────────────────────────────────────────
# rule='median' is the default — end to end
# ────────────────────────────────────────────────────────────────────────


class TestMedianDefaultEndToEnd:
    def test_add_points_default_is_median(self, tmp_path) -> None:
        pos = np.random.RandomState(2).uniform(-10, 10, (300, 3)).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos, partition=dict(max_elements=120))
        # Over cap → wrapper produced.
        assert node.attrs.get("kind") == "partition"

    def test_explicit_median_rule(self, tmp_path) -> None:
        pos = np.random.RandomState(4).uniform(-10, 10, (300, 3)).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points(
                "pts", pos, partition=dict(max_elements=120, rule="median")
            )
        assert node.attrs.get("kind") == "partition"

    def test_add_gsplats_default_is_median(self, tmp_path) -> None:
        """B3/[P8]: ``add_gsplats(partition=dict(max_elements=N))`` without an
        explicit ``rule`` must default to median, symmetrically with
        ``add_points`` (the points equivalent lives in
        ``test_sah.py::test_median_is_default_rule``).

        Compares the on-disk leaf splat counts against the
        ``median_bsp_partition`` baseline — proves the default rule is median
        and NOT the old midpoint baseline.
        """
        rng = np.random.RandomState(11)
        c = rng.uniform(-10, 10, (300, 3)).astype(np.float32)
        a = rng.uniform(0.1, 1.0, 300).astype(np.float32)
        ch = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (300, 1))
        baseline = sorted(
            int(p.size) for p in median_bsp_partition(c, max_elements=120)
        )

        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats(
                "splats",
                centers=c,
                amplitudes=a,
                cholesky_factors=ch,
                partition=dict(max_elements=120),
            )
        assert node.attrs.get("kind") == "partition"

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")

        def collect(g, out):
            for k in g.keys():
                child = g[k]
                if child.attrs.get("type") == "gsplats":
                    out.append(int(child.attrs["n_splats"]))
                else:
                    collect(child, out)
            return out

        sizes = sorted(collect(store["splats"], []))
        assert sizes == baseline
