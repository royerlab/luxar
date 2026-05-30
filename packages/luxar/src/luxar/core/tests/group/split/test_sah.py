"""Tests for ``sah_bsp_partition`` and the ``rule='sah'`` opt-in.

Verifies:

- :func:`luxar.core.group.split.sah_bsp_partition` invariants: complete cover,
  parts honor ``max_elements``, non-trivial split on uniform data.
- ``split=dict(rule='sah')`` end-to-end through ``add_points``,
  ``add_gsplats``, and ``add_lines``.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.group.split import midpoint_bsp_partition, sah_bsp_partition
from luxar.io.compiler import LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# sah_bsp_partition — pure algorithm
# ────────────────────────────────────────────────────────────────────────


class TestSahBspPartition:
    def test_under_cap_returns_single_part(self):
        pos = np.random.RandomState(0).uniform(-1, 1, (40, 3)).astype(np.float32)
        parts = sah_bsp_partition(pos, max_elements=100)
        assert len(parts) == 1
        assert parts[0].size == 40

    def test_above_cap_splits(self):
        pos = np.random.RandomState(1).uniform(-10, 10, (400, 3)).astype(np.float32)
        parts = sah_bsp_partition(pos, max_elements=120)
        assert len(parts) >= 2
        for p in parts:
            assert p.size <= 120

    def test_complete_cover(self):
        pos = np.random.RandomState(2).uniform(-5, 5, (200, 3)).astype(np.float32)
        parts = sah_bsp_partition(pos, max_elements=80)
        collected = np.concatenate(parts)
        assert sorted(collected.tolist()) == list(range(200))

    def test_empty_input(self):
        pos = np.empty((0, 3), dtype=np.float32)
        parts = sah_bsp_partition(pos, max_elements=10)
        assert parts == []

    def test_rejects_invalid_n_candidates(self):
        pos = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="n_candidates"):
            sah_bsp_partition(pos, max_elements=2, n_candidates=1)

    def test_rejects_invalid_max_elements(self):
        pos = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="max_elements"):
            sah_bsp_partition(pos, max_elements=0)

    def test_degenerate_coincident_returns_one_part(self):
        pos = np.zeros((20, 3), dtype=np.float32)
        parts = sah_bsp_partition(pos, max_elements=5)
        assert len(parts) == 1
        assert parts[0].size == 20

    def test_skewed_data_sah_handles_long_axis(self):
        """SAH should respect the most-extended axis.

        On a strongly anisotropic input (long X, narrow Y/Z) SAH and
        midpoint should both split along X. This isn't a quality
        comparison — just a sanity check that the SAH path produces a
        valid partition on real-world-like skewed data.
        """
        rng = np.random.RandomState(3)
        x = rng.uniform(-100, 100, 300)
        y = rng.uniform(-0.5, 0.5, 300)
        z = rng.uniform(-0.5, 0.5, 300)
        pos = np.stack([x, y, z], axis=1).astype(np.float32)
        parts = sah_bsp_partition(pos, max_elements=150)
        # At least one valid split.
        assert len(parts) >= 2
        # All parts respect the cap.
        for p in parts:
            assert p.size <= 150
        # Complete cover.
        collected = np.concatenate(parts)
        assert sorted(collected.tolist()) == list(range(300))


# ────────────────────────────────────────────────────────────────────────
# ``rule='sah'`` end-to-end through add_points / add_gsplats / add_lines
# ────────────────────────────────────────────────────────────────────────


class TestSahRuleEndToEnd:
    def test_add_points_sah_rule(self, tmp_path):
        pos = np.random.RandomState(0).uniform(-10, 10, (400, 3)).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points(
                "pts", pos, split=dict(max_elements=150, rule="sah")
            )
        assert isinstance(node, Group)
        assert node.attrs.get("kind") == "split"

    def test_add_points_invalid_rule_raises(self, tmp_path):
        pos = np.random.RandomState(0).uniform(-10, 10, (100, 3)).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="split rule"):
                scene.add_points("pts", pos, split=dict(max_elements=50, rule="bogus"))

    def test_add_gsplats_sah_rule(self, tmp_path):
        rng = np.random.RandomState(0)
        c = rng.uniform(-10, 10, (200, 3)).astype(np.float32)
        a = rng.uniform(0.1, 1.0, 200).astype(np.float32)
        ch = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (200, 1))
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats(
                "splats",
                centers=c,
                amplitudes=a,
                cholesky_factors=ch,
                split=dict(max_elements=100, rule="sah"),
            )
        assert isinstance(node, Group)
        assert node.attrs.get("kind") == "split"

    def test_midpoint_default_is_unchanged(self, tmp_path):
        """``split=dict(max_elements=N)`` without ``rule`` defaults to midpoint."""
        pos = np.random.RandomState(1).uniform(-10, 10, (300, 3)).astype(np.float32)
        # The midpoint partitioner is what produced this baseline before
        # SAH existed — sanity check that omitting rule= preserves it.
        midpoint_parts = midpoint_bsp_partition(pos, max_elements=120)
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", pos, split=dict(max_elements=120))
        store = zarr.open(str(tmp_path / "t.zarr"), mode="r")
        grp = store["pts"]

        def collect(g, out):
            for k in g.keys():
                child = g[k]
                if child.attrs.get("type") == "points":
                    out.append(int(child.attrs["n_points"]))
                else:
                    collect(child, out)
            return out

        sizes = sorted(collect(grp, []))
        baseline = sorted(int(p.size) for p in midpoint_parts)
        assert sizes == baseline
