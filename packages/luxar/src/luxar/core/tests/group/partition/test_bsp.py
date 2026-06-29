"""Tests for the kind=partition ``Group`` (BSP-decomposition wrapper).

Covers:

- ``midpoint_bsp_partition`` directly — single part below cap, multi-part
  recursion, degenerate (all-coincident) positions, axis selection.
- The ``partition=`` convenience kwarg on ``add_points`` and ``add_gsplats``:
  end-to-end round-trip through the writer.
- Homogeneity / display_type derivation and parent ``position_bounds``
  union.
- ``layer=True`` propagation onto the wrapper (not the leaf parts).
- The standalone ``add_partition_group`` builder.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.group.partition import (
    DEFAULT_MAX_ELEMENTS,
    midpoint_bsp_partition,
    validate_partition_group,
)
from luxar.io.compiler import LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# midpoint_bsp_partition — pure algorithm
# ────────────────────────────────────────────────────────────────────────


class TestMidpointBspPartition:
    def test_single_part_when_under_cap(self) -> None:
        """N ≤ max_elements → a single part covering the whole input."""
        pos = np.random.RandomState(0).rand(50, 3).astype(np.float32)
        parts = midpoint_bsp_partition(pos, max_elements=100)
        assert len(parts) == 1
        assert parts[0].size == 50
        np.testing.assert_array_equal(np.sort(parts[0]), np.arange(50))

    def test_two_parts_when_over_cap(self) -> None:
        """N > max_elements → recursive partition until each part fits."""
        pos = np.random.RandomState(1).rand(200, 3).astype(np.float32)
        parts = midpoint_bsp_partition(pos, max_elements=120)
        assert len(parts) >= 2
        for p in parts:
            assert p.size <= 120
        total = sum(int(p.size) for p in parts)
        assert total == 200
        # Concatenation is a permutation of all original indices.
        all_idx = np.concatenate(parts)
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(200))

    def test_partition_partitions_along_longest_axis(self) -> None:
        """A long-X data block should partition along X first."""
        # 100 points spread along X (range 100), short on Y and Z (range 1).
        rng = np.random.RandomState(2)
        pos = np.stack(
            [
                rng.uniform(0, 100, 100),
                rng.uniform(0, 1, 100),
                rng.uniform(0, 1, 100),
            ],
            axis=1,
        ).astype(np.float32)
        parts = midpoint_bsp_partition(pos, max_elements=60)
        assert len(parts) == 2
        # Each part should occupy roughly the lower or upper half of X.
        # Data-agnostic: midpoint splits at the geometric midpoint of the
        # actual X range, not a hardcoded 50.
        x_mid = float((pos[:, 0].min() + pos[:, 0].max()) * 0.5)
        x_means = [float(pos[p][:, 0].mean()) for p in parts]
        assert min(x_means) < x_mid < max(x_means)

    def test_coincident_positions_emit_as_oversized_single_part(self) -> None:
        """All-coincident positions cannot partition further; one (oversized) part."""
        pos = np.zeros((10, 3), dtype=np.float32)
        parts = midpoint_bsp_partition(pos, max_elements=5)
        assert len(parts) == 1
        assert parts[0].size == 10

    def test_zero_elements_returns_empty_list(self) -> None:
        pos = np.zeros((0, 3), dtype=np.float32)
        parts = midpoint_bsp_partition(pos, max_elements=10)
        assert parts == []

    def test_single_element_returns_single_part(self) -> None:
        """B10/[P5]: N=1 boundary — one element under any cap is one part."""
        pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        parts = midpoint_bsp_partition(pos, max_elements=100)
        assert len(parts) == 1
        np.testing.assert_array_equal(parts[0], np.array([0]))

    def test_max_elements_one_splits_to_singletons(self) -> None:
        """B10/[P5]: max_elements=1 (minimum valid cap) on distinct points
        recurses to maximal depth — every part is a single element and the
        cover stays complete."""
        pos = np.array([[i, 0.0, 0.0] for i in range(8)], dtype=np.float32)
        parts = midpoint_bsp_partition(pos, max_elements=1)
        assert len(parts) == 8
        assert all(p.size == 1 for p in parts)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(8))

    def test_rejects_non_2d_positions(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            midpoint_bsp_partition(np.zeros((10,), dtype=np.float32), max_elements=5)

    def test_rejects_fewer_than_3_spatial_dims(self) -> None:
        with pytest.raises(ValueError, match="at least 3 spatial dimensions"):
            midpoint_bsp_partition(np.zeros((10, 2), dtype=np.float32), max_elements=5)

    def test_rejects_zero_max_elements(self) -> None:
        with pytest.raises(ValueError, match="max_elements must be >= 1"):
            midpoint_bsp_partition(np.zeros((10, 3), dtype=np.float32), max_elements=0)

    def test_max_elements_default(self) -> None:
        """Default cap is large enough that a 10K-point input stays as one part."""
        rng = np.random.RandomState(3)
        pos = rng.rand(10_000, 3).astype(np.float32)
        parts = midpoint_bsp_partition(pos, max_elements=DEFAULT_MAX_ELEMENTS)
        assert len(parts) == 1


# ────────────────────────────────────────────────────────────────────────
# partition= kwarg on add_points
# ────────────────────────────────────────────────────────────────────────


class TestAddPointsPartition:
    def test_none_is_default_no_wrapping(self, tmp_path) -> None:
        """``partition=None`` (default) writes a plain Points node."""
        rng = np.random.RandomState(0)
        pos = rng.rand(50, 3).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos)
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        assert store["pts"].attrs["type"] == "points"
        assert store["pts"].attrs["n_points"] == 50
        # Return value is the leaf type when no partitionting happened.
        assert type(node).__name__ == "Points"

    def test_under_cap_falls_through_to_single_leaf(self, tmp_path) -> None:
        """``partition=True`` on a small input stays as a single Points node."""
        rng = np.random.RandomState(1)
        pos = rng.rand(50, 3).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos, partition=True)
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        # Default cap is 1M; 50 points → single part, no wrapper.
        assert store["pts"].attrs["type"] == "points"
        assert type(node).__name__ == "Points"

    def test_oversized_coincident_part_warns(self, tmp_path, capsys) -> None:
        """L2: coincident data that can't be split below the cap is written as
        one oversized part WITH a warning (was silent)."""
        pos = np.zeros((300, 3), dtype=np.float32)  # all coincident
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos, partition=dict(max_elements=120))
        # Single oversized leaf (no wrapper), but the cap violation is surfaced.
        assert type(node).__name__ == "Points"
        out = capsys.readouterr().out
        assert "oversized" in out and "max_elements" in out

    def test_over_cap_creates_partition_wrapper(self, tmp_path) -> None:
        """``partition=dict(max_elements=N)`` over the cap produces a wrapper Group."""
        rng = np.random.RandomState(2)
        pos = rng.rand(300, 3).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos, partition=dict(max_elements=120))
            assert isinstance(node, Group)
            assert node.attrs.get("kind") == "partition"

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["pts"]
        assert grp.attrs["type"] == "group"
        assert grp.attrs["kind"] == "partition"
        assert grp.attrs["display_type"] == "points"
        assert grp.attrs["max_elements"] == 120
        # Children are part_0, part_1, ... and each is a Points node.
        children = sorted(grp.keys())
        assert all(c.startswith("part_") for c in children)
        assert all(grp[c].attrs["type"] == "points" for c in children)
        # All children together total 300 points.
        total = sum(int(grp[c].attrs["n_points"]) for c in children)
        assert total == 300

    def test_position_bounds_is_union_of_children(self, tmp_path) -> None:
        """Wrapper's position_bounds spans every child's bbox."""
        rng = np.random.RandomState(3)
        pos = rng.uniform(-10, 10, (250, 3)).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", pos, partition=dict(max_elements=100))

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["pts"]
        parent_pb = grp.attrs["position_bounds"]
        # The parent bbox should contain every data point.
        for ax in range(3):
            assert parent_pb["min"][ax] <= float(pos[:, ax].min()) + 1e-5
            assert parent_pb["max"][ax] >= float(pos[:, ax].max()) - 1e-5

    def test_layer_flag_lands_on_wrapper(self, tmp_path) -> None:
        """``layer=True`` rides onto the wrapper, not each part."""
        rng = np.random.RandomState(4)
        pos = rng.rand(300, 3).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", pos, partition=dict(max_elements=120), layer=True)

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["pts"]
        assert grp.attrs.get("layer") is True
        # Each part should NOT carry layer=true.
        for child in grp.keys():
            assert grp[child].attrs.get("layer") is not True

    def test_per_point_arrays_slice_correctly(self, tmp_path) -> None:
        """Per-point arrays slice per part — total count matches input."""
        rng = np.random.RandomState(5)
        n = 300
        pos = rng.uniform(0, 100, (n, 3)).astype(np.float32)
        # Per-point scalars to verify the slicing path runs (the writer's
        # quantization makes exact-value comparisons brittle; rely on
        # per-part counts and total).
        scalars = rng.rand(n).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                pos,
                scalars=scalars,
                partition=dict(max_elements=120),
                colormap="viridis",
            )

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["pts"]
        total = 0
        for child in sorted(grp.keys()):
            assert "scalars" in grp[child]
            total += grp[child]["scalars"].shape[0]
        assert total == n

    def test_image_labels_rejected(self, tmp_path) -> None:
        """image_labels alongside partition= raises (sparse-dict slicing untrivial)."""
        rng = np.random.RandomState(6)
        pos = rng.rand(300, 3).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="image_labels is not supported"):
                scene.add_points(
                    "pts",
                    pos,
                    image_labels=[b"x"] * 300,
                    partition=dict(max_elements=120),
                )

    def test_invalid_spec_raises(self, tmp_path) -> None:
        """partition= must be None, True, or dict.

        ``add_points`` wraps internal errors and re-raises as ``ValueError``;
        the underlying ``TypeError`` is the chained cause.
        """
        pos = np.zeros((300, 3), dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="partition must be None"):
                scene.add_points("pts", pos, partition="auto")  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────────
# partition= kwarg on add_gsplats
# ────────────────────────────────────────────────────────────────────────


class TestAddGSplatsPartition:
    @staticmethod
    def _make_gsplats(n: int = 300, seed: int = 0) -> tuple:
        rng = np.random.RandomState(seed)
        centers = rng.uniform(-10, 10, (n, 3)).astype(np.float32)
        amplitudes = rng.uniform(0.1, 1.0, n).astype(np.float32)
        # Packed isotropic Cholesky: [1, 0, 1, 0, 0, 1] broadcast.
        chol = np.tile(np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1))
        return centers, amplitudes, chol

    def test_over_cap_creates_partition_wrapper(self, tmp_path) -> None:
        c, a, ch = self._make_gsplats(n=300)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats(
                "splats",
                centers=c,
                amplitudes=a,
                cholesky_factors=ch,
                partition=dict(max_elements=120),
            )
            assert isinstance(node, Group)
            assert node.attrs.get("kind") == "partition"

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["splats"]
        assert grp.attrs["display_type"] == "gsplats"
        total = sum(int(grp[c].attrs["n_splats"]) for c in grp.keys())
        assert total == 300

    def test_amplitudes_and_cholesky_slice_correctly(self, tmp_path) -> None:
        """Per-splat amplitudes and (N, k) Cholesky arrays partition correctly.

        The writer quantizes amplitudes (``bounded_scalar_uint8``) so the
        exact pre-write values aren't recoverable; we verify per-part
        counts + total instead.
        """
        c, a, ch = self._make_gsplats(n=300, seed=10)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "splats",
                centers=c,
                amplitudes=a,
                cholesky_factors=ch,
                partition=dict(max_elements=120),
            )

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["splats"]
        total_amps = 0
        for child in sorted(grp.keys()):
            assert grp[child].attrs["type"] == "gsplats"
            total_amps += grp[child]["amplitudes"].shape[0]
            # Cholesky factors are present on each part. v3.1 stores them as a
            # diagonal + off-diagonal split (the writer compresses
            # uniform-Cholesky inputs, so the row count varies).
            assert "cholesky_factors_diag" in grp[child]
        assert total_amps == 300


# ────────────────────────────────────────────────────────────────────────
# Standalone builder: add_partition_group + validator
# ────────────────────────────────────────────────────────────────────────


class TestAddPartitionGroup:
    def test_builder_writes_kind_partition(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            grp = scene.add_partition_group(
                "manual",
                display_type="points",
                max_elements=500,
            )
            assert isinstance(grp, Group)
            assert grp.attrs["kind"] == "partition"
            assert grp.attrs["display_type"] == "points"
            assert grp.attrs["max_elements"] == 500

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        attrs = store["manual"].attrs
        assert attrs["type"] == "group"
        assert attrs["kind"] == "partition"

    def test_invalid_display_type_rejected(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="display_type for a partition"):
                scene.add_partition_group(
                    "bad",
                    display_type="meshes",
                    max_elements=10,  # type: ignore[arg-type]
                )

    def test_invalid_max_elements_rejected(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="max_elements must be"):
                scene.add_partition_group("bad", display_type="points", max_elements=0)

    def test_validate_empty_children_raises(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            grp = scene.add_partition_group(
                "empty", display_type="points", max_elements=100
            )
            with pytest.raises(ValueError, match="has no children"):
                validate_partition_group(grp)

    def test_validate_homogeneous_children_passes(self, tmp_path) -> None:
        c, a, ch = TestAddGSplatsPartition._make_gsplats(n=10)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            grp = scene.add_partition_group(
                "homog", display_type="gsplats", max_elements=100
            )
            grp.add_gsplats("part_0", centers=c, amplitudes=a, cholesky_factors=ch)
            grp.add_gsplats("part_1", centers=c, amplitudes=a, cholesky_factors=ch)
            validate_partition_group(grp)  # no raise

    def test_validate_mixed_children_raises(self, tmp_path) -> None:
        """A child whose resolved display_type differs from the parent must fail."""
        c, a, ch = TestAddGSplatsPartition._make_gsplats(n=10)
        pos = np.zeros((5, 3), dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            grp = scene.add_partition_group(
                "bad", display_type="gsplats", max_elements=100
            )
            grp.add_gsplats("part_0", centers=c, amplitudes=a, cholesky_factors=ch)
            grp.add_points("part_1", pos)
            with pytest.raises(ValueError, match="non-homogeneous"):
                validate_partition_group(grp)


# ────────────────────────────────────────────────────────────────────────
# Compiler-level auto-partition heuristic (opt-in)
# ────────────────────────────────────────────────────────────────────────


class TestCompilerAutoPartition:
    """``LuxarZarrCompiler(auto_partition_max_elements=N)`` opt-in heuristic.

    The compiler-level threshold synthesizes a ``partition=dict(max_elements=N)``
    on ``add_points`` / ``add_gsplats`` when the input exceeds N and the
    user did NOT pass ``partition=`` at the call site. Below the threshold or
    with a user-explicit ``partition=``, behavior is unchanged.
    """

    def test_below_threshold_writes_single_leaf(self, tmp_path) -> None:
        pos = np.random.RandomState(0).uniform(-10, 10, (50, 3)).astype(np.float32)
        with LuxarZarrCompiler(
            tmp_path / "t.luxar.zarr", auto_partition_max_elements=100
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos)
        # Single leaf (no wrapper)
        assert node.attrs.get("kind") != "partition"
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        assert store["pts"].attrs["type"] == "points"

    def test_above_threshold_wraps_in_partition(self, tmp_path) -> None:
        pos = np.random.RandomState(1).uniform(-10, 10, (300, 3)).astype(np.float32)
        with LuxarZarrCompiler(
            tmp_path / "t.luxar.zarr", auto_partition_max_elements=100
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos)
        assert isinstance(node, Group)
        assert node.attrs["kind"] == "partition"
        assert node.attrs["display_type"] == "points"
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["pts"]
        total = sum(int(grp[c].attrs["n_points"]) for c in grp.keys())
        assert total == 300

    def test_user_explicit_partition_wins(self, tmp_path) -> None:
        """User-explicit ``partition=`` always wins, even with smaller threshold."""
        pos = np.random.RandomState(2).uniform(-10, 10, (300, 3)).astype(np.float32)
        with LuxarZarrCompiler(
            tmp_path / "t.luxar.zarr", auto_partition_max_elements=50
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            # User says 200 → cap of 200. Compiler threshold (50) is
            # ignored — auto-partition would otherwise produce many more
            # leaves.
            scene.add_points("pts", pos, partition=dict(max_elements=200))
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["pts"]

        # Walk to leaf points nodes (the BSP recurses, so leaves can be
        # nested several levels deep) and check the user-explicit cap.
        def collect_leaves(g, out):
            for child_name in g.keys():
                child = g[child_name]
                t = child.attrs.get("type")
                if t == "points":
                    out.append(int(child.attrs["n_points"]))
                else:
                    collect_leaves(child, out)
            return out

        counts = collect_leaves(grp, [])
        assert len(counts) >= 2
        for n in counts:
            assert n <= 200, (
                f"leaf with {n} points exceeds user cap (200); "
                "auto-partition must not override an explicit user partition="
            )
        # Total reconstructs the input.
        assert sum(counts) == 300
        # And the leaf count is small — well below what cap=50 would
        # have produced (≈6+).
        assert len(counts) <= 4

    def test_auto_partition_applies_to_gsplats(self, tmp_path) -> None:
        c, a, ch = TestAddGSplatsPartition._make_gsplats(n=300)
        with LuxarZarrCompiler(
            tmp_path / "t.luxar.zarr", auto_partition_max_elements=120
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats(
                "splats", centers=c, amplitudes=a, cholesky_factors=ch
            )
        assert isinstance(node, Group)
        assert node.attrs["kind"] == "partition"

    def test_default_none_disables_auto_partition(self, tmp_path) -> None:
        """No threshold = no partition, regardless of size."""
        pos = np.random.RandomState(3).uniform(-10, 10, (5000, 3)).astype(np.float32)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points("pts", pos)
        assert node.attrs.get("kind") != "partition"

    def test_invalid_threshold_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            LuxarZarrCompiler(
                tmp_path / "bad.luxar.zarr", auto_partition_max_elements=0
            )
        with pytest.raises(ValueError, match="must be positive"):
            LuxarZarrCompiler(
                tmp_path / "bad2.luxar.zarr", auto_partition_max_elements=-1
            )
