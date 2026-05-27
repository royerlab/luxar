"""Tests for Group class with add_* methods."""

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.node import Node
from luxar.io.compiler import LuxarZarrCompiler


class TestGroupAddData:
    """Test that Group nodes can add data children."""

    def test_group_add_points(self, tmp_path) -> None:
        """Test group.add_points() writes data under group path."""
        output_path = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            pts = group.add_points("pts", positions)

            assert pts.n_elements == 2
            assert pts.parent == group

        # Verify zarr hierarchy
        store = zarr.open(str(output_path), mode="r")
        assert "grp" in store
        assert "pts" in store["grp"]
        assert store["grp"]["pts"].attrs.get("type") == "points"

    def test_group_add_lines(self, tmp_path) -> None:
        """Test group.add_lines() writes data under group path."""
        output_path = tmp_path / "test.zarr"
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            lines = group.add_lines("lines", vertices, widths=0.1)

            assert lines.n_elements == 2

        store = zarr.open(str(output_path), mode="r")
        assert "lines" in store["grp"]

    def test_group_add_gsplats(self, tmp_path) -> None:
        """Test group.add_gsplats() writes data under group path."""
        output_path = tmp_path / "test.zarr"
        centers = np.array([[1, 2, 3]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        cholesky = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            gsplats = group.add_gsplats("splats", centers, amplitudes, cholesky)

            assert gsplats.n_splats == 1

        store = zarr.open(str(output_path), mode="r")
        assert "splats" in store["grp"]
        assert store["grp"]["splats"].attrs.get("type") == "gsplats"

    def test_nested_groups(self, tmp_path) -> None:
        """Test nested groups: group.add_group('sub').add_points(...)."""
        output_path = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("level1")
            sub = group.add_group("level2")
            sub.add_points("pts", positions)

        store = zarr.open(str(output_path), mode="r")
        assert "pts" in store["level1"]["level2"]

    def test_dimension_validation_through_group(self, tmp_path) -> None:
        """Test that dimension validation works when adding data to a group."""
        output_path = tmp_path / "test.zarr"
        positions_2d = np.array([[1, 2]], dtype=np.float32)  # 2D, but scene is 3D

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")

            with pytest.raises(ValueError, match="Dimension mismatch"):
                group.add_points("bad", positions_2d)


class TestGroupNotAttachedToScene:
    """Test error when Group is not part of a Scene hierarchy."""

    def test_detached_group_raises(self) -> None:
        """Group not attached to Scene should raise on add_*."""
        group = Group("orphan")
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with pytest.raises(ValueError, match="not attached to a Scene"):
            group.add_points("pts", positions)


class TestParentArgument:
    """Test that Scene add_* methods accept parent groups."""

    def test_scene_add_points_with_parent(self, tmp_path) -> None:
        """scene.add_points('name', data, parent=group)."""
        output_path = tmp_path / "test.zarr"
        positions = np.array([[1, 2, 3]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            pts = scene.add_points("pts", positions, parent=group)

            assert pts.parent == group

        store = zarr.open(str(output_path), mode="r")
        assert "pts" in store["grp"]

    def test_scene_add_gsplats_with_parent(self, tmp_path) -> None:
        """scene.add_gsplats('name', ..., parent=group)."""
        output_path = tmp_path / "test.zarr"
        centers = np.array([[1, 2, 3]], dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        cholesky = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            gsplats = scene.add_gsplats(
                "splats", centers, amplitudes, cholesky, parent=group
            )

            assert gsplats.parent == group


class TestGroupIsGroup:
    """Test that add_group() returns Group instances."""

    def test_add_group_returns_group(self, tmp_path) -> None:
        output_path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")

            assert isinstance(group, Group)
            assert isinstance(group, Node)  # Group is also a Node
            assert hasattr(group, "add_points")
            assert hasattr(group, "add_lines")
            assert hasattr(group, "add_gsplats")
            assert hasattr(group, "add_gsplats_from_data")


class TestMultiLODGSplats:
    """Test multi-LOD GSplatData writing through add_gsplats_from_data."""

    @staticmethod
    def _make_multi_lod_gsplat_data():
        """Create a multi-LOD GSplatData with 2 LODs."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        rng = np.random.RandomState(42)
        lod0 = AdditiveSubLOD(
            centers=rng.rand(5, 3).astype(np.float32) * 100,
            amplitudes=rng.rand(5).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (5, 1)
            ),
            stats={"pass_index": 0, "cumulative_psnr_db": 25.0},
        )
        lod1 = AdditiveSubLOD(
            centers=rng.rand(3, 3).astype(np.float32) * 100,
            amplitudes=rng.rand(3).astype(np.float32),
            cholesky_factors=np.tile(
                np.array([0.5, 0, 0.5, 0, 0, 0.5], dtype=np.float32), (3, 1)
            ),
            stats={"pass_index": 1, "cumulative_psnr_db": 30.0},
        )
        return GSplatData.from_additive_sublods([lod0, lod1])

    def test_multi_lod_writes_subgroups(self, tmp_path) -> None:
        """Multi-additive-LOD GSplatData writes ``additive_<i>`` cells flat under the node."""
        output_path = tmp_path / "test.zarr"
        data = self._make_multi_lod_gsplat_data()

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data("splats", data)

            assert node.n_splats == 8  # 5 + 3

        store = zarr.open(str(output_path), mode="r")
        grp = store["splats"]
        assert grp.attrs["type"] == "gsplats"
        assert grp.attrs["n_additive_sublods"] == 2
        assert grp.attrs["n_splats"] == 8

        # Flat layout: additive_<i> directly under the gsplats node.
        # No substitutive wrapper.
        assert "additive_0" in grp
        assert "additive_1" in grp
        assert "substitutive_0" not in grp
        assert grp["additive_0"].attrs["n_splats"] == 5
        assert grp["additive_1"].attrs["n_splats"] == 3
        # Per-additive arrays
        assert "centers" in grp["additive_0"]
        assert "amplitudes" in grp["additive_0"]
        assert "cholesky_factors" in grp["additive_0"]

    def test_multi_lod_preserves_per_lod_stats(self, tmp_path) -> None:
        """Per-additive-sub-LOD stats land on each ``additive_<i>`` group."""
        output_path = tmp_path / "test.zarr"
        data = self._make_multi_lod_gsplat_data()

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data)

        store = zarr.open(str(output_path), mode="r")
        grp = store["splats"]
        lod0_stats = grp["additive_0"].attrs.get("lod_stats", {})
        assert lod0_stats.get("pass_index") == 0
        lod1_stats = grp["additive_1"].attrs.get("lod_stats", {})
        assert lod1_stats.get("pass_index") == 1

    def test_single_lod_uses_flat_layout(self, tmp_path) -> None:
        """Single-LOD GSplatData uses flat layout (no additive subgroups)."""
        from luxar.gsplats.gsplat_data import GSplatData

        output_path = tmp_path / "test.zarr"
        data = GSplatData(
            centers=np.array([[1, 2, 3]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data)

        store = zarr.open(str(output_path), mode="r")
        grp = store["splats"]
        assert grp.attrs["type"] == "gsplats"
        # No LOD subgroups — flat layout
        assert "additive_0" not in grp
        assert "n_additive_sublods" not in grp.attrs
        # Arrays at top level
        assert "centers" in grp
        assert "amplitudes" in grp

    def test_multi_lod_in_group(self, tmp_path) -> None:
        """Multi-additive-LOD gsplats can be added under a group node."""
        output_path = tmp_path / "test.zarr"
        data = self._make_multi_lod_gsplat_data()

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            group.add_gsplats_from_data("splats", data)

        store = zarr.open(str(output_path), mode="r")
        grp = store["grp"]["splats"]
        assert "additive_0" in grp
        assert "additive_1" in grp

    @pytest.mark.parametrize(
        "removed_attr",
        [
            "n_substitutive",
            "default_substitutive",
            "n_additive_sublods_default",
            "n_splats_total",
            "format_version",
        ],
    )
    def test_multi_lod_drops_legacy_substitutive_attrs(
        self, tmp_path, removed_attr: str
    ) -> None:
        """No vestigial substitutive metadata leaks onto the multi-LOD scene gsplats node.

        These attrs belong to the standalone ``.gsplats.zarr`` processing
        format only. Scene zarrs (post-PR-1) must not carry them.
        """
        output_path = tmp_path / "test.zarr"
        data = self._make_multi_lod_gsplat_data()

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data)

        store = zarr.open(str(output_path), mode="r")
        grp = store["splats"]
        assert removed_attr not in grp.attrs, (
            f"Unexpected {removed_attr!r} on multi-LOD scene gsplats node — "
            f"this attribute belongs to the standalone .gsplats.zarr format only."
        )

    @pytest.mark.parametrize(
        "removed_attr",
        [
            "n_substitutive",
            "default_substitutive",
            "n_additive_sublods_default",
            "n_additive_sublods",
            "n_splats_total",
            "format_version",
        ],
    )
    def test_single_lod_drops_legacy_substitutive_attrs(
        self, tmp_path, removed_attr: str
    ) -> None:
        """Single-LOD gsplats nodes carry no LOD-machinery metadata at all."""
        from luxar.gsplats.gsplat_data import GSplatData

        output_path = tmp_path / "test.zarr"
        data = GSplatData(
            centers=np.array([[1, 2, 3]], dtype=np.float32),
            amplitudes=np.array([1.0], dtype=np.float32),
            cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data)

        store = zarr.open(str(output_path), mode="r")
        grp = store["splats"]
        assert removed_attr not in grp.attrs

    def test_multi_lod_drops_substitutive_levels_silently(self, tmp_path) -> None:
        """Substitutive levels in input are dropped; only the default level is written.

        The convention (documented on ``add_gsplats_from_data``): if the
        input GSplatData carries multiple substitutive levels, only the
        default substitutive level's additive ladder is written into
        the scene zarr. Other substitutive levels are silently dropped.
        Substitutive alternatives belong to the standalone processing
        format; expressing them in a scene is a separate concern.
        """
        from luxar.gsplats.gsplat_data import (
            AdditiveSubLOD,
            GSplatData,
            SubstitutiveLevel,
        )

        # Two substitutive levels: finest (default, 5 splats), coarsest (1 splat).
        finest = SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=np.array(
                        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                        dtype=np.float32,
                    ),
                    amplitudes=np.full(5, 1.0, dtype=np.float32),
                    cholesky_factors=np.tile(
                        np.array([0.5, 0, 0.5, 0, 0, 0.5], dtype=np.float32), (5, 1)
                    ),
                )
            ],
            compression_factor=1,
            level_index=0,
        )
        coarsest = SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
                    amplitudes=np.array([5.0], dtype=np.float32),
                    cholesky_factors=np.array(
                        [[1.0, 0, 1.0, 0, 0, 1.0]], dtype=np.float32
                    ),
                )
            ],
            compression_factor=5,
            level_index=1,
        )
        data = GSplatData.from_substitutive_levels(
            [finest, coarsest], default_substitutive=0
        )
        assert data.n_substitutive == 2  # sanity

        output_path = tmp_path / "test.zarr"
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data)

        store = zarr.open(str(output_path), mode="r")
        grp = store["splats"]
        # Only the finest (default) substitutive level survived; coarse splat is gone.
        assert grp.attrs["n_splats"] == 5
        # No substitutive wrapper or attrs.
        assert "substitutive_0" not in grp
        assert "substitutive_1" not in grp
        assert "n_substitutive" not in grp.attrs
