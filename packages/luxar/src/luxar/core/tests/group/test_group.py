"""Tests for Group class with add_* methods."""

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.gsplats import GSplats
from luxar.core.node import Node
from luxar.io.compiler import LuxarZarrCompiler


class TestGroupAddData:
    """Test that Group nodes can add data children."""

    def test_group_add_points(self, tmp_path) -> None:
        """Test group.add_points() writes data under group path."""
        output_path = tmp_path / "test.luxar.zarr"
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            pts = group.add_points("pts", positions)

            assert pts.n_elements == 2
            assert pts.parent == group

        # Verify zarr hierarchy AND that the positions round-tripped to disk
        # (B11/[P2]: type+count alone don't prove the data was written).
        store = zarr.open(str(output_path), mode="r")
        assert "grp" in store
        assert "pts" in store["grp"]
        node = store["grp"]["pts"]
        assert node.attrs.get("type") == "points"
        assert node.attrs["n_points"] == 2
        stored = node["positions"][:]
        assert stored.shape == (2, 3)
        np.testing.assert_allclose(stored, positions)

    def test_group_add_lines(self, tmp_path) -> None:
        """Test group.add_lines() writes data under group path."""
        output_path = tmp_path / "test.luxar.zarr"
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")
            lines = group.add_lines("lines", vertices, widths=0.1)

            assert lines.n_elements == 2

        store = zarr.open(str(output_path), mode="r")
        assert "lines" in store["grp"]
        # B11/[P2]: verify the vertices actually round-tripped, not just presence.
        node = store["grp"]["lines"]
        assert node.attrs.get("type") == "lines"
        assert node.attrs["n_vertices"] == 2
        verts = node["vertices"][:]
        assert verts.shape == (2, 3)
        np.testing.assert_allclose(verts, vertices)

    def test_group_add_gsplats(self, tmp_path) -> None:
        """Test group.add_gsplats() writes data under group path."""
        output_path = tmp_path / "test.luxar.zarr"
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
        node = store["grp"]["splats"]
        assert node.attrs.get("type") == "gsplats"
        # B11/[P2]: verify centers round-tripped and the per-splat arrays are
        # present with the right count (was: only type + n_splats).
        assert node.attrs["n_splats"] == 1
        ctrs = node["centers"][:]
        assert ctrs.shape == (1, 3)
        np.testing.assert_allclose(ctrs, centers)
        assert node["amplitudes"].shape[0] == 1
        # v3.1: Cholesky stored as a diagonal + off-diagonal split.
        assert "cholesky_factors_diag" in node
        assert "cholesky_factors_offdiag" in node

    def test_nested_groups(self, tmp_path) -> None:
        """Test nested groups: group.add_group('sub').add_points(...)."""
        output_path = tmp_path / "test.luxar.zarr"
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
        output_path = tmp_path / "test.luxar.zarr"
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
        output_path = tmp_path / "test.luxar.zarr"
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
        output_path = tmp_path / "test.luxar.zarr"
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
        output_path = tmp_path / "test.luxar.zarr"
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
        output_path = tmp_path / "test.luxar.zarr"
        data = self._make_multi_lod_gsplat_data()

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data("splats", data)

            # Single-substitutive path returns GSplats, not LODGroup.
            assert isinstance(node, GSplats)
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
        # Per-additive arrays (v3.1 splits Cholesky into diag + offdiag)
        assert "centers" in grp["additive_0"]
        assert "amplitudes" in grp["additive_0"]
        assert "cholesky_factors_diag" in grp["additive_0"]

    def test_multi_lod_preserves_per_lod_stats(self, tmp_path) -> None:
        """Per-additive-sub-LOD stats land on each ``additive_<i>`` group."""
        output_path = tmp_path / "test.luxar.zarr"
        data = self._make_multi_lod_gsplat_data()

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data)

        store = zarr.open(str(output_path), mode="r")
        grp = store["splats"]
        # B11/[P2]: assert the stats dict + key are actually present, so a
        # missing lod_stats can't pass via ``.get(...)`` returning a default.
        lod0_stats = grp["additive_0"].attrs["lod_stats"]
        assert "pass_index" in lod0_stats
        assert lod0_stats["pass_index"] == 0
        lod1_stats = grp["additive_1"].attrs["lod_stats"]
        assert "pass_index" in lod1_stats
        assert lod1_stats["pass_index"] == 1

    def test_single_lod_uses_flat_layout(self, tmp_path) -> None:
        """Single-LOD GSplatData uses flat layout (no additive subgroups)."""
        from luxar.gsplats.gsplat_data import GSplatData

        output_path = tmp_path / "test.luxar.zarr"
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
        output_path = tmp_path / "test.luxar.zarr"
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
        output_path = tmp_path / "test.luxar.zarr"
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

        output_path = tmp_path / "test.luxar.zarr"
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

    def test_multi_lod_auto_lowers_substitutive_levels(self, tmp_path) -> None:
        """Substitutive levels in input are auto-lowered to a kind=lod Group.

        The convention (documented on ``add_gsplats_from_data``): if the
        input GSplatData carries multiple substitutive levels and the
        caller does not specify ``lod_group``, the default now auto-lowers
        the pyramid into a ``kind=lod`` Group with one child per level
        (no expensive substitutive work discarded). ``lod_group=False``
        collapses to the finest level instead.
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
        data = GSplatData.from_substitutive_levels([finest, coarsest])
        assert data.n_substitutive == 2  # sanity

        output_path = tmp_path / "test.luxar.zarr"
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data)

        store = zarr.open(str(output_path), mode="r")
        grp = store["splats"]
        # Auto-lowered to a kind=lod Group with one child per substitutive level.
        assert grp.attrs["type"] == "group"
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "gsplats"
        # Coarsest first: child_0 = 1 splat (coarsest), child_1 = 5 splats (finest).
        assert grp["child_0"].attrs["n_splats"] == 1
        assert grp["child_1"].attrs["n_splats"] == 5


# ────────────────────────────────────────────────────────────────────────
# Convenience API — lod_group= / additive_lod= resolution table
# ────────────────────────────────────────────────────────────────────────


def _make_flat_gsplat_data(n: int = 8, seed: int = 0):
    """Build a single-substitutive, single-additive GSplatData with N splats."""
    from luxar.gsplats.gsplat_data import GSplatData

    rng = np.random.RandomState(seed)
    return GSplatData(
        centers=rng.rand(n, 3).astype(np.float32) * 100,
        amplitudes=rng.rand(n).astype(np.float32).clip(0.1, 1.0),
        cholesky_factors=np.tile(
            np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
        ),
    )


def _make_multi_substitutive_gsplat_data():
    """Build a 2-level substitutive (no additive ladders) GSplatData.

    Index 0 = finest (8 splats), index 1 = coarsest (2 splats), matching the
    ``make_substitutive_lod`` convention.
    """
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )

    rng = np.random.RandomState(1)
    finest = SubstitutiveLevel(
        additive_sublods=[
            AdditiveSubLOD(
                centers=rng.rand(8, 3).astype(np.float32),
                amplitudes=np.ones(8, dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (8, 1)
                ),
            )
        ],
        compression_factor=1,
        level_index=0,
    )
    coarsest = SubstitutiveLevel(
        additive_sublods=[
            AdditiveSubLOD(
                centers=rng.rand(2, 3).astype(np.float32),
                amplitudes=np.ones(2, dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
                ),
            )
        ],
        compression_factor=4,
        level_index=1,
    )
    return GSplatData.from_substitutive_levels([finest, coarsest])


class TestLodGroupAxis:
    """``lod_group=`` semantics on ``add_gsplats_from_data``."""

    def test_none_auto_lowers_to_lod_group(self, tmp_path) -> None:
        """``None`` (default) auto-lowers a multi-substitutive pyramid to a
        ``kind=lod`` Group instead of silently dropping levels."""
        data = _make_multi_substitutive_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data("splats", data)  # lod_group=None
        # Auto-lowered to a kind=lod Group with one child per substitutive level.
        assert isinstance(node, Group)
        assert node.attrs.get("kind") == "lod"
        assert len(node.children) == 2
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        assert store["splats"].attrs["type"] == "group"
        assert store["splats"].attrs["kind"] == "lod"

    def test_none_passthrough_single_substitutive(self, tmp_path) -> None:
        """``None`` on single-substitutive data stays a flat GSplats node."""
        data = _make_flat_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data("splats", data)  # lod_group=None
        assert type(node).__name__ == "GSplats"
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        assert store["splats"].attrs["type"] == "gsplats"

    def test_true_requires_stored_levels(self, tmp_path) -> None:
        """``True`` raises when input is single-substitutive."""
        data = _make_flat_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="lod_group=True requires"):
                scene.add_gsplats_from_data("splats", data, lod_group=True)

    def test_true_uses_stored_levels(self, tmp_path) -> None:
        """``True`` with stored levels produces a kind=lod Group."""
        data = _make_multi_substitutive_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data("multires", data, lod_group=True)
            assert isinstance(node, Group)
            assert node.attrs.get("kind") == "lod"
            assert len(node.children) == 2

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["multires"]
        assert grp.attrs["type"] == "group"
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "gsplats"
        # Coarsest first → child_0 has fewer splats than child_1.
        assert grp["child_0"].attrs["n_splats"] == 2
        assert grp["child_1"].attrs["n_splats"] == 8
        # Auto-derived coverage fractions (sqrt(N_i/N_finest)): coarsest = 0.0,
        # finest = 1.0.
        assert grp["child_0"].attrs["coverage_fraction"] == 0.0
        assert grp["child_1"].attrs["coverage_fraction"] == pytest.approx(1.0)

    def test_false_collapses_to_finest(self, tmp_path) -> None:
        """``False`` keeps only the finest substitutive level (index 0)."""
        data = _make_multi_substitutive_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data("splats", data, lod_group=False)
        assert type(node).__name__ == "GSplats"
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        # Finest had 8 splats; coarsest had 2. We kept the finest.
        assert store["splats"].attrs["n_splats"] == 8

    def test_dict_computes_from_flat(self, tmp_path) -> None:
        """``dict(...)`` computes substitutive levels when input is flat."""
        data = _make_flat_gsplat_data(n=8)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data(
                "multires",
                data,
                lod_group=dict(compression_factor=2, levels=1),
            )
            assert isinstance(node, Group)
            assert node.attrs.get("kind") == "lod"
            # K=2, L=1 → 2 substitutive levels total
            assert len(node.children) == 2

    def test_dict_with_stored_and_compute_kwargs_raises(self, tmp_path) -> None:
        """``dict(...)`` carrying compute kwargs against stored levels raises.

        Silently dropping ``compression_factor`` / ``levels`` when stored
        substitutive levels are present is a footgun — the user's intent is
        ambiguous. Require an explicit ``recompute=True`` to discard the
        stored pyramid, or drop the compute kwargs to reuse it.
        """
        data = _make_multi_substitutive_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="compute kwargs"):
                scene.add_gsplats_from_data(
                    "multires",
                    data,
                    lod_group=dict(compression_factor=4, levels=3),
                )

    def test_dict_empty_with_stored_reuses_stored(self, tmp_path) -> None:
        """``dict()`` with no compute kwargs and stored levels reuses stored.

        Companion to ``test_dict_with_stored_and_compute_kwargs_raises``:
        a bare ``dict()`` (or one carrying only ``coverage_fractions``) is
        unambiguous — reuse the stored pyramid.
        """
        data = _make_multi_substitutive_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data(
                "multires",
                data,
                lod_group=dict(),
            )
            assert isinstance(node, Group)
            assert node.attrs.get("kind") == "lod"
            assert len(node.children) == 2

    def test_dict_recompute_forces_recomputation(self, tmp_path) -> None:
        """``dict(..., recompute=True)`` recomputes even when stored levels exist."""
        # Use FLAT data (no stored substitutive levels) so we can verify
        # recompute=True still works on flat input identically to the
        # compute-if-absent path. The recompute flag's main job is to
        # NOT short-circuit when levels are present; on flat input,
        # behaviour is the same.
        data = _make_flat_gsplat_data(n=16)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data(
                "multires",
                data,
                lod_group=dict(compression_factor=2, levels=2, recompute=True),
            )
            # K=2, L=2 → 3 substitutive levels total
            assert len(node.children) == 3

    def test_coverage_fractions_override(self, tmp_path) -> None:
        """Explicit ``coverage_fractions`` overrides the auto-derived defaults."""
        data = _make_multi_substitutive_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data(
                "multires",
                data,
                lod_group=dict(coverage_fractions=[0.0, 0.5]),
            )
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["multires"]
        assert grp["child_0"].attrs["coverage_fraction"] == 0.0
        assert grp["child_1"].attrs["coverage_fraction"] == 0.5

    def test_coverage_fractions_wrong_length_raises(self, tmp_path) -> None:
        """Length mismatch between coverage_fractions and # of substitutive levels."""
        data = _make_multi_substitutive_gsplat_data()  # 2 levels
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="coverage_fractions has"):
                scene.add_gsplats_from_data(
                    "splats",
                    data,
                    lod_group=dict(coverage_fractions=[0.0, 0.5, 1.0]),  # 3 entries
                )

    def test_coverage_fractions_not_strictly_ascending_raises(self, tmp_path) -> None:
        """Explicit ``coverage_fractions`` must be strictly increasing coarsest→finest.

        The resolver checks this directly so callers fail at the
        ``add_gsplats_from_data`` call site, not at a later
        ``LODGroup.validate()`` invocation the user may never make.
        """
        data = _make_multi_substitutive_gsplat_data()  # 2 levels
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="strictly increasing"):
                scene.add_gsplats_from_data(
                    "splats",
                    data,
                    lod_group=dict(coverage_fractions=[0.5, 0.1]),  # decreasing
                )

    def test_coverage_fraction_attr_rejected_on_multi_substitutive(
        self, tmp_path
    ) -> None:
        """Passing ``coverage_fraction=`` to multi-substitutive path raises.

        Multi-substitutive paths derive ``coverage_fraction`` per child. An
        explicit value here is ambiguous — and worse, used to provoke
        ``TypeError: multiple values for keyword argument 'coverage_fraction'``
        inside the LODGroup expansion.
        """
        data = _make_multi_substitutive_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="coverage_fraction"):
                scene.add_gsplats_from_data(
                    "multires",
                    data,
                    lod_group=True,
                    coverage_fraction=0.5,
                )

    def test_unknown_spec_type_raises(self, tmp_path) -> None:
        """Non-None/bool/dict spec is a TypeError."""
        data = _make_flat_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(TypeError, match="lod_group must be"):
                scene.add_gsplats_from_data("splats", data, lod_group="auto")  # type: ignore[arg-type]


class TestAdditiveLodAxis:
    """``additive_lod=`` semantics on ``add_gsplats_from_data``."""

    @staticmethod
    def _make_multi_additive_data(n: int = 8, n_lods: int = 3):
        """Flat data with an additive ladder of n_lods levels (cumulative)."""
        from luxar.gsplats.lod.additive import make_additive_lod

        return make_additive_lod(_make_flat_gsplat_data(n=n), n_lods=n_lods)

    def test_none_passes_through(self, tmp_path) -> None:
        """``None`` keeps the additive ladder as-is."""
        data = self._make_multi_additive_data(n=8, n_lods=3)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data)  # additive_lod=None
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["splats"]
        # Existing multi-additive path → additive_<i>/ subgroups.
        assert grp.attrs.get("n_additive_sublods") == 3
        assert "additive_0" in grp

    def test_true_requires_stored_ladder(self, tmp_path) -> None:
        """``True`` raises when input has only a single additive level."""
        data = _make_flat_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="additive_lod=True requires"):
                scene.add_gsplats_from_data("splats", data, additive_lod=True)

    def test_true_uses_stored_ladder(self, tmp_path) -> None:
        """``True`` with a stored ladder writes the additive subgroups."""
        data = self._make_multi_additive_data(n=8, n_lods=3)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data, additive_lod=True)
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        assert store["splats"].attrs["n_additive_sublods"] == 3

    def test_false_flattens_ladder(self, tmp_path) -> None:
        """``False`` collapses the additive ladder to a single LOD."""
        data = self._make_multi_additive_data(n=8, n_lods=3)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data, additive_lod=False)
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["splats"]
        # No additive subgroups; flat layout, all 8 splats at the top level.
        assert "additive_0" not in grp
        assert "n_additive_sublods" not in grp.attrs
        assert grp.attrs["n_splats"] == 8

    def test_dict_computes_ladder_from_flat(self, tmp_path) -> None:
        """``dict(n_lods=2)`` computes a 2-level ladder from flat input."""
        data = _make_flat_gsplat_data(n=8)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("splats", data, additive_lod=dict(n_lods=2))
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        assert store["splats"].attrs["n_additive_sublods"] == 2


class TestCombinedAxes:
    """Both ``lod_group=`` and ``additive_lod=`` together."""

    def test_compute_substitutive_then_additive(self, tmp_path) -> None:
        """Compute substitutive levels then add an additive ladder on each."""
        data = _make_flat_gsplat_data(n=16)
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_gsplats_from_data(
                "multires",
                data,
                lod_group=dict(compression_factor=2, levels=1),
                additive_lod=dict(n_lods=2),
            )
            assert isinstance(node, Group)
            assert node.attrs.get("kind") == "lod"
            assert len(node.children) == 2  # K=2, L=1

        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["multires"]
        # Each substitutive child should have a 2-step additive ladder.
        for child_name in ["child_0", "child_1"]:
            child = grp[child_name]
            assert child.attrs.get("n_additive_sublods") == 2
            assert "additive_0" in child
            assert "additive_1" in child
            assert "coverage_fraction" in child.attrs

    def test_compositing_attrs_land_on_lod_group(self, tmp_path) -> None:
        """opacity/gamma/etc. ride onto the kind=lod Group, not the children."""
        data = _make_multi_substitutive_gsplat_data()
        with LuxarZarrCompiler(tmp_path / "t.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data(
                "multires", data, lod_group=True, opacity=0.5, gamma=2.0
            )
        store = zarr.open(str(tmp_path / "t.luxar.zarr"), mode="r")
        grp = store["multires"]
        # The kind=lod Group carries the compositing attrs.
        assert grp.attrs["opacity"] == 0.5
        assert grp.attrs["gamma"] == 2.0
        # Children do NOT (they inherit via composition at render time).
        # Note: the writer auto-defaults missing values, so we can't easily
        # assert "absence" on the children — but the value on the parent
        # is the authoritative source.
