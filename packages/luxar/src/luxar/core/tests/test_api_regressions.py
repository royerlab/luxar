"""Regression tests pinning down core API invariants.

Each class guards a specific behavior that must hold:
- Points metadata exposes the "ndim" key (never "dims"), matching Lines/GSplats.
- Node property setters (opacity/gamma/blending_mode) persist to the zarr store.
- Nodes from different scenes are never equal, even with identical paths.
- Scene.dimensions returns the live Dimensions object and enforces initialization.
- Scene.to_zarr finalizes and copies the backing store with safe destination checks.
- GSplatData and fit_gaussian_splats are importable from the top-level luxar package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler

# ── Points metadata uses the "ndim" key ─────────────────────────────


class TestPointsMetadataNdimKey:
    """Guard: Points metadata must use 'ndim' (not 'dims') for consistency
    with Lines and GSplats."""

    def test_points_metadata_has_ndim_key(self, tmp_path: Path) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            pts = scene.add_points("pts", np.zeros((5, 3), dtype=np.float32))
            assert "ndim" in pts.metadata
            assert pts.metadata["ndim"] == 3

    def test_points_metadata_no_dims_key(self, tmp_path: Path) -> None:
        """The old 'dims' key must not be present."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            pts = scene.add_points("pts", np.zeros((5, 3), dtype=np.float32))
            assert "dims" not in pts.metadata

    def test_points_ndim_property_works(self, tmp_path: Path) -> None:
        """DataNode.ndim property should work for Points."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            pts = scene.add_points("pts", np.zeros((5, 3), dtype=np.float32))
            assert pts.ndim == 3

    def test_lines_and_gsplats_also_use_ndim(self, tmp_path: Path) -> None:
        """Lines and GSplats should also use 'ndim' key."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            lines = scene.add_lines(
                "lines",
                np.zeros((4, 3), dtype=np.float32),
                widths=0.1,
            )
            assert "ndim" in lines.metadata
            assert lines.ndim == 3

            # 3D packed Cholesky has k=6 elements: [L00, L10, L11, L20, L21, L22]
            chol = np.array([[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32)
            gsplats = scene.add_gsplats(
                "splats",
                np.zeros((2, 3), dtype=np.float32),
                amplitudes=np.ones(2, dtype=np.float32),
                cholesky_factors=chol,
            )
            assert "ndim" in gsplats.metadata
            assert gsplats.ndim == 3


# ── Property setters persist to zarr ─────────────────────────────────


class TestPropertySettersPersistToZarr:
    """Guard: Setting opacity/gamma/blending_mode after creation must
    persist the change to the zarr store on disk."""

    def test_opacity_setter_persists(self, tmp_path: Path) -> None:
        store_path = tmp_path / "test.luxar.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")

            # Set opacity AFTER creation
            group.opacity = 0.42

            # Verify in-memory
            assert group.opacity == pytest.approx(0.42)

        # Verify on disk (after compiler is closed)
        store = zarr.open_group(str(store_path), mode="r")
        assert float(store["grp"].attrs["opacity"]) == pytest.approx(0.42)

    def test_gamma_setter_persists(self, tmp_path: Path) -> None:
        store_path = tmp_path / "test.luxar.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")

            group.gamma = 1.5

            assert group.gamma == pytest.approx(1.5)

        store = zarr.open_group(str(store_path), mode="r")
        assert float(store["grp"].attrs["gamma"]) == pytest.approx(1.5)

    def test_blending_mode_setter_persists(self, tmp_path: Path) -> None:
        store_path = tmp_path / "test.luxar.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("grp")

            group.blending_mode = "max"

            assert group.blending_mode == "max"

        store = zarr.open_group(str(store_path), mode="r")
        assert store["grp"].attrs["blending_mode"] == "max"

    def test_opacity_setter_on_points_node(self, tmp_path: Path) -> None:
        """Property setters should also work on DataNode subclasses."""
        store_path = tmp_path / "test.luxar.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            pts = scene.add_points("pts", np.zeros((3, 3), dtype=np.float32))

            pts.opacity = 0.75

            assert pts.opacity == pytest.approx(0.75)

        store = zarr.open_group(str(store_path), mode="r")
        assert float(store["pts"].attrs["opacity"]) == pytest.approx(0.75)


# ── Cross-scene node inequality ──────────────────────────────────────


class TestCrossSceneNodeInequality:
    """Guard: Nodes from different scenes with the same path must NOT
    be considered equal."""

    def test_same_name_different_scenes_not_equal(self, tmp_path: Path) -> None:
        with LuxarZarrCompiler(tmp_path / "a.luxar.zarr") as c1:
            scene1 = c1.create_scene(dimensions=Dimensions.default_3d())
            g1 = scene1.add_group("foo")

        with LuxarZarrCompiler(tmp_path / "b.luxar.zarr") as c2:
            scene2 = c2.create_scene(dimensions=Dimensions.default_3d())
            g2 = scene2.add_group("foo")

        assert g1 != g2

    def test_same_name_different_scenes_different_hash(self, tmp_path: Path) -> None:
        with LuxarZarrCompiler(tmp_path / "a.luxar.zarr") as c1:
            scene1 = c1.create_scene(dimensions=Dimensions.default_3d())
            g1 = scene1.add_group("foo")

        with LuxarZarrCompiler(tmp_path / "b.luxar.zarr") as c2:
            scene2 = c2.create_scene(dimensions=Dimensions.default_3d())
            g2 = scene2.add_group("foo")

        assert hash(g1) != hash(g2)

    def test_same_scene_same_node_still_equal(self, tmp_path: Path) -> None:
        """Regression: same node in same scene must still be == to itself."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("foo")

            assert group == group

    def test_root_nodes_from_different_scenes_not_equal(self, tmp_path: Path) -> None:
        with LuxarZarrCompiler(tmp_path / "a.luxar.zarr") as c1:
            scene1 = c1.create_scene(dimensions=Dimensions.default_3d())

        with LuxarZarrCompiler(tmp_path / "b.luxar.zarr") as c2:
            scene2 = c2.create_scene(dimensions=Dimensions.default_3d())

        assert scene1 != scene2

    def test_cross_scene_nodes_in_set(self, tmp_path: Path) -> None:
        """Nodes from different scenes should both appear in a set."""
        with LuxarZarrCompiler(tmp_path / "a.luxar.zarr") as c1:
            scene1 = c1.create_scene(dimensions=Dimensions.default_3d())
            g1 = scene1.add_group("foo")

        with LuxarZarrCompiler(tmp_path / "b.luxar.zarr") as c2:
            scene2 = c2.create_scene(dimensions=Dimensions.default_3d())
            g2 = scene2.add_group("foo")

        node_set = {g1, g2}
        assert len(node_set) == 2


# ── Scene.dimensions returns the live Dimensions object ──────────────


class TestSceneDimensionsProperty:
    """Guard: Scene.dimensions must return the Dimensions object directly."""

    def test_dimensions_returns_correct_object(self, tmp_path: Path) -> None:
        dims = Dimensions.default_3d()
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            assert scene.dimensions is dims
            assert scene.dimensions.ndim == 3
            assert scene.dimensions.names == ["x", "y", "z"]

    def test_dimensions_getter_enforces_initialization_invariant(
        self, tmp_path: Path
    ) -> None:
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene._dimensions = None  # type: ignore[assignment]

            with pytest.raises(RuntimeError, match="dimensions are not initialized"):
                _ = scene.dimensions


# ── Scene.to_zarr exports a finalized backing store ──────────────────


class TestSceneToZarrExport:
    """Guard: Scene.to_zarr must be a functional export API, not a stub."""

    def test_to_zarr_finalizes_and_copies_store(self, tmp_path: Path) -> None:
        source = tmp_path / "source.luxar.zarr"
        export = tmp_path / "exported.luxar.zarr"

        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts", np.zeros((3, 3), dtype=np.float32), labels=["a", "b", "c"]
            )

            scene.to_zarr(export)

        assert export.exists()
        store = zarr.open_group(str(export), mode="r")
        assert store.attrs["type"] == "scene"
        assert "scene_dimensions" in store.attrs
        assert "content_hash" in store.attrs
        assert "pts" in store
        assert store["pts/positions"].shape == (3, 3)
        # Finalization should auto-inject hover overlay before copying.
        assert "overlays/__hover_text" in store

    def test_to_zarr_refuses_existing_destination(self, tmp_path: Path) -> None:
        source = tmp_path / "source.luxar.zarr"
        export = tmp_path / "exported.luxar.zarr"
        export.mkdir()

        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            with pytest.raises(FileExistsError, match="Destination already exists"):
                scene.to_zarr(export)

    def test_to_zarr_refuses_destination_inside_source(self, tmp_path: Path) -> None:
        source = tmp_path / "source.luxar.zarr"

        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            with pytest.raises(ValueError, match="cannot be inside source"):
                scene.to_zarr(source / "nested.luxar.zarr")


# ── GSplatData re-exported at the top level ──────────────────────────


class TestGSplatDataTopLevelExport:
    """Guard: GSplatData and fit_gaussian_splats must be accessible
    directly from the luxar package."""

    def test_gsplatdata_importable_from_luxar(self) -> None:
        import luxar

        assert hasattr(luxar, "GSplatData")

    def test_fit_gaussian_splats_importable_from_luxar(self) -> None:
        import luxar

        assert hasattr(luxar, "fit_gaussian_splats")

    def test_gsplatdata_in_all(self) -> None:
        import luxar

        assert "GSplatData" in luxar.__all__
        assert "fit_gaussian_splats" in luxar.__all__
