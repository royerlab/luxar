"""Tests for colormap support in the compiler."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


class TestColormapPointsCompiler:
    """Test colormap support when writing points."""

    def test_points_with_colormap_string(self) -> None:
        """Points with a named colormap and scalars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(100, 3).astype(np.float32)
                scalars = np.random.rand(100).astype(np.float32)
                scene.add_points(
                    "pts",
                    positions,
                    scalars=scalars,
                    colormap="viridis",
                    layer=True,
                )

            # Verify zarr contents
            store = zarr.open(str(path), mode="r")
            assert store["pts"].attrs["colormap"] == "viridis"
            assert "scalar_data_range" in store["pts"].attrs
            assert "scalars" in store["pts"]
            # Critical: has_scalars must be in zarr attrs for viewer to enable colormap
            assert store["pts"].attrs.get("has_scalars") is True

    def test_points_colormap_without_scalars(self) -> None:
        """Points with colormap but no scalars — valid (uses default uniform)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                scene.add_points("pts", positions, colormap="green")

            store = zarr.open(str(path), mode="r")
            assert store["pts"].attrs["colormap"] == "green"

    def test_points_colors_and_colormap_raises(self) -> None:
        """Cannot set both colors and colormap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                colors = np.random.rand(50, 3).astype(np.float32)
                with pytest.raises(ValueError, match="colors.*colormap"):
                    scene.add_points("pts", positions, colors=colors, colormap="green")

    def test_points_scalars_without_colormap_raises(self) -> None:
        """Scalars require a colormap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                scalars = np.random.rand(50).astype(np.float32)
                with pytest.raises(ValueError, match="scalars.*colormap"):
                    scene.add_points("pts", positions, scalars=scalars)

    def test_points_custom_colormap_array(self) -> None:
        """Custom colormap as numpy array is stored as LUT dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            custom_lut = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                scene.add_points("pts", positions, colormap=custom_lut)

            store = zarr.open(str(path), mode="r")
            assert store["pts"].attrs["colormap"] == "custom"
            assert "colormap_lut" in store["pts"]
            lut = np.array(store["pts"]["colormap_lut"])
            assert lut.shape == (256, 3)
            np.testing.assert_array_equal(lut, custom_lut)

    def test_points_matplotlib_colormap_stored_as_custom(self) -> None:
        """Non-built-in colormap names (e.g. matplotlib) are resolved to LUT."""
        pytest.importorskip("matplotlib")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                pts = scene.add_points("pts", positions, colormap="cividis")
                # Node should see "custom" (not "cividis")
                assert pts.colormap == "custom"

            # Zarr should have "custom" attr and colormap_lut dataset
            store = zarr.open(str(path), mode="r")
            assert store["pts"].attrs["colormap"] == "custom"
            assert "colormap_lut" in store["pts"]
            lut = np.array(store["pts"]["colormap_lut"])
            assert lut.shape == (256, 3)

    def test_points_builtin_colormap_stored_as_name(self) -> None:
        """Built-in colormaps are stored by name (no LUT dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                pts = scene.add_points("pts", positions, colormap="viridis")
                # Node should see "viridis" (built-in, no resolution needed)
                assert pts.colormap == "viridis"

            store = zarr.open(str(path), mode="r")
            assert store["pts"].attrs["colormap"] == "viridis"
            assert "colormap_lut" not in store["pts"]


class TestColormapToneMappingWarning:
    """The compiler warns about ACES hue distortion when a colormap LUT is used."""

    def test_warns_under_default_aces(self) -> None:
        """A colormap with the default tone-mapping (None → ACES) warns once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with pytest.warns(UserWarning, match="ACES"):
                with LuxarZarrCompiler(path) as c:
                    scene = c.create_scene(dimensions=dims)
                    positions = np.random.rand(50, 3).astype(np.float32)
                    scene.add_points("pts", positions, colormap="viridis")

    def test_warns_only_once_for_multiple_nodes(self) -> None:
        """The warning fires at most once per compile session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with pytest.warns(UserWarning) as record:
                with LuxarZarrCompiler(path) as c:
                    scene = c.create_scene(dimensions=dims)
                    positions = np.random.rand(50, 3).astype(np.float32)
                    scene.add_points("pts1", positions, colormap="viridis")
                    scene.add_points("pts2", positions, colormap="magma")
            aces_warnings = [w for w in record if "ACES" in str(w.message)]
            assert len(aces_warnings) == 1

    def test_no_warning_when_neutral_selected(self) -> None:
        """When the author pins Neutral tone-mapping, no warning is emitted."""
        import warnings as _warnings

        from luxar.core.viewer_config import ViewerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            vc = ViewerConfig(tone_mapping="Neutral")
            with _warnings.catch_warnings(record=True) as record:
                _warnings.simplefilter("always")
                with LuxarZarrCompiler(path) as c:
                    scene = c.create_scene(dimensions=dims, viewer_config=vc)
                    positions = np.random.rand(50, 3).astype(np.float32)
                    scene.add_points("pts", positions, colormap="viridis")
            assert not [w for w in record if "ACES" in str(w.message)]

    @pytest.mark.parametrize(
        "tone_mapping", ["ACES", "AgX", "Reinhard", "Linear", "None"]
    )
    def test_no_warning_when_tone_mapping_chosen_explicitly(
        self, tone_mapping: str
    ) -> None:
        """ANY explicit tone-mapping silences the warning, including 'ACES'.

        The warning exists to catch authors who never considered the choice, and
        its text speaks of "the viewer's default" — which only applies when
        nothing was set. Second-guessing a deliberate selection (ACES is the
        recommended default for most scenes) is just noise on every build.
        """
        import warnings as _warnings

        from luxar.core.viewer_config import ViewerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            vc = ViewerConfig(tone_mapping=tone_mapping)
            with _warnings.catch_warnings(record=True) as record:
                _warnings.simplefilter("always")
                with LuxarZarrCompiler(path) as c:
                    scene = c.create_scene(dimensions=dims, viewer_config=vc)
                    positions = np.random.rand(50, 3).astype(np.float32)
                    scene.add_points("pts", positions, colormap="viridis")
            assert not [w for w in record if "ACES" in str(w.message)]


class TestColormapGSplatsCompiler:
    """Test colormap support when writing gsplats."""

    def test_gsplats_default_gray_colormap(self) -> None:
        """GSplats without colors or colormap get default gray."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                centers = np.random.rand(50, 3).astype(np.float32)
                amplitudes = np.random.rand(50).astype(np.float32)
                cholesky = np.eye(3)[np.newaxis, :, :].repeat(50, axis=0)
                # Extract lower triangular: L00, L10, L11, L20, L21, L22
                chol_packed = np.column_stack(
                    [
                        cholesky[:, 0, 0],
                        cholesky[:, 1, 0],
                        cholesky[:, 1, 1],
                        cholesky[:, 2, 0],
                        cholesky[:, 2, 1],
                        cholesky[:, 2, 2],
                    ]
                ).astype(np.float32)
                scene.add_gsplats("gs", centers, amplitudes, chol_packed)

            store = zarr.open(str(path), mode="r")
            assert store["gs"].attrs["colormap"] == "gray"

    def test_gsplats_with_explicit_colormap(self) -> None:
        """GSplats with explicit colormap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                centers = np.random.rand(50, 3).astype(np.float32)
                amplitudes = np.random.rand(50).astype(np.float32)
                cholesky = np.eye(3)[np.newaxis, :, :].repeat(50, axis=0)
                chol_packed = np.column_stack(
                    [
                        cholesky[:, 0, 0],
                        cholesky[:, 1, 0],
                        cholesky[:, 1, 1],
                        cholesky[:, 2, 0],
                        cholesky[:, 2, 1],
                        cholesky[:, 2, 2],
                    ]
                ).astype(np.float32)
                scene.add_gsplats(
                    "gs", centers, amplitudes, chol_packed, colormap="magenta"
                )

            store = zarr.open(str(path), mode="r")
            assert store["gs"].attrs["colormap"] == "magenta"

    def test_gsplats_colors_and_colormap_raises(self) -> None:
        """Cannot set both colors and colormap on gsplats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                centers = np.random.rand(50, 3).astype(np.float32)
                amplitudes = np.random.rand(50).astype(np.float32)
                cholesky = np.eye(3)[np.newaxis, :, :].repeat(50, axis=0)
                chol_packed = np.column_stack(
                    [
                        cholesky[:, 0, 0],
                        cholesky[:, 1, 0],
                        cholesky[:, 1, 1],
                        cholesky[:, 2, 0],
                        cholesky[:, 2, 1],
                        cholesky[:, 2, 2],
                    ]
                ).astype(np.float32)
                colors = np.random.rand(50, 3).astype(np.float32)
                with pytest.raises(ValueError, match="colors.*colormap"):
                    scene.add_gsplats(
                        "gs",
                        centers,
                        amplitudes,
                        chol_packed,
                        colors=colors,
                        colormap="green",
                    )

    def test_gsplats_with_colors_no_default_gray(self) -> None:
        """GSplats with explicit colors should NOT get default gray colormap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                centers = np.random.rand(50, 3).astype(np.float32)
                amplitudes = np.random.rand(50).astype(np.float32)
                cholesky = np.eye(3)[np.newaxis, :, :].repeat(50, axis=0)
                chol_packed = np.column_stack(
                    [
                        cholesky[:, 0, 0],
                        cholesky[:, 1, 0],
                        cholesky[:, 1, 1],
                        cholesky[:, 2, 0],
                        cholesky[:, 2, 1],
                        cholesky[:, 2, 2],
                    ]
                ).astype(np.float32)
                colors = np.random.rand(50, 3).astype(np.float32)
                scene.add_gsplats("gs", centers, amplitudes, chol_packed, colors=colors)

            store = zarr.open(str(path), mode="r")
            # Should NOT have colormap attr when colors are explicitly provided
            assert "colormap" not in store["gs"].attrs

    def test_gsplats_invalid_colormap_name_raises(self) -> None:
        """Invalid colormap name should raise at write time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                centers = np.random.rand(50, 3).astype(np.float32)
                amplitudes = np.random.rand(50).astype(np.float32)
                cholesky = np.eye(3)[np.newaxis, :, :].repeat(50, axis=0)
                chol_packed = np.column_stack(
                    [
                        cholesky[:, 0, 0],
                        cholesky[:, 1, 0],
                        cholesky[:, 1, 1],
                        cholesky[:, 2, 0],
                        cholesky[:, 2, 1],
                        cholesky[:, 2, 2],
                    ]
                ).astype(np.float32)
                with pytest.raises(ValueError, match="Unknown colormap"):
                    scene.add_gsplats(
                        "gs",
                        centers,
                        amplitudes,
                        chol_packed,
                        colormap="totally_fake_colormap_xyz",
                    )


class TestColormapLinesCompiler:
    """Test colormap support when writing lines."""

    def test_lines_with_colormap_and_scalars(self) -> None:
        """Lines with colormap and scalar values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                vertices = np.random.rand(100, 3).astype(np.float32)
                widths = np.full(100, 0.1, dtype=np.float32)
                scalars = np.random.rand(100).astype(np.float32)
                scene.add_lines(
                    "ln",
                    vertices,
                    widths,
                    scalars=scalars,
                    colormap="inferno",
                )

            store = zarr.open(str(path), mode="r")
            assert store["ln"].attrs["colormap"] == "inferno"
            assert "scalar_data_range" in store["ln"].attrs
            assert "scalars" in store["ln"]
            # Critical: has_scalars must be in zarr attrs for viewer to enable colormap
            assert store["ln"].attrs.get("has_scalars") is True

    def test_lines_invalid_colormap_name_raises(self) -> None:
        """Invalid colormap name should raise at write time, not silently pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                vertices = np.random.rand(50, 3).astype(np.float32)
                widths = np.full(50, 0.1, dtype=np.float32)
                with pytest.raises(ValueError, match="Unknown colormap"):
                    scene.add_lines(
                        "ln",
                        vertices,
                        widths,
                        colormap="definitely_not_a_real_colormap",
                    )


class TestScalarsReorderWithSpatialOrdering:
    """Regression tests for scalars reordering during spatial ordering.

    When spatial ordering (Morton/Hilbert) is applied, per-element arrays
    (positions, colors, radii, scalars) must all be reordered by the same
    sort_order. A bug (fixed) left scalars unreordered, causing per-point
    scalar values to be misaligned with their positions.
    """

    def test_points_scalars_aligned_after_spatial_ordering(self) -> None:
        """Scalars must be reordered consistently with positions.

        The encoder may quantize scalars (e.g., to uint8), so we use the
        Luxar decoder to read them back as float32 for comparison.
        """
        from luxar.encoding.decoder import ArrayDecoder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            # Create points with known pattern: scalar[i] = position[i].sum()
            n = 500
            positions = np.random.rand(n, 3).astype(np.float32)
            scalars = positions.sum(axis=1).astype(np.float32)

            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                scene.add_points("pts", positions, scalars=scalars, colormap="viridis")

            # Read back using decoder to handle quantization
            store = zarr.open(str(path), mode="r")
            decoder = ArrayDecoder()
            read_pos = decoder.decode(store["pts"]["positions"])
            read_scalars = decoder.decode(store["pts"]["scalars"])

            # Each scalar should still equal the sum of its position's coordinates
            # (with tolerance for quantization)
            expected_scalars = read_pos.sum(axis=1)
            np.testing.assert_allclose(
                read_scalars,
                expected_scalars,
                atol=0.05,
                err_msg="Scalars are not aligned with positions after spatial reordering",
            )

    def test_lines_scalars_aligned_after_spatial_ordering(self) -> None:
        """Lines scalars must be reordered consistently with vertices."""
        from luxar.encoding.decoder import ArrayDecoder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            n = 200
            vertices = np.random.rand(n, 3).astype(np.float32)
            widths = np.full(n, 0.1, dtype=np.float32)
            scalars = vertices[:, 0].copy()  # scalar = x coordinate

            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                scene.add_lines(
                    "ln", vertices, widths, scalars=scalars, colormap="plasma"
                )

            store = zarr.open(str(path), mode="r")
            decoder = ArrayDecoder()
            read_verts = decoder.decode(store["ln"]["vertices"])
            read_scalars = decoder.decode(store["ln"]["scalars"])

            # Each scalar should still equal the x-coordinate of its vertex
            np.testing.assert_allclose(
                read_scalars,
                read_verts[:, 0],
                atol=0.05,
                err_msg="Scalars are not aligned with vertices after spatial reordering",
            )


class TestColormapLinesScalarsBug:
    """Regression tests for the Lines scalars bug (vertices vs positions key)."""

    def test_lines_scalars_does_not_crash(self) -> None:
        """Lines with scalars must not crash — _write_scalars_dataset must find 'vertices'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                vertices = np.random.rand(20, 3).astype(np.float32)
                widths = np.full(20, 0.05, dtype=np.float32)
                scalars = np.random.rand(20).astype(np.float32)
                ln = scene.add_lines(
                    "ln", vertices, widths, scalars=scalars, colormap="plasma"
                )
                assert ln.has_scalars

            store = zarr.open(str(path), mode="r")
            assert "scalars" in store["ln"]
            assert store["ln"].attrs["colormap"] == "plasma"
            sr = store["ln"].attrs["scalar_data_range"]
            assert sr[0] <= sr[1]


class TestColormapNodeProperty:
    """Test Node.colormap property."""

    def test_colormap_set_at_creation(self) -> None:
        """Colormap set via attrs at node creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                pts = scene.add_points("pts", positions, colormap="viridis")
                assert pts.colormap == "viridis"

    def test_colormap_setter_accepts_string(self) -> None:
        """Colormap setter accepts string names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                pts = scene.add_points("pts", positions, colormap="viridis")
                pts.colormap = "magenta"
                assert pts.colormap == "magenta"

    def test_colormap_setter_rejects_array(self) -> None:
        """Colormap property setter must reject numpy arrays (JSON serialization)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                pts = scene.add_points("pts", positions, colormap="viridis")
                with pytest.raises(TypeError, match="string names"):
                    pts.colormap = np.zeros((256, 3), dtype=np.uint8)

    def test_colormap_default_none(self) -> None:
        """Colormap defaults to None when not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(50, 3).astype(np.float32)
                colors = np.random.rand(50, 3).astype(np.float32)
                pts = scene.add_points("pts", positions, colors=colors)
                assert pts.colormap is None


class TestColormapMultiLodAdditive:
    """Custom (ndarray / non-builtin) colormaps on the additive-LOD writers.

    Regression: ``write_points_multi_lod`` / ``write_lines_multi_lod`` dumped
    the raw attrs to JSON without the colormap-LUT resolution the flat writers
    do, so ``additive_lod= + colormap=<ndarray>`` crashed with
    ``Object of type ndarray is not JSON serializable``.
    """

    def test_points_additive_lod_custom_colormap_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            custom_lut = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(500, 3).astype(np.float32)
                scalars = np.random.rand(500).astype(np.float32)
                pts = scene.add_points(
                    "pts",
                    positions,
                    scalars=scalars,
                    colormap=custom_lut,
                    additive_lod=dict(method="random", n_lods=2, seed=0),
                )
                # Adder-side sync: the returned node mirrors the writer.
                assert pts.colormap == "custom"

            store = zarr.open(str(path), mode="r")
            assert store["pts"].attrs["colormap"] == "custom"
            assert store["pts"].attrs["n_additive_sublods"] == 2
            lut = np.array(store["pts"]["colormap_lut"])
            np.testing.assert_array_equal(lut, custom_lut)
            # The LUT rides on the parent (the logical node); children carry none.
            for i in range(2):
                assert "colormap_lut" not in store[f"pts/additive_{i}"]

    def test_lines_additive_lod_custom_colormap_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            custom_lut = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                # Many 2-vertex segments: the lines additive ladder is
                # per-polyline, so >1 polyline is required for >1 level.
                vertices = np.random.rand(400, 3).astype(np.float32)
                scalars = np.random.rand(400).astype(np.float32)
                lines = scene.add_lines(
                    "lns",
                    vertices,
                    0.05,
                    scalars=scalars,
                    colormap=custom_lut,
                    line_type="segments",
                    additive_lod=dict(method="random", n_lods=2, seed=0),
                )
                assert lines.colormap == "custom"

            store = zarr.open(str(path), mode="r")
            assert store["lns"].attrs["colormap"] == "custom"
            assert store["lns"].attrs["n_additive_sublods"] == 2
            lut = np.array(store["lns"]["colormap_lut"])
            np.testing.assert_array_equal(lut, custom_lut)
            for i in range(2):
                assert "colormap_lut" not in store[f"lns/additive_{i}"]

    def test_points_additive_lod_matplotlib_name(self) -> None:
        # The same missing LUT call also broke non-builtin string names.
        pytest.importorskip("matplotlib")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            with LuxarZarrCompiler(path) as c:
                scene = c.create_scene(dimensions=dims)
                positions = np.random.rand(300, 3).astype(np.float32)
                scalars = np.random.rand(300).astype(np.float32)
                pts = scene.add_points(
                    "pts",
                    positions,
                    scalars=scalars,
                    colormap="cividis",
                    additive_lod=dict(method="random", n_lods=2, seed=0),
                )
                assert pts.colormap == "custom"

            store = zarr.open(str(path), mode="r")
            assert store["pts"].attrs["colormap"] == "custom"
            assert "colormap_lut" in store["pts"]


class TestSignedColormapScalars:
    """Regression tests for issue #730: colormap scalars are legitimately
    signed (z-scores, velocities, divergence).

    Before the fix, the ``scalars`` dataset was written as POSITIVE_SCALAR,
    whose encoder rejects negative values — but only AFTER ``positions`` had
    already been written to disk. The result was a partial, corrupt node (a
    group with only ``positions`` and no ``type`` attr) that ``finalize()``
    consolidated into ``.zmetadata``. Scalars are now BOUNDED_SCALAR, which
    accepts signed values.
    """

    def test_points_signed_scalar_array_roundtrips(self) -> None:
        """A signed float scalar array writes cleanly and decodes back."""
        from luxar.encoding.decoder import ArrayDecoder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            positions = np.random.rand(20, 3).astype(np.float32)
            scalars = np.linspace(-1.0, 1.0, 20).astype(np.float32)

            with LuxarZarrCompiler(path) as c:
                c.create_scene(dimensions=dims)
                c.write_points("neg", positions, scalars=scalars)

            store = zarr.open(str(path), mode="r")
            # Node is complete: type attr present, scalars dataset present.
            assert store["neg"].attrs["type"] == "points"
            assert "scalars" in store["neg"]
            sr = store["neg"].attrs["scalar_data_range"]
            assert sr[0] == pytest.approx(-1.0)
            assert sr[1] == pytest.approx(1.0)

            decoder = ArrayDecoder()
            read_scalars = decoder.decode(store["neg"]["scalars"])
            assert np.all(np.isfinite(read_scalars))

            # Spatial ordering may permute elements; compare the SORTED sets.
            # Values round-trip within uint8-over-span-2.0 tolerance
            # (step ≈ 2/255 ≈ 0.0078).
            np.testing.assert_allclose(
                np.sort(read_scalars),
                np.sort(scalars),
                atol=2.0 / 255 + 1e-4,
                err_msg="Signed scalars differ beyond uint8 quantization tolerance",
            )

    def test_lines_signed_scalar_array_roundtrips(self) -> None:
        """Signed scalars on Lines also write cleanly (vertices key path)."""
        from luxar.encoding.decoder import ArrayDecoder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            vertices = np.random.rand(30, 3).astype(np.float32)
            widths = np.full(30, 0.05, dtype=np.float32)
            scalars = np.linspace(-3.0, 2.0, 30).astype(np.float32)

            with LuxarZarrCompiler(path) as c:
                c.create_scene(dimensions=dims)
                c.write_lines("ln", vertices, widths, scalars=scalars)

            store = zarr.open(str(path), mode="r")
            assert store["ln"].attrs["type"] == "lines"
            assert "scalars" in store["ln"]

            decoder = ArrayDecoder()
            read_scalars = decoder.decode(store["ln"]["scalars"])
            assert np.all(np.isfinite(read_scalars))
            assert float(read_scalars.min()) < 0.0  # signed values preserved

    def test_points_signed_broadcast_scalar(self) -> None:
        """A uniform NEGATIVE broadcast scalar value writes without error."""
        from luxar.encoding.decoder import ArrayDecoder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            positions = np.random.rand(15, 3).astype(np.float32)

            with LuxarZarrCompiler(path) as c:
                c.create_scene(dimensions=dims)
                c.write_points("neg", positions, scalars=-0.5)

            store = zarr.open(str(path), mode="r")
            assert store["neg"].attrs["type"] == "points"
            assert "scalars" in store["neg"]
            sr = store["neg"].attrs["scalar_data_range"]
            assert sr[0] == pytest.approx(-0.5)
            assert sr[1] == pytest.approx(-0.5)

            decoder = ArrayDecoder()
            read_scalars = decoder.decode(store["neg"]["scalars"])
            np.testing.assert_allclose(read_scalars, -0.5)

    def test_signed_scalars_leave_no_partial_node(self) -> None:
        """The previously-failing case must leave a COMPLETE node.

        Asserts the full expected surface (type attr + positions + scalars +
        has_scalars) so a regression to POSITIVE_SCALAR — which would abort
        mid-write after positions — is caught.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            positions = np.random.rand(40, 3).astype(np.float32)
            scalars = np.random.uniform(-5.0, 5.0, 40).astype(np.float32)

            with LuxarZarrCompiler(path) as c:
                c.create_scene(dimensions=dims)
                c.write_points("z", positions, scalars=scalars, colormap="viridis")

            store = zarr.open(str(path), mode="r")
            grp = store["z"]
            assert grp.attrs["type"] == "points"
            assert grp.attrs.get("has_scalars") is True
            assert "positions" in grp
            assert "scalars" in grp
            assert grp.attrs["colormap"] == "viridis"

    def test_float32_overflow_array_fails_fast_no_partial_node(self) -> None:
        """A float64 scalar array beyond the float32 range is rejected BEFORE
        any write — the exact #730 signature via a different trigger.

        Scalars are stored as float32; ``1e40`` overflows to inf on cast. The
        pre-write gate must reject it so ``positions`` is never committed and
        no partial node is left for ``finalize()`` to consolidate.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            positions = np.random.rand(10, 3).astype(np.float32)
            scalars = np.array([1e40] * 10, dtype=np.float64)

            with LuxarZarrCompiler(path) as c:
                c.create_scene(dimensions=dims)
                with pytest.raises(ValueError, match="float32"):
                    c.write_points("p", positions, scalars=scalars)

            # Store finalized on clean context exit — no partial node "p".
            store = zarr.open(str(path), mode="r")
            assert "p" not in store, "float32-overflow left a partial node"

    def test_float32_overflow_broadcast_fails_fast_no_partial_node(self) -> None:
        """A broadcast scalar beyond the float32 range is also rejected up front."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.luxar.zarr"
            dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
            positions = np.random.rand(10, 3).astype(np.float32)

            with LuxarZarrCompiler(path) as c:
                c.create_scene(dimensions=dims)
                with pytest.raises(ValueError, match="float32"):
                    c.write_points("p", positions, scalars=1e40)

            store = zarr.open(str(path), mode="r")
            assert "p" not in store, "float32-overflow left a partial node"
