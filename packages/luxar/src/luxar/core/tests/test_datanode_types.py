"""Tests for DataNode types: Lines and GSplats."""

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.datanode import DataNode
from luxar.core.gsplats import GSplats
from luxar.core.lines import Lines
from luxar.core.mesh import Mesh
from luxar.core.points import Points


# [Python-R1/core-MAJOR / feedback_geometry_symmetry] Points / Lines /
# GSplats / Mesh expose a parallel "common interface" of properties
# (n_elements, has_colors, has_labels, has_image_labels) and follow the
# same metadata-key convention. The original tests in this file exercise
# Lines and GSplats individually; Points and Mesh have no dedicated
# coverage at the class level. A drift in any one type (e.g., a refactor
# that renamed has_colors to has_color on Lines but not Points) would
# have no symmetric test to surface it.
#
# The parametrize matrix below builds a minimal-metadata instance of
# each type and asserts the four common properties are wired up to the
# same metadata keys with the same boolean semantics. Type-specific
# extras (Points.has_radii, Lines.has_widths, GSplats.has_amplitudes)
# stay in their per-type tests below.
@pytest.mark.parametrize(
    "geometry_cls",
    [Points, Lines, GSplats, Mesh],
    ids=["points", "lines", "gsplats", "mesh"],
)
class TestGeometryTypeParity:
    def test_common_properties_exist(self, geometry_cls) -> None:
        node = geometry_cls(
            "p",
            metadata={"n_points": 7, "n_vertices": 7, "n_splats": 7, "ndim": 3},
        )
        # Every type exposes the same four "has-X" booleans plus
        # n_elements with the same integer semantics.
        assert isinstance(node.n_elements, int)
        assert hasattr(node, "has_colors")
        assert hasattr(node, "has_labels")
        assert hasattr(node, "has_image_labels")

    def test_has_colors_reads_metadata_key(self, geometry_cls) -> None:
        node_no_colors = geometry_cls(
            "p", metadata={"n_points": 0, "n_vertices": 0, "n_splats": 0}
        )
        assert node_no_colors.has_colors is False

        node_with_colors = geometry_cls(
            "p",
            metadata={
                "n_points": 0,
                "n_vertices": 0,
                "n_splats": 0,
                "has_colors": True,
            },
        )
        assert node_with_colors.has_colors is True

    def test_has_labels_reads_metadata_key(self, geometry_cls) -> None:
        no_labels = geometry_cls(
            "p", metadata={"n_points": 0, "n_vertices": 0, "n_splats": 0}
        )
        assert no_labels.has_labels is False

        with_labels = geometry_cls(
            "p",
            metadata={
                "n_points": 0,
                "n_vertices": 0,
                "n_splats": 0,
                "has_labels": True,
            },
        )
        assert with_labels.has_labels is True

    def test_has_image_labels_reads_metadata_key(self, geometry_cls) -> None:
        no = geometry_cls("p", metadata={"n_points": 0, "n_vertices": 0, "n_splats": 0})
        assert no.has_image_labels is False

        yes = geometry_cls(
            "p",
            metadata={
                "n_points": 0,
                "n_vertices": 0,
                "n_splats": 0,
                "has_image_labels": True,
            },
        )
        assert yes.has_image_labels is True


class TestDataNodeNdim:
    """Test DataNode.ndim property."""

    def test_ndim_raises_on_missing_metadata(self) -> None:
        """Test that ndim raises ValueError when metadata has no dims key."""

        class _StubDataNode(DataNode):
            @property
            def n_elements(self) -> int:
                return 0

        node = _StubDataNode("test_node", metadata={})
        with pytest.raises(ValueError, match="no dimensionality metadata"):
            _ = node.ndim

    def test_ndim_works_with_ndim_key(self) -> None:
        """Test that ndim reads from 'ndim' key."""

        class _StubDataNode(DataNode):
            @property
            def n_elements(self) -> int:
                return 0

        node = _StubDataNode("test_node", metadata={"ndim": 5})
        assert node.ndim == 5


class TestLinesNode:
    """Test Lines node class and Scene.add_lines()."""

    def test_add_lines_polyline(self, tmp_path) -> None:
        """Test creating a polyline."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )
        widths = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lines = scene.add_lines("polyline", vertices, widths, line_type="polyline")

            # Test properties
            assert lines.n_elements == 4  # vertex count (geometry-neutral)
            assert lines.n_segments == 3  # polyline: N-1 segments
            assert lines.line_type == "polyline"
            assert lines.ndim == 3
            assert lines.has_colors is False
            assert lines.has_sharpness is False
            assert lines.max_width == pytest.approx(0.1)

    def test_add_lines_segments(self, tmp_path) -> None:
        """Test creating independent segments."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32
        )
        widths = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lines = scene.add_lines("segments", vertices, widths, line_type="segments")

            assert lines.n_elements == 4
            assert lines.n_segments == 2  # segments: N//2
            assert lines.line_type == "segments"
            assert lines.max_width == pytest.approx(0.2)

    def test_add_lines_loop(self, tmp_path) -> None:
        """Test creating a closed loop."""
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        widths = np.full(4, 0.05, dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            lines = scene.add_lines("loop", vertices, widths, line_type="loop")

            assert lines.n_elements == 4
            assert lines.n_segments == 4  # loop: N segments (including wrap)
            assert lines.line_type == "loop"
            assert lines.ndim == 2

    def test_add_lines_indexed(self, tmp_path) -> None:
        """Test creating indexed lines."""
        # Triangle: 3 vertices, 3 edges
        vertices = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=np.float32)
        widths = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        indices = np.array([0, 1, 1, 2, 2, 0], dtype=np.uint32)  # 3 segments

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            lines = scene.add_lines(
                "triangle", vertices, widths, indices=indices, line_type="indexed"
            )

            assert lines.n_elements == 3
            assert lines.n_segments == 3  # indexed: len(indices)//2
            assert lines.line_type == "indexed"

    def test_add_lines_with_colors(self, tmp_path) -> None:
        """Test lines with per-vertex colors."""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        widths = np.array([0.1, 0.1], dtype=np.float32)
        colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # Red to green

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lines = scene.add_lines(
                "gradient", vertices, widths, colors=colors, line_type="polyline"
            )

            assert lines.has_colors is True
            assert lines.n_elements == 2

    def test_add_lines_with_sharpness(self, tmp_path) -> None:
        """Test lines with per-vertex sharpness."""
        vertices = np.array([[0, 0], [1, 0]], dtype=np.float32)
        widths = np.array([0.1, 0.2], dtype=np.float32)
        sharpness = np.array([0.3, 0.8], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            lines = scene.add_lines(
                "tapered", vertices, widths, sharpness=sharpness, line_type="polyline"
            )

            assert lines.has_sharpness is True

    def test_add_lines_broadcast_width(self, tmp_path) -> None:
        """Test lines with single width value (broadcast)."""
        vertices = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        width = 0.5  # Single value

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            lines = scene.add_lines("uniform", vertices, width, line_type="polyline")

            assert lines.n_elements == 3
            assert lines.max_width == pytest.approx(0.5)

    def test_lines_validation_segments_odd_vertices(self, tmp_path) -> None:
        """Test that segments with odd vertex count raises error."""
        vertices = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)  # Odd!
        widths = np.full(3, 0.1, dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            with pytest.raises(ValueError, match="even number of vertices"):
                scene.add_lines("bad", vertices, widths, line_type="segments")

    def test_lines_validation_indexed_missing_indices(self, tmp_path) -> None:
        """Test that indexed type without indices raises error."""
        vertices = np.array([[0, 0], [1, 0]], dtype=np.float32)
        widths = np.array([0.1, 0.1], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            with pytest.raises(ValueError, match="requires indices"):
                scene.add_lines("bad", vertices, widths, line_type="indexed")

    def test_lines_validation_invalid_line_type(self, tmp_path) -> None:
        """Test that invalid line_type raises error."""
        vertices = np.array([[0, 0], [1, 0]], dtype=np.float32)
        widths = np.array([0.1, 0.1], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            with pytest.raises(ValueError, match="Invalid line_type"):
                scene.add_lines("bad", vertices, widths, line_type="invalid")


class TestGSplatsNode:
    """Test GSplats node class and Scene.add_gsplats()."""

    def test_add_gsplats_2d(self, tmp_path) -> None:
        """Test creating 2D Gaussian splats."""
        centers = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        amplitudes = np.array([1.0, 2.0, 1.5], dtype=np.float32)
        # 2D Cholesky: k = 2*(2+1)/2 = 3 (L00, L10, L11)
        cholesky = np.array([[1, 0, 1], [1, 0.5, 1], [1, 0, 1]], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            gsplats = scene.add_gsplats("splats2d", centers, amplitudes, cholesky)

            assert gsplats.n_elements == 3  # splat count (geometry-neutral)
            assert gsplats.ndim == 2
            assert gsplats.has_colors is False
            assert gsplats.amplitude_range["min"] == 1.0
            assert gsplats.amplitude_range["max"] == 2.0

    def test_add_gsplats_3d(self, tmp_path) -> None:
        """Test creating 3D Gaussian splats."""
        centers = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        amplitudes = np.array([1.0, 1.0], dtype=np.float32)
        # 3D Cholesky: k = 3*(3+1)/2 = 6
        cholesky = np.array([[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            gsplats = scene.add_gsplats("splats3d", centers, amplitudes, cholesky)

            assert gsplats.n_elements == 2
            assert gsplats.ndim == 3

    def test_add_gsplats_with_colors(self, tmp_path) -> None:
        """Test gsplats with RGB colors."""
        centers = np.array([[0, 0]], dtype=np.float32)  # 2D
        amplitudes = np.array([1.0], dtype=np.float32)
        cholesky = np.array([[1, 0, 1]], dtype=np.float32)  # 2D: k=3
        colors = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            gsplats = scene.add_gsplats(
                "colored", centers, amplitudes, cholesky, colors=colors
            )

            assert gsplats.has_colors is True

    def test_add_gsplats_broadcast_amplitude(self, tmp_path) -> None:
        """Test gsplats with single amplitude value (broadcast)."""
        centers = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)  # 2D
        amplitude = 2.0  # Single value
        cholesky = np.array(
            [1, 0, 1], dtype=np.float32
        )  # Broadcast cholesky: 1D array will be broadcast

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            gsplats = scene.add_gsplats("uniform", centers, amplitude, cholesky)

            assert gsplats.n_elements == 3
            assert gsplats.amplitude_range["min"] == 2.0
            assert gsplats.amplitude_range["max"] == 2.0

    def test_gsplats_validation_negative_amplitude(self, tmp_path) -> None:
        """Test that negative amplitudes raise error."""
        centers = np.array([[0, 0]], dtype=np.float32)
        amplitudes = np.array([-1.0], dtype=np.float32)
        cholesky = np.array([[1, 0, 1]], dtype=np.float32)  # 2D: k=3

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            with pytest.raises(ValueError, match="must be non-negative"):
                scene.add_gsplats("bad", centers, amplitudes, cholesky)

    def test_gsplats_validation_wrong_cholesky_shape(self, tmp_path) -> None:
        """Test that wrong cholesky shape raises error."""
        centers = np.array([[0, 0]], dtype=np.float32)  # 2D
        amplitudes = np.array([1.0], dtype=np.float32)
        cholesky = np.array([[1, 0]], dtype=np.float32)  # Wrong! 2D needs k=3

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            with pytest.raises(ValueError, match="Cholesky factors shape"):
                scene.add_gsplats("bad", centers, amplitudes, cholesky)

    def test_gsplats_broadcast_cholesky(self, tmp_path) -> None:
        """Test gsplats with broadcast cholesky factors."""
        centers = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)  # 2D, 3 splats
        amplitudes = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # Single cholesky: will be broadcast to all splats
        cholesky = np.array(
            [1, 0, 1], dtype=np.float32
        )  # 1D array: shape (3,) → 2D: k=3

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            gsplats = scene.add_gsplats("isotropic", centers, amplitudes, cholesky)

            assert gsplats.n_elements == 3

    def test_gsplats_uniform_cholesky_multichunk_write(self, tmp_path) -> None:
        """Regression for issue #729: writing uniform-Cholesky splats whose
        count exceeds one spatial-index chunk must not crash.

        The uniform-Cholesky convenience keeps a single shared (1, k) row; the
        spatial-ordering chunk-bounds pass used to slice it positionally, which
        yields an empty (0, k) array (and a broadcast error) for every chunk
        after the first. N=3000 3D splats span multiple chunks.
        """
        from luxar.typing_utils import TARGET_CHUNK_BYTES

        n_splats = 3000
        rng = np.random.default_rng(729)
        centers = (rng.random((n_splats, 3)) * 100).astype(np.float32)  # 3D
        amplitudes = np.ones(n_splats, dtype=np.float32)
        # Single shared 1D cholesky (k=6 for 3D) → broadcast to all splats.
        cholesky = np.array([1, 0, 1, 0, 0, 1], dtype=np.float32)

        # The bug only triggers when the splats span MORE THAN ONE spatial-index
        # chunk. Pin that here so the test can't silently degrade to a single
        # chunk (and stop covering the regression) if TARGET_CHUNK_BYTES grows.
        # Mirrors the chunk_size formula in io/_compiler/gsplat_assembly.py.
        n_dims = 3
        expected_k = n_dims * (n_dims + 1) // 2
        bytes_per_splat = n_dims * 4 + 4 + expected_k * 4 + 16
        chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_splat)
        assert chunk_size < n_splats, (chunk_size, n_splats)

        with LuxarZarrCompiler(
            tmp_path / "uniform.zarr", enable_spatial_index=True
        ) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            gsplats = scene.add_gsplats("uniform", centers, amplitudes, cholesky)

            assert gsplats.n_elements == n_splats

    def test_add_gsplats_from_data(self, tmp_path) -> None:
        """Test adding gsplats from GSplatData."""
        from luxar.gsplats.gsplat_data import GSplatData

        # Create a result object
        centers = np.array([[0, 0], [1, 1]], dtype=np.float32)
        amplitudes = np.array([1.0, 2.0], dtype=np.float32)
        cholesky = np.array([[1, 0, 1], [1, 0.5, 1]], dtype=np.float32)
        colors = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float32)

        result = GSplatData(
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky,
            colors=colors,
            stats={"test": "value"},
        )

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_2d())
            gsplats = scene.add_gsplats_from_data("from_result", result)

            # Single-substitutive path returns GSplats, not LODGroup.
            assert isinstance(gsplats, GSplats)
            assert gsplats.n_elements == 2
            assert gsplats.ndim == 2
            assert gsplats.has_colors is True

    def test_add_gsplats_from_file(self, tmp_path) -> None:
        """Test adding gsplats from .gsplats.zarr file."""
        from luxar.gsplats.gsplat_data import GSplatData

        # Create and save a result
        centers = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        amplitudes = np.array([1.0, 1.5], dtype=np.float32)
        cholesky = np.array([[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32)

        result = GSplatData(
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky,
            stats={},
        )

        # Save to file
        gsplats_path = tmp_path / "fitted.gsplats.zarr"
        result.save(gsplats_path, ordering="none")

        # Load into scene
        with LuxarZarrCompiler(tmp_path / "scene.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            gsplats = scene.add_gsplats_from_file("from_file", gsplats_path)

            assert gsplats.n_elements == 2
            assert gsplats.ndim == 3
            assert gsplats.has_colors is False

    def test_add_gsplats_from_file_not_found(self, tmp_path) -> None:
        """Test error when file doesn't exist."""
        with LuxarZarrCompiler(tmp_path / "scene.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            with pytest.raises(FileNotFoundError):
                scene.add_gsplats_from_file(
                    "missing", tmp_path / "nonexistent.gsplats.zarr"
                )

    def test_add_gsplats_from_data_invalid_type(self, tmp_path) -> None:
        """Test error when passing wrong type to add_gsplats_from_data."""
        with LuxarZarrCompiler(tmp_path / "scene.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            with pytest.raises(TypeError, match="Expected GSplatData"):
                scene.add_gsplats_from_data("invalid", "not_a_result")


class TestDataNodeAbstraction:
    """Test DataNode abstract base class."""

    def test_datanode_n_elements_abstraction(self, tmp_path) -> None:
        """Test that n_elements works correctly for all data node types."""
        positions = np.random.rand(100, 3).astype(np.float32)
        radii = np.full(100, 0.1, dtype=np.float32)
        vertices = np.random.rand(50, 3).astype(np.float32)
        widths = np.full(50, 0.05, dtype=np.float32)
        centers = np.random.rand(30, 3).astype(np.float32)
        amplitudes = np.ones(30, dtype=np.float32)
        cholesky = np.tile([1, 0, 1, 0, 0, 1], (30, 1)).astype(np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            points = scene.add_points("pts", positions, radii=radii)
            lines = scene.add_lines("lns", vertices, widths)
            gsplats = scene.add_gsplats("spl", centers, amplitudes, cholesky)

            # All should have n_elements property
            assert points.n_elements == 100
            assert lines.n_elements == 50
            assert gsplats.n_elements == 30

            # All should have ndim property
            assert points.ndim == 3
            assert lines.ndim == 3
            assert gsplats.ndim == 3

    def test_datanode_metadata_property(self, tmp_path) -> None:
        """Test that metadata property is accessible on all data nodes."""
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        radii = np.array([0.1], dtype=np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            points = scene.add_points("pts", positions, radii=radii)

            # metadata property should be accessible
            assert "n_points" in points.metadata
            assert "ndim" in points.metadata
            assert points.metadata["n_points"] == 1
            assert points.metadata["ndim"] == 3
