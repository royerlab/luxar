"""Advanced tests for Scene class to improve coverage.

Tests cover uncovered lines in scene.py:
- Scene initialization error handling
- add_points input handling (extend_to_all tests are in test_extend_to_all.py)
- add_lines validation and extend_to_all
- add_gsplats validation
- _analyze_extend_candidates method
- get_store_path method
- dimensions property setter
- to_zarr method
"""

import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.dimensions import Dimension
from luxar.core.scene import Scene


class TestSceneInitialization:
    """Tests for Scene initialization error paths."""

    def test_scene_requires_writer(self) -> None:
        """Test that Scene requires a writer."""
        with pytest.raises(ValueError, match="Writer is required"):
            Scene(writer=cast(Any, None), dimensions=Dimensions.default_3d())


class TestAddPointsInputHandling:
    """Tests for add_points input handling (non-extend_to_all).

    Note: extend_to_all tests live in test_extend_to_all.py.
    """

    def test_add_points_1d_positions_error(self) -> None:
        """Test add_points with 1D positions raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                positions = np.random.rand(100).astype(np.float32)  # 1D, not 2D

                with pytest.raises(ValueError, match="shape.*N, D"):
                    scene.add_points("test_points", positions)

    def test_add_points_list_input(self) -> None:
        """Test add_points with list input (converted to array)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Pass list instead of numpy array
                positions = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

                points = scene.add_points("test_points", positions)
                assert points is not None


class TestAddLinesValidation:
    """Tests for add_lines validation paths."""

    def test_add_lines_1d_vertices_error(self) -> None:
        """Test add_lines with 1D vertices raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                vertices = np.random.rand(100).astype(np.float32)  # 1D, not 2D

                with pytest.raises(ValueError, match="shape.*N, D"):
                    scene.add_lines("test_lines", vertices, widths=0.1)

    def test_add_lines_list_input(self) -> None:
        """Test add_lines with list input (converted to array)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Pass list instead of numpy array
                vertices = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

                lines = scene.add_lines("test_lines", vertices, widths=0.1)
                assert lines is not None


class TestAddLinesExtendToAll:
    """Tests for add_lines with various extend_to_all options."""

    def test_extend_to_all_with_string_all(self) -> None:
        """Test add_lines with extend_to_all='all'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 9)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # 4D line segment vertices (2 vertices per segment)
                vertices = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0],
                        [2.0, 2.0, 2.0, 0.0],
                    ],
                    dtype=np.float32,
                )

                lines = scene.add_lines(
                    "test_lines",
                    vertices,
                    widths=0.1,
                    extend_to_all="all",
                )
                assert lines is not None
                # Verify extend_to_all is stored in attrs
                assert "extend_to_all" in lines.attrs
                assert lines.attrs["extend_to_all"] == ["time"]

    def test_extend_to_all_explicit_list(self) -> None:
        """Test add_lines with explicit extend_to_all list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 9)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # 4D vertices - all at time=0
                vertices = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [2.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )

                lines = scene.add_lines(
                    "test_lines",
                    vertices,
                    widths=0.1,
                    extend_to_all=["time"],
                )
                assert lines is not None
                assert "extend_to_all" in lines.attrs
                assert lines.attrs["extend_to_all"] == ["time"]

    def test_extend_to_all_empty_list_no_attrs(self) -> None:
        """Test add_lines with extend_to_all=[] doesn't store attr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 9)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                vertices = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                )

                lines = scene.add_lines(
                    "test_lines",
                    vertices,
                    widths=0.1,
                    extend_to_all=[],  # Empty list - explicit no extension
                )
                assert lines is not None
                # Empty list should not store extend_to_all attr
                assert lines.attrs.get("extend_to_all") is None

    def test_extend_to_all_invalid_value(self) -> None:
        """Test add_lines with invalid extend_to_all value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                vertices = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=np.float32,
                )

                with pytest.raises(ValueError, match="Invalid extend_to_all"):
                    scene.add_lines(
                        "test_lines",
                        vertices,
                        widths=0.1,
                        extend_to_all="invalid_value",
                    )

    def test_extend_to_all_with_warning_for_candidates(self) -> None:
        """Test add_lines warns when extend_to_all candidates exist but not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 9)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # 4D vertices with only one time value (candidate for extend_to_all)
                vertices = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                )

                with pytest.warns(UserWarning, match="time"):
                    lines = scene.add_lines(
                        "test_lines",
                        vertices,
                        widths=0.1,
                        # extend_to_all not specified - should warn
                    )
                assert lines is not None

    def test_extend_to_all_multiple_dimensions(self) -> None:
        """Test add_lines with extend_to_all for multiple dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 9)
                    ),
                    Dimension(
                        "channel", unit="", display=False, discrete=True, range=(0, 3)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # 5D vertices - all at time=0, channel=0
                vertices = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )

                lines = scene.add_lines(
                    "test_lines",
                    vertices,
                    widths=0.1,
                    extend_to_all=["time", "channel"],
                )
                assert lines is not None
                assert "extend_to_all" in lines.attrs
                assert set(lines.attrs["extend_to_all"]) == {"time", "channel"}


class TestAddGSplatsValidation:
    """Tests for add_gsplats validation paths."""

    def test_add_gsplats_1d_centers_error(self) -> None:
        """Test add_gsplats with 1D centers raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                centers = np.random.rand(100).astype(np.float32)  # 1D, not 2D
                cholesky = np.random.rand(100, 6).astype(np.float32)

                with pytest.raises(ValueError, match="shape.*N, D"):
                    scene.add_gsplats(
                        "test_gsplats",
                        centers,
                        amplitudes=1.0,
                        cholesky_factors=cholesky,
                    )

    def test_add_gsplats_list_input(self) -> None:
        """Test add_gsplats with list input (converted to array)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Pass list instead of numpy array
                centers = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
                # Cholesky factors: 3D -> k = 3*(3+1)/2 = 6
                cholesky = np.array(
                    [[1.0, 0, 0, 1.0, 0, 1.0], [1.0, 0, 0, 1.0, 0, 1.0]]
                )

                gsplats = scene.add_gsplats(
                    "test_gsplats",
                    centers,
                    amplitudes=1.0,
                    cholesky_factors=cholesky,
                )
                assert gsplats is not None


class TestAnalyzeExtendCandidates:
    """Tests for _analyze_extend_candidates method."""

    def test_analyze_no_dimensions(self) -> None:
        """Test analyze returns empty when scene has only displayed dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                positions = np.random.rand(100, 3).astype(np.float32)

                # No non-displayed dimensions, so no candidates
                result = scene._analyze_extend_candidates(positions)
                assert result == []

    def test_analyze_multiple_values_no_candidate(self) -> None:
        """Test analyze returns empty when dimension has multiple values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 4)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # Create positions with multiple time values - NOT a candidate
                n_per_time = 20
                positions = []
                for t in range(5):  # 0, 1, 2, 3, 4
                    pts = np.random.rand(n_per_time, 3).astype(np.float32)
                    time_col = np.full((n_per_time, 1), t, dtype=np.float32)
                    positions.append(np.hstack([pts, time_col]))

                all_positions = np.vstack(positions)

                # Multiple time values means NOT a candidate
                result = scene._analyze_extend_candidates(all_positions)
                assert "time" not in result

    def test_analyze_single_value_with_range_is_candidate(self) -> None:
        """Test analyze detects single-value dimension with larger range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 99)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # All points at single time value, but range is [0, 99]
                positions = np.random.rand(5, 4).astype(np.float32)
                positions[:, 3] = 5  # All at time=5

                result = scene._analyze_extend_candidates(positions)
                # Single value with larger range = candidate
                assert "time" in result

    def test_analyze_no_range_not_candidate(self) -> None:
        """Test analyze doesn't flag dimension without defined range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("time", unit="s", display=False),  # No range
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                positions = np.random.rand(5, 4).astype(np.float32)
                positions[:, 3] = 0  # All at time=0

                result = scene._analyze_extend_candidates(positions)
                # No range defined, so not a candidate
                assert "time" not in result


class TestDimensionsProperty:
    """Tests for dimensions property getter and setter."""

    def test_dimensions_setter_none(self) -> None:
        """Test that setting dimensions to None raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                assert scene.dimensions is not None

                # Setting to None should raise ValueError
                with pytest.raises(ValueError, match="dimensions cannot be None"):
                    scene.dimensions = cast(Any, None)

    def test_dimensions_setter_new_dims(self) -> None:
        """Test setting new dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                new_dims = Dimensions(
                    [
                        Dimension("a", unit="mm", display=True),
                        Dimension("b", unit="mm", display=True),
                    ]
                )

                scene.dimensions = new_dims
                assert scene.dimensions is not None
                assert len(scene.dimensions.dimensions) == 2


class TestToZarr:
    """Tests for to_zarr method.

    The full behavioral contract (FileExistsError, inside-source rejection,
    finalize semantics) is covered in ``test_api_regressions.py``.
    """

    def test_to_zarr_finalizes_and_copies_store(self) -> None:
        """Scene.to_zarr finalizes the backing store and copies it elsewhere.

        The detailed behavior (refusing existing destinations, refusing
        nested destinations, etc.) is exercised in test_api_regressions.py;
        this test pins down the smoke-level happy path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"
            export_path = Path(tmpdir) / "export.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.to_zarr(export_path)

            assert export_path.exists()
            assert export_path.is_dir()
            assert (export_path / ".zgroup").exists()
            assert (export_path / ".zattrs").exists()


class TestGetStorePath:
    """Tests for get_store_path method."""

    def test_get_store_path_returns_path(self) -> None:
        """Test that get_store_path returns the store path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                result = scene.get_store_path()
                assert str(store_path) in result
