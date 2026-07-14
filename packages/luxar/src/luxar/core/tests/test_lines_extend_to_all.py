"""Tests for extend_to_all functionality in Scene.add_lines().

The Lines sibling of test_extend_to_all.py (Points) and
test_gsplats_extend_to_all.py (GSplats) — one dedicated file per geometry,
per the three-geometry symmetry rule.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.dimensions import Dimension


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
