"""Tests for nD transform bounds expansion in the compiler.

Verifies that the compiler's finalize() correctly applies world nd_transforms
to per-node position bounds, producing world-space scene bounds where
non-displayed dimensions reflect transformed ranges.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.dimensions import Dimension


class TestNdTransformBoundsExpansion:
    """Test bounds expansion with nD transforms."""

    def test_bounds_expansion_affine(self) -> None:
        """Affine nd_transform should transform non-displayed dim bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("Time", unit="s", display=False, discrete=True, step=1.0),
                ]
            )

            # Points with Time bounds [0, 50]
            positions = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [10.0, 10.0, 10.0, 50.0],
                ],
                dtype=np.float32,
            )

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points(
                    "pts",
                    positions,
                    nd_transform={"Time": {"scale": 2.0, "offset": 10.0}},
                )

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # Displayed dims unchanged
            assert bounds["min"][0] == pytest.approx(0.0)
            assert bounds["max"][0] == pytest.approx(10.0)
            assert bounds["min"][1] == pytest.approx(0.0)
            assert bounds["max"][1] == pytest.approx(10.0)
            assert bounds["min"][2] == pytest.approx(0.0)
            assert bounds["max"][2] == pytest.approx(10.0)

            # Time dim: min = 2*0 + 10 = 10, max = 2*50 + 10 = 110
            assert bounds["min"][3] == pytest.approx(10.0)
            assert bounds["max"][3] == pytest.approx(110.0)

            # Node-level bounds should still be local
            node_bounds = store["pts"].attrs["position_bounds"]
            assert node_bounds["min"][3] == pytest.approx(0.0)
            assert node_bounds["max"][3] == pytest.approx(50.0)

    def test_bounds_expansion_hierarchical(self) -> None:
        """Parent group nd_transform should compose with child for bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("Time", unit="s", display=False, discrete=True, step=1.0),
                ]
            )

            # Points with Time bounds [0, 10]
            positions = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [5.0, 5.0, 5.0, 10.0],
                ],
                dtype=np.float32,
            )

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                # Parent group with scale=2 on Time
                group = scene.add_group(
                    "parent",
                    nd_transform={"Time": {"scale": 2.0, "offset": 0.0}},
                )
                # Child points (inherit parent's nd_transform)
                group.add_points("child_pts", positions)

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # Time dim: parent scale=2 -> min=2*0=0, max=2*10=20
            assert bounds["min"][3] == pytest.approx(0.0)
            assert bounds["max"][3] == pytest.approx(20.0)

    def test_bounds_expansion_no_transform_unchanged(self) -> None:
        """Without nd_transform, bounds should be identical to local space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("Time", unit="s", display=False, discrete=True, step=1.0),
                ]
            )

            positions = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [10.0, 20.0, 30.0, 50.0],
                ],
                dtype=np.float32,
            )

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points("pts", positions)

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # All dims should match local bounds exactly
            assert bounds["min"] == [0.0, 0.0, 0.0, 0.0]
            assert bounds["max"] == [10.0, 20.0, 30.0, 50.0]

    def test_bounds_expansion_multiple_nodes(self) -> None:
        """Multiple nodes with different nd_transforms: union of world bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("Time", unit="s", display=False, discrete=True, step=1.0),
                ]
            )

            # Node 1: Time [0, 10], transform offset=5 -> world [5, 15]
            positions1 = np.array(
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 10.0]], dtype=np.float32
            )
            # Node 2: Time [0, 20], transform offset=100 -> world [100, 120]
            positions2 = np.array(
                [[2.0, 2.0, 2.0, 0.0], [3.0, 3.0, 3.0, 20.0]], dtype=np.float32
            )

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points(
                    "pts1",
                    positions1,
                    nd_transform={"Time": {"offset": 5.0}},
                )
                scene.add_points(
                    "pts2",
                    positions2,
                    nd_transform={"Time": {"offset": 100.0}},
                )

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # X: union of [0,1] and [2,3] = [0, 3]
            assert bounds["min"][0] == pytest.approx(0.0)
            assert bounds["max"][0] == pytest.approx(3.0)

            # Time: union of [5,15] and [100,120] = [5, 120]
            assert bounds["min"][3] == pytest.approx(5.0)
            assert bounds["max"][3] == pytest.approx(120.0)

    def test_bounds_expansion_negative_scale(self) -> None:
        """Negative scale should correctly flip min/max."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension("Time", unit="s", display=False, discrete=True, step=1.0),
                ]
            )

            # Points with Time bounds [10, 50]
            positions = np.array(
                [
                    [0.0, 0.0, 0.0, 10.0],
                    [5.0, 5.0, 5.0, 50.0],
                ],
                dtype=np.float32,
            )

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points(
                    "pts",
                    positions,
                    nd_transform={"Time": {"scale": -1.0, "offset": 100.0}},
                )

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # Time: scale=-1, offset=100
            # min = -1*50 + 100 = 50, max = -1*10 + 100 = 90
            # (flipped because negative scale)
            assert bounds["min"][3] == pytest.approx(50.0)
            assert bounds["max"][3] == pytest.approx(90.0)
