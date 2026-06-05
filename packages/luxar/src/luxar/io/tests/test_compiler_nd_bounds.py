"""Tests for transform bounds expansion in the compiler.

Verifies that the compiler's finalize() correctly applies world-space
transforms to per-node position bounds, producing world-space scene bounds:

- ``nd_transform`` (affine scale/offset) moves the non-displayed dims.
- The 4x4 spatial ``transform`` (translate/rotate/scale) moves the displayed
  dims, applied via an 8-corner box transform so rotation expands the extent.

The spatial-transform coverage is the regression guard for the bug where
translated/scaled child nodes were left in local space in the scene bounds,
making the viewer's dynamic-clipping sphere too small and clipping peripheral
geometry as the camera rotated.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler, transforms
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


class TestSpatialTransformBoundsExpansion:
    """Test scene bounds expansion with 4x4 spatial transforms.

    Regression guard: a node's 4x4 ``transform`` (which moves the displayed
    x/y/z dims) must be folded into the scene-level ``position_bounds``.
    """

    @staticmethod
    def _unit_cube() -> np.ndarray:
        """8 corners of the [-0.5, 0.5]^3 cube."""
        return np.array(
            [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
            dtype=np.float32,
        )

    def test_translation_shifts_displayed_bounds(self) -> None:
        """A translated node must report world-space (shifted) scene bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points(
                    "moved",
                    self._unit_cube(),
                    transform=transforms.translate(3, 0, 0),
                )

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # Scene bounds are world-space: cube shifted +3 in x.
            assert bounds["min"][0] == pytest.approx(2.5)
            assert bounds["max"][0] == pytest.approx(3.5)
            assert bounds["min"][1] == pytest.approx(-0.5)
            assert bounds["max"][1] == pytest.approx(0.5)

            # Per-node bounds remain local.
            node_bounds = store["moved"].attrs["position_bounds"]
            assert node_bounds["min"][0] == pytest.approx(-0.5)
            assert node_bounds["max"][0] == pytest.approx(0.5)

    def test_rotation_expands_bounds_via_eight_corners(self) -> None:
        """A 45-deg rotation must grow the x/y extent to the half-diagonal.

        Transforming only the (min, max) corner pair would underestimate the
        rotated box — this asserts the 8-corner transform is used.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points(
                    "spun",
                    self._unit_cube(),
                    transform=transforms.rotate_z(45),
                )

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            half_diag = np.sqrt(2) / 2  # ~0.707, vs 0.5 for the naive corner pair
            assert bounds["max"][0] == pytest.approx(half_diag, abs=1e-4)
            assert bounds["min"][0] == pytest.approx(-half_diag, abs=1e-4)
            assert bounds["max"][1] == pytest.approx(half_diag, abs=1e-4)
            # Untouched z stays at the local extent.
            assert bounds["max"][2] == pytest.approx(0.5)

    def test_hierarchical_transform_composition(self) -> None:
        """A parent-group transform must compose with the child's."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                group = scene.add_group(
                    "parent", transform=transforms.translate(0, 0, 3)
                )
                group.add_points(
                    "child",
                    self._unit_cube(),
                    transform=transforms.translate(0, 0, 1),
                )

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # World z = local +1 (child) +3 (parent) = +4 -> [3.5, 4.5].
            assert bounds["min"][2] == pytest.approx(3.5)
            assert bounds["max"][2] == pytest.approx(4.5)

    def test_multiple_nodes_union_world_bounds(self) -> None:
        """Scene bounds are the union of every node's world-space box."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points(
                    "left", self._unit_cube(), transform=transforms.translate(-3, 0, 0)
                )
                scene.add_points(
                    "right", self._unit_cube(), transform=transforms.translate(3, 0, 0)
                )

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # Union spans both translated cubes: x in [-3.5, 3.5].
            assert bounds["min"][0] == pytest.approx(-3.5)
            assert bounds["max"][0] == pytest.approx(3.5)

    def test_no_transform_bounds_unchanged(self) -> None:
        """Without any transform, scene bounds equal the local union exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            positions = np.array(
                [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]], dtype=np.float32
            )
            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("pts", positions)

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]
            assert bounds["min"] == [0.0, 0.0, 0.0]
            assert bounds["max"] == [10.0, 20.0, 30.0]

    def test_combined_spatial_and_nd_transform(self) -> None:
        """Spatial 4x4 (displayed dims) and nd_transform (non-displayed dim)
        must both be applied to the same node's bounds."""
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
            # Cube corners in x/y/z plus a Time column spanning [0, 10].
            cube = self._unit_cube()
            time_col = np.array([[0.0]] * 4 + [[10.0]] * 4, dtype=np.float32)
            positions = np.hstack([cube, time_col]).astype(np.float32)

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points(
                    "pts",
                    positions,
                    transform=transforms.translate(3, 0, 0),
                    nd_transform={"Time": {"scale": 2.0, "offset": 5.0}},
                )

            store = zarr.open_group(zarr_path, mode="r")
            bounds = store.attrs["position_bounds"]

            # Displayed x: spatial translate +3 -> [2.5, 3.5].
            assert bounds["min"][0] == pytest.approx(2.5)
            assert bounds["max"][0] == pytest.approx(3.5)
            # Non-displayed Time: 2*t + 5 over [0, 10] -> [5, 25].
            assert bounds["min"][3] == pytest.approx(5.0)
            assert bounds["max"][3] == pytest.approx(25.0)
