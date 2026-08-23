"""Round-trip coverage for BSP metadata written by native scene adders."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group.adders.lines import add_lines_partition_wrapper_impl
from luxar.core.group.partition import (
    serialized_bsp_leaf_labels,
    serialized_bsp_tree_separates,
    serialized_bsp_tree_straddles_centers,
)
from luxar.io.compiler import LuxarZarrCompiler


def _part_boxes(group: zarr.Group) -> list[tuple[np.ndarray, np.ndarray]]:
    children = [group[name] for name in group.keys() if name.startswith("part_")]
    children.sort(key=lambda child: int(child.attrs["child_index"]))
    return [
        (
            np.asarray(child.attrs["position_bounds"]["min"], dtype=np.float64),
            np.asarray(child.attrs["position_bounds"]["max"], dtype=np.float64),
        )
        for child in children
    ]


@pytest.mark.parametrize("rule", ["median", "midpoint", "sah"])
def test_native_partition_adders_write_valid_bsp_trees(tmp_path, rule: str) -> None:
    """Native adders persist exact point trees and centroid-valid surface trees."""
    centers = np.array(
        [
            [-6.2, -0.1, 0.0],
            [-5.8, 0.1, 0.0],
            [-2.2, -0.1, 0.0],
            [-1.8, 0.1, 0.0],
            [1.8, -0.1, 0.0],
            [2.2, 0.1, 0.0],
            [5.8, -0.1, 0.0],
            [6.2, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    line_vertices = np.concatenate(
        [
            np.column_stack(
                (
                    np.linspace(center - 10.0, center + 10.0, 20),
                    np.full(20, row, dtype=np.float64),
                    np.zeros(20),
                )
            )
            for row, center in enumerate((-6.0, -2.0, 2.0, 6.0))
        ]
    ).astype(np.float32)
    line_indices = np.array(
        [
            (base + offset, base + offset + 1)
            for base in range(0, 80, 20)
            for offset in range(19)
        ],
        dtype=np.uint32,
    )
    mesh_vertices = np.array(
        [
            point
            for center in (-6.0, -2.0, 2.0, 6.0)
            for point in (
                (center - 10.0, -1.0, 0.0),
                (center + 10.0, -1.0, 0.0),
                (center, 2.0, 0.0),
            )
        ],
        dtype=np.float32,
    )
    mesh_faces = np.arange(12, dtype=np.uint32).reshape(4, 3)

    output = tmp_path / "native-partitions.luxar.zarr"
    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("points", centers, partition={"max_elements": 2, "rule": rule})
        scene.add_lines(
            "lines",
            line_vertices,
            widths=0.05,
            indices=line_indices,
            line_type="indexed",
            partition={"max_elements": 20, "rule": rule},
        )
        scene.add_mesh(
            "mesh",
            mesh_vertices,
            mesh_faces,
            partition={"max_elements": 1, "rule": rule},
        )
        scene.add_gsplats(
            "gsplats",
            centers=centers,
            amplitudes=np.ones(centers.shape[0], dtype=np.float32),
            cholesky_factors=np.array([1, 0, 1, 0, 0, 1], dtype=np.float32),
            partition={"max_elements": 2, "rule": rule},
        )

    root = zarr.open_group(str(output), mode="r")
    geometry_names = ("points", "lines", "mesh", "gsplats")
    missing = [name for name in geometry_names if "bsp_tree" not in root[name].attrs]
    assert missing == []
    for name in geometry_names:
        group = root[name]
        tree = group.attrs["bsp_tree"]
        boxes = _part_boxes(group)
        assert serialized_bsp_tree_straddles_centers(tree, boxes)
        if name in ("points", "gsplats"):
            assert serialized_bsp_tree_separates(tree, boxes)
        else:
            assert not serialized_bsp_tree_separates(tree, boxes)


@pytest.mark.parametrize("geometry", ["points", "gsplats"])
def test_empty_partition_uses_the_canonical_writer_error(
    tmp_path, geometry: str
) -> None:
    """Partitioning must not replace the leaf writer's actionable empty error."""
    output = tmp_path / f"empty-{geometry}.luxar.zarr"
    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="Cannot write empty points") as exc:
            if geometry == "points":
                scene.add_points(
                    "empty",
                    np.empty((0, 3), dtype=np.float32),
                    partition={"max_elements": 2},
                )
            else:
                scene.add_gsplats(
                    "empty",
                    centers=np.empty((0, 3), dtype=np.float32),
                    amplitudes=np.empty(0, dtype=np.float32),
                    cholesky_factors=np.empty((0, 6), dtype=np.float32),
                    partition={"max_elements": 2},
                )
        assert "spatial_bsp_tree" not in str(exc.value)


def test_line_partition_prunes_and_renumbers_an_empty_region(tmp_path) -> None:
    """A skipped line region cannot leave the stored tree naming stale children."""
    vertices = np.array(
        [[-2.2, 0.0], [-1.8, 0.0], [1.8, 0.0], [2.2, 0.0]],
        dtype=np.float32,
    )
    bsp_tree = {
        "axis": 0,
        "split": -4.0,
        "left": {"part": 0},
        "right": {
            "axis": 0,
            "split": 0.0,
            "left": {"part": 1},
            "right": {"part": 2},
        },
    }

    output = tmp_path / "pruned-lines.luxar.zarr"
    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_2d())
        add_lines_partition_wrapper_impl(
            scene,
            name="lines",
            vert_arr=vertices,
            polyline_indices=[
                np.empty(0, dtype=np.intp),
                np.array([0, 1], dtype=np.intp),
                np.array([2, 3], dtype=np.intp),
            ],
            polyline_parts=[[0], [1], [2]],
            n_vertices=4,
            widths=0.05,
            colors=None,
            sharpness=None,
            scalars=None,
            labels=None,
            indices=None,
            line_type="segments",
            parent=None,
            extend_to_all=None,
            max_elements=2,
            bsp_tree=bsp_tree,
        )

    group = zarr.open_group(str(output), mode="r")["lines"]
    stored_tree = group.attrs["bsp_tree"]
    assert serialized_bsp_leaf_labels(stored_tree) == [0, 1]
    assert serialized_bsp_tree_separates(stored_tree, _part_boxes(group))
