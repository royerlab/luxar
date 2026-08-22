"""Round-trip coverage for BSP metadata written by native scene adders."""

from __future__ import annotations

import numpy as np
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group.partition import serialized_bsp_tree_separates
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


def test_native_partition_adders_write_separating_bsp_trees(tmp_path) -> None:
    """Every native leaf adder persists the BSP it used to create its parts."""
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
    line_vertices = np.stack(
        [
            centers[[0, 1]],
            centers[[2, 3]],
            centers[[4, 5]],
            centers[[6, 7]],
        ]
    ).reshape(-1, 3)
    mesh_vertices = np.array(
        [[x - 0.1, -0.1, 0.0] for x in (-6.0, -2.0, 2.0, 6.0) for _ in (0,)]
        + [[x + 0.1, -0.1, 0.0] for x in (-6.0, -2.0, 2.0, 6.0)]
        + [[x, 0.1, 0.0] for x in (-6.0, -2.0, 2.0, 6.0)],
        dtype=np.float32,
    )
    mesh_faces = np.array([[i, i + 4, i + 8] for i in range(4)], dtype=np.uint32)

    output = tmp_path / "native-partitions.luxar.zarr"
    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("points", centers, partition={"max_elements": 2})
        scene.add_lines(
            "lines",
            line_vertices,
            widths=0.05,
            line_type="segments",
            partition={"max_elements": 2},
        )
        scene.add_mesh(
            "mesh",
            mesh_vertices,
            mesh_faces,
            partition={"max_elements": 1},
        )
        scene.add_gsplats(
            "gsplats",
            centers=centers,
            amplitudes=np.ones(centers.shape[0], dtype=np.float32),
            cholesky_factors=np.array([1, 0, 1, 0, 0, 1], dtype=np.float32),
            partition={"max_elements": 2},
        )

    root = zarr.open_group(str(output), mode="r")
    geometry_names = ("points", "lines", "mesh", "gsplats")
    missing = [name for name in geometry_names if "bsp_tree" not in root[name].attrs]
    assert missing == []
    for name in geometry_names:
        group = root[name]
        assert serialized_bsp_tree_separates(
            group.attrs["bsp_tree"], _part_boxes(group)
        )
