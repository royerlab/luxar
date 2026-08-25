#!/usr/bin/env python3
"""Create a small shaded tetrahedron mesh example."""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main() -> None:
    output_path = get_examples_output_dir() / "mesh_basic_example.luxar.zarr"
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    colors = np.array(
        [
            [1.0, 0.25, 0.2],
            [0.2, 0.8, 0.35],
            [0.25, 0.45, 1.0],
            [1.0, 0.8, 0.2],
        ],
        dtype=np.float32,
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(
            "tetrahedron",
            vertices,
            faces,
            colors=colors,
            shading="smooth",
            layer=True,
        )

    aprint(f"Created mesh example: {output_path}")


if __name__ == "__main__":
    main()
