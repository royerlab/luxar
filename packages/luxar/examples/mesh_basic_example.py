#!/usr/bin/env python3
"""Mesh Basic Example — a minimal shaded tetrahedron.

This example demonstrates:
- Creating a triangle surface with ``add_mesh``.
- Flat, view-anchored shading without a lighting rig.
- Per-vertex RGB colours interpolated across each face.
"""

import numpy as np
from _overlay_style import add_explainer
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
            layer=True,
        )
        add_explainer(
            scene,
            title="Basic mesh",
            body="A tetrahedron built from four vertices and four triangular faces, "
            "using Luxar's light-free flat shading.",
            observe=[
                "Four crisp triangular faces form a closed surface.",
                "The shading needs no authored lights or normals.",
                "Vertex colours blend smoothly across each face.",
            ],
        )

    aprint(f"Created mesh example: {output_path}")


if __name__ == "__main__":
    main()
