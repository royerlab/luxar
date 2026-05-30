#!/usr/bin/env python3
"""Lines Indexed Example — explicit connectivity via the indices array.

This example demonstrates the **fourth** ``line_type`` for ``add_lines``:

- ``polyline`` — vertices interpreted as a single connected chain.
- ``segments`` — every pair of consecutive vertices is one segment.
- ``loop`` — polyline that closes back on its starting vertex.
- ``indexed`` (this example) — vertices stored once; an ``indices`` array
  of even length pairs them into segments. Lets you share vertices
  between multiple segments (e.g. a graph, a triangulated mesh edge
  set, or a star where every spoke shares the same center vertex).

Scene: a star graph with 1 center vertex and 8 outer vertices.
Drawing this as ``segments`` would need 16 vertices (8 segments × 2
endpoints, with the center duplicated 8 times). With ``indexed``, only
9 vertices are stored; ``indices`` lists the (center, spoke_i) pairs.

Educational value:
- Discover the ``indexed`` line type and the ``indices`` kwarg.
- Understand when ``indexed`` saves storage compared to ``segments``.
- Foundation for graph/network visualizations (Points + indexed Lines).
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main() -> None:
    """Draw an 8-spoke star using indexed line connectivity."""
    output_path = get_examples_output_dir() / "lines_indexed_example.zarr"
    aprint(f"Writing indexed-lines example to {output_path}")

    # One center vertex + 8 outer vertices on a circle of radius 4.
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    outer = np.column_stack(
        [
            4.0 * np.cos(angles),
            4.0 * np.sin(angles),
            np.zeros_like(angles),
        ]
    )
    vertices = np.vstack(
        [np.array([[0.0, 0.0, 0.0]], dtype=np.float32), outer.astype(np.float32)]
    )
    # 8 spokes: pair index 0 (center) with each of 1..8 (outer rim).
    # The indices array is flat: (a0, b0, a1, b1, …).
    indices = np.empty(16, dtype=np.uint32)
    indices[0::2] = 0
    indices[1::2] = np.arange(1, 9, dtype=np.uint32)

    # Per-vertex widths: thick at the center, thinner at the rim, so the
    # spokes visibly taper outward.
    widths = np.full(9, 0.05, dtype=np.float32)
    widths[0] = 0.2

    # Rainbow per-vertex colors so each spoke has its own hue.
    colors = np.zeros((9, 3), dtype=np.float32)
    colors[0] = [1.0, 1.0, 1.0]  # white center
    for i in range(8):
        h = i / 8.0 * 6.0
        sector = int(h) % 6
        f = h - sector
        table = {
            0: (1.0, f, 0.0),
            1: (1.0 - f, 1.0, 0.0),
            2: (0.0, 1.0, f),
            3: (0.0, 1.0 - f, 1.0),
            4: (f, 0.0, 1.0),
            5: (1.0, 0.0, 1.0 - f),
        }
        colors[i + 1] = table[sector]

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_lines(
            "star_graph",
            vertices=vertices,
            widths=widths,
            colors=colors,
            indices=indices,
            line_type="indexed",
            sharpness=1.0,
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("8 spokes share a single center vertex via the indices array.")


if __name__ == "__main__":
    main()
