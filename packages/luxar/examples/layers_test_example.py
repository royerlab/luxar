#!/usr/bin/env python3
"""Layers Test Example - Generates a dataset with layer=True for E2E testing.

Creates two point clouds with layer=True and different blending modes,
required by the layers-panel, colormap-system, and blending-modes E2E tests.
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main():
    """Create a minimal scene with two layers for E2E testing."""
    output_path = get_examples_output_dir() / "layers_test_example.zarr"
    aprint(f"Generating layers test example at: {output_path}")

    np.random.seed(42)

    with LuxarZarrCompiler(str(output_path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Red cloud with additive blending
        pts1 = np.random.randn(500, 3).astype(np.float32)
        colors1 = np.tile([1.0, 0.0, 0.0], (500, 1)).astype(np.float32)
        scene.add_points(
            "RedCloud",
            positions=pts1,
            colors=colors1,
            radii=0.1,
            layer=True,
            blending_mode="additive",
        )

        # Blue cloud with normal blending, offset so they don't fully overlap
        pts2 = np.random.randn(500, 3).astype(np.float32) + 3.0
        colors2 = np.tile([0.0, 0.0, 1.0], (500, 1)).astype(np.float32)
        scene.add_points(
            "BlueCloud",
            positions=pts2,
            colors=colors2,
            radii=0.1,
            layer=True,
            blending_mode="normal",
        )

    aprint(f"Done: {output_path}")


if __name__ == "__main__":
    main()
