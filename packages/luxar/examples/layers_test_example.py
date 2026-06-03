#!/usr/bin/env python3
"""Layers Test Example - Generates a dataset with layer=True for E2E testing.

Creates two point-cloud layers plus a composite group layer (containing
two non-layer data children) and exercises `visible=False` at creation
time. Required by the layers-panel, colormap-system, and blending-modes
E2E tests.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main():
    """Create a minimal scene with data-node layers, a group layer, and a hidden layer."""
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

        # Composite group layer — its eye toggle must fan out to both children.
        # The children themselves are NOT layers, so the only way to hide them
        # from the panel is via the group's control.
        group = scene.add_group("CompositeLayer", layer=True)
        pts3 = np.random.randn(300, 3).astype(np.float32) - 3.0
        colors3 = np.tile([0.0, 1.0, 0.0], (300, 1)).astype(np.float32)
        group.add_points(
            "GreenPart",
            positions=pts3,
            colors=colors3,
            radii=0.1,
        )
        pts4 = np.random.randn(300, 3).astype(np.float32) + np.array([0.0, -3.0, 0.0])
        colors4 = np.tile([1.0, 1.0, 0.0], (300, 1)).astype(np.float32)
        group.add_points(
            "YellowPart",
            positions=pts4,
            colors=colors4,
            radii=0.1,
        )

        # Hidden-at-creation layer — verifies that `visible=False` takes effect
        # in the viewer's initial state.
        pts5 = np.random.randn(200, 3).astype(np.float32) + np.array([3.0, -3.0, 0.0])
        colors5 = np.tile([1.0, 0.0, 1.0], (200, 1)).astype(np.float32)
        scene.add_points(
            "HiddenLayer",
            positions=pts5,
            colors=colors5,
            radii=0.1,
            layer=True,
            visible=False,
        )

        add_explainer(
            scene,
            title="Layer Panel",
            body=(
                "Two data-node layers, one composite group layer (two non-layer "
                "children), and one layer created with <code>visible=False</code> "
                "exercise the viewer's layers panel."
            ),
            observe=[
                "The layers panel lists RedCloud, BlueCloud, CompositeLayer.",
                "HiddenLayer starts hidden until its eye toggle is clicked.",
                "Toggling CompositeLayer hides both green and yellow children.",
            ],
            observe_label="Verify",
        )

    aprint(f"Done: {output_path}")


if __name__ == "__main__":
    main()
