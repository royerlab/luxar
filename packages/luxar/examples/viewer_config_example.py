#!/usr/bin/env python3
"""Viewer Config Example — embed a default camera, theme, and UI panel state.

This example demonstrates:
- ``LuxarZarrCompiler.create_scene(viewer_config=ViewerConfig(...))`` —
  bundle viewer hints into the zarr file so the scene loads with the
  author's chosen camera, tone mapping, bloom, and panel visibility.
- ``CameraConfig`` — initial camera ``position`` / ``target`` (or
  ``target_node`` to auto-fit to a scene node's bounding box).
- ``UIConfig`` — start with specific panels open (layers, dimensions,
  scale bar, …).
- The priority chain: user localStorage > zarr viewer_config > viewer
  built-in defaults. Authors set sensible starting state; users can
  always override interactively.

Educational value:
- Show how to ship a scene that opens "looking right" without the user
  needing to find the right camera angle by hand.
- See the breadth of authorable viewer state (bloom, exposure, tone
  mapping, background) — the kind of polish that makes scenes feel
  curated.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import (
    CameraConfig,
    Dimensions,
    LuxarZarrCompiler,
    UIConfig,
    ViewerConfig,
)
from luxar.utils.paths import get_examples_output_dir


def main() -> None:
    """Author a tiny scene with a curated viewer configuration."""
    output_path = get_examples_output_dir() / "viewer_config_example.zarr"
    aprint(f"Writing viewer-config example to {output_path}")

    # A simple lit blob the camera config will target by name.
    rng = np.random.default_rng(seed=0)
    positions = rng.normal(0.0, 1.5, (10_000, 3)).astype(np.float32)
    colors = rng.uniform(0.3, 1.0, (10_000, 3)).astype(np.float32)

    viewer_config = ViewerConfig(
        # Start the camera looking at the blob from a fixed angle. Using
        # ``target_node`` (instead of an explicit ``target``) means the
        # viewer auto-fits to that node's bounding box on load — no
        # matter how the blob is positioned, the camera frames it.
        camera=CameraConfig(
            position=(8.0, 6.0, 12.0),
            target_node="blob",
            fov=45.0,
        ),
        # Subtle bloom + ACES tone mapping for a polished look without
        # touching individual point colors.
        tone_mapping="ACES",
        exposure=0.5,
        background_color="#0a0a14",
        # Start with the Layers and Dimensions panels closed, scale bar
        # visible. The user can always toggle these at runtime.
        ui=UIConfig(
            show_layers=False,
            show_dimensions=False,
            show_scale_bar=True,
            show_help=False,
        ),
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=viewer_config,
        )
        scene.add_points(
            "blob",
            positions,
            colors=colors,
            radii=0.04,
            sharpness=0.5,
        )

        # Explainer overlay: confirm the authored viewer state on load.
        add_explainer(
            scene,
            title="Authored Viewer Config",
            body=(
                "The scene ships a <code>ViewerConfig</code> so it opens "
                "<strong>looking right</strong>: a fixed camera framing the "
                "<code>blob</code> node, ACES tone mapping, exposure, a dark "
                "backdrop, and a chosen panel layout. Users can still override."
            ),
            observe=[
                "Camera auto-frames the blob via <code>target_node='blob'</code>.",
                "Background is dark navy with ACES tone mapping applied.",
                "Layers and Dimensions panels start closed; scale bar is shown.",
                "Reloading restores this state unless localStorage overrides it.",
            ],
            observe_label="Verify",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Scene opens with a fixed camera, ACES tone mapping, and dark backdrop.")


if __name__ == "__main__":
    main()
