#!/usr/bin/env python3
"""Mesh Reflections Example — a chrome and a glass sphere lit by the data around them.

This example demonstrates:
- ``viewer_config.environment = EnvironmentConfig(source="scene")``: the viewer
  captures the scene itself from a probe (an exact ``CubeCamera`` render) and uses
  it as the environment, so the chrome sphere reflects the emissive swirl and the
  glass sphere refracts and reflects it — the data is the light
  (``MESH_PHYSICAL_MATERIALS_SPEC.md`` §3.3, Phase 4).
- The capture is exact (real shaders, colormaps, intensity) and never per frame: it
  reruns on data commit, slice change and appearance change once the loader settles.
- ``luxar env bake <this store>`` runs the same capture headlessly and stores the six
  faces in the store; a scene edited after the bake ignores the stale map.

Educational value:
- Luxar's geometry is emissive by design, so a capture of the scene IS a radiance
  map of the data; nothing is added to the graph, and house-shaded meshes, points,
  lines and splats never read the environment.
- This store is the ``luxar env bake`` target used in the CLI documentation.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, EnvironmentConfig, ViewerConfig
from luxar.encoding import EncodingMode
from luxar.mesh.primitives import icosphere
from luxar.utils.paths import get_examples_output_dir

#: Icosphere refinement: 642 vertices / 1280 faces reads as a sphere at this size.
SUBDIVISIONS = 3
#: Sphere radius in scene units; the swirl's outer radius is a multiple of it.
RADIUS = 1.0
SWIRL_RADIUS = RADIUS * 6.0
SWIRL_HEIGHT = RADIUS * 3.0
#: Enough points to read as a bright continuous ribbon in the reflection.
SWIRL_POINT_COUNT = 60_000
SWIRL_TURNS = 3.0
SWIRL_SEED = 11
#: Where the glass sphere sits relative to the chrome one (which is at the centre).
GLASS_OFFSET = np.array([RADIUS * 3.2, RADIUS * 0.4, RADIUS * 0.8], dtype=np.float32)


def swirl_points(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A spiral ribbon of points with a warm-to-cool colour ramp along it.

    Returns ``(positions, colors)`` in float32; colours are linear RGB in [0, 1]
    and deliberately bright, because they ARE the light source of this scene.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(count)).astype(np.float32)
    angle = t * SWIRL_TURNS * 2.0 * np.pi
    radius = SWIRL_RADIUS * (0.35 + 0.65 * t)
    jitter = rng.normal(scale=RADIUS * 0.18, size=(count, 3)).astype(np.float32)
    positions = np.stack(
        [radius * np.cos(angle), (t - 0.5) * SWIRL_HEIGHT, radius * np.sin(angle)],
        axis=1,
    ).astype(np.float32)
    positions += jitter
    # Warm orange at the inner end, cool cyan at the outer end.
    warm = np.array([1.0, 0.55, 0.15], dtype=np.float32)
    cool = np.array([0.2, 0.75, 1.0], dtype=np.float32)
    colors = (1.0 - t)[:, None] * warm[None, :] + t[:, None] * cool[None, :]
    return positions, colors.astype(np.float32)


def create_scene(output_path) -> None:
    """Write the swirl + two spheres scene to ``output_path``."""
    vertices, faces, normals = icosphere(SUBDIVISIONS, radius=RADIUS)
    positions, colors = swirl_points(SWIRL_POINT_COUNT, SWIRL_SEED)
    aprint(f"icosphere: {len(vertices)} vertices; swirl: {len(positions)} points")

    with LuxarZarrCompiler(output_path, encoding_mode=EncodingMode.MEMORY) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=ViewerConfig(
                tone_mapping="ACES",
                # The whole point: light the physical spheres with an exact capture
                # of THIS scene from its centre.
                environment=EnvironmentConfig(source="scene", probe="auto"),
                camera=CameraConfig(
                    position=(
                        SWIRL_RADIUS * 0.9,
                        SWIRL_RADIUS * 0.55,
                        SWIRL_RADIUS * 1.6,
                    ),
                    target=(0.0, 0.0, 0.0),
                    up=(0.0, 1.0, 0.0),
                ),
            ),
        )
        scene.add_points(
            "swirl",
            positions,
            colors=colors,
            radii=np.full(len(positions), RADIUS * 0.05, dtype=np.float32),
            blending_mode="additive",
            intensity=1.6,
            layer=True,
        )
        scene.add_mesh(
            "chrome",
            vertices,
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            colors=np.tile(
                np.array([[0.95, 0.95, 0.97]], dtype=np.float32), (len(vertices), 1)
            ),
            shading="smooth",
            material="physical",
            metalness=1.0,
            roughness=0.05,
            layer=True,
        )
        scene.add_mesh(
            "glass",
            vertices + GLASS_OFFSET[None, :],
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            colors=np.tile(
                np.array([[1.0, 1.0, 1.0]], dtype=np.float32), (len(vertices), 1)
            ),
            shading="smooth",
            material="physical",
            roughness=0.02,
            transmission=1.0,
            ior=1.5,
            thickness=RADIUS * 2.0,
            layer=True,
        )
        add_explainer(
            scene,
            title="Mesh reflections — the data is the light",
            body="A chrome and a glass sphere inside a bright emissive swirl, lit by "
            "an exact capture of the scene itself "
            '(<code>viewer_config.environment.source = "scene"</code>). Also the '
            "<code>luxar env bake</code> target.",
            observe=[
                "The swirl moves in the chrome as a reflection, not as data.",
                "Orbiting costs nothing: the cube map is captured once, not per frame.",
                "Layers panel (L): drag the swirl's Intensity and the reflection follows.",
            ],
        )

    aprint(f"Created mesh reflections example: {output_path}")


def main() -> None:
    output_path = get_examples_output_dir() / "mesh_reflections_example.luxar.zarr"
    create_scene(output_path)


if __name__ == "__main__":
    main()
