#!/usr/bin/env python3
"""Mesh Demo: Reflections — a chrome sphere and a glass sphere lit by the data around them

A bright emissive point swirl with two ``material="physical"`` spheres in it, and one
line of ``viewer_config``: ``environment.source = "scene"``. The viewer then captures
the scene itself from the probe (a `CubeCamera`, six low-resolution renders with the
real shaders) and uses that as the environment, so the chrome sphere reflects the
swirl and the glass sphere refracts and reflects it — the data is the light
(``docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md`` §3.3, Phase 4). No dataset,
NumPy only, under a second.

================================================================================
WHAT THIS DEMO SHOWS
================================================================================

**The data is the light.** Luxar's geometry is emissive by design, so a capture of the
scene IS a radiance map of the data. Nothing is added to the graph — no lights, no
HDRI — and house-shaded meshes, points, lines and splats never read the environment,
so the swirl itself renders exactly as it would without the spheres.

**Exact, not approximate, and never per frame.** The capture reuses the real shaders
(colormaps, intensity, opacity, LOD energy compensation and all) and runs on data
commit, slice change and appearance change once the loader settles — a cube map from a
fixed probe is view-independent, so orbiting the camera costs nothing. Open the Layers
panel (**L**), drag the swirl's Intensity, and watch the reflections follow.

**Baking.** ``luxar env bake <this scene>`` drives the same capture headlessly and
stores the six faces in the store; the viewer then prefilters the baked map at load in
milliseconds and skips the live capture. A scene that changes after the bake ignores the
stale map (it records the scene digest it was baked against) and captures live again.

WORKFLOW:
=========

1. **Build** a spiral point swirl with a warm-to-cool colour ramp.
2. **Add** a chrome sphere (``metalness=1``, ``roughness=0.05``) at the swirl's centre
   and a clear glass sphere (``transmission=1``) off to one side.
3. **Set** ``viewer_config.environment = {"source": "scene", "probe": "auto"}``.
4. **Visualize** — orbit; the swirl moves in the chrome as a reflection, not as data.
"""

DEMO_META = {
    "key": "mesh_reflections",
    "title": "Mesh Reflections",
    "description": (
        "A chrome sphere and a glass sphere inside a bright emissive point swirl, lit by "
        "an exact capture of the scene itself (viewer_config.environment.source='scene'): "
        "the data is the light. Also the `luxar env bake` target."
    ),
    "category": "synthetic",
    "geometry": "mixed",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["mesh_reflections"],
}

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, EnvironmentConfig, ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer, parse_demo_flags
from luxar.encoding import EncodingMode
from luxar.mesh.primitives import icosphere
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

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

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]

Arbol.max_depth = 5


# =============================================================================
# Scene construction
# =============================================================================


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
        [
            radius * np.cos(angle),
            (t - 0.5) * SWIRL_HEIGHT,
            radius * np.sin(angle),
        ],
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

    with asection("Writing scene"):
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.MEMORY
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions.default_3d(),
                viewer_config=ViewerConfig(
                    tone_mapping="ACES",
                    # The whole point of the demo: light the physical spheres with an
                    # exact capture of THIS scene from its centre.
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
            aprint("added 'swirl' (the light source)")

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
            aprint("added 'chrome' (metalness=1, roughness=0.05)")

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
            aprint("added 'glass' (transmission=1, ior=1.5)")

            add_demo_caption(scene, "Mesh reflections • the data is the light", None)

    aprint(f"Scene written to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Build the scene (unless ``--serve-only``) and open the viewer."""
    output_path = get_demos_output_dir() / "mesh_reflections.luxar.zarr"

    with asection("Mesh Reflections Demo"):
        if SERVE_ONLY:
            if not output_path.exists():
                aprint(f"No scene at {output_path} — run without --serve-only first.")
                return
            aprint("Serving existing scene (--serve-only)")
        else:
            create_scene(output_path)

        if not NO_SERVE:
            launch_viewer(output_path)


if __name__ == "__main__":
    main()
