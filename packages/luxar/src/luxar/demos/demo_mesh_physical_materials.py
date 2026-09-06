#!/usr/bin/env python3
"""Mesh Demo: Physical Materials — metal, lacquer, pearl and velvet spheres

A row of icospheres, one per look the opt-in ``material="physical"`` family can
give a mesh (``docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md``, Phase 1),
next to one sphere rendered by the house shader for comparison. No dataset: the
subject *is* the material, so the geometry is the analytic sphere from
``luxar.mesh.primitives.icosphere`` and the demo runs in under a second on
NumPy alone.

================================================================================
WHAT THIS DEMO SHOWS
================================================================================

**Two material families.** Every Luxar mesh renders by default with the
light-free, view-anchored key of ``MESH_NODE_SPEC.md`` §6.2 — the first sphere.
That model reads a surface without any light in the scene, but it cannot express
how a surface *reflects an environment*: no Fresnel rim, no metal, no pearl.
``material="physical"`` hands a mesh to three.js's own physically based material
(``MeshPhysicalMaterial`` on WebGL, ``MeshPhysicalNodeMaterial`` on WebGPU), lit
by a scene environment the viewer builds lazily the first time it meets one. The
other spheres are that.

**The marker-shell case.** The second sphere is the reason Phase 1 exists: a
dark, translucent base with ``clearcoat=1`` wrapped around a small emissive point
cluster. The clearcoat's Fresnel term brightens the rim exactly where the surface
turns away from the camera — a *view-relative* effect the fixed house key cannot
produce — so the shell reads as a glassy bubble around the data while the points
inside stay crisp (they draw on top: three's transmission pass never sees Luxar's
transparent point materials, spec §3.4, which is why refraction is a later phase).

**Metal, pearl, velvet.** ``metalness`` with low and high ``roughness`` gives
polished gold and brushed steel; ``iridescence`` a thin-film pearl; ``sheen`` with
a ``sheen_color`` a velvet whose highlight lives at grazing angles. Each knob is a
plain node attr in ``[0, 1]``, written only when set, and the Layers panel
(press **L**) lists a physical layer's knobs read-only in place of the house
shader's Ambient / Shade falloff / Specular / Shininess sliders.

**What stays the same.** Picking, ``layer_order``, opacity, per-vertex colours,
``double_sided`` and nD slicing all work on a physical mesh exactly as on a house
one; the environment is invisible to points, lines, splats and house meshes, so
adding one physical sphere changes nothing about how the rest of a scene renders.

WORKFLOW:
=========

1. **Build** one welded icosphere (642 vertices / 1280 faces).
2. **Add** it six times with a per-sphere colour and material spec.
3. **Add** an emissive point cluster inside the clearcoat shell.
4. **Visualize** — orbit; watch the rim of the dark shell and the highlights of
   the metals move with the camera while the house sphere's key stays put.
"""

DEMO_META = {
    "key": "mesh_physical_materials",
    "title": "Mesh Physical Materials",
    "description": (
        "Six icospheres side by side: the house shader next to clearcoat, gold, "
        "brushed steel, pearl and velvet rendered by three's physically based "
        "material lit from a scene environment (material='physical', Phase 1)."
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
    "outputs": ["mesh_physical_materials"],
}

from typing import Any, Dict, List

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer, parse_demo_flags
from luxar.encoding import EncodingMode
from luxar.mesh.primitives import icosphere
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

#: Icosphere refinement: 642 vertices / 1280 faces reads as a sphere at this size
#: without the silhouette polygons a coarser level shows.
SUBDIVISIONS = 3

#: Sphere radius and the centre-to-centre spacing of the row, in scene units.
RADIUS = 1.0
SPACING = 2.6

#: Points inside the clearcoat shell — enough to read as a cluster, few enough
#: that the shell, not the cloud, is what the eye lands on.
SHELL_POINT_COUNT = 400
SHELL_POINT_SEED = 7

#: The row, in order. ``material`` is absent on the first entry on purpose: it is
#: the house-shader reference, and an absent attr (not ``"luxar"``) is what every
#: existing scene carries. Colours are linear RGB in [0, 1]; the shell's fourth
#: component is per-vertex alpha, which the physical material consumes as
#: opacity exactly as the house shader does.
SPHERES: List[Dict[str, Any]] = [
    {
        "name": "house_shader",
        "label": "House shader (default)",
        "color": (0.72, 0.70, 0.66),
    },
    {
        "name": "clearcoat_shell",
        "label": "Dark clearcoat shell (Fresnel rim)",
        "color": (0.05, 0.07, 0.09, 0.35),
        "material": {
            "material": "physical",
            "roughness": 0.6,
            "metalness": 0.0,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.05,
        },
    },
    {
        "name": "polished_gold",
        "label": "Polished gold",
        "color": (1.0, 0.78, 0.35),
        "material": {"material": "physical", "metalness": 1.0, "roughness": 0.15},
    },
    {
        "name": "brushed_steel",
        "label": "Brushed steel",
        "color": (0.74, 0.75, 0.77),
        "material": {"material": "physical", "metalness": 1.0, "roughness": 0.55},
    },
    {
        "name": "pearl",
        "label": "Iridescent pearl",
        "color": (0.92, 0.90, 0.86),
        "material": {
            "material": "physical",
            "roughness": 0.2,
            "metalness": 0.0,
            "iridescence": 1.0,
        },
    },
    {
        "name": "velvet",
        "label": "Velvet sheen",
        "color": (0.26, 0.03, 0.08),
        "material": {
            "material": "physical",
            "roughness": 1.0,
            "metalness": 0.0,
            "sheen": 1.0,
            "sheen_color": "#ff4d6d",
        },
    },
]

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]

Arbol.max_depth = 5


# =============================================================================
# Scene construction
# =============================================================================


def sphere_centre(index: int) -> np.ndarray:
    """Centre of the ``index``-th sphere: the row runs along x, centred on 0."""
    offset = (index - (len(SPHERES) - 1) / 2.0) * SPACING
    return np.array([offset, 0.0, 0.0], dtype=np.float32)


def vertex_colors(color: Any, n_vertices: int) -> np.ndarray:
    """Broadcast one RGB or RGBA colour to every vertex, as float32."""
    rgb = np.asarray(color, dtype=np.float32)
    return np.tile(rgb[None, :], (n_vertices, 1))


def shell_points(centre: np.ndarray, radius: float) -> np.ndarray:
    """A Gaussian point cluster well inside the shell (3 sigma at 60% of the radius)."""
    rng = np.random.default_rng(SHELL_POINT_SEED)
    offsets = rng.normal(scale=radius * 0.2, size=(SHELL_POINT_COUNT, 3))
    return (centre[None, :] + offsets).astype(np.float32)


def create_scene(output_path) -> None:
    """Write the six-sphere scene to ``output_path``."""
    vertices, faces, normals = icosphere(SUBDIVISIONS, radius=RADIUS)
    aprint(f"icosphere: {len(vertices)} vertices, {len(faces)} faces")

    row_half_width = (len(SPHERES) - 1) / 2.0 * SPACING + RADIUS

    with asection("Writing scene"):
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.MEMORY
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions.default_3d(),
                viewer_config=ViewerConfig(
                    # ACES set explicitly — the house default, and stating it keeps
                    # the compiler's "nothing was chosen" LUT notice quiet.
                    tone_mapping="ACES",
                    # Slightly above the row and off to one side, so every sphere
                    # shows a lit face, a rim and a highlight at once.
                    camera=CameraConfig(
                        position=(
                            row_half_width * 0.35,
                            row_half_width * 0.55,
                            row_half_width * 1.7,
                        ),
                        target=(0.0, 0.0, 0.0),
                        up=(0.0, 1.0, 0.0),
                    ),
                ),
            )

            for index, spec in enumerate(SPHERES):
                centre = sphere_centre(index)
                scene.add_mesh(
                    spec["name"],
                    vertices + centre[None, :],
                    faces,
                    normals=normals,
                    normal_dims=[0, 1, 2],
                    colors=vertex_colors(spec["color"], len(vertices)),
                    shading="smooth",
                    # One row each in the Layers panel (press L): a physical
                    # layer lists its knobs read-only where the house sphere
                    # shows its Ambient / Shade falloff / Specular / Shininess.
                    layer=True,
                    **spec.get("material", {}),
                )
                aprint(f"added '{spec['name']}' ({spec['label']})")

            shell_index = next(
                i for i, s in enumerate(SPHERES) if s["name"] == "clearcoat_shell"
            )
            points = shell_points(sphere_centre(shell_index), RADIUS)
            scene.add_points(
                "shell_points",
                points,
                colors=np.tile(
                    np.array([[0.35, 0.9, 1.0]], dtype=np.float32), (len(points), 1)
                ),
                radii=np.full(len(points), RADIUS * 0.035, dtype=np.float32),
                blending_mode="additive",
                layer=True,
            )
            aprint(
                f"added 'shell_points' ({len(points)} emissive points inside the shell)"
            )

            add_demo_caption(scene, "Mesh physical materials • Phase 1", None)

    aprint(f"Scene written to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Build the scene (unless ``--serve-only``) and open the viewer."""
    output_path = get_demos_output_dir() / "mesh_physical_materials.luxar.zarr"

    with asection("Mesh Physical Materials Demo"):
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
