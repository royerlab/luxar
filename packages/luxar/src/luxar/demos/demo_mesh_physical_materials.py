#!/usr/bin/env python3
"""Mesh Demo: Physical Materials — metal, lacquer, pearl, velvet and glass spheres

A row of icospheres, one per look the opt-in ``material="physical"`` family can
give a mesh (``docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md``, Phases 1 and
2), next to one sphere rendered by the house shader for comparison, in front of
a house-shaded checkerboard for the glass to refract. No dataset: the subject
*is* the material, so the geometry is the analytic sphere from
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
transparent point materials, spec §3.4 — true of the glass spheres too).

**Metal, pearl, velvet.** ``metalness`` with low and high ``roughness`` gives
polished gold and brushed steel; ``iridescence`` a thin-film pearl; ``sheen`` with
a ``sheen_color`` a velvet whose highlight lives at grazing angles. Each knob is a
plain node attr in ``[0, 1]``, written only when set, and the Layers panel
(press **L**) shows a physical layer's knobs as LIVE sliders in place of the
house shader's Ambient / Shade falloff / Specular / Shininess sliders — drag
``Transmission`` on the gold sphere and it turns to glass; Reset restores what
this script authored.

**Glass (Phase 2).** ``transmission`` hands the surface to three's transmission
pass: the last three spheres refract the checkerboard behind them. ``ior`` sets
how strongly, ``thickness`` how far the ray travels inside, ``attenuation_color``
with ``attenuation_distance`` tints what passes through (the amber sphere), and
``dispersion`` splits it into chromatic fringes (the crystal). What glass does
NOT do — and the clear sphere shows it with a point cluster inside — is refract
Luxar's points, lines and splats: they are drawn on top, crisp and unrefracted
(spec §3.4). Only the background and other meshes bend.

**What stays the same.** Picking, ``layer_order``, opacity, per-vertex colours,
``double_sided`` and nD slicing all work on a physical mesh exactly as on a house
one; the environment is invisible to points, lines, splats and house meshes, so
adding one physical sphere changes nothing about how the rest of a scene renders.

WORKFLOW:
=========

1. **Build** one welded icosphere (642 vertices / 1280 faces).
2. **Add** it nine times with a per-sphere colour and material spec.
3. **Add** an emissive point cluster inside the clearcoat shell and inside the
   clear glass sphere, and a house-shaded checkerboard behind the row.
4. **Visualize** — orbit; watch the rim of the dark shell and the highlights of
   the metals move with the camera while the house sphere's key stays put, and
   the checkerboard swim through the glass while the points inside do not.
"""

DEMO_META = {
    "key": "mesh_physical_materials",
    "title": "Mesh Physical Materials",
    "description": (
        "Nine icospheres in front of a checkerboard: the house shader next to "
        "clearcoat, gold, brushed steel, pearl, velvet, clear glass, amber glass "
        "and a dispersive crystal rendered by three's physically based material "
        "lit from a scene environment (material='physical', Phases 1-2)."
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
from luxar.demos._cinematic_camera import pull_in
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
    # --- Phase 2: the glass family (spec §3.4). Thickness is in scene units, so
    # it is stated against RADIUS: a solid sphere refracts through ~one diameter.
    {
        "name": "clear_glass",
        "label": "Clear glass (refracts the backdrop; points inside stay crisp)",
        "color": (1.0, 1.0, 1.0),
        "material": {
            "material": "physical",
            "roughness": 0.05,
            "metalness": 0.0,
            "transmission": 1.0,
            "ior": 1.5,
            "thickness": RADIUS * 2.0,
        },
    },
    {
        "name": "amber_glass",
        "label": "Thick amber glass (volume attenuation)",
        "color": (1.0, 0.92, 0.7),
        "material": {
            "material": "physical",
            "roughness": 0.1,
            "metalness": 0.0,
            "transmission": 1.0,
            "ior": 1.5,
            "thickness": RADIUS * 2.0,
            "attenuation_color": "#f6d148",
            "attenuation_distance": RADIUS * 0.8,
        },
    },
    {
        "name": "dispersive_crystal",
        "label": "Dispersive crystal (chromatic fringes)",
        "color": (1.0, 1.0, 1.0),
        "material": {
            "material": "physical",
            "roughness": 0.0,
            "metalness": 0.0,
            "transmission": 1.0,
            "ior": 2.0,
            "thickness": RADIUS * 2.0,
            "dispersion": 0.6,
        },
    },
]

#: The spheres that get an emissive point cluster inside them: the clearcoat shell
#: (the Phase 1 marker case) and the clear glass sphere, which shows the §3.4
#: contract honestly — points draw crisply on top of the glass, unrefracted, while
#: the backdrop behind the sphere IS refracted.
CLUSTER_SPHERES = ("clearcoat_shell", "clear_glass")

#: The house-shaded checkerboard behind the row: glass needs something with edges
#: to bend. Tiles are unshared-vertex quads so each tile is one flat colour.
BACKDROP_TILES_X = 24
BACKDROP_TILES_Y = 6
BACKDROP_DEPTH = -RADIUS * 3.0

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


def backdrop_panel(
    half_width: float, half_height: float, depth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A checkerboard plane facing +z at ``z = depth``.

    Returns ``(vertices, faces, normals, colors)`` with four UNSHARED vertices per
    tile, so every tile is one flat colour — the edges the glass spheres bend.
    """
    xs = np.linspace(-half_width, half_width, BACKDROP_TILES_X + 1, dtype=np.float32)
    ys = np.linspace(-half_height, half_height, BACKDROP_TILES_Y + 1, dtype=np.float32)
    light = np.array([0.62, 0.60, 0.56], dtype=np.float32)
    dark = np.array([0.16, 0.17, 0.20], dtype=np.float32)
    vertices, faces, colors = [], [], []
    for j in range(BACKDROP_TILES_Y):
        for i in range(BACKDROP_TILES_X):
            base = len(vertices)
            for x, y in (
                (xs[i], ys[j]),
                (xs[i + 1], ys[j]),
                (xs[i + 1], ys[j + 1]),
                (xs[i], ys[j + 1]),
            ):
                vertices.append((x, y, depth))
            faces.append((base, base + 1, base + 2))
            faces.append((base, base + 2, base + 3))
            colors.extend([light if (i + j) % 2 == 0 else dark] * 4)
    v = np.asarray(vertices, dtype=np.float32)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(v), 1))
    return (
        v,
        np.asarray(faces, dtype=np.uint32),
        normals,
        np.asarray(colors, dtype=np.float32),
    )


def create_scene(output_path) -> None:
    """Write the six-sphere scene to ``output_path``."""
    vertices, faces, normals = icosphere(SUBDIVISIONS, radius=RADIUS)
    aprint(f"icosphere: {len(vertices)} vertices, {len(faces)} faces")

    row_half_width = (len(SPHERES) - 1) / 2.0 * SPACING + RADIUS
    # Nine spheres make a wide row: frame it from a little further out and higher
    # than the six-sphere Phase 1 layout so the whole row and the backdrop fit.

    with asection("Writing scene"):
        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.MEMORY
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions.default_3d(),
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    # ACES set explicitly — the house default, and stating it keeps
                    # the compiler's "nothing was chosen" LUT notice quiet.
                    tone_mapping="ACES",
                    # Slightly above the row and off to one side, so every sphere
                    # shows a lit face, a rim and a highlight at once.
                    camera=CameraConfig(
                        position=pull_in(
                            (
                                row_half_width * 0.25,
                                row_half_width * 0.45,
                                row_half_width * 1.9,
                            )
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
                    # layer shows live knob sliders where the house sphere
                    # shows its Ambient / Shade falloff / Specular / Shininess.
                    layer=True,
                    **spec.get("material", {}),
                )
                aprint(f"added '{spec['name']}' ({spec['label']})")

            for sphere_name in CLUSTER_SPHERES:
                index = next(
                    i for i, s in enumerate(SPHERES) if s["name"] == sphere_name
                )
                points = shell_points(sphere_centre(index), RADIUS)
                node_name = (
                    "shell_points"
                    if sphere_name == "clearcoat_shell"
                    else "glass_points"
                )
                scene.add_points(
                    node_name,
                    points,
                    colors=np.tile(
                        np.array([[0.35, 0.9, 1.0]], dtype=np.float32), (len(points), 1)
                    ),
                    radii=np.full(len(points), RADIUS * 0.035, dtype=np.float32),
                    blending_mode="additive",
                    layer=True,
                )
                aprint(
                    f"added '{node_name}' ({len(points)} emissive points inside "
                    f"'{sphere_name}')"
                )

            bv, bf, bn, bc = backdrop_panel(
                row_half_width + RADIUS, RADIUS * 2.2, BACKDROP_DEPTH
            )
            scene.add_mesh(
                "backdrop",
                bv,
                bf,
                normals=bn,
                normal_dims=[0, 1, 2],
                colors=bc,
                shading="flat",
                layer=True,
            )
            aprint(f"added 'backdrop' ({len(bf)} house-shaded checkerboard triangles)")

            add_demo_caption(scene, "Mesh physical materials • Phases 1–2", None)

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
