#!/usr/bin/env python3
"""Mesh Glass Lens Example — glass that refracts the data behind it.

This example demonstrates:
- ``add_mesh(..., material="physical", transmission=1.0, refract_data=True)``: a
  glass mesh that draws AFTER the emissive data and refracts it (spec
  ``MESH_PHYSICAL_MATERIALS_SPEC.md`` §3.4, Phase 3).
- The Phase 2 default beside it: the same glass WITHOUT ``refract_data`` draws
  first, so the data behind it stays crisp and unrefracted.
- A tinted bubble: ``attenuation_color`` with ``attenuation_distance`` colours what
  the lens transmits.
- The documented limit: emissive layers write no depth, so data IN FRONT of a
  refracting glass is painted over — orbit until lattice rows cross in front of the
  lens to see it.

Educational value:
- A regular point lattice makes refraction legible: straight rows bend and magnify
  through the lens, and the two spheres side by side show exactly what the flag
  changes and what it does not (compositing, opacity and picking are the same).
- Faint, non-overlapping points are used on purpose: additive data brighter than
  1.0 clips to white under tone mapping, and a lens of saturated data is a white disk.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.mesh.primitives import icosphere
from luxar.utils.paths import get_examples_output_dir

#: Icosphere refinement: 642 vertices / 1280 faces reads as a sphere at this size.
SUBDIVISIONS = 3
#: Sphere radius in scene units.
RADIUS = 1.2
#: The lattice behind the spheres: an N×N×3 grid of small points.
LATTICE_N = 15
LATTICE_HALF = 4.5
LATTICE_Z = (-5.0, -3.5, -2.0)
POINT_RADIUS = 0.07
#: Below the tone-mapping knee, so isolated points keep their colour through glass.
LATTICE_INTENSITY = 0.8
#: The three spheres, in front of the lattice (the camera looks down -z).
SPHERE_Z = 1.0
SPHERE_X = (-3.0, 0.0, 3.0)


def lattice_points() -> tuple[np.ndarray, np.ndarray]:
    """A regular lattice of points with a colour ramp along x, as float32."""
    axis = np.linspace(-LATTICE_HALF, LATTICE_HALF, LATTICE_N, dtype=np.float32)
    xs, ys, zs = np.meshgrid(axis, axis, np.asarray(LATTICE_Z, dtype=np.float32))
    positions = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(
        np.float32
    )
    t = (positions[:, 0] + LATTICE_HALF) / (2.0 * LATTICE_HALF)
    warm = np.array([1.0, 0.6, 0.25], dtype=np.float32)
    cool = np.array([0.35, 0.8, 1.0], dtype=np.float32)
    colors = (1.0 - t)[:, None] * warm[None, :] + t[:, None] * cool[None, :]
    return positions, colors.astype(np.float32)


def create_scene(output_path) -> None:
    """Write the lattice + three glass spheres to ``output_path``."""
    vertices, faces, normals = icosphere(SUBDIVISIONS, radius=RADIUS)
    positions, colors = lattice_points()
    white = np.tile(np.array([[1.0, 1.0, 1.0]], dtype=np.float32), (len(vertices), 1))
    amber = np.tile(np.array([[1.0, 0.92, 0.7]], dtype=np.float32), (len(vertices), 1))

    def at(x: float) -> np.ndarray:
        return vertices + np.array([[x, 0.0, SPHERE_Z]], dtype=np.float32)

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(),
            viewer_config=ViewerConfig(
                tone_mapping="ACES",
                # Straight on and a little above, so every sphere has lattice behind it.
                camera=CameraConfig(
                    position=(0.0, 1.5, 12.0),
                    target=(0.0, 0.0, -1.0),
                    up=(0.0, 1.0, 0.0),
                ),
            ),
        )
        scene.add_points(
            "lattice",
            positions,
            colors=colors,
            radii=np.full(len(positions), POINT_RADIUS, dtype=np.float32),
            blending_mode="additive",
            intensity=LATTICE_INTENSITY,
            layer=True,
        )
        # Phase 3: the lens draws after the lattice and refracts it.
        scene.add_mesh(
            "lens",
            at(SPHERE_X[0]),
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            colors=white,
            shading="smooth",
            material="physical",
            roughness=0.02,
            transmission=1.0,
            ior=1.5,
            thickness=RADIUS * 2.0,
            refract_data=True,
            layer=True,
        )
        # Phase 2: the same glass without the flag — the lattice stays crisp on top.
        scene.add_mesh(
            "glass_first",
            at(SPHERE_X[1]),
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            colors=white,
            shading="smooth",
            material="physical",
            roughness=0.02,
            transmission=1.0,
            ior=1.5,
            thickness=RADIUS * 2.0,
            layer=True,
        )
        # A tinted bubble: what it transmits is coloured by the volume it crosses.
        scene.add_mesh(
            "amber_bubble",
            at(SPHERE_X[2]),
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            colors=amber,
            shading="smooth",
            material="physical",
            roughness=0.05,
            transmission=1.0,
            ior=1.4,
            thickness=RADIUS * 2.0,
            attenuation_color="#f6d148",
            attenuation_distance=RADIUS * 0.9,
            refract_data=True,
            layer=True,
        )
        add_explainer(
            scene,
            title="Glass that refracts the data",
            body="Three physical glass spheres in front of a point lattice. The left "
            "lens and the right amber bubble are authored with "
            "<code>refract_data=True</code>: they draw after the data and bend it. "
            "The middle sphere is the Phase 2 default: the lattice stays crisp on top.",
            observe=[
                "Lattice rows bend and magnify through the left lens, inverted.",
                "Through the middle sphere the same rows stay straight and sharp.",
                "The amber bubble tints what it transmits.",
                "Orbit until rows pass IN FRONT of a lens: they are painted over — "
                "emissive layers write no depth, which is why the flag is opt-in.",
                "Layers panel (L): the Refract data switch flips a sphere live.",
            ],
        )

    aprint(f"Created glass lens example: {output_path}")


def main() -> None:
    output_path = get_examples_output_dir() / "mesh_glass_lens_example.luxar.zarr"
    create_scene(output_path)


if __name__ == "__main__":
    main()
