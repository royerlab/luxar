"""Appearance baked from geometry at scene-authoring time.

Luxar's emissive geometry types — Points, Lines, GSplats — have shaders that know
nothing about neighbouring geometry, so shape cues that a shaded renderer would
get for free have to be computed by the author and written into the scene. This
package holds those bakes.

The dividing line is deliberate and worth stating once: **bake what the geometry
knows, leave to the shader what the camera knows.** Ambient occlusion is a scalar
function of the geometry alone, so the value computed offline is correct from
every camera and belongs here. A key light depends on a direction relative to the
viewer, so baking one fixes it in world space and it stops reading the moment the
camera orbits — that belongs in a material (see the mesh shader's view-space key),
not in a store.

This also sits on the appearance side of Luxar's data/appearance split. A
``.gsplats.zarr`` is a *reconstruction*; colormaps, tone mapping and occlusion are
authored on the way into a scene. Keeping the bake here rather than in the gsplat
toolbox avoids making it another per-element sidecar for ``reencode``, ``lod``,
``decimate`` and refits to reorder or invalidate, and means the caller has the
scene's :class:`~luxar.core.dimensions.Dimensions` in hand to say which axes are
spatial.

Quick start
-----------
::

    from luxar.shading import bake_ambient_occlusion

    shade = bake_ambient_occlusion(positions, mass=amplitudes)
    colors = (base_colors * shade[:, None]).astype(np.float32)
"""

from .occlusion import (
    bake_ambient_occlusion,
    directional_optical_depth,
    sphere_directions,
)

__all__ = [
    "bake_ambient_occlusion",
    "directional_optical_depth",
    "sphere_directions",
]
