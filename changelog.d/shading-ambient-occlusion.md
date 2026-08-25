#### `luxar.shading`: bake ambient occlusion for the emissive geometry types

Points, Lines and GSplats are emissive — their shaders know nothing about
neighbouring geometry, so a dense shell accumulates into a flat glow and the eye
loses the shape. Three demos had each hand-rolled a fix for this independently
(`demo_mandelbulb` from its distance estimator, `demo_volumetric_cloud` from an
optical-depth grid, `demo_lsystem_forest` from vertex normals). The reusable part
of that is now a package.

`bake_ambient_occlusion(positions, ...)` returns a per-element multiplier in
`[0, 1]` computed as a windowed Beer–Lambert column integral over a spherical
direction set. It is deliberately **ambient only**: occlusion is a scalar
function of the geometry, identical from every camera, which is what makes baking
it sound. A directional key light is not offered, because baking one fixes it in
world space and it stops reading the moment the camera orbits — a key light has
to follow the viewer, which is a material concern, and the mesh shader already
handles it that way in view space.

The finite occlusion `radius` is what makes this occlusion rather than a depth
map: an unbounded integral reports how deep an element sits inside the whole
object, so a flat sheet and a crevice at the same depth score identically. Three
choices follow from the integral being ambient rather than lit — the anti-banding
blur runs only across the axes transverse to each ray, so no element occludes
itself; the cell size is fixed once from the data extent and reused for every
direction, so no direction integrates at a coarser scale than another; and that
blur clamps at the grid edge rather than wrapping, so mass on one face cannot
leak onto the opposite one. `extinction` defaults to `"auto"`, which calibrates
the scale so the same call works unchanged on a 50k-point sketch and a 3M-splat
fit, whatever the mass units.

Normals are optional and never invented. Over the full sphere none are needed,
which is the right reading for volumetric data and avoids estimating normals by
local PCA where they would be meaningless. Where the subject really is a surface
and the caller already holds them, passing them switches to a cosine-weighted
hemisphere and roughly doubles the discrimination: scored against the
Mandelbulb's own distance-estimator AO over its surface points, correlation rises
from `+0.31` to `+0.61` at a 0.05 radius and from `+0.44` to `+0.68` at 0.20.

Occlusion is baked at scene-authoring time and stays out of the gsplat toolbox on
purpose. A `.gsplats.zarr` is a reconstruction; colormaps, tone mapping and now
occlusion are authored on the way into a scene. Keeping it here avoids adding
another per-element sidecar for `reencode`, `lod`, `decimate` and refits to carry,
reorder or invalidate, and leaves the caller holding the scene's `Dimensions` so
it can say which axes are spatial — `group_by` then keeps occlusion from crossing
a time or channel axis, the same hazard `--coarsen-dims` exists for on the LOD
side.

Three demos exercise it. A new `ambient_occlusion` demo puts the same gyroid
surface on screen three times — unshaded, full-sphere, cosine-hemisphere — with
identical colour, radii and blending, so the only variable is the baked
multiplier; its exposure is derived from the measured deepest sightline rather
than hardcoded, because a fixed gain clips at high `--resolution` and the clipping
would hide the very term the demo is about. `mesh_isosurface_cells3d` gains
per-vertex occlusion from the marching-cubes normals it already computed, baked
per channel so an independently-toggled layer never wears its neighbour's
shadows. `nuclear_pore_complex` gains burial shading: occlusion over a sphere of
directions is a close correlate of how much solvent could reach an atom, so the
darkening tracks a real property rather than decorating one — and because CPK is
a categorical hue encoding, a scalar multiplier changes lightness while leaving
element identity readable. Both demos normalize against the term's own maximum,
so occlusion is spent on contrast instead of dimming an authored exposure.

`atp_synthase` gains the same burial shading. It renders `volumetric`, which
might look like double-counting, and the reason it is not is the framing the
package is built on: **emissivity is a function of ambient illumination.**
Emission–absorption transport has two terms; the blending mode supplies the
outgoing attenuation, and for matter lit from outside rather than glowing the
emission term is `albedo × incident irradiance` — which is exactly what ambient
occlusion measures. In-scattered source and outgoing attenuation are halves of
one equation, not two darkenings. That also fixes what `strength` means:
`1 - strength` is the indirect, multiply-scattered ambient that reaches even a
fully enclosed element, the same quantity the cloud demo spends three tuned
radiance terms on and the mandelbulb writes as its ambient floor of 0.32.

The multiplier is premultiplied into linear-light colour, which is the only route
available today. Carrying it as a per-element attribute so the viewer could scale
it live is the better contract, and needs format, loader and shader work that does
not exist yet — the new demo sidesteps that entirely, since a fixed side-by-side
wants each panel's answer baked in anyway.

The auto calibration's target sits at 0.25 rather than a timider 0.5, picked by
measuring contrast and the 5th percentile across two deliberately different
regimes (a thin shell read through the hemisphere path, a solid ball read through
the full sphere) so the default is not tuned to one shape. It roughly doubles
contrast in both while keeping p5 near 0.2; below it the gain comes from clipping
the dark end rather than from revealing structure. Points then take a higher
`strength` than mesh — 0.85 against 0.45 — because a depth-tested surface puts
one element in each pixel while a point cloud blends soft overlapping sprites,
and an additive one shows the ray-averaged shade, costing about 30% of the
contrast. Mesh was measured and deliberately left where it was.
