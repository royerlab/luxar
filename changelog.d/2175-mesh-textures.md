#### Mesh textures

Mesh gains UV coordinates and a texture, as a third mutually-exclusive base
colour alongside per-vertex colours and a colormap LUT. Two payload shapes behind
one `texture_encoding` attr: `raw` `(H, W, C)` arrays including HDR float, and
`png`/`webp`/`jpeg` bytes in a uint8 array — the same shape `image_label_bytes`
already uses, so encoded textures reuse a shipped mechanism rather than adding
loose files that `content_hash` would not cover.

The texture is sampled **per fragment**, which is the structural difference from
the colormap LUT. That LUT is a vertex-stage lookup, and for a LUT that is a fair
model — one scalar per vertex, so interpolating the resulting colour approximates
interpolating the scalar. An image has structure *between* vertices, so a
per-vertex fetch would resolve exactly one texel per vertex and reproduce the
point-cloud limitation this exists to remove.

`shading` gains a third arm, `"none"`. An unlit mesh is what every other Luxar
geometry type already is (the other three are purely emissive) and what a data
basemap needs, since a view-anchored key makes a colour-coded surface read
differently as the camera moves. Never a default. Under it no normal is computed
at all — not computed and multiplied by zero: the `normal` attribute stays out of
the vertex layout and the derivative pair is never evaluated.

Texture alpha multiplies coverage, so an RGBA texture gets real cutout holes under
`opaque` — and the pick pass samples the texture too, because otherwise a hole the
user can see stays pickable and depth-occluding.

`texture_filter` and `texture_wrap` are authorable, defaulting to linear with
mipmaps and to repeat-in-u / clamp-in-v. That asymmetry is deliberate: longitude
is periodic and must wrap for an equirectangular dateline seam to close, latitude
is not, and a wrapping `v` bleeds the north pole into the south.

Two limits are enforced at authoring time rather than left to fail at load. A
texture over 16384 pixels on either axis is refused, because a GPU silently
*clamps* a larger one and renders the wrong image with no diagnostic; and a
texture whose decoded surface alone exceeds the viewer's per-node budget is
refused too, since `16000x16000` sits inside the per-axis limit and still decodes
to 977 MiB. To go beyond one texture, split the surface across several mesh nodes
— a partition cannot carry one, because a part re-indexes vertices while the image
is node-level.

#### The four Earth globes are textured meshes

`demo_earthquakes_3d`, `demo_ocean_currents_earth`, `demo_global_rivers_earth` and
`demo_biodiversity_planetary_scale` rendered the planet as point clouds — two of
them at 8M points — because a point cloud resolves a texture at roughly one sample
per point. They are now UV spheres with a 16384x8192 Blue Marble basemap across
multiple nodes (four for the relief globe), at two to three orders of magnitude fewer elements, and the element
budget goes back to the data being visualised.

Each carries a real cloud deck from NASA's Blue Marble composite at altitude,
where the texture's alpha *is* the cloud fraction. Rivers uses 15x relief
exaggeration as a displaced mesh with smooth shading, and its cloud shell's
altitude is derived from the relief's own maximum — Everest sits at about 2.1%
of the radius there.

Every translucent layer over a globe — cloud deck, current ribbons, sea surface —
uses `luminous` blending. That is additive **and** depth-tested, which is the
distinction that matters: the far side stays hidden behind the opaque planet, so
nothing is given up relative to `normal`. What is gained is that additive
composition is commutative, so a scene with 11M current segments plus a cloud
shell has no depth-order to get wrong; and on the currents the accumulation is
informative, since a boundary current concentrates flow and therefore brightens.

Note that `strength` means something different under the two modes: alpha-over at
0.22 is a lerp 22% of the way to white whatever is behind, while additive at 0.22
is an addition of 0.22 that tone mapping then compresses along with the rest of
the highlights. Additive needs more amplitude for the same apparent density.
