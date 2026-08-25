# `luxar.shading` — appearance baked from geometry

Luxar's Points, Lines and GSplats are **emissive**: their shaders know nothing
about neighbouring geometry. A dense shell therefore accumulates into a flat glow
and the eye loses the shape — the fold structure of a fractal, the lobes of a
cloud, the interior of a light-sheet fit. This package computes the shape cues a
shaded renderer would get for free, so an author can write them into the scene.

## Emissivity as a function of ambient illumination

This is not decoration bolted onto the renderer — it is the missing half of the
transport the renderer already implements.

Emission–absorption rendering has two terms. Luxar's blending modes supply the
attenuation one: radiance is absorbed on its way *out* to the eye. The emission
term is the other, and for matter that is **lit from outside** rather than
genuinely glowing, the physically correct source is `albedo × incident
irradiance` — and the incident irradiance at a point is exactly what ambient
occlusion measures. So an emissive scene is best read as one whose *emissivity is
a function of ambient illumination*, and this package computes that function.

Three consequences, which is why the API looks the way it does:

- The result belongs multiplied into the **emission** (colour × intensity), never
  into opacity or absorption — those are the *other* term.
- It **composes with** an absorbing blending mode rather than double-counting it.
  `volumetric` answers "what is in front of what", which changes as the camera
  moves; occlusion answers "how much environment can reach here", which does not.
- **`strength` is physical, not taste.** `1 - strength` is the *indirect* ambient
  — multiply-scattered light that reaches even a fully enclosed point. It is the
  same quantity `demo_volumetric_cloud` spends three tuned radiance terms on, and
  that `demo_mandelbulb` writes as its ambient floor of 0.32.

## The dividing line

> **Bake what the geometry knows. Leave to the shader what the camera knows.**

Ambient occlusion is a scalar function of the geometry alone — *how enclosed is
this element* — so the value computed once offline is correct from every camera.
That is what makes it safe to bake.

A **key light is not offered here, on purpose.** Baking one fixes it in world
space, so orbiting to the unlit side darkens the scene for no reason. A key light
has to follow the viewer, which makes it a material concern; the mesh shader
already does this properly with a view-space key direction.

This package also sits on the *appearance* side of Luxar's data/appearance split.
A `.gsplats.zarr` is a reconstruction; colormaps, tone mapping and occlusion are
authored on the way into a scene. Baking occlusion here rather than into the
gsplat store keeps it from becoming another per-element sidecar that `reencode`,
`lod`, `decimate` and refits have to carry, reorder or invalidate — and it means
the caller still holds the scene's `Dimensions` and can say which axes are
spatial.

## Quick Start

```python
import numpy as np
from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.shading import bake_ambient_occlusion

shade = bake_ambient_occlusion(positions, mass=amplitudes)

with LuxarZarrCompiler(out) as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats(
        "Specimen",
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky,
        colors=(base_colors * shade[:, None]).astype(np.float32),
        blending_mode="volumetric",
        layer=True,
    )
```

`shade` is a **multiplier** in `[0, 1]`: `1.0` out in the open, lower where an
element is enclosed. Multiplying it into linear-light colour is the zero-machinery
route available today. Carrying it as a separate per-element attribute so the
viewer can scale it live is the better long-term contract, and needs format,
loader and shader work that does not exist yet.

## Knobs that matter

| Argument | Why you would touch it |
|---|---|
| `radius` | **The** knob. The scale of structure AO responds to; defaults to 5% of the bounding-box diagonal. Too small and only the tightest creases darken; too large and the integral degenerates into a depth map. |
| `normals` | **Pass these if the data is a surface and you have them.** Switches from the full sphere to a cosine-weighted hemisphere. Roughly doubles the discrimination on shells — see below. |
| `mass` | Occluding material per element — **the only channel through which an element's own appearance enters** (see below). Defaults to ones, i.e. pure count density. |
| `group_by` | **Required for nD data.** Pass the timepoint index for a timelapse, or occlusion crosses the time axis and the whole sequence shades as one solid. The same hazard `--coarsen-dims` exists for on the LOD side. |
| `strength` | Scales the darkening. `0.0` returns all ones. |
| `n_directions` | Sphere directions averaged. Cost and memory are linear in it. |

Two things worth knowing before judging a result:

- **AO always lowers mean brightness**, so authored exposure has to absorb it —
  `shade.mean()` is the factor to compensate for. A frame that merely looks
  crisper may just be darker; compare against the unshaded original.
- `extinction="auto"` (the default) solves for the value that puts the median
  element at `AUTO_TARGET_TRANSMITTANCE`, which makes the same call work
  unchanged on a 50k-point sketch and a 3M-splat fit, whatever the mass units.
  That constant is the global contrast dial, and it was picked by measurement
  against two deliberately different shapes — see its docstring for the table.
  **Judge a lower target on the 5th percentile, not on the contrast number**: a
  target low enough to crush the dark end to black reports more contrast while
  showing less structure.

### What `mass` does and does not capture

Nothing here reads a node's radii, sharpness or opacity. Per-element appearance
enters **only** through `mass`:

| type | pass |
|---|---|
| GSplats | `amplitudes` (× per-splat alpha if RGBA) |
| Points | `radii ** 3` when radii vary; otherwise leave `None` |
| Lines | `widths ** 2 * segment_length` per vertex |
| Mesh | leave `None` — a vertex has no extent of its own |

Two things need no folding in. A **uniform** factor — node opacity, or a constant
radius or sharpness — cancels entirely, because `extinction="auto"` calibrates
against the population's own median. And **sharpness** only changes an element's
profile *shape*, a modest constant unless it varies per element.

One real approximation: mass is deposited at each element's **centre**, so extent
is a weight and not a footprint. That holds while the render radius is small next
to the grid cell (`extent / grid_cells`) — measured at 0.18, 0.36 and 0.44 of a
cell in the three bundled point demos. Raise `grid_cells` if your elements span
cells.

### Points need more of it than a mesh does

Worth knowing before copying a strength between geometry types. A shaded,
depth-tested surface puts **one** element in each pixel, so the full per-element
multiplier reaches the screen. A point cloud blends soft overlapping sprites, and
an additive one sums along the ray — a pixel then shows the *ray-averaged* shade,
which costs about 30% of the contrast (measured 0.280 additive against 0.403
nearest-element on the gyroid). So the same value that looks right on a mesh reads
as barely-there on points. The bundled demos land at `0.45` for mesh and `0.85`
for the point clouds for exactly this reason.

## Method

A windowed Beer–Lambert column integral over a spherical direction set. For each
direction: splat mass onto a grid aligned to it, integrate density along that axis
over a finite `radius` window (exclusive of the element's own cell), and take
`exp(-tau)`. The ambient term is the mean transmittance over all directions.

### Volumetric or surface? Pass `normals` if you have them

Averaged over the full sphere the integral needs **no surface normals**, and
nothing here invents any — normals estimated by local PCA are meaningless for a
volumetric point cloud, so the default simply does without.

But that default is the *volumetric* reading of occlusion, and it under-serves
surfaces. On a thin shell every element is surrounded by the same in-plane
material, which dominates the sphere average and washes the signal out. Supplying
normals switches to a cosine-weighted hemisphere. Measured against the
Mandelbulb's own distance-estimator AO over its 27k surface points:

| occlusion radius | full sphere | cosine hemisphere |
|---|---|---|
| 0.05 | +0.31 | **+0.61** |
| 0.20 | +0.44 | **+0.68** |

So: volumetric data (light-sheet fits, clouds, fractal interiors) → leave
`normals` alone. Surfaces where you already hold orientations (mesh normals,
marching-cubes gradients, a distance-field gradient) → pass them. Either way the
result is view-independent; a normal describes the surface, not the camera.

The **finite window** is what makes this occlusion rather than a depth map. An
unbounded integral reports how deep an element sits inside the whole object, so
two elements at equal depth read identically however differently shaped their
surroundings are — exactly the flat-sheet-versus-crevice confusion AO exists to
resolve. `directional_optical_depth(..., radius=None)` is the unbounded form, and
is the right tool only when the direction is a real light that belongs to the
subject.

Three details differ from the same integral written for a single light, and all
three are corrections rather than preferences:

- The anti-banding blur runs **only across the two axes transverse to the ray**.
  Blurring *along* it smears an element's own mass into the adjacent slices, which
  the integral then counts — so every element partly occludes itself and the whole
  field picks up a constant pedestal that has to be dialled back out with a magic
  strength and floor. Transverse-only keeps the self-exclusion exact; an open-space
  probe measures exactly zero.
- The **cell size is fixed once** from the data extent and reused for every
  direction. Deriving it per direction from that direction's own rotated bounding
  box — the natural thing to write when there is only one direction — makes each
  direction integrate at its own scale and biases the mean.
- That blur **clamps at the grid edge** instead of wrapping, so mass on one face
  of the object cannot leak onto the opposite one.

The column is also normalized into units of *cells of typical material
traversed*, which is what keeps `extinction` an O(1) knob instead of a
per-dataset constant in the thousands that shifts with the element count. It stays
a **sum** over the window and not a mean: dividing by the window would make a
one-cell-thick sheet — which is what a surface-sampled point cloud is made of —
block only `1/window` of the light, and block less the wider the radius.

Reference: the transmittance-toward-a-direction formulation follows Harris &
Lastra (2001), *Real-Time Cloud Rendering*, computed on a grid rather than by
rendering from the light's point of view — which suits a caller that already holds
every element as an array.

## Related

- `packages/luxar/src/luxar/demos/demo_mandelbulb.py` — bakes distance-field AO
  from the fractal's own distance estimator. Strictly better than a density field
  *when you have an analytic oracle*, and inapplicable when you do not.
- `packages/luxar/src/luxar/demos/demo_volumetric_cloud.py` — an unbounded
  optical-depth integral toward a sun, plus sky and ground terms. Correctly
  demo-local: its sun is part of the subject, and its density normalization is
  tied to how the parcels were seeded.
- `packages/luxar-viewer/src/rendering/materials/mesh/appearance.ts` — the mesh
  material's view-space key light, i.e. the other side of the dividing line.
