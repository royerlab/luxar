---
name: luxar-visualization
description: >-
  Build a Luxar visualization for a dataset. Use when a user wants to turn data
  (point clouds, trajectories/lines, nD images, volumes, time series, multichannel
  stacks) into a .luxar.zarr scene and view it in the web viewer. Covers the
  LuxarZarrCompiler → create_scene → add_points/add_lines/add_gsplats →
  serve/export flow, Dimensions, transforms, hierarchy, baked ambient occlusion
  (luxar.shading) for emissive geometry that reads flat, and the demo/example
  patterns that are the canonical source of know-how.
---

# Build a Luxar visualization

Luxar compiles nD scientific scenes (Points, Lines, GSplats, Mesh) to a `.luxar.zarr`
archive served to a WebGL viewer. This skill builds a scene from a dataset.

**The repo's demos and examples are the canonical know-how** — when in doubt, read a
matching one before writing code:
- `packages/luxar/src/luxar/demos/demo_*.py` (89 complete demos)
- `packages/luxar/examples/*_example.py` (53 focused examples)

## The canonical pattern (every demo follows this)

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions
from luxar.demos import launch_viewer   # convenience: shells to `luxar serve --viewer`

output_path = "my_scene.luxar.zarr"

with LuxarZarrCompiler(output_path) as compiler:        # 1. open the compiler
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())  # 2. dims REQUIRED
    scene.add_points(                                   # 3. add geometry nodes
        "MyData",
        positions,                                      # (N, 3) float32
        colors=colors,                                  # (N, 3) float32/uint8
        radii=radii,                                    # (N,) or scalar
    )
# 4. context exit finalizes the store

launch_viewer(output_path)                              # 5. view it
```

That is the whole shape. Build data → `LuxarZarrCompiler` → `create_scene(dimensions=)`
→ `add_points` / `add_lines` / `add_gsplats` → serve. See `references/scene-api.md`
for every method signature, `**attrs`, transforms, and overlays.

## Pick the geometry type

| Data | Node | Required arrays |
| --- | --- | --- |
| Point cloud, particles, cells | `add_points` | `positions` (N,D) |
| Trajectories, skeletons, vectors, meshedges | `add_lines` | `vertices` (N,D) + `widths` |
| Triangle surfaces, isosurfaces | `add_mesh` | `vertices` (V,D) + `faces` (F,3); optional normals (with `normal_dims`), colors, scalars |
| Volumes / dense nD images (fitted to Gaussians) | `add_gsplats*` | centers/amplitudes/cholesky, or a fitted `.gsplats.zarr` |

For raw volumes, fit Gaussian splats first — see the **`luxar-gsplat-pipeline`**
skill, then `add_gsplats_from_file` / `add_gsplats_from_volume` here.

## Dimensions = the coordinate system

`Dimensions` is the single source of truth for the scene's axes. Displayed dims are
the viewport; non-displayed dims (time, channel, z) become nD slider axes.

```python
from luxar import Dimensions, Dimension

# 3D spatial
dims = Dimensions.default_3d()

# 4D time series: t is a non-displayed, discrete slider axis
dims = Dimensions([
    Dimension("t", unit="s", display=False, discrete=True, range=(0, 100), step=1),
    Dimension("z", unit="um", display=True),
    Dimension("y", unit="um", display=True),
    Dimension("x", unit="um", display=True),
])
```

`positions` columns must match the dimension order. Use `extend_to_all=` on a node to
broadcast geometry across a non-displayed dim (e.g. show the same points at every t).

## Common recipes

```python
# Scalars -> colormap (instead of explicit colors)
scene.add_points("field", pts, scalars=values, colormap="viridis")

# Lines: polyline trajectory with tapering width
scene.add_lines("track", verts, widths=w, line_type="polyline", blending_mode="additive")

# Hierarchy: a group with a transform; children compose under it
from luxar import transforms
g = scene.add_group("left", transform=transforms.translate(x=-50))
g.add_points("nuclei", pts)

# Fitted gsplats from a volume, in one step
scene.add_gsplats_from_volume("embryo", volume, seeds=8000, n_iters=5000, device="cuda")

# On-screen label
scene.add_text("Embryo, t=0", position=(0.02, 0.02), font_size=0.05, anchor="top-left")
```

## Demo helpers (for scripts that fit/serve)

From `luxar.demos` (the barrel that re-exports these — the single spelling; never
import the owning `luxar.demos._support.*` modules directly from a demo):
- `launch_viewer(output_path, open_browser=True)` — serve via the CLI viewer.
- `parse_demo_flags()` — standard `--recompute` / `--no-serve` / `--serve-only` flags.
- `load_precomputed_gsplats(demo_name, file_names, recompute=...)` — load cached
  (Git-LFS) fitted splats, refitting only when `--recompute`.
- `create_lorenz_attractor`, `create_random_spheres`, `create_time_series_demo`,
  `detect_device`, `warn_if_no_cuda_gpu`.

From `luxar.utils.paths`: `get_demos_output_dir()`, `get_examples_output_dir()` for
output locations (never commit `.luxar.zarr`).

### Shipping a demo's data

`ensure_dataset("<key>")` — also from the `luxar.demos` barrel — resolves a demo's
payload cache → in-repo (git-LFS) → Zenodo, verifying the manifest sha256 at every
step, and returns the paths. The key comes from
`packages/luxar/src/luxar/demos/data_manifest.json`. It only resolves `zenodo`
keys: a `local-compute` / `regenerate` one raises `LocalComputeDataset` instead,
so those demos have to build their data themselves.

**That manifest is GENERATED — never hand-edit it.** Declare the dataset in
`scripts/gen_data_manifest.py` (bucket, record, license, source, attribution) and
run `make gen-data-manifest`; in-repo files are measured from the tree, while a
`zenodo` entry's `sha256` and byte count pin the published record. Legacy
`hosted_sha256` metadata is carried forward when present because it cannot be
derived from the tree.
A hand-edited metadata field always fails
`test_committed_manifest_matches_generator`; a hand-edited checksum does not when
the payload directory is absent or empty in your checkout, since the generator
then reuses the committed file list verbatim rather than wiping it. Re-serializing
it yourself with the wrong `indent` rewrites all ~600 lines into diff noise (the
generator uses `indent=2`). Replacing a payload in place is then just: copy the
new file over the LFS-tracked one, regenerate, and check the diff is only the
sha256/bytes lines.

## Appearance is authored, not defaulted

The Layers panel (press `L`) is where you dial a scene in — but it rewrites the
node's uniforms at every load, so whatever you tune there must be written back into
the `add_*` call or it is lost. Five things about that round-trip surprise people:

- **The display window is stored as an `intensity`/`offset` PAIR, not a gain.** For
  a window `[lo, hi]`: `intensity = 1/(hi-lo)`, `offset = -lo/(hi-lo)`. Passing the
  panel's max straight into `intensity=` stores a window `hi²` times too narrow and
  the scene renders blown out. Read it back the same way:
  `lo = -offset/intensity`, `hi = (1-offset)/intensity`.
- **Amplitudes MUST be normalised into `[0, ~1]` before the node enters the
  scene, and no viewer control can substitute for it.** A fitted
  `.gsplats.zarr` stores amplitudes in RAW SOURCE UNITS — the fitter multiplies
  its `[0,1]` working copy back out by the volume's intensity range, so a fit
  from a uint16 detector stack carries detector counts, in the hundreds or
  thousands. In the shader the stored amplitude does two jobs and the display
  window only reaches one of them: it picks the colormap LUT index
  (`t = clamp((A - min) * scale, 0, 1)` — clamped, so it selects a *colour* and
  can never scale brightness), and it sets **emitted radiance and, under
  `volumetric`, optical depth** (`tau = absorption * opacity * intensity` with
  `intensity ∝ A`), which nothing windows. An amplitude of 800 therefore emits
  1000x the radiance of 0.8 and saturates `1 - exp(-tau)` to an opaque shell.
  `add_gsplats_from_data` / `add_gsplats_from_file` /
  `add_gsplats_from_volume` / a graft now normalise by default (robust p99.9 →
  1.0, one factor for the whole structure, a no-op on data already in range).
  A child inserted directly into a `kind=lod` or `kind=partition` group instead
  defaults to no normalisation so its exposure stays shared with its siblings;
  pass `normalize_amplitudes=True` to override that rule or `False` to preserve
  raw units elsewhere. Note this is the *opposite* of what this bullet used to
  claim — the viewer does **not** normalise by the stored maximum.
- **`opacity` is the exposure lever, and with normalised amplitudes it lives in
  its natural range.** That is the point of normalising: against raw counts
  `opacity` has to carry a ~1/500 factor on a `[0,1]` control, which is
  unauthorable in the panel and is why so much older shipped appearance is a
  magic tiny constant.
- **For a colormapped node, the display window is NOT an exposure lever, and
  reaching for it first is the classic wrong turn.** Under `volumetric` (or any sum
  projection) a pixel accumulates along the ray, while the window only picks each
  element's LUT index. The symptom that tells the two apart: if widening the window
  *dims the whole object toward the colormap's dark foot and shrinks its footprint*
  rather than spreading it across the LUT, you are over-accumulated and want
  `opacity`. A frame whose bright regions are genuinely clipped flat is the
  window's problem. With explicit `colors=`, the pair is a direct color gain and
  offset: a `[0, hi]` window is pure pre-gamma gain. Under additive at
  `gamma=1`, it scales RGB like lowering `opacity`; under `volumetric`, `opacity`
  also lowers optical depth and coverage. Any nonzero `lo` carries an offset that
  shifts the authored colors. Prefer `opacity`, which the panel round-trips as
  exposure rather than as a window.
- **`absorption` (volumetric blending) is optical depth and ACCUMULATES along the
  ray**, so the right value depends on how deep the object is, not on how bright it
  is. It is not portable between datasets: a value tuned on a 170 µm brain will
  over- or under-attenuate on a different one. Retune it, and remember the pair —
  raising absorption usually wants the window's max lowered with it.
- **A stored window only means anything for that exact store.** Any refit, rescale
  or re-encode moves it. Say so in a comment next to the constant.

Framing is authored too. The viewer's default fits the bounding box with margin,
which leaves an object small; bake a `CameraConfig` derived from the LOADED bounds
(never hard-coded numbers — the centres carry whatever origin the pipeline left):

```python
from luxar.core.viewer_config import CameraConfig, ViewerConfig
# visible_height(d) = 2·d·tan(fov/2);  visible_width(d) = that · aspect
# Fit at the NEAR FACE, then add the half-depth back:
# d = max((W/fill)/(2 tan(fov/2)·aspect), (H/fill)/(2 tan(fov/2))) + depth/2
scene = compiler.create_scene(
    dimensions=dims,
    viewer_config=ViewerConfig(
        tone_mapping="ACES",
        camera=CameraConfig(position=(cx, cy, cz + d), target=(cx, cy, cz), fov=47.0),
    ),
)
```

**In a bundled demo this exact call fails required tests.** Demos run under the
cinematic 35 mm preset, and `test_demos_cinematic_mode.py` refuses an authored
pose that pins `fov`/`fov_preset` or that is not visibly composed for the 63°
lens. It also requires literal `cinematic_mode=True` on every `ViewerConfig`
built under `luxar/demos/`. Leave the FOV unset, derive the distance at
`CINEMATIC_FOV_DEG` directly, and show the full wrapper:

```python
import math

from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG

half_fov = math.radians(CINEMATIC_FOV_DEG) / 2
d = (
    max(
        (W / fill) / (2 * math.tan(half_fov) * aspect),
        (H / fill) / (2 * math.tan(half_fov)),
    )
    + depth / 2
)
camera = CameraConfig(position=(cx, cy, cz + d), target=(cx, cy, cz))
scene = compiler.create_scene(
    dimensions=dims,
    viewer_config=ViewerConfig(
        cinematic_mode=True, tone_mapping="ACES", camera=camera
    ),
)
```

The guard also recognises an inline `position=pull_in(...)` when carrying over an
empirically tuned distance from another FOV. For a derived distance, importing
`CINEMATIC_FOV_DEG` or `framing_scale` vouches for the module and lets the pose use
a local variable normally.

Two decisions this formula makes explicit:

- **`fov` is VERTICAL**, so visible width scales with the LIVE viewport aspect
  while a baked distance cannot. A wide object therefore has no single distance
  that fills it everywhere: at the aspect you calibrate for it occupies `fill` of
  the width, a WIDER window leaves margin, a NARROWER one crops. (That is the
  direction — it is easy to write it backwards.) The design aspect is a declared
  calibration point, not a safety margin, so say which one you picked. Framing for
  a square viewport is the no-crop-ever choice; the viewer default uses that same
  shorter-axis rule with `fill = 0.75`, while an explicit camera can choose a
  tighter fill and a declared landscape calibration aspect. Work out both numbers
  before choosing.
- **Fit at the NEAR FACE, not the target plane.** This matches the viewer default:
  the frustum narrows towards the camera, so a deep object's camera-facing side has
  the least room. Fit at `d - depth/2` and add the half-depth back; otherwise the
  near corners silently leave the frame.

A test that merely asserts the camera block *exists* is vacuous — a silent revert
to the default still loads a valid scene. Assert the geometry, and prefer one
assertion that re-derives nothing: project the eight bbox corners through the
camera and require every one inside the frustum. That catches the near-face error;
a test that repeats the distance formula cannot.

If a node's data sits diagonally in its own footprint, level it before authoring the
camera: the amplitude-weighted principal axis of the cloud in the view plane gives
the correction angle directly (`gsplat transform --rotate-z <-angle>`), and it can
shrink the bounding box dramatically — 483×508 → 663×303 µm on a fly brain, which is
the difference between framing the specimen and framing empty corners.

### Thin-line scenes need `allow_high_dpr=True`

The viewer renders at CSS resolution (device pixel ratio 1.0) by default, even on
a Retina display, because a 2x panel costs 4x the fragment work and soft-edged
emissive geometry barely rewards it. **Line-dominant scenes are the documented
exception — set `allow_high_dpr=True` on them.**

```python
viewer_config=ViewerConfig(
    cinematic_mode=True,
    allow_high_dpr=True,   # river networks / tractograms / wiring diagrams
)
```

Measured on a Retina panel against DPR 2, brightness and coverage hold to within
2.5% on every geometry type; the entire visible effect of the cap is a 15-35%
loss of high-frequency detail. On points and gsplats that reads as slightly
softer. On dense thin lines it reads as **mush** — individual rivers, streamlines
or edges stop being separable, which loses information rather than polish.

It is also the cheapest place to spend the pixels. Line scenes are not
fill-bound, so they gain least from the cap to begin with: a trajectory scene
measured 1.06-1.17x faster at DPR 1, against 2.6-2.7x for a point cloud. You buy
the detail back for almost no frame time.

Leave it off for points, gsplats and mesh unless a specific scene proves
otherwise — that is where the default earns its keep. Mesh is the one worth
checking by eye, since it is shaded with hard silhouette edges rather than soft
sprites, and the "emissive geometry barely rewards it" argument does not cover it.

## Baked ambient occlusion (`luxar.shading`)

**Reach for this when a dense scene renders as an even glow and the shape is
gone.** Points, Lines and GSplats are emissive — their shaders know nothing about
neighbouring geometry, so there is no shading term at all and no exposure tweak
will put one there. `bake_ambient_occlusion` computes the missing one offline.

```python
from luxar.shading import bake_ambient_occlusion

shade = bake_ambient_occlusion(positions, mass=amplitudes)   # volumetric data
shade = bake_ambient_occlusion(                              # a surface
    vertices, normals=normals, occluder="opaque", radius=4.0, strength=0.85
)
colors = (base_colors * shade[:, None]).astype(np.float32)   # multiply into EMISSION
```

The mental model that makes the knobs obvious: **emissivity is a function of
ambient illumination.** Emission–absorption transport has two terms; the blending
mode already supplies the outgoing attenuation, and for matter lit from outside
rather than glowing the emission term is `albedo × incident irradiance` — which is
what occlusion measures. So it multiplies into COLOUR, never into `opacity` or
`absorption`, and it *composes* with `volumetric` instead of double-counting it.

Choosing settings:

- **`occluder`** — `"density"` (default) for a medium; `"opaque"` for a **surface
  sampled as points**. Not cosmetic: under Beer–Lambert a one-cell-thick shell
  only attenuates by `exp(-k)`, so a *wall* passes about half the light, and no
  single `k` both blocks walls and spares thick regions.
- **`normals`** — pass them if you have them (mesh normals, marching-cubes or
  distance-field gradients). Roughly doubles the discrimination on a shell. Never
  invent them: local-PCA normals on a volumetric cloud are meaningless, and
  full-sphere is the correct reading there anyway.
- **`radius`** — the scale of structure it responds to, and the easiest thing to
  get wrong. **Size it against the geometry you actually built, not the science.**
  A demo whose radius came from the real complex's 120 nm ring, on a scaled-down
  25 nm model, sat at 23% of the object and measured *worse* contrast than a
  radius four times smaller — past a point the window stops reporting enclosure
  and starts reporting depth.
- **`group_by`** — required for nD. Pass the timepoint index or occlusion crosses
  the time axis and the whole sequence shades as one solid.
- **`strength`** — physical, not taste: `1 - strength` is the indirect,
  multiply-scattered ambient reaching even an enclosed element. Points want more
  than mesh (≈0.85 vs ≈0.45): a depth-tested surface puts one element in each
  pixel, while a point cloud blends overlapping sprites and an additive one shows
  the ray-*averaged* shade.

Three traps worth knowing before you judge a result:

- **AO always lowers mean brightness.** Dividing by the term's own maximum
  (`normalized = shade / shade.max()`) preserves peak brightness, not the mean;
  scale node intensity by `1 / normalized.mean()` if the authored mean exposure
  must survive.
- **A frame that merely looks crisper may just be darker.** Compare against the
  unshaded original, and judge a stronger setting on the 5th percentile, not the
  contrast number — crushing the dark end to black raises contrast while showing
  less.
- **Check your sprite radius before blaming the occlusion.** If the render radius
  is under half the sample spacing the sprites never touch, the surface renders as
  stipple, and that per-pixel noise drowns the gradient completely. This has
  already cost one debugging session.

`directional_optical_depth` is the lower-level primitive AO is built from — the
column toward ONE direction. Use it only when the direction is a real light that
belongs to the subject (`demo_volumetric_cloud`'s sun). It is **not** a general
appearance path: a single baked direction is locked to world space, so it stops
reading the moment the camera orbits. A key light has to follow the viewer, which
makes it a material concern — the mesh shader does it properly in view space.

Occlusion is authored at SCENE time and deliberately absent from the gsplat
toolbox: a `.gsplats.zarr` is a reconstruction, and a per-element sidecar there
would be one more thing for `reencode`/`lod`/`decimate`/refits to reorder or
invalidate. Working demos: `demo_exotic_surfaces` (surfaces, opaque),
`demo_nuclear_pore_complex` and `demo_atp_synthase` (burial shading on PDB atoms),
`demo_mesh_isosurface_cells3d` (per-vertex, on top of the mesh key light).

## Serve & export

```bash
luxar serve my_scene.luxar.zarr --viewer --open     # interactive viewer
luxar export my_scene.luxar.zarr -o out/ --open     # standalone offline folder
luxar info my_scene.luxar.zarr --stats              # inspect a built scene
```

## Gotchas (from the codebase)

- `create_scene` REQUIRES `dimensions=`.
- GSplat `cholesky_factors` are the packed lower-triangular factor **L of the
  covariance** (Σ = L·Lᵀ): 3D = `(N,6)` `[L00, L10, L11, L20, L21, L22]`. The
  diagonal is **scale-like** — isotropic std σ -> `[σ,0,σ,0,0,σ]`, NOT `1/σ`.
  This is the same contract documented by `Group.add_gsplats` and
  the `AdditiveSubLOD` docstring.
- `transforms.compose(t1, t2, t3)` applies `t1` FIRST.
- 4D/nD scenes can show 0 elements at a given slice — navigate to a populated slice,
  or use `extend_to_all` to broadcast across a non-displayed dim.
- To hand-author a stacked time/channel axis from lower-dimensional splats, use
  `dim_order=["x", "y", "z"]`, `fill={"time": t}`,
  `fill_sigma={"time": 0.0}`, and `extend_to_all=[]` (omitting the last argument
  auto-broadcasts the unmapped axis). Embedding regularizes the semantic zero to
  `1e-7`. For full scene-dimensional input without `dim_order` embedding, use a
  strictly positive diagonal smaller than the coordinate step; the writer rejects
  zero.
- Match `positions` column order to the `Dimensions` order.
- **Always pass `n_iters` to `add_gsplats_from_volume` / `fit_gaussian_splats`** (the
  example above uses 5000). The default is 1000 — *below* the CLI's lowest preset —
  and on thin structures it leaves splats at their isotropic seed shape and renders
  filaments as bead chains. See the `luxar-gsplat-pipeline` skill.
- **A demo that caches a fit must key the cache on every parameter that changes the
  result**, the optimizer schedule included. Seeds/floor/retention are the obvious
  ones, but the schedule changes splat SHAPES while leaving the COUNT identical, so a
  key blind to it silently returns the old fit and makes a retune look like a no-op.
  Fold the parameters into the filename (a short digest) rather than relying on
  anyone remembering `--recompute`.

See `references/scene-api.md` for exact signatures and `**attrs`. For volume→splats,
use the `luxar-gsplat-pipeline` skill.
