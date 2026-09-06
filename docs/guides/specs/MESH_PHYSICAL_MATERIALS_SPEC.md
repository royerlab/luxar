# Mesh Physical Materials Spec — glass, metal and iridescence on Luxar meshes

**Status:** Phase 1 (§4) is **implemented**; Phases 2–4 are design. Written after
the `esm3_protein_stories` demo wanted translucent marker shells around clusters
and got them from the existing light-free mesh model (see §2); this document is
the plan for the materials that model cannot express, and — for Phase 1 — the
record of what shipped and where it deviates from the sketch (the "Phase 1
implementation notes" in §3.1, §3.2, §3.5 and §3.6).

## 1. Motivation

Luxar's four geometry types are emissive by design: points, lines and splats
have no lighting at all, and the mesh (`MESH_NODE_SPEC.md` §6.2) shades with a
light-free, view-anchored key so a surface reads without any light object in the
scene. That is the right default for scientific data. It is also a ceiling: a
mesh cannot look like glass, brushed metal, a soap bubble or a lens, because all
of those are statements about how a surface reflects and transmits an
*environment*, and Luxar has no environment.

three.js has all of it in `MeshPhysicalMaterial` (`transmission`, `ior`,
`thickness`, `attenuationColor`, `dispersion`, `clearcoat`, `iridescence`,
`sheen`, `roughness`, `metalness`) and in its node twin
`MeshPhysicalNodeMaterial`, on both backends Luxar ships. The question is not
whether the shading exists but how it enters a light-free, additively blended,
post-processed, dual-backend viewer without breaking the contracts the four
types share. This spec answers that in phases.

## 2. What the current model already gives (baseline, shipped)

Before adding a material system, note what the existing knobs do, because a
surprising amount of "translucent marker" is already there:

- Per-vertex **RGBA** colours: alpha is per-vertex opacity, consumed by every
  blending mode.
- `blending_mode` ∈ `opaque | normal | additive | luminous | max` (no
  `volumetric` on meshes).
- `shading` ∈ `smooth | flat | none`, with the wrapped diffuse term
  `mix(ambient, 1, pow(dot(N, L)·0.5 + 0.5, shade_exponent))` under a fixed
  view-space key `L`, plus an additive Blinn–Phong highlight
  (`specular`, `shininess`). `ambient = 0` removes the floor so only the lit
  side of a shell shows; `shininess` shapes the highlight.
- `double_sided`, `layer_order` bands, and the nD vertex array (a marker can be
  pinned to a hidden-dimension slot like any other node).

The stories demo's marker spheres are exactly this: an icosphere per cluster,
story colour at alpha 0.10, `additive`, `ambient = 0`, `shade_exponent = 2`,
`specular = 0.35`, in a depth band between backdrop and highlight. What this
cannot do: Fresnel rim falloff (the key is fixed, not view-relative), refraction
of what lies behind, reflections, dispersion.

## 3. Design

### 3.1 Authoring: an opt-in `material` on `add_mesh`

```python
scene.add_mesh(
    "lens", vertices, faces, normals=normals, normal_dims=[0, 1, 2],
    colors=rgb,                       # base colour (vertex colours), as today
    material="physical",              # opt-in; default stays the house shader
    roughness=0.05, metalness=0.0,
    transmission=1.0, ior=1.5, thickness=0.4,
    attenuation_color="#f6d148", attenuation_distance=0.3,
    dispersion=0.5,
    clearcoat=0.0, iridescence=0.0,
)
```

`material` is a node attr with two values, `"luxar"` (default, today's shader)
and `"physical"`. The physical parameters are plain snake_case attrs validated
in `add_mesh` (ranges per three's documentation: `roughness`/`metalness`/
`transmission`/`clearcoat`/`iridescence` in [0, 1], `ior` in [1, 2.333],
`thickness` ≥ 0, `dispersion` ≥ 0, `attenuation_distance` > 0 or absent) and
written only when set. Unknown material attrs are refused, as `Waypoint.rendering`
refuses unknown keys: an audience is the wrong place to discover a typo.

A physical mesh keeps everything else a mesh has: nD vertices and the §5
slab cull, `layer_order`, `opacity`, labels/picking, partition and LOD groups,
`double_sided`. It drops the house-shader-only knobs (`shading`, `ambient`,
`shade_exponent`, `specular`, `shininess`, `blending_mode`, colormaps): those
are refused when `material="physical"`, not silently ignored.

**Phase 1 implementation notes** (what shipped; the contract where it differs
from the sketch above):

- The Phase 1 knob set is `roughness`, `metalness`, `clearcoat`,
  `clearcoat_roughness`, `iridescence`, `sheen` — each a finite float in
  `[0, 1]` — plus **`sheen_color`** (`"#rrggbb"`), which the sketch omitted:
  three's default sheen colour is black, so `sheen` alone renders nothing, and a
  knob that can never render anything is exactly the trap this validation exists
  to refuse. The viewer defaults `sheenColor` to white when unset for the same
  reason.
- **Phase 2 (delivered)** adds the glass family: `transmission` (`[0, 1]`), `ior`
  (`[1, 2.333]`), `thickness` (`>= 0`), `attenuation_color` (`"#rrggbb"`),
  `attenuation_distance` (`> 0`; absent = none) and `dispersion` (`>= 0`). One
  more pairing rule: `thickness`, `attenuation_*` and `dispersion` are refused
  without a `transmission` above zero, because three compiles them only inside
  its `USE_TRANSMISSION` block — written, they would be silently ignored. `ior`
  stands alone (it also sets an opaque surface's specular reflectance).
- **`shading` is kept, not refused**, except for `"none"`. Flat-versus-smooth
  normals is a property of any lit surface, not of the house key; it maps onto
  three's `flatShading`. Only the unlit `"none"` is meaningless on an
  environment-lit surface.
- **`alpha_cutoff` is mapped, not refused**: it becomes three's `alphaTest`, and
  on an RGBA mesh its presence selects the opaque cutout path over
  translucency — the meaning the house `opaque` mode gives the same pair.
- Refused under `material="physical"` (a `ValueError` naming the knob and why,
  from the mesh adder next to the volumetric refusal): `ambient`,
  `shade_exponent`, `specular`, `shininess`, `blending_mode`, `colormap`, a
  `texture`, `shading="none"`. A physical knob without `material="physical"` is
  refused too, as is an unknown `material` value. `material="luxar"` is accepted
  and written as given; an absent attr means the house shader.
- The vocabulary lives in `luxar.validation.types` (`MESH_MATERIALS`,
  `PHYSICAL_MATERIAL_ATTRS`, `HOUSE_SHADER_ONLY_ATTRS`) and every key is also in
  `MESH_ONLY_APPEARANCE_ATTRS`, so points, lines, splats and groups refuse it.
  (`Waypoint.rendering`, cited above as the style precedent, is not on `main`;
  the shipped refusals follow `validate_render_attrs`'s.)

### 3.2 Viewer: a third material family, not a variant

`createMeshNode` reads `attrs.material`. For `"physical"` it constructs three's
own material — `MeshPhysicalMaterial` on WebGL, `MeshPhysicalNodeMaterial` on
WebGPU — with `vertexColors = true`, `transparent = transmission > 0 || opacity < 1`,
`side` from `double_sided`, and the attrs mapped one to one. It is registered
in `material-manager/factories.ts` as its own entry (`meshPhysical`) rather than a
variant of `MeshMaterial`, because none of the house contracts apply to it:

- **No codegen snapshots.** The snapshot harness pins TSL Luxar writes. Three's
  own materials are three's to pin.
- **Picking** still uses the house `mesh-pick` material: picking renders
  geometry, not appearance, so a physical mesh picks exactly like a house mesh.
- **Depth sorting.** A physical mesh is not registered with the triangle sorter
  (`normal` mode's per-triangle sort is a house-shader feature). Three handles
  transmissive objects in its own render list order (§3.4).
- **Layer bands.** `renderOrder` from `layer_order` applies unchanged; three
  sorts within each of its lists by `renderOrder` first.

**Phase 1 implementation notes.** Three's material is wrapped, not used bare:
`PhysicalMeshMaterial extends MeshPhysicalMaterial` and
`PhysicalMeshTSLMaterial extends MeshPhysicalNodeMaterial`
(`rendering/materials/mesh-physical/`), sharing ONE attr→property mapping. The
wrapper exists because the viewer's generic paths assume a leaf-material surface:
`MaterialManager` broadcasts camera params, the Layers panel / exposure / LOD
cross-fade gate on `updateIntensity` + `updateGamma`, and a layer reset calls
`applyBlendingMode` (falling back to writing the HOUSE blend state when the
method is missing). So the wrapper implements exactly that surface with physical
meanings — `intensity` → a scalar base-colour gain, `offset` → emissive,
`gamma` recorded only, `applyBlendingMode` a no-op — and nothing more. It has
no camera surface (the near fade is a house-shader feature) and stamps
`userData.blendingMode = 'opaque'` only when opaque, so the depth-sort
coordinator releases it in every case while the `layer_order` band rank still
applies. Translucency is read off the data (`opacity < 1`, `transmission > 0`,
or per-vertex alpha without an `alpha_cutoff`); translucent surfaces do not
write depth. Vertex alpha is learned at the first commit
(`applyMeshVertexAlpha`), since the attrs record `has_colors` but not the
component count.

**Phase 2 implementation notes.** The knobs are ONE table
(`PHYSICAL_MESH_KNOBS` in `config.ts`: attr, three property, domain, default,
slider presentation, program-affecting flag) that the construction mapping, the
live Layers-panel sliders and the tests all read. `setPhysicalKnob` is the single
write path and flags `needsUpdate` when a program-affecting knob crosses zero:
three's WebGL setters version-bump on that crossing already, but
`MeshPhysicalNodeMaterial` stores plain properties and bakes `useTransmission`
into its lighting model at setup, so the rule lives in the shared helper for
both twins. Glass composites as translucent — the §3.4 contract made concrete: a
depth-writing shell would HIDE the cluster inside it, the worse failure for the
marker case — and stamps `userData.drawBeforeEmissive`, which the depth-sort
coordinator turns into "first within its `layer_order` band" (renderOrder −1
when unranked); see §3.4 for why that is needed on WebGPU.

### 3.3 Lighting: a scene environment, not scene lights

Physical materials render black without light. Luxar stays light-free in the
sense that matters (no light objects in the graph, nothing per-node), and gains
one scene-level input: `scene.environment`.

- **Default:** three's procedural `RoomEnvironment` through `PMREMGenerator`,
  built lazily the first time a physical mesh is created and cached on the
  `SceneManager`. It needs no asset, gives believable reflections and a neutral
  key, and is what three's own examples use.
- **Authored (Phase 4):** `viewer_config.environment = {"source": "room" |
  "scene" | "hdri", "probe": "auto" | [x, y, z] | "node:<path>", "resolution":
  128, "intensity": 1.0, "url": ...}`. `"room"` is today's default. `"hdri"`
  loads an equirectangular image from the store (the loader already fetches
  opaque files for overlays and textures). `"scene"` is the interesting one and
  is described next.
- `environmentIntensity` rides `viewer_config.environment.intensity`.

**Scene-derived environments (`"source": "scene"`).** Luxar's data is emissive,
so a capture of the scene *is* a radiance map of the data: a chrome sphere or a
glass shell reflects the cloud it sits in, and the light-free principle holds —
the data is the light, nothing is added to the graph. The capture is EXACT and
reuses the real shaders: a `CubeCamera` renders the scene from the probe point
into a cube render target, `PMREMGenerator.fromCubemap` prefilters it, and the
result goes onto `scene.environment` through the same lazy
`SceneEnvironment.ensure()` seam the room uses. An approximate alternative
(projecting the resident emitters into a small equirect on the CPU, no render)
was considered and **rejected**: it would re-implement the appearance pipeline
(colormaps, intensity, opacity, gamma, textures, LOD energy compensation) a
second time and keep it in step with the shaders forever — the two-copies drift
this codebase is organised to make impossible — for a saving that is only real
per frame, and the capture is never per frame.

Four details make the exact capture cheap and correct:

- **Trigger, not cadence.** Capture on camera settle and on data commit (and on
  slice change), at 64–128 pixel faces. Six low-resolution draws a few times a
  minute; never per frame unless the probe moves.
- **Sprite sizes.** Point and line footprints come from the camera params the
  material manager broadcasts (fov, resolution). The capture pushes the cube
  camera's params before the six renders and restores them after.
- **No self-reflection.** Physical meshes are masked out of their own capture
  with the cube camera's `layers` bit (a mesh that reflects a stale copy of
  itself is the failure mode). Blending the room in is just the room's boxes
  added to the capture scene.
- **Parallax.** A cube map is exact only at its probe. `"auto"` is the scene
  centre (the showcase case); `"node:<path>"` captures from a node's bounding-box
  centre, which is what a marker shell around a cluster wants. Depth-sorted
  `normal` / `volumetric` layers are ordered for the main camera, so their
  reflection is slightly mis-ordered — invisible under any real roughness.

**Baking at compile time: `luxar env bake`.** The same capture, driven headlessly,
moves the cost off the viewer entirely for a published scene:

```bash
luxar env bake scene.luxar.zarr                       # probe auto, 128 px faces
luxar env bake scene.luxar.zarr --probe node:clusters/shell_3
```

It drives the viewer through Playwright the way the gallery capture does (so it
runs unattended on a GPU box), using a `?bake-env&probe=...` URL parameter that
makes the viewer capture on load settle, read the six faces back as FLOATS
(`readRenderTargetPixels` — a PNG would clip everything above 1.0 and destroy the
highlights that make metals read) and download one binary blob with a JSON header
(probe, resolution, scene `content_hash`, appearance state). `luxar env attach
scene.luxar.zarr faces.bin` is the manual half of the same loop: it writes the
faces as an `environment/faces` zarr array of shape `(6, H, W, 3)` (float16) with
the header as attrs. At load the viewer builds a `CubeTexture` from it and runs
`PMREMGenerator.fromCubemap` — milliseconds — so what is stored is the raw
capture and the prefilter math stays three's. Idempotent (an unchanged scene
rewrites nothing) and honest about staleness: the viewer ignores a baked map
whose `content_hash` does not match the scene, says so in the console, and falls
back to the live capture or the room. What baking freezes: one appearance (a
Layers-panel colormap change does not reach the reflections until re-baked) and
one slice — a scene sliced on a hidden dimension gets one map for the slice it
was baked at, or one per slice under `environment/<slice>/`. Fine for a published
demo, which is where baking pays; for authoring sessions the live capture is the
default.

House-shaded meshes, points, lines and splats ignore `scene.environment`
entirely (their shaders never read it), so enabling it changes nothing about
existing scenes.

### 3.4 Transmission and what it can see

This is the load-bearing limitation and must be stated before anyone authors a
"glass sphere around a point cloud".

Three renders transmission by drawing the scene into a transmission render
target and sampling it, blurred by roughness, behind the transmissive surface.
Both backends do this natively (WebGL `renderTransmissionPass`, WebGPU
`Renderer` with `viewportSharedTexture`). The target is filled from the
**opaque and transmissive** render lists. **Transparent objects are excluded.**
Every Luxar point, line and splat material is `transparent: true` (additive,
luminous, normal, max). Therefore:

> A physical mesh with `transmission > 0` refracts the background, opaque
> meshes and other physical meshes. It does **not** refract points, lines or
> splats. They are drawn afterwards, on top, unrefracted.

For the stories demo's "glass around a cluster", the cluster's points would sit
crisply *on* the glass rather than *inside* it. Whether that reads well is a
design question, not a bug; it is also why the additive rim shell (§2) is the
right marker for the kiosk.

**Phase 2 (delivered) — what makes that statement true on BOTH backends.** On
WebGL it is free: the transmissive list draws before the transparent list, a
glass fragment lands with alpha 1 (`transmissionAlpha` is 1 because the
transmission target is cleared at alpha 1), and the glass writes no depth, so the
emissive draws that follow land on top. On WebGPU a transmissive mesh joins the
SAME transparent list as every point, line and splat layer, sorted back-to-front
by object depth — so a cluster whose centre sorts farther than the sphere would be
drawn first and then painted over by the alpha-1 glass. The physical material
therefore stamps `userData.drawBeforeEmissive` while `transmission > 0`, and the
depth-sort coordinator's cross-node pass orders such a mesh first within its
`layer_order` band (renderOrder −1 when unranked, since unranked emissive layers
sit at three's default 0). One more asymmetry worth recording: WebGL exposes
`renderer.transmissionResolutionScale` (left at three's 1.0; the 0.5 default
belongs to Phase 3's cost work), WebGPU has no equivalent and always mip-chains a
full-resolution framebuffer copy, so its transmission cost is not tunable.

Closing the gap is Phase 3 (§4). An earlier draft of this section proposed taking
over three's pass wholesale (a manual render target and a `scene.overrideMaterial`
dance inside `post-processing-manager/pipeline.ts`). Reading three's code shows
that is not needed; the two backends want two small, different moves:

- **WebGPU is almost free.** There, anything with `transmission > 0` lives in
  the *transparent* list and samples `viewportSharedTexture`, a copy of the
  framebuffer taken at the moment the glass is drawn. Whatever drew before it is
  visible through it. Give the glass a higher `layer_order` band than the data it
  wraps and three's own ordering refracts the data — a renderOrder rule the
  depth-sort coordinator already applies.
- **WebGL needs one trick, not a takeover.** Its `renderTransmissionPass` draws
  only the *opaque* list into a mip-mapped half-float target (sized by
  `renderer.transmissionResolutionScale`, tone mapping off). So split the scene
  pass: render everything except transmissive meshes into the HDR target as
  today, then render only the transmissive meshes on top, into the same target
  without clearing, with one screen-space quad textured with pass one's result
  added to that second scene. Three's pass draws the quad as its "opaque" content,
  so the glass refracts the data; the quad's own draw paints back pixels that are
  already there. A `layers` mask and a quad, isolated in `pipeline.ts`; no shader
  edits. SSAA jitter is unaffected (both renders share the jittered camera and
  the quad is screen-space) and so is adaptive DPR (the targets already follow
  it).

**Cost.** Paid only in frames where a transmissive mesh is visible; every other
scene keeps today's single pass. When paid: three's mip generation on the
transmission target plus the glass drawn twice — on the order of a millisecond at
1080p at `transmissionResolutionScale = 0.5` (the right default; roughness blur
hides it), several at 4K with SSAA at full scale. The adaptive DPR manager
regulates frame time, so a heavy case degrades resolution rather than stuttering.
Measure on the perf harness before trusting these numbers.

**The limits are about correctness, not cost**, and they are what keeps Phase 3
deferred. Which emissive layers write depth decides what a glass surface drawn
after them can be occluded by (`rendering/blending-state.ts`):

| Mode | depthTest | depthWrite |
| --- | --- | --- |
| `additive` | off | off |
| `luminous` | on | off |
| `max` | on | off |
| `volumetric` | on | off |
| `normal` | on | only at opacity ≥ 0.99 (points, gsplats; lines never) |
| `opaque` | on | on |

So data *in front of* a glass shell is painted over by the glass — it appears
refracted instead of crisp — for `additive`, `luminous`, `max` and `volumetric`
layers, which is the default for clouds and splats; it is depth-tested correctly
against `opaque` layers and against `normal` at (near) full opacity. No pass
design fixes that without depth for points. And depth-sorted layers are ordered
for the main camera, so through a clear lens their order is slightly off
(invisible under roughness). Phase 2 already gives glass that refracts the
background and other meshes, and the additive rim shell already does the marker
job; refracting the data *inside* matters for a lens or bubble showpiece. Hence:
opt-in, built when a demo wants it, prototyped first to measure the two costs.

### 3.5 Colour pipeline

The viewer renders linear and tone-maps in post (ACES by default). Three's
physical materials output linear radiance and read `renderer.toneMapping`, which
Luxar leaves at `NoToneMapping` for the scene pass, so a physical mesh flows
through bloom and the tone-mapping pass like any emissive node. Two consequences
to test, not assume: the bloom threshold treats a bright specular highlight as a
light source (usually desirable on glass, possibly not on metal), and `AgX`/
`Neutral` tone mapping shifts the saturated `attenuationColor` of thick glass.

**Phase 1 measurement** (the demo scene, `ab-webgpu-vs-webgl.mjs --bloom`,
1280×720, `dpr=1`, ACES, default bloom threshold 0.01). With bloom OFF (the
default) the physical spheres' specular highlights occupy ~1.5% of the frame as
near-white pixels against ~0.04% for a house-shaded fixture: the highlights ARE
bright, as expected of a polished metal under a room environment. With bloom ON
they bloom heavily — near-white rises to ~4.8% on WebGL — but so does everything
else at that threshold (the house fixture's own lit surfaces go to ~8.5%), so
the default threshold, not the material, decides. A metal that reads too hot
under bloom is a `bloom_threshold` matter for the scene author. Separately, the
bloom pass itself diverges between backends (WebGL blooms noticeably more than
WebGPU on the SAME scene, including a house-only one: SSIM 0.87–0.91 with
bloom on vs 0.99–1.00 with it off); that is a pre-existing post-pipeline
difference, not a physical-material one, and is tracked as #2563 rather than fixed here.

### 3.6 Dual-backend parity

`MeshPhysicalMaterial` and `MeshPhysicalNodeMaterial` are three's own and are
close but not pixel-identical (transmission blur and dispersion differ in
sampling). The acceptance test is a real-WebGPU A/B on the stories scene with a
structural-similarity floor, not pixel equality; the mesh spec's §11 row 6 A/B
harness is the tool.

**Phase 1 result.** The §11 row 6 run was manual, so the tool now exists as a
script: `packages/luxar-viewer/scripts/ab-webgpu-vs-webgl.mjs` launches the
system Chrome (the only headless browser with a WebGPU adapter), captures the
same scene on WebGL and on real WebGPU — verifying the WebGPU arm actually got a
WebGPU backend — and reports SSIM, normalised cross-correlation and lit-pixel
counts on a 256² greyscale downscale. On `demo_mesh_physical_materials` (six
spheres: house shader, dark clearcoat shell around an emissive cluster, gold,
steel, pearl, velvet) the two backends score **SSIM 0.986, NCC 0.998**, both
with the environment built and 7,680 triangles committed; the Fresnel rim on the
dark shell is visible on both. The stories scene is not on `main` yet, so the
demo carries the A/B until it is.

**Phase 2 result.** The same run on the nine-sphere demo (three glass spheres in
front of a house-shaded checkerboard, an emissive cluster inside the clear
one; 11,808 triangles, environment built on both arms, real WebGPU verified):
**SSIM 0.988, NCC 0.998**, lit-pixel share 15.6% (WebGL) vs 18.4% (WebGPU). Both
arms refract the checkerboard through all three glass spheres, the amber
sphere's volume attenuation reads on both, and the cluster inside the clear
glass is drawn crisp and unrefracted on both — the draw-before-emissive rule
doing its job on WebGPU. Where the arms differ most is the dispersive crystal
(`ior=2.0`, `dispersion=0.6`): WebGPU's transmission sample comes off a coarser
level of its framebuffer mip chain there and shows the checkerboard's tiles
through the sphere where WebGL shows a smoother frosted refraction. That is the
sampling difference this section predicts, not a defect in either mapping, and
it is why the acceptance floor is structural similarity rather than pixels.

## 4. Phases

| Phase | Scope | Unlocks | Cost |
| --- | --- | --- | --- |
| 1 — **delivered** | `material="physical"`, `roughness`, `metalness`, `clearcoat`, `clearcoat_roughness`, `iridescence`, `sheen`, `sheen_color`; default `RoomEnvironment` built lazily; Python validation; Layers-panel shows the physical knobs read-only; `demo_mesh_physical_materials`; the A/B script | Metals, lacquer, pearlescent shells; a true Fresnel rim via `clearcoat` on a dark base | Small: one factory entry, one env builder, attrs plumbing |
| 2 — **delivered** | `transmission`, `ior`, `thickness`, `attenuation_*`, `dispersion` with three's stock pass; glass ordered first in its band on both backends; the physical knobs as LIVE Layers-panel sliders (§6 item 2); three glass spheres and a checkerboard backdrop in the demo | Glass and lenses that refract the background and other meshes; tuning a material without a rebuild | Small; the §3.4 caveat is in the Layers panel tooltip |
| 3 — opt-in, deferred | Data through glass: `layer_order` ordering on WebGPU, a two-render split with an injected quad on WebGL, `transmissionResolutionScale` 0.5 (§3.4) | Glass that refracts the data behind it (data in front stays a known limit) | Small–medium, isolated in `pipeline.ts`; build when a demo needs it, measure first |
| 4 — designed | `viewer_config.environment.source = room \| scene \| hdri` with `probe`; live exact cube capture on settle/commit; `luxar env bake` / `?bake-env` / `luxar env attach` storing `environment/faces` with a `content_hash` guard (§3.3) | Metals and glass that reflect the data they sit in; zero-cost baked environments for published scenes | Small: one more builder behind `SceneEnvironment.ensure()`, one CLI pair, one Playwright driver |

## 5. Explicitly out of scope

- Physical materials on points, lines or splats. They are emissive primitives;
  the mesh is the only surface.
- Scene lights (`DirectionalLight` etc.). The environment is the light.
- Shadows.
- Replacing the house mesh shader. It stays the default: it is faster, has no
  environment dependency, and is what every existing scene renders with.
- Approximate environments computed from the resident arrays instead of rendered
  (emitter projection to an equirect, spherical harmonics from the data).
  Rejected in §3.3: they duplicate the appearance pipeline for a saving the
  exact capture does not need, since it is triggered rather than per frame.
- Depth writes for emissive point, line and splat layers so that glass can be
  occluded by data in front of it. It would break the additive sum those modes
  are (§3.4); the limit is documented instead.

## 6. Open questions

1. Should `material="physical"` force `opacity`-based transparency through
   three's `transparent` flag, or map Luxar `opacity` onto `transmission`? The
   former is literal and predictable; the latter is what "translucent glass"
   usually means. Proposal: literal, with `transmission` explicit.
   **Phase 1: literal.** `opacity < 1` (or per-vertex alpha without a cutoff)
   sets `transparent`; `transmission` stays a Phase 2 knob.
2. Does the Layers panel expose the physical knobs as live sliders (they map
   cleanly to `setLayer` patches) or read-only? Proposal: sliders in Phase 2
   once two demos exist to tune against. **Phase 1: read-only**, in place of
   the house shading sliders; Gamma and Blend hide too, since neither has a
   meaning on this material. Opacity stays live. **Phase 2: live sliders**, one
   per numeric knob from the shared knob table (ranges, steps and a log track
   for `attenuation_distance` whose top stop means "none"); the two colour knobs
   stay read-only rows; Reset restores the authored values.
3. Which demo carries Phase 1? Candidate: a bubble/lens marker variant of
   `esm3_protein_stories`, which already has the story dimension to switch
   marker styles on. **Phase 1: `demo_mesh_physical_materials`**, a numpy-only
   row of icospheres (house shader, dark clearcoat shell around an emissive
   point cluster, gold, steel, pearl, velvet), because the stories demo was not
   on `main`. Switching one stories shell to `material="physical"` is a
   one-line change once it lands; the shared `luxar.mesh.primitives.icosphere`
   exists for it.
4. Should a scene-derived environment be computed from the arrays (compile time,
   approximate) or rendered (exact)? **Decided: rendered, exact**, on both the
   live path and the baked one (§3.3). An approximation buys nothing once the
   exact capture is triggered rather than per frame, and costs a second copy of
   the appearance pipeline. Compile-time baking exists only as
   `luxar env bake`, which drives the same exact capture headlessly.
5. Where do probes live — on the scene manager or on the node? **Decided: the
   probe is a knob of the environment config** (`"auto"` = scene centre,
   `[x, y, z]`, or `"node:<path>"`), so one implementation serves the showcase
   case and the marker-shell case; a per-node probe is a second capture, not a
   second mechanism. Whether several probes can be active at once (one physical
   mesh per probe) is left to Phase 4's first demo.
6. Is Phase 3 worth its intrusion? **Decided: as re-formulated in §3.4 it is not
   intrusive** (a renderOrder rule on WebGPU, a two-render split with a quad on
   WebGL, isolated in `pipeline.ts`), and its cost is paid only when a
   transmissive mesh is visible. What defers it is the correctness limit that
   emissive layers do not write depth, so data in front of a glass surface is
   painted over. It stays opt-in and waits for a demo that wants a lens.
