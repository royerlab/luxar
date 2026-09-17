# Mesh Physical Materials Spec — glass, metal and iridescence on Luxar meshes

**Status:** Phases 1–4 are **implemented** (Phase 3 opt-in per mesh, with the data
in front of the glass partitioned out and drawn crisp). Written after the `esm3_protein_stories` demo wanted translucent marker
shells around clusters and got them from the existing light-free mesh model (see
§2); this document is the plan for the materials that model cannot express, and —
for the delivered phases — the record of what shipped and where it deviates from
the sketch (the implementation notes in §3.1, §3.2, §3.3, §3.5 and §3.6).

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
- **Phase 3 (delivered)** adds one boolean, `refract_data`: the glass draws AFTER
  the emissive data and refracts it (§3.4). Absent = false, so every Phase 2 scene
  keeps its glass-first behaviour. It joins the pairing rule — refused without a
  `transmission` above zero, since a surface that transmits nothing has nothing to
  refract — and is refused as a number (`refract_data=1` reads like a slider value).
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
  built lazily the first time a physical mesh is created and cached by the active
  viewer host (`SceneManager` or `LuxarLayer`). Layer mode preserves a host-supplied
  `scene.environment`; otherwise its environment also lights the host's own
  lighting-model materials until the layer is disposed. It needs no asset, gives
  believable reflections and a neutral key, and is what three's own examples use.
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
runs unattended on a GPU box), using a `?bakeEnv&probe=...` URL parameter that
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

**Phase 4 implementation notes** (what shipped; where it differs from the text
above):

- **No explicit `PMREMGenerator.fromCubemap` call.** Both backends re-prefilter a
  cube texture assigned to `scene.environment` whenever its `pmremVersion` moves,
  and `CubeCamera.update` bumps it — so the live capture is "render six faces,
  assign the target's texture", and a baked map is "build a half-float
  `CubeTexture`, assign it". The room still goes through `fromScene` once.
- **Visibility, not a `layers` bit, hides physical meshes during the capture.**
  A persistent bit would need the pick camera and the main camera to agree on it;
  each mesh's PREVIOUS `visible` is restored, so a user- or LOD-hidden mesh stays
  hidden. (The one `layers` use in the viewer, Phase 3's refraction split, is
  transient inside a single render call and restored before this capture can run.)
- **Camera settle is NOT a trigger.** A cube map from a fixed probe is
  view-independent; the only camera-driven change — LOD residency — arrives as a
  geometry commit. Triggers are: geometry commit (`eventBus 'geometry-committed'`),
  slice change, appearance change; a 250 ms debounce; and the loader's settled
  predicate. The first light is an immediate capture of whatever is resident.
- **The capture pushes its own camera params** to the material manager directly
  (90° fov, a square drawing buffer, pixel ratio 1) and restores the main camera's
  through the ordinary path, because the shared helper reads the canvas size.
- **The store contract.** `environment/` is a root-level group with no `type` and
  no `kind` (the viewer skips such groups as sidecars; `RESERVED_ROOT_GROUPS` makes
  every general Python walker that enumerates root children as nodes do the same,
  and the compiler refuses a user node by that name). The faces array is
  `environment/faces-<xxh64[:8]>`, shape `(6, H, W, 4)`,
  zarr dtype **`uint16` holding IEEE half-float bits** rather than `float16`:
  zarrita throws on `<f2` without a `Float16Array`, while the GPU readback and
  three's `HalfFloatType` cube texture already speak half bits, so `uint16` needs
  no conversion at either end. RGBA rather than RGB because the readback is RGBA.
  The digest suffix makes a re-bake a NEW path (a caching viewer cannot serve stale
  faces); the group's `faces` attr names the live array.
- **The environment group is EXCLUDED from the scene `content_hash`** (both
  hashing walks). That is what makes the header's `scene_content_hash` an exact
  guard the viewer can evaluate, keeps `luxar env attach` from invalidating every
  visitor's warm cache, and makes attaching idempotent. The group carries its own
  digest for tooling.
- **`luxar env bake`** serves the store and the built viewer from one process
  (a stoppable `served_store`), drives a headless browser to
  `?bakeEnv&probe=…&envResolution=…` through the viewer's Playwright
  (`scripts/bake-env.mjs`, Node — Playwright is a devDependency of the viewer, not
  a Python dependency, so the command needs a development checkout), pulls the
  `LXENV001` container back through `page.evaluate` as base64 rather than a
  download event, and attaches it. `luxar env attach` is the manual half. A
  bake ignores any previously attached map (it captures the SCENE), and a
  stale bake is refused unless `--force`.
- Precedence at load: a valid baked map > `viewer_config.environment.source` > the
  room. `viewer_config.environment` round-trips through the Python
  `EnvironmentConfig` dataclass and the Ctrl+Shift+S export.

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
> splats — they are drawn afterwards, on top, unrefracted — **unless it is
> authored with `refract_data=True`** (Phase 3, below), in which case it draws
> after them and refracts what is behind it, while what is in front of it is
> drawn crisp on top (the depth partition, below).

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

**Phase 3 (delivered) — data through glass, opt-in.** `refract_data=True` flips
the ordering stamp to `userData.drawAfterEmissive`. The spike
(`spike/glass-refracts-data`) settled the mechanism and its cost before this was
built; the owner's review of the first delivery then removed the one limit the
spike had accepted (data in front of the glass painted over), with the partition
described last below.

- **Rank last.** The depth-sort coordinator gives a refracting glass a rank even
  when nothing else would (the unranked emissive layers sit at renderOrder 0) and a
  `last` flag that mirrors Phase 2's `first`: after everything else in its
  `layer_order` band, before any higher band. The containment hoist (§6.3 of the
  mesh spec) skips edges that touch a glass-last group — a lens enclosing a cluster
  is exactly "container contains content", and hoisting the lens first would paint
  the cluster crisp on top instead of refracting it.
- **The split.** `PostProcessingManager` owns a `DataRefractionSplit` that runs on
  BOTH backends when a visible glass asks to refract (`collectRefractingGlass`, read
  off the coordinator's node registry, not a scene traversal). Pass G renders the
  refracting glass's FRONT faces, depth only, through depth-only proxies into a
  target wrapping the one shared depth texture (`materials/_shared/glass-partition.ts`;
  cleared depth 1.0 = "no glass here"). Pass A renders everything but the refracting
  glass into the HDR target with the data in partition mode 1. On WebGL — whose
  `renderTransmissionPass` draws only the *opaque* list into a mip-mapped half-float
  target — the colour is blitted into a copy target, and pass B renders the glass
  plus one OPAQUE screen-space quad textured with that copy, so three draws the quad
  into the transmission texture and the glass refracts the data; on WebGPU three
  samples the live framebuffer and pass B is the glass alone. Pass C renders the data
  again in partition mode 2. Pass A's depth survives, so an opaque mesh in front
  still occludes the glass. The refracting glass sits on `Object3D.layers` bit 1 and
  the meshes three's own materials draw (Phase 2 glass, opaque physical surfaces —
  they carry no Luxar shader and cannot classify themselves) on bit 2 for the
  duration (`rendering/render-layers.ts`), drawn in pass A and left out of pass C;
  everything borrowed — the partition mode first, the layer masks, the camera mask,
  `autoClear`, the scene background, the transmission scale, the quads' scene
  membership — goes back in a `finally`, so nothing outside that one render call
  can observe any of it: picking, the environment capture, the blend warm-up and
  scene disposal are untouched. The raw-HDR capture path shares the same stage, so
  an EXR refracts too.
- **The depth partition — how data in front of the glass stays crisp.** Emissive
  layers write no depth (additive mode has the depth test off altogether), so
  nothing in the hardware can tell a lattice row in front of the lens from one
  behind it. Every Luxar data fragment shader (points, quad and capsule lines,
  splats, the house mesh; GLSL and TSL) therefore classifies ITSELF against the
  pass-G texture, first thing in its fragment stage:
  `inFront = glassDepth < 1.0 && fragmentDepth < glassDepth`. Mode 1 discards the
  fragments in front; mode 2 discards the rest. The two modes are complements of one
  predicate, so every data fragment is drawn exactly once across passes A and C, and
  the glass paints over nothing nearer than itself. The mode is a runtime uniform
  (`uGlassPartition`, flipped several times per frame, never a define), the sampler
  `uGlassDepth` binds ONE module-level `DepthTexture` that the pass-G target is
  created with, so no material is ever rebound; the pick materials deliberately
  carry neither — picking must see all the data, and it runs while the mode is 0.
  GLSL reads its own window depth from `gl_FragCoord.z` and the texture with an
  exact `texelFetch` at its pixel; TSL carries the clip `zw` as a varying (the data
  graphs replace three's vertex stage, so three's depth nodes would read a
  quad-corner attribute) remapped to window depth per coordinate system at
  build time, and samples at `screenUV` exactly as three's `viewportDepthTexture`
  does, so the depth texture's Y flip on the GLSL builder is applied for free.
  The depth map stores only the nearest refracting front face, so overlapping
  refracting glasses can paint over data in front of the farther glass. Viewed from
  inside a double-sided refracting glass, the front-face proxy writes no depth and the
  shell can paint over enclosed data.
- **Two things three does that the split must undo.** (1) Both renderers FORCE a
  clear at the start of every `render()` whose scene has a colour background,
  whatever `autoClear` says — passes B and C run with `scene.background` taken away
  (found by bisecting a black frame: the original two-pass split only survived
  because the WebGL quad repainted the frame after that clear). (2) Under MSAA the
  WebGL renderer resolves a multisampled target's colour to its texture at the end of
  every `render()` and INVALIDATES the multisampled colour attachment (depth is
  resolved and kept), so passes B and C each draw onto undefined colour until a
  full-screen quad repaints it from a copy; the pass-B quad and a second blit +
  restore quad before pass C do that, which is why the quads' `depthTest:false`,
  `depthWrite:false`, `transparent:false` and `frustumCulled:false` are pinned by
  test. `WebGPURenderer`'s WebGL2 fallback (`?webgpuForceWebgl`) invalidates too
  and cannot take the GLSL quads, so that one combination with MSAA on falls back to
  a single pass with a warning.
- **`transmissionResolutionScale` 0.5, pass B only** (config
  `renderingControls.refraction`, validated in `(0, 1]`). Three reads it per
  transmission pass, so Phase 2 glass in pass A keeps three's default and its
  pixels are untouched; WebGPU has no equivalent knob. SSAA is plain supersampling
  here (there is no camera jitter), so all passes render at the same size.

**The alpha finding — the one thing the spike changed about this design.** Luxar's
`additive`, `luminous` and `normal` point and line shaders are alpha-weighted (RGB
unweighted, alpha = intensity·opacity, blended SrcAlpha/One with One/One on the
alpha channel), so the HDR framebuffer alpha behind a point cloud is an overdraw
count — 8 to 18 was measured, `Infinity` on dense scenes. The house already treats
that channel as undefined: the mega shader reads RGB only and the EXR capture forces
alpha to 1 before export. Three's transmission, meant for transparent canvases,
copies the sampled alpha into the glass fragment alpha (`transmission_fragment`'s
`transmissionAlpha` on WebGL, `PhysicalLightingModel.start`'s `diffuseColor.a`
multiply on WebGPU), an UNCLAMPED blend factor on a half-float target: a glass drawn
after the data came out thousands of times too bright, or negative, on both
backends. **Decision: the physical family refuses that channel too.** The WebGL
wrapper's `onBeforeCompile` expands the `transmission_fragment` include with the one
mix line pinned to `1.0` (a unit test on three's real chunk turns a future drift
into a red build; a fixed `customProgramCacheKey` keeps the warm-up fingerprint
stable); the WebGPU wrapper's lighting model saves `diffuseColor.a` into a property
node before three's `start` and restores it after (a `toVar` would be declared
after the multiply). Always on, and a pixel no-op for the glass-first path, whose
transmission target only ever holds the alpha-1 clear and opaque meshes; an
`opaque`-mode house layer or a cutout mesh seen through glass now follows the same
rule. Redefining HDR alpha as coverage was considered and rejected: it needs
premultiplied output plus separate alpha blend factors on every emissive mode on
both backends (the gsplat code records that separate alpha-channel blend state trips
the WebGPU renderer's WebGL2 bridge), and with the clear alpha at 1 it would yield
the same pixels.

**Cost.** Paid only in frames where a refracting glass is visible; every other
scene keeps today's single pass. Measured on `mesh_glass_lens_example` (three
spheres, 675 points) at 1280×720, dpr 1, Apple M4 Max, median of 7 trials × 20
`PostProcessingManager.render()` calls behind a GPU barrier (a one-pixel
`readPixels` on WebGL — `gl.finish()` returns in 0 ms on ANGLE/Metal and is not a
barrier — and `queue.onSubmittedWorkDone()` on WebGPU):

| | refracting glass hidden (one pass) | the partitioned split | the earlier two-pass split (partition off, in-page) |
| --- | --- | --- | --- |
| WebGL | 1.7 ms | 4.2 ms | 5.2 ms |
| native WebGPU | 0.8 ms | 1.5 ms | 1.5 ms |

Switching the passes off one at a time attributes the whole difference to pass B —
the glass draw with three's transmission render — on both backends; the depth
pre-pass, the partition compare (one texel fetch per data fragment under a uniform
branch) and pass C add nothing measurable on this scene. What pass C does add in
general is the data's vertex work a second time, most of whose fragments discard at
once: on a multi-million-point cloud that is a share of the data draw in exactly
the frames where a refracting glass is visible, and the adaptive DPR manager
regulates frame time in any case. (The spike had measured the two-pass split at
0.68 ms on a one-sphere scene; the 3.5 ms pass B here is pre-existing and a
follow-up, not part of the partition.)

**What the depth table now means.** Which emissive layers write depth decides what
a glass surface drawn after them is occluded by in pass B
(`rendering/blending-state.ts`):

| Mode | depthTest | depthWrite |
| --- | --- | --- |
| `additive` | off | off |
| `luminous` | on | off |
| `max` | on | off |
| `volumetric` | on | off |
| `normal` | on | only at opacity ≥ 0.99 (points, gsplats; lines never) |
| `opaque` | on | on |

Before the partition this table was the limit: data in `additive`, `luminous`,
`max` and `volumetric` layers in front of a refracting glass was painted over by
it. The partition removes that dependency — the in-front fragments are held back
from pass A and drawn in pass C whatever their mode — so the table only says which
layers ALSO occlude the glass through the hardware depth test (`opaque`, and
`normal` at near-full opacity). Depth-sorted layers are ordered for the main
camera, so through a clear lens their order is slightly off (invisible under
roughness). The flag stays opt-in per mesh because refraction is a look, not a
default: a lens of saturated data is a white disk — additive data above 1.0 clips
under tone mapping — so a showcase needs faint, non-overlapping points, and the
frame pays pass B.

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
counts on a 256² greyscale downscale. On `mesh_physical_materials_example` (then six
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

**Phase 4 result** (`mesh_reflections_example`: a 60k-point emissive swirl with a
chrome and a glass sphere, `environment.source = "scene"`, 1280×720, dpr 1,
bloom off). Live capture: **SSIM 0.986, NCC 0.9995** WebGL vs real WebGPU, the
environment built on both arms — the chrome reflects the swirl on both. Then a
real headless bake through `scripts/bake-env.mjs` against the dev viewer (a 64 px
cube, 197 KB container), attached with `luxar env attach`, and the same frames
captured again with the viewer reporting `kind = baked`: baked-vs-live agreement
is **SSIM 0.9985 (WebGL) / 0.9989 (WebGPU)** on the same backend (an
unrelated-scene control scores 0.23), so the stored faces rebuild the texture the
capture produced — orientation included — and with the baked map the two
backends render identically (**SSIM 1.000**), since neither captures anything.
`LuxarScene.nodes` and `luxar info` list no `environment` node, the root
`content_hash` is unchanged by the attach, and a second attach is a no-op.

**Phase 3 result** (spike, `spike/glass-refracts-data`, on a faint point lattice
with a refracting glass sphere, 1280×720, dpr 1). With the flag off the two
backends agree at **SSIM 0.999** inside the glass; with it on, **0.888–0.917**
(the lens is a magnified, inverted image of soft point sprites, and the two
transmission samplers blur it differently). On the saturated reflections swirl
the lens is a white disk on both and they agree at 0.994–0.998. Before the alpha
pin the same crops scored 0.25–0.58 and the HDR pixels inside the glass read
`[-15504, -21808, -22336, 65504]`. The verification run of the delivered code on
`mesh_glass_lens_example` is recorded in the Phase 3 PR. **With the depth
partition** (WebGL vs native WebGPU on the lens example, whole frame): **SSIM
0.991, NCC 0.997**. The partition itself, from behind the spheres with the lattice
between the camera and the lens: of the 22 lattice points projecting inside the
lens disk, sampled with the lens shown and hidden, the mean per-channel difference
is 7/255 with one point over 24 (the glass brightening under a point where it
refracts the neighbouring sphere) on both backends; with the partition switched
off in-page the same points differ by 51 on average, nine of them over 24, up to
207 — the glass's refracted image of the lattice painted over them. Those are the
thresholds `glass-refraction-partition.spec.ts` sits between.

## 4. Phases

| Phase | Scope | Unlocks | Cost |
| --- | --- | --- | --- |
| 1 — **delivered** | `material="physical"`, `roughness`, `metalness`, `clearcoat`, `clearcoat_roughness`, `iridescence`, `sheen`, `sheen_color`; default `RoomEnvironment` built lazily; Python validation; Layers-panel shows the physical knobs read-only; `mesh_physical_materials_example`; the A/B script | Metals, lacquer, pearlescent shells; a true Fresnel rim via `clearcoat` on a dark base | Small: one factory entry, one env builder, attrs plumbing |
| 2 — **delivered** | `transmission`, `ior`, `thickness`, `attenuation_*`, `dispersion` with three's stock pass; glass ordered first in its band on both backends; the physical knobs as LIVE Layers-panel sliders (§6 item 2); three glass spheres and a checkerboard backdrop in the demo | Glass and lenses that refract the background and other meshes; tuning a material without a rebuild | Small; the §3.4 caveat is in the Layers panel tooltip |
| 3 — **delivered** (opt-in) | `refract_data`: glass ranked LAST in its band, the `DataRefractionSplit` on both backends (glass depth pre-pass, data behind, glass — with a screen quad on WebGL — data in front), the per-fragment depth partition in every data shader (`uGlassPartition` / `uGlassDepth`, GLSL and TSL), `transmissionResolutionScale` 0.5 in pass B only, the transmitted alpha pinned to 1 on both backends (§3.4); a live Layers-panel switch; `mesh_glass_lens_example`; the `glass-refraction-partition` E2E spec | Glass that refracts the data behind it while the data in front of it stays crisp | Medium: one split module owned by `PostProcessingManager`, one `last` flag + two queries in the coordinator, a guard in ten fragment shaders, two shader hooks |
| 4 — **delivered** | `viewer_config.environment.source = room \| scene \| hdri` with `probe`, `resolution`, `intensity`, `url`; live exact cube capture on commit / slice / appearance change once settled; `luxar env bake` / `?bakeEnv` / `luxar env attach` storing `environment/faces-<digest>` (uint16 half bits) with an exact `scene_content_hash` guard, the group excluded from the scene digest (§3.3 Phase 4 notes); `mesh_reflections_example` | Metals and glass that reflect the data they sit in; zero-cost baked environments for published scenes | Small: one more builder behind `SceneEnvironment.ensure()`, one CLI pair, one Playwright driver |

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
  are (§3.4); the refraction split partitions the data by depth against the glass
  in the data shaders instead, which writes no depth.

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
   stay read-only rows; Reset restores the authored values. **Phase 3: a live
   "Refract data" switch** after the sliders, inert (with the reason) until
   Transmission is above 0; the per-row Reset restores the physical knobs and the
   switch (it had only ever reset the readouts).
3. Which demo carries Phase 1? Candidate: a bubble/lens marker variant of
   `esm3_protein_stories`, which already has the story dimension to switch
   marker styles on. **Phase 1: `mesh_physical_materials_example`** (a demo until
   Phase 3 moved the physical scenes to `packages/luxar/examples/`), a numpy-only
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
   mesh per probe) is left to Phase 4's first demo. **Phase 4: one probe per
   scene.** `mesh_reflections_example` puts the chrome sphere AT the probe (the
   scene centre) and the glass sphere beside it, which is exact where it matters
   and a small parallax error where it does not; a per-mesh probe stays a
   possible follow-up, not a need any demo has shown.
6. Is Phase 3 worth its intrusion? **Decided: as re-formulated in §3.4 it is not
   intrusive** (a renderOrder rule on WebGPU, a two-render split with a quad on
   WebGL, isolated in `pipeline.ts`), and its cost is paid only when a
   transmissive mesh is visible. What defers it is the correctness limit that
   emissive layers do not write depth, so data in front of a glass surface is
   painted over. **Phase 3: built, opt-in per mesh (`refract_data`)**, after a
   spike measured the cost (none) and the limit (real, visible) and found the
   alpha defect below. **The limit itself was then removed** (owner review of the
   first delivery: "when 'refract data' is on, it refracts even data in front of
   it?!"): the per-fragment depth partition in §3.4 draws the data in front of the
   glass crisp on top, on both backends. `mesh_glass_lens_example` is the lens.
7. What does the HDR framebuffer's alpha channel MEAN? **Decided: nothing — it is
   an overdraw count, and consumers must not read it.** The mega shader and the
   EXR capture already followed that rule; Phase 3 makes the physical glass follow
   it too (§3.4, the alpha finding). Redefining it as coverage is a possible future
   refactor of every emissive blend mode, not a Phase 3 prerequisite: with the
   clear alpha at 1 it would change no pixel the glass produces.
