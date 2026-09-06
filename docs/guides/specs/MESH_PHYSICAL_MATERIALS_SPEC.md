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
  reason. Phase 2's `transmission` / `ior` / `thickness` / `attenuation_*` /
  `dispersion` are not accepted yet.
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
applies. Translucency is read off the data (`opacity < 1`, or per-vertex alpha
without an `alpha_cutoff`); translucent surfaces do not write depth. Vertex
alpha is learned at the first commit (`applyMeshVertexAlpha`), since the attrs
record `has_colors` but not the component count.

### 3.3 Lighting: a scene environment, not scene lights

Physical materials render black without light. Luxar stays light-free in the
sense that matters (no light objects in the graph, nothing per-node), and gains
one scene-level input: `scene.environment`.

- **Default:** three's procedural `RoomEnvironment` through `PMREMGenerator`,
  built lazily the first time a physical mesh is created and cached on the
  `SceneManager`. It needs no asset, gives believable reflections and a neutral
  key, and is what three's own examples use.
- **Authored:** `viewer_config.environment = {"preset": "room" | "neutral" |
  "studio"}` first; an HDRI URL later if a demo needs one (the loader already
  fetches opaque files from the store for overlays and textures).
- `environmentIntensity` rides `viewer_config.environment.intensity`.

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

Closing the gap is Phase 3 (§4): a Luxar-owned transmission pass that renders
the point layers into the transmission target before three samples it. It means
taking over three's pass (`renderer.transmissionResolutionScale`, a manual
`renderTarget` and `scene.overrideMaterial` dance) inside
`post-processing-manager/pipeline.ts`, and it interacts with the SSAA jitter and
the adaptive DPR ceiling. It is deliberately not Phase 1.

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

## 4. Phases

| Phase | Scope | Unlocks | Cost |
| --- | --- | --- | --- |
| 1 — **delivered** | `material="physical"`, `roughness`, `metalness`, `clearcoat`, `clearcoat_roughness`, `iridescence`, `sheen`, `sheen_color`; default `RoomEnvironment` built lazily; Python validation; Layers-panel shows the physical knobs read-only; `demo_mesh_physical_materials`; the A/B script | Metals, lacquer, pearlescent shells; a true Fresnel rim via `clearcoat` on a dark base | Small: one factory entry, one env builder, attrs plumbing |
| 2 | `transmission`, `ior`, `thickness`, `attenuation_*`, `dispersion` with three's stock pass | Glass and lenses that refract the background and other meshes | Small; ships with the §3.4 caveat documented in the Layers panel tooltip |
| 3 | Luxar-owned transmission pass including the point/line/splat layers | Glass that refracts the data inside it | Medium–large; touches the post pipeline, SSAA, DPR |
| 4 | Authored HDRI environments (`viewer_config.environment.url`) | Scene-specific reflections | Small once Phase 1 exists |

## 5. Explicitly out of scope

- Physical materials on points, lines or splats. They are emissive primitives;
  the mesh is the only surface.
- Scene lights (`DirectionalLight` etc.). The environment is the light.
- Shadows.
- Replacing the house mesh shader. It stays the default: it is faster, has no
  environment dependency, and is what every existing scene renders with.

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
   meaning on this material. Opacity stays live.
3. Which demo carries Phase 1? Candidate: a bubble/lens marker variant of
   `esm3_protein_stories`, which already has the story dimension to switch
   marker styles on. **Phase 1: `demo_mesh_physical_materials`**, a numpy-only
   row of icospheres (house shader, dark clearcoat shell around an emissive
   point cluster, gold, steel, pearl, velvet), because the stories demo was not
   on `main`. Switching one stories shell to `material="physical"` is a
   one-line change once it lands; the shared `luxar.mesh.primitives.icosphere`
   exists for it.
