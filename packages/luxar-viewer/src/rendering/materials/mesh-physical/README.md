# Physical Mesh Material — three's PBR behind the Luxar leaf surface

The mesh's **second material family**. `material="physical"` on `add_mesh` hands a mesh
to three.js's own physically based material — `MeshPhysicalMaterial` on WebGL,
`MeshPhysicalNodeMaterial` on WebGPU — lit by the scene environment the viewer builds
lazily the first time it meets one (`rendering/environment/`). Design and phases:
`docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md`; the house shader it is an
alternative to: `../mesh/README.md` and `docs/specs/MESH_NODE_SPEC.md` §6.2.

## Files

| File               | Purpose                                                                                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.ts`        | The ONE mapping from Luxar attrs onto three's properties, the defaults, the compositing rule (`derivePhysicalCompositing`), the transmitted-alpha pin (`pinTransmittedAlphaGlsl`) and the `LuxarMaterial` surface as shared functions. Imports no `three/webgpu`. |
| `material-glsl.ts` | `PhysicalMeshMaterial extends THREE.MeshPhysicalMaterial` — the WebGL wrapper. Thin: every method delegates to `config.ts`; its `onBeforeCompile` applies the alpha pin.                                  |
| `material-tsl.ts`  | `PhysicalMeshTSLMaterial extends MeshPhysicalNodeMaterial` — the WebGPU twin, in the lazy `three/webgpu` cone, reached only through `rendering/tsl/registry.ts`; its lighting-model subclass applies the alpha pin. |

Registered as `VISUAL_FACTORIES.meshPhysical` in `rendering/material-manager/factories.ts`
and constructed by `MaterialManager.getMeshPhysicalMaterial`. There is deliberately **no**
`PICKING_FACTORIES.meshPhysical`: picking renders geometry, not appearance, so a physical
mesh picks through the house mesh-pick material exactly like a house mesh.

## Why a wrapper at all, and why so thin

The spec's first instinct was three's material unwrapped. The viewer's generic code
paths make that a crash or a silent skip: `MaterialManager.getMeshMaterial` calls
`updateCameraParams` unconditionally, the Layers panel / exposure / LOD cross-fade gate
on `isLuxarMaterial` (`updateIntensity` + `updateGamma`), and `resetAllLayers` calls
`applyBlendingMode` on every layer — falling back to writing the HOUSE shader's blend
state when the method is missing. So the wrapper implements exactly that surface, each
method with a physical meaning, and nothing else:

| Luxar method        | Three property                                      | Meaning                                                                                                                                                        |
| ------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `updateOpacity`     | `opacity`, `transparent`, `depthWrite`, `alphaTest` | Re-derives the whole compositing decision (below).                                                                                                             |
| `updateIntensity`   | `color` (scalar)                                    | With `vertexColors` on, base colour is `color × vertexColor`, so a scalar `color` is a gain on the authored colour — what `uIntensity` is on the house shader. |
| `updateOffset`      | `emissive` (scalar)                                 | An additive brightness shift IS an emitted radiance. Negative (black-level subtraction) has no physical counterpart and clamps to 0.                           |
| `updateGamma`       | `userData.gamma` only                               | Recorded, never applied — a PBR material has no gamma term. The panel hides the slider for a physical layer.                                                   |
| `applyBlendingMode` | nothing                                             | Deliberate no-op: a physical mesh has no Luxar blending mode (refused at authoring; an inherited one is ignored with a notice).                                |
| `getOpacity`        | —                                                   | The LOD cross-fade's fade base.                                                                                                                                |
| `updateRefractData` | `userData.drawAfterEmissive` / `drawBeforeEmissive` | Luxar `refract_data`, live: the glass draws after (true) or before (false) the emissive data. Plain state, no rebuild.                                         |

Absent on purpose: `updateCameraParams` (the near fade is a house-shader feature, so
`register()` files the material as static), `updateShading` / `updateBaseColorTexture` /
`updateColormapTexture` / the four shade knobs (house-shader features; the commit path
and the panel already no-op when the methods are missing).

## The compositing rule

A physical mesh has no blending mode, so translucency is read off the **data**
(`derivePhysicalCompositing`):

- `opacity < 1` → translucent.
- Per-vertex alpha → translucent, **unless** an `alpha_cutoff` was authored, in which
  case the alpha is a cutout (`alphaTest`) and the surface stays opaque — the meaning
  the house `opaque` mode gives the same pair.
- Translucent → `transparent = true`, `depthWrite = false`, `userData.blendingMode`
  UNSET. Opaque → `depthWrite = true`, `userData.blendingMode = 'opaque'`.

The `userData.blendingMode` stamp is what the rest of the viewer keys on: the depth-sort
coordinator releases a node whose stamp is `opaque` or absent (a physical mesh is **never**
triangle-sorted — spec §3.2 — but still earns its `layer_order` band rank), and the pick
pass applies the cutout iff the stamp is `opaque`, matching the screen.

Vertex alpha is known only at the first **commit** (`MeshMetadata` says `has_colors`,
not how many components), so `commit-mesh-geometry.ts` calls `applyMeshVertexAlpha`
next to `applyMeshTexture`; a no-op for the house material and on every later commit.

## Knobs and defaults

One table, `PHYSICAL_MESH_KNOBS` in `config.ts`, describes every numeric knob — attr name,
three property, domain, default, slider presentation and whether crossing zero selects a
shader variant — and the construction mapping, the live Layers-panel sliders and the tests
all read it. In panel order: `roughness`, `metalness`, `clearcoat`, `clearcoat_roughness`,
`iridescence`, `sheen` (each `[0, 1]`), then the glass family `transmission` (`[0, 1]`),
`ior` (`[1, 2.333]`), `thickness` (`>= 0`), `attenuation_distance` (`> 0`, default
`Infinity` = none; the slider's top stop means ∞) and `dispersion` (`>= 0`). Values are
clamped with the same NaN-to-default policy as the house knobs. The two colours,
`sheen_color` and `attenuation_color` (`#rrggbb`), default to **white** — three's own for
attenuation, and deliberately NOT three's black for sheen, because a black sheen is a
no-op and `sheen=1` alone would render nothing. `alpha_cutoff` maps to `alphaTest`;
`shading` maps its flat/smooth half to `flatShading` (`none` is refused at authoring). The
one boolean, `refract_data` (Phase 3), is not a knob-table entry: it is an input to the
compositing decision and the Layers panel's "Refract data" switch.

`setPhysicalKnob` is the one write path for construction AND the live sliders: it clamps,
assigns, re-derives compositing for `transmission`, and flags `needsUpdate` when a
`programAffecting` knob (`clearcoat`, `iridescence`, `sheen`, `transmission`, `dispersion`)
crosses zero — three's WebGL setters version-bump on that crossing already, but
`MeshPhysicalNodeMaterial` bakes the flags into its lighting model at setup, so the rule
lives here for both twins.

## Glass (transmission, spec §3.4)

`transmission > 0` composites as translucent (no depth write) and stamps
`userData.drawBeforeEmissive`, which the depth-sort coordinator turns into "first in its
`layer_order` band" (renderOrder −1 when unranked). Three renders transmission by sampling
a copy of what was drawn before the glass — the opaque list on WebGL, the framebuffer just
before the first transmissive draw on WebGPU — so by default glass refracts the background
and other meshes, never Luxar's transparent point, line and splat materials: those draw on
top, crisp and unrefracted, on both backends. On WebGPU the glass shares the transparent
list with the emissive layers and lands its fragments with alpha 1, which is why it must
draw first.

**`refract_data` (Phase 3)** flips the stamp to `userData.drawAfterEmissive`. The
coordinator then ranks the glass LAST in its band (a rank even when nothing else would
earn one — the unranked emissive layers sit at 0), which on WebGPU is the whole
mechanism; on WebGL `PostProcessingManager`'s `DataRefractionSplit`
(`post-processing/post-processing-manager/refraction-split.ts`) renders the scene twice
with a screen-space quad so three's transmission target holds the data. Data in front of
a refracting glass is painted over — emissive layers write no depth — which is why the
flag is opt-in. Compositing is otherwise identical: translucent, no depth write,
`NormalBlending`. `transmissionResolutionScale` (config `renderingControls.refraction`,
0.5) applies to the split's second pass only; WebGPU has no equivalent knob.

**The transmitted alpha is pinned to 1 on both backends, always.** Luxar's HDR
framebuffer alpha is an overdraw count, not coverage (the additive, luminous and normal
point and line shaders are alpha-weighted; 8–18 was measured behind a point cloud), and
the house already treats it as undefined: the mega shader reads RGB only and the EXR
capture forces alpha to 1. Three copies the sampled alpha into the glass fragment alpha,
an unclamped blend factor on the half-float target — a glass drawn after the data came out
thousands of times too bright or negative. So the WebGL wrapper's `onBeforeCompile`
expands the `transmission_fragment` include with `TRANSMISSION_ALPHA_MIX_LINE` pinned
(`pinTransmittedAlphaGlsl`; a unit test on three's real chunk turns a future drift into a
red build; a fixed `customProgramCacheKey` keeps the warm-up fingerprint stable), and the
WebGPU wrapper's `LuxarPhysicalLightingModel` saves `diffuseColor.a` into a property node
before three's `start` and restores it after. A pixel no-op for the glass-first path,
whose transmission target only ever holds alpha-1 content; an `opaque`-mode house layer
or a cutout mesh seen THROUGH glass now follows the same rule.

## Colour pipeline

The scene pass runs with `renderer.toneMapping = NoToneMapping` into the linear HDR
target, so a physical mesh's radiance flows through bloom and the mega-shader tone map
exactly like an emissive node (`toneMapped = false` is stated on the material too).
`sheen_color` is parsed by `Color.set('#rrggbb')`, i.e. read as sRGB and converted to the
linear working space — the human reading of a hex tint.

## Testing

- `tests/unit/rendering/materials/mesh-physical/` — the config mapping on both twins from
  one table, the compositing rule (glass included), the knob table against three's own
  defaults, the slider mapping, the zero-crossing rebuild rule, the `LuxarMaterial` surface,
  the stamp rule.
- `tests/unit/rendering/node-factory/create-mesh-node.test.ts` — the family branch, the
  glass pass-through, the house pick material, the placeholder bookkeeping, the
  inherited-mode notice.
- `tests/unit/rendering/depth-sort-coordinator.test.ts` — glass ordered first in its band,
  and at renderOrder −1 when unranked; refracting glass ranked last, the containment-hoist
  skip, and `collectRefractingGlass`.
- `tests/unit/rendering/post-processing/post-processing-manager/pipeline.test.ts` — the
  refraction split's pass order, borrowed-state restoration (including on a throw) and the
  quad's MSAA-critical flags.
- `tests/unit/ui/layers/layers-panel.test.ts` — the live sliders and the Refract data switch
  on a REAL wrapper, and both Resets restoring the authored values.
- `tests/unit/rendering/environment/` — the environment is lazy, cached, invisible to house
  materials, and disposed.
- No codegen snapshot: the snapshot harness pins TSL Luxar writes; three's materials are
  three's to pin.
