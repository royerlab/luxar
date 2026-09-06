# Scene Environment — the one lighting input a light-free viewer has

Luxar's four geometry types are emissive and the house mesh shader lights itself from a
fixed view-space key, so the scene has never held a light or an environment map. A
physically based material (`materials/mesh-physical/`) renders black without one.
Rather than add light objects to the graph, this module sets `scene.environment` to a
prefiltered (PMREM) copy of three's procedural `RoomEnvironment` — no asset, a neutral
key, believable reflections, and what three's own examples light with. Design:
`docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3.

## Files

| File                   | Purpose                                                                                                                                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scene-environment.ts` | `SceneEnvironment` (`ensure()` / `isReady()` / `rebuild()` / `dispose()`) and `createSceneEnvironment(renderer, backend, scene)`, which picks the WebGL or the WebGPU PMREM generator for the active renderer. |

## Two load-bearing properties

- **Lazy.** Nothing is built until `ensure()` is called, and the only caller is the
  material manager's `onPhysicalMaterialCreated` hook, wired by `SceneManager`. A scene
  with no physical mesh keeps `scene.environment === null` and renders byte-identically to
  before this module existed.
- **Invisible to house materials.** `scene.environment` is read only by three's
  lighting-model materials. Every Luxar material is a `ShaderMaterial` / `NodeMaterial`
  with its own fragment code that never samples an environment, so setting it changes
  nothing about points, lines, splats or house meshes. The unit tests assert this rather
  than assume it.

## Backends

The PMREM generator is backend-specific: `three`'s for `WebGLRenderer`, `three/webgpu`'s
for `WebGPURenderer`. The WebGPU one is fetched through `rendering/tsl/registry.ts`
(`environment.PMREMGenerator`) so the lazy `three/webgpu` chunk stays lazy; it is always
loaded before a WebGPU material — hence before this — can exist. `RoomEnvironment` itself
imports only `three` core and is converted to node materials by the WebGPU renderer.

## Lifetime

Owned by `SceneManager`, created in `init()` next to the scene, disposed in `dispose()`
before the renderer (the PMREM target is that renderer's GPU resource). A WebGL context
restore rebuilds an existing target in place but preserves laziness when no physical
material has requested one. It survives a dataset switch — it is scene-level and cheap
to keep.
