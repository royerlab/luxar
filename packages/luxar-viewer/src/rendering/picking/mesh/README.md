# Mesh picking

Per-geometry picking sources for meshes. Self-contained — no cross-geometry imports beyond
`materials/mesh/appearance` (the shared cutoff default and the fraction clamp) and
`materials/_shared/glsl-lib` (the shared 16-bit id split).

| File                  | Role                                                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `material.ts`         | GLSL3 `THREE.ShaderMaterial` wrapper (`MeshPickingMaterial`)                                                                                           |
| `material-tsl.ts`     | WebGPU `NodeMaterial` counterpart (`MeshPickingTSLMaterial`). No rebuild path — every mode-dependent value is a runtime uniform                        |
| `pick.tsl.ts`         | TSL node factory (`meshPickWebGPUFactory` + `buildMeshPickTSLNodesFromUniforms`). Used by the WebGPU material above and by the GLSL/TSL parity harness |
| `shaders.ts`          | GLSL3 vertex/fragment source + `MESH_PICK_SOURCE: ShaderSource`. Element id from `gl_VertexID`; the same alpha cutout as the visual shader             |
| `pick-mode.ts`        | `MeshPickAwareMaterial` + `resolveMeshPickModeState` — the blending mode's two pick-pass consequences, derived in one place for both backends          |
| `provoking-vertex.ts` | Aligns the WebGL `flat` provoking vertex with WebGPU's when the extension exists; otherwise mesh index triples rotate cyclically                       |

## What makes this the odd one out

Three differences from the point/line/gsplat pick materials, all from
`docs/specs/MESH_NODE_SPEC.md` §6.5:

- **The element id is a built-in, not an attribute.** Mesh depth sorting permutes whole index
  triples, so there is no `aSortedIndex` indirection or `luxarElementIdParts()` call — the id is
  `gl_VertexID` (a stable per-**vertex** ordinal under an indexed draw), split by the shared
  `luxarElementIdSplit()`. A face ordinal would be renumbered on every slice change, since only
  the index buffer is rewritten; a vertex ordinal is invariant, and it indexes the per-vertex
  label CSR directly.
- **Camera-aware for half the usual reason.** A mesh has no screen-space footprint to size, so
  there is no resolution/FOV uniform, and the near fade's ortho test reads the
  projection matrix (`luxarIsOrthoProjection()`) — but `uNearCull` is bound and the material
  joins the camera broadcast, because the pick pass has to reproduce the visual near fade or a
  surface fading out of view would stay fully pickable. Matching the visual mesh material, which
  is camera-aware for exactly the same half. The fade is a runtime switch (`uNearFade`) that the
  pick render turns off for a `material="physical"` node: three's PBR material has no near fade,
  so a physical surface right in front of the camera is fully visible and must stay pickable.
- **`side` is synced from the visual material** rather than pinned to `DoubleSide`. The siblings'
  quads are view-facing; a mesh's back faces may be culled on screen, and a pick pass that
  rasterized them anyway would make an invisible interior face both pickable and
  depth-occluding.

## Two runtime uniforms, never defines

| Uniform         | Derived from                                 | Effect                                                                    |
| --------------- | -------------------------------------------- | ------------------------------------------------------------------------- |
| `uAlphaCutout`  | `isOpaqueMode(mode)`                         | applies the visual shader's identical `a < uAlphaCutoff` discard          |
| `uSurfaceDepth` | `isNormalMode(mode) \|\| isOpaqueMode(mode)` | real projected depth (front-most wins) vs brightness-as-depth (brightest) |

Both are pushed by `PickingSystem.renderPickBuffer` from the visual material's
`userData.blendingMode`, every pick render, through the single `setPickMode(mode)` entry point.
Keeping them uniforms is what makes a layers-panel mode switch a uniform write instead of a
recompile — and it is why `mesh-pick` is ONE codegen snapshot variant covering every mode.

## Pick corner and depth

- **Which triangle corner a pick reports.** WebGL and core WebGPU report the original _first_
  vertex. Compatibility-mode WebGPU samples `flat, either`, so its id is only guaranteed to be a
  corner of the hit triangle (§6.5).
  `provoking-vertex.ts` aligns GL when `WEBGL_provoking_vertex` exists. Otherwise the shared
  mesh index triples rotate cyclically, preserving winding while putting that corner last for GL.
- **The surface-depth VALUE.** GLSL writes `gl_FragCoord.z`; the TSL twin writes
  `pickFragmentDepthTSL()` (three's perspective or orthographic depth of the reconstructed
  view-space z, chosen per draw from the projection matrix — the same choice for every pick
  graph, so depths from graphs built under different cameras stay comparable). Both are monotone in distance over `[near, far] → [0, 1]`, and the
  pick buffer's depth is only ever used to ORDER fragments within one render, so "front-most wins"
  resolves identically. See the module doc in `shaders.ts` for when this would stop being benign.
