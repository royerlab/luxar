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
| `provoking-vertex.ts` | Aligns the WebGL `flat` provoking vertex with WebGPU's, so a pick reports the same triangle corner on both backends where the platform allows          |

## What makes this the odd one out

Three differences from the point/line/gsplat pick materials, all from
`docs/specs/MESH_NODE_SPEC.md` §6.5:

- **The element id is a built-in, not an attribute.** Mesh has no per-triangle depth sort, so
  there is no `aSortedIndex` indirection and no `luxarElementIdParts()` call — the id is
  `gl_VertexID` (a stable per-**vertex** ordinal under an indexed draw), split by the shared
  `luxarElementIdSplit()`. A face ordinal would be renumbered on every slice change, since only
  the index buffer is rewritten; a vertex ordinal is invariant, and it indexes the per-vertex
  label CSR directly.
- **Not camera-aware.** A mesh has no screen-space footprint to size, so there is no
  resolution/FOV/near-cull uniform and no camera broadcast — matching the visual mesh material,
  which the material manager routes to `staticMaterials` for the same reason.
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

## Known divergences, both deliberate

- **Which triangle corner a pick reports.** A `flat` varying comes from one corner, and GL fixes
  it to the _last_ vertex where WGSL samples the _first_. `provoking-vertex.ts` aligns them when
  `WEBGL_provoking_vertex` exists; where it does not, the contract stands as "_a_ corner vertex of
  the front-most triangle under the cursor" — the cursor is over the face, so every corner is an
  equally valid answer and no consumer may assume one.
- **The surface-depth VALUE.** GLSL writes `gl_FragCoord.z`; the TSL twin's `depth` node expands
  to a linear view-space depth. Both are monotone in distance over `[near, far] → [0, 1]`, and the
  pick buffer's depth is only ever used to ORDER fragments within one render, so "front-most wins"
  resolves identically. See the module doc in `shaders.ts` for when this would stop being benign.
