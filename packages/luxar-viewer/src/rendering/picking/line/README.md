# Line picking

Per-geometry picking sources for lines. Self-contained — no cross-geometry imports.

| File                  | Role                                                                                                                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `material.ts`         | GLSL3 `THREE.ShaderMaterial` wrapper (`LinePickingMaterial`). Selects the screen-space or capsule shader pair from the same `?linePrimitive=` resolution the visual material uses. Recompiles on `needsUpdate` |
| `material-tsl.ts`     | WebGPU `NodeMaterial` counterpart (`LinePickingTSLMaterial`). Same `CameraAwareMaterial` contract; every rebuild (ortho flip, texture rebind, clone) dispatches to the factory matching the resolved primitive |
| `pick.tsl.ts`         | TSL node factory for the screen-space primitive (`linePickWebGPUFactory` + `buildLinePickTSLNodesFromUniforms`). Used by the WebGPU material above and by the GLSL/TSL parity harness                          |
| `shaders.ts`          | Screen-space GLSL3 vertex/fragment source strings + `LINE_PICK_SOURCE: ShaderSource`. Uses full pick width (lines are already narrow); cap factor in fragment matches the visual shader                        |
| `pick-capsule.tsl.ts` | TSL node factory for the capsule primitive (#1352, `capsuleLinePickWebGPUFactory`): the capsule stencil vertex (half-disc bisector joints) with pick IDs, quartic-profile brightness fragment                  |
| `shaders-capsule.ts`  | Capsule GLSL3 sources + `CAPSULE_LINE_PICK_SOURCE: ShaderSource` — the GLSL twin of `pick-capsule.tsl.ts`                                                                                                      |

(The #1352 campaign also shipped a `volumetric` pick pair mirroring the
closed-form volumetric primitive's stadium stencil; it was deleted with that
primitive — see `../../materials/line/README.md` for the history and the git
branch point.)

## The capsule pick pass (#1352 — the default since the flip)

The capsule pick shaders duplicate the visual
capsule's vertex stage exactly — stencil-local corners, half-disc bisector
joints (same shared joint-code cap rule, same AA-ramp deficit composition) — and the
fragment shades the same quartic
profile of the 2D point-to-segment distance, so `brightness = profile ×
fade` tracks the visible pixels one-for-one (the capsule is peak-shaped by
construction; there is no separate peak lane to select). Per-element alpha
and node opacity are ignored, matching the pick contract of the
screen-space primitive. Output contract:
`vec4(nodeId, elementId-low16, brightness, elementId-high16)`,
`gl_FragDepth = 1 − brightness`. Footprint
agreement is pinned in `tsl-shader-parity.spec.ts` against the fat visual
footprint (1-px quantisation ribbon).
