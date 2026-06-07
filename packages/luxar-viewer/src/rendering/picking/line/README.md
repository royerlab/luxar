# Line picking

Per-geometry picking sources for lines. Self-contained — no cross-geometry imports.

| File              | Role                                                                                                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `material.ts`     | GLSL3 `THREE.ShaderMaterial` wrapper (`LinePickingMaterial`). Recompiles on `needsUpdate`                                                                                  |
| `material-tsl.ts` | WebGPU `NodeMaterial` counterpart (`LinePickingTSLMaterial`). Same `CameraAwareMaterial` contract; rebuilds the TSL node graph on ortho flip                               |
| `pick.tsl.ts`     | TSL node factory (`linePickWebGPUFactory` + `buildLinePickTSLNodesFromUniforms`). Used by the WebGPU material above and by the GLSL/TSL parity harness                     |
| `shaders.ts`      | GLSL3 vertex/fragment source strings + `LINE_PICK_SOURCE: ShaderSource`. Uses full pick width (lines are already narrow); cap factor in fragment matches the visual shader |
