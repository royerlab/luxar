# Point picking

Per-geometry picking sources for points. Self-contained — no cross-geometry imports.

| File              | Role                                                                                                                                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `material.ts`     | GLSL3 `THREE.ShaderMaterial` wrapper (`PointPickingMaterial`). Camera-aware; tighter 80%-radius truncation than the visual material                                                                  |
| `material-tsl.ts` | WebGPU `NodeMaterial` counterpart (`PointPickingTSLMaterial`). Mirrors the GLSL wrapper one-for-one over the same `CameraAwareMaterial` contract                                                     |
| `pick.tsl.ts`     | TSL node factory (`pointPickWebGPUFactory` + `buildPointPickTSLNodesFromUniforms`). Used by the WebGPU material above and by the GLSL/TSL parity harness                                             |
| `shaders.ts`      | GLSL3 vertex/fragment source strings + `POINT_PICK_SOURCE: ShaderSource`. Reuses `GLSL_SANITIZE_FUNCTIONS` for footprint parity with the visual side; references `pick.tsl.ts` for the WebGPU branch |
