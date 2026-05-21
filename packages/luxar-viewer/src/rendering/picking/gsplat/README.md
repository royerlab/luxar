# GSplat picking

Per-geometry picking sources for Gaussian splats. Self-contained — no cross-geometry imports.

| File              | Role                                                                                                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `material.ts`     | GLSL3 `THREE.ShaderMaterial` wrapper (`GSplatPickingMaterial`). Tighter 1.5σ truncation than the visual material; always max-projection (no ray-integration)                                                 |
| `material-tsl.ts` | WebGPU `NodeMaterial` counterpart (`GSplatPickingTSLMaterial`). Mirrors the GLSL wrapper one-for-one over the same `CameraAwareMaterial` contract; always 1.5σ + max-projection                              |
| `pick.tsl.ts`     | TSL node factory (`gsplatPickWebGPUFactory` + `buildGSplatPickTSLNodesFromUniforms`). Used by the WebGPU material above and by the GLSL/TSL parity harness                                                   |
| `shaders.ts`      | GLSL3 vertex/fragment source strings + `GSPLAT_PICK_SOURCE: ShaderSource`. Source-of-truth for the GLSL path; reuses `GLSL_SANITIZE_FUNCTIONS` + `invalidCov2D` parity guards from the visual gsplat shaders |
