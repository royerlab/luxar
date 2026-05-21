# Mega-shader

The heaviest post-processing effect: a fused fragment shader that runs
chromatic lens distortion → bloom mix → detector noise → EOG → tone
mapping → vignette → sRGB encoding in a single pass. Self-contained —
every intra-quartet import is a sibling within `mega/`.

| File              | Role                                                                                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `material.ts`     | `MegaShaderMaterial extends THREE.ShaderMaterial` — uniform layout, `#define` toggles for each effect (bloom/lens/noise/vignette/tone-mapping-mode/capture)     |
| `material-tsl.ts` | `MegaShaderTSLMaterial extends NodeMaterial` — WebGPU counterpart mirroring the same setter surface                                                              |
| `shader.glsl.ts`  | GLSL3 vertex + fragment + `MEGA_SOURCE: ShaderSource`. **Picking-parity contract**: `applyDistortion` (lines 117–128) is byte-equivalent to `picking-system/lens-distortion.ts` — do not edit without updating the picking port |
| `shader.tsl.ts`   | TSL/WebGPU factory (`megaWebGPUFactory`) and `LuxarToneMappingMode` enum — referenced by `material-tsl.ts` and by `shader.glsl.ts` (for the WebGPU branch of `MEGA_SOURCE`) |

Consumers: `material-manager` + `material-manager/factories.ts` (instantiates
the two material classes), the orchestrator (holds an instance as `megaShader`
+ `megaPass`), and the GLSL/TSL parity harness (`MEGA_SOURCE` + `megaWebGPUFactory`).
