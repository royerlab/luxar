# Mega-shader

The heaviest post-processing effect: a fused fragment shader that runs
bloom mix → chromatic lens distortion → detector noise → EOG → tone
mapping → vignette → sRGB encoding in a single pass. Bloom is mixed
into the HDR sample *before* lens distortion samples the buffer, so a
distorted sample picks up bloom at its distorted UV too. The four files
here reference only each other (plus the shared `ShaderSource` type
from `../../materials/_shared/`).

| File              | Role                                                                                                                                                                                                                                                                         |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `material.ts`     | `MegaShaderMaterial extends THREE.ShaderMaterial` — uniform layout, `#define` toggles for each effect (bloom/lens/noise/vignette/tone-mapping-mode/capture)                                                                                                                  |
| `material-tsl.ts` | `MegaShaderTSLMaterial extends NodeMaterial` — WebGPU counterpart mirroring the same setter surface                                                                                                                                                                          |
| `shader.glsl.ts`  | GLSL3 vertex + fragment + `MEGA_SOURCE: ShaderSource`. **Picking-parity contract**: the `applyDistortion` helper (Brown-Conrady radial + intrinsic matrix) is byte-equivalent to `picking/picking-system/lens-distortion.ts` — do not edit without updating the picking port |
| `shader.tsl.ts`   | TSL/WebGPU factory (`megaWebGPUFactory`), the `LuxarToneMappingMode` type, and the `MegaTSLConfig` toggle interface — referenced by `material-tsl.ts` and by `shader.glsl.ts` (for the WebGPU branch of `MEGA_SOURCE`)                                                       |

Consumers: `material-manager/factories.ts` (instantiates the two material
classes via the `MEGA_SHADER_FACTORIES` `{ glsl, tsl }` map), the
post-processing orchestrator (`post-processing-manager.ts` holds the active
instance as `megaShader` and the fullscreen pass as `megaPass`), and the
GLSL/TSL parity harness (`tests/e2e/harnesses/tsl-harness.ts`, via
`MEGA_SOURCE` + `megaWebGPUFactory`).
