# FXAA effect

Single-pass FXAA Quality preset applied to the LDR output of the mega-shader.
Self-contained — every intra-trio import is a sibling within `fxaa/`.

| File          | Role                                                                                                             |
| ------------- | ---------------------------------------------------------------------------------------------------------------- |
| `pass.ts`     | `FxaaPass` class — owns one FullscreenPass with the FXAA material; exposes `setSize` + `render(renderer, input)` |
| `shaders.ts`  | GLSL3 vertex + fragment + `FXAA_SOURCE: ShaderSource` record                                                     |
| `fxaa.tsl.ts` | TSL/WebGPU factory (`fxaaWebGPUFactory`) — referenced by `shaders.ts`                                            |

Consumers: orchestrator + `post-processing-manager/resource-lifecycle.ts`
(allocates/disposes the pass), and the GLSL/TSL parity harness (`FXAA_SOURCE`).
