# Bloom effect

Threshold + mipmap-pyramid bloom that produces the soft-glow texture the
mega-shader additively blends into the LDR output. Self-contained — every
intra-trio import is a sibling within `bloom/`.

| File           | Role                                                                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chain.ts`     | `BloomChain` class — owns the mip pyramid, threshold pass, and the downsample/upsample chain. Material-swap semantics through a single FullscreenPass |
| `shaders.ts`   | GLSL3 vertex + threshold/downsample/upsample fragment shaders + three `ShaderSource` records (one per pass)                                            |
| `bloom.tsl.ts` | TSL/WebGPU factories (`bloomThresholdWebGPUFactory`, `bloomDownsampleWebGPUFactory`, `bloomUpsampleWebGPUFactory`) — referenced by `shaders.ts`        |

Consumers: orchestrator (`post-processing-manager.ts`), helper `resource-lifecycle.ts`
(allocates the chain), `settings.ts` (live-update bloom uniforms), and the GLSL/TSL
parity harness (imports `BLOOM_THRESHOLD_SOURCE`).
