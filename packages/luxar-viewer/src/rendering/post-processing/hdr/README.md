# HDR readback + capture utilities

Shared HDR plumbing: a unified backend-aware pixel-readback primitive
plus the EXR-log helper that goes with it.

| File             | Role                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pixel-utils.ts` | `halfFloatToFloat32` / `float32ToHalfFloat` converters + the pure row helpers `flipPixelsVerticallyRGBA` / `compactWebGPUReadbackRows` + `readPixelsCompactAsync(renderer, caps, opts)` — unified WebGL2/WebGPU readback (keyed by the `TexelKind` discriminator `'rgba8' \| 'rgba16f' \| 'rgba32f'`) that hides backend dispatch, WebGPU 256-byte row-padding compaction, and canonical top-down Y orientation in one place |
| `capture.ts`     | `formatHDRExrLogLine(width, height, isHalfFloat, byteLength)` — the size-tagged log line printed after a successful EXR encode                                                                                                                                                                                                                                                                                               |

Consumers: `post-processing-manager/capture.ts` (all three capture paths),
**`rendering/picking/picking-system.ts`** (uses `readPixelsCompactAsync` for the
pick-buffer readback — the shared primitive keeps post-processing and picking on
exactly the same WebGL2/WebGPU branch and row-padding semantics).
