# HDR readback + capture utilities

Shared HDR plumbing: a unified backend-aware pixel-readback primitive
plus the EXR-log helper that goes with it.

| File             | Role                                                                                                                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pixel-utils.ts` | `halfFloatToFloat32` / `float32ToHalfFloat` converters + `readPixelsCompactAsync(renderer, capabilities, opts)` — unified WebGL2/WebGPU readback that hides backend dispatch, WebGPU padding compaction, and canonical Y orientation |
| `capture.ts`     | `formatHDRExrLogLine(width, height, isHalfFloat, byteLength)` — the size-tagged log line printed after a successful EXR encode                                                                                                       |

Consumers: `post-processing-manager/capture.ts` (all three capture paths),
**`rendering/picking/picking-system.ts`** (uses `readPixelsCompactAsync` for the
pick-buffer readback — the shared primitive keeps post-processing and picking on
exactly the same WebGL2/WebGPU branch and row-padding semantics).
