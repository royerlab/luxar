# Data-Worker Projection

WASM-backed nD→3D projection kernels for **Lines** and **GSplats**. These
are the worker-side bodies that the main-thread `data/scene-loader/process/`
stage dispatches to via Comlink (`projectLinesTo3D`, `projectGSplatsTo3D`
on the `DataWorkerAPI`), or runs in-process via `in-process.ts`.

**Points are not here.** Points projection is memory-bandwidth-bound and
pairs with a zero-allocation accumulator, so it runs on the **main thread**
(still WASM-accelerated) in `data/points/projection.ts` — see that module
and `getPointsBackend` in `in-process.ts`. All three geometries thus share
the same WASM kernels; only Lines/GSplats are worker-offloaded.

## Files

| File             | Geometry | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lines.ts`       | Lines    | Clips segments with `clip_segments_batch` (slicePosition + tolerance + displayDims), interpolates clipped endpoints to 3D via `interpolate_clipped_positions`, then interpolates per-vertex colors / widths / sharpness / scalars with `interpolate_colors_batch` + `interpolate_scalars_batch`, finishes with `calculate_segment_lengths` and `compute_joint_codes` (per-endpoint joint code: `0` free end, `-1` slice-clipped, `-2` degree-≥​3 hub, or a signed reference to the partner segment's storage slot — purely topological, so the shader measures the bend angle itself in screen space). |
| `gsplats.ts`     | GSplats  | Pre-filters discrete hidden dims (half-step threshold, TS), coerces colors to normalized f32, then makes a **single fused `project_gsplats_nd_to_3d` call** (discrete gate → continuous Mahalanobis attenuation → visibility → compacted centers / 3D Cholesky / amplitudes / colors). A standard-3D fast path (`ndim===3`, `displayDims===[0,1,2]`) skips the kernel with a straight copy and **no** amplitude filtering. Hidden-dim classification is shared via `hidden-dims.ts`.                                                                                                                   |
| `hidden-dims.ts` | shared   | `classifyHiddenDims` (partitions non-displayed dims into extend_to_all / discrete / continuous, preserving displayDims order) and `isExtendToAll`. Used by `gsplats.ts` and the data-processor's param builder.                                                                                                                                                                                                                                                                                                                                                                                        |
| `constants.ts`   | shared   | Numeric thresholds shared across the dispatchers: `MIN_AMPLITUDE`, `EXTEND_TO_ALL_THRESHOLD`. (The gsplat truncation default now lives in `src/config/constants.ts` as `GSPLAT_DEFAULT_TRUNCATION_RADIUS`.)                                                                                                                                                                                                                                                                                                                                                                                            |
| `in-process.ts`  | shared   | Runs the dispatchers **on the main thread** against a lazily-initialized `WasmCtx` (`projectGSplatsInProcess` / `projectLinesInProcess`). The data-processors use this for the `useWebWorkers=false` / below-threshold path and as the worker-failure fallback — the same kernel code, no second copy. Also exposes `getPointsBackend(ndim)`, which resolves the WASM backend (compiled, or TS-reference for `ndim>16`) that `data/points/projection.ts` runs its kernels on.                                                                                                                          |

## Public surface

All three functions take a `ctx: WasmCtx` (from `../state.ts`) plus a
`params` object and return a Comlink-`transfer()`-wrapped result whose
typed-array buffers move to the main thread without copying.

```typescript
// Both share the same ProjectionViewState shape from ../types.ts.
// displayDims + slicePosition are consumed by every kernel; tolerance
// is read by Lines (clip bounds) but is required by GSplats only for
// API parity. (Points projection lives in data/points/projection.ts.)
export async function projectLinesTo3D(
  ctx,
  params
): Promise<{
  startPositions;
  endPositions;
  startColors;
  endColors;
  startWidths;
  endWidths;
  startSharpness;
  endSharpness;
  startScalars;
  endScalars; // per-vertex colormap scalars
  segmentLengths;
  startJointCode;
  endJointCode;
  visibleSegmentCount;
}>;
export async function projectGSplatsTo3D(
  ctx,
  params
): Promise<{
  centers3D;
  choleskyFactors3D;
  amplitudes;
  colors;
  sharpness;
  visibleCount;
}>;
```

## Invariants

- **Three-geometry symmetry.** One file per geometry, same name shape
  (`<geometry>.ts`), same `(ctx, params) → transfer(result)` signature,
  same `ProjectionViewState` input. Shared boundary checks live in
  `../validation.ts` (`validateProjectionInputs`,
  `validateLineSegmentReferences`); shared color/scalar coercion lives
  in `../../color-utils.ts` (`coerceColorsToFloat32`,
  `coerceScalarsToFloat32`, `fillColorsWhite`).
- **WASM is required, not optional.** Each kernel calls
  `requireWasm(ctx)` first — a worker without WASM throws here rather
  than silently falling back. The TypeScript fallback path is one level
  up, in the `initWasm` shim in `../../../wasm/`, which substitutes a pure-
  TS implementation of every batch function before the worker reaches
  these files.
- **Zero-copy results.** Every return goes through `comlink.transfer()`
  with the full list of result-array buffers, so the main thread adopts
  them without an intermediate copy. (Points' zero-alloc accumulator
  `outputBuffers` path lives with its main-thread projection in
  `data/points/projection.ts`.)
- **`extend_to_all` is geometry-aware.** GSplats receives an explicit
  `extendToAllDims` index list and removes those dims from the active
  hidden-dim set before Mahalanobis attenuation. Lines has no
  `extend_to_all` step here — the caller already mutates the tolerance
  array to `EXTEND_TO_ALL_TOLERANCE` upstream in
  `scene-loader/process/data-processor-lines.ts`. (Points fold extend
  dims into the display set for `calculate_effective_radii`, on the main
  thread in `data/points/projection.ts`.)
- **Early-exit on empty visibility.** When the visibility mask resolves
  to zero, both kernels return empty typed arrays of the correct element
  type instead of running the compact / interpolate steps. This keeps
  the worker fast on hidden-slice navigation.

## See also

- `../README.md` — overview of all `data-worker/` task groups.
- `../validation.ts` — shared JS→WASM boundary checks.
- `../../color-utils.ts` — `coerceColorsToFloat32`, `fillColorsWhite`.
- `../../../data/scene-loader/process/README.md` — main-thread half of
  the projection (`projectLinesTo3DUsingWorker`,
  `projectGSplatsTo3DUsingWorker` dispatchers + worker thresholds).
- `../../../data/lines/projection.ts`,
  `../../../data/gsplats/projection.ts` — main-thread fallback
  kernels that mirror these files when `useWebWorkers` is off.
- `../../../wasm/` — compiled WASM module + the pure-TS fallback that
  satisfies the same `wasmModule.*` surface these files call.
