# Data-Worker Projection

WASM-backed nD→3D projection kernels for the three Luxar geometry
types — **Points**, **Lines**, **GSplats**. One file per geometry,
parallel shape, shared validation and color helpers. These are the
worker-side bodies that the main-thread `data/scene-loader/process/`
stage dispatches to via Comlink (`projectPointsTo3D`,
`projectLinesTo3D`, `projectGSplatsTo3D` on the `DataWorkerAPI`).

## Files

| File         | Geometry | Role                                                                                                                                                                                                                                                                                                                                                                              |
| ------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `points.ts`  | Points   | Extracts 3D positions via `extract_3d_positions`, optionally runs `calculate_effective_radii` (treating `extend_to_all` dims as displayed), filters zero-radius points with `radii_to_visibility_mask` + `compact_by_mask`, computes bounds via `calculate_bounds_3d`. Supports the TransferableAccumulator zero-allocation path via `outputBuffers`.                             |
| `lines.ts`   | Lines    | Clips segments with `clip_segments_batch` (slicePosition + tolerance + displayDims), interpolates clipped endpoints to 3D via `interpolate_clipped_positions`, then interpolates per-vertex colors / widths / sharpness / scalars with `interpolate_colors_batch` + `interpolate_scalars_batch`, finishes with `calculate_segment_lengths` and `mark_clipped_endpoints`.          |
| `gsplats.ts` | GSplats  | Pre-filters discrete hidden dims (half-step threshold, TS), runs `compute_gsplats_attenuation` over the continuous hidden dims for Mahalanobis-based visibility + attenuation, then `extract_3d_positions` + `compact_by_mask` for centers, `extract_visible_cholesky_3d` for the 3D Cholesky submatrices, and `compact_attenuated_amplitudes` for the per-splat amplitude scale. |

## Public surface

All three functions take a `ctx: WasmCtx` (from `../state.ts`) plus a
`params` object and return a Comlink-`transfer()`-wrapped result whose
typed-array buffers move to the main thread without copying.

```typescript
// All three share the same ProjectionViewState shape from ../types.ts.
// displayDims + slicePosition are consumed by every kernel; tolerance
// is read by Points (extend_to_all detection) and Lines (clip bounds)
// but is required by GSplats only for API parity.
export async function projectPointsTo3D(
  ctx,
  params
): Promise<{
  positions3D;
  colors;
  radii;
  sharpness;
  visibleCount;
  bounds;
  outputBuffers?: PointsOutputBuffers; // TransferableAccumulator
}>;
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
  startClipped;
  endClipped;
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
  them without an intermediate copy. Points additionally supports a
  caller-provided `outputBuffers` for fully pre-allocated zero-alloc
  loops (the TransferableAccumulator pattern).
- **`extend_to_all` is geometry-aware.** Points detects extend-to-all
  via `tolerance[d] >= 1e9` and treats those dims as displayed inside
  `calculate_effective_radii`. GSplats receives an explicit
  `extendToAllDims` index list and removes those dims from the active
  hidden-dim set before Mahalanobis attenuation. Lines has no
  `extend_to_all` step here — the caller already mutates the tolerance
  array to `EXTEND_TO_ALL_TOLERANCE` upstream in
  `scene-loader/process/data-processor-lines.ts`.
- **Early-exit on empty visibility.** When the visibility mask resolves
  to zero, every kernel returns empty typed arrays of the correct
  element type (preserving the input color dtype for Points) instead of
  running the compact / interpolate steps. This keeps the worker fast
  on hidden-slice navigation.
- **No Points process step upstream.** Unlike Lines and GSplats, Points
  has no `data-processor-points.ts` — the spatial-index loader already
  produces 3D-ready buffers, so `projectPointsTo3D` is only invoked by
  loaders that _do_ want async nD compaction (effective-radius +
  extend-to-all paths). See
  `../../../data/scene-loader/commit/commit-points-geometry.ts` for the
  rationale.

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
