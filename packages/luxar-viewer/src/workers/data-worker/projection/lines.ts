/**
 * Project Lines from nD to 3D with segment clipping.
 *
 * The single Lines projection implementation — run on a worker, or on the
 * main thread via `in-process.ts` (the former hand-written main-thread
 * copy in `data/lines/projection.ts` was deleted in W4). Uses WASM batch
 * functions:
 * 1. Clips all segments to the nD slice
 * 2. Projects clipped endpoints to 3D
 * 3. Interpolates per-vertex attributes (colors, widths, sharpness, scalars)
 * 4. Calculates segment lengths and tracks clipping
 *
 * WASM batch functions provide 3-5x speedup over per-segment TypeScript loops.
 */

import { transfer } from 'comlink';
import { pickBackend, type WasmCtx } from '../state';
import { validateProjectionInputs, validateLineSegmentReferences } from '../validation';
import { coerceColorsToFloat32, coerceScalarsToFloat32, fillColorsWhite } from '../../color-utils';
import type { ProjectionViewState } from '../types';

export async function projectLinesTo3D(
  ctx: WasmCtx,
  params: {
    positions: Float32Array;
    segments: Uint32Array;
    widths: Float32Array;
    colors: Float32Array | Uint8Array | Uint16Array | null;
    sharpness: Float32Array | null;
    /**
     * Optional per-vertex scalar attribute for colormap-mode Lines.
     * Accepted dtypes: `Float32Array` (passes through zero-copy),
     * `Float16Array` (element-wise expand), `Uint8Array` (normalized by
     * `1/255` to align with the colormap shader's `[0, 1]` contract).
     * When `null` the worker emits empty `startScalars`/`endScalars`
     * arrays and the loader leaves the geometry's scalar attribute
     * unallocated.
     */
    scalars: Float32Array | Float16Array | Uint8Array | null;
    /**
     * View state for projection. Same `ProjectionViewState` shape every
     * `project*To3D` worker function accepts — keeps the worker API
     * uniform across node types. All three fields are consumed: the
     * WASM `clip_segments_batch` step reads `slicePosition` + `tolerance`
     * + `displayDims` to decide per-segment visibility and to clip
     * partially-visible segments at the slice boundary.
     */
    viewState: ProjectionViewState;
    ndim: number;
    segmentCount: number;
  }
): Promise<{
  startPositions: Float32Array;
  endPositions: Float32Array;
  startColors: Float32Array;
  endColors: Float32Array;
  startWidths: Float32Array;
  endWidths: Float32Array;
  startSharpness: Float32Array;
  endSharpness: Float32Array;
  /** Per-segment start scalar (empty Float32Array when input scalars=null). */
  startScalars: Float32Array;
  /** Per-segment end scalar (empty Float32Array when input scalars=null). */
  endScalars: Float32Array;
  segmentLengths: Float32Array;
  startClipped: Uint8Array;
  endClipped: Uint8Array;
  visibleSegmentCount: number;
}> {
  const wasmModule = pickBackend(ctx, params.ndim); // >16D -> uncapped TS reference

  const { positions, segments, widths, colors, sharpness, scalars, viewState, ndim, segmentCount } =
    params;
  const { displayDims, slicePosition, tolerance } = viewState;

  // The shared projection validator handles displayDims and the basic
  // ndim/positions sanity check; we then check segment-vertex bounds
  // explicitly because positions length depends on max referenced vertex
  // (not numItems = 1), and finally the per-vertex attribute lengths.
  validateProjectionInputs(
    'projectLinesTo3D',
    positions,
    displayDims,
    slicePosition,
    ndim,
    1,
    ndim
  );
  if (tolerance.length < ndim) {
    throw new Error(
      `projectLinesTo3D: tolerance too short (got ${tolerance.length}, expected ≥ ${ndim})`
    );
  }
  validateLineSegmentReferences('projectLinesTo3D', segments, segmentCount, positions, ndim, {
    widths,
    colors: colors ?? undefined,
    sharpness: sharpness ?? undefined,
    scalars: scalars ?? undefined,
  });

  // Convert input arrays to WASM-compatible formats
  const slicePosF32 = new Float32Array(slicePosition);
  const toleranceF32 = new Float32Array(tolerance);
  const displayDimsU32 = new Uint32Array(displayDims);

  // Step 1: Batch clip all segments using WASM
  const visibility = new Uint8Array(segmentCount);
  const t1Params = new Float32Array(segmentCount);
  const t2Params = new Float32Array(segmentCount);

  const visibleCount = wasmModule.clip_segments_batch(
    positions,
    segments,
    slicePosF32,
    toleranceF32,
    displayDimsU32,
    ndim,
    segmentCount,
    visibility,
    t1Params,
    t2Params
  );

  // Early exit if no visible segments
  if (visibleCount === 0) {
    const emptyPositions = new Float32Array(0);
    const emptyColors = new Float32Array(0);
    const emptyScalars = new Float32Array(0);
    const emptyFlags = new Uint8Array(0);

    return transfer(
      {
        startPositions: emptyPositions,
        endPositions: new Float32Array(0),
        startColors: emptyColors,
        endColors: new Float32Array(0),
        startWidths: emptyScalars,
        endWidths: new Float32Array(0),
        startSharpness: new Float32Array(0),
        endSharpness: new Float32Array(0),
        startScalars: new Float32Array(0),
        endScalars: new Float32Array(0),
        segmentLengths: new Float32Array(0),
        startClipped: emptyFlags,
        endClipped: new Uint8Array(0),
        visibleSegmentCount: 0,
      },
      [emptyPositions.buffer, emptyColors.buffer, emptyScalars.buffer, emptyFlags.buffer]
    );
  }

  // Step 2: Interpolate clipped positions to 3D using WASM
  const startPositions = new Float32Array(visibleCount * 3);
  const endPositions = new Float32Array(visibleCount * 3);

  wasmModule.interpolate_clipped_positions(
    positions,
    segments,
    visibility,
    t1Params,
    t2Params,
    displayDimsU32,
    ndim,
    segmentCount,
    startPositions,
    endPositions
  );

  // Step 3: Interpolate colors using WASM
  // Convert colors to Float32Array if needed (WASM expects Float32Array)
  const startColors = new Float32Array(visibleCount * 3);
  const endColors = new Float32Array(visibleCount * 3);

  if (colors) {
    wasmModule.interpolate_colors_batch(
      coerceColorsToFloat32(colors),
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startColors,
      endColors
    );
  } else {
    fillColorsWhite(startColors, visibleCount);
    fillColorsWhite(endColors, visibleCount);
  }

  // Step 4: Interpolate widths using WASM
  const startWidths = new Float32Array(visibleCount);
  const endWidths = new Float32Array(visibleCount);

  wasmModule.interpolate_scalars_batch(
    widths,
    segments,
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startWidths,
    endWidths
  );

  // Step 5: Interpolate sharpness using WASM
  const startSharpness = new Float32Array(visibleCount);
  const endSharpness = new Float32Array(visibleCount);

  if (sharpness) {
    wasmModule.interpolate_scalars_batch(
      sharpness,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startSharpness,
      endSharpness
    );
  } else {
    // Default sharpness is 1.0
    startSharpness.fill(1.0);
    endSharpness.fill(1.0);
  }

  // Step 6: Interpolate per-vertex scalars using WASM (colormap mode).
  // Empty arrays are returned when `scalars` is null so the loader can
  // skip allocating the geometry's scalar attribute. We coerce
  // Float16/Uint8 to Float32 first since `interpolate_scalars_batch`
  // expects Float32 inputs (same path widths/sharpness take).
  let startScalars: Float32Array;
  let endScalars: Float32Array;
  if (scalars) {
    const scalarsF32 = coerceScalarsToFloat32(scalars);
    startScalars = new Float32Array(visibleCount);
    endScalars = new Float32Array(visibleCount);
    wasmModule.interpolate_scalars_batch(
      scalarsF32,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startScalars,
      endScalars
    );
  } else {
    startScalars = new Float32Array(0);
    endScalars = new Float32Array(0);
  }

  // Step 7: Calculate segment lengths using WASM
  const segmentLengths = new Float32Array(visibleCount);
  wasmModule.calculate_segment_lengths(startPositions, endPositions, visibleCount, segmentLengths);

  // Step 8: Mark clipped endpoints using WASM
  const startClipped = new Uint8Array(visibleCount);
  const endClipped = new Uint8Array(visibleCount);

  wasmModule.mark_clipped_endpoints(
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startClipped,
    endClipped
  );

  // Build transferable list
  const transferables: ArrayBuffer[] = [
    startPositions.buffer as ArrayBuffer,
    endPositions.buffer as ArrayBuffer,
    startColors.buffer as ArrayBuffer,
    endColors.buffer as ArrayBuffer,
    startWidths.buffer as ArrayBuffer,
    endWidths.buffer as ArrayBuffer,
    startSharpness.buffer as ArrayBuffer,
    endSharpness.buffer as ArrayBuffer,
    startScalars.buffer as ArrayBuffer,
    endScalars.buffer as ArrayBuffer,
    segmentLengths.buffer as ArrayBuffer,
    startClipped.buffer as ArrayBuffer,
    endClipped.buffer as ArrayBuffer,
  ];

  return transfer(
    {
      startPositions,
      endPositions,
      startColors,
      endColors,
      startWidths,
      endWidths,
      startSharpness,
      endSharpness,
      startScalars,
      endScalars,
      segmentLengths,
      startClipped,
      endClipped,
      visibleSegmentCount: visibleCount,
    },
    transferables
  );
}
