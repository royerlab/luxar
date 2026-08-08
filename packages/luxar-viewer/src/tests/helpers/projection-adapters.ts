/**
 * Test-only adapters bridging the legacy main-thread projection call
 * shapes to the surviving production code.
 *
 * The hand-written main-thread projection copies in
 * `data/{gsplats,lines}/projection.ts` were deleted in W4 — the nD → 3D
 * math now lives solely in the worker dispatchers
 * (`workers/data-worker/projection/{gsplats,lines}.ts`) and the WASM /
 * TS-reference kernels. These adapters let the still-valuable behavioral
 * suites (clip-segment property tests, scalar end-to-end, golden
 * equivalence) keep their assertions while exercising the *live* code:
 *
 *   - `clipSegmentToSlice` wraps the TS-reference clipper
 *     (`wasm/typescript/lines-clipping.ts`), which is the implementation
 *     the worker uses as its fallback.
 *   - `projectGSplatsViaDispatcher` / `projectLinesViaDispatcher` run the
 *     worker dispatchers in-process against a caller-supplied backend
 *     (TypeScript fallback or compiled WASM), returning the same
 *     `Processed*Data` shape the data-processors map to.
 *
 * Production code never imports this module.
 *
 * @module tests/helpers/projection-adapters
 */

import { clip_segment_single } from '../../wasm/typescript/lines-clipping';
import { isExtendToAll } from '../../workers/data-worker/projection/hidden-dims';
import { projectGSplatsTo3D } from '../../workers/data-worker/projection/gsplats';
import { projectLinesTo3D } from '../../workers/data-worker/projection/lines';
import type { WasmCtx } from '../../workers/data-worker/state';
import type { WasmModule } from '../../wasm/types';
import type {
  LoadedGSplatsData,
  GSplatsViewState,
  ProcessedGSplatsData,
} from '../../types/gsplats';
import type { LoadedLinesData, ProcessedLinesData, ClippedSegment } from '../../types/lines';

// ---------------------------------------------------------------------------
// Per-segment clip primitive (legacy `clipSegmentToSlice` object shape)
// ---------------------------------------------------------------------------

/**
 * Adapter reproducing the deleted `clipSegmentToSlice` object API on top
 * of the TS-reference `clip_segment_single` (which returns `[visible,
 * t1, t2]`). The clipped 3D display positions are reconstructed from the
 * interpolation parameters, exactly as the old function did.
 */
export function clipSegmentToSlice(
  p1: number[],
  p2: number[],
  slicePosition: readonly number[],
  tolerance: readonly number[],
  displayDims: readonly number[]
): ClippedSegment {
  const ndim = p1.length;
  const res = clip_segment_single(
    Float32Array.from(p1),
    Float32Array.from(p2),
    Float32Array.from(slicePosition as number[]),
    Float32Array.from(tolerance as number[]),
    Uint32Array.from(displayDims as number[]),
    ndim
  );
  if (res[0] !== 1.0) {
    return { p1: [], p2: [], t1: 0, t2: 0, visible: false };
  }
  const t1 = res[1];
  const t2 = res[2];
  const d1 = displayDims.map((d) => p1[d] + t1 * (p2[d] - p1[d]));
  const d2 = displayDims.map((d) => p1[d] + t2 * (p2[d] - p1[d]));
  while (d1.length < 3) d1.push(0);
  while (d2.length < 3) d2.push(0);
  return { p1: d1, p2: d2, t1, t2, visible: true };
}

// ---------------------------------------------------------------------------
// In-process dispatcher adapters (legacy `projectGSplats` / `projectLinesTo3D`)
// ---------------------------------------------------------------------------

/** Wrap a backend module as a main-thread `WasmCtx` for the dispatchers. */
function ctxFor(backend: WasmModule): WasmCtx {
  return { wasm: backend, tsFallback: backend };
}

/**
 * Run the GSplats dispatcher in-process against `backend`, deriving the
 * discrete / extend_to_all dim sets from the view state exactly as the
 * data-processor does, and mapping to `ProcessedGSplatsData`.
 */
export async function projectGSplatsViaDispatcher(
  backend: WasmModule,
  loaded: LoadedGSplatsData,
  viewState: GSplatsViewState,
  truncate = 3.0
): Promise<ProcessedGSplatsData> {
  const discreteDims: number[] = [];
  const discreteSteps: Record<number, number> = {};
  const extendToAllDims: number[] = [];
  if (viewState.dimensions) {
    for (let d = 0; d < viewState.dimensions.length; d++) {
      if (viewState.displayDims.includes(d)) continue;
      if (isExtendToAll(viewState.tolerance[d])) {
        extendToAllDims.push(d);
      } else if (viewState.dimensions[d]?.discrete) {
        discreteDims.push(d);
        discreteSteps[d] = viewState.dimensions[d].step ?? 1.0;
      }
    }
  }
  const result = await projectGSplatsTo3D(ctxFor(backend), {
    positions: loaded.positions,
    choleskyFactors: loaded.choleskyFactors,
    amplitudes: loaded.amplitudes,
    colors: loaded.colors,
    sharpness: null,
    viewState: {
      displayDims: viewState.displayDims,
      slicePosition: viewState.slicePosition,
      tolerance: viewState.tolerance,
    },
    ndim: loaded.ndim,
    splatCount: loaded.splatCount,
    discreteDims,
    discreteSteps,
    extendToAllDims,
    truncate,
  });
  return {
    centers3D: result.centers3D,
    choleskyFactors3D: result.choleskyFactors3D,
    amplitudes: result.amplitudes,
    colors: result.colors,
    splatCount: result.visibleCount,
  };
}

/**
 * Run the Lines dispatcher in-process against `backend`, mapping to
 * `ProcessedLinesData` (scalars omitted when the source had none).
 */
export async function projectLinesViaDispatcher(
  backend: WasmModule,
  loaded: LoadedLinesData,
  slicePosition: readonly number[],
  tolerance: readonly number[],
  displayDims: readonly number[]
): Promise<ProcessedLinesData> {
  const result = await projectLinesTo3D(ctxFor(backend), {
    positions: loaded.positions,
    segments: loaded.segments,
    widths: loaded.widths,
    colors: loaded.colors,
    sharpness: loaded.sharpness,
    scalars: loaded.scalars ?? null,
    viewState: { displayDims, slicePosition, tolerance },
    ndim: loaded.ndim,
    segmentCount: loaded.segmentCount,
  });
  // Presence follows the SOURCE alone (mirrors toProcessedLines in
  // data-processor-lines.ts): an empty slice keeps empty-but-defined
  // scalar fields so the geometry's hasScalars stamp survives.
  const hasScalars = !!loaded.scalars;
  return {
    startPositions: result.startPositions,
    endPositions: result.endPositions,
    startColors: result.startColors,
    endColors: result.endColors,
    startWidths: result.startWidths,
    endWidths: result.endWidths,
    startSharpness: result.startSharpness,
    endSharpness: result.endSharpness,
    startScalars: hasScalars ? result.startScalars : undefined,
    endScalars: hasScalars ? result.endScalars : undefined,
    segmentLengths: result.segmentLengths,
    startJointCode: result.startJointCode,
    endJointCode: result.endJointCode,
    segmentCount: result.visibleSegmentCount,
  };
}
