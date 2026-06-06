/**
 * Lines "empty payload" factory.
 *
 * The nD → 3D line clipping/interpolation math lives in exactly one
 * place: the worker dispatcher `workers/data-worker/projection/lines.ts`
 * (run on a worker, or on the main thread via
 * `workers/data-worker/projection/in-process.ts`), which batches through
 * the WASM `clip_segments_batch` / `interpolate_*` kernels (with the
 * uncapped `wasm/typescript/lines-clipping.ts` reference as fallback).
 *
 * This module used to carry a second, hand-written main-thread copy of
 * that math (`projectLinesTo3D`, `projectLinesTo3DWASM`,
 * `clipSegmentToSlice`, `lerp`/`lerpVec3`/`distance3D`, the
 * `getWasmModuleSync` footgun, and `initLinesWASM`). It was deleted so
 * there is a single projection implementation to keep in sync. Only the
 * tiny "no visible lines" constructor remains, alongside the loader that
 * consumes it.
 *
 * @module data/lines/projection
 */

import type { LinesMetadata, LoadedLinesData } from '../../types/lines';

/**
 * Construct the canonical "no visible lines at this slice" payload.
 * Mirrors `createEmptyPointsData` in `points/projection.ts` and
 * `createEmptyGSplatsData` in `gsplats/projection.ts`.
 */
export function createEmptyLinesData(attrs: LinesMetadata): LoadedLinesData {
  // `scalars` is omitted (optional + undefined) to match
  // `LoadedPointsData`. The downstream truthy check
  // (`if (data.scalars)`) treats undefined and null identically, so
  // existing call sites are unaffected.
  return {
    positions: new Float32Array(0),
    segments: new Uint32Array(0),
    widths: new Float32Array(0),
    colors: null,
    sharpness: null,
    segmentCount: 0,
    vertexCount: 0,
    ndim: attrs.ndim,
  };
}
