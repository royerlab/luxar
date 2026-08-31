/**
 * GSplats "empty payload" factory.
 *
 * The nD → 3D GSplat projection math lives in exactly one place: the
 * worker dispatcher `workers/data-worker/projection/gsplats.ts` (run on
 * a worker, or on the main thread via
 * `workers/data-worker/projection/in-process.ts`). This module used to
 * carry a second, hand-written main-thread copy of that math; it was
 * deleted so there is a single projection implementation to keep in sync
 * with the WASM / TS-reference kernels. Only the tiny "no visible splats"
 * constructor remains here, alongside the loader that consumes it.
 *
 * @module data/gsplats/projection
 */

import type { GSplatsMetadata, LoadedGSplatsData } from '../../types/gsplats';

/**
 * Construct the canonical "no visible splats at this slice" payload.
 * Mirrors `createEmptyPointsData` in `points/projection.ts` and
 * `createEmptyLinesData` in `lines/projection.ts`.
 */
export function createEmptyGSplatsData(attrs: GSplatsMetadata): LoadedGSplatsData {
  return {
    positions: new Float32Array(0),
    amplitudes: new Float32Array(0),
    choleskyFactors: new Float32Array(0),
    colors: null,
    labelIndices: attrs.has_label_ids ? new Uint32Array(0) : undefined,
    labelVocabulary: attrs.label_vocabulary
      ? Object.entries(attrs.label_vocabulary).map(([id, name]) => ({ id, name }))
      : undefined,
    splatCount: 0,
    ndim: attrs.ndim,
  };
}
