/**
 * Worker-side types passed across the Comlink RPC boundary.
 *
 * `EffectiveRadiusConfig` is sourced from the canonical definition
 * in `types/points.ts` so worker and main-thread paths can't drift.
 *
 * `PointsOutputBuffers` is re-exported from `data-worker.ts` because
 * the TransferableAccumulator pattern on the main thread needs the
 * shape.
 */

import type { EffectiveRadiusConfig } from '../../types/points';

export type { EffectiveRadiusConfig };

/**
 * View state for projection (subset of main thread ViewState).
 *
 * The worker accepts the minimal subset of ViewState fields its
 * projection kernels actually consume — keeping the worker payload
 * narrow rather than serializing the full main-thread ViewState.
 */
export interface ProjectionViewState {
  displayDims: readonly number[];
  slicePosition: readonly number[];
  tolerance: readonly number[];
}

/**
 * Pre-allocated output buffers for TransferableAccumulator pattern.
 * When provided, projection writes directly to these buffers for zero-allocation.
 */
export interface PointsOutputBuffers {
  positions3D: Float32Array;
  colors?: Float32Array | Uint8Array | Uint16Array | null;
  radii?: Float32Array | null;
  sharpness?: Float32Array | null;
}
