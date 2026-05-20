/**
 * Worker-side types passed across the Comlink RPC boundary.
 *
 * `EffectiveRadiusConfig` and `ProjectionViewState` are also defined
 * authoritatively in `data/points/effective-radius-calculator.ts`
 * (canonical) — the duplicate here keeps the worker's input/output
 * shape self-documenting at task boundaries. Worker-internal use only.
 *
 * `PointsOutputBuffers` is re-exported from `data-worker.ts` because
 * the TransferableAccumulator pattern on the main thread needs the
 * shape.
 */

/**
 * Configuration for effective radius calculation (passed from main thread)
 */
export interface EffectiveRadiusConfig {
  spatialExtendDims: boolean[];
  maxRadius: number;
}

/**
 * View state for projection (subset of main thread ViewState)
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
