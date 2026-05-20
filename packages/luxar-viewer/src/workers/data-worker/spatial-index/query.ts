/**
 * Spatial-index chunk query. Finds which chunks intersect the current
 * nD view frustum. Generic for Points/Lines/GSplats — the underlying
 * WASM kernel reads the chunk bounds buffer the same way for all three.
 */

import { requireWasm, type WasmCtx } from '../state';
import { validateChunkQueryInputs } from '../validation';

export async function querySpatialIndex(
  ctx: WasmCtx,
  params: {
    chunkBounds: Float32Array;
    slicePosition: Float32Array;
    tolerance: Float32Array;
    numChunks: number;
    ndim: number;
  }
): Promise<Uint32Array> {
  const wasmModule = requireWasm(ctx);

  const { chunkBounds, slicePosition, tolerance, numChunks, ndim } = params;

  validateChunkQueryInputs(
    'querySpatialIndex',
    chunkBounds,
    slicePosition,
    tolerance,
    ndim,
    numChunks
  );

  // Output buffer for matching chunk indices
  const matchingChunks = new Uint32Array(numChunks); // Max size

  // Call WASM (or TypeScript fallback)
  const count = wasmModule.query_chunks_for_view(
    chunkBounds,
    slicePosition,
    tolerance,
    ndim,
    numChunks,
    matchingChunks
  );

  return matchingChunks.subarray(0, count);
}
