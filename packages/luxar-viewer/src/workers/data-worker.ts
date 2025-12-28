/**
 * Data processing worker for CPU-intensive spatial queries and visibility computation.
 *
 * Phase 2: Worker infrastructure with TypeScript WASM fallbacks
 * Phase 3: Upgrade to actual WASM module for 3-5x speedup
 *
 * CRITICAL: ArrayDecoder stays on main thread (needs zarr.Array objects)!
 *
 * Worker Responsibilities:
 * - Spatial index queries (chunk bounding box tests)
 * - nD visibility computation (hypersphere intersection testing)
 * - Future: Attribute interleaving, data packing
 *
 * NOT handled here (stays on main thread):
 * - Zarr chunk fetching (async IO)
 * - ArrayDecoder decoding (needs zarr context)
 * - Accumulator buffer management
 */

import { expose } from 'comlink';
import { initWasm, type WasmModule } from '../wasm';

// Worker-side persistent state
let wasmModule: WasmModule | null = null;

// Persistent buffers (avoid per-task allocations)
let visibilityMaskBuffer: Uint8Array | null = null;

/**
 * Initialize worker (called once at startup)
 *
 * Phase 2: Loads TypeScript fallback
 * Phase 3: Loads actual WASM module (REQUIRED - fails if unavailable)
 */
async function initialize(): Promise<void> {
  console.log('[DataWorker] Initializing...');

  // Load WASM module (Phase 2: TypeScript fallback, Phase 3: actual WASM)
  try {
    wasmModule = await initWasm();
    console.log('[DataWorker] WASM module loaded successfully');
  } catch (error) {
    console.error('[DataWorker] WASM initialization FAILED:', error);
    throw new Error(
      'WASM unavailable. Luxar requires WebAssembly support. ' +
        'Please use a modern browser (Chrome 57+, Firefox 52+, Safari 11+).'
    );
  }

  // Pre-allocate visibility buffer (will grow as needed)
  visibilityMaskBuffer = new Uint8Array(100000); // 100K elements max

  console.log('[DataWorker] Ready');
}

/**
 * Task 1: Query spatial index (generic for all types)
 *
 * Finds which chunks intersect the current nD view frustum.
 */
async function querySpatialIndex(params: {
  chunkBounds: Float32Array;
  slicePosition: Float32Array;
  tolerance: Float32Array;
  numChunks: number;
  ndim: number;
}): Promise<Uint32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { chunkBounds, slicePosition, tolerance, numChunks, ndim } = params;

  // Output buffer for matching chunk indices
  const matchingChunks = new Uint32Array(numChunks); // Max size

  // Call WASM (Phase 2: TypeScript fallback, Phase 3: actual WASM)
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

/**
 * Task 2: Compute nD visibility for Points
 *
 * NOTE: Receives ALREADY DECODED data (ArrayDecoder runs on main thread).
 * Computes which points are visible in the current nD slice using hypersphere intersection.
 */
async function computeNDVisibilityPoints(params: {
  positions: Float32Array;
  radii: Float32Array;
  slicePosition: Float32Array;
  tolerance: Float32Array;
  ndim: number;
  numPoints: number;
}): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { positions, radii, slicePosition, tolerance, ndim, numPoints } = params;

  // Ensure buffer capacity
  if (!visibilityMaskBuffer || visibilityMaskBuffer.length < numPoints) {
    visibilityMaskBuffer = new Uint8Array(Math.ceil(numPoints * 1.5));
  }

  // Call WASM (Phase 2: TypeScript fallback, Phase 3: actual WASM)
  const visibleCount = wasmModule.compute_nd_visibility_points(
    positions,
    radii,
    slicePosition,
    tolerance,
    ndim,
    numPoints,
    visibilityMaskBuffer
  );

  return {
    visibilityMask: visibilityMaskBuffer.subarray(0, numPoints),
    visibleCount,
  };
}

/**
 * Task 3: Compute nD visibility for Lines (check segment endpoints)
 */
async function computeNDVisibilityLines(params: {
  vertices: Float32Array;
  segments: Uint32Array;
  widths: Float32Array;
  slicePosition: Float32Array;
  tolerance: Float32Array;
  ndim: number;
  numSegments: number;
}): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { vertices, segments, widths, slicePosition, tolerance, ndim, numSegments } = params;

  // Ensure buffer capacity
  if (!visibilityMaskBuffer || visibilityMaskBuffer.length < numSegments) {
    visibilityMaskBuffer = new Uint8Array(Math.ceil(numSegments * 1.5));
  }

  // Call WASM (checks if EITHER endpoint is visible)
  const visibleCount = wasmModule.compute_nd_visibility_lines(
    vertices,
    segments,
    widths,
    slicePosition,
    tolerance,
    ndim,
    numSegments,
    visibilityMaskBuffer
  );

  return {
    visibilityMask: visibilityMaskBuffer.subarray(0, numSegments),
    visibleCount,
  };
}

/**
 * Task 4: Compute nD visibility for GSplats (check center + ellipsoid extent)
 */
async function computeNDVisibilityGSplats(params: {
  centers: Float32Array;
  choleskyFactors: Float32Array;
  slicePosition: Float32Array;
  tolerance: Float32Array;
  ndim: number;
  numSplats: number;
}): Promise<{ visibilityMask: Uint8Array; visibleCount: number }> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { centers, choleskyFactors, slicePosition, tolerance, ndim, numSplats } = params;

  // Ensure buffer capacity
  if (!visibilityMaskBuffer || visibilityMaskBuffer.length < numSplats) {
    visibilityMaskBuffer = new Uint8Array(Math.ceil(numSplats * 1.5));
  }

  // Call WASM (computes ellipsoid extent from Cholesky factors)
  const visibleCount = wasmModule.compute_nd_visibility_gsplats(
    centers,
    choleskyFactors,
    slicePosition,
    tolerance,
    ndim,
    numSplats,
    visibilityMaskBuffer
  );

  return {
    visibilityMask: visibilityMaskBuffer.subarray(0, numSplats),
    visibleCount,
  };
}

/**
 * Expose worker API via Comlink
 * NO decoding functions - ArrayDecoder stays on main thread!
 */
const workerAPI = {
  initialize,
  querySpatialIndex,
  computeNDVisibilityPoints,
  computeNDVisibilityLines,
  computeNDVisibilityGSplats,
};

expose(workerAPI);

export type DataWorkerAPI = typeof workerAPI;
