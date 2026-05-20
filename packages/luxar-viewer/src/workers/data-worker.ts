/**
 * Data processing worker for CPU-intensive spatial queries and visibility computation.
 *
 * Loads the compiled WASM module via initWasm() (with TypeScript fallback when WASM
 * is unavailable) and exposes projection/decoding/visibility tasks to the main thread
 * through Comlink.
 *
 * Worker Responsibilities:
 * - Spatial index queries (chunk bounding box tests)
 * - nD visibility computation (hypersphere intersection testing)
 * - Array decoding (LUT, quantization, log-space) - CPU intensive!
 *
 * NOT handled here (stays on main thread):
 * - Zarr chunk fetching (needs caching store)
 * - Accumulator buffer management
 * - GPU buffer updates
 */

import { expose, transfer } from 'comlink';
import { state, requireWasm } from './data-worker/state';
import { initialize as initializeImpl } from './data-worker/initialize';
import { querySpatialIndex as querySpatialIndexImpl } from './data-worker/spatial-index/query';
import { computeNDVisibilityPoints as computeNDVisibilityPointsImpl } from './data-worker/visibility/points';
import { computeNDVisibilityLines as computeNDVisibilityLinesImpl } from './data-worker/visibility/lines';
import { computeNDVisibilityGSplats as computeNDVisibilityGSplatsImpl } from './data-worker/visibility/gsplats';
import { projectPointsTo3D as projectPointsTo3DImpl } from './data-worker/projection/points';
import { projectLinesTo3D as projectLinesTo3DImpl } from './data-worker/projection/lines';
import { projectGSplatsTo3D as projectGSplatsTo3DImpl } from './data-worker/projection/gsplats';

// Validation + color helpers live in ./data-worker/validation and ./color-utils.
// The worker imports them as bare identifiers (used in projection and
// decode paths below) and re-exports the color helpers for the
// existing unit test.
import { validateDecodeArgs } from './data-worker/validation';
import { coerceColorsToFloat32, coerceScalarsToFloat32, fillColorsWhite } from './color-utils';
export { coerceColorsToFloat32, coerceScalarsToFloat32, fillColorsWhite };



// ============================================================================
// PROJECTION FUNCTIONS (nD → 3D, CPU-intensive)
// ============================================================================

import type {
  EffectiveRadiusConfig,
  ProjectionViewState,
  PointsOutputBuffers,
} from './data-worker/types';

export type { EffectiveRadiusConfig, ProjectionViewState };




// Note: packedIndexWorker, extractCholeskySubmatrixWorker, mahalanobisDistanceWorker removed - using WASM batch functions instead

// ============================================================================
// DECODING FUNCTIONS (WASM-accelerated, CPU-intensive, offloaded from main thread)
// ============================================================================

/**
 * Decode quantized data (uint8/uint16) to float32 using WASM.
 *
 * Main thread fetches raw bytes from cache, worker dequantizes via WASM.
 * Uses Comlink.transfer for zero-copy return.
 *
 * WASM provides 2-3x speedup for large arrays (>10K elements).
 */
async function decodeQuantized(params: {
  data: Uint8Array | Uint16Array;
  bounds: [number, number];
  dtype: 'uint8' | 'uint16';
}): Promise<Float32Array> {
  const wasmModule = requireWasm(state);

  const { data, bounds, dtype } = params;
  validateDecodeArgs('decodeQuantized', data, {
    boundsPair: { name: 'bounds', bounds },
  });
  const [minVal, maxVal] = bounds;

  const result = new Float32Array(data.length);

  // Use WASM for decoding
  if (dtype === 'uint8') {
    wasmModule.decode_quantized_u8(data as Uint8Array, minVal, maxVal, result);
  } else {
    wasmModule.decode_quantized_u16(data as Uint16Array, minVal, maxVal, result);
  }

  // Transfer ownership to main thread (zero-copy)
  return transfer(result, [result.buffer]);
}

/**
 * Decode log-space quantized data (uint8/uint16) to float32 using WASM.
 *
 * Used for positive scalars with wide dynamic range (e.g., radii).
 * Decoding: expm1(normalized * maxLog)
 *
 * WASM provides 2-3x speedup for large arrays.
 */
async function decodeLogScalar(params: {
  data: Uint8Array | Uint16Array;
  maxLog: number;
  dtype: 'uint8' | 'uint16';
}): Promise<Float32Array> {
  const wasmModule = requireWasm(state);

  const { data, maxLog, dtype } = params;
  validateDecodeArgs('decodeLogScalar', data, {
    finiteScalar: { name: 'maxLog', value: maxLog },
  });

  const result = new Float32Array(data.length);

  // Use WASM for decoding
  if (dtype === 'uint8') {
    wasmModule.decode_log_scalar_u8(data as Uint8Array, maxLog, result);
  } else {
    wasmModule.decode_log_scalar_u16(data as Uint16Array, maxLog, result);
  }

  // Transfer ownership to main thread (zero-copy)
  return transfer(result, [result.buffer]);
}

/**
 * Decode LUT-encoded data (indices → values via lookup table) using WASM.
 *
 * Supports two modes:
 * - "row": One index per row → k values (e.g., one index per point → xyz)
 * - "scalar": One index per element → 1 value
 *
 * WASM provides 2-4x speedup for large arrays.
 */
async function decodeLUT(params: {
  indices: Uint8Array | Uint16Array;
  lut: number[];
  k: number;
  lutMode: 'row' | 'scalar';
  dtype?: 'uint8' | 'uint16'; // Optional - inferred from indices type if not provided
}): Promise<Float32Array> {
  const wasmModule = requireWasm(state);

  const { indices, lut, k, lutMode } = params;
  validateDecodeArgs('decodeLUT', indices, {
    positiveInt: { name: 'k', value: k },
    lut: {
      name: 'lut',
      values: lut,
      // Row mode requires the LUT to have ≥ k columns per entry; the
      // table is flat with `lut.length` total entries (= rows × k).
      minLength: lutMode === 'row' ? k : 1,
    },
  });
  if (lutMode !== 'row' && lutMode !== 'scalar') {
    throw new Error(`decodeLUT: lutMode='${lutMode}' must be 'row' or 'scalar'`);
  }

  // Scan indices for out-of-range values BEFORE handing off to WASM.
  // Rust functions (decode_lut_scalar_*, decode_lut_row_*) index
  // `lut[indices[i] as usize]` directly; an out-of-range index panics
  // or traps inside WASM. JS-side rejection turns malformed encoded
  // data into a clear error at the worker boundary.
  if (lutMode === 'row' && lut.length % k !== 0) {
    throw new Error(`decodeLUT: row-mode lut length ${lut.length} is not divisible by k=${k}`);
  }
  const entryCount = lutMode === 'row' ? Math.floor(lut.length / k) : lut.length;
  for (let i = 0; i < indices.length; i++) {
    if (indices[i] >= entryCount) {
      throw new Error(
        `decodeLUT: indices[${i}]=${indices[i]} out of range for ${entryCount} LUT ` +
          `entr${entryCount === 1 ? 'y' : 'ies'} (lutMode=${lutMode})`
      );
    }
  }

  // Infer dtype from indices type if not explicitly provided
  const dtype = params.dtype ?? (indices instanceof Uint8Array ? 'uint8' : 'uint16');
  const n = indices.length;

  // Convert LUT to Float32Array for WASM
  const lutF32 = new Float32Array(lut);

  if (lutMode === 'scalar') {
    // Scalar mode: one index per element, output size = n
    const result = new Float32Array(n);

    if (dtype === 'uint8') {
      wasmModule.decode_lut_scalar_u8(indices as Uint8Array, lutF32, result);
    } else {
      wasmModule.decode_lut_scalar_u16(indices as Uint16Array, lutF32, result);
    }

    return transfer(result, [result.buffer]);
  } else {
    // Row mode: one index per row, output size = n * k
    const result = new Float32Array(n * k);

    if (dtype === 'uint8') {
      wasmModule.decode_lut_row_u8(indices as Uint8Array, lutF32, k, result);
    } else {
      wasmModule.decode_lut_row_u16(indices as Uint16Array, lutF32, k, result);
    }

    return transfer(result, [result.buffer]);
  }
}

/**
 * Decode broadcasted data (single value replicated to all points) using WASM.
 *
 * Used for uniform attributes (e.g., all points same color).
 *
 * WASM provides speedup for large point counts.
 */
async function decodeBroadcasted(params: {
  value: Float32Array;
  numPoints: number;
  elementsPerPoint: number;
}): Promise<Float32Array> {
  const wasmModule = requireWasm(state);

  const { value, numPoints, elementsPerPoint } = params;
  if (!Number.isInteger(numPoints) || numPoints < 0) {
    throw new Error(`decodeBroadcasted: numPoints=${numPoints} must be a non-negative integer`);
  }
  if (!Number.isInteger(elementsPerPoint) || elementsPerPoint < 1) {
    throw new Error(
      `decodeBroadcasted: elementsPerPoint=${elementsPerPoint} must be a positive integer`
    );
  }
  if (value.length < elementsPerPoint) {
    throw new Error(
      `decodeBroadcasted: value too short (got ${value.length}, expected ≥ ${elementsPerPoint})`
    );
  }
  const result = new Float32Array(numPoints * elementsPerPoint);

  // Use WASM for broadcasting
  wasmModule.decode_broadcasted(value, numPoints, elementsPerPoint, result);

  return transfer(result, [result.buffer]);
}

// ============================================================================

/**
 * Expose worker API via Comlink
 */
export const workerAPI = {
  initialize: (): Promise<void> => initializeImpl(state),
  querySpatialIndex: (p: Parameters<typeof querySpatialIndexImpl>[1]) =>
    querySpatialIndexImpl(state, p),
  computeNDVisibilityPoints: (p: Parameters<typeof computeNDVisibilityPointsImpl>[1]) =>
    computeNDVisibilityPointsImpl(state, p),
  computeNDVisibilityLines: (p: Parameters<typeof computeNDVisibilityLinesImpl>[1]) =>
    computeNDVisibilityLinesImpl(state, p),
  computeNDVisibilityGSplats: (p: Parameters<typeof computeNDVisibilityGSplatsImpl>[1]) =>
    computeNDVisibilityGSplatsImpl(state, p),
  // Decoding functions (main thread fetches, worker decodes)
  decodeQuantized,
  decodeLogScalar,
  decodeLUT,
  decodeBroadcasted,
  // Projection functions (nD → 3D, CPU-intensive)
  projectPointsTo3D: (p: Parameters<typeof projectPointsTo3DImpl>[1]) =>
    projectPointsTo3DImpl(state, p),
  projectLinesTo3D: (p: Parameters<typeof projectLinesTo3DImpl>[1]) =>
    projectLinesTo3DImpl(state, p),
  projectGSplatsTo3D: (p: Parameters<typeof projectGSplatsTo3DImpl>[1]) =>
    projectGSplatsTo3DImpl(state, p),
};

expose(workerAPI);

export type DataWorkerAPI = typeof workerAPI;

// Export types for TransferableAccumulator pattern
export type { PointsOutputBuffers };
