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

// Validation + color helpers live in ./data-worker/validation and ./color-utils.
// The worker imports them as bare identifiers (used in projection and
// decode paths below) and re-exports the color helpers for the
// existing unit test.
import { validateProjectionInputs, validateDecodeArgs } from './data-worker/validation';
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



// ============================================================================
// GSPLATS PROJECTION (nD → 3D with WASM batch functions)
// ============================================================================

/**
 * Project GSplats from nD to 3D with Mahalanobis-based visibility filtering.
 *
 * This function uses WASM batch functions for high performance:
 * 1. compute_gsplats_attenuation - computes visibility and attenuation for all splats
 * 2. extract_3d_positions - extracts 3D centers (via display dims)
 * 3. extract_visible_cholesky_3d - extracts 3D Cholesky submatrices
 * 4. compact_attenuated_amplitudes - compacts amplitudes by visibility
 * 5. compact_by_mask - compacts other arrays by visibility
 *
 * WASM batch functions provide 3-5x speedup over per-splat TypeScript loops.
 */
async function projectGSplatsTo3D(params: {
  positions: Float32Array;
  choleskyFactors: Float32Array;
  amplitudes: Float32Array;
  colors: Float32Array | Uint8Array | Uint16Array | null;
  sharpness: Float32Array | null;
  /**
   * View state for projection. Same `ProjectionViewState` shape every
   * `project*To3D` worker function accepts. GSplats consumes
   * `displayDims` + `slicePosition`; `tolerance` is required for API
   * parity but isn't consulted by the WASM kernel (per-axis hidden-
   * dim attenuation is computed from cholesky factors instead).
   */
  viewState: ProjectionViewState;
  ndim: number;
  splatCount: number;
  /** Indices of hidden dimensions that are discrete (binary visibility) */
  discreteDims?: readonly number[];
  /** Per-dimension step sizes for discrete dims (keyed by dim index) */
  discreteSteps?: Record<number, number>;
  /** Indices of dimensions to skip entirely (extend_to_all — always visible) */
  extendToAllDims?: readonly number[];
  /** Truncation radius in sigmas for shifted Gaussian attenuation (default 3.0) */
  truncate?: number;
}): Promise<{
  centers3D: Float32Array;
  choleskyFactors3D: Float32Array;
  amplitudes: Float32Array;
  colors: Float32Array;
  sharpness: Float32Array;
  visibleCount: number;
}> {
  const wasmModule = requireWasm(state);

  const { positions, choleskyFactors, amplitudes, colors, viewState, ndim, splatCount } = params;
  const { displayDims, slicePosition } = viewState;

  validateProjectionInputs(
    'projectGSplatsTo3D',
    positions,
    displayDims,
    slicePosition,
    ndim,
    splatCount
  );
  // Cholesky factors: packed lower-triangular = ndim × (ndim + 1) / 2 per splat.
  const expectedCholesky = splatCount * ((ndim * (ndim + 1)) / 2);
  if (choleskyFactors.length < expectedCholesky) {
    throw new Error(
      'projectGSplatsTo3D: choleskyFactors too short ' +
        `(got ${choleskyFactors.length}, expected ≥ ${expectedCholesky})`
    );
  }
  if (amplitudes.length < splatCount) {
    throw new Error(
      `projectGSplatsTo3D: amplitudes too short (got ${amplitudes.length}, expected ≥ ${splatCount})`
    );
  }
  // RGB triplet per splat — short colors reach WASM compact_by_mask
  // unchecked and read past the buffer end.
  if (colors && colors.length < splatCount * 3) {
    throw new Error(
      `projectGSplatsTo3D: colors too short (got ${colors.length}, expected ≥ ${splatCount * 3})`
    );
  }

  // Compute hidden dimensions (all dims not in displayDims)
  const hiddenDims: number[] = [];
  for (let d = 0; d < ndim; d++) {
    if (!displayDims.includes(d)) {
      hiddenDims.push(d);
    }
  }

  // preserve requested displayDims order (matches main-thread
  // processor + Points/Lines convention). Hidden dims are still sorted
  // for the WASM Mahalanobis path which expects ascending indices.
  const orderedDisplayDims = [...displayDims];
  const sortedHiddenDims = [...hiddenDims].sort((a, b) => a - b);

  // Separate hidden dims into discrete (binary visibility) and continuous (Gaussian attenuation).
  // Discrete dimensions use a half-step threshold; continuous use WASM Mahalanobis.
  // extend_to_all dimensions are skipped entirely — splats are always visible there.
  const extendSet = new Set(params.extendToAllDims ?? []);
  const discreteSet = new Set(params.discreteDims ?? []);
  const activeHiddenDims = sortedHiddenDims.filter((d) => !extendSet.has(d));
  const continuousHiddenDims = activeHiddenDims.filter((d) => !discreteSet.has(d));
  const discreteHiddenDims = activeHiddenDims.filter((d) => discreteSet.has(d));

  // Convert to WASM-compatible arrays
  const slicePosF32 = new Float32Array(slicePosition);
  const continuousHiddenDimsU32 = new Uint32Array(continuousHiddenDims);
  const displayDimsU32 = new Uint32Array(orderedDisplayDims);

  // Minimum amplitude threshold
  const minAmplitude = 1e-6;

  // Step 0: Pre-filter discrete dimensions (TypeScript, before WASM).
  // Splats whose center is more than half a step away in any discrete dim are invisible.
  const discreteVisibility = new Uint8Array(splatCount);
  if (discreteHiddenDims.length > 0) {
    const discreteSteps = params.discreteSteps ?? {};
    for (let i = 0; i < splatCount; i++) {
      const centerOffset = i * ndim;
      let vis = true;
      for (let dIdx = 0; dIdx < discreteHiddenDims.length; dIdx++) {
        const dim = discreteHiddenDims[dIdx];
        const step = discreteSteps[dim] ?? 1.0;
        if (Math.abs(slicePosF32[dim] - positions[centerOffset + dim]) > step * 0.5) {
          vis = false;
          break;
        }
      }
      discreteVisibility[i] = vis ? 1 : 0;
    }
  } else {
    discreteVisibility.fill(1);
  }

  // Step 1: Compute attenuation and visibility for CONTINUOUS hidden dims using WASM
  const visibility = new Uint8Array(splatCount);
  const attenuation = new Float32Array(splatCount);

  const truncate = params.truncate ?? 3.0;
  wasmModule.compute_gsplats_attenuation(
    positions,
    choleskyFactors,
    amplitudes,
    slicePosF32,
    continuousHiddenDimsU32,
    ndim,
    splatCount,
    minAmplitude,
    truncate,
    visibility,
    attenuation
  );

  // Combine discrete and continuous visibility masks
  let visibleCount = 0;
  for (let i = 0; i < splatCount; i++) {
    if (discreteVisibility[i] === 0) {
      visibility[i] = 0;
      attenuation[i] = 0.0;
    }
    if (visibility[i] !== 0) visibleCount++;
  }

  // Early exit if no visible splats
  if (visibleCount === 0) {
    const emptyF32 = new Float32Array(0);
    return transfer(
      {
        centers3D: emptyF32,
        choleskyFactors3D: new Float32Array(0),
        amplitudes: new Float32Array(0),
        colors: new Float32Array(0),
        sharpness: new Float32Array(0),
        visibleCount: 0,
      },
      [emptyF32.buffer]
    );
  }

  // Step 2: Extract 3D centers using WASM
  // First extract all centers, then compact by visibility
  const allCenters3D = new Float32Array(splatCount * 3);
  wasmModule.extract_3d_positions(positions, displayDimsU32, ndim, splatCount, allCenters3D);

  // Compact centers by visibility
  const centers3D = new Float32Array(visibleCount * 3);
  wasmModule.compact_by_mask(allCenters3D, visibility, splatCount, 3, centers3D);

  // Step 3: Extract 3D Cholesky submatrices for visible splats using WASM
  const choleskyFactors3D = new Float32Array(visibleCount * 6);
  wasmModule.extract_visible_cholesky_3d(
    choleskyFactors,
    visibility,
    displayDimsU32,
    ndim,
    splatCount,
    choleskyFactors3D
  );

  // Step 4: Compact attenuated amplitudes using WASM
  const outAmplitudes = new Float32Array(visibleCount);
  wasmModule.compact_attenuated_amplitudes(
    amplitudes,
    attenuation,
    visibility,
    splatCount,
    outAmplitudes
  );

  // Step 5: Handle colors
  const outColors = new Float32Array(visibleCount * 3);
  if (colors) {
    // Compact colors by visibility (WASM expects Float32 input)
    wasmModule.compact_by_mask(coerceColorsToFloat32(colors), visibility, splatCount, 3, outColors);
  } else {
    fillColorsWhite(outColors, visibleCount);
  }

  // Build transferable list
  const transferables: ArrayBuffer[] = [
    centers3D.buffer as ArrayBuffer,
    choleskyFactors3D.buffer as ArrayBuffer,
    outAmplitudes.buffer as ArrayBuffer,
    outColors.buffer as ArrayBuffer,
  ];

  return transfer(
    {
      centers3D,
      choleskyFactors3D,
      amplitudes: outAmplitudes,
      colors: outColors,
      sharpness: new Float32Array(0), // Kept for API compatibility but unused
      visibleCount,
    },
    transferables
  );
}

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
  projectGSplatsTo3D,
};

expose(workerAPI);

export type DataWorkerAPI = typeof workerAPI;

// Export types for TransferableAccumulator pattern
export type { PointsOutputBuffers };
