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
 * - nD to 3D projection (points, lines, GSplats)
 * - Data decoding (quantized, LUT, log-space)
 * - Effective radii calculation for sliced hyperspheres
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
let effectiveRadiiBuffer: Float32Array | null = null;
let positions3DBuffer: Float32Array | null = null;
let attenuationBuffer: Float32Array | null = null;
let t1Buffer: Float32Array | null = null;
let t2Buffer: Float32Array | null = null;

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
  effectiveRadiiBuffer = new Float32Array(100000);
  positions3DBuffer = new Float32Array(300000); // 100K * 3
  attenuationBuffer = new Float32Array(100000);
  t1Buffer = new Float32Array(100000);
  t2Buffer = new Float32Array(100000);

  console.log('[DataWorker] Ready');
}

// ============================================================================
// SPATIAL INDEX QUERIES
// ============================================================================

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

// ============================================================================
// ND VISIBILITY COMPUTATION
// ============================================================================

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

// ============================================================================
// PROJECTION FUNCTIONS - nD to 3D
// ============================================================================

/**
 * Project Points from nD to 3D.
 *
 * Pipeline:
 * 1. extract_3d_positions() - Extract 3D from nD
 * 2. calculate_effective_radii() - Compute visible radii
 * 3. radii_to_visibility_mask() - Create visibility mask
 * 4. compact_by_mask() - Filter positions (call 1)
 * 5. compact_by_mask() - Filter radii (call 2)
 * 6. compact_by_mask() - Filter sharpness (call 3)
 * 7. calculate_bounds_3d() - Compute axis-aligned bounds
 */
async function projectPointsTo3D(params: {
  positions: Float32Array;
  radii: Float32Array;
  sharpness?: Float32Array;
  displayDims: Uint32Array;
  slicePosition: Float32Array;
  spatialExtendDims: Uint8Array;
  ndim: number;
  numPoints: number;
  minRadius?: number;
}): Promise<{
  positions3D: Float32Array;
  radii: Float32Array;
  sharpness?: Float32Array;
  bounds: Float32Array;
  visibleCount: number;
}> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const {
    positions,
    radii,
    sharpness,
    displayDims,
    slicePosition,
    spatialExtendDims,
    ndim,
    numPoints,
    minRadius = 0.0001,
  } = params;

  // Ensure buffer capacity
  if (!positions3DBuffer || positions3DBuffer.length < numPoints * 3) {
    positions3DBuffer = new Float32Array(Math.ceil(numPoints * 1.5 * 3));
  }
  if (!effectiveRadiiBuffer || effectiveRadiiBuffer.length < numPoints) {
    effectiveRadiiBuffer = new Float32Array(Math.ceil(numPoints * 1.5));
  }
  if (!visibilityMaskBuffer || visibilityMaskBuffer.length < numPoints) {
    visibilityMaskBuffer = new Uint8Array(Math.ceil(numPoints * 1.5));
  }

  // Step 1: Extract 3D positions from nD
  wasmModule.extract_3d_positions(
    positions,
    displayDims,
    ndim,
    numPoints,
    positions3DBuffer
  );

  // Step 2: Calculate effective radii for sliced hyperspheres
  wasmModule.calculate_effective_radii(
    positions,
    radii,
    displayDims,
    slicePosition,
    spatialExtendDims,
    ndim,
    numPoints,
    effectiveRadiiBuffer
  );

  // Step 3: Create visibility mask from effective radii
  const visibleCount = wasmModule.radii_to_visibility_mask(
    effectiveRadiiBuffer,
    minRadius,
    numPoints,
    visibilityMaskBuffer
  );

  if (visibleCount === 0) {
    return {
      positions3D: new Float32Array(0),
      radii: new Float32Array(0),
      sharpness: sharpness ? new Float32Array(0) : undefined,
      bounds: new Float32Array([0, 0, 0, 0, 0, 0]),
      visibleCount: 0,
    };
  }

  // Step 4: Compact positions by visibility mask
  const compactedPositions = new Float32Array(visibleCount * 3);
  wasmModule.compact_by_mask(
    positions3DBuffer,
    visibilityMaskBuffer,
    numPoints,
    3,
    compactedPositions
  );

  // Step 5: Compact radii by visibility mask
  const compactedRadii = new Float32Array(visibleCount);
  wasmModule.compact_by_mask(
    effectiveRadiiBuffer,
    visibilityMaskBuffer,
    numPoints,
    1,
    compactedRadii
  );

  // Step 6: Compact sharpness if provided
  let compactedSharpness: Float32Array | undefined;
  if (sharpness) {
    compactedSharpness = new Float32Array(visibleCount);
    wasmModule.compact_by_mask(
      sharpness,
      visibilityMaskBuffer,
      numPoints,
      1,
      compactedSharpness
    );
  }

  // Step 7: Calculate bounds for visible points
  const bounds = new Float32Array(6);
  wasmModule.calculate_bounds_3d(compactedPositions, visibleCount, bounds);

  return {
    positions3D: compactedPositions,
    radii: compactedRadii,
    sharpness: compactedSharpness,
    bounds,
    visibleCount,
  };
}

/**
 * Project Lines from nD to 3D.
 *
 * Pipeline (7 WASM batch functions):
 * 1. clip_segments_batch() - Clip all segments to nD slice
 * 2. interpolate_clipped_positions() - Project clipped positions to 3D
 * 3. interpolate_colors_batch() - Interpolate colors at clipped endpoints
 * 4. interpolate_scalars_batch() - Interpolate widths (call 1)
 * 5. interpolate_scalars_batch() - Interpolate sharpness (call 2)
 * 6. calculate_segment_lengths() - Compute 3D segment lengths
 * 7. mark_clipped_endpoints() - Track which endpoints were clipped
 */
async function projectLinesTo3D(params: {
  positions: Float32Array;
  segments: Uint32Array;
  colors?: Float32Array;
  widths?: Float32Array;
  sharpness?: Float32Array;
  displayDims: Uint32Array;
  slicePosition: Float32Array;
  tolerance: Float32Array;
  ndim: number;
  numSegments: number;
}): Promise<{
  startPositions: Float32Array;
  endPositions: Float32Array;
  startColors?: Float32Array;
  endColors?: Float32Array;
  startWidths?: Float32Array;
  endWidths?: Float32Array;
  startSharpness?: Float32Array;
  endSharpness?: Float32Array;
  segmentLengths: Float32Array;
  startClipped: Uint8Array;
  endClipped: Uint8Array;
  visibleCount: number;
}> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const {
    positions,
    segments,
    colors,
    widths,
    sharpness,
    displayDims,
    slicePosition,
    tolerance,
    ndim,
    numSegments,
  } = params;

  // Ensure buffer capacity
  if (!visibilityMaskBuffer || visibilityMaskBuffer.length < numSegments) {
    visibilityMaskBuffer = new Uint8Array(Math.ceil(numSegments * 1.5));
  }
  if (!t1Buffer || t1Buffer.length < numSegments) {
    t1Buffer = new Float32Array(Math.ceil(numSegments * 1.5));
  }
  if (!t2Buffer || t2Buffer.length < numSegments) {
    t2Buffer = new Float32Array(Math.ceil(numSegments * 1.5));
  }

  // Step 1: Clip all segments and get interpolation parameters
  const visibleCount = wasmModule.clip_segments_batch(
    positions,
    segments,
    slicePosition,
    tolerance,
    displayDims,
    ndim,
    numSegments,
    visibilityMaskBuffer,
    t1Buffer,
    t2Buffer
  );

  if (visibleCount === 0) {
    return {
      startPositions: new Float32Array(0),
      endPositions: new Float32Array(0),
      startColors: colors ? new Float32Array(0) : undefined,
      endColors: colors ? new Float32Array(0) : undefined,
      startWidths: widths ? new Float32Array(0) : undefined,
      endWidths: widths ? new Float32Array(0) : undefined,
      startSharpness: sharpness ? new Float32Array(0) : undefined,
      endSharpness: sharpness ? new Float32Array(0) : undefined,
      segmentLengths: new Float32Array(0),
      startClipped: new Uint8Array(0),
      endClipped: new Uint8Array(0),
      visibleCount: 0,
    };
  }

  // Step 2: Interpolate clipped positions to 3D
  const startPositions = new Float32Array(visibleCount * 3);
  const endPositions = new Float32Array(visibleCount * 3);
  wasmModule.interpolate_clipped_positions(
    positions,
    segments,
    visibilityMaskBuffer,
    t1Buffer,
    t2Buffer,
    displayDims,
    ndim,
    numSegments,
    startPositions,
    endPositions
  );

  // Step 3: Interpolate colors if provided
  let startColors: Float32Array | undefined;
  let endColors: Float32Array | undefined;
  if (colors) {
    startColors = new Float32Array(visibleCount * 3);
    endColors = new Float32Array(visibleCount * 3);
    wasmModule.interpolate_colors_batch(
      colors,
      segments,
      visibilityMaskBuffer,
      t1Buffer,
      t2Buffer,
      numSegments,
      startColors,
      endColors
    );
  }

  // Step 4: Interpolate widths if provided
  let startWidths: Float32Array | undefined;
  let endWidths: Float32Array | undefined;
  if (widths) {
    startWidths = new Float32Array(visibleCount);
    endWidths = new Float32Array(visibleCount);
    wasmModule.interpolate_scalars_batch(
      widths,
      segments,
      visibilityMaskBuffer,
      t1Buffer,
      t2Buffer,
      numSegments,
      startWidths,
      endWidths
    );
  }

  // Step 5: Interpolate sharpness if provided
  let startSharpness: Float32Array | undefined;
  let endSharpness: Float32Array | undefined;
  if (sharpness) {
    startSharpness = new Float32Array(visibleCount);
    endSharpness = new Float32Array(visibleCount);
    wasmModule.interpolate_scalars_batch(
      sharpness,
      segments,
      visibilityMaskBuffer,
      t1Buffer,
      t2Buffer,
      numSegments,
      startSharpness,
      endSharpness
    );
  }

  // Step 6: Calculate 3D segment lengths
  const segmentLengths = new Float32Array(visibleCount);
  wasmModule.calculate_segment_lengths(
    startPositions,
    endPositions,
    visibleCount,
    segmentLengths
  );

  // Step 7: Mark clipped endpoints
  const startClipped = new Uint8Array(visibleCount);
  const endClipped = new Uint8Array(visibleCount);
  wasmModule.mark_clipped_endpoints(
    visibilityMaskBuffer,
    t1Buffer,
    t2Buffer,
    numSegments,
    startClipped,
    endClipped
  );

  return {
    startPositions,
    endPositions,
    startColors,
    endColors,
    startWidths,
    endWidths,
    startSharpness,
    endSharpness,
    segmentLengths,
    startClipped,
    endClipped,
    visibleCount,
  };
}

/**
 * Project GSplats from nD to 3D.
 *
 * Pipeline (6 WASM batch functions):
 * 1. compute_gsplats_attenuation() - Compute visibility + attenuation for all splats
 * 2. extract_3d_positions() - Extract 3D centers from nD
 * 3. compact_by_mask() - Compact centers by visibility
 * 4. extract_visible_cholesky_3d() - Extract 3D Cholesky submatrices
 * 5. compact_attenuated_amplitudes() - Compact amplitudes with attenuation
 * 6. compact_by_mask() - Compact colors and sharpness
 */
async function projectGSplatsTo3D(params: {
  centers: Float32Array;
  choleskyFactors: Float32Array;
  amplitudes: Float32Array;
  colors?: Float32Array;
  sharpness: Float32Array;
  displayDims: Uint32Array;
  slicePosition: Float32Array;
  ndim: number;
  numSplats: number;
  minAmplitude?: number;
}): Promise<{
  centers3D: Float32Array;
  choleskyFactors3D: Float32Array;
  amplitudes: Float32Array;
  colors?: Float32Array;
  sharpness: Float32Array;
  visibleCount: number;
}> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const {
    centers,
    choleskyFactors,
    amplitudes,
    colors,
    sharpness,
    displayDims,
    slicePosition,
    ndim,
    numSplats,
    minAmplitude = 0.001,
  } = params;

  // Ensure buffer capacity
  if (!visibilityMaskBuffer || visibilityMaskBuffer.length < numSplats) {
    visibilityMaskBuffer = new Uint8Array(Math.ceil(numSplats * 1.5));
  }
  if (!attenuationBuffer || attenuationBuffer.length < numSplats) {
    attenuationBuffer = new Float32Array(Math.ceil(numSplats * 1.5));
  }
  if (!positions3DBuffer || positions3DBuffer.length < numSplats * 3) {
    positions3DBuffer = new Float32Array(Math.ceil(numSplats * 1.5 * 3));
  }

  // Compute hidden dimensions (all dims not in displayDims)
  const hiddenDims: number[] = [];
  const displaySet = new Set(Array.from(displayDims));
  for (let d = 0; d < ndim; d++) {
    if (!displaySet.has(d)) {
      hiddenDims.push(d);
    }
  }
  const hiddenDimsArray = new Uint32Array(hiddenDims);

  // Step 1: Compute attenuation and visibility for all splats
  const visibleCount = wasmModule.compute_gsplats_attenuation(
    centers,
    choleskyFactors,
    amplitudes,
    sharpness,
    slicePosition,
    hiddenDimsArray,
    ndim,
    numSplats,
    minAmplitude,
    visibilityMaskBuffer,
    attenuationBuffer
  );

  if (visibleCount === 0) {
    return {
      centers3D: new Float32Array(0),
      choleskyFactors3D: new Float32Array(0),
      amplitudes: new Float32Array(0),
      colors: colors ? new Float32Array(0) : undefined,
      sharpness: new Float32Array(0),
      visibleCount: 0,
    };
  }

  // Step 2: Extract 3D positions from nD centers
  wasmModule.extract_3d_positions(
    centers,
    displayDims,
    ndim,
    numSplats,
    positions3DBuffer
  );

  // Step 3: Compact centers by visibility
  const compactedCenters = new Float32Array(visibleCount * 3);
  wasmModule.compact_by_mask(
    positions3DBuffer,
    visibilityMaskBuffer,
    numSplats,
    3,
    compactedCenters
  );

  // Step 4: Extract 3D Cholesky submatrices for visible splats
  const compactedCholesky = new Float32Array(visibleCount * 6); // 3x3 packed = 6 elements
  wasmModule.extract_visible_cholesky_3d(
    choleskyFactors,
    visibilityMaskBuffer,
    displayDims,
    ndim,
    numSplats,
    compactedCholesky
  );

  // Step 5: Compact amplitudes with attenuation applied
  const compactedAmplitudes = new Float32Array(visibleCount);
  wasmModule.compact_attenuated_amplitudes(
    amplitudes,
    attenuationBuffer,
    visibilityMaskBuffer,
    numSplats,
    compactedAmplitudes
  );

  // Step 6: Compact colors if provided
  let compactedColors: Float32Array | undefined;
  if (colors) {
    compactedColors = new Float32Array(visibleCount * 3);
    wasmModule.compact_by_mask(
      colors,
      visibilityMaskBuffer,
      numSplats,
      3,
      compactedColors
    );
  }

  // Step 7: Compact sharpness
  const compactedSharpness = new Float32Array(visibleCount);
  wasmModule.compact_by_mask(
    sharpness,
    visibilityMaskBuffer,
    numSplats,
    1,
    compactedSharpness
  );

  return {
    centers3D: compactedCenters,
    choleskyFactors3D: compactedCholesky,
    amplitudes: compactedAmplitudes,
    colors: compactedColors,
    sharpness: compactedSharpness,
    visibleCount,
  };
}

// ============================================================================
// DECODE FUNCTIONS - Dequantize compressed data formats
// ============================================================================

/**
 * Decode quantized uint8 to float32.
 * Maps [0,255] -> [minVal,maxVal]
 */
async function decodeQuantizedU8(params: {
  data: Uint8Array;
  minVal: number;
  maxVal: number;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { data, minVal, maxVal } = params;
  const output = new Float32Array(data.length);

  wasmModule.decode_quantized_u8(data, minVal, maxVal, output);

  return output;
}

/**
 * Decode quantized uint16 to float32.
 * Maps [0,65535] -> [minVal,maxVal]
 */
async function decodeQuantizedU16(params: {
  data: Uint16Array;
  minVal: number;
  maxVal: number;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { data, minVal, maxVal } = params;
  const output = new Float32Array(data.length);

  wasmModule.decode_quantized_u16(data, minVal, maxVal, output);

  return output;
}

/**
 * Decode log-space quantized uint8.
 * Result = expm1(normalized * maxLog)
 */
async function decodeLogScalarU8(params: {
  data: Uint8Array;
  maxLog: number;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { data, maxLog } = params;
  const output = new Float32Array(data.length);

  wasmModule.decode_log_scalar_u8(data, maxLog, output);

  return output;
}

/**
 * Decode log-space quantized uint16.
 * Result = expm1(normalized * maxLog)
 */
async function decodeLogScalarU16(params: {
  data: Uint16Array;
  maxLog: number;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { data, maxLog } = params;
  const output = new Float32Array(data.length);

  wasmModule.decode_log_scalar_u16(data, maxLog, output);

  return output;
}

/**
 * Decode LUT indices (uint8) to scalar float values.
 */
async function decodeLUTScalarU8(params: {
  indices: Uint8Array;
  lut: Float32Array;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { indices, lut } = params;
  const output = new Float32Array(indices.length);

  wasmModule.decode_lut_scalar_u8(indices, lut, output);

  return output;
}

/**
 * Decode LUT indices (uint16) to scalar float values.
 */
async function decodeLUTScalarU16(params: {
  indices: Uint16Array;
  lut: Float32Array;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { indices, lut } = params;
  const output = new Float32Array(indices.length);

  wasmModule.decode_lut_scalar_u16(indices, lut, output);

  return output;
}

/**
 * Decode LUT indices (uint8) to k-element vectors.
 */
async function decodeLUTRowU8(params: {
  indices: Uint8Array;
  lut: Float32Array;
  k: number;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { indices, lut, k } = params;
  const output = new Float32Array(indices.length * k);

  wasmModule.decode_lut_row_u8(indices, lut, k, output);

  return output;
}

/**
 * Decode LUT indices (uint16) to k-element vectors.
 */
async function decodeLUTRowU16(params: {
  indices: Uint16Array;
  lut: Float32Array;
  k: number;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { indices, lut, k } = params;
  const output = new Float32Array(indices.length * k);

  wasmModule.decode_lut_row_u16(indices, lut, k, output);

  return output;
}

/**
 * Broadcast a value to all points.
 */
async function decodeBroadcasted(params: {
  value: Float32Array;
  numPoints: number;
  elementsPerPoint: number;
}): Promise<Float32Array> {
  if (!wasmModule) {
    throw new Error('[DataWorker] Not initialized - call initialize() first');
  }

  const { value, numPoints, elementsPerPoint } = params;
  const output = new Float32Array(numPoints * elementsPerPoint);

  wasmModule.decode_broadcasted(value, numPoints, elementsPerPoint, output);

  return output;
}

/**
 * Expose worker API via Comlink
 * NO ArrayDecoder methods - those stay on main thread!
 */
const workerAPI = {
  // Initialization
  initialize,

  // Spatial queries
  querySpatialIndex,

  // nD visibility computation
  computeNDVisibilityPoints,
  computeNDVisibilityLines,
  computeNDVisibilityGSplats,

  // Projection functions
  projectPointsTo3D,
  projectLinesTo3D,
  projectGSplatsTo3D,

  // Decode functions
  decodeQuantizedU8,
  decodeQuantizedU16,
  decodeLogScalarU8,
  decodeLogScalarU16,
  decodeLUTScalarU8,
  decodeLUTScalarU16,
  decodeLUTRowU8,
  decodeLUTRowU16,
  decodeBroadcasted,
};

expose(workerAPI);

export type DataWorkerAPI = typeof workerAPI;
