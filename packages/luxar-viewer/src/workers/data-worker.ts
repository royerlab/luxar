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
import { initWasm, type WasmModule } from '../wasm';
import { log, Modules } from '../utils/log';

// Worker-side persistent state
let wasmModule: WasmModule | null = null;

// Persistent buffers (avoid per-task allocations)
let visibilityMaskBuffer: Uint8Array | null = null;

// Validation + color helpers live in ./validation and ./color-utils.
// The worker imports them as bare identifiers (used in projection and
// decode paths below) and re-exports the color helpers for the
// existing unit test.
import {
  MAX_WASM_DIMS,
  validateNDArrays,
  validateProjectionInputs,
  validateDecodeArgs,
  validateLineSegmentReferences,
  validateChunkQueryInputs,
} from './validation';
import { coerceColorsToFloat32, coerceScalarsToFloat32, fillColorsWhite } from './color-utils';
export { coerceColorsToFloat32, coerceScalarsToFloat32, fillColorsWhite };

/**
 * Single source of truth for the "task called before initialize()"
 * error. Each task entry point inlines the `if (!wasmModule) throw`
 * check so TypeScript narrowing persists for the rest of the
 * function body — `asserts` clauses don't apply to module-scoped
 * `let` variables, so an extracted guard helper would lose narrowing.
 */
const NOT_INITIALIZED_MSG =
  '[DataWorker] Not initialized - call initialize() first';

/**
 * Initialize worker (called once at startup).
 *
 * Loads the WASM module via initWasm(); if that fails catastrophically (including
 * the TypeScript fallback), throws — the viewer requires a functioning worker.
 */
async function initialize(): Promise<void> {
  log.info(Modules.WORKER_POOL, 'DataWorker initializing...');

  // Load WASM module (falls back to TypeScript implementation if compiled WASM missing)
  try {
    wasmModule = await initWasm();
    log.info(Modules.WORKER_POOL, 'DataWorker WASM module loaded successfully');
  } catch (error) {
    log.error(Modules.WORKER_POOL, 'DataWorker WASM initialization failed', error);
    throw new Error(
      'WASM unavailable. Luxar requires WebAssembly support. ' +
        'Please use a modern browser (Chrome 57+, Firefox 52+, Safari 11+).'
    );
  }

  // Pre-allocate visibility buffer (will grow as needed)
  visibilityMaskBuffer = new Uint8Array(100000); // 100K elements max

  log.info(Modules.WORKER_POOL, 'DataWorker ready');
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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

  const { positions, radii, slicePosition, tolerance, ndim, numPoints } = params;
  validateNDArrays(
    'computeNDVisibilityPoints',
    positions,
    slicePosition,
    tolerance,
    ndim,
    numPoints,
    ndim,
    radii
  );

  // Ensure buffer capacity
  if (!visibilityMaskBuffer || visibilityMaskBuffer.length < numPoints) {
    visibilityMaskBuffer = new Uint8Array(Math.ceil(numPoints * 1.5));
  }

  // Call WASM (or TypeScript fallback)
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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

  const { vertices, segments, widths, slicePosition, tolerance, ndim, numSegments } = params;
  if (!Number.isInteger(ndim) || ndim < 1 || ndim > MAX_WASM_DIMS) {
    throw new Error(`computeNDVisibilityLines: ndim=${ndim} out of range [1, ${MAX_WASM_DIMS}]`);
  }
  if (slicePosition.length < ndim || tolerance.length < ndim) {
    throw new Error(`computeNDVisibilityLines: slicePosition/tolerance too short for ndim=${ndim}`);
  }
  // Validates segments[i] < vertex-count, plus widths length, against the
  // max referenced vertex (a stronger check than `>= numSegments`, which
  // earlier code did).
  validateLineSegmentReferences(
    'computeNDVisibilityLines',
    segments,
    numSegments,
    vertices,
    ndim,
    { widths }
  );

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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

  const { centers, choleskyFactors, slicePosition, tolerance, ndim, numSplats } = params;
  validateNDArrays(
    'computeNDVisibilityGSplats',
    centers,
    slicePosition,
    tolerance,
    ndim,
    numSplats
  );
  // Cholesky factors: packed lower-triangular has ndim*(ndim+1)/2 entries per splat.
  const expectedCholesky = numSplats * ((ndim * (ndim + 1)) / 2);
  if (choleskyFactors.length < expectedCholesky) {
    throw new Error(
      'computeNDVisibilityGSplats: choleskyFactors too short ' +
        `(got ${choleskyFactors.length}, expected ≥ ${expectedCholesky})`
    );
  }

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
// PROJECTION FUNCTIONS (nD → 3D, CPU-intensive)
// ============================================================================

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
interface PointsOutputBuffers {
  positions3D: Float32Array;
  colors?: Float32Array | Uint8Array | Uint16Array | null;
  radii?: Float32Array | null;
  sharpness?: Float32Array | null;
}

/**
 * Project Points from nD to 3D with visibility filtering
 *
 * This function:
 * 1. Extracts 3D positions from nD using displayDims
 * 2. Calculates effective radii (if config provided)
 * 3. Filters out zero-radius points
 * 4. Returns compacted arrays ready for GPU
 *
 * TransferableAccumulator pattern:
 * - If `outputBuffers` is provided, writes directly to those buffers (zero-allocation)
 * - If not, allocates new arrays for the response
 * - Returns `outputBuffers` for transfer back to main thread
 */
async function projectPointsTo3D(params: {
  positions: Float32Array;
  colors: Float32Array | Uint8Array | Uint16Array | null;
  radii: Float32Array | null;
  sharpness: Float32Array | null;
  viewState: ProjectionViewState;
  effectiveRadiusConfig: EffectiveRadiusConfig | null;
  ndim: number;
  numPoints: number;
  /** Optional pre-allocated buffers for zero-allocation (TransferableAccumulator) */
  outputBuffers?: PointsOutputBuffers;
}): Promise<{
  positions3D: Float32Array;
  colors: Float32Array | Uint8Array | Uint16Array | null;
  radii: Float32Array | null;
  sharpness: Float32Array | null;
  visibleCount: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  /** Returned for TransferableAccumulator - same as input if provided */
  outputBuffers?: PointsOutputBuffers;
}> {
  const {
    positions,
    colors,
    radii,
    sharpness,
    viewState,
    effectiveRadiusConfig,
    ndim,
    numPoints,
    outputBuffers,
  } = params;
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);
  const { displayDims, slicePosition } = viewState;

  validateProjectionInputs(
    'projectPointsTo3D',
    positions,
    displayDims,
    slicePosition,
    ndim,
    numPoints
  );
  if (radii && radii.length < numPoints) {
    throw new Error(
      `projectPointsTo3D: radii too short (got ${radii.length}, expected ≥ ${numPoints})`
    );
  }
  if (sharpness && sharpness.length < numPoints) {
    throw new Error(
      `projectPointsTo3D: sharpness too short (got ${sharpness.length}, expected ≥ ${numPoints})`
    );
  }
  // RGB triplet per point — short colors silently produce NaN/0 fill in
  // the TS-side compaction and corrupt the rendered point colors.
  if (colors && colors.length < numPoints * 3) {
    throw new Error(
      `projectPointsTo3D: colors too short (got ${colors.length}, expected ≥ ${numPoints * 3})`
    );
  }

  // Validate effective-radius config inputs before they reach WASM.
  // The Uint8 array built from spatialExtendDims is fed to
  // calculate_effective_radii, which expects ndim entries; a short
  // array silently treats trailing dims as non-extended. maxRadius is
  // not used here directly but flows into per-vertex math elsewhere
  // and must be finite to avoid NaN propagation.
  if (effectiveRadiusConfig) {
    if (effectiveRadiusConfig.spatialExtendDims.length < ndim) {
      throw new Error(
        `projectPointsTo3D: effectiveRadiusConfig.spatialExtendDims too short (got ${effectiveRadiusConfig.spatialExtendDims.length}, expected ≥ ${ndim})`
      );
    }
    if (!Number.isFinite(effectiveRadiusConfig.maxRadius)) {
      throw new Error(
        `projectPointsTo3D: effectiveRadiusConfig.maxRadius=${effectiveRadiusConfig.maxRadius} must be a finite number`
      );
    }
    // Tolerance is read by index per non-displayed dim in the
    // extend_to_all detection path further down; a malformed worker
    // payload that omits it or provides a short array would either
    // throw a generic `Cannot read properties of undefined` or
    // silently treat missing entries as non-extend, changing
    // effective-radius semantics. Production callers pass the right
    // shape; this is a worker-boundary validation belt for direct
    // callers and malformed payloads.
    if (
      !viewState ||
      !viewState.tolerance ||
      viewState.tolerance.length < ndim
    ) {
      throw new Error(
        `projectPointsTo3D: viewState.tolerance too short for effective radius (got ${viewState?.tolerance?.length ?? 0}, expected ≥ ${ndim})`
      );
    }
  }

  // Determine if using pre-allocated buffers (TransferableAccumulator pattern)
  const usePreallocated = !!outputBuffers;

  // Step 1: Extract 3D positions from nD using WASM
  // Use pre-allocated buffer if available, otherwise allocate
  const positions3D =
    usePreallocated && outputBuffers.positions3D.length >= numPoints * 3
      ? outputBuffers.positions3D
      : new Float32Array(numPoints * 3);

  // Use WASM for 3D extraction (faster for large arrays)
  const displayDimsU32 = new Uint32Array(displayDims);
  wasmModule!.extract_3d_positions(positions, displayDimsU32, ndim, numPoints, positions3D);

  // Step 2: Calculate effective radii (if config provided and radii exist) using WASM
  let effectiveRadii: Float32Array | null = null;
  let usedEffectiveRadius = false;

  if (radii && effectiveRadiusConfig) {
    effectiveRadii = new Float32Array(numPoints);
    // Convert to WASM-compatible arrays
    const slicePositionF32 = new Float32Array(slicePosition);
    const spatialExtendDimsU8 = new Uint8Array(
      effectiveRadiusConfig.spatialExtendDims.map((b) => (b ? 1 : 0))
    );

    // For effective radius calculation, extend_to_all dims should be treated
    // as "display" dims (completely skipped — no discrete check, no distance).
    // Build an augmented display dims array that includes extend_to_all dims.
    const extendToAllDims: number[] = [];
    for (let d = 0; d < ndim; d++) {
      if (!displayDims.includes(d) && viewState.tolerance[d] >= 1e9) {
        extendToAllDims.push(d);
      }
    }
    const effectiveRadiiDisplayDims =
      extendToAllDims.length > 0
        ? new Uint32Array([...displayDims, ...extendToAllDims])
        : displayDimsU32;

    // Use WASM for effective radius calculation
    wasmModule!.calculate_effective_radii(
      positions,
      radii,
      effectiveRadiiDisplayDims,
      slicePositionF32,
      spatialExtendDimsU8,
      ndim,
      numPoints,
      effectiveRadii
    );
    usedEffectiveRadius = true;
  } else if (radii) {
    // No effective radius config - just copy radii
    effectiveRadii = new Float32Array(radii);
  }

  // Step 3: Filter out zero-radius points using WASM
  let visibleCount = numPoints;
  let filteredPositions3D = positions3D;
  let filteredColors: Float32Array | Uint8Array | Uint16Array | null = colors;
  let filteredRadii = effectiveRadii;
  let filteredSharpness = sharpness;

  if (usedEffectiveRadius && effectiveRadii) {
    const threshold = 0.0001;

    // Create visibility mask using WASM
    const visibilityMask = new Uint8Array(numPoints);
    visibleCount = wasmModule!.radii_to_visibility_mask(
      effectiveRadii,
      threshold,
      numPoints,
      visibilityMask
    );

    if (visibleCount < numPoints && visibleCount > 0) {
      // Compact positions using WASM
      filteredPositions3D = new Float32Array(visibleCount * 3);
      wasmModule!.compact_by_mask(positions3D, visibilityMask, numPoints, 3, filteredPositions3D);

      // Compact radii using WASM
      filteredRadii = new Float32Array(visibleCount);
      wasmModule!.compact_by_mask(effectiveRadii, visibilityMask, numPoints, 1, filteredRadii);

      // Compact colors (keep TypeScript for multi-type support)
      if (colors) {
        // Build valid indices for color compaction (colors can be Uint8/Uint16/Float32)
        const validIndices: number[] = [];
        for (let i = 0; i < numPoints; i++) {
          if (visibilityMask[i] !== 0) validIndices.push(i);
        }

        if (colors instanceof Uint8Array) {
          filteredColors = new Uint8Array(visibleCount * 3);
        } else if (colors instanceof Uint16Array) {
          filteredColors = new Uint16Array(visibleCount * 3);
        } else {
          filteredColors = new Float32Array(visibleCount * 3);
        }
        for (let i = 0; i < visibleCount; i++) {
          const srcIdx = validIndices[i];
          filteredColors[i * 3] = colors[srcIdx * 3];
          filteredColors[i * 3 + 1] = colors[srcIdx * 3 + 1];
          filteredColors[i * 3 + 2] = colors[srcIdx * 3 + 2];
        }
      }

      // Compact sharpness using WASM
      if (sharpness) {
        filteredSharpness = new Float32Array(visibleCount);
        wasmModule!.compact_by_mask(sharpness, visibilityMask, numPoints, 1, filteredSharpness);
      }
    } else if (visibleCount === 0) {
      // All filtered out - preserve original color type for empty arrays
      filteredPositions3D = new Float32Array(0);
      filteredRadii = new Float32Array(0);
      if (colors) {
        if (colors instanceof Uint8Array) {
          filteredColors = new Uint8Array(0);
        } else if (colors instanceof Uint16Array) {
          filteredColors = new Uint16Array(0);
        } else {
          filteredColors = new Float32Array(0);
        }
      } else {
        filteredColors = null;
      }
      filteredSharpness = sharpness ? new Float32Array(0) : null;
    }
  }

  // Step 4: Calculate bounds using WASM
  const boundsOutput = new Float32Array(6);
  wasmModule!.calculate_bounds_3d(filteredPositions3D, visibleCount, boundsOutput);

  const bounds = {
    min: [boundsOutput[0], boundsOutput[1], boundsOutput[2]] as [number, number, number],
    max: [boundsOutput[3], boundsOutput[4], boundsOutput[5]] as [number, number, number],
  };

  // Build transferable list (cast to ArrayBuffer since we only use regular ArrayBuffers)
  const transferables: ArrayBuffer[] = [filteredPositions3D.buffer as ArrayBuffer];
  if (filteredColors) transferables.push(filteredColors.buffer as ArrayBuffer);
  if (filteredRadii) transferables.push(filteredRadii.buffer as ArrayBuffer);
  if (filteredSharpness) transferables.push(filteredSharpness.buffer as ArrayBuffer);

  // Build result with optional outputBuffers for TransferableAccumulator
  const result: {
    positions3D: Float32Array;
    colors: Float32Array | Uint8Array | Uint16Array | null;
    radii: Float32Array | null;
    sharpness: Float32Array | null;
    visibleCount: number;
    bounds: { min: [number, number, number]; max: [number, number, number] };
    outputBuffers?: PointsOutputBuffers;
  } = {
    positions3D: filteredPositions3D,
    colors: filteredColors,
    radii: filteredRadii,
    sharpness: filteredSharpness,
    visibleCount,
    bounds,
  };

  // If using TransferableAccumulator pattern, include outputBuffers for adoption
  if (usePreallocated && outputBuffers) {
    // Update outputBuffers with the actual buffers used (may be same or different due to filtering)
    result.outputBuffers = {
      positions3D: filteredPositions3D,
      colors: filteredColors,
      radii: filteredRadii,
      sharpness: filteredSharpness,
    };
  }

  return transfer(result, transferables);
}

// Note: calculateEffectiveRadiiWorker removed - using WASM calculate_effective_radii instead

/**
 * Project Lines from nD to 3D with segment clipping.
 *
 * This function mirrors the main-thread `data/lines/projection.ts::projectLinesTo3D`
 * but runs in a worker:
 * 1. Clips all segments to the nD slice using WASM batch functions
 * 2. Projects clipped endpoints to 3D
 * 3. Interpolates per-vertex attributes (colors, widths, sharpness)
 * 4. Calculates segment lengths and tracks clipping
 *
 * WASM batch functions provide 3-5x speedup over per-segment TypeScript loops.
 */
async function projectLinesTo3D(params: {
  positions: Float32Array;
  segments: Uint32Array;
  widths: Float32Array;
  colors: Float32Array | Uint8Array | Uint16Array | null;
  sharpness: Float32Array | null;
  /**
   * Optional per-vertex scalar attribute for colormap-mode Lines.
   * Accepted dtypes: `Float32Array` (passes through zero-copy),
   * `Float16Array` (element-wise expand), `Uint8Array` (normalized by
   * `1/255` to align with the colormap shader's `[0, 1]` contract).
   * When `null` the worker emits empty `startScalars`/`endScalars`
   * arrays and the loader leaves the geometry's scalar attribute
   * unallocated.
   */
  scalars: Float32Array | Float16Array | Uint8Array | null;
  /**
   * View state for projection. Same `ProjectionViewState` shape every
   * `project*To3D` worker function accepts — keeps the worker API
   * uniform across node types. `tolerance` is required for parity
   * even though Lines doesn't consult it directly (chunk bounds
   * handle the visibility test); the field is validated for length
   * to catch malformed payloads at the boundary.
   */
  viewState: ProjectionViewState;
  ndim: number;
  segmentCount: number;
}): Promise<{
  startPositions: Float32Array;
  endPositions: Float32Array;
  startColors: Float32Array;
  endColors: Float32Array;
  startWidths: Float32Array;
  endWidths: Float32Array;
  startSharpness: Float32Array;
  endSharpness: Float32Array;
  /** Per-segment start scalar (empty Float32Array when input scalars=null). */
  startScalars: Float32Array;
  /** Per-segment end scalar (empty Float32Array when input scalars=null). */
  endScalars: Float32Array;
  segmentLengths: Float32Array;
  startClipped: Uint8Array;
  endClipped: Uint8Array;
  visibleSegmentCount: number;
}> {
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

  const {
    positions,
    segments,
    widths,
    colors,
    sharpness,
    scalars,
    viewState,
    ndim,
    segmentCount,
  } = params;
  const { displayDims, slicePosition, tolerance } = viewState;

  // The shared projection validator handles displayDims and the basic
  // ndim/positions sanity check; we then check segment-vertex bounds
  // explicitly because positions length depends on max referenced vertex
  // (not numItems = 1), and finally the per-vertex attribute lengths.
  validateProjectionInputs(
    'projectLinesTo3D',
    positions,
    displayDims,
    slicePosition,
    ndim,
    1,
    ndim
  );
  if (tolerance.length < ndim) {
    throw new Error(
      `projectLinesTo3D: tolerance too short (got ${tolerance.length}, expected ≥ ${ndim})`
    );
  }
  validateLineSegmentReferences(
    'projectLinesTo3D',
    segments,
    segmentCount,
    positions,
    ndim,
    {
      widths,
      colors: colors ?? undefined,
      sharpness: sharpness ?? undefined,
    }
  );

  // Convert input arrays to WASM-compatible formats
  const slicePosF32 = new Float32Array(slicePosition);
  const toleranceF32 = new Float32Array(tolerance);
  const displayDimsU32 = new Uint32Array(displayDims);

  // Step 1: Batch clip all segments using WASM
  const visibility = new Uint8Array(segmentCount);
  const t1Params = new Float32Array(segmentCount);
  const t2Params = new Float32Array(segmentCount);

  const visibleCount = wasmModule.clip_segments_batch(
    positions,
    segments,
    slicePosF32,
    toleranceF32,
    displayDimsU32,
    ndim,
    segmentCount,
    visibility,
    t1Params,
    t2Params
  );

  // Early exit if no visible segments
  if (visibleCount === 0) {
    const emptyPositions = new Float32Array(0);
    const emptyColors = new Float32Array(0);
    const emptyScalars = new Float32Array(0);
    const emptyFlags = new Uint8Array(0);

    return transfer(
      {
        startPositions: emptyPositions,
        endPositions: new Float32Array(0),
        startColors: emptyColors,
        endColors: new Float32Array(0),
        startWidths: emptyScalars,
        endWidths: new Float32Array(0),
        startSharpness: new Float32Array(0),
        endSharpness: new Float32Array(0),
        startScalars: new Float32Array(0),
        endScalars: new Float32Array(0),
        segmentLengths: new Float32Array(0),
        startClipped: emptyFlags,
        endClipped: new Uint8Array(0),
        visibleSegmentCount: 0,
      },
      [emptyPositions.buffer, emptyColors.buffer, emptyScalars.buffer, emptyFlags.buffer]
    );
  }

  // Step 2: Interpolate clipped positions to 3D using WASM
  const startPositions = new Float32Array(visibleCount * 3);
  const endPositions = new Float32Array(visibleCount * 3);

  wasmModule.interpolate_clipped_positions(
    positions,
    segments,
    visibility,
    t1Params,
    t2Params,
    displayDimsU32,
    ndim,
    segmentCount,
    startPositions,
    endPositions
  );

  // Step 3: Interpolate colors using WASM
  // Convert colors to Float32Array if needed (WASM expects Float32Array)
  const startColors = new Float32Array(visibleCount * 3);
  const endColors = new Float32Array(visibleCount * 3);

  if (colors) {
    wasmModule.interpolate_colors_batch(
      coerceColorsToFloat32(colors),
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startColors,
      endColors
    );
  } else {
    fillColorsWhite(startColors, visibleCount);
    fillColorsWhite(endColors, visibleCount);
  }

  // Step 4: Interpolate widths using WASM
  const startWidths = new Float32Array(visibleCount);
  const endWidths = new Float32Array(visibleCount);

  wasmModule.interpolate_scalars_batch(
    widths,
    segments,
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startWidths,
    endWidths
  );

  // Step 5: Interpolate sharpness using WASM
  const startSharpness = new Float32Array(visibleCount);
  const endSharpness = new Float32Array(visibleCount);

  if (sharpness) {
    wasmModule.interpolate_scalars_batch(
      sharpness,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startSharpness,
      endSharpness
    );
  } else {
    // Default sharpness is 1.0
    startSharpness.fill(1.0);
    endSharpness.fill(1.0);
  }

  // Step 6: Interpolate per-vertex scalars using WASM (colormap mode).
  // Empty arrays are returned when `scalars` is null so the loader can
  // skip allocating the geometry's scalar attribute. We coerce
  // Float16/Uint8 to Float32 first since `interpolate_scalars_batch`
  // expects Float32 inputs (same path widths/sharpness take).
  let startScalars: Float32Array;
  let endScalars: Float32Array;
  if (scalars) {
    const scalarsF32 = coerceScalarsToFloat32(scalars);
    startScalars = new Float32Array(visibleCount);
    endScalars = new Float32Array(visibleCount);
    wasmModule.interpolate_scalars_batch(
      scalarsF32,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startScalars,
      endScalars
    );
  } else {
    startScalars = new Float32Array(0);
    endScalars = new Float32Array(0);
  }

  // Step 7: Calculate segment lengths using WASM
  const segmentLengths = new Float32Array(visibleCount);
  wasmModule.calculate_segment_lengths(startPositions, endPositions, visibleCount, segmentLengths);

  // Step 8: Mark clipped endpoints using WASM
  const startClipped = new Uint8Array(visibleCount);
  const endClipped = new Uint8Array(visibleCount);

  wasmModule.mark_clipped_endpoints(
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startClipped,
    endClipped
  );

  // Build transferable list
  const transferables: ArrayBuffer[] = [
    startPositions.buffer as ArrayBuffer,
    endPositions.buffer as ArrayBuffer,
    startColors.buffer as ArrayBuffer,
    endColors.buffer as ArrayBuffer,
    startWidths.buffer as ArrayBuffer,
    endWidths.buffer as ArrayBuffer,
    startSharpness.buffer as ArrayBuffer,
    endSharpness.buffer as ArrayBuffer,
    startScalars.buffer as ArrayBuffer,
    endScalars.buffer as ArrayBuffer,
    segmentLengths.buffer as ArrayBuffer,
    startClipped.buffer as ArrayBuffer,
    endClipped.buffer as ArrayBuffer,
  ];

  return transfer(
    {
      startPositions,
      endPositions,
      startColors,
      endColors,
      startWidths,
      endWidths,
      startSharpness,
      endSharpness,
      startScalars,
      endScalars,
      segmentLengths,
      startClipped,
      endClipped,
      visibleSegmentCount: visibleCount,
    },
    transferables
  );
}

// Note: clipSegmentToSliceWorker and lerpWorker removed - using WASM batch functions instead

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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

  const {
    positions,
    choleskyFactors,
    amplitudes,
    colors,
    viewState,
    ndim,
    splatCount,
  } = params;
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
    wasmModule.compact_by_mask(
      coerceColorsToFloat32(colors),
      visibility,
      splatCount,
      3,
      outColors
    );
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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

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
    throw new Error(
      `decodeLUT: row-mode lut length ${lut.length} is not divisible by k=${k}`
    );
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
  if (!wasmModule) throw new Error(NOT_INITIALIZED_MSG);

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
  initialize,
  querySpatialIndex,
  computeNDVisibilityPoints,
  computeNDVisibilityLines,
  computeNDVisibilityGSplats,
  // Decoding functions (main thread fetches, worker decodes)
  decodeQuantized,
  decodeLogScalar,
  decodeLUT,
  decodeBroadcasted,
  // Projection functions (nD → 3D, CPU-intensive)
  projectPointsTo3D,
  projectLinesTo3D,
  projectGSplatsTo3D,
};

expose(workerAPI);

export type DataWorkerAPI = typeof workerAPI;

// Export types for TransferableAccumulator pattern
export type { PointsOutputBuffers };
