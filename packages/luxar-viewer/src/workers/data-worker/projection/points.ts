/**
 * Project Points from nD to 3D with visibility filtering.
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

import { transfer } from 'comlink';
import { requireWasm, type WasmCtx } from '../state';
import { validateProjectionInputs } from '../validation';
import type { EffectiveRadiusConfig, ProjectionViewState, PointsOutputBuffers } from '../types';

export async function projectPointsTo3D(
  ctx: WasmCtx,
  params: {
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
  }
): Promise<{
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
  const wasmModule = requireWasm(ctx);
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
    if (!viewState || !viewState.tolerance || viewState.tolerance.length < ndim) {
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
  wasmModule.extract_3d_positions(positions, displayDimsU32, ndim, numPoints, positions3D);

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
    wasmModule.calculate_effective_radii(
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
    visibleCount = wasmModule.radii_to_visibility_mask(
      effectiveRadii,
      threshold,
      numPoints,
      visibilityMask
    );

    if (visibleCount < numPoints && visibleCount > 0) {
      // Compact positions using WASM
      filteredPositions3D = new Float32Array(visibleCount * 3);
      wasmModule.compact_by_mask(positions3D, visibilityMask, numPoints, 3, filteredPositions3D);

      // Compact radii using WASM
      filteredRadii = new Float32Array(visibleCount);
      wasmModule.compact_by_mask(effectiveRadii, visibilityMask, numPoints, 1, filteredRadii);

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
        wasmModule.compact_by_mask(sharpness, visibilityMask, numPoints, 1, filteredSharpness);
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
  wasmModule.calculate_bounds_3d(filteredPositions3D, visibleCount, boundsOutput);

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
