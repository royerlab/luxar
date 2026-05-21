/**
 * Project GSplats from nD to 3D with Mahalanobis-based visibility filtering.
 *
 * Uses WASM batch functions for high performance:
 * 1. compute_gsplats_attenuation - computes visibility and attenuation
 * 2. extract_3d_positions - extracts 3D centers (via displayDims)
 * 3. extract_visible_cholesky_3d - extracts 3D Cholesky submatrices
 * 4. compact_attenuated_amplitudes - compacts amplitudes by visibility
 * 5. compact_by_mask - compacts other arrays by visibility
 *
 * WASM batch functions provide 3-5x speedup over per-splat TypeScript loops.
 */

import { transfer } from 'comlink';
import { requireWasm, type WasmCtx } from '../state';
import { validateProjectionInputs } from '../validation';
import { coerceColorsToFloat32, fillColorsWhite } from '../../color-utils';
import type { ProjectionViewState } from '../types';

export async function projectGSplatsTo3D(
  ctx: WasmCtx,
  params: {
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
  }
): Promise<{
  centers3D: Float32Array;
  choleskyFactors3D: Float32Array;
  amplitudes: Float32Array;
  colors: Float32Array;
  sharpness: Float32Array;
  visibleCount: number;
}> {
  const wasmModule = requireWasm(ctx);

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
