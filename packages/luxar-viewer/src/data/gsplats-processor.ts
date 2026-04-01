/**
 * GSplats Processor for nD → 3D Conversion
 *
 * This module handles the conversion of nD gsplats data to 3D for rendering.
 * The key operations are:
 * 1. Extract 3D center from nD center using display dimensions
 * 2. Extract 3D Cholesky submatrix from nD Cholesky
 * 3. Attenuate amplitude based on distance to hyperplane in hidden dimensions
 *
 * @module data/gsplats-processor
 */

import type { LoadedGSplatsData, ProcessedGSplatsData, GSplatsViewState } from '../types/gsplats';

/**
 * Compute the packed index for a Cholesky element L[row, col].
 * Packed lower-triangular: [L00, L10, L11, L20, L21, L22, ...]
 * Formula: row * (row + 1) / 2 + col (for col <= row)
 */
function packedIndex(row: number, col: number): number {
  return (row * (row + 1)) / 2 + col;
}

/** Maximum supported dimensions (must match WASM MAX_SUPPORTED_DIMS). */
const MAX_SUPPORTED_DIMS = 16;

/** Epsilon for degenerate diagonal detection during Cholesky factorization. */
const CHOLESKY_EPSILON = 1e-10;

/**
 * Pre-allocated workspace buffers for computeMarginalCholesky.
 * Avoids per-call allocation when called in tight loops (100K+ splats).
 * Module-level singletons, safe because JS is single-threaded.
 */
const _sigmaWorkspace = new Float32Array(MAX_SUPPORTED_DIMS * MAX_SUPPORTED_DIMS);
const _lSubWorkspace = new Float32Array((MAX_SUPPORTED_DIMS * (MAX_SUPPORTED_DIMS + 1)) / 2);

/**
 * Compute the correct marginal Cholesky factor for a subset of dimensions.
 *
 * For Σ = L·Lᵀ, the marginal covariance for dimensions S is:
 *   Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]
 *
 * This function reconstructs Σ_S and then Cholesky-factorizes it.
 * Simply extracting L elements is INCORRECT when there are cross-dimension correlations.
 *
 * Uses module-level workspace buffers to avoid per-call allocation.
 *
 * @param packed - Full packed Cholesky factor array
 * @param offset - Start offset in packed for this splat
 * @param keepDims - Dimension indices to keep (sorted ascending)
 * @param output - Output array to write to
 * @param outputOffset - Start offset in output array
 */
function computeMarginalCholesky(
  packed: Float32Array,
  offset: number,
  keepDims: number[],
  output: Float32Array,
  outputOffset: number
): void {
  const subNdim = keepDims.length;

  // Step 1: Reconstruct marginal covariance Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]
  // Uses pre-allocated workspace (zeroing only the region we use)
  for (let i = 0; i < subNdim; i++) {
    const si = keepDims[i];
    for (let j = 0; j <= i; j++) {
      const sj = keepDims[j];
      const kMax = Math.min(si, sj);
      let sum = 0;
      for (let k = 0; k <= kMax; k++) {
        const lSiK = packed[offset + packedIndex(si, k)];
        const lSjK = packed[offset + packedIndex(sj, k)];
        sum += lSiK * lSjK;
      }
      _sigmaWorkspace[i * MAX_SUPPORTED_DIMS + j] = sum;
      _sigmaWorkspace[j * MAX_SUPPORTED_DIMS + i] = sum;
    }
  }

  // Step 2: Cholesky-Crout factorization of Σ_S → L_S
  const subPackedSize = (subNdim * (subNdim + 1)) / 2;

  for (let i = 0; i < subNdim; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = _sigmaWorkspace[i * MAX_SUPPORTED_DIMS + j];
      for (let k = 0; k < j; k++) {
        sum -= _lSubWorkspace[packedIndex(i, k)] * _lSubWorkspace[packedIndex(j, k)];
      }
      if (i === j) {
        _lSubWorkspace[packedIndex(i, i)] =
          sum > CHOLESKY_EPSILON ? Math.sqrt(sum) : Math.sqrt(CHOLESKY_EPSILON);
      } else {
        const diag = _lSubWorkspace[packedIndex(j, j)];
        _lSubWorkspace[packedIndex(i, j)] = diag > CHOLESKY_EPSILON ? sum / diag : 0;
      }
    }
  }

  // Copy to output from workspace
  for (let k = 0; k < subPackedSize; k++) {
    output[outputOffset + k] = _lSubWorkspace[k];
  }
}

/**
 * Compute Mahalanobis distance reusing a pre-allocated y buffer.
 * Avoids per-call allocation, matching the pattern in gsplats_processing.ts.
 */
function mahalanobisDistanceReuse(
  diff: number[],
  packedL: Float32Array,
  offset: number,
  ndim: number,
  yBuffer: number[]
): number {
  for (let i = 0; i < ndim; i++) {
    let val = diff[i];
    for (let j = 0; j < i; j++) {
      val -= packedL[offset + packedIndex(i, j)] * yBuffer[j];
    }
    const diag = packedL[offset + packedIndex(i, i)];
    yBuffer[i] = diag > 1e-10 ? val / diag : 0;
  }

  let sumSq = 0;
  for (let i = 0; i < ndim; i++) {
    sumSq += yBuffer[i] * yBuffer[i];
  }
  return Math.sqrt(sumSq);
}

/**
 * Process loaded nD gsplats data to 3D for rendering.
 *
 * This function:
 * 1. Filters splats based on visibility (amplitude attenuation > threshold)
 * 2. Extracts 3D centers from nD centers
 * 3. Extracts 3D Cholesky submatrices from nD Cholesky
 * 4. Computes attenuated amplitudes based on hidden dimension distance
 *
 * @param loaded - Raw gsplats data from zarr
 * @param viewState - Current view state with display dims and slice position
 * @param truncate - Truncation radius in sigmas for shifted Gaussian (default 3.0)
 * @returns Processed 3D gsplats data ready for GPU
 */
export function processGSplatsTo3D(
  loaded: LoadedGSplatsData,
  viewState: GSplatsViewState,
  truncate: number = 3.0
): ProcessedGSplatsData {
  const { displayDims, slicePosition } = viewState;
  const ndim = loaded.ndim;
  const splatCount = loaded.splatCount;

  // Compute hidden dimensions (all dims not in displayDims)
  const hiddenDims = [];
  for (let d = 0; d < ndim; d++) {
    if (!displayDims.includes(d)) {
      hiddenDims.push(d);
    }
  }

  // Sort dimensions for consistent submatrix extraction
  const sortedDisplayDims = [...displayDims].sort((a, b) => a - b);
  const sortedHiddenDims = [...hiddenDims].sort((a, b) => a - b);

  // Separate hidden dims into discrete (binary visibility) and continuous (Gaussian attenuation).
  // Discrete dimensions (e.g., time, channel) use step-based in/out: if the splat's center
  // is within half a step of slicePosition, it's fully visible; otherwise invisible.
  // Continuous dimensions use the existing marginal Cholesky + Mahalanobis attenuation.
  // Dimensions with extend_to_all (tolerance >= 1e9) are skipped entirely — splats are
  // always visible in those dimensions regardless of position.
  // When no dimension metadata is available, all hidden dims default to continuous (backward compat).
  const discreteHiddenDims: number[] = [];
  const continuousHiddenDims: number[] = [];
  for (const dim of sortedHiddenDims) {
    if (viewState.tolerance[dim] >= 1e9) continue; // extend_to_all: always visible
    if (viewState.dimensions?.[dim]?.discrete) {
      discreteHiddenDims.push(dim);
    } else {
      continuousHiddenDims.push(dim);
    }
  }
  const numContinuousHidden = continuousHiddenDims.length;

  // Compute packed sizes
  const fullPackedSize = (ndim * (ndim + 1)) / 2;
  const display3DPackedSize = 6; // 3D Cholesky has 6 elements
  const continuousHiddenPackedSize = (numContinuousHidden * (numContinuousHidden + 1)) / 2;

  // Minimum amplitude threshold (splats with lower amplitude are invisible)
  const minAmplitude = 1e-6;

  // Pre-allocate reusable temporary buffers ONCE (avoid per-splat GC pressure)
  // Buffers sized for continuous hidden dims only (discrete dims don't need Cholesky)
  const hiddenCholesky =
    numContinuousHidden > 0 ? new Float32Array(continuousHiddenPackedSize) : null;
  const diff = numContinuousHidden > 0 ? new Array<number>(numContinuousHidden) : null;
  const yBuffer = numContinuousHidden > 0 ? new Array<number>(numContinuousHidden) : null;

  // Shifted Gaussian constants (hoisted outside loop — these depend only on truncation radius)
  const shiftedGaussianC = Math.exp(-0.5 * truncate * truncate);
  const shiftedGaussianInvOneMinusC = 1.0 / (1.0 - shiftedGaussianC);

  // Single pass: compute attenuation for ALL splats, cache the values,
  // and collect visible indices. This avoids recomputing the expensive
  // marginal Cholesky + Mahalanobis distance in a second pass.
  const attenuations = new Float32Array(splatCount);
  let visibleCount = 0;
  const visibleIndices: number[] = [];

  for (let i = 0; i < splatCount; i++) {
    const centerOffset = i * ndim;

    // Step 1: Binary visibility check for discrete hidden dimensions.
    // If the splat's center is more than half a step away in any discrete dim,
    // it belongs to a different slice and should be invisible.
    let discreteVisible = true;
    for (let dIdx = 0; dIdx < discreteHiddenDims.length; dIdx++) {
      const dim = discreteHiddenDims[dIdx];
      const step = viewState.dimensions?.[dim]?.step ?? 1.0;
      const absDiff = Math.abs(slicePosition[dim] - loaded.positions[centerOffset + dim]);
      if (absDiff > step * 0.5) {
        discreteVisible = false;
        break;
      }
    }

    if (!discreteVisible) {
      attenuations[i] = 0.0;
      continue;
    }

    // Step 2: Gaussian attenuation for continuous hidden dimensions only.
    let attenuation = 1.0;

    if (numContinuousHidden > 0) {
      // Fill diff vector for continuous hidden dims only
      for (let hIdx = 0; hIdx < numContinuousHidden; hIdx++) {
        diff![hIdx] =
          slicePosition[continuousHiddenDims[hIdx]] -
          loaded.positions[centerOffset + continuousHiddenDims[hIdx]];
      }

      // Compute marginal Cholesky for continuous hidden dims
      computeMarginalCholesky(
        loaded.choleskyFactors,
        i * fullPackedSize,
        continuousHiddenDims,
        hiddenCholesky!,
        0
      );

      // Compute Mahalanobis distance using reusable y buffer
      const mahalDist = mahalanobisDistanceReuse(
        diff!,
        hiddenCholesky!,
        0,
        numContinuousHidden,
        yBuffer!
      );

      // Shifted Gaussian attenuation: scale · max(0, exp(-0.5·D²) - C)
      const rawExp = Math.exp(-0.5 * mahalDist * mahalDist);
      attenuation = Math.max(0.0, shiftedGaussianInvOneMinusC * (rawExp - shiftedGaussianC));
    }

    attenuations[i] = attenuation;
    const attenuatedAmplitude = loaded.amplitudes[i] * attenuation;

    if (attenuatedAmplitude >= minAmplitude) {
      visibleIndices.push(i);
      visibleCount++;
    }
  }

  // Allocate output arrays
  const centers3D = new Float32Array(visibleCount * 3);
  const choleskyFactors3D = new Float32Array(visibleCount * display3DPackedSize);
  const amplitudes = new Float32Array(visibleCount);
  const colors = new Float32Array(visibleCount * 3);

  // Color normalization factor (computed once, not per-splat)
  const normFactor = loaded.colors
    ? loaded.colors instanceof Uint8Array
      ? 1 / 255
      : loaded.colors instanceof Uint16Array
        ? 1 / 65535
        : 1
    : 0;

  // Second pass: extract visible splat data (attenuation reused from cache, NOT recomputed)
  for (let outIdx = 0; outIdx < visibleCount; outIdx++) {
    const srcIdx = visibleIndices[outIdx];
    const srcCenterOffset = srcIdx * ndim;
    const srcCholeskyOffset = srcIdx * fullPackedSize;

    // Extract 3D center using SORTED display dimensions (must match Cholesky order)
    const dstCenterOffset = outIdx * 3;
    for (let d = 0; d < 3 && d < sortedDisplayDims.length; d++) {
      centers3D[dstCenterOffset + d] = loaded.positions[srcCenterOffset + sortedDisplayDims[d]];
    }

    // Compute marginal Cholesky for display dimensions (3D)
    computeMarginalCholesky(
      loaded.choleskyFactors,
      srcCholeskyOffset,
      sortedDisplayDims,
      choleskyFactors3D,
      outIdx * display3DPackedSize
    );

    // Use cached attenuation from first pass (no recomputation!)
    amplitudes[outIdx] = loaded.amplitudes[srcIdx] * attenuations[srcIdx];

    // Copy colors with normalization (default to white if not present)
    const srcColorOffset = srcIdx * 3;
    const dstColorOffset = outIdx * 3;
    if (loaded.colors) {
      colors[dstColorOffset] = loaded.colors[srcColorOffset] * normFactor;
      colors[dstColorOffset + 1] = loaded.colors[srcColorOffset + 1] * normFactor;
      colors[dstColorOffset + 2] = loaded.colors[srcColorOffset + 2] * normFactor;
    } else {
      colors[dstColorOffset] = 1.0;
      colors[dstColorOffset + 1] = 1.0;
      colors[dstColorOffset + 2] = 1.0;
    }
  }

  return {
    centers3D,
    amplitudes,
    choleskyFactors3D,
    colors,
    splatCount: visibleCount,
  };
}

/**
 * Process gsplats for 3D-only datasets (no nD slicing needed).
 *
 * This is an optimized path when ndim === 3 and there are no hidden dimensions.
 * It simply copies the data with minimal transformation.
 *
 * @param loaded - Raw 3D gsplats data from zarr
 * @returns Processed 3D gsplats data ready for GPU
 */
export function processGSplats3DOnly(loaded: LoadedGSplatsData): ProcessedGSplatsData {
  if (loaded.ndim !== 3) {
    throw new Error(`processGSplats3DOnly requires ndim=3, got ${loaded.ndim}`);
  }

  const splatCount = loaded.splatCount;

  // Allocate output arrays
  const centers3D = new Float32Array(splatCount * 3);
  const choleskyFactors3D = new Float32Array(splatCount * 6);
  const amplitudes = new Float32Array(splatCount);
  const colors = new Float32Array(splatCount * 3);

  // Copy centers (already 3D)
  centers3D.set(loaded.positions);

  // Copy Cholesky factors (already 3D, 6 elements per splat)
  choleskyFactors3D.set(loaded.choleskyFactors);

  // Copy amplitudes
  amplitudes.set(loaded.amplitudes);

  // Copy colors with normalization (default to white if not present)
  if (loaded.colors) {
    // Normalize Uint8 (0-255) and Uint16 (0-65535) to Float32 (0-1)
    if (loaded.colors instanceof Float32Array) {
      colors.set(loaded.colors);
    } else {
      const normFactor = loaded.colors instanceof Uint8Array ? 1 / 255 : 1 / 65535;
      for (let i = 0; i < loaded.colors.length; i++) {
        colors[i] = loaded.colors[i] * normFactor;
      }
    }
  } else {
    colors.fill(1.0);
  }

  return {
    centers3D,
    amplitudes,
    choleskyFactors3D,
    colors,
    splatCount,
  };
}

/**
 * Smart processor that chooses the optimal path based on data dimensionality.
 *
 * @param loaded - Raw gsplats data from zarr
 * @param viewState - Current view state
 * @returns Processed 3D gsplats data ready for GPU
 */
export function processGSplats(
  loaded: LoadedGSplatsData,
  viewState: GSplatsViewState
): ProcessedGSplatsData {
  // Use optimized path for pure 3D data
  if (loaded.ndim === 3 && viewState.displayDims.length === 3) {
    // Check if displayDims are [0, 1, 2] (no dimension remapping needed)
    const isStandard3D =
      viewState.displayDims[0] === 0 &&
      viewState.displayDims[1] === 1 &&
      viewState.displayDims[2] === 2;

    if (isStandard3D) {
      return processGSplats3DOnly(loaded);
    }
  }

  // General nD → 3D processing
  return processGSplatsTo3D(loaded, viewState);
}
