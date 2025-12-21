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

/**
 * Extract a submatrix from packed lower-triangular Cholesky factors.
 *
 * @param packed - Full packed Cholesky (k = ndim*(ndim+1)/2 elements per splat)
 * @param offset - Start offset in the packed array for this splat
 * @param ndim - Original dimensionality
 * @param keepDims - Indices of dimensions to keep (sorted ascending)
 * @param output - Output array to write to
 * @param outputOffset - Start offset in output array
 */
function extractCholeskySubmatrix(
  packed: Float32Array,
  offset: number,
  ndim: number,
  keepDims: number[],
  output: Float32Array,
  outputOffset: number
): void {
  // Silence the unused parameter warning - ndim is kept for clarity and future use
  void ndim;

  const subNdim = keepDims.length;
  let outIdx = outputOffset;

  for (let subRow = 0; subRow < subNdim; subRow++) {
    const origRow = keepDims[subRow];
    for (let subCol = 0; subCol <= subRow; subCol++) {
      const origCol = keepDims[subCol];
      output[outIdx++] = packed[offset + packedIndex(origRow, origCol)];
    }
  }
}

/**
 * Compute Mahalanobis distance for a point using packed Cholesky factor.
 *
 * Given L (lower-triangular Cholesky of covariance),
 * Mahalanobis distance = ||L⁻¹ · (x - μ)||
 *
 * We use forward substitution to solve L·y = d, then ||y|| is the Mahalanobis distance.
 *
 * @param diff - Difference vector (x - μ) for the dimensions
 * @param packedL - Packed Cholesky factor [L00, L10, L11, ...]
 * @param offset - Start offset in packedL
 * @param ndim - Dimensionality of the Cholesky
 * @returns Mahalanobis distance
 */
function mahalanobisDistance(
  diff: number[],
  packedL: Float32Array,
  offset: number,
  ndim: number
): number {
  // Forward substitution: solve L · y = diff
  const y = new Array(ndim);

  for (let i = 0; i < ndim; i++) {
    let val = diff[i];
    for (let j = 0; j < i; j++) {
      val -= packedL[offset + packedIndex(i, j)] * y[j];
    }
    const diag = packedL[offset + packedIndex(i, i)];
    y[i] = diag > 1e-10 ? val / diag : 0;
  }

  // Compute ||y||
  let sumSq = 0;
  for (let i = 0; i < ndim; i++) {
    sumSq += y[i] * y[i];
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
 * @returns Processed 3D gsplats data ready for GPU
 */
export function processGSplatsTo3D(
  loaded: LoadedGSplatsData,
  viewState: GSplatsViewState
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

  // Compute packed sizes
  const fullPackedSize = (ndim * (ndim + 1)) / 2;
  const display3DPackedSize = 6; // 3D Cholesky has 6 elements
  const hiddenPackedSize = (sortedHiddenDims.length * (sortedHiddenDims.length + 1)) / 2;

  // Minimum amplitude threshold (splats with lower amplitude are invisible)
  const minAmplitude = 1e-6;

  // First pass: count visible splats
  let visibleCount = 0;
  const visibleIndices: number[] = [];

  for (let i = 0; i < splatCount; i++) {
    const sharpness = loaded.sharpness?.[i] ?? 2.0;

    // Compute hidden dimension distance if there are hidden dims
    let attenuation = 1.0;
    if (sortedHiddenDims.length > 0) {
      // Extract hidden dimension center components
      const centerOffset = i * ndim;
      const diff = sortedHiddenDims.map((d) => slicePosition[d] - loaded.centers[centerOffset + d]);

      // Extract hidden Cholesky submatrix
      const hiddenCholesky = new Float32Array(hiddenPackedSize);
      extractCholeskySubmatrix(
        loaded.choleskyFactors,
        i * fullPackedSize,
        ndim,
        sortedHiddenDims,
        hiddenCholesky,
        0
      );

      // Compute Mahalanobis distance in hidden dims
      const mahalDist = mahalanobisDistance(diff, hiddenCholesky, 0, sortedHiddenDims.length);

      // Attenuation: exp(-½ · mahal^sharpness)
      attenuation = Math.exp(-0.5 * Math.pow(mahalDist, sharpness));
    }

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
  const sharpness = new Float32Array(visibleCount);
  const colors = new Float32Array(visibleCount * 3);

  // Second pass: extract visible splat data
  for (let outIdx = 0; outIdx < visibleCount; outIdx++) {
    const srcIdx = visibleIndices[outIdx];
    const srcCenterOffset = srcIdx * ndim;
    const srcCholeskyOffset = srcIdx * fullPackedSize;

    // Extract 3D center using SORTED display dimensions (must match Cholesky order)
    const dstCenterOffset = outIdx * 3;
    for (let d = 0; d < 3 && d < sortedDisplayDims.length; d++) {
      centers3D[dstCenterOffset + d] = loaded.centers[srcCenterOffset + sortedDisplayDims[d]];
    }

    // Extract 3D Cholesky submatrix
    extractCholeskySubmatrix(
      loaded.choleskyFactors,
      srcCholeskyOffset,
      ndim,
      sortedDisplayDims,
      choleskyFactors3D,
      outIdx * display3DPackedSize
    );

    // Compute attenuated amplitude
    const srcSharpness = loaded.sharpness?.[srcIdx] ?? 2.0;
    sharpness[outIdx] = srcSharpness;

    let attenuation = 1.0;
    if (sortedHiddenDims.length > 0) {
      const diff = sortedHiddenDims.map(
        (d) => slicePosition[d] - loaded.centers[srcCenterOffset + d]
      );
      const hiddenCholesky = new Float32Array(hiddenPackedSize);
      extractCholeskySubmatrix(
        loaded.choleskyFactors,
        srcCholeskyOffset,
        ndim,
        sortedHiddenDims,
        hiddenCholesky,
        0
      );
      const mahalDist = mahalanobisDistance(diff, hiddenCholesky, 0, sortedHiddenDims.length);
      attenuation = Math.exp(-0.5 * Math.pow(mahalDist, srcSharpness));
    }

    amplitudes[outIdx] = loaded.amplitudes[srcIdx] * attenuation;

    // Copy colors (default to white if not present)
    const srcColorOffset = srcIdx * 3;
    const dstColorOffset = outIdx * 3;
    if (loaded.colors) {
      colors[dstColorOffset] = loaded.colors[srcColorOffset];
      colors[dstColorOffset + 1] = loaded.colors[srcColorOffset + 1];
      colors[dstColorOffset + 2] = loaded.colors[srcColorOffset + 2];
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
    sharpness,
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
  const sharpness = new Float32Array(splatCount);
  const colors = new Float32Array(splatCount * 3);

  // Copy centers (already 3D)
  centers3D.set(loaded.centers);

  // Copy Cholesky factors (already 3D, 6 elements per splat)
  choleskyFactors3D.set(loaded.choleskyFactors);

  // Copy amplitudes
  amplitudes.set(loaded.amplitudes);

  // Copy sharpness (default to 2.0 if not present)
  if (loaded.sharpness) {
    sharpness.set(loaded.sharpness);
  } else {
    sharpness.fill(2.0);
  }

  // Copy colors (default to white if not present)
  if (loaded.colors) {
    colors.set(loaded.colors);
  } else {
    for (let i = 0; i < splatCount; i++) {
      colors[i * 3] = 1.0;
      colors[i * 3 + 1] = 1.0;
      colors[i * 3 + 2] = 1.0;
    }
  }

  return {
    centers3D,
    amplitudes,
    choleskyFactors3D,
    colors,
    sharpness,
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
