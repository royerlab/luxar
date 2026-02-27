/**
 * GSplats Processing for nD → 3D Conversion
 *
 * TypeScript reference implementation matching gsplats_processing.rs
 *
 * High-performance implementations of GSplats processing operations:
 * - Mahalanobis distance computation (forward substitution)
 * - Cholesky submatrix extraction
 * - Batch attenuation computation for visibility filtering
 */

/** Maximum supported dimensions (must match Rust MAX_SUPPORTED_DIMS). */
const MAX_SUPPORTED_DIMS = 16;

/** Epsilon for degenerate diagonal detection during Cholesky factorization. */
const CHOLESKY_EPSILON = 1e-10;

/**
 * Compute the packed index for a Cholesky element L[row, col].
 * Packed lower-triangular: [L00, L10, L11, L20, L21, L22, ...]
 * Formula: row * (row + 1) / 2 + col (for col <= row)
 */
function packedIndex(row: number, col: number): number {
  return (row * (row + 1)) / 2 + col;
}

/**
 * Compute the correct marginal Cholesky factor for a subset of dimensions.
 *
 * For Σ = L·Lᵀ, the marginal covariance for dimensions S is:
 *   Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]
 *
 * This function reconstructs Σ_S and then Cholesky-factorizes it.
 * Simply extracting L elements (as extract_cholesky_submatrix does)
 * is INCORRECT when there are cross-dimension correlations.
 *
 * @param fullPackedL - Full packed Cholesky factor array
 * @param fullPackedOffset - Offset into fullPackedL for this splat
 * @param keepDims - Dimension indices to keep (sorted ascending)
 * @param subNdim - Number of dimensions to keep
 * @param output - Output packed marginal Cholesky [subNdim*(subNdim+1)/2]
 * @param outputOffset - Start offset in output array
 */
export function computeMarginalCholesky(
  fullPackedL: Float32Array,
  fullPackedOffset: number,
  keepDims: Uint32Array | number[],
  subNdim: number,
  output: Float32Array,
  outputOffset: number
): void {
  // Step 1: Reconstruct marginal covariance Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]
  const sigma = new Float32Array(MAX_SUPPORTED_DIMS * MAX_SUPPORTED_DIMS);

  for (let i = 0; i < subNdim; i++) {
    const si = keepDims[i];
    for (let j = 0; j <= i; j++) {
      const sj = keepDims[j];
      const kMax = Math.min(si, sj);
      let sum = 0;
      for (let k = 0; k <= kMax; k++) {
        const lSiK = fullPackedL[fullPackedOffset + packedIndex(si, k)];
        const lSjK = fullPackedL[fullPackedOffset + packedIndex(sj, k)];
        sum += lSiK * lSjK;
      }
      sigma[i * MAX_SUPPORTED_DIMS + j] = sum;
      sigma[j * MAX_SUPPORTED_DIMS + i] = sum;
    }
  }

  // Step 2: Cholesky-Crout factorization of Σ_S → L_S
  const subPackedSize = (subNdim * (subNdim + 1)) / 2;
  const lSub = new Float32Array(subPackedSize);

  for (let i = 0; i < subNdim; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = sigma[i * MAX_SUPPORTED_DIMS + j];
      for (let k = 0; k < j; k++) {
        sum -= lSub[packedIndex(i, k)] * lSub[packedIndex(j, k)];
      }
      if (i === j) {
        lSub[packedIndex(i, i)] =
          sum > CHOLESKY_EPSILON ? Math.sqrt(sum) : Math.sqrt(CHOLESKY_EPSILON);
      } else {
        const diag = lSub[packedIndex(j, j)];
        lSub[packedIndex(i, j)] = diag > CHOLESKY_EPSILON ? sum / diag : 0;
      }
    }
  }

  // Copy to output
  for (let k = 0; k < subPackedSize; k++) {
    output[outputOffset + k] = lSub[k];
  }
}

/**
 * Compute Mahalanobis distance for a single point using packed Cholesky factor.
 *
 * Given L (lower-triangular Cholesky of covariance),
 * Mahalanobis distance = ||L⁻¹ · (x - μ)||
 *
 * We use forward substitution to solve L·y = d, then ||y|| is the Mahalanobis distance.
 *
 * @param diff - Difference vector (x - μ) for the dimensions [ndim]
 * @param packedL - Packed Cholesky factor [packedSize]
 * @param ndim - Dimensionality of the Cholesky
 * @returns Mahalanobis distance
 */
export function mahalanobis_distance(
  diff: Float32Array,
  packedL: Float32Array,
  ndim: number
): number {
  // Forward substitution: solve L · y = diff
  const y = new Float32Array(ndim);

  for (let i = 0; i < ndim; i++) {
    let val = diff[i];
    for (let j = 0; j < i; j++) {
      val -= packedL[packedIndex(i, j)] * y[j];
    }
    const diag = packedL[packedIndex(i, i)];
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
 * Extract a Cholesky submatrix for specified dimensions.
 *
 * **WARNING**: This extracts raw L elements, NOT the correct marginal Cholesky factor.
 * For Σ = L·Lᵀ, the Cholesky of marginal covariance Σ_S ≠ submatrix of L when there
 * are cross-dimension correlations. Use `computeMarginalCholesky()` instead for
 * correct results with correlated covariances.
 *
 * @param packed - Full packed Cholesky [packedSize]
 * @param keepDims - Indices of dimensions to keep (must be sorted ascending) [subNdim]
 * @param subNdim - Number of dimensions to keep
 * @param output - Output packed submatrix [subPackedSize]
 */
export function extract_cholesky_submatrix(
  packed: Float32Array,
  keepDims: Uint32Array,
  subNdim: number,
  output: Float32Array
): void {
  let outIdx = 0;

  for (let subRow = 0; subRow < subNdim; subRow++) {
    const origRow = keepDims[subRow];
    for (let subCol = 0; subCol <= subRow; subCol++) {
      const origCol = keepDims[subCol];
      output[outIdx++] = packed[packedIndex(origRow, origCol)];
    }
  }
}

/**
 * Compute attenuation factors for all GSplats based on hidden dimension distance.
 *
 * For each splat, computes:
 * 1. Difference vector in hidden dimensions
 * 2. Mahalanobis distance using hidden Cholesky submatrix
 * 3. Attenuation = exp(-0.5 * mahal^sharpness)
 * 4. Visibility = (amplitude * attenuation) >= threshold
 *
 * @param positions - Splat centers [splatCount * ndim]
 * @param cholesky - Packed Cholesky factors [splatCount * packedSize]
 * @param amplitudes - Splat amplitudes [splatCount]
 * @param sharpness - Per-splat sharpness values [splatCount]
 * @param slicePosition - Current slice position [ndim]
 * @param hiddenDims - Indices of hidden dimensions (sorted) [numHidden]
 * @param ndim - Total dimensionality
 * @param splatCount - Number of splats
 * @param minAmplitude - Visibility threshold
 * @param outputVisibility - Output visibility mask [splatCount]
 * @param outputAttenuation - Output attenuation factors [splatCount]
 * @returns Number of visible splats
 */
export function compute_gsplats_attenuation(
  positions: Float32Array,
  cholesky: Float32Array,
  amplitudes: Float32Array,
  sharpness: Float32Array,
  slicePosition: Float32Array,
  hiddenDims: Uint32Array,
  ndim: number,
  splatCount: number,
  minAmplitude: number,
  outputVisibility: Uint8Array,
  outputAttenuation: Float32Array
): number {
  const numHidden = hiddenDims.length;
  const fullPackedSize = (ndim * (ndim + 1)) / 2;
  const hiddenPackedSize = (numHidden * (numHidden + 1)) / 2;

  // Temporary buffers (pre-allocated, reused across all splats)
  const diff = new Float32Array(numHidden);
  const hiddenCholesky = new Float32Array(hiddenPackedSize);
  const y = new Float32Array(numHidden); // Forward substitution buffer

  let visibleCount = 0;

  for (let i = 0; i < splatCount; i++) {
    const centerOffset = i * ndim;
    const choleskyOffset = i * fullPackedSize;
    const splatSharpness = sharpness[i];

    let attenuation: number;

    if (numHidden === 0) {
      // No hidden dimensions, full visibility
      attenuation = 1.0;
    } else {
      // Compute difference vector in hidden dimensions
      for (let hIdx = 0; hIdx < numHidden; hIdx++) {
        const d = hiddenDims[hIdx];
        diff[hIdx] = slicePosition[d] - positions[centerOffset + d];
      }

      // Compute marginal Cholesky for hidden dimensions
      computeMarginalCholesky(cholesky, choleskyOffset, hiddenDims, numHidden, hiddenCholesky, 0);

      // Compute Mahalanobis distance (reuses pre-allocated y buffer)
      const mahalDist = mahalanobisDistanceInternal(diff, hiddenCholesky, numHidden, y);

      // Attenuation = exp(-0.5 * mahal^sharpness)
      attenuation = Math.exp(-0.5 * Math.pow(mahalDist, splatSharpness));
    }

    outputAttenuation[i] = attenuation;

    const attenuatedAmplitude = amplitudes[i] * attenuation;
    const visible = attenuatedAmplitude >= minAmplitude;
    outputVisibility[i] = visible ? 1 : 0;
    if (visible) {
      visibleCount++;
    }
  }

  return visibleCount;
}

/**
 * Internal Mahalanobis distance (matches Rust internal function).
 * Accepts an optional pre-allocated buffer to avoid per-call allocation.
 */
function mahalanobisDistanceInternal(
  diff: Float32Array,
  packedL: Float32Array,
  ndim: number,
  yBuffer?: Float32Array
): number {
  const y = yBuffer ?? new Float32Array(ndim);

  for (let i = 0; i < ndim; i++) {
    let val = diff[i];
    for (let j = 0; j < i; j++) {
      val -= packedL[packedIndex(i, j)] * y[j];
    }
    const diag = packedL[packedIndex(i, i)];
    y[i] = diag > 1e-10 ? val / diag : 0;
  }

  let sumSq = 0;
  for (let i = 0; i < ndim; i++) {
    sumSq += y[i] * y[i];
  }
  return Math.sqrt(sumSq);
}

/**
 * Extract 3D Cholesky submatrices for visible splats.
 *
 * @param cholesky - Packed Cholesky factors [splatCount * packedSize]
 * @param visibility - Visibility mask [splatCount]
 * @param displayDims - Display dimension indices (sorted) [3]
 * @param ndim - Total dimensionality
 * @param splatCount - Number of splats
 * @param output - Output 3D Cholesky factors [visibleCount * 6]
 * @returns Number of visible splats processed
 */
export function extract_visible_cholesky_3d(
  cholesky: Float32Array,
  visibility: Uint8Array,
  displayDims: Uint32Array,
  ndim: number,
  splatCount: number,
  output: Float32Array
): number {
  const fullPackedSize = (ndim * (ndim + 1)) / 2;
  let outSplat = 0;

  for (let i = 0; i < splatCount; i++) {
    if (visibility[i] === 0) {
      continue;
    }

    const srcOffset = i * fullPackedSize;
    const dstOffset = outSplat * 6;

    // Compute marginal Cholesky for display dimensions (3D)
    computeMarginalCholesky(cholesky, srcOffset, displayDims, 3, output, dstOffset);

    outSplat++;
  }

  return outSplat;
}

/**
 * Compact amplitudes by visibility mask, applying attenuation.
 *
 * @param amplitudes - Original amplitudes [splatCount]
 * @param attenuation - Attenuation factors [splatCount]
 * @param visibility - Visibility mask [splatCount]
 * @param splatCount - Number of splats
 * @param output - Output attenuated amplitudes [visibleCount]
 * @returns Number of visible splats
 */
export function compact_attenuated_amplitudes(
  amplitudes: Float32Array,
  attenuation: Float32Array,
  visibility: Uint8Array,
  splatCount: number,
  output: Float32Array
): number {
  let outIdx = 0;

  for (let i = 0; i < splatCount; i++) {
    if (visibility[i] !== 0) {
      output[outIdx++] = amplitudes[i] * attenuation[i];
    }
  }

  return outIdx;
}
