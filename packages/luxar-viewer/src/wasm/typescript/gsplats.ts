/**
 * Gaussian splat visibility computation using ellipsoid extent.
 *
 * TypeScript reference implementation matching gsplats.rs
 */

/**
 * Compute nD visibility for GSplats using ellipsoid extent.
 *
 * Uses an optimistic estimate (max diagonal of L underestimates the true
 * ellipsoid extent for correlated covariances). This is acceptable as a
 * pre-filter since precise attenuation is computed later by
 * `compute_gsplats_attenuation`.
 *
 * @param centers - Splat centers [numSplats * ndim]
 * @param choleskyFactors - Packed Cholesky factors [numSplats * k] where k = ndim*(ndim+1)/2
 * @param slicePosition - Current slice position [ndim]
 * @param tolerance - Tolerance per dimension [ndim]
 * @param ndim - Number of dimensions
 * @param numSplats - Total number of splats
 * @param output - Output visibility mask [numSplats]
 * @returns Number of visible splats
 */
export function compute_nd_visibility_gsplats(
  centers: Float32Array,
  choleskyFactors: Float32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  ndim: number,
  numSplats: number,
  output: Uint8Array
): number {
  const choleskySize = (ndim * (ndim + 1)) / 2;
  let visibleCount = 0;

  for (let splatIdx = 0; splatIdx < numSplats; splatIdx++) {
    const centerOffset = splatIdx * ndim;
    const choleskyOffset = splatIdx * choleskySize;

    // Compute maximum ellipsoid extent from Cholesky factors
    // Diagonal elements of L give axis scales (L is lower triangular)
    let maxExtent = 0;
    let cholIdx = 0;

    for (let dim = 0; dim < ndim; dim++) {
      // Diagonal element L[dim,dim] is at packed position:
      // dim + (dim-1) + (dim-2) + ... + 0 = dim*(dim+1)/2
      const diagPos = choleskyOffset + cholIdx + dim;
      const scale = Math.abs(choleskyFactors[diagPos]);
      maxExtent = Math.max(maxExtent, scale);
      cholIdx += dim + 1; // Move to next row's diagonal
    }

    // Check if center + max extent is within tolerance
    let distSq = 0;
    let shouldBreak = false;

    for (let dim = 0; dim < ndim; dim++) {
      const delta = centers[centerOffset + dim] - slicePosition[dim];
      const effectiveTolerance = tolerance[dim] + maxExtent;

      if (effectiveTolerance > 0) {
        const normalized = delta / effectiveTolerance;
        distSq += normalized * normalized;
      } else if (Math.abs(delta) > 1e-6) {
        distSq = Infinity;
        shouldBreak = true;
        break;
      }
    }

    if (shouldBreak) {
      output[splatIdx] = 0;
      continue;
    }

    const visible = distSq <= 1.0;
    output[splatIdx] = visible ? 1 : 0;
    if (visible) {
      visibleCount++;
    }
  }

  return visibleCount;
}
