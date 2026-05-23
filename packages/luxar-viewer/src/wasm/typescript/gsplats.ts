/**
 * Gaussian splat visibility computation using ellipsoid extent.
 *
 * TypeScript reference implementation matching gsplats.rs
 */

/**
 * Compute nD visibility for GSplats using ellipsoid extent.
 *
 * Uses row norms of L to estimate the marginal standard deviation per axis.
 * Row norm of row i = sqrt(sum_j L[i,j]^2) gives the correct extent even
 * for correlated covariances. The max row norm is used as a conservative
 * pre-filter; precise attenuation is computed later by
 * `compute_gsplats_attenuation`.
 *
 * Limitation (MED-45): taking the max of the row norms produces a single
 * scalar extent applied isotropically along every axis. For highly
 * anisotropic splats — e.g. a needle-shaped Gaussian stretched along
 * one axis — this bound is tight along the stretched axis and loose
 * along the others, so the visibility set is a strict superset of the
 * true visible splats. This is **intentional and acceptable** because
 * (a) this routine is a coarse pre-filter, not the final attenuation
 * gate, and (b) the precise per-splat attenuation that follows in
 * `compute_gsplats_attenuation` discards any splat whose contribution
 * is negligible at the current slice. A tighter per-axis bound would
 * cost more work here and save very little downstream.
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

    // Compute maximum ellipsoid extent from Cholesky row norms.
    // Row norm of row i = sqrt(sum_j L[i,j]^2) gives the marginal
    // standard deviation along axis i (correct for correlated covariances).
    let maxExtent = 0;

    for (let dim = 0; dim < ndim; dim++) {
      // Row `dim` has elements at packed positions dim*(dim+1)/2 + col for col in 0..=dim
      const rowStart = choleskyOffset + (dim * (dim + 1)) / 2;
      let rowNormSq = 0;
      for (let col = 0; col <= dim; col++) {
        const val = choleskyFactors[rowStart + col];
        rowNormSq += val * val;
      }
      maxExtent = Math.max(maxExtent, Math.sqrt(rowNormSq));
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
