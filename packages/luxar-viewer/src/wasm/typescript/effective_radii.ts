/**
 * Effective radius calculation for nD hypersphere slicing.
 *
 * TypeScript reference implementation matching effective_radii.rs
 *
 * When an nD hypersphere of radius R is intersected by a hyperplane at distance D,
 * the effective radius in the slice is: R_effective = sqrt(R² - D²)
 */

/**
 * Calculate effective radii for nD points when sliced.
 *
 * @param positions - Point positions [numPoints * ndim]
 * @param radii - Original point radii [numPoints]
 * @param displayDims - Dimensions to display (typically [0,1,2]) [numDisplayDims]
 * @param slicePosition - Current slice position [ndim]
 * @param spatialExtendDims - Which dims are spatial (1) vs discrete (0) [ndim]
 * @param ndim - Total number of dimensions
 * @param numPoints - Number of points
 * @param output - Output effective radii [numPoints]
 * @returns Number of points with non-zero effective radius (visible points)
 */
export function calculate_effective_radii(
  positions: Float32Array,
  radii: Float32Array,
  displayDims: Uint32Array,
  slicePosition: Float32Array,
  spatialExtendDims: Uint8Array,
  ndim: number,
  numPoints: number,
  output: Float32Array
): number {
  const discreteTolerance = 0.5;
  let visibleCount = 0;

  // Create display dims lookup (Uint8Array bitmap is faster than Set in hot loops)
  const isDisplayDim = new Uint8Array(ndim);
  for (let i = 0; i < displayDims.length; i++) {
    isDisplayDim[displayDims[i]] = 1;
  }

  for (let i = 0; i < numPoints; i++) {
    const originalRadius = radii[i];
    const posOffset = i * ndim;

    // Fused loop: check discrete match AND compute spatial distance in one pass
    // Matches the Rust implementation in effective_radii.rs
    let discreteMatch = true;
    let distanceSquared = 0;
    for (let d = 0; d < ndim; d++) {
      if (isDisplayDim[d]) continue;

      // Check if this is a spatial or discrete dimension
      const isSpatial = d < spatialExtendDims.length ? spatialExtendDims[d] !== 0 : true;

      const value = positions[posOffset + d];
      const target = slicePosition[d];

      if (isSpatial) {
        // Spatial dimension: accumulate squared distance
        const diff = value - target;
        distanceSquared += diff * diff;
      } else {
        // Discrete dimension: must match exactly (within tolerance)
        if (Math.abs(value - target) > discreteTolerance) {
          discreteMatch = false;
          break;
        }
      }
    }

    if (!discreteMatch) {
      output[i] = 0;
      continue;
    }

    // Apply Pythagorean theorem: R_effective = sqrt(R² - D²)
    const radiusSquared = originalRadius * originalRadius;
    if (distanceSquared >= radiusSquared) {
      output[i] = 0;
    } else {
      output[i] = Math.sqrt(radiusSquared - distanceSquared);
      visibleCount++;
    }
  }

  return visibleCount;
}
