/**
 * Point visibility computation using hypersphere intersection.
 *
 * TypeScript reference implementation matching points.rs
 */

/**
 * Compute nD visibility for points using hypersphere intersection.
 *
 * @param positions - Point positions [numPoints * ndim]
 * @param radii - Point radii [numPoints]
 * @param slicePosition - Current slice position [ndim]
 * @param tolerance - Tolerance per dimension [ndim]
 * @param ndim - Number of dimensions
 * @param numPoints - Total number of points
 * @param output - Output visibility mask [numPoints] (1=visible, 0=hidden)
 * @returns Number of visible points
 */
export function compute_nd_visibility_points(
  positions: Float32Array,
  radii: Float32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  ndim: number,
  numPoints: number,
  output: Uint8Array
): number {
  let visibleCount = 0;

  for (let ptIdx = 0; ptIdx < numPoints; ptIdx++) {
    const ptOffset = ptIdx * ndim;
    const radius = radii[ptIdx];

    // Compute normalized distance in nD space
    let distSq = 0;
    let shouldBreak = false;

    for (let dim = 0; dim < ndim; dim++) {
      const delta = positions[ptOffset + dim] - slicePosition[dim];
      const effectiveTolerance = tolerance[dim] + radius;

      // Avoid division by zero
      if (effectiveTolerance > 0) {
        const normalized = delta / effectiveTolerance;
        distSq += normalized * normalized;
      } else if (Math.abs(delta) > 1e-6) {
        // Point is far from slice with zero tolerance - not visible
        distSq = Infinity;
        shouldBreak = true;
        break;
      }
    }

    if (shouldBreak) {
      output[ptIdx] = 0;
      continue;
    }

    const visible = distSq <= 1.0;
    output[ptIdx] = visible ? 1 : 0;
    if (visible) {
      visibleCount++;
    }
  }

  return visibleCount;
}
