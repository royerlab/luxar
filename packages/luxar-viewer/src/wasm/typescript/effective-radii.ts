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
 * @param spatialExtendDims - Which dims are spatial (1) vs discrete (0).
 *   Callers SHOULD pass an array of length `ndim` for explicit control.
 *   For dimensions `d >= spatialExtendDims.length`, this implementation
 *   DEFAULTS to spatial (treats the missing entry as `1`). This is the more
 *   permissive fallback (extra dims contribute to the Pythagorean distance
 *   instead of silently dropping the point as a discrete mismatch); production
 *   callers in this codebase enforce `length >= ndim` at the worker boundary
 *   (see `projectPointsTo3D`). [length: SHOULD be ndim]
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
  // Rust's `discrete_tolerance` is `0.5_f32`; 0.5 is exactly representable, so
  // no rounding is needed on the constant itself.
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
        // Spatial dimension: accumulate squared distance.
        // Rust rounds EVERY step to f32 — the subtraction, the square, and each
        // partial sum — while a plain JS chain keeps f64 all the way. That is not
        // a cosmetic ulp: `distanceSquared` is compared against `radiusSquared`
        // below, so a point a hair outside its own radius can be VISIBLE on one
        // backend and culled on the other, changing `visibleCount` and the
        // compaction the caller derives from it. (`value`/`target` come out of
        // Float32Arrays, so their difference is normally exact in f64 — but Rust
        // still rounds it, hence the fround on the subtraction too. That one is
        // load-bearing whenever the slice position is not zero: dropping it
        // alone moves 212/20000 outputs, by up to 2527 ulp, on the randomized
        // sweep in `tests/unit/wasm/wasm-vs-typescript.test.ts`. With an
        // all-zero slice position it is a no-op, which is why that fixture
        // deliberately does not use one.)
        const diff = Math.fround(value - target);
        distanceSquared = Math.fround(distanceSquared + Math.fround(diff * diff));
      } else {
        // Discrete dimension: must match exactly (within tolerance)
        if (Math.abs(Math.fround(value - target)) > discreteTolerance) {
          discreteMatch = false;
          break;
        }
      }
    }

    if (!discreteMatch) {
      output[i] = 0;
      continue;
    }

    // Apply Pythagorean theorem: R_effective = sqrt(R² - D²).
    // Both the square and the difference are f32 operations in Rust; the
    // difference in particular cancels catastrophically for a point near the
    // rim of its own radius, which is exactly where the two backends used to
    // disagree: 615/20000 outputs, up to 1003 ulp, on the randomized sweep in
    // `tests/unit/wasm/wasm-vs-typescript.test.ts` before this fix.
    const radiusSquared = Math.fround(originalRadius * originalRadius);
    if (distanceSquared >= radiusSquared) {
      output[i] = 0;
    } else {
      output[i] = Math.sqrt(Math.fround(radiusSquared - distanceSquared));
      visibleCount++;
    }
  }

  return visibleCount;
}
