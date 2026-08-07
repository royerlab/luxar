/**
 * nD to 3D projection.
 *
 * TypeScript reference implementation matching projection.rs
 */

/**
 * Extract 3D positions from nD positions using display dimension indices.
 */
export function extract_3d_positions(
  positionsNd: Float32Array,
  displayDims: Uint32Array,
  ndim: number,
  numPoints: number,
  output: Float32Array
): void {
  const numDisplayDims = Math.min(displayDims.length, 3);

  for (let i = 0; i < numPoints; i++) {
    const srcOffset = i * ndim;
    const dstOffset = i * 3;

    // Extract displayed dimensions
    for (let j = 0; j < numDisplayDims; j++) {
      const dimIdx = displayDims[j];
      output[dstOffset + j] = positionsNd[srcOffset + dimIdx];
    }

    // Fill remaining with zeros
    for (let j = numDisplayDims; j < 3; j++) {
      output[dstOffset + j] = 0;
    }
  }
}
