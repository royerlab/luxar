/**
 * Spatial index queries and bounding box intersection tests.
 *
 * TypeScript reference implementation matching spatial.rs
 */

/**
 * Query chunks whose bounding boxes intersect the nD slice.
 *
 * @param chunkBounds - Flattened chunk bounds [numChunks * ndim * 2] (min/max pairs)
 * @param slicePosition - Current slice position in nD space [ndim]
 * @param tolerance - Tolerance per dimension [ndim]
 * @param ndim - Number of dimensions
 * @param numChunks - Total number of chunks
 * @param output - Output buffer for matching chunk indices [numChunks]
 * @returns Number of matching chunks
 */
export function query_chunks_for_view(
  chunkBounds: Float32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  ndim: number,
  numChunks: number,
  output: Uint32Array
): number {
  let count = 0;

  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    const boundsOffset = chunkIdx * ndim * 2;
    let intersects = true;

    // Check if chunk bounding box intersects the nD slice
    for (let dim = 0; dim < ndim; dim++) {
      const minBound = chunkBounds[boundsOffset + dim * 2];
      const maxBound = chunkBounds[boundsOffset + dim * 2 + 1];
      const slicePos = slicePosition[dim];
      const tol = tolerance[dim];

      // Check intersection: [minBound, maxBound] intersects [slicePos - tol, slicePos + tol]
      if (maxBound < slicePos - tol || minBound > slicePos + tol) {
        intersects = false;
        break;
      }
    }

    if (intersects) {
      output[count++] = chunkIdx;
    }
  }

  return count;
}
