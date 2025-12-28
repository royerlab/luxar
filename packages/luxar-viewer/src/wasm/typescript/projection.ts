/**
 * nD to 3D projection and bounds calculation.
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

/**
 * Calculate axis-aligned bounding box for 3D positions.
 *
 * @returns Number of points processed
 */
export function calculate_bounds_3d(
  positions3d: Float32Array,
  numPoints: number,
  output: Float32Array
): number {
  if (numPoints === 0) {
    for (let i = 0; i < 6; i++) {
      output[i] = 0;
    }
    return 0;
  }

  // Initialize with first point
  output[0] = positions3d[0]; // min_x
  output[1] = positions3d[1]; // min_y
  output[2] = positions3d[2]; // min_z
  output[3] = positions3d[0]; // max_x
  output[4] = positions3d[1]; // max_y
  output[5] = positions3d[2]; // max_z

  // Process remaining points
  for (let i = 1; i < numPoints; i++) {
    const offset = i * 3;
    const x = positions3d[offset];
    const y = positions3d[offset + 1];
    const z = positions3d[offset + 2];

    output[0] = Math.min(output[0], x);
    output[1] = Math.min(output[1], y);
    output[2] = Math.min(output[2], z);
    output[3] = Math.max(output[3], x);
    output[4] = Math.max(output[4], y);
    output[5] = Math.max(output[5], z);
  }

  return numPoints;
}

/**
 * Compact arrays by removing elements where mask[i] == 0.
 *
 * @returns Number of visible elements in output
 */
export function compact_by_mask(
  input: Float32Array,
  mask: Uint8Array,
  count: number,
  stride: number,
  output: Float32Array
): number {
  let outIdx = 0;

  for (let i = 0; i < count; i++) {
    if (mask[i] !== 0) {
      const srcOffset = i * stride;
      const dstOffset = outIdx * stride;

      for (let j = 0; j < stride; j++) {
        output[dstOffset + j] = input[srcOffset + j];
      }

      outIdx++;
    }
  }

  return outIdx;
}

/**
 * Count visible elements (non-zero mask values).
 */
export function count_visible(mask: Uint8Array, count: number): number {
  let visible = 0;
  for (let i = 0; i < count; i++) {
    if (mask[i] !== 0) {
      visible++;
    }
  }
  return visible;
}

/**
 * Create visibility mask from effective radii (radius > threshold is visible).
 *
 * @returns Number of visible elements
 */
export function radii_to_visibility_mask(
  radii: Float32Array,
  threshold: number,
  count: number,
  output: Uint8Array
): number {
  let visible = 0;

  for (let i = 0; i < count; i++) {
    if (radii[i] > threshold) {
      output[i] = 1;
      visible++;
    } else {
      output[i] = 0;
    }
  }

  return visible;
}
