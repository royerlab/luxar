/**
 * Line segment visibility computation (endpoint-based).
 *
 * TypeScript reference implementation matching lines.rs
 */

/**
 * Check if a single vertex is visible in the nD slice.
 */
function checkPointVisibility(
  vertices: Float32Array,
  vertexIdx: number,
  width: number,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  ndim: number
): boolean {
  const offset = vertexIdx * ndim;
  let distSq = 0;

  for (let dim = 0; dim < ndim; dim++) {
    const delta = vertices[offset + dim] - slicePosition[dim];
    const effectiveTolerance = tolerance[dim] + width;

    if (effectiveTolerance > 0) {
      const normalized = delta / effectiveTolerance;
      distSq += normalized * normalized;
    } else if (Math.abs(delta) > 1e-6) {
      return false;
    }
  }

  return distSq <= 1.0;
}

/**
 * Compute nD visibility for line segments (endpoint-based).
 *
 * A segment is visible if EITHER endpoint is visible.
 *
 * @param vertices - Vertex positions [numVertices * ndim]
 * @param segments - Segment indices [numSegments * 2] (pairs of vertex indices)
 * @param widths - Per-vertex widths [numVertices]
 * @param slicePosition - Current slice position [ndim]
 * @param tolerance - Tolerance per dimension [ndim]
 * @param ndim - Number of dimensions
 * @param numSegments - Total number of segments
 * @param output - Output visibility mask [numSegments]
 * @returns Number of visible segments
 */
export function compute_nd_visibility_lines(
  vertices: Float32Array,
  segments: Uint32Array,
  widths: Float32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  ndim: number,
  numSegments: number,
  output: Uint8Array
): number {
  let visibleCount = 0;

  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    const v0Idx = segments[segIdx * 2];
    const v1Idx = segments[segIdx * 2 + 1];

    // Get per-vertex widths
    const width0 = widths[v0Idx];
    const width1 = widths[v1Idx];

    // Check if EITHER endpoint is visible
    const v0Visible = checkPointVisibility(vertices, v0Idx, width0, slicePosition, tolerance, ndim);
    const v1Visible = checkPointVisibility(vertices, v1Idx, width1, slicePosition, tolerance, ndim);

    const visible = v0Visible || v1Visible;
    output[segIdx] = visible ? 1 : 0;
    if (visible) {
      visibleCount++;
    }
  }

  return visibleCount;
}
