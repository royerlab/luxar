/**
 * Lines Clipping for nD → 3D Conversion
 *
 * TypeScript reference implementation matching lines_clipping.rs
 *
 * ## Clipping Algorithm (Liang-Barsky based)
 *
 * For each segment, handles 5 visibility cases:
 * - A: Both endpoints IN slice → Full segment visible
 * - B: P1 IN, P2 OUT → Clip P2 to boundary
 * - C: P1 OUT, P2 IN → Clip P1 to boundary
 * - D: Both OUT, opposite sides → Clip both (segment crosses slice)
 * - E: Both OUT, same side → Invisible
 */

/**
 * Threshold below which a segment is treated as parallel to the slice in a
 * hidden dimension (so clipping in that dim is skipped). MUST stay identical to
 * the Rust crate's `common::SEGMENT_PARALLEL_EPSILON` — a mismatch silently
 * diverges visibility for near-parallel segments (this was previously 1e-10 here
 * vs 1e-7 in Rust).
 */
const SEGMENT_PARALLEL_EPSILON = 1e-7;

/**
 * Clip a single segment to the nD slice and return interpolation parameters.
 *
 * Returns [visible, t1, t2] where:
 * - visible: 1.0 if any part of segment intersects slice, 0.0 otherwise
 * - t1: interpolation parameter for clipped start (0.0 = original start)
 * - t2: interpolation parameter for clipped end (1.0 = original end)
 *
 * MED-20: pass `workspace` (length >= ndim) to avoid per-call Uint8Array
 * allocation in hot loops; the buffer is zeroed by this function before use.
 */
export function clip_segment_single(
  p1: Float32Array,
  p2: Float32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  displayDims: Uint32Array,
  ndim: number,
  workspace?: Uint8Array
): Float32Array {
  let t1 = 0.0;
  let t2 = 1.0;

  // MED-20: reuse caller-provided workspace when available, otherwise
  // allocate one. We zero the first `ndim` slots before populating —
  // the caller may reuse the same buffer across many segments and we
  // must not see stale 1s for non-display dims.
  let displaySet: Uint8Array;
  if (workspace !== undefined && workspace.length >= ndim) {
    displaySet = workspace;
    for (let i = 0; i < ndim; i++) displaySet[i] = 0;
  } else {
    displaySet = new Uint8Array(ndim);
  }
  for (let i = 0; i < displayDims.length; i++) {
    displaySet[displayDims[i]] = 1;
  }

  for (let dim = 0; dim < ndim; dim++) {
    if (displaySet[dim]) {
      continue; // Skip displayed dimensions
    }

    const tol = tolerance[dim];
    const sliceCenter = slicePosition[dim];
    const sliceMin = sliceCenter - tol;
    const sliceMax = sliceCenter + tol;

    const v1 = p1[dim];
    const v2 = p2[dim];

    // Classify endpoints relative to slice
    const p1In = v1 >= sliceMin && v1 <= sliceMax;
    const p2In = v2 >= sliceMin && v2 <= sliceMax;

    if (p1In && p2In) {
      continue; // Both in - no clipping for this dimension
    }

    if (!p1In && !p2In) {
      // Both out - check if on same side (Case E: invisible)
      if ((v1 < sliceMin && v2 < sliceMin) || (v1 > sliceMax && v2 > sliceMax)) {
        return new Float32Array([0.0, 0.0, 0.0]); // [visible=0, t1, t2]
      }
      // Opposite sides - will clip both (Case D)
    }

    // Compute intersection parameters
    const dv = v2 - v1;
    if (Math.abs(dv) < SEGMENT_PARALLEL_EPSILON) {
      continue; // Parallel to slice
    }

    const tMin = (sliceMin - v1) / dv;
    const tMax = (sliceMax - v1) / dv;

    // Clip t1 (entry) and t2 (exit)
    if (dv > 0) {
      t1 = Math.max(t1, tMin);
      t2 = Math.min(t2, tMax);
    } else {
      t1 = Math.max(t1, tMax);
      t2 = Math.min(t2, tMin);
    }

    if (t1 >= t2) {
      return new Float32Array([0.0, 0.0, 0.0]); // No valid range
    }
  }

  return new Float32Array([1.0, t1, t2]); // [visible=1, t1, t2]
}

/**
 * Batch clip all segments and output visibility mask and interpolation parameters.
 *
 * @param positions - Vertex positions [numVertices * ndim]
 * @param segments - Segment indices [numSegments * 2]
 * @param slicePosition - Current slice position [ndim]
 * @param tolerance - Per-dimension tolerance [ndim]
 * @param displayDims - Which dimensions to display [numDisplayDims]
 * @param ndim - Number of dimensions
 * @param numSegments - Number of segments
 * @param outputVisibility - Output visibility mask [numSegments]
 * @param outputT1 - Output t1 parameters [numSegments]
 * @param outputT2 - Output t2 parameters [numSegments]
 * @returns Number of visible segments
 */
export function clip_segments_batch(
  positions: Float32Array,
  segments: Uint32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  displayDims: Uint32Array,
  ndim: number,
  numSegments: number,
  outputVisibility: Uint8Array,
  outputT1: Float32Array,
  outputT2: Float32Array
): number {
  const displaySetBatch = new Uint8Array(ndim);
  for (let i = 0; i < displayDims.length; i++) {
    displaySetBatch[displayDims[i]] = 1;
  }
  let visibleCount = 0;

  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    const v0 = segments[segIdx * 2];
    const v1 = segments[segIdx * 2 + 1];

    const p1Offset = v0 * ndim;
    const p2Offset = v1 * ndim;

    let t1 = 0.0;
    let t2 = 1.0;
    let visible = true;

    for (let dim = 0; dim < ndim; dim++) {
      if (displaySetBatch[dim]) {
        continue;
      }

      const tol = tolerance[dim];
      const sliceCenter = slicePosition[dim];
      const sliceMin = sliceCenter - tol;
      const sliceMax = sliceCenter + tol;

      const v1Val = positions[p1Offset + dim];
      const v2Val = positions[p2Offset + dim];

      const p1In = v1Val >= sliceMin && v1Val <= sliceMax;
      const p2In = v2Val >= sliceMin && v2Val <= sliceMax;

      if (p1In && p2In) {
        continue;
      }

      if (!p1In && !p2In) {
        if ((v1Val < sliceMin && v2Val < sliceMin) || (v1Val > sliceMax && v2Val > sliceMax)) {
          visible = false;
          break;
        }
      }

      const dv = v2Val - v1Val;
      if (Math.abs(dv) < SEGMENT_PARALLEL_EPSILON) {
        continue;
      }

      const tMin = (sliceMin - v1Val) / dv;
      const tMax = (sliceMax - v1Val) / dv;

      if (dv > 0) {
        t1 = Math.max(t1, tMin);
        t2 = Math.min(t2, tMax);
      } else {
        t1 = Math.max(t1, tMax);
        t2 = Math.min(t2, tMin);
      }

      if (t1 >= t2) {
        visible = false;
        break;
      }
    }

    outputVisibility[segIdx] = visible ? 1 : 0;
    outputT1[segIdx] = t1;
    outputT2[segIdx] = t2;

    if (visible) {
      visibleCount++;
    }
  }

  return visibleCount;
}

/**
 * Interpolate clipped positions to 3D display space.
 *
 * @param positions - Vertex positions [numVertices * ndim]
 * @param segments - Segment indices [numSegments * 2]
 * @param visibility - Visibility mask [numSegments]
 * @param t1Params - Start interpolation parameters [numSegments]
 * @param t2Params - End interpolation parameters [numSegments]
 * @param displayDims - Which dimensions to display [3]
 * @param ndim - Number of dimensions
 * @param numSegments - Total number of segments
 * @param outputStart - Output start positions [visibleCount * 3]
 * @param outputEnd - Output end positions [visibleCount * 3]
 * @returns Number of visible segments written
 */
export function interpolate_clipped_positions(
  positions: Float32Array,
  segments: Uint32Array,
  visibility: Uint8Array,
  t1Params: Float32Array,
  t2Params: Float32Array,
  displayDims: Uint32Array,
  ndim: number,
  numSegments: number,
  outputStart: Float32Array,
  outputEnd: Float32Array
): number {
  const numDisplay = Math.min(displayDims.length, 3);
  let outIdx = 0;

  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    if (visibility[segIdx] === 0) {
      continue;
    }

    const v0 = segments[segIdx * 2];
    const v1 = segments[segIdx * 2 + 1];
    const t1 = t1Params[segIdx];
    const t2 = t2Params[segIdx];

    const p1Offset = v0 * ndim;
    const p2Offset = v1 * ndim;

    // Interpolate to clipped positions, then project to display dims
    for (let outD = 0; outD < numDisplay; outD++) {
      const d = displayDims[outD];
      const p1Val = positions[p1Offset + d];
      const p2Val = positions[p2Offset + d];

      // Clipped start: p1 + t1 * (p2 - p1)
      outputStart[outIdx * 3 + outD] = p1Val + t1 * (p2Val - p1Val);
      // Clipped end: p1 + t2 * (p2 - p1)
      outputEnd[outIdx * 3 + outD] = p1Val + t2 * (p2Val - p1Val);
    }

    // Pad to 3D if fewer than 3 display dims
    for (let outD = numDisplay; outD < 3; outD++) {
      outputStart[outIdx * 3 + outD] = 0.0;
      outputEnd[outIdx * 3 + outD] = 0.0;
    }

    outIdx++;
  }

  return outIdx;
}

/**
 * Linear interpolation helper (scalar).
 */
export function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

/**
 * Linear interpolation for 3D vectors.
 * Returns interpolated vector as Float32Array [x, y, z].
 */
export function lerp_vec3(a: Float32Array, b: Float32Array, t: number): Float32Array {
  return new Float32Array([
    a[0] + t * (b[0] - a[0]),
    a[1] + t * (b[1] - a[1]),
    a[2] + t * (b[2] - a[2]),
  ]);
}

/**
 * Calculate 3D Euclidean distance.
 */
export function distance_3d(a: Float32Array, b: Float32Array): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Batch interpolate scalar attributes for visible segments.
 *
 * @param values - Per-vertex attribute values [numVertices]
 * @param segments - Segment indices [numSegments * 2]
 * @param visibility - Visibility mask [numSegments]
 * @param t1Params - Start interpolation parameters [numSegments]
 * @param t2Params - End interpolation parameters [numSegments]
 * @param numSegments - Total number of segments
 * @param outputStart - Output interpolated start values [visibleCount]
 * @param outputEnd - Output interpolated end values [visibleCount]
 * @returns Number of visible segments written
 */
export function interpolate_scalars_batch(
  values: Float32Array,
  segments: Uint32Array,
  visibility: Uint8Array,
  t1Params: Float32Array,
  t2Params: Float32Array,
  numSegments: number,
  outputStart: Float32Array,
  outputEnd: Float32Array
): number {
  let outIdx = 0;

  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    if (visibility[segIdx] === 0) {
      continue;
    }

    const v0 = segments[segIdx * 2];
    const v1 = segments[segIdx * 2 + 1];
    const t1 = t1Params[segIdx];
    const t2 = t2Params[segIdx];

    const val0 = values[v0];
    const val1 = values[v1];

    outputStart[outIdx] = val0 + t1 * (val1 - val0);
    outputEnd[outIdx] = val0 + t2 * (val1 - val0);

    outIdx++;
  }

  return outIdx;
}

/**
 * Batch interpolate RGB color attributes for visible segments.
 *
 * @param colors - Per-vertex RGB colors [numVertices * 3]
 * @param segments - Segment indices [numSegments * 2]
 * @param visibility - Visibility mask [numSegments]
 * @param t1Params - Start interpolation parameters [numSegments]
 * @param t2Params - End interpolation parameters [numSegments]
 * @param numSegments - Total number of segments
 * @param outputStart - Output interpolated start colors [visibleCount * 3]
 * @param outputEnd - Output interpolated end colors [visibleCount * 3]
 * @returns Number of visible segments written
 */
export function interpolate_colors_batch(
  colors: Float32Array,
  segments: Uint32Array,
  visibility: Uint8Array,
  t1Params: Float32Array,
  t2Params: Float32Array,
  numSegments: number,
  outputStart: Float32Array,
  outputEnd: Float32Array
): number {
  let outIdx = 0;

  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    if (visibility[segIdx] === 0) {
      continue;
    }

    const v0 = segments[segIdx * 2];
    const v1 = segments[segIdx * 2 + 1];
    const t1 = t1Params[segIdx];
    const t2 = t2Params[segIdx];

    for (let c = 0; c < 3; c++) {
      const c0 = colors[v0 * 3 + c];
      const c1 = colors[v1 * 3 + c];

      outputStart[outIdx * 3 + c] = c0 + t1 * (c1 - c0);
      outputEnd[outIdx * 3 + c] = c0 + t2 * (c1 - c0);
    }

    outIdx++;
  }

  return outIdx;
}

/**
 * Calculate 3D segment lengths for visible segments.
 *
 * @param startPositions - Clipped start positions [visibleCount * 3]
 * @param endPositions - Clipped end positions [visibleCount * 3]
 * @param visibleCount - Number of visible segments
 * @param output - Output segment lengths [visibleCount]
 */
export function calculate_segment_lengths(
  startPositions: Float32Array,
  endPositions: Float32Array,
  visibleCount: number,
  output: Float32Array
): void {
  for (let i = 0; i < visibleCount; i++) {
    const dx = endPositions[i * 3] - startPositions[i * 3];
    const dy = endPositions[i * 3 + 1] - startPositions[i * 3 + 1];
    const dz = endPositions[i * 3 + 2] - startPositions[i * 3 + 2];
    output[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
}

/**
 * Unit direction pointing from an endpoint back along its own segment.
 *
 * One unit direction per visible segment, start -> end. A degenerate
 * (zero-length or non-finite) segment gets a zero direction, whose dot product
 * is 0, which yields suppression 0 — the cap is kept, the wanted fallback.
 */
function segmentDirections(
  visibleCount: number,
  startPositions: Float32Array,
  endPositions: Float32Array
): Float32Array {
  const dirs = new Float32Array(visibleCount * 3);
  for (let i = 0; i < visibleCount; i++) {
    const o = i * 3;
    const dx = endPositions[o] - startPositions[o];
    const dy = endPositions[o + 1] - startPositions[o + 1];
    const dz = endPositions[o + 2] - startPositions[o + 2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (Number.isFinite(len) && len > 0) {
      dirs[o] = dx / len;
      dirs[o + 1] = dy / len;
      dirs[o + 2] = dz / len;
    }
  }
  return dirs;
}

/**
 * Per-endpoint cap suppression in [0, 1] (drives the shader cap factor).
 *
 * TypeScript reference for the Rust `compute_cap_suppression` kernel — see
 * `wasm/rust/src/lines_clipping.rs` for the full rationale. In short: the line
 * fragment shader dims each segment towards `0.5` at its own endpoints, which
 * is only correct where a neighbouring quad overlaps and adds the missing half
 * back. Collinear neighbours tile rather than overlap, so this returns `1.0`
 * (suppress the dimming) for clipped endpoints and straight-through interior
 * joints, `0.0` (keep it) for free ends, branch points, and sharp bends, and
 * `clamp(-dot(awayA, awayB), 0, 1)` in between.
 *
 * Only degree-2 vertices count as joints, and only endpoints that actually
 * reach the shared vertex (untrimmed, on a visible segment) participate.
 *
 * @param segments - Vertex index pairs [numSegments * 2]
 * @param visibility - Visibility mask [numSegments]
 * @param t1Params - Start interpolation parameters [numSegments]
 * @param t2Params - End interpolation parameters [numSegments]
 * @param numSegments - Total number of segments
 * @param numVertices - Total number of source vertices
 * @param startPositions - Clipped start positions [visibleCount * 3]
 * @param endPositions - Clipped end positions [visibleCount * 3]
 * @param outputStart - Output start suppression [visibleCount]
 * @param outputEnd - Output end suppression [visibleCount]
 * @returns Number of visible segments written
 */
export function compute_cap_suppression(
  segments: Uint32Array,
  visibility: Uint8Array,
  t1Params: Float32Array,
  t2Params: Float32Array,
  numSegments: number,
  numVertices: number,
  startPositions: Float32Array,
  endPositions: Float32Array,
  outputStart: Float32Array,
  outputEnd: Float32Array
): number {
  // Endpoint code: (outIdx << 1) | endBit. `codeSum` accumulates the codes of
  // the endpoints landing exactly on each vertex and `degree` counts them
  // (saturating at 3, so branch points stay distinguishable from ordinary
  // joints). At degree 2 the partner is simply `codeSum - myCode` — one
  // scattered array instead of two, halving this pass's cache traffic. Both are
  // left zero-initialised: degree gates every read, so a sentinel fill would be
  // pure cost. Int32 holds the sums with room to spare: a code is at most 2x the visible
  // segment count, so two of them stay under 2^31 for any scene that fits in
  // memory.
  const codeSum = new Int32Array(numVertices);
  const degree = new Uint8Array(numVertices);

  const registerTouch = (vertex: number, code: number): void => {
    if (vertex >= numVertices) return; // upstream validation rejects these
    const d = degree[vertex];
    if (d < 2) {
      codeSum[vertex] += code;
      degree[vertex] = d + 1;
    } else {
      degree[vertex] = 3; // branch point — the sum is no longer meaningful
    }
  };

  let outIdx = 0;
  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    if (visibility[segIdx] === 0) continue;
    const code = outIdx << 1;
    if (t1Params[segIdx] <= 0) registerTouch(segments[segIdx * 2], code);
    if (t2Params[segIdx] >= 1) registerTouch(segments[segIdx * 2 + 1], code | 1);
    outIdx++;
  }

  const visibleCount = outIdx;

  // Normalise ONCE per segment; the joint test is then a single dot product.
  // The "away" vector at an endpoint is +dir at a start and -dir at an end, so
  //   dot(awayMine, awayPartner) = sMine * sPartner * dot(dirMine, dirPartner)
  // and sMine * sPartner is +1 exactly when the two endpoint bits agree.
  const dirs = segmentDirections(visibleCount, startPositions, endPositions);

  const jointSuppression = (vertex: number, myCode: number): number => {
    if (vertex >= numVertices || degree[vertex] !== 2) return 0;
    const partner = codeSum[vertex] - myCode;
    if (partner === myCode) return 0; // self-segment registered both its ends here
    const mo = (myCode >> 1) * 3;
    const po = (partner >> 1) * 3;
    if (mo + 2 >= dirs.length || po + 2 >= dirs.length) return 0;
    const dot = dirs[mo] * dirs[po] + dirs[mo + 1] * dirs[po + 1] + dirs[mo + 2] * dirs[po + 2];
    const sign = (myCode & 1) === (partner & 1) ? 1 : -1;
    return Math.min(Math.max(-(sign * dot), 0), 1);
  };

  outIdx = 0;
  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    if (visibility[segIdx] === 0) continue;
    const code = outIdx << 1;
    outputStart[outIdx] = t1Params[segIdx] > 0 ? 1 : jointSuppression(segments[segIdx * 2], code);
    outputEnd[outIdx] =
      t2Params[segIdx] < 1 ? 1 : jointSuppression(segments[segIdx * 2 + 1], code | 1);
    outIdx++;
  }

  return outIdx;
}
