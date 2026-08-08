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
 * `f32::max` / `f32::min`, which IGNORE a NaN operand and return the other —
 * unlike `Math.max` / `Math.min`, which propagate it.
 *
 * The t-param accumulation below must match the Rust kernel exactly, and this
 * mirror is not a fallback: it is the PRODUCTION backend above 16 dimensions,
 * so a NaN t-param is producible here and not on WASM (a NaN tolerance or
 * slice position makes both in-slab tests fail and the same-side rejection
 * miss, so `tMin`/`tMax` come out NaN). Propagating it would leave a segment
 * with NaN clip parameters that the two backends then disagree about.
 */
const maxIgnoringNaN = (a: number, b: number): number => (b > a ? b : a);
const minIgnoringNaN = (a: number, b: number): number => (b < a ? b : a);

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

    // #806: a non-finite (NaN or ±Inf) coordinate on a slicing (non-displayed)
    // dimension cannot be localized against the slice, so the segment is
    // treated as invisible. Enforced here, identically in the Rust backend
    // (`lines_clipping.rs`), so the two backends stay in parity — without this
    // the comparisons below are all false for NaN, the "both out, same side"
    // check falls through, and NaN t-params leak out as a "visible" result.
    if (!Number.isFinite(v1) || !Number.isFinite(v2)) {
      return new Float32Array([0.0, 0.0, 0.0]); // [visible=0, t1, t2]
    }

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
      t1 = maxIgnoringNaN(t1, tMin);
      t2 = minIgnoringNaN(t2, tMax);
    } else {
      t1 = maxIgnoringNaN(t1, tMax);
      t2 = minIgnoringNaN(t2, tMin);
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

      // #806: a non-finite (NaN or ±Inf) coordinate on a slicing (non-displayed)
      // dimension cannot be localized against the slice, so the segment is
      // treated as invisible. Enforced here, identically in the Rust backend
      // (`lines_clipping.rs`), so the two backends stay in parity.
      if (!Number.isFinite(v1Val) || !Number.isFinite(v2Val)) {
        visible = false;
        break;
      }

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
        t1 = maxIgnoringNaN(t1, tMin);
        t2 = minIgnoringNaN(t2, tMax);
      } else {
        t1 = maxIgnoringNaN(t1, tMax);
        t2 = minIgnoringNaN(t2, tMin);
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

/** Free polyline end: keep the soft cap. */
export const JOINT_FREE_END = 0;

/**
 * Largest partner slot a joint code can name and still survive its own storage.
 * Mirrors the Rust `MAX_EXACT_JOINT_SLOT` — see that constant for why the bound
 * is `slot + 3 <= 2^24` and why it must be enforced rather than assumed (a
 * 32768-class device reaches a 22.35M per-node line capacity, where a measured
 * 12.5% of codes above the bound mis-decode). Both sides of a pair are tested
 * against it, so a joint straddling the bound degrades on BOTH endpoints and
 * never leaves one of them mitring alone.
 */
export const MAX_EXACT_JOINT_SLOT = (1 << 24) - 3;
/** Slice-clipped endpoint: no neighbour will arrive, so suppress the cap. */
export const JOINT_CLIPPED = -1;
/** Degree->=3 hub: several quads already stack here, so keep the cap. */
export const JOINT_HUB = -2;

/**
 * Joint code for one unclipped endpoint sitting on `vertex`.
 *
 * Mirrors the Rust `joint_code`; exposed (that twin is private) because the
 * `MAX_EXACT_JOINT_SLOT` boundary is only reachable through a >16.7M-segment
 * scene, which no unit test can build.
 *
 * @param vertex - Source vertex index the endpoint lands on
 * @param myCode - This endpoint's own `(slot << 1) | endBit` code
 * @param numVertices - Length of the touch tables
 * @param visibleCount - Number of visible segments (bounds a named slot)
 * @param codeSum - Per-vertex sum of the codes registered on it
 * @param degree - Per-vertex count of registered endpoints, saturating at 3
 * @returns The joint code, or a sentinel when no partner can be named
 */
export function jointCodeForEndpoint(
  vertex: number,
  myCode: number,
  numVertices: number,
  visibleCount: number,
  codeSum: Int32Array,
  degree: Uint8Array
): number {
  if (vertex >= numVertices) return JOINT_FREE_END; // unregistered
  const d = degree[vertex];
  if (d < 2) return JOINT_FREE_END;
  if (d > 2) return JOINT_HUB;
  const partner = codeSum[vertex] - myCode;
  const slot = partner >> 1;
  // Three ways the difference can fail to name a real partner:
  //
  // - out of range: the sum did not contain `myCode`, so the difference is
  //   arbitrary. Both passes now run the IDENTICAL `t <= 0` / `t >= 1` tests,
  //   so an endpoint can no longer read tables it never registered in — but
  //   the predecessor kernel bounded the same arithmetic against its
  //   direction table's length, and keeping an equivalent bound means a code
  //   can never name a slot outside the stream it indexes, independent of the
  //   texel writer's capacity clamp.
  // - `slot === my slot`: a zero-length or looping segment registered BOTH of
  //   its own endpoints here, so the difference is its own other endpoint (the
  //   two codes differ only in the end bit, which is why comparing whole codes
  //   is not enough). The angle-only scalar this replaced survived the case by
  //   returning a plausible number; a code gets dereferenced, and a segment
  //   mitered against itself is the asymmetric-join case that produces flaps.
  // - either slot past `MAX_EXACT_JOINT_SLOT`: the outputs are Float32Arrays,
  //   so a code past 2^24 rounds AT THE STORE — and it rounds to a valid,
  //   in-range slot that no downstream consumer can tell from a deliberate
  //   one. The texel writer cannot help; it reads the already-rounded value.
  //   BOTH slots are tested, not just the partner's: the pair degrades
  //   together only if each side asks the same question, and my own slot is
  //   what the partner's code has to name.
  if (
    slot < 0 ||
    slot >= visibleCount ||
    slot > MAX_EXACT_JOINT_SLOT ||
    myCode >> 1 > MAX_EXACT_JOINT_SLOT ||
    slot === myCode >> 1
  )
    return JOINT_FREE_END;
  // endBit 0 = the partner's START touches this vertex, 1 = its END does.
  return (partner & 1) === 0 ? slot + 1 : -(slot + 3);
}

/**
 * Per-endpoint **joint code** (drives the shader's join geometry and cap).
 *
 * TypeScript reference for the Rust `compute_joint_codes` kernel — see the Rust
 * doc comment in `wasm/rust/src/lines_clipping.rs` for the full rationale. This
 * mirror is not just a WASM-missing fallback: it is the production backend for
 * `ndim > 16`, which the fixed-size Rust kernels cannot serve, so the two must
 * agree exactly.
 *
 * Agreement is trivial here in a way it was not for the scalar this replaced:
 * the output is integer index arithmetic, so there is no f32/f64 accumulation
 * order to reconcile between the backends.
 *
 * | value         | meaning                                                    |
 * |---------------|------------------------------------------------------------|
 * | ` 0`          | free polyline end — keep the soft cap                       |
 * | `-1`          | slice-clipped — suppress the cap entirely                   |
 * | `-2`          | degree->=3 hub — keep the cap                               |
 * | `+(slot + 1)` | joins visible segment `slot`, at that segment's START       |
 * | `-(slot + 3)` | joins visible segment `slot`, at that segment's END         |
 *
 * `slot` is a storage slot in the line texture, so it survives the depth-sort
 * worker's draw-order permutation without adjustment.
 *
 * @param segments - Vertex index pairs [numSegments * 2]
 * @param visibility - Visibility mask [numSegments]
 * @param t1Params - Start interpolation parameters [numSegments]
 * @param t2Params - End interpolation parameters [numSegments]
 * @param numSegments - Total number of segments
 * @param numVertices - Total number of source vertices (bounds the touch tables)
 * @param outputStart - Output start joint codes [visibleCount]
 * @param outputEnd - Output end joint codes [visibleCount]
 * @returns Number of visible segments written
 */
export function compute_joint_codes(
  segments: Uint32Array,
  visibility: Uint8Array,
  t1Params: Float32Array,
  t2Params: Float32Array,
  numSegments: number,
  numVertices: number,
  outputStart: Float32Array,
  outputEnd: Float32Array
): number {
  // Endpoint code: (outIdx << 1) | endBit. `codeSum` accumulates the codes of
  // the endpoints landing exactly on each vertex and `degree` counts them
  // (saturating at 3, so hubs stay distinguishable from ordinary joints). At
  // degree 2 the partner is simply `codeSum - myCode` — one scattered array
  // instead of two, halving this pass's cache traffic. Both are left
  // zero-initialised: degree gates every read, so a sentinel fill would be pure
  // cost. Int32 holds the sums with room to spare: a code is at most 2x the
  // visible segment count, so two of them stay under 2^31 for any scene that
  // fits in memory.
  const codeSum = new Int32Array(numVertices);
  const degree = new Uint8Array(numVertices);

  const registerTouch = (vertex: number, code: number): void => {
    if (vertex >= numVertices) return; // upstream validation rejects these
    const d = degree[vertex];
    if (d < 2) {
      codeSum[vertex] += code;
      degree[vertex] = d + 1;
    } else {
      degree[vertex] = 3; // hub — the sum is no longer meaningful
    }
  };

  let outIdx = 0;
  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    if (visibility[segIdx] === 0) continue;
    const code = outIdx << 1;
    if (t1Params[segIdx] <= 0.0) registerTouch(segments[segIdx * 2], code);
    if (t2Params[segIdx] >= 1.0) registerTouch(segments[segIdx * 2 + 1], code | 1);
    outIdx++;
  }

  const visibleCount = outIdx;

  outIdx = 0;
  for (let segIdx = 0; segIdx < numSegments; segIdx++) {
    if (visibility[segIdx] === 0) continue;
    const code = outIdx << 1;
    // The SAME predicates the registering pass used, not their complements.
    // `t <= 0` and `!(t > 0)` agree for every ordinary float but BOTH go false
    // for NaN, so the complementary spelling let an endpoint that never
    // registered still read the shared vertex and name a real but unrelated
    // partner slot. Repeating the predicate makes that desync structurally
    // impossible: an endpoint reads the tables only if it put its own code
    // into them.
    outputStart[outIdx] =
      t1Params[segIdx] <= 0.0
        ? jointCodeForEndpoint(
            segments[segIdx * 2],
            code,
            numVertices,
            visibleCount,
            codeSum,
            degree
          )
        : JOINT_CLIPPED;
    outputEnd[outIdx] =
      t2Params[segIdx] >= 1.0
        ? jointCodeForEndpoint(
            segments[segIdx * 2 + 1],
            code | 1,
            numVertices,
            visibleCount,
            codeSum,
            degree
          )
        : JOINT_CLIPPED;
    outIdx++;
  }

  return outIdx;
}
