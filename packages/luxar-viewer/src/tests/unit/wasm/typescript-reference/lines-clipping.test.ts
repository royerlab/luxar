/**
 * Tests for `src/wasm/typescript/lines_clipping.ts`
 * (nD segment clipping: clip_segment_single, clip_segments_batch,
 * interpolate_clipped_positions, lerp / lerp_vec3 / distance_3d primitives,
 * interpolate_scalars_batch, interpolate_colors_batch,
 * calculate_segment_lengths, compute_cap_suppression).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D9 — final phase): the 1714-line
 * mega-file has been fully split into per-source-module test files
 * mirroring `src/wasm/typescript/`.
 */

import { describe, it, expect } from 'vitest';
import {
  clip_segment_single,
  clip_segments_batch,
  interpolate_clipped_positions,
  lerp,
  lerp_vec3,
  distance_3d,
  interpolate_scalars_batch,
  interpolate_colors_batch,
  calculate_segment_lengths,
  compute_cap_suppression,
} from '../../../../wasm/typescript';

// ============================================================================
// LINES CLIPPING TESTS (nD segment clipping and interpolation)
// ============================================================================

describe('lines_clipping: clip_segment_single', () => {
  it('should return visible with t1=0, t2=1 when both endpoints in slice', () => {
    // 4D segment, both endpoints in slice
    const p1 = new Float32Array([0, 0, 0, 5.2]);
    const p2 = new Float32Array([10, 10, 10, 4.8]);
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);

    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);

    expect(result[0]).toBe(1.0); // visible
    expect(result[1]).toBeCloseTo(0.0, 5); // t1
    expect(result[2]).toBeCloseTo(1.0, 5); // t2
  });

  it('should return invisible when both endpoints below slice', () => {
    const p1 = new Float32Array([0, 0, 0, 0]);
    const p2 = new Float32Array([10, 10, 10, 2]);
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);

    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);

    expect(result[0]).toBe(0.0); // not visible
  });

  it('should clip segment crossing slice correctly', () => {
    const p1 = new Float32Array([0, 0, 0, 0]);
    const p2 = new Float32Array([10, 10, 10, 10]);
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);

    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);

    expect(result[0]).toBe(1.0); // visible
    expect(result[1]).toBeCloseTo(0.45, 5); // t1 = (4.5 - 0) / 10
    expect(result[2]).toBeCloseTo(0.55, 5); // t2 = (5.5 - 0) / 10
  });

  it('MED-20: reusing a pre-allocated workspace buffer yields identical results across calls', () => {
    // Hot loops call clip_segment_single per segment; passing a single
    // workspace Uint8Array avoids per-call allocation while producing
    // the same result as the allocating overload. The function must
    // also zero stale bytes between calls (a previous call's display
    // dims may differ from the current call's).
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const workspace = new Uint8Array(4);

    // First call: displayDims=[0,1,2], one dim clipped
    const p1a = new Float32Array([0, 0, 0, 0]);
    const p2a = new Float32Array([10, 10, 10, 10]);
    const ddA = new Uint32Array([0, 1, 2]);
    const refA = clip_segment_single(p1a, p2a, slicePos, tolerance, ddA, 4);
    const wsA = clip_segment_single(p1a, p2a, slicePos, tolerance, ddA, 4, workspace);
    expect(Array.from(wsA)).toEqual(Array.from(refA));

    // Second call with DIFFERENT displayDims — the workspace must be
    // re-zeroed inside the function, otherwise dim 0 would still be
    // marked as a display dim and the clip math would diverge.
    const p1b = new Float32Array([0, 0, 5.2, 0]);
    const p2b = new Float32Array([10, 10, 4.8, 10]);
    const slicePosB = new Float32Array([0, 0, 5, 0]);
    const toleranceB = new Float32Array([0.5, 1e10, 1e10, 1e10]);
    const ddB = new Uint32Array([1, 2, 3]);
    const refB = clip_segment_single(p1b, p2b, slicePosB, toleranceB, ddB, 4);
    const wsB = clip_segment_single(p1b, p2b, slicePosB, toleranceB, ddB, 4, workspace);
    expect(Array.from(wsB)).toEqual(Array.from(refB));
  });
});

describe('lines_clipping: clip_segments_batch', () => {
  it('should batch clip multiple segments', () => {
    // Two segments: one fully visible, one crossing slice
    const positions = new Float32Array([
      0,
      0,
      0,
      5, // v0: in slice
      10,
      10,
      10,
      5, // v1: in slice
      20,
      20,
      20,
      0, // v2: out of slice
    ]);
    const segments = new Uint32Array([0, 1, 1, 2]); // seg0: v0-v1, seg1: v1-v2
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);

    const visibility = new Uint8Array(2);
    const t1 = new Float32Array(2);
    const t2 = new Float32Array(2);

    const count = clip_segments_batch(
      positions,
      segments,
      slicePos,
      tolerance,
      displayDims,
      4,
      2,
      visibility,
      t1,
      t2
    );

    expect(count).toBe(2);
    // seg0: both in slice
    expect(visibility[0]).toBe(1);
    expect(t1[0]).toBeCloseTo(0.0, 5);
    expect(t2[0]).toBeCloseTo(1.0, 5);

    // seg1: crosses slice (v1 at 5.0, v2 at 0.0)
    expect(visibility[1]).toBe(1);
    expect(t1[1]).toBeCloseTo(0.0, 5); // v1 is inside
    expect(t2[1]).toBeCloseTo(0.1, 5); // t where dim3 crosses 4.5: 5 + t*(0-5) = 4.5 -> t = 0.1
  });
});

describe('lines_clipping: interpolate_clipped_positions', () => {
  it('should interpolate clipped positions to 3D', () => {
    const positions = new Float32Array([
      0,
      0,
      0,
      0, // v0
      10,
      20,
      30,
      10, // v1
    ]);
    const segments = new Uint32Array([0, 1]);
    const visibility = new Uint8Array([1]);
    const t1Params = new Float32Array([0.25]);
    const t2Params = new Float32Array([0.75]);
    const displayDims = new Uint32Array([0, 1, 2]);

    const outputStart = new Float32Array(3);
    const outputEnd = new Float32Array(3);

    const count = interpolate_clipped_positions(
      positions,
      segments,
      visibility,
      t1Params,
      t2Params,
      displayDims,
      4,
      1,
      outputStart,
      outputEnd
    );

    expect(count).toBe(1);
    // Start: p1 + 0.25 * (p2 - p1) = [0,0,0] + 0.25 * [10,20,30] = [2.5, 5, 7.5]
    expect(outputStart[0]).toBeCloseTo(2.5, 5);
    expect(outputStart[1]).toBeCloseTo(5.0, 5);
    expect(outputStart[2]).toBeCloseTo(7.5, 5);
    // End: p1 + 0.75 * (p2 - p1) = [0,0,0] + 0.75 * [10,20,30] = [7.5, 15, 22.5]
    expect(outputEnd[0]).toBeCloseTo(7.5, 5);
    expect(outputEnd[1]).toBeCloseTo(15.0, 5);
    expect(outputEnd[2]).toBeCloseTo(22.5, 5);
  });
});

describe('lines_clipping: lerp', () => {
  // [wasm.md/O5][P4] Was four anonymous expect() lines in a single it().
  // Each case has independent observable identity (endpoints + midpoint +
  // symmetric range), and an it.each surfaces the failing case.
  it.each([
    { a: 0, b: 10, t: 0, expected: 0 },
    { a: 0, b: 10, t: 1, expected: 10 },
    { a: 0, b: 10, t: 0.5, expected: 5 },
    { a: -10, b: 10, t: 0.5, expected: 0 },
  ])('lerp($a, $b, $t) === $expected', ({ a, b, t, expected }) => {
    expect(lerp(a, b, t)).toBe(expected);
  });
});

describe('lines_clipping: lerp_vec3', () => {
  it('should interpolate 3D vectors', () => {
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([10, 20, 30]);

    const result = lerp_vec3(a, b, 0.5);

    expect(result[0]).toBe(5);
    expect(result[1]).toBe(10);
    expect(result[2]).toBe(15);
  });
});

describe('lines_clipping: distance_3d', () => {
  // wasm.md O7[P4]: parametrize the previously-bundled two distance_3d
  // cases via it.each so a single failure surfaces by label rather than
  // both being lumped under "should calculate Euclidean distance".
  it.each([
    { label: '3-4-5 triangle (exact integer root)', a: [0, 0, 0], b: [3, 4, 0], expected: 5 },
    { label: 'unit diagonal sqrt(3)', a: [0, 0, 0], b: [1, 1, 1], expected: Math.sqrt(3) },
  ])('Euclidean distance: $label', ({ a, b, expected }) => {
    const va = new Float32Array(a);
    const vb = new Float32Array(b);
    expect(distance_3d(va, vb)).toBeCloseTo(expected, 5);
  });
});

describe('lines_clipping: interpolate_scalars_batch', () => {
  it('should interpolate scalar attributes', () => {
    const values = new Float32Array([0.1, 0.3, 0.5]);
    const segments = new Uint32Array([0, 1, 1, 2]);
    const visibility = new Uint8Array([1, 1]);
    const t1Params = new Float32Array([0.0, 0.5]);
    const t2Params = new Float32Array([1.0, 1.0]);

    const outputStart = new Float32Array(2);
    const outputEnd = new Float32Array(2);

    const count = interpolate_scalars_batch(
      values,
      segments,
      visibility,
      t1Params,
      t2Params,
      2,
      outputStart,
      outputEnd
    );

    expect(count).toBe(2);
    // seg0: t1=0 -> val0=0.1, t2=1 -> val0 + 1.0*(val1-val0) = 0.3
    expect(outputStart[0]).toBeCloseTo(0.1, 5);
    expect(outputEnd[0]).toBeCloseTo(0.3, 5);
    // seg1: t1=0.5 -> val1 + 0.5*(val2-val1) = 0.3 + 0.1 = 0.4
    expect(outputStart[1]).toBeCloseTo(0.4, 5);
  });
});

describe('lines_clipping: interpolate_colors_batch', () => {
  it('should interpolate RGB colors', () => {
    const colors = new Float32Array([
      1,
      0,
      0, // Red
      0,
      1,
      0, // Green
    ]);
    const segments = new Uint32Array([0, 1]);
    const visibility = new Uint8Array([1]);
    const t1Params = new Float32Array([0.0]);
    const t2Params = new Float32Array([0.5]);

    const outputStart = new Float32Array(3);
    const outputEnd = new Float32Array(3);

    const count = interpolate_colors_batch(
      colors,
      segments,
      visibility,
      t1Params,
      t2Params,
      1,
      outputStart,
      outputEnd
    );

    expect(count).toBe(1);
    // Start: t1=0 -> red (1,0,0)
    expect(outputStart[0]).toBeCloseTo(1.0, 5);
    expect(outputStart[1]).toBeCloseTo(0.0, 5);
    expect(outputStart[2]).toBeCloseTo(0.0, 5);
    // End: t2=0.5 -> midway (0.5, 0.5, 0)
    expect(outputEnd[0]).toBeCloseTo(0.5, 5);
    expect(outputEnd[1]).toBeCloseTo(0.5, 5);
    expect(outputEnd[2]).toBeCloseTo(0.0, 5);
  });
});

describe('lines_clipping: calculate_segment_lengths', () => {
  it('should calculate 3D segment lengths', () => {
    const startPos = new Float32Array([0, 0, 0, 0, 0, 0]);
    const endPos = new Float32Array([3, 4, 0, 1, 1, 1]);
    const output = new Float32Array(2);

    calculate_segment_lengths(startPos, endPos, 2, output);

    expect(output[0]).toBe(5); // 3-4-5 triangle
    expect(output[1]).toBeCloseTo(Math.sqrt(3), 5);
  });

  it('returns the finite true length on huge coordinates (no f32 squared-length overflow)', () => {
    // Component delta 2e30 squares to 4e60 — far past f32::MAX (~3.4e38).
    // The reference reads f32 inputs but accumulates in f64 (the contract
    // the Rust kernel must match — see the WASM parity twin, #793).
    const startPos = new Float32Array([-1e30, 0, 0]);
    const endPos = new Float32Array([1e30, 0, 0]);
    const output = new Float32Array(1);

    calculate_segment_lengths(startPos, endPos, 1, output);

    expect(Number.isFinite(output[0])).toBe(true);
    expect(output[0]).toBe(Math.fround(endPos[0] - startPos[0]));
  });
});

describe('lines_clipping: compute_cap_suppression', () => {
  it('should mark clipped endpoints correctly', () => {
    const visibility = new Uint8Array([1, 1, 0, 1]);
    const t1Params = new Float32Array([0.0, 0.5, 0.0, 0.25]);
    const t2Params = new Float32Array([1.0, 0.75, 1.0, 1.0]);
    // Disjoint segments (vertices 0..7) so nothing forms a joint — this test
    // pins the clipped-flag half of the contract in isolation.
    const segments = Uint32Array.from({ length: 8 }, (_, i) => i);
    const startPositions = new Float32Array([0, 0, 0, 10, 0, 0, 20, 0, 0]);
    const endPositions = new Float32Array([1, 0, 0, 11, 0, 0, 21, 0, 0]);

    const startCapSuppression = new Float32Array(3);
    const endCapSuppression = new Float32Array(3);

    const count = compute_cap_suppression(
      segments,
      visibility,
      t1Params,
      t2Params,
      4,
      8,
      startPositions,
      endPositions,
      startCapSuppression,
      endCapSuppression
    );

    expect(count).toBe(3); // 3 visible
    // seg0: t1=0 (not clipped), t2=1 (not clipped), no neighbour → free ends
    expect(startCapSuppression[0]).toBe(0);
    expect(endCapSuppression[0]).toBe(0);
    // seg1: t1=0.5 (clipped), t2=0.75 (clipped)
    expect(startCapSuppression[1]).toBe(1);
    expect(endCapSuppression[1]).toBe(1);
    // seg3: t1=0.25 (clipped), t2=1.0 (not clipped, free end)
    expect(startCapSuppression[2]).toBe(1);
    expect(endCapSuppression[2]).toBe(0);
  });

  it('suppresses the cap at a straight-through interior joint', () => {
    // Two collinear segments sharing vertex 1: v0 --> v1 --> v2 along +x.
    const segments = new Uint32Array([0, 1, 1, 2]);
    const visibility = new Uint8Array([1, 1]);
    const t1Params = new Float32Array([0, 0]);
    const t2Params = new Float32Array([1, 1]);
    const startPositions = new Float32Array([0, 0, 0, 1, 0, 0]);
    const endPositions = new Float32Array([1, 0, 0, 2, 0, 0]);
    const outStart = new Float32Array(2);
    const outEnd = new Float32Array(2);

    compute_cap_suppression(
      segments,
      visibility,
      t1Params,
      t2Params,
      2,
      3,
      startPositions,
      endPositions,
      outStart,
      outEnd
    );

    // Free outer ends keep the cap; the shared joint is fully suppressed.
    expect(outStart[0]).toBe(0);
    expect(outEnd[0]).toBeCloseTo(1, 6);
    expect(outStart[1]).toBeCloseTo(1, 6);
    expect(outEnd[1]).toBe(0);
  });

  it('keeps the cap at a 90-degree bend and interpolates in between', () => {
    // v0 --> v1 along +x, then v1 --> v2 along +y (a right-angle turn).
    const right = new Float32Array(2);
    compute_cap_suppression(
      new Uint32Array([0, 1, 1, 2]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      3,
      new Float32Array([0, 0, 0, 1, 0, 0]),
      new Float32Array([1, 0, 0, 1, 1, 0]),
      right,
      new Float32Array(2)
    );
    expect(right[1]).toBe(0); // 90 degrees → dot 0 → cap preserved

    // A gentle 45-degree turn lands between the two regimes.
    const gentle = new Float32Array(2);
    const d = Math.SQRT1_2;
    compute_cap_suppression(
      new Uint32Array([0, 1, 1, 2]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      3,
      new Float32Array([0, 0, 0, 1, 0, 0]),
      new Float32Array([1, 0, 0, 1 + d, d, 0]),
      gentle,
      new Float32Array(2)
    );
    expect(gentle[1]).toBeCloseTo(Math.SQRT1_2, 5);
  });

  it('clamps a 180-degree fold-back joint to 0 (the only case with a negative raw value)', () => {
    // v0 -> v1 travelling +x, then v1 -> v2 travelling BACK along -x. The two
    // "away" vectors coincide, so dot(dirA, dirB) = -1 and the endpoint bits
    // differ (sign = -1), making the raw value -1. Every other geometry keeps
    // it in [0, 1], so this is the sole case that exercises the LOWER clamp —
    // without it a fold-back would emit a negative suppression, which the
    // shader's per-endpoint mix(0.5 + 0.5 * ramp, 1.0, s) would turn into a
    // cap BELOW 0.5 (a darker-than-intended notch) instead of the full cap
    // the overlap needs.
    const foldStart = new Float32Array(2);
    const foldEnd = new Float32Array(2);
    compute_cap_suppression(
      new Uint32Array([0, 1, 1, 2]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      3,
      new Float32Array([0, 0, 0, 1, 0, 0]),
      new Float32Array([1, 0, 0, 0, 0, 0]),
      foldStart,
      foldEnd
    );
    expect(foldEnd[0]).toBe(0);
    expect(foldStart[1]).toBe(0);
  });

  it('keeps the cap at a branch point (three segments meeting)', () => {
    // Three segments all starting at vertex 0 — a star hub. Suppressing here
    // would stack three quads into a bright nub.
    const outStart = new Float32Array(3);
    compute_cap_suppression(
      new Uint32Array([0, 1, 0, 2, 0, 3]),
      new Uint8Array([1, 1, 1]),
      new Float32Array([0, 0, 0]),
      new Float32Array([1, 1, 1]),
      3,
      4,
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
      new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      outStart,
      new Float32Array(3)
    );
    expect(Array.from(outStart)).toEqual([0, 0, 0]);
  });

  it('does not treat a culled or trimmed neighbour as a joint', () => {
    // v0-v1-v2 collinear, but the second segment is invisible: v1 is a real
    // visible free end and must keep its cap.
    const culled = new Float32Array(2);
    compute_cap_suppression(
      new Uint32Array([0, 1, 1, 2]),
      new Uint8Array([1, 0]),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      3,
      new Float32Array([0, 0, 0, 1, 0, 0]),
      new Float32Array([1, 0, 0, 2, 0, 0]),
      new Float32Array(2),
      culled
    );
    expect(culled[0]).toBe(0);

    // Same geometry, but the neighbour is visible and trimmed away from the
    // shared vertex (t1 > 0) — it no longer reaches v1, so still no joint.
    const trimmed = new Float32Array(2);
    compute_cap_suppression(
      new Uint32Array([0, 1, 1, 2]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0.4]),
      new Float32Array([1, 1]),
      2,
      3,
      new Float32Array([0, 0, 0, 1.4, 0, 0]),
      new Float32Array([1, 0, 0, 2, 0, 0]),
      new Float32Array(2),
      trimmed
    );
    expect(trimmed[0]).toBe(0);
  });

  it('handles a shared vertex reached from both segments by the same endpoint', () => {
    // Both segments END at vertex 1 (v0 -> v1 <- v2): the polyline is stored
    // with opposing orientation. Geometrically this is still a straight
    // continuation, so the "away" vectors are opposite and it suppresses.
    const outEnd = new Float32Array(2);
    compute_cap_suppression(
      new Uint32Array([0, 1, 2, 1]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      3,
      new Float32Array([0, 0, 0, 2, 0, 0]),
      new Float32Array([1, 0, 0, 1, 0, 0]),
      new Float32Array(2),
      outEnd
    );
    expect(outEnd[0]).toBeCloseTo(1, 6);
    expect(outEnd[1]).toBeCloseTo(1, 6);
  });

  it('uses compacted endpoint codes when visibility is non-contiguous', () => {
    // seg0 and seg2 survive and share v1; the disjoint middle segment is
    // culled. The position arrays are compacted to the two survivors, so a
    // regression that keys direction-table codes on source segIdx would read
    // the wrong row and lose this straight-through joint.
    const outStart = new Float32Array(2);
    const outEnd = new Float32Array(2);
    const count = compute_cap_suppression(
      new Uint32Array([0, 1, 3, 4, 1, 2]),
      new Uint8Array([1, 0, 1]),
      new Float32Array([0, 0, 0]),
      new Float32Array([1, 1, 1]),
      3,
      5,
      new Float32Array([0, 0, 0, 1, 0, 0]),
      new Float32Array([1, 0, 0, 2, 0, 0]),
      outStart,
      outEnd
    );

    expect(count).toBe(2);
    expect(outStart[0]).toBe(0);
    expect(outEnd[0]).toBeCloseTo(1, 6);
    expect(outStart[1]).toBeCloseTo(1, 6);
    expect(outEnd[1]).toBe(0);
  });

  it('keeps malformed unregistered endpoint codes finite', () => {
    // The first two starts register degree 2 at v0. The third start has a
    // NaN t1, so it does NOT register but later queries the same vertex with
    // myCode=4. codeSum=0+2 then yields partner=-2. JavaScript negative array
    // indexing returns undefined, which used to poison the dot product and
    // emit NaN; Rust rejects the out-of-range partner and returns 0.
    const outStart = new Float32Array(3);
    compute_cap_suppression(
      new Uint32Array([0, 1, 0, 2, 0, 3]),
      new Uint8Array([1, 1, 1]),
      new Float32Array([0, 0, Number.NaN]),
      new Float32Array([1, 1, 1]),
      3,
      4,
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
      new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      outStart,
      new Float32Array(3)
    );

    expect(outStart[2]).toBe(0);
    expect(Number.isFinite(outStart[2])).toBe(true);
  });

  it('keeps the cap on a degenerate zero-length neighbour', () => {
    const outEnd = new Float32Array(2);
    compute_cap_suppression(
      new Uint32Array([0, 1, 1, 2]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      3,
      new Float32Array([0, 0, 0, 1, 0, 0]),
      new Float32Array([1, 0, 0, 1, 0, 0]), // second segment has zero length
      new Float32Array(2),
      outEnd
    );
    expect(outEnd[0]).toBe(0);
  });
});
