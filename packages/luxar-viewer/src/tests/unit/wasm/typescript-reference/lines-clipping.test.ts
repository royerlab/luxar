/**
 * Tests for `src/wasm/typescript/lines_clipping.ts`
 * (nD segment clipping: clip_segment_single, clip_segments_batch,
 * interpolate_clipped_positions, interpolate_scalars_batch,
 * interpolate_colors_batch, calculate_segment_lengths,
 * compute_joint_codes).
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
  interpolate_scalars_batch,
  interpolate_colors_batch,
  calculate_segment_lengths,
  compute_joint_codes,
  JOINT_CLIPPED,
  JOINT_FREE_END,
  JOINT_HUB,
  MAX_EXACT_JOINT_SLOT,
  jointCodeForEndpoint,
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

  it('#806: returns invisible when NaN is on the FIRST endpoint of a non-displayed dim', () => {
    // Both endpoints coincide/inside on the displayed dims [0,1,2]; the FIRST
    // endpoint (p1) carries a NaN on the slicing (non-displayed) dim 3. Such a
    // segment cannot be localized against the slice, so it must be invisible
    // — NOT a "visible" result with NaN interpolation params.
    //
    // Pins the `!Number.isFinite(v1)` guard branch: without it the old code
    // returns [1, NaN, NaN] (dv=5-NaN=NaN is not < epsilon, so no `continue`;
    // NaN t-params never trip the `t1 >= t2` gate), so this assertion is
    // genuinely red on the pre-#806 kernel.
    const p1 = new Float32Array([1, 2, 3, NaN]);
    const p2 = new Float32Array([1, 2, 3, 5]);
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);

    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);

    expect(result[0]).toBe(0.0); // invisible
    expect(result[1]).toBe(0.0);
    expect(result[2]).toBe(0.0);
    expect(Number.isNaN(result[1])).toBe(false);
    expect(Number.isNaN(result[2])).toBe(false);
  });

  it('#806: returns invisible when NaN is on the SECOND endpoint of a non-displayed dim', () => {
    // Mirror of the above, pinning the `!Number.isFinite(v2)` guard branch: the
    // FIRST endpoint is finite and inside the slice, the SECOND (p2) carries a
    // NaN on dim 3. Old code: p1_in is true so the both-out short-circuit is
    // skipped, dv=NaN-5=NaN is not < epsilon (no `continue`), and the NaN
    // t-params never trip `t1 >= t2` → it returns [1, NaN, NaN] (visible with
    // NaN params). Genuinely red without the guard.
    const p1 = new Float32Array([1, 2, 3, 5]);
    const p2 = new Float32Array([1, 2, 3, NaN]);
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);

    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);

    expect(result[0]).toBe(0.0); // invisible
    expect(result[1]).toBe(0.0);
    expect(result[2]).toBe(0.0);
    expect(Number.isNaN(result[1])).toBe(false);
    expect(Number.isNaN(result[2])).toBe(false);
  });

  it('#806: returns invisible when a non-displayed dim is +/-Inf on the FIRST endpoint', () => {
    // The non-finite value must sit on the FIRST endpoint (p1). With +/-Inf on
    // p1 and a finite in-slice p2, the old code produces dv=+/-Inf, whose
    // t_min/t_max become NaN (finite/Inf mixes divide to NaN), so it returns
    // [1, NaN, NaN] — genuinely red without the guard. (Note: putting -Inf on
    // p2 instead would spuriously pass via the legacy `t1 >= t2` path, so it
    // would NOT exercise the guard.)
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);

    // +Inf on the first endpoint's non-displayed dim.
    const posInf = clip_segment_single(
      new Float32Array([1, 2, 3, Infinity]),
      new Float32Array([1, 2, 3, 5]),
      slicePos,
      tolerance,
      displayDims,
      4
    );
    expect(posInf[0]).toBe(0.0);
    expect(posInf[1]).toBe(0.0);
    expect(posInf[2]).toBe(0.0);

    // -Inf on the first endpoint's non-displayed dim.
    const negInf = clip_segment_single(
      new Float32Array([1, 2, 3, -Infinity]),
      new Float32Array([1, 2, 3, 5]),
      slicePos,
      tolerance,
      displayDims,
      4
    );
    expect(negInf[0]).toBe(0.0);
    expect(negInf[1]).toBe(0.0);
    expect(negInf[2]).toBe(0.0);
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

  it('#806: marks a segment invisible when a non-displayed dim is NaN', () => {
    // seg0: normal, both in slice. seg1: v2 carries NaN on non-displayed dim 3.
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
      NaN, // v2: NaN on the slicing dim
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

    expect(count).toBe(1); // only seg0 visible
    expect(visibility[0]).toBe(1);
    expect(visibility[1]).toBe(0); // NaN segment invisible, not NaN-visible
    expect(Number.isNaN(t1[1])).toBe(false);
    expect(Number.isNaN(t2[1])).toBe(false);
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

describe('lines_clipping: compute_joint_codes', () => {
  // Mirrors the Rust kernel's own tests (wasm/rust/src/lines_clipping.rs) —
  // this module is not merely a WASM-missing fallback but the production
  // backend for ndim > 16, so the two must agree exactly. Agreement is trivial
  // now in a way it was not for the angle scalar this replaced: the output is
  // integer index arithmetic, with no f32/f64 accumulation order to reconcile.

  it('emits sentinels for disjoint segments (no shared vertices)', () => {
    const segments = Uint32Array.from({ length: 8 }, (_, i) => i);
    const outStart = new Float32Array(4).fill(9);
    const outEnd = new Float32Array(4).fill(9);

    const count = compute_joint_codes(
      segments,
      new Uint8Array([1, 1, 1, 1]),
      new Float32Array([0.0, 0.3, 0.0, 0.2]),
      new Float32Array([1.0, 1.0, 0.7, 0.8]),
      4,
      8,
      outStart,
      outEnd
    );

    expect(count).toBe(4);
    // Reaches both vertices, but nothing shares them.
    expect(outStart[0]).toBe(JOINT_FREE_END);
    expect(outEnd[0]).toBe(JOINT_FREE_END);
    // t1=0.3 → start trimmed off its vertex.
    expect(outStart[1]).toBe(JOINT_CLIPPED);
    expect(outEnd[1]).toBe(JOINT_FREE_END);
    // t2=0.7 → end trimmed.
    expect(outStart[2]).toBe(JOINT_FREE_END);
    expect(outEnd[2]).toBe(JOINT_CLIPPED);
    // Both trimmed.
    expect(outStart[3]).toBe(JOINT_CLIPPED);
    expect(outEnd[3]).toBe(JOINT_CLIPPED);
  });

  it('names the partner and which of ITS endpoints is shared', () => {
    // v0 → v1, v1 → v2: the two inner endpoints meet at v1.
    const outStart = new Float32Array(2).fill(9);
    const outEnd = new Float32Array(2).fill(9);

    compute_joint_codes(
      new Uint32Array([0, 1, 1, 2]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      3,
      outStart,
      outEnd
    );

    // Outer ends are free.
    expect(outStart[0]).toBe(JOINT_FREE_END);
    expect(outEnd[1]).toBe(JOINT_FREE_END);
    // Segment 0's END joins segment 1 at segment 1's START → +(1 + 1).
    expect(outEnd[0]).toBe(2);
    // Segment 1's START joins segment 0 at segment 0's END → −(0 + 3).
    expect(outStart[1]).toBe(-3);
  });

  it('is angle-independent: the code is a fact about topology', () => {
    // The kernel no longer reads positions at all, so a straight chain, a
    // right-angle bend and a 180-degree fold-back with the same connectivity
    // must all produce identical codes. The bend term is the vertex stage's
    // business now, measured in SCREEN space so it tracks the camera (#795).
    const run = (): [Float32Array, Float32Array] => {
      const s = new Float32Array(2).fill(9);
      const e = new Float32Array(2).fill(9);
      compute_joint_codes(
        new Uint32Array([0, 1, 1, 2]),
        new Uint8Array([1, 1]),
        new Float32Array([0, 0]),
        new Float32Array([1, 1]),
        2,
        3,
        s,
        e
      );
      return [s, e];
    };
    const [s1, e1] = run();
    const [s2, e2] = run();
    expect(Array.from(s1)).toEqual(Array.from(s2));
    expect(Array.from(e1)).toEqual(Array.from(e2));
  });

  it('treats a degree->=3 hub as a hub, not a joint', () => {
    const hubStart = new Float32Array(3).fill(9);
    const hubEnd = new Float32Array(3).fill(9);
    compute_joint_codes(
      new Uint32Array([0, 1, 0, 2, 0, 3]),
      new Uint8Array([1, 1, 1]),
      new Float32Array([0, 0, 0]),
      new Float32Array([1, 1, 1]),
      3,
      4,
      hubStart,
      hubEnd
    );
    expect(Array.from(hubStart)).toEqual([JOINT_HUB, JOINT_HUB, JOINT_HUB]);
    expect(Array.from(hubEnd)).toEqual([JOINT_FREE_END, JOINT_FREE_END, JOINT_FREE_END]);
  });

  it('does not anchor a joint on a neighbour trimmed away from the vertex', () => {
    // Mitering against a neighbour that does not reach the shared vertex would
    // build an edge the neighbour never draws.
    const outStart = new Float32Array(2).fill(9);
    const outEnd = new Float32Array(2).fill(9);
    compute_joint_codes(
      new Uint32Array([0, 1, 1, 2]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0.4]),
      new Float32Array([1, 1]),
      2,
      3,
      outStart,
      outEnd
    );
    expect(outEnd[0]).toBe(JOINT_FREE_END);
    expect(outStart[1]).toBe(JOINT_CLIPPED);
  });

  it('does not anchor a joint on an invisible neighbour, and slots stay contiguous', () => {
    // v0-v1, v1-v2 (culled), v2-v3 → visible segments occupy slots 0 and 1, so
    // an emitted slot always indexes a WRITTEN texel.
    const outStart = new Float32Array(2).fill(9);
    const outEnd = new Float32Array(2).fill(9);
    const count = compute_joint_codes(
      new Uint32Array([0, 1, 1, 2, 2, 3]),
      new Uint8Array([1, 0, 1]),
      new Float32Array([0, 0, 0]),
      new Float32Array([1, 1, 1]),
      3,
      4,
      outStart,
      outEnd
    );
    expect(count).toBe(2);
    expect(outEnd[0]).toBe(JOINT_FREE_END);
    expect(outStart[1]).toBe(JOINT_FREE_END);
  });

  it('reports the partner END for two segments meeting end-to-end', () => {
    // seg0: v0 → v1, seg1: v2 → v1. Both END on v1.
    const outStart = new Float32Array(2).fill(9);
    const outEnd = new Float32Array(2).fill(9);
    compute_joint_codes(
      new Uint32Array([0, 1, 2, 1]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Float32Array([1, 1]),
      2,
      3,
      outStart,
      outEnd
    );
    expect(outEnd[0]).toBe(-4); // joins slot 1 at its end → −(1 + 3)
    expect(outEnd[1]).toBe(-3); // joins slot 0 at its end → −(0 + 3)
  });

  it('never names itself when a segment registers both endpoints on one vertex', () => {
    // The two codes differ only in their end bit, so comparing whole codes is
    // not enough — the guard compares SLOTS. The angle-only scalar this replaced
    // survived the case by returning a plausible number; a code is dereferenced,
    // and a segment mitered against itself is the asymmetric-join case that
    // rasterises as a flap.
    const outStart = new Float32Array(1).fill(9);
    const outEnd = new Float32Array(1).fill(9);
    compute_joint_codes(
      new Uint32Array([1, 1]),
      new Uint8Array([1]),
      new Float32Array([0]),
      new Float32Array([1]),
      1,
      3,
      outStart,
      outEnd
    );
    expect(outStart[0]).toBe(JOINT_FREE_END);
    expect(outEnd[0]).toBe(JOINT_FREE_END);
  });

  it('reads nothing at an endpoint a NaN clip param kept out of the touch tables', () => {
    // Both passes run the SAME `t <= 0` test, so a NaN endpoint registers
    // nothing and reports JOINT_CLIPPED. With the reading pass on the
    // complement `!(t > 0)` — also false for NaN — it read anyway, and the
    // code-sum difference (which does not contain its own code) decoded to an
    // arbitrary slot. `slot < 0` never caught that: the difference is negative
    // only when the unregistered endpoint's code is the larger one.
    //
    // Here slot 0 carries the NaN while the two endpoints actually registered
    // on v5 are slots 1 and 2, so the difference is large and POSITIVE
    // (6 - 0 = 6 -> slot 3). FOUR segments, so that bogus slot is IN range and
    // names a real-but-unrelated segment: the `visibleCount` bound cannot see
    // it.
    const outStart = new Float32Array(4).fill(9);
    const outEnd = new Float32Array(4).fill(9);
    compute_joint_codes(
      new Uint32Array([5, 8, 5, 6, 5, 7, 0, 1]),
      new Uint8Array([1, 1, 1, 1]),
      new Float32Array([NaN, 0, 0, 0]),
      new Float32Array([1, 1, 1, 1]),
      4,
      9,
      outStart,
      outEnd
    );

    expect(outStart[0]).toBe(JOINT_CLIPPED);

    // The other slots are unaffected: v5 still holds exactly the two endpoints
    // that DID register, and they pair with each other at their STARTs.
    expect(outStart[1]).toBe(3); // slot 1 start joins slot 2's START: +(2 + 1)
    expect(outStart[2]).toBe(2); // slot 2 start joins slot 1's START: +(1 + 1)
    expect(outStart[3]).toBe(JOINT_FREE_END); // v0 is touched once
    for (const code of outEnd) expect(code).toBe(JOINT_FREE_END);

    // Belt and braces: no code may name a slot outside the stream.
    for (const code of [...outStart, ...outEnd]) {
      if (code > 0.5) expect(code - 1).toBeLessThan(4);
      else if (code < -2.5) expect(-code - 3).toBeLessThan(4);
    }
  });

  it('degrades BOTH sides of a joint whose slots straddle the f32 exact bound', () => {
    // Codes land in Float32Arrays, so a slot past 2^24 is rounded AT THE STORE
    // onto a valid neighbour — indistinguishable downstream from a deliberate
    // reference. Rejecting only the PARTNER's slot degrades one side of the
    // pair: the over-bound endpoint would see a representable partner, miter,
    // and rotate its end edge onto a miter line the other side never matched.
    // Both slots are therefore tested.
    //
    // Driven through `jointCodeForEndpoint` directly, mirroring the Rust unit
    // test on `joint_code`: reaching it via compute_joint_codes would need a
    // >16.7M-segment fixture.
    const degree = new Uint8Array([2]);
    const myCode = 5 << 1; // slot 5, start
    const over = MAX_EXACT_JOINT_SLOT + 1;
    const partnerCode = (over << 1) | 1; // the partner's END touches the vertex

    // Seen from the representable side: the PARTNER is unrepresentable.
    expect(
      jointCodeForEndpoint(0, myCode, 1, over + 1, new Int32Array([myCode + partnerCode]), degree)
    ).toBe(JOINT_FREE_END);

    // Seen from the over-bound side: the partner (slot 5) IS representable, so
    // only the own-slot test can reject it.
    const theirCode = (5 << 1) | 1;
    const myOverCode = over << 1;
    expect(
      jointCodeForEndpoint(
        0,
        myOverCode,
        1,
        over + 1,
        new Int32Array([myOverCode + theirCode]),
        degree
      )
    ).toBe(JOINT_FREE_END);

    // Sensitivity control: the largest REPRESENTABLE slot must still encode, or
    // both assertions above would pass for the wrong reason.
    const atBound = MAX_EXACT_JOINT_SLOT;
    const partnerOk = (atBound << 1) | 1;
    expect(
      jointCodeForEndpoint(0, myCode, 1, atBound + 1, new Int32Array([myCode + partnerOk]), degree)
    ).toBe(-(atBound + 3));
  });

  it('falls back to the free end for out-of-range vertex indices', () => {
    const outStart = new Float32Array(1).fill(9);
    const outEnd = new Float32Array(1).fill(9);
    compute_joint_codes(
      new Uint32Array([7, 9]),
      new Uint8Array([1]),
      new Float32Array([0]),
      new Float32Array([1]),
      1,
      2, // numVertices = 2 → both indices out of range
      outStart,
      outEnd
    );
    expect(outStart[0]).toBe(JOINT_FREE_END);
    expect(outEnd[0]).toBe(JOINT_FREE_END);
  });
});
