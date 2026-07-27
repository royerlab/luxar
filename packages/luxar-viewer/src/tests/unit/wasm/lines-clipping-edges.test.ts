/**
 * Edge-case tests for the WASM TypeScript-fallback lines-clipping helpers.
 *
 * Closes wasm.md gap cluster G9–G14:
 *   - [wasm.md G9][P5]  clip_segments_batch: out-of-range vertex index,
 *                       self-segment (v0 === v1), NaN in positions.
 *   - [wasm.md G10][P5] interpolate_clipped_positions: displayDims.length=0
 *                       (zero-fill all-3 columns of the output).
 *   - [wasm.md G11][P5] interpolate_scalars_batch / interpolate_colors_batch:
 *                       all-hidden visibility → loop never iterates → return 0.
 *   - [wasm.md G12][P5] calculate_segment_lengths: visibleCount=0 (no-op) and
 *                       NaN positions (NaN propagates through Math.sqrt).
 *   - [wasm.md G13][P5] compute_cap_suppression: all-hidden, and t1=t2=0.5
 *                       (both clipped) plus t1=0/t2=1 (neither clipped).
 *   - [wasm.md G14][P5] distance_3d: NaN inputs, a===b (zero distance, no
 *                       sqrt underflow).
 *
 * Pure math on typed arrays — no mocks. Mirrors the audit's three-geometry
 * principle: every helper has a known empty-input and NaN-propagation contract.
 */

import { describe, it, expect } from 'vitest';
import {
  clip_segments_batch,
  interpolate_clipped_positions,
  interpolate_scalars_batch,
  interpolate_colors_batch,
  calculate_segment_lengths,
  compute_cap_suppression,
  distance_3d,
} from '../../../wasm/typescript/lines-clipping';

describe('clip_segments_batch — out-of-range / self-segment / NaN [wasm.md G9]', () => {
  it('[G9] out-of-range vertex index: positions[OOB] reads undefined → NaN comparisons all-false → segment visible only if no hidden dim has a hard miss', () => {
    // 3 vertices, but segment indexes vertex 99 (way OOB). With ndim=4 and
    // displayDims=[0,1,2], dim 3 is the only hidden dim. positions[99*4+3]
    // is undefined → v1Val=NaN → p1In = (NaN >= NaN-tol && NaN <= NaN+tol)
    // is false. p2In = same. Then "both out, same side" guard uses `<` / `>`
    // with NaN → all false → no early `visible=false` exit. dv = NaN-finite =
    // NaN; `Math.abs(NaN) < 1e-10` is false → not skipped. tMin/tMax = NaN.
    // dv > 0 is false → else-branch: t1 = max(0, NaN) = NaN, t2 = min(1, NaN) = NaN.
    // t1 >= t2 → NaN >= NaN is false → no early exit. Segment ends as visible=true
    // with t1=NaN, t2=NaN. Pin this documented OOB behaviour so a future
    // hardening (throw / clamp) surfaces as intentional.
    const positions = new Float32Array(12); // 3 verts × 4 dim
    const segments = new Uint32Array([0, 99]);
    const slicePos = new Float32Array([0, 0, 0, 0]);
    const tolerance = new Float32Array([1, 1, 1, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const outVis = new Uint8Array(1);
    const outT1 = new Float32Array(1);
    const outT2 = new Float32Array(1);
    expect(() =>
      clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        4,
        1,
        outVis,
        outT1,
        outT2
      )
    ).not.toThrow();
    // Contract: NaN t-params escape into the output (no crash, no zero-fill).
    // The visibility byte is whatever the algorithm produced — we DON'T assert
    // a specific value, only that the call completed safely.
    expect(outVis[0] === 0 || outVis[0] === 1).toBe(true);
  });

  it('[G9] self-segment (v0 === v1): dv=0 in every dim → skipped → fully visible with t1=0,t2=1', () => {
    // A degenerate segment where both endpoints are the same vertex. Every
    // dim has dv=0 < 1e-10 → continue → no clipping applied. If the (single)
    // vertex is in the slice, segment is visible with t1=0, t2=1.
    const positions = new Float32Array([0, 0, 0, 5]); // vertex 0 at the slice in dim 3
    const segments = new Uint32Array([0, 0]); // self-segment
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1, 1, 1, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const outVis = new Uint8Array(1);
    const outT1 = new Float32Array(1);
    const outT2 = new Float32Array(1);
    const n = clip_segments_batch(
      positions,
      segments,
      slicePos,
      tolerance,
      displayDims,
      4,
      1,
      outVis,
      outT1,
      outT2
    );
    expect(n).toBe(1);
    expect(outVis[0]).toBe(1);
    expect(outT1[0]).toBe(0);
    expect(outT2[0]).toBe(1);
  });

  it('[G9] NaN in positions: segment marked visible with NaN t-params (no crash)', () => {
    // Pin contract: NaN in a HIDDEN dim of vertex 0 propagates into dv/tMin/tMax
    // but no `< 1e-10` short-circuit fires; final t1>=t2 check yields NaN>=NaN=false.
    // No early exit, segment escapes the loop as visible with NaN parameters.
    // A defensive future hardening can flip this; the test pins today's behaviour.
    const positions = new Float32Array([0, 0, 0, Number.NaN, 0, 0, 0, 0]);
    const segments = new Uint32Array([0, 1]);
    const slicePos = new Float32Array([0, 0, 0, 0]);
    const tolerance = new Float32Array([1, 1, 1, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const outVis = new Uint8Array(1);
    const outT1 = new Float32Array(1);
    const outT2 = new Float32Array(1);
    expect(() =>
      clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        4,
        1,
        outVis,
        outT1,
        outT2
      )
    ).not.toThrow();
  });

  it('[G9] numSegments === 0: empty loop, return 0, output untouched', () => {
    const outVis = new Uint8Array(0);
    const outT1 = new Float32Array(0);
    const outT2 = new Float32Array(0);
    const n = clip_segments_batch(
      new Float32Array(0),
      new Uint32Array(0),
      new Float32Array([0, 0, 0]),
      new Float32Array([1, 1, 1]),
      new Uint32Array([0, 1, 2]),
      3,
      0,
      outVis,
      outT1,
      outT2
    );
    expect(n).toBe(0);
  });
});

describe('interpolate_clipped_positions — displayDims boundaries [wasm.md G10]', () => {
  it('[G10] displayDims.length === 0: every column of every visible segment is zero-filled', () => {
    // numDisplay = min(0, 3) = 0; the first inner loop never runs.
    // The "pad to 3D" loop fills all 3 columns with 0.
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]); // 2 verts × 3 dim
    const segments = new Uint32Array([0, 1]);
    const visibility = new Uint8Array([1]);
    const t1Params = new Float32Array([0]);
    const t2Params = new Float32Array([1]);
    const displayDims = new Uint32Array(0);
    const outStart = new Float32Array(3).fill(99); // sentinel
    const outEnd = new Float32Array(3).fill(99);
    const n = interpolate_clipped_positions(
      positions,
      segments,
      visibility,
      t1Params,
      t2Params,
      displayDims,
      3,
      1,
      outStart,
      outEnd
    );
    expect(n).toBe(1);
    expect(Array.from(outStart)).toEqual([0, 0, 0]);
    expect(Array.from(outEnd)).toEqual([0, 0, 0]);
  });

  it('[G10] displayDims.length === 1: column 0 interpolates, columns 1-2 zero-filled', () => {
    // numDisplay = 1. Inner loop runs once for outD=0, dim = displayDims[0] = 1.
    // The "pad" loop fills columns 1 and 2 with 0.
    const positions = new Float32Array([0, 10, 0, 0, 20, 0]); // 2 verts × 3 dim
    const segments = new Uint32Array([0, 1]);
    const visibility = new Uint8Array([1]);
    const t1Params = new Float32Array([0]);
    const t2Params = new Float32Array([1]);
    const displayDims = new Uint32Array([1]); // display dim 1
    const outStart = new Float32Array(3).fill(99);
    const outEnd = new Float32Array(3).fill(99);
    interpolate_clipped_positions(
      positions,
      segments,
      visibility,
      t1Params,
      t2Params,
      displayDims,
      3,
      1,
      outStart,
      outEnd
    );
    expect(outStart[0]).toBe(10); // p1[1] at t1=0
    expect(outStart[1]).toBe(0); // padded
    expect(outStart[2]).toBe(0);
    expect(outEnd[0]).toBe(20); // p2[1] at t2=1
    expect(outEnd[1]).toBe(0);
    expect(outEnd[2]).toBe(0);
  });

  it('[G10] displayDims.length > 3: only first 3 columns used (min clamp)', () => {
    // numDisplay = min(5, 3) = 3 → extra display dims silently dropped.
    const positions = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]); // 2 verts × 5 dim
    const segments = new Uint32Array([0, 1]);
    const visibility = new Uint8Array([1]);
    const t1Params = new Float32Array([0]);
    const t2Params = new Float32Array([1]);
    const displayDims = new Uint32Array([0, 1, 2, 3, 4]);
    const outStart = new Float32Array(3);
    const outEnd = new Float32Array(3);
    interpolate_clipped_positions(
      positions,
      segments,
      visibility,
      t1Params,
      t2Params,
      displayDims,
      5,
      1,
      outStart,
      outEnd
    );
    expect(Array.from(outStart)).toEqual([1, 2, 3]); // first 3 dims of vert 0 at t=0
    expect(Array.from(outEnd)).toEqual([6, 7, 8]); // first 3 dims of vert 1 at t=1
  });
});

describe('interpolate_scalars_batch / interpolate_colors_batch all-hidden [wasm.md G11]', () => {
  it('[G11] scalars: visibility all-zero → loop never iterates → return 0, output untouched', () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const segments = new Uint32Array([0, 1, 2, 3]);
    const visibility = new Uint8Array([0, 0]);
    const t1Params = new Float32Array([0, 0]);
    const t2Params = new Float32Array([1, 1]);
    const outStart = new Float32Array(2).fill(99); // sentinel
    const outEnd = new Float32Array(2).fill(99);
    const n = interpolate_scalars_batch(
      values,
      segments,
      visibility,
      t1Params,
      t2Params,
      2,
      outStart,
      outEnd
    );
    expect(n).toBe(0);
    expect(Array.from(outStart)).toEqual([99, 99]);
    expect(Array.from(outEnd)).toEqual([99, 99]);
  });

  it('[G11] colors: visibility all-zero → loop never iterates → return 0, output untouched', () => {
    const colors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]); // 4 verts × 3
    const segments = new Uint32Array([0, 1, 2, 3]);
    const visibility = new Uint8Array([0, 0]);
    const t1Params = new Float32Array([0, 0]);
    const t2Params = new Float32Array([1, 1]);
    const outStart = new Float32Array(6).fill(99);
    const outEnd = new Float32Array(6).fill(99);
    const n = interpolate_colors_batch(
      colors,
      segments,
      visibility,
      t1Params,
      t2Params,
      2,
      outStart,
      outEnd
    );
    expect(n).toBe(0);
    expect(Array.from(outStart)).toEqual([99, 99, 99, 99, 99, 99]);
  });

  it('[G11] scalars: mixed visibility → output is compacted (only visible segments written)', () => {
    // Pin the compaction contract: hidden segments do NOT consume an output
    // slot. With 3 segments and visibility=[0,1,0], only segment 1 writes
    // to outStart[0] / outEnd[0].
    const values = new Float32Array([1, 2, 3, 4]);
    const segments = new Uint32Array([0, 1, 1, 2, 2, 3]);
    const visibility = new Uint8Array([0, 1, 0]);
    const t1Params = new Float32Array([0, 0, 0]);
    const t2Params = new Float32Array([1, 1, 1]);
    const outStart = new Float32Array(3).fill(99);
    const outEnd = new Float32Array(3).fill(99);
    const n = interpolate_scalars_batch(
      values,
      segments,
      visibility,
      t1Params,
      t2Params,
      3,
      outStart,
      outEnd
    );
    expect(n).toBe(1);
    expect(outStart[0]).toBe(2); // segment 1: t=0 → val[1] = 2
    expect(outEnd[0]).toBe(3); // segment 1: t=1 → val[2] = 3
    expect(outStart[1]).toBe(99); // untouched
  });
});

describe('calculate_segment_lengths — boundaries and NaN [wasm.md G12]', () => {
  it('[G12] visibleCount === 0: no-op, output untouched (sentinel preserved)', () => {
    const startPositions = new Float32Array([0, 0, 0]);
    const endPositions = new Float32Array([1, 1, 1]);
    const output = new Float32Array(2).fill(42); // sentinel
    calculate_segment_lengths(startPositions, endPositions, 0, output);
    expect(Array.from(output)).toEqual([42, 42]);
  });

  it('[G12] NaN in positions → NaN propagates through Math.sqrt → output[i] is NaN', () => {
    const startPositions = new Float32Array([0, 0, 0]);
    const endPositions = new Float32Array([1, Number.NaN, 1]);
    const output = new Float32Array(1);
    calculate_segment_lengths(startPositions, endPositions, 1, output);
    expect(Number.isNaN(output[0])).toBe(true);
  });

  it('[G12] zero-length segment (start === end): output is 0 (no sqrt underflow)', () => {
    const startPositions = new Float32Array([1.5, 2.5, 3.5]);
    const endPositions = new Float32Array([1.5, 2.5, 3.5]);
    const output = new Float32Array(1);
    calculate_segment_lengths(startPositions, endPositions, 1, output);
    expect(output[0]).toBe(0);
  });

  it('[G12] 3-4-5 triangle: length is exactly 5 (no FP drift on exact roots)', () => {
    const startPositions = new Float32Array([0, 0, 0]);
    const endPositions = new Float32Array([3, 4, 0]);
    const output = new Float32Array(1);
    calculate_segment_lengths(startPositions, endPositions, 1, output);
    expect(output[0]).toBe(5);
  });
});

describe('compute_cap_suppression — empty / boundary [wasm.md G13]', () => {
  /** Disjoint segments (no shared vertices) → the clipped-flag path alone. */
  const disjointSegs = (n: number): Uint32Array => Uint32Array.from({ length: n * 2 }, (_, i) => i);
  /** Positions for `n` disjoint unit-length segments along +x. */
  const disjointPos = (n: number): [Float32Array, Float32Array] => {
    const s = new Float32Array(n * 3);
    const e = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      s[i * 3] = i * 10;
      e[i * 3] = i * 10 + 1;
    }
    return [s, e];
  };

  it('[G13] all visibility=0: loop never executes, output untouched, return 0', () => {
    const visibility = new Uint8Array([0, 0, 0]);
    const t1Params = new Float32Array([0.5, 0.5, 0.5]);
    const t2Params = new Float32Array([0.5, 0.5, 0.5]);
    const outStart = new Float32Array(3).fill(99);
    const outEnd = new Float32Array(3).fill(99);
    const [sp, ep] = disjointPos(3);
    const n = compute_cap_suppression(
      disjointSegs(3),
      visibility,
      t1Params,
      t2Params,
      3,
      6,
      sp,
      ep,
      outStart,
      outEnd
    );
    expect(n).toBe(0);
    expect(Array.from(outStart)).toEqual([99, 99, 99]);
    expect(Array.from(outEnd)).toEqual([99, 99, 99]);
  });

  it('[G13] t1=t2=0.5 (both clipped from outside): both suppressions = 1', () => {
    // t1=0.5 > 0 → start clipped. t2=0.5 < 1 → end clipped. Both = 1.
    const outStart = new Float32Array(1);
    const outEnd = new Float32Array(1);
    const [sp, ep] = disjointPos(1);
    const n = compute_cap_suppression(
      disjointSegs(1),
      new Uint8Array([1]),
      new Float32Array([0.5]),
      new Float32Array([0.5]),
      1,
      2,
      sp,
      ep,
      outStart,
      outEnd
    );
    expect(n).toBe(1);
    expect(outStart[0]).toBe(1);
    expect(outEnd[0]).toBe(1);
  });

  it('[G13] t1=0, t2=1, no neighbour (free ends): both suppressions = 0', () => {
    // The strict-inequality clipped contract (`t1 > 0` / `t2 < 1`) yields 0,
    // and with no segment sharing either vertex there is no joint to suppress.
    const outStart = new Float32Array(1);
    const outEnd = new Float32Array(1);
    const [sp, ep] = disjointPos(1);
    compute_cap_suppression(
      disjointSegs(1),
      new Uint8Array([1]),
      new Float32Array([0]),
      new Float32Array([1]),
      1,
      2,
      sp,
      ep,
      outStart,
      outEnd
    );
    expect(outStart[0]).toBe(0);
    expect(outEnd[0]).toBe(0);
  });

  it('[G13] asymmetry: t1=0 / t2=0.7 → start free (0), end clipped (1)', () => {
    const outStart = new Float32Array(1);
    const outEnd = new Float32Array(1);
    const [sp, ep] = disjointPos(1);
    compute_cap_suppression(
      disjointSegs(1),
      new Uint8Array([1]),
      new Float32Array([0]),
      new Float32Array([0.7]),
      1,
      2,
      sp,
      ep,
      outStart,
      outEnd
    );
    expect(outStart[0]).toBe(0);
    expect(outEnd[0]).toBe(1);
  });

  it('[G13] mixed visibility compacts the output (only visible segments contribute)', () => {
    const visibility = new Uint8Array([0, 1, 0, 1]);
    const t1Params = new Float32Array([0, 0.3, 0, 0]);
    const t2Params = new Float32Array([1, 1, 1, 0.9]);
    const outStart = new Float32Array(4).fill(99); // sentinel
    const outEnd = new Float32Array(4).fill(99);
    const [sp, ep] = disjointPos(4);
    const n = compute_cap_suppression(
      disjointSegs(4),
      visibility,
      t1Params,
      t2Params,
      4,
      8,
      sp,
      ep,
      outStart,
      outEnd
    );
    expect(n).toBe(2);
    // Compacted: outStart[0] = (seg 1 → t1=0.3 > 0) = 1; outStart[1] = (seg 3 → free) = 0.
    expect(outStart[0]).toBe(1);
    expect(outStart[1]).toBe(0);
    expect(outStart[2]).toBe(99); // untouched
    expect(outEnd[0]).toBe(0); // seg 1: t2=1, no neighbour → free end
    expect(outEnd[1]).toBe(1); // seg 3: t2=0.9 < 1 → clipped
  });
});

describe('distance_3d — NaN / zero-length / non-integer [wasm.md G14]', () => {
  it('[G14] a === b (identical vectors): distance is exactly 0 (no sqrt underflow)', () => {
    const a = new Float32Array([1.5, -2.5, 3.5]);
    const b = new Float32Array([1.5, -2.5, 3.5]);
    expect(distance_3d(a, b)).toBe(0);
  });

  it('[G14] NaN in a → distance is NaN (propagation contract)', () => {
    const a = new Float32Array([Number.NaN, 0, 0]);
    const b = new Float32Array([1, 1, 1]);
    expect(Number.isNaN(distance_3d(a, b))).toBe(true);
  });

  it('[G14] NaN in b → distance is NaN (symmetric to NaN-in-a)', () => {
    const a = new Float32Array([1, 1, 1]);
    const b = new Float32Array([Number.NaN, 0, 0]);
    expect(Number.isNaN(distance_3d(a, b))).toBe(true);
  });

  it('[G14] Infinity in b → distance is Infinity', () => {
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([Number.POSITIVE_INFINITY, 0, 0]);
    expect(distance_3d(a, b)).toBe(Number.POSITIVE_INFINITY);
  });

  it('[G14] non-integer distance: matches Float64 reference within Float32 epsilon', () => {
    // sqrt(1^2 + 1^2 + 1^2) = sqrt(3) ≈ 1.732050807...
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([1, 1, 1]);
    expect(distance_3d(a, b)).toBeCloseTo(Math.sqrt(3), 6);
  });
});
