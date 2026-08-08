/**
 * Edge-case tests for the WASM TypeScript-fallback lines-clipping helpers.
 *
 * Closes wasm.md gap cluster G9–G14:
 *   - [wasm.md G9][P5]  clip_segments_batch: out-of-range vertex index and
 *                       NaN in positions (both #806: non-finite on a hidden
 *                       dim → segment invisible), self-segment (v0 === v1).
 *   - [wasm.md G10][P5] interpolate_clipped_positions: displayDims.length=0
 *                       (zero-fill all-3 columns of the output).
 *   - [wasm.md G11][P5] interpolate_scalars_batch / interpolate_colors_batch:
 *                       all-hidden visibility → loop never iterates → return 0.
 *   - [wasm.md G12][P5] calculate_segment_lengths: visibleCount=0 (no-op) and
 *                       NaN positions (NaN propagates through Math.sqrt).
 *   - [wasm.md G13][P5] compute_joint_codes: all-hidden, and t1=t2=0.5
 *                       (both clipped) plus t1=0/t2=1 (neither clipped).
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
  compute_joint_codes,
  MAX_EXACT_JOINT_SLOT,
  JOINT_CLIPPED,
  JOINT_FREE_END,
} from '../../../wasm/typescript/lines-clipping';

describe('clip_segments_batch — out-of-range / self-segment / NaN [wasm.md G9]', () => {
  it('[G9/#806] out-of-range vertex index: positions[OOB] reads undefined → non-finite on a hidden dim → segment invisible', () => {
    // 3 vertices, but segment indexes vertex 99 (way OOB). With ndim=4 and
    // displayDims=[0,1,2], dim 3 is the only hidden dim. positions[99*4+3]
    // is undefined → v2Val is non-finite. Under the #806 contract a non-finite
    // coordinate on a slicing (non-displayed) dimension cannot be localized
    // against the slice, so the guard marks the segment invisible up front
    // (visible=0) BEFORE any NaN can propagate into the t-params. This pins
    // that behaviour: no crash, no NaN leaking into the output, segment culled.
    const positions = new Float32Array(12); // 3 verts × 4 dim
    const segments = new Uint32Array([0, 99]);
    const slicePos = new Float32Array([0, 0, 0, 0]);
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
    // #806 contract: invisible, and NO NaN t-params leak into the output.
    expect(n).toBe(0);
    expect(outVis[0]).toBe(0);
    expect(Number.isNaN(outT1[0])).toBe(false);
    expect(Number.isNaN(outT2[0])).toBe(false);
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

  it('[G9/#806] NaN in a hidden dim: segment marked invisible, no NaN t-params leaked', () => {
    // #806 contract: a NaN on a HIDDEN (non-displayed) dim of vertex 0 cannot
    // be localized against the slice, so the guard marks the segment invisible
    // (visible=0) up front — it does NOT escape the loop as visible with NaN
    // t-params (the pre-#806 behaviour, which silently vanished the segment
    // downstream when the NaN reached the interpolation stage).
    const positions = new Float32Array([0, 0, 0, Number.NaN, 0, 0, 0, 0]);
    const segments = new Uint32Array([0, 1]);
    const slicePos = new Float32Array([0, 0, 0, 0]);
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
    expect(n).toBe(0);
    expect(outVis[0]).toBe(0);
    expect(Number.isNaN(outT1[0])).toBe(false);
    expect(Number.isNaN(outT2[0])).toBe(false);
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

describe('joint-code f32 representability bound', () => {
  // The bound itself is exercised as a BRANCH on the Rust side
  // (`slot_past_the_f32_exact_bound_degrades_to_free_end`), where `joint_code`
  // is a free function. Its TypeScript mirror is a closure inside
  // `computeJointCodes`, and reaching the branch through the public kernel
  // would need a >16.7M-segment fixture — so what is pinned here is the part
  // that can silently DRIFT between the two hand-written implementations: the
  // constant, and the arithmetic reason it sits where it does.
  it('is the largest slot whose END encoding survives a float32 round-trip', () => {
    const decode = (c: number) => (c > 0 ? Math.trunc(c) - 1 : Math.trunc(-c) - 3);
    const atEnd = (slot: number) => -(slot + 3);
    const atStart = (slot: number) => slot + 1;

    expect(MAX_EXACT_JOINT_SLOT).toBe((1 << 24) - 3);
    // At the bound: both encodings round-trip exactly.
    expect(decode(Math.fround(atEnd(MAX_EXACT_JOINT_SLOT)))).toBe(MAX_EXACT_JOINT_SLOT);
    expect(decode(Math.fround(atStart(MAX_EXACT_JOINT_SLOT)))).toBe(MAX_EXACT_JOINT_SLOT);
    // One past it, the END encoding — the larger magnitude, so the binding one —
    // rounds to a neighbour and decodes to the WRONG slot. That wrong slot is
    // in range and indistinguishable downstream, which is why the kernel has to
    // refuse to emit it rather than letting a consumer notice.
    const over = MAX_EXACT_JOINT_SLOT + 1;
    expect(decode(Math.fround(atEnd(over)))).not.toBe(over);
  });
});

describe('compute_joint_codes — empty / boundary [wasm.md G13]', () => {
  /** Disjoint segments (no shared vertices) → the sentinel path alone. */
  const disjointSegs = (n: number): Uint32Array => Uint32Array.from({ length: n * 2 }, (_, i) => i);

  it('[G13] all visibility=0: loop never executes, output untouched, return 0', () => {
    const visibility = new Uint8Array([0, 0, 0]);
    const t1Params = new Float32Array([0.5, 0.5, 0.5]);
    const t2Params = new Float32Array([0.5, 0.5, 0.5]);
    const outStart = new Float32Array(3).fill(99);
    const outEnd = new Float32Array(3).fill(99);
    const n = compute_joint_codes(
      disjointSegs(3),
      visibility,
      t1Params,
      t2Params,
      3,
      6,
      outStart,
      outEnd
    );
    expect(n).toBe(0);
    expect(Array.from(outStart)).toEqual([99, 99, 99]);
    expect(Array.from(outEnd)).toEqual([99, 99, 99]);
  });

  it('[G13] t1=t2=0.5 (both clipped from outside): both codes = JOINT_CLIPPED', () => {
    // t1=0.5 > 0 → start trimmed off its vertex. t2=0.5 < 1 → end trimmed. A
    // trimmed endpoint can never be a joint: no neighbour meets it there.
    const outStart = new Float32Array(1);
    const outEnd = new Float32Array(1);
    const n = compute_joint_codes(
      disjointSegs(1),
      new Uint8Array([1]),
      new Float32Array([0.5]),
      new Float32Array([0.5]),
      1,
      2,
      outStart,
      outEnd
    );
    expect(n).toBe(1);
    expect(outStart[0]).toBe(JOINT_CLIPPED);
    expect(outEnd[0]).toBe(JOINT_CLIPPED);
  });

  it('[G13] t1=0, t2=1, no neighbour: both codes = JOINT_FREE_END', () => {
    // The strict-inequality clipped contract (`t1 > 0` / `t2 < 1`) means these
    // endpoints DO reach their vertices — but nothing shares them, so there is
    // no partner to name.
    const outStart = new Float32Array(1);
    const outEnd = new Float32Array(1);
    compute_joint_codes(
      disjointSegs(1),
      new Uint8Array([1]),
      new Float32Array([0]),
      new Float32Array([1]),
      1,
      2,
      outStart,
      outEnd
    );
    expect(outStart[0]).toBe(JOINT_FREE_END);
    expect(outEnd[0]).toBe(JOINT_FREE_END);
  });

  it('[G13] asymmetry: t1=0 / t2=0.7 → start free, end clipped', () => {
    const outStart = new Float32Array(1);
    const outEnd = new Float32Array(1);
    compute_joint_codes(
      disjointSegs(1),
      new Uint8Array([1]),
      new Float32Array([0]),
      new Float32Array([0.7]),
      1,
      2,
      outStart,
      outEnd
    );
    expect(outStart[0]).toBe(JOINT_FREE_END);
    expect(outEnd[0]).toBe(JOINT_CLIPPED);
  });

  it('[G13] mixed visibility compacts the output (only visible segments contribute)', () => {
    const visibility = new Uint8Array([0, 1, 0, 1]);
    const t1Params = new Float32Array([0, 0.3, 0, 0]);
    const t2Params = new Float32Array([1, 1, 1, 0.9]);
    const outStart = new Float32Array(4).fill(99); // sentinel
    const outEnd = new Float32Array(4).fill(99);
    const n = compute_joint_codes(
      disjointSegs(4),
      visibility,
      t1Params,
      t2Params,
      4,
      8,
      outStart,
      outEnd
    );
    expect(n).toBe(2);
    // Compacted into slots 0 and 1: the emitted partner slots are therefore
    // VISIBLE-stream indices, not source segment indices — which is exactly
    // what makes a code a line-texture storage slot.
    expect(outStart[0]).toBe(JOINT_CLIPPED); // seg 1: t1=0.3 > 0
    expect(outStart[1]).toBe(JOINT_FREE_END); // seg 3: reaches, unshared
    expect(outStart[2]).toBe(99); // untouched
    expect(outEnd[0]).toBe(JOINT_FREE_END); // seg 1: t2=1, no neighbour
    expect(outEnd[1]).toBe(JOINT_CLIPPED); // seg 3: t2=0.9 < 1
  });
});
