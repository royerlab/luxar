/**
 * Edge-case tests for the WASM TypeScript-fallback paths.
 *
 * Despite the historical "decode-and-fallback-edges" file name, this file
 * covers four distinct TS-fallback areas (wasm.md O12 → renamed to
 * `tsfallback-edges.test.ts` to match):
 *   - decode_quantized/decode_log_scalar boundaries (degenerate ranges)
 *   - mahalanobis_distance ndim=1 (forward-sub identity)
 *   - compute_gsplats_attenuation numHidden=ndim
 *   - clip_segment_single parallel-segment + lines-clipping edges
 *
 * Pure math on typed arrays — no mocks. All cases exercise the TS fallback
 * code path; the WASM binary, when present, must match these results
 * (covered separately by wasm-vs-typescript.test.ts).
 */

import { describe, it, expect } from 'vitest';
import {
  decode_quantized_u8,
  decode_quantized_u16,
  decode_log_scalar_u8,
  decode_log_scalar_u16,
  decode_lut_scalar_u8,
  decode_lut_scalar_u16,
  decode_lut_row_u8,
  decode_lut_row_u16,
} from '../../../wasm/typescript/decode';
import { mahalanobis_distance } from '../../../wasm/typescript/gsplats-processing';
import { compute_gsplats_attenuation } from '../../../wasm/typescript/gsplats-processing';
import { clip_segment_single, lerp_vec3 } from '../../../wasm/typescript/lines-clipping';

describe('decode_quantized — degenerate range (minVal === maxVal)', () => {
  // When minVal === maxVal the scale becomes zero; every output entry must
  // collapse to the constant value, irrespective of the input byte.
  it('u8: emits the constant value for every input byte', () => {
    const data = new Uint8Array([0, 64, 128, 255]);
    const output = new Float32Array(4);
    decode_quantized_u8(data, 3.14, 3.14, output);
    // Float32 rounds 3.14 → ~3.140000104904175; use closeness.
    for (let i = 0; i < 4; i++) {
      expect(output[i]).toBeCloseTo(3.14, 5);
    }
  });

  it('u16: emits the constant value for every input word (-1.5 is exact in Float32)', () => {
    const data = new Uint16Array([0, 32768, 65535]);
    const output = new Float32Array(3);
    decode_quantized_u16(data, -1.5, -1.5, output);
    // -1.5 is exactly representable in Float32.
    expect(Array.from(output)).toEqual([-1.5, -1.5, -1.5]);
  });

  it('u8: zero-length input leaves output untouched', () => {
    const output = new Float32Array(3).fill(42);
    decode_quantized_u8(new Uint8Array(0), 0, 10, output);
    expect(Array.from(output)).toEqual([42, 42, 42]);
  });
});

describe('decode_log_scalar — maxLog boundary', () => {
  // maxLog=0 collapses every output to expm1(0) = 0.
  it('u8: maxLog=0 produces 0 for every input byte', () => {
    const data = new Uint8Array([0, 64, 128, 255]);
    const output = new Float32Array(4);
    decode_log_scalar_u8(data, 0, output);
    expect(Array.from(output)).toEqual([0, 0, 0, 0]);
  });

  it('u16: maxLog=0 produces 0 for every input word', () => {
    const data = new Uint16Array([0, 1000, 65535]);
    const output = new Float32Array(3);
    decode_log_scalar_u16(data, 0, output);
    expect(Array.from(output)).toEqual([0, 0, 0]);
  });

  // Negative maxLog produces decreasing-then-negative outputs:
  // expm1(positive * negative) = expm1(negative) ∈ (-1, 0).
  it('u8: negative maxLog produces monotonically non-positive outputs', () => {
    const data = new Uint8Array([0, 255]);
    const output = new Float32Array(2);
    decode_log_scalar_u8(data, -2.0, output);
    expect(output[0]).toBeCloseTo(0, 5); // expm1(0) = 0
    expect(output[1]).toBeCloseTo(Math.expm1(-2.0), 5); // ≈ -0.8647
    expect(output[1]).toBeLessThan(output[0]); // monotone decreasing
  });
});

describe('mahalanobis_distance — ndim=1', () => {
  // For ndim=1 the packed Cholesky is a single diagonal entry L[0,0].
  // Forward substitution reduces to y[0] = diff[0] / L[0,0], and
  // ||y|| = |y[0]|. Pins the trivial 1D case so a refactor of the
  // packedIndex helper doesn't silently break the degenerate path.
  it('ndim=1: returns |diff/L00|', () => {
    const diff = new Float32Array([4]);
    const packedL = new Float32Array([2]); // L[0,0] = 2
    expect(mahalanobis_distance(diff, packedL, 1)).toBeCloseTo(2, 5); // 4/2 = 2
  });

  it('ndim=1: negative diff returns positive distance (norm)', () => {
    const diff = new Float32Array([-3]);
    const packedL = new Float32Array([1.5]);
    expect(mahalanobis_distance(diff, packedL, 1)).toBeCloseTo(2, 5); // |-3/1.5| = 2
  });

  it('ndim=1: zero diff returns 0 distance', () => {
    const diff = new Float32Array([0]);
    const packedL = new Float32Array([1]);
    expect(mahalanobis_distance(diff, packedL, 1)).toBe(0);
  });

  it('ndim=1: degenerate diagonal (L00 ~ 0) yields 0 distance (defensive)', () => {
    // Source uses `diag > 1e-10 ? val / diag : 0` to clamp degenerate
    // Cholesky cells. A tiny diagonal collapses the contribution.
    const diff = new Float32Array([10]);
    const packedL = new Float32Array([1e-12]); // below epsilon
    expect(mahalanobis_distance(diff, packedL, 1)).toBe(0);
  });
});

describe('compute_gsplats_attenuation — numHidden=ndim', () => {
  // Boundary: every dimension is hidden, no displayed dim. The marginal
  // Cholesky covers the full ndim and the attenuation depends on full-
  // dimensional Mahalanobis distance to the slice. Pins the "all hidden"
  // case so a mutant that special-cases numHidden < ndim only would fail.
  it('every dim hidden: splat at slice yields attenuation 1, far splat yields ~0', () => {
    const ndim = 3;
    // 2 splats, 3D, with identity Cholesky (packed lower-tri, 6 entries each).
    const positions = new Float32Array([
      0, 0, 0, // splat 0 at slice
      10, 10, 10, // splat 1 far
    ]);
    const choleskyEntries = [
      1, // L00
      0, 1, // L10 L11
      0, 0, 1, // L20 L21 L22
    ];
    const cholesky = new Float32Array([...choleskyEntries, ...choleskyEntries]);
    const amplitudes = new Float32Array([1, 1]);
    const slicePos = new Float32Array([0, 0, 0]);
    const hiddenDims = new Uint32Array([0, 1, 2]); // all hidden

    const visibility = new Uint8Array(2);
    const attenuation = new Float32Array(2);

    const count = compute_gsplats_attenuation(
      positions,
      cholesky,
      amplitudes,
      slicePos,
      hiddenDims,
      ndim,
      2,
      0.01,
      3.0,
      visibility,
      attenuation
    );

    // Splat 0: diff=0 in every dim → Mahalanobis 0 → attenuation 1.
    expect(attenuation[0]).toBeCloseTo(1.0, 5);
    expect(visibility[0]).toBe(1);

    // Splat 1: diff=(10,10,10), Mahalanobis ≈ sqrt(300) ≈ 17.3,
    // well beyond 3σ truncation → attenuation = 0 by C0 clamp.
    expect(attenuation[1]).toBe(0);
    expect(visibility[1]).toBe(0);
    expect(count).toBe(1);
  });
});

describe('clip_segment_single — parallel-to-slice edge case (dv < 1e-10)', () => {
  // When p1[dim] === p2[dim] in a hidden dimension, the segment is
  // parallel to the slice in that dimension. The source skips that
  // dim via `if (Math.abs(dv) < 1e-10) continue;`. A mutant that
  // dropped this guard would divide by zero and yield NaN/Infinity
  // t-values downstream.
  it('parallel hidden-dim with both endpoints inside slice is fully visible', () => {
    // 4D segment; dim 3 is hidden. Both endpoints sit at exactly the
    // slice center → dv = 0 in dim 3.
    const p1 = new Float32Array([0, 0, 0, 5]);
    const p2 = new Float32Array([10, 10, 10, 5]); // same dim-3 value
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);
    expect(result[0]).toBe(1.0); // visible
    expect(result[1]).toBeCloseTo(0.0, 5);
    expect(result[2]).toBeCloseTo(1.0, 5);
  });

  it('parallel hidden-dim with both endpoints OUTSIDE the slice is invisible', () => {
    // Same dv=0 contract, but now both endpoints sit outside slice.tol.
    // Source classifies both endpoints as "not in slice" on the same
    // side → Case E (invisible). The parallel skip only kicks in for
    // the intersection-parameter path, not the same-side guard, so the
    // segment is rejected before reaching the `dv` branch.
    const p1 = new Float32Array([0, 0, 0, 10]);
    const p2 = new Float32Array([10, 10, 10, 10]); // both at hidden=10
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);
    expect(result[0]).toBe(0.0); // not visible
  });
});

// [wasm.md/G9][P5] Case D: both endpoints OUT but on OPPOSITE sides of the
// slice; the segment must cross the slice and be visible with t1>0 / t2<1.
// The docstring's 5-case enumeration listed this case but no test previously
// asserted it for `clip_segment_single`. A mutant that mis-classified
// opposite-sides as same-side (Case E) would silently hide every line
// crossing the slice.
describe('clip_segment_single — Case D opposite-sides crossing', () => {
  it('endpoints on opposite sides of a hidden dim emit clipped t1, t2 in (0,1)', () => {
    // 4D: dim 3 hidden, slice center at 0 with tol 0.5 → slice ∈ [-0.5, 0.5].
    // Endpoints at dim3 = -2 (below) and dim3 = +2 (above). Linearly,
    // the segment hits sliceMin at t = (−0.5 − (−2)) / (2 − (−2)) = 1.5/4 = 0.375
    // and sliceMax at t = (0.5 − (−2)) / 4 = 2.5/4 = 0.625.
    const p1 = new Float32Array([0, 0, 0, -2]);
    const p2 = new Float32Array([10, 10, 10, 2]);
    const slicePos = new Float32Array([0, 0, 0, 0]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);
    expect(result[0]).toBe(1.0); // visible
    expect(result[1]).toBeCloseTo(0.375, 5);
    expect(result[2]).toBeCloseTo(0.625, 5);
    // Both clipping flags should fire: t1 > 0 AND t2 < 1.
    expect(result[1]).toBeGreaterThan(0);
    expect(result[2]).toBeLessThan(1);
  });

  it('asymmetric opposite-sides crossing — close-to-slice endpoint clips later', () => {
    // p1 just below slice min, p2 far above. The exit t (t2) lands very
    // close to the start of the segment; the entry t (t1) is at the very
    // start (clamped to 0 since p1 is already below the slice min).
    const p1 = new Float32Array([0, 0, 0, -0.51]); // just below sliceMin=-0.5
    const p2 = new Float32Array([10, 10, 10, 100]);
    const slicePos = new Float32Array([0, 0, 0, 0]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);
    expect(result[0]).toBe(1.0);
    // t1 = (sliceMin − v1) / dv = (−0.5 − (−0.51)) / 100.51 = 0.01/100.51 ≈ 9.95e-5
    expect(result[1]).toBeGreaterThan(0);
    expect(result[1]).toBeLessThan(0.001);
    // t2 = (sliceMax − v1) / dv = (0.5 − (−0.51)) / 100.51 ≈ 0.01005
    expect(result[2]).toBeGreaterThan(result[1]);
    expect(result[2]).toBeLessThan(0.02);
  });
});

// [wasm.md/G9][P5] Zero-length segment (p1 === p2 in every dim). All dv=0;
// the parallel-skip guard runs in every hidden dim. If both endpoints sit
// inside the slice, the segment is fully visible; if both sit outside on
// the same side, it's invisible (Case E). The corner case where p1 == p2
// is not directly enumerated in the 5-case docstring but exercises every
// parallel branch.
describe('clip_segment_single — zero-length segment', () => {
  it('p1 === p2 inside the slice: visible with t1=0, t2=1', () => {
    const p1 = new Float32Array([1, 2, 3, 5]);
    const p2 = new Float32Array([1, 2, 3, 5]); // identical to p1
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);
    expect(result[0]).toBe(1.0);
    expect(result[1]).toBe(0);
    expect(result[2]).toBe(1);
  });

  it('p1 === p2 outside the slice in a hidden dim: invisible (Case E)', () => {
    const p1 = new Float32Array([1, 2, 3, 100]);
    const p2 = new Float32Array([1, 2, 3, 100]); // both far from slice
    const slicePos = new Float32Array([0, 0, 0, 0]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const result = clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);
    expect(result[0]).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// [wasm.md G6] decode round-trip (quantize → decode ≈ x within step).
// The encoder is exercised by python tests; the TS-fallback decoder is
// tested in isolation. This block pins the round-trip invariant for both
// uint8 and uint16 encodings. A swapped `(maxVal-minVal)/N` vs `N/(maxVal-
// minVal)` in either direction would survive each-direction-only tests but
// fail here.
// ---------------------------------------------------------------------------
describe('decode_quantized round-trip [wasm.md G6]', () => {
  // Inverse encoders mirroring the documented mapping in decode.ts:11/27.
  function quantize_u8(values: Float32Array, minVal: number, maxVal: number): Uint8Array {
    const inv = 255 / (maxVal - minVal);
    const out = new Uint8Array(values.length);
    for (let i = 0; i < values.length; i++) {
      out[i] = Math.max(0, Math.min(255, Math.round((values[i] - minVal) * inv)));
    }
    return out;
  }
  function quantize_u16(values: Float32Array, minVal: number, maxVal: number): Uint16Array {
    const inv = 65535 / (maxVal - minVal);
    const out = new Uint16Array(values.length);
    for (let i = 0; i < values.length; i++) {
      out[i] = Math.max(0, Math.min(65535, Math.round((values[i] - minVal) * inv)));
    }
    return out;
  }

  it('[G6] u8: decode(quantize(x)) lands within one quantization step of x', () => {
    const minVal = 0;
    const maxVal = 10;
    const step = (maxVal - minVal) / 255;
    const values = new Float32Array([0, 1.234, 5.0, 7.777, 9.999]);
    const encoded = quantize_u8(values, minVal, maxVal);
    const decoded = new Float32Array(values.length);
    decode_quantized_u8(encoded, minVal, maxVal, decoded);
    for (let i = 0; i < values.length; i++) {
      expect(Math.abs(decoded[i] - values[i])).toBeLessThanOrEqual(step);
    }
  });

  it('[G6] u16: decode(quantize(x)) lands within one (tighter) quantization step of x', () => {
    const minVal = -3.14;
    const maxVal = 2.71;
    const step = (maxVal - minVal) / 65535;
    const values = new Float32Array([-3.14, -1.0, 0.0, 1.0, 2.71]);
    const encoded = quantize_u16(values, minVal, maxVal);
    const decoded = new Float32Array(values.length);
    decode_quantized_u16(encoded, minVal, maxVal, decoded);
    for (let i = 0; i < values.length; i++) {
      expect(Math.abs(decoded[i] - values[i])).toBeLessThanOrEqual(step);
    }
  });

  it('[G6] u8: decode is monotonic in input bytes (mutation-killer for sign-flipped scale)', () => {
    // 0 → minVal, 255 → maxVal. A sign-flipped `(minVal - maxVal)/255` would
    // produce a decreasing function — caught here.
    const out = new Float32Array(3);
    decode_quantized_u8(new Uint8Array([0, 128, 255]), -1, 1, out);
    expect(out[0]).toBeLessThan(out[1]);
    expect(out[1]).toBeLessThan(out[2]);
    expect(out[0]).toBeCloseTo(-1, 5);
    expect(out[2]).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// [wasm.md G7] LUT-decode out-of-bounds index and boundary cases.
// decode_lut_scalar_* and decode_lut_row_* perform no bounds-checking on
// the index — a stale or corrupt index would read past the LUT and store
// `undefined`, which Float32Array stores as 0. Pin this documented (or
// implicit) behaviour.
// ---------------------------------------------------------------------------
describe('decode_lut OOB and empty boundaries [wasm.md G7]', () => {
  // Note: the audit hypothesised that Float32Array stores `undefined` as 0,
  // but JS actually coerces `undefined → NaN` on assignment to a numeric
  // typed array (TypedArray spec: ToNumber(undefined) = NaN, then ToFloat32(NaN)
  // = NaN). So OOB LUT reads silently produce NaN. Pin this contract so any
  // future hardening (e.g. throw on OOB, clamp to 0) surfaces as intentional.
  it('[G7] scalar_u8: index >= lut.length reads undefined → Float32 stores NaN', () => {
    const lut = new Float32Array([10, 20, 30]);
    const indices = new Uint8Array([0, 5, 1]); // index 5 is OOB for lut.length=3
    const out = new Float32Array(3);
    decode_lut_scalar_u8(indices, lut, out);
    expect(out[0]).toBe(10);
    expect(Number.isNaN(out[1])).toBe(true);
    expect(out[2]).toBe(20);
  });

  it('[G7] scalar_u16: OOB symmetric to u8 path', () => {
    const lut = new Float32Array([10, 20, 30]);
    const indices = new Uint16Array([0, 50000, 1]); // way OOB
    const out = new Float32Array(3);
    decode_lut_scalar_u16(indices, lut, out);
    expect(out[0]).toBe(10);
    expect(Number.isNaN(out[1])).toBe(true);
    expect(out[2]).toBe(20);
  });

  it('[G7] row_u8: index*k + j >= lut.length reads undefined → row filled with NaN past boundary', () => {
    // k=2, lut has 4 floats (rows 0..1). indices=[0, 1, 2]. Index 2 is OOB
    // (row 2 would start at lut[4]); both slots become NaN.
    const lut = new Float32Array([1, 2, 3, 4]);
    const indices = new Uint8Array([0, 1, 2]);
    const out = new Float32Array(6);
    decode_lut_row_u8(indices, lut, 2, out);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(2);
    expect(out[2]).toBe(3);
    expect(out[3]).toBe(4);
    expect(Number.isNaN(out[4])).toBe(true);
    expect(Number.isNaN(out[5])).toBe(true);
  });

  it('[G7] row_u16: parallel to u8 path', () => {
    const lut = new Float32Array([1, 2, 3, 4]);
    const indices = new Uint16Array([0, 1, 30000]);
    const out = new Float32Array(6);
    decode_lut_row_u16(indices, lut, 2, out);
    expect(out[0]).toBe(1);
    expect(out[3]).toBe(4);
    expect(Number.isNaN(out[4])).toBe(true);
    expect(Number.isNaN(out[5])).toBe(true);
  });

  it('[G7] empty lut (lut.length === 0): every index reads OOB → all-NaN output', () => {
    const lut = new Float32Array(0);
    const indices = new Uint8Array([0, 1, 2]);
    const out = new Float32Array(3);
    decode_lut_scalar_u8(indices, lut, out);
    for (let i = 0; i < 3; i++) expect(Number.isNaN(out[i])).toBe(true);
  });

  it('[G7] k === 0 (row-mode boundary): inner loop never executes, output untouched', () => {
    // k=0 means each row has zero floats. Pre-fill out with sentinels; verify
    // unchanged after decode_lut_row_u8 (the inner `for j < 0` is empty).
    const lut = new Float32Array([1, 2, 3]);
    const indices = new Uint8Array([0, 1, 2]);
    const out = new Float32Array([99, 99, 99]);
    decode_lut_row_u8(indices, lut, 0, out);
    expect(Array.from(out)).toEqual([99, 99, 99]);
  });
});

// ---------------------------------------------------------------------------
// [wasm.md G15] lerp_vec3 boundary and extrapolation coverage.
// Prior round only had a happy-path t=0.5 test. Pin t=0/t=1 endpoints and
// the algebraic extrapolation behaviour (t outside [0,1] is supported).
// ---------------------------------------------------------------------------
describe('lerp_vec3 boundary and extrapolation [wasm.md G15]', () => {
  it('[G15] t=0 returns a (exactly, no FP drift)', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([10, 20, 30]);
    const result = lerp_vec3(a, b, 0);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('[G15] t=1 returns b (exactly)', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([10, 20, 30]);
    const result = lerp_vec3(a, b, 1);
    // `a + 1 * (b - a) = a + b - a = b`; Float32 round-trip should hit b exactly.
    expect(result[0]).toBeCloseTo(10, 5);
    expect(result[1]).toBeCloseTo(20, 5);
    expect(result[2]).toBeCloseTo(30, 5);
  });

  it('[G15] t < 0 extrapolates BEHIND a (away from b)', () => {
    // t=-1: result = a + (-1) * (b-a) = 2a - b. For a=(0,0,0), b=(2,4,6):
    // result = (0,0,0) - (2,4,6) = (-2,-4,-6).
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([2, 4, 6]);
    const result = lerp_vec3(a, b, -1);
    expect(result[0]).toBeCloseTo(-2, 5);
    expect(result[1]).toBeCloseTo(-4, 5);
    expect(result[2]).toBeCloseTo(-6, 5);
  });

  it('[G15] t > 1 extrapolates PAST b (away from a)', () => {
    // t=2: result = a + 2 * (b-a) = 2b - a. For a=(0,0,0), b=(1,1,1):
    // result = (2,2,2).
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([1, 1, 1]);
    const result = lerp_vec3(a, b, 2);
    expect(result[0]).toBeCloseTo(2, 5);
    expect(result[1]).toBeCloseTo(2, 5);
    expect(result[2]).toBeCloseTo(2, 5);
  });

  it('[G15] linearity in t: lerp(a, b, t1+t2) = a + (t1+t2)(b-a) equals 2*lerp(a,b,(t1+t2)/2) - a', () => {
    // Property-style check on a single fixture. With a=(0,0,0), b=(10,10,10),
    // lerp(t=0.3) + lerp(t=0.7) should equal lerp(t=1.0) + lerp(t=0) = b + a = b.
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([10, 10, 10]);
    const r03 = lerp_vec3(a, b, 0.3);
    const r07 = lerp_vec3(a, b, 0.7);
    for (let i = 0; i < 3; i++) {
      expect(r03[i] + r07[i]).toBeCloseTo(b[i] + a[i], 5);
    }
  });
});
