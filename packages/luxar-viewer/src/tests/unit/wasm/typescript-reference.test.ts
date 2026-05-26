/**
 * WASM vs TypeScript Comparison Tests
 *
 * These tests verify that:
 * 1. TypeScript reference implementations are correct
 * 2. WASM implementations produce identical results to TypeScript
 *
 * Strategy:
 * - TypeScript implementations serve as the "ground truth"
 * - Each function is tested with multiple inputs covering edge cases
 * - When WASM is available (browser E2E), outputs must match exactly
 *
 * Structure mirrors src/wasm/rust/src/ and src/wasm/typescript/:
 * - spatial tests     -> spatial.rs / spatial.ts
 * - points tests      -> points.rs / points.ts
 * - lines tests       -> lines.rs / lines.ts
 * - gsplats tests     -> gsplats.rs / gsplats.ts
 * - effective_radii tests -> effective_radii.rs / effective-radii.ts
 */

import { describe, it, expect } from 'vitest';
import {
  compute_nd_visibility_gsplats,
  decode_quantized_u8,
  decode_quantized_u16,
  decode_log_scalar_u8,
  decode_log_scalar_u16,
  decode_lut_scalar_u8,
  decode_lut_scalar_u16,
  decode_lut_row_u8,
  decode_lut_row_u16,
  decode_broadcasted,
  clip_segment_single,
  clip_segments_batch,
  interpolate_clipped_positions,
  lerp,
  lerp_vec3,
  distance_3d,
  interpolate_scalars_batch,
  interpolate_colors_batch,
  calculate_segment_lengths,
  mark_clipped_endpoints,
} from '../../../wasm/typescript';

describe('TypeScript Reference Implementation Tests', () => {
  // SPATIAL tests moved to ./typescript-reference/spatial.test.ts (wasm.md O1).

  // POINTS tests moved to ./typescript-reference/points.test.ts (wasm.md O1).

  // LINES tests moved to ./typescript-reference/lines.test.ts (wasm.md O1).

  // ============================================================================
  // GSPLATS TESTS (ellipsoid visibility)
  // ============================================================================
  describe('gsplats: compute_nd_visibility_gsplats', () => {
    it('should filter splats based on center proximity and ellipsoid extent', () => {
      // 3D test matching Rust test: gsplats.rs::test_gsplat_visibility_basic
      const ndim = 3;
      const numSplats = 2;

      const centers = new Float32Array([
        0.0,
        0.0,
        0.0, // Splat 0 at origin
        10.0,
        10.0,
        10.0, // Splat 1 far away
      ]);
      // 3D cholesky: 6 elements per splat [L00, L10, L11, L20, L21, L22]
      // Using identity-ish (1.0 on diagonals)
      const choleskyFactors = new Float32Array([
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0, // Splat 0
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0, // Splat 1
      ]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0]);
      const tolerance = new Float32Array([2.0, 2.0, 2.0]);

      const output = new Uint8Array(numSplats);
      const count = compute_nd_visibility_gsplats(
        centers,
        choleskyFactors,
        slicePos,
        tolerance,
        ndim,
        numSplats,
        output
      );

      expect(output[0]).toBe(1); // Splat 0 visible (at origin)
      expect(output[1]).toBe(0); // Splat 1 hidden (far away)
      expect(count).toBe(1);
    });

    it('should handle 4D splats with hidden time dimension', () => {
      // 4D test matching Rust test: gsplats.rs::test_gsplat_visibility_4d
      const ndim = 4;
      const numSplats = 2;

      const centers = new Float32Array([
        0.0,
        0.0,
        0.0,
        0.0, // Splat 0 at t=0
        0.0,
        0.0,
        0.0,
        10.0, // Splat 1 at t=10
      ]);
      // 4D cholesky: 10 elements per splat
      const choleskyFactors = new Float32Array([
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0, // Splat 0
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0, // Splat 1
      ]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0]);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 2.0]); // Infinite for XYZ, 2.0 for T

      const output = new Uint8Array(numSplats);
      const count = compute_nd_visibility_gsplats(
        centers,
        choleskyFactors,
        slicePos,
        tolerance,
        ndim,
        numSplats,
        output
      );

      expect(output[0]).toBe(1); // Splat 0 visible (t=0)
      expect(output[1]).toBe(0); // Splat 1 hidden (t=10 > tolerance+extent)
      expect(count).toBe(1);
    });
  });

  // EFFECTIVE_RADII tests moved to ./typescript-reference/effective-radii.test.ts (wasm.md O1).

  // ============================================================================
  // DECODE TESTS (quantized, LUT, log-space)
  // ============================================================================
  describe('decode: quantized functions', () => {
    it('should decode uint8 linear quantization', () => {
      const data = new Uint8Array([0, 128, 255]);
      const output = new Float32Array(3);

      decode_quantized_u8(data, 0.0, 10.0, output);

      expect(output[0]).toBeCloseTo(0.0, 2);
      expect(output[1]).toBeCloseTo(5.02, 1); // 128/255 * 10
      expect(output[2]).toBeCloseTo(10.0, 2);
    });

    it('should decode uint16 linear quantization', () => {
      const data = new Uint16Array([0, 32768, 65535]);
      const output = new Float32Array(3);

      decode_quantized_u16(data, -1.0, 1.0, output);

      expect(output[0]).toBeCloseTo(-1.0, 2);
      expect(output[1]).toBeCloseTo(0.0, 2);
      expect(output[2]).toBeCloseTo(1.0, 2);
    });

    it('should decode log-space quantization', () => {
      const data = new Uint8Array([0, 128, 255]);
      const maxLog = 5.0;
      const output = new Float32Array(3);

      decode_log_scalar_u8(data, maxLog, output);

      // 0 -> expm1(0) = 0
      expect(output[0]).toBeCloseTo(0, 2);
      // 128 -> expm1(128/255 * 5) ≈ expm1(2.51) ≈ 11.3
      expect(output[1]).toBeGreaterThan(10);
      expect(output[1]).toBeLessThan(13);
      // 255 -> expm1(5) ≈ 147.4
      expect(output[2]).toBeCloseTo(Math.expm1(5), 1);
    });

    it('should decode log-space quantization uint16', () => {
      const data = new Uint16Array([0, 32768, 65535]);
      const maxLog = 5.0;
      const output = new Float32Array(3);

      decode_log_scalar_u16(data, maxLog, output);

      // 0 -> expm1(0) = 0
      expect(output[0]).toBeCloseTo(0, 2);
      // 32768 -> expm1(32768/65535 * 5) ≈ expm1(2.5) ≈ 11.2
      expect(output[1]).toBeGreaterThan(10);
      expect(output[1]).toBeLessThan(13);
      // 65535 -> expm1(5) ≈ 147.4
      expect(output[2]).toBeCloseTo(Math.expm1(5), 1);
    });
  });

  describe('decode: LUT functions', () => {
    it('should decode scalar LUT indices uint8', () => {
      const indices = new Uint8Array([0, 2, 1]);
      const lut = new Float32Array([1.0, 2.0, 3.0]);
      const output = new Float32Array(3);

      decode_lut_scalar_u8(indices, lut, output);

      expect(output[0]).toBe(1.0);
      expect(output[1]).toBe(3.0);
      expect(output[2]).toBe(2.0);
    });

    it('should decode scalar LUT indices uint16', () => {
      const indices = new Uint16Array([0, 2, 1]);
      const lut = new Float32Array([1.0, 2.0, 3.0]);
      const output = new Float32Array(3);

      decode_lut_scalar_u16(indices, lut, output);

      expect(output[0]).toBe(1.0);
      expect(output[1]).toBe(3.0);
      expect(output[2]).toBe(2.0);
    });

    it('should decode row LUT indices (vector attributes) uint8', () => {
      const indices = new Uint8Array([0, 1]);
      // LUT with 2 entries, each with 3 values (rgb)
      const lut = new Float32Array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]); // red, green
      const output = new Float32Array(6);

      decode_lut_row_u8(indices, lut, 3, output);

      // First row: red
      expect(output[0]).toBe(1.0);
      expect(output[1]).toBe(0.0);
      expect(output[2]).toBe(0.0);
      // Second row: green
      expect(output[3]).toBe(0.0);
      expect(output[4]).toBe(1.0);
      expect(output[5]).toBe(0.0);
    });

    it('should decode row LUT indices (vector attributes) uint16', () => {
      const indices = new Uint16Array([0, 1]);
      // LUT with 2 entries, each with 3 values (rgb)
      const lut = new Float32Array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]); // red, green
      const output = new Float32Array(6);

      decode_lut_row_u16(indices, lut, 3, output);

      // First row: red
      expect(output[0]).toBe(1.0);
      expect(output[1]).toBe(0.0);
      expect(output[2]).toBe(0.0);
      // Second row: green
      expect(output[3]).toBe(0.0);
      expect(output[4]).toBe(1.0);
      expect(output[5]).toBe(0.0);
    });
  });

  describe('decode: broadcast function', () => {
    it('should broadcast scalar to all points', () => {
      const value = new Float32Array([0.5]);
      const output = new Float32Array(5);

      decode_broadcasted(value, 5, 1, output);

      for (let i = 0; i < 5; i++) {
        expect(output[i]).toBe(0.5);
      }
    });

    it('should broadcast vector to all points', () => {
      const value = new Float32Array([0.5, 0.6, 0.7]); // rgb
      const output = new Float32Array(9); // 3 points * 3 elements

      decode_broadcasted(value, 3, 3, output);

      for (let i = 0; i < 3; i++) {
        expect(output[i * 3]).toBeCloseTo(0.5, 5);
        expect(output[i * 3 + 1]).toBeCloseTo(0.6, 5);
        expect(output[i * 3 + 2]).toBeCloseTo(0.7, 5);
      }
    });

    // Regression: MED-19 — reject ambiguous middle-length inputs. Previously
    // `value.length=2, elementsPerPoint=3` would silently produce a row of
    // `[v0, v1, v0]` (mixed broadcast). Now it throws.
    it('should throw on ambiguous value.length between 1 and elementsPerPoint', () => {
      const value = new Float32Array([0.1, 0.2]); // length 2
      const output = new Float32Array(9); // 3 points * 3 elements

      expect(() => decode_broadcasted(value, 3, 3, output)).toThrow(
        /value\.length must be 1.*or elementsPerPoint \(3\), got 2/
      );
    });

    it('should throw on value.length greater than elementsPerPoint', () => {
      const value = new Float32Array([0.1, 0.2, 0.3, 0.4]); // length 4
      const output = new Float32Array(6); // 2 points * 3 elements

      expect(() => decode_broadcasted(value, 2, 3, output)).toThrow(
        /value\.length must be 1.*or elementsPerPoint \(3\), got 4/
      );
    });
  });

  // PROJECTION tests moved to ./typescript-reference/projection.test.ts (wasm.md O1).

  // GSPLATS_PROCESSING tests moved to ./typescript-reference/gsplats-processing.test.ts (wasm.md O1).

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
  });

  describe('lines_clipping: mark_clipped_endpoints', () => {
    it('should mark clipped endpoints correctly', () => {
      const visibility = new Uint8Array([1, 1, 0, 1]);
      const t1Params = new Float32Array([0.0, 0.5, 0.0, 0.25]);
      const t2Params = new Float32Array([1.0, 0.75, 1.0, 1.0]);

      const startClipped = new Uint8Array(3);
      const endClipped = new Uint8Array(3);

      const count = mark_clipped_endpoints(
        visibility,
        t1Params,
        t2Params,
        4,
        startClipped,
        endClipped
      );

      expect(count).toBe(3); // 3 visible
      // seg0: t1=0 (not clipped), t2=1 (not clipped)
      expect(startClipped[0]).toBe(0);
      expect(endClipped[0]).toBe(0);
      // seg1: t1=0.5 (clipped), t2=0.75 (clipped)
      expect(startClipped[1]).toBe(1);
      expect(endClipped[1]).toBe(1);
      // seg3: t1=0.25 (clipped), t2=1.0 (not clipped)
      expect(startClipped[2]).toBe(1);
      expect(endClipped[2]).toBe(0);
    });
  });

});

// wasm.md O2/O13: helpers (`arraysEqual`, `arraysAlmostEqual`,
// `generateRandomPoints`) moved to `src/tests/helpers/array-compare.ts`
// — they were unused inside this file and no other test imported them
// here. New code should import directly from that helper module.
