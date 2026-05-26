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
  decode_quantized_u8,
  decode_quantized_u16,
  decode_log_scalar_u8,
  decode_log_scalar_u16,
  decode_lut_scalar_u8,
  decode_lut_scalar_u16,
  decode_lut_row_u8,
  decode_lut_row_u16,
  decode_broadcasted,
  extract_3d_positions,
  calculate_bounds_3d,
  compact_by_mask,
  count_visible,
  radii_to_visibility_mask,
  mahalanobis_distance,
  extract_cholesky_submatrix,
  computeMarginalCholesky,
  compute_gsplats_attenuation,
  extract_visible_cholesky_3d,
  compact_attenuated_amplitudes,
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

  // GSPLATS tests moved to ./typescript-reference/gsplats.test.ts (wasm.md O1).

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

  // ============================================================================
  // PROJECTION TESTS (nD to 3D projection and bounds)
  // ============================================================================
  describe('projection: extract_3d_positions', () => {
    it('should extract 3D positions from 5D data', () => {
      // 5D data: 2 points
      const positionsNd = new Float32Array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0, // point 0
        6.0,
        7.0,
        8.0,
        9.0,
        10.0, // point 1
      ]);
      const displayDims = new Uint32Array([0, 2, 4]); // X=dim0, Y=dim2, Z=dim4
      const output = new Float32Array(6);

      extract_3d_positions(positionsNd, displayDims, 5, 2, output);

      expect(output[0]).toBe(1.0); // point0.x = dim0
      expect(output[1]).toBe(3.0); // point0.y = dim2
      expect(output[2]).toBe(5.0); // point0.z = dim4
      expect(output[3]).toBe(6.0); // point1.x = dim0
      expect(output[4]).toBe(8.0); // point1.y = dim2
      expect(output[5]).toBe(10.0); // point1.z = dim4
    });

    it('should handle 2D display (z=0)', () => {
      // 4D data displayed as 2D
      const positionsNd = new Float32Array([1.0, 2.0, 3.0, 4.0]);
      const displayDims = new Uint32Array([0, 1]); // Only X and Y
      const output = new Float32Array(3);

      extract_3d_positions(positionsNd, displayDims, 4, 1, output);

      expect(output[0]).toBe(1.0); // x
      expect(output[1]).toBe(2.0); // y
      expect(output[2]).toBe(0.0); // z = 0 (default)
    });

    it('should handle reordered dimensions', () => {
      const positionsNd = new Float32Array([10.0, 20.0, 30.0]);
      const displayDims = new Uint32Array([2, 0, 1]); // Z->X, X->Y, Y->Z
      const output = new Float32Array(3);

      extract_3d_positions(positionsNd, displayDims, 3, 1, output);

      expect(output[0]).toBe(30.0); // x = dim2
      expect(output[1]).toBe(10.0); // y = dim0
      expect(output[2]).toBe(20.0); // z = dim1
    });
  });

  describe('projection: calculate_bounds_3d', () => {
    it('should calculate correct bounds', () => {
      const positions = new Float32Array([
        -1.0,
        2.0,
        3.0, // point 0
        4.0,
        -5.0,
        6.0, // point 1
        7.0,
        8.0,
        -9.0, // point 2
      ]);
      const output = new Float32Array(6);

      const count = calculate_bounds_3d(positions, 3, output);

      expect(count).toBe(3);
      expect(output[0]).toBe(-1.0); // min_x
      expect(output[1]).toBe(-5.0); // min_y
      expect(output[2]).toBe(-9.0); // min_z
      expect(output[3]).toBe(7.0); // max_x
      expect(output[4]).toBe(8.0); // max_y
      expect(output[5]).toBe(6.0); // max_z
    });

    it('should handle empty input', () => {
      const output = new Float32Array(6).fill(999);

      const count = calculate_bounds_3d(new Float32Array(0), 0, output);

      expect(count).toBe(0);
      for (let i = 0; i < 6; i++) {
        expect(output[i]).toBe(0);
      }
    });

    it('should handle single point', () => {
      const positions = new Float32Array([1.5, 2.5, 3.5]);
      const output = new Float32Array(6);

      const count = calculate_bounds_3d(positions, 1, output);

      expect(count).toBe(1);
      expect(output[0]).toBe(1.5); // min_x = max_x
      expect(output[1]).toBe(2.5); // min_y = max_y
      expect(output[2]).toBe(3.5); // min_z = max_z
      expect(output[3]).toBe(1.5);
      expect(output[4]).toBe(2.5);
      expect(output[5]).toBe(3.5);
    });
  });

  describe('projection: compact_by_mask', () => {
    it('should compact arrays by visibility mask', () => {
      const input = new Float32Array([
        1.0,
        2.0,
        3.0, // visible
        4.0,
        5.0,
        6.0, // hidden
        7.0,
        8.0,
        9.0, // visible
      ]);
      const mask = new Uint8Array([1, 0, 1]);
      const output = new Float32Array(6);

      const visible = compact_by_mask(input, mask, 3, 3, output);

      expect(visible).toBe(2);
      expect(output[0]).toBe(1.0);
      expect(output[1]).toBe(2.0);
      expect(output[2]).toBe(3.0);
      expect(output[3]).toBe(7.0);
      expect(output[4]).toBe(8.0);
      expect(output[5]).toBe(9.0);
    });

    it('should handle scalar stride', () => {
      const input = new Float32Array([10, 20, 30, 40, 50]);
      const mask = new Uint8Array([1, 0, 1, 0, 1]);
      const output = new Float32Array(3);

      const visible = compact_by_mask(input, mask, 5, 1, output);

      expect(visible).toBe(3);
      expect(output[0]).toBe(10);
      expect(output[1]).toBe(30);
      expect(output[2]).toBe(50);
    });

    it('should handle all hidden', () => {
      const input = new Float32Array([1, 2, 3]);
      const mask = new Uint8Array([0, 0, 0]);
      const output = new Float32Array(3);

      const visible = compact_by_mask(input, mask, 3, 1, output);

      expect(visible).toBe(0);
    });
  });

  describe('projection: count_visible', () => {
    it('should count non-zero mask values', () => {
      const mask = new Uint8Array([1, 0, 1, 0, 1, 1]);
      expect(count_visible(mask, 6)).toBe(4);
    });

    it('should handle all visible', () => {
      const mask = new Uint8Array([1, 1, 1]);
      expect(count_visible(mask, 3)).toBe(3);
    });

    it('should handle all hidden', () => {
      const mask = new Uint8Array([0, 0, 0, 0]);
      expect(count_visible(mask, 4)).toBe(0);
    });

    it('should handle empty', () => {
      expect(count_visible(new Uint8Array(0), 0)).toBe(0);
    });
  });

  describe('projection: radii_to_visibility_mask', () => {
    it('should create mask from radii threshold', () => {
      const radii = new Float32Array([0.5, 0.0001, 0.2, 0.0]);
      const output = new Uint8Array(4);

      const visible = radii_to_visibility_mask(radii, 0.0001, 4, output);

      expect(visible).toBe(2); // 0.5 and 0.2 > 0.0001
      expect(output[0]).toBe(1);
      expect(output[1]).toBe(0); // exactly at threshold = not visible
      expect(output[2]).toBe(1);
      expect(output[3]).toBe(0);
    });

    it('should handle zero threshold', () => {
      const radii = new Float32Array([0.001, 0.0, 0.5]);
      const output = new Uint8Array(3);

      const visible = radii_to_visibility_mask(radii, 0, 3, output);

      expect(visible).toBe(2);
      expect(output[0]).toBe(1);
      expect(output[1]).toBe(0); // 0 is not > 0
      expect(output[2]).toBe(1);
    });

    it('should handle large threshold (all hidden)', () => {
      const radii = new Float32Array([1.0, 2.0, 3.0]);
      const output = new Uint8Array(3);

      const visible = radii_to_visibility_mask(radii, 100, 3, output);

      expect(visible).toBe(0);
      expect(output[0]).toBe(0);
      expect(output[1]).toBe(0);
      expect(output[2]).toBe(0);
    });
  });

  // ============================================================================
  // GSPLATS PROCESSING TESTS (Mahalanobis distance, Cholesky extraction)
  // ============================================================================
  describe('gsplats_processing: mahalanobis_distance', () => {
    it('should compute Euclidean distance with identity Cholesky', () => {
      // Identity Cholesky (L = I): Mahalanobis = Euclidean
      // 3D: packed = [1, 0, 1, 0, 0, 1]
      const packedL = new Float32Array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
      const diff = new Float32Array([3.0, 4.0, 0.0]); // Distance should be 5.0

      const dist = mahalanobis_distance(diff, packedL, 3);
      expect(dist).toBeCloseTo(5.0, 5);
    });

    it('should scale distance with scaled Cholesky', () => {
      // Scaled Cholesky: L = diag(2, 2, 2)
      // packed = [2, 0, 2, 0, 0, 2]
      // Mahalanobis = ||L⁻¹ · diff|| = ||diff / 2||
      const packedL = new Float32Array([2.0, 0.0, 2.0, 0.0, 0.0, 2.0]);
      const diff = new Float32Array([4.0, 0.0, 0.0]); // Mahalanobis should be 4/2 = 2

      const dist = mahalanobis_distance(diff, packedL, 3);
      expect(dist).toBeCloseTo(2.0, 5);
    });

    it('should handle 2D case', () => {
      // 2D identity: packed = [1, 0, 1]
      const packedL = new Float32Array([1.0, 0.0, 1.0]);
      const diff = new Float32Array([3.0, 4.0]);

      const dist = mahalanobis_distance(diff, packedL, 2);
      expect(dist).toBeCloseTo(5.0, 5);
    });
  });

  describe('gsplats_processing: extract_cholesky_submatrix', () => {
    it('should extract 2D submatrix from 4D Cholesky', () => {
      // 4D Cholesky: 10 elements
      // [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
      const packed = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

      // Extract dims [0, 2] (2D submatrix)
      const keepDims = new Uint32Array([0, 2]);
      const output = new Float32Array(3); // 2D packed = 3 elements

      extract_cholesky_submatrix(packed, keepDims, 2, output);

      // Expected: [L00, L20, L22] = [1.0, 4.0, 6.0]
      expect(output[0]).toBe(1.0); // L[0,0]
      expect(output[1]).toBe(4.0); // L[2,0]
      expect(output[2]).toBe(6.0); // L[2,2]
    });

    it('should extract 3D submatrix from 5D Cholesky', () => {
      // 5D Cholesky: 15 elements
      // Build with identity-like values for easy verification
      const packed = new Float32Array(15);
      for (let i = 0; i < 15; i++) {
        packed[i] = i + 1;
      }

      // Extract dims [0, 1, 2] (first 3 dims)
      const keepDims = new Uint32Array([0, 1, 2]);
      const output = new Float32Array(6); // 3D packed = 6 elements

      extract_cholesky_submatrix(packed, keepDims, 3, output);

      // Should extract [L00, L10, L11, L20, L21, L22] = [1, 2, 3, 4, 5, 6]
      expect(output[0]).toBe(1.0);
      expect(output[1]).toBe(2.0);
      expect(output[2]).toBe(3.0);
      expect(output[3]).toBe(4.0);
      expect(output[4]).toBe(5.0);
      expect(output[5]).toBe(6.0);
    });
  });

  describe('gsplats_processing: compute_gsplats_attenuation', () => {
    it('should have full attenuation with no hidden dimensions', () => {
      // 3D splats with no hidden dimensions
      const positions = new Float32Array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
      const cholesky = new Float32Array([
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
      const amplitudes = new Float32Array([1.0, 0.5]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0]);
      const hiddenDims = new Uint32Array([]); // No hidden dims

      const visibility = new Uint8Array(2);
      const attenuation = new Float32Array(2);

      const count = compute_gsplats_attenuation(
        positions,
        cholesky,
        amplitudes,
        slicePos,
        hiddenDims,
        3,
        2,
        0.1,
        3.0,
        visibility,
        attenuation
      );

      expect(count).toBe(2); // Both visible
      expect(attenuation[0]).toBe(1.0); // No attenuation
      expect(attenuation[1]).toBe(1.0);
    });

    it('should attenuate splats far in hidden dimension', () => {
      // 4D splats with dim 3 as hidden
      const positions = new Float32Array([
        0.0,
        0.0,
        0.0,
        0.0, // Splat 0: at slice
        0.0,
        0.0,
        0.0,
        5.0, // Splat 1: far in hidden dim
      ]);
      // 4D Cholesky: 10 elements, identity
      const cholesky = new Float32Array([
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
      const amplitudes = new Float32Array([1.0, 1.0]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0]);
      const hiddenDims = new Uint32Array([3]); // Dim 3 is hidden

      const visibility = new Uint8Array(2);
      const attenuation = new Float32Array(2);

      const count = compute_gsplats_attenuation(
        positions,
        cholesky,
        amplitudes,
        slicePos,
        hiddenDims,
        4,
        2,
        0.01, // Low threshold
        3.0,
        visibility,
        attenuation
      );

      // Splat 0: mahal = 0, attenuation = 1.0
      expect(visibility[0]).toBe(1);
      expect(attenuation[0]).toBeCloseTo(1.0, 5);

      // Splat 1: mahal = 5.0, beyond 3σ truncation → attenuation ≈ 0
      expect(visibility[1]).toBe(0);
      expect(attenuation[1]).toBeLessThan(0.001);

      expect(count).toBe(1);
    });
  });

  describe('gsplats_processing: extract_visible_cholesky_3d', () => {
    it('should extract 3D Cholesky for visible splats only', () => {
      // 4D Cholesky: 10 elements per splat
      const cholesky = new Float32Array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0, // Splat 0
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0, // Splat 1
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
        26.0,
        27.0,
        28.0,
        29.0,
        30.0, // Splat 2
      ]);
      const visibility = new Uint8Array([1, 0, 1]); // Splats 0 and 2 visible
      const displayDims = new Uint32Array([0, 1, 2]);
      const output = new Float32Array(12); // 2 visible * 6 elements

      const count = extract_visible_cholesky_3d(cholesky, visibility, displayDims, 4, 3, output);

      expect(count).toBe(2);
      // Splat 0: [L00, L10, L11, L20, L21, L22] = [1, 2, 3, 4, 5, 6]
      expect(output[0]).toBe(1.0);
      expect(output[1]).toBe(2.0);
      expect(output[2]).toBe(3.0);
      expect(output[3]).toBe(4.0);
      expect(output[4]).toBe(5.0);
      expect(output[5]).toBe(6.0);
      // Splat 2: [L00, L10, L11, L20, L21, L22] = [21, 22, 23, 24, 25, 26]
      expect(output[6]).toBe(21.0);
      expect(output[7]).toBe(22.0);
      expect(output[8]).toBe(23.0);
      expect(output[9]).toBe(24.0);
      expect(output[10]).toBe(25.0);
      expect(output[11]).toBe(26.0);
    });
  });

  describe('gsplats_processing: compact_attenuated_amplitudes', () => {
    it('should compact amplitudes with attenuation', () => {
      const amplitudes = new Float32Array([1.0, 2.0, 3.0, 4.0]);
      const attenuation = new Float32Array([0.5, 0.25, 0.75, 0.1]);
      const visibility = new Uint8Array([1, 0, 1, 0]);
      const output = new Float32Array(2);

      const count = compact_attenuated_amplitudes(amplitudes, attenuation, visibility, 4, output);

      expect(count).toBe(2);
      expect(output[0]).toBeCloseTo(0.5, 5); // 1.0 * 0.5
      expect(output[1]).toBeCloseTo(2.25, 5); // 3.0 * 0.75
    });
  });

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

  // ============================================================================
  // MARGINAL CHOLESKY TESTS (correlated covariance correctness)
  // ============================================================================
  describe('gsplats_processing: computeMarginalCholesky', () => {
    it('should match raw extraction for diagonal Cholesky', () => {
      // 4D diagonal L: diag(2, 3, 5, 7)
      // Packed: [2, 0,3, 0,0,5, 0,0,0,7]
      const packed = new Float32Array([2, 0, 3, 0, 0, 5, 0, 0, 0, 7]);
      const keepDims = new Uint32Array([0, 2]);
      const output = new Float32Array(3);

      computeMarginalCholesky(packed, 0, keepDims, 2, output, 0);

      // Σ = diag(4,9,25,49), marginal for [0,2] = diag(4,25)
      // Cholesky = diag(2, 5) → packed [2, 0, 5]
      expect(output[0]).toBeCloseTo(2.0, 4);
      expect(output[1]).toBeCloseTo(0.0, 4);
      expect(output[2]).toBeCloseTo(5.0, 4);
    });

    it('should produce correct marginal for correlated Cholesky', () => {
      // 3D L with correlations (matches Rust test):
      // L = [[2, 0, 0], [1, 3, 0], [0.5, 0.5, 4]]
      // Packed: [2, 1,3, 0.5,0.5,4]
      const packed = new Float32Array([2, 1, 3, 0.5, 0.5, 4]);
      const keepDims = new Uint32Array([0, 2]);
      const output = new Float32Array(3);

      computeMarginalCholesky(packed, 0, keepDims, 2, output, 0);

      // Σ_S = [[4, 1], [1, 16.5]]
      // L_S[0,0] = 2, L_S[1,0] = 0.5, L_S[1,1] = sqrt(16.25) ≈ 4.031
      expect(output[0]).toBeCloseTo(2.0, 4);
      expect(output[1]).toBeCloseTo(0.5, 4);
      expect(output[2]).toBeCloseTo(Math.sqrt(16.25), 3);
    });

    it('should differ from raw extraction for correlated Cholesky', () => {
      // Same correlated L as above
      const packed = new Float32Array([2, 1, 3, 0.5, 0.5, 4]);
      const keepDims = new Uint32Array([0, 2]);

      const rawOutput = new Float32Array(3);
      extract_cholesky_submatrix(packed, keepDims, 2, rawOutput);

      const marginalOutput = new Float32Array(3);
      computeMarginalCholesky(packed, 0, keepDims, 2, marginalOutput, 0);

      // Raw gives [L[0,0], L[2,0], L[2,2]] = [2, 0.5, 4]
      expect(rawOutput[2]).toBeCloseTo(4.0, 5);
      // Marginal gives sqrt(16.25) ≈ 4.031 ≠ 4.0
      expect(Math.abs(marginalOutput[2] - rawOutput[2])).toBeGreaterThan(0.01);
    });

    it('should give correct Mahalanobis distance with marginal Cholesky', () => {
      // Same 3D correlated L
      const packed = new Float32Array([2, 1, 3, 0.5, 0.5, 4]);
      const keepDims = new Uint32Array([0, 2]);

      const marginalL = new Float32Array(3);
      computeMarginalCholesky(packed, 0, keepDims, 2, marginalL, 0);

      const diff = new Float32Array([1.0, 0.0]);
      const dist = mahalanobis_distance(diff, marginalL, 2);

      // Forward substitution: y[0]=1/2=0.5, y[1]=(0-0.5*0.5)/4.031≈-0.0621
      // ||y|| ≈ sqrt(0.25 + 0.00386) ≈ 0.504
      expect(dist).toBeCloseTo(0.5, 1);
    });

    it('should produce correct attenuation with correlated Cholesky', () => {
      // 4D splat with correlated L (matches Rust test)
      // L = [[2,0,0,0], [1,3,0,0], [0,0,2,0], [0.5,0.5,0,4]]
      const positions = new Float32Array([0, 0, 0, 0]);
      const cholesky = new Float32Array([2, 1, 3, 0, 0, 2, 0.5, 0.5, 0, 4]);
      const amplitudes = new Float32Array([1.0]);
      const slicePos = new Float32Array([0, 0, 0, 1]); // slice at dim3 = 1
      const hiddenDims = new Uint32Array([3]);

      const visibility = new Uint8Array(1);
      const attenuation = new Float32Array(1);

      compute_gsplats_attenuation(
        positions,
        cholesky,
        amplitudes,
        slicePos,
        hiddenDims,
        4,
        1,
        0.001,
        3.0,
        visibility,
        attenuation
      );

      // Marginal for dim [3]: Σ_33 = 0.25+0.25+0+16 = 16.5
      // L_S = sqrt(16.5) ≈ 4.062
      // Mahalanobis: 1/4.062 ≈ 0.2462
      // Attenuation: exp(-0.5 * 0.2462^2) ≈ 0.970
      expect(attenuation[0]).toBeGreaterThan(0.9);
      expect(attenuation[0]).toBeLessThan(1.0);
      expect(visibility[0]).toBe(1);
    });
  });
});

// wasm.md O2/O13: helpers (`arraysEqual`, `arraysAlmostEqual`,
// `generateRandomPoints`) moved to `src/tests/helpers/array-compare.ts`
// — they were unused inside this file and no other test imported them
// here. New code should import directly from that helper module.
