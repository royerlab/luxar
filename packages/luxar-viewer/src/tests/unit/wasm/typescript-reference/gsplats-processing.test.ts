/**
 * Tests for `src/wasm/typescript/gsplats_processing.ts`
 * (Mahalanobis distance, Cholesky submatrix extraction, marginal Cholesky,
 * gsplat attenuation, visible-cholesky compaction, attenuated-amplitude compaction).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D8): the 1714-line mega-file
 * is being split into per-source-module test files mirroring
 * `src/wasm/typescript/`.
 */

import { describe, it, expect } from 'vitest';
import {
  mahalanobis_distance,
  extract_cholesky_submatrix,
  computeMarginalCholesky,
  compute_gsplats_attenuation,
  extract_visible_cholesky_3d,
  compact_attenuated_amplitudes,
} from '../../../../wasm/typescript';

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
