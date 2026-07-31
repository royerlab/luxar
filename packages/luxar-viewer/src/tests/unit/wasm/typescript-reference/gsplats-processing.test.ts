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
  project_gsplats_nd_to_3d,
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

/**
 * Fewer than 3 display dims (2D / 1D scenes).
 *
 * These live in the TS-only suite ON PURPOSE. The cross-language parity tests in
 * `wasm-vs-typescript.test.ts` are `skipIf(!wasmFilesExist)`, so on any machine or
 * CI leg without built WASM artifacts they vanish — and the TS reference is not
 * merely a fallback, it is the production backend for ndim > 16. Without these,
 * a TS-side regression in the display-marginal padding would ship silently.
 *
 * Contract (mirrors `gsplats_processing.rs::compute_display_cholesky_3d`): the n-D
 * marginal occupies the first n·(n+1)/2 packed slots; the remaining rows get zero
 * off-diagonals and a phantom diagonal equal to the geometric mean of the real
 * Cholesky pivots. The phantom is deliberately NOT an epsilon — an ε-thin splat is
 * invisible in sum/additive blending because the shader scales amplitude by the
 * Gaussian's extent along the view ray.
 */
describe('gsplats_processing: fewer than 3 display dims (TS reference)', () => {
  const SQRT_EPS = Math.sqrt(1e-10);

  it('extract_visible_cholesky_3d pads a 2D marginal with a scale-matched phantom axis', () => {
    const cholesky = new Float32Array([2.0, 0.5, 1.5]); // packed 2D [L00, L10, L11]
    const output = new Float32Array(6);

    const count = extract_visible_cholesky_3d(
      cholesky,
      new Uint8Array([1]),
      new Uint32Array([0, 1]),
      2,
      1,
      output
    );

    expect(count).toBe(1);
    // Keeping ALL dims reproduces the input factor.
    expect(output[0]).toBeCloseTo(2.0, 5);
    expect(output[1]).toBeCloseTo(0.5, 5);
    expect(output[2]).toBeCloseTo(1.5, 5);
    // Phantom row: uncorrelated, scale-matched.
    expect(output[3]).toBe(0);
    expect(output[4]).toBe(0);
    expect(output[5]).toBeCloseTo(Math.sqrt(2.0 * 1.5), 5);
    // The regression this guards: an epsilon here renders the scene black.
    expect(output[5]).toBeGreaterThan(SQRT_EPS * 1000);
  });

  it('project_gsplats_nd_to_3d handles a 2D scene (no OOB read, z-padded centers)', () => {
    const one = [2.0, 0.5, 1.5];
    const splatCount = 2;
    const centers = new Float32Array(splatCount * 3);
    const chol = new Float32Array(splatCount * 6);
    const amps = new Float32Array(splatCount);
    const cols = new Float32Array(splatCount * 3);

    const count = project_gsplats_nd_to_3d(
      new Float32Array([0, 0, 5, -3]),
      new Float32Array([...one, ...one]),
      new Float32Array([1.0, 0.8]),
      new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
      new Uint8Array([1, 1]),
      new Float32Array([0, 0]),
      new Uint32Array([]), // no hidden dims
      new Uint32Array([0, 1]), // 2D scene: two display dims
      2,
      splatCount,
      3,
      1e-6,
      3.0,
      centers,
      chol,
      amps,
      cols
    );

    expect(count).toBe(2);
    // Centers zero-padded in the missing z component.
    expect(Array.from(centers.subarray(0, 6))).toEqual([0, 0, 0, 5, -3, 0]);
    for (let s = 0; s < 2; s++) {
      const c = s * 6;
      expect(chol[c + 3]).toBe(0);
      expect(chol[c + 4]).toBe(0);
      expect(chol[c + 5]).toBeCloseTo(Math.sqrt(2.0 * 1.5), 5);
    }
  });

  it('pads BOTH missing rows for a single display dim', () => {
    const output = new Float32Array(6);
    extract_visible_cholesky_3d(
      new Float32Array([2.0, 0.5, 1.5]),
      new Uint8Array([1]),
      new Uint32Array([0]),
      2,
      1,
      output
    );

    // Marginal over dim 0 alone is [sqrt(L00^2)] = [2]; phantom = 2 on both rows.
    expect(Array.from(output)).toEqual([2, 0, 2, 0, 0, 2]);
  });

  it('keeps the phantom axis proportional to the splat (not a constant)', () => {
    const phantomFor = (scale: number): number => {
      const output = new Float32Array(6);
      extract_visible_cholesky_3d(
        new Float32Array([2.0 * scale, 0.5 * scale, 1.5 * scale]),
        new Uint8Array([1]),
        new Uint32Array([0, 1]),
        2,
        1,
        output
      );
      return output[5];
    };

    expect(phantomFor(2) / phantomFor(1)).toBeCloseTo(2.0, 4);
  });

  it('yields a finite positive phantom for a fully degenerate marginal', () => {
    const output = new Float32Array(6);
    extract_visible_cholesky_3d(
      new Float32Array([0, 0, 0]),
      new Uint8Array([1]),
      new Uint32Array([0, 1]),
      2,
      1,
      output
    );

    // Crout floors the diagonals at sqrt(eps) before the mean is taken, so the
    // covariance stays non-singular instead of collapsing to zero.
    expect(Number.isFinite(output[5])).toBe(true);
    expect(output[5]).toBeGreaterThan(0);
    expect(output[5]).toBeCloseTo(SQRT_EPS, 10);
  });

  it('is rotation-invariant: an in-plane rotation does not change the phantom axis', () => {
    const [sx, sy] = [3.0, 0.75];
    const phantoms = [0, 17, 45, 73, 90].map((deg) => {
      const t = (deg * Math.PI) / 180;
      const [s, c] = [Math.sin(t), Math.cos(t)];
      const a = c * c * sx * sx + s * s * sy * sy;
      const b = c * s * (sx * sx - sy * sy);
      const d = s * s * sx * sx + c * c * sy * sy;
      const l00 = Math.sqrt(a);
      const l10 = b / l00;
      const l11 = Math.sqrt(d - l10 * l10);

      const output = new Float32Array(6);
      extract_visible_cholesky_3d(
        new Float32Array([l00, l10, l11]),
        new Uint8Array([1]),
        new Uint32Array([0, 1]),
        2,
        1,
        output
      );
      return output[5];
    });

    // Taken over the Cholesky pivots the mean is (det Sigma)^(1/4) = sqrt(sx*sy).
    for (const p of phantoms) {
      expect(p).toBeCloseTo(Math.sqrt(sx * sy), 4);
    }
  });
});

/**
 * Regression for issue #725: the TS reference is the UNCAPPED >16-dimension
 * backend (`pickBackend` routes ndim > MAX_SUPPORTED_DIMS here because the WASM
 * kernel panics above 16), yet its scratch buffers and the marginal-covariance
 * matrix stride were hardwired to MAX_SUPPORTED_DIMS (16). With more than 16
 * CONTINUOUS hidden dims the workspaces overflowed, the marginal Cholesky read
 * garbage, and amplitudes came back NaN — and since `NaN < minAmplitude` is
 * false EVERY splat was emitted as "visible" with a NaN amplitude (silent
 * corruption, no error).
 */
describe('gsplats_processing: >16 continuous hidden dims (issue #725)', () => {
  // Packed lower-triangular index, matching the module's packedIndex().
  const packedIdx = (row: number, col: number): number => (row * (row + 1)) / 2 + col;

  // Identity Cholesky (L = I) packed for `ndim` dims: unit diagonal, zero off-diag.
  const identityPackedCholesky = (ndim: number): Float32Array => {
    const packed = new Float32Array((ndim * (ndim + 1)) / 2);
    for (let i = 0; i < ndim; i++) packed[packedIdx(i, i)] = 1.0;
    return packed;
  };

  it('returns finite ~1.0 amplitudes for on-slice splats at ndim=20 (17 continuous hidden dims)', () => {
    const ndim = 20;
    const displayDims = new Uint32Array([0, 1, 2]);
    // Dims 3..19 (17 dims) are continuous hidden — one MORE than MAX_SUPPORTED_DIMS.
    const continuousHiddenDims = new Uint32Array(Array.from({ length: ndim - 3 }, (_, k) => k + 3));
    expect(continuousHiddenDims.length).toBe(17);

    const splatCount = 2;
    const fullPacked = identityPackedCholesky(ndim);
    // Both splats sit exactly on the slice (all coords 0) → hidden distance 0.
    const positions = new Float32Array(splatCount * ndim); // all zeros
    const cholesky = new Float32Array(splatCount * fullPacked.length);
    cholesky.set(fullPacked, 0);
    cholesky.set(fullPacked, fullPacked.length);
    const amplitudes = new Float32Array([1.0, 1.0]);
    const colors = new Float32Array(splatCount * 3).fill(1.0);
    const discreteVisibility = new Uint8Array([1, 1]);
    const slicePosition = new Float32Array(ndim); // all zeros

    const outCenters3d = new Float32Array(splatCount * 3);
    const outCholesky3d = new Float32Array(splatCount * 6);
    const outAmplitudes = new Float32Array(splatCount);
    const outColors = new Float32Array(splatCount * 3);

    const count = project_gsplats_nd_to_3d(
      positions,
      cholesky,
      amplitudes,
      colors,
      discreteVisibility,
      slicePosition,
      continuousHiddenDims,
      displayDims,
      ndim,
      splatCount,
      3,
      1e-6,
      4.0,
      outCenters3d,
      outCholesky3d,
      outAmplitudes,
      outColors
    );

    // On the buggy code the workspaces overflow, mahalDist is NaN, attenuation is
    // NaN, and `NaN < minAmplitude` is false → both splats emitted with NaN amps.
    // The NaN emission would still yield count===2, so the real assertion is that
    // the amplitudes are FINITE and ~1.0 (not NaN).
    expect(count).toBe(2);
    expect(Number.isFinite(outAmplitudes[0])).toBe(true);
    expect(Number.isFinite(outAmplitudes[1])).toBe(true);
    expect(outAmplitudes[0]).toBeCloseTo(1.0, 5);
    expect(outAmplitudes[1]).toBeCloseTo(1.0, 5);

    // GROWN-BUFFER REUSE: after the 17-D hidden marginal grows the workspaces,
    // the SAME call does a subNdim=3 display marginal (computeDisplayCholesky3D).
    // With an identity input the packed-3D display Cholesky must be exactly the
    // identity [1,0,1,0,0,1] — proving the grown workspaces are re-zeroed and no
    // stale 17-D data leaks into the later 3-D computation.
    for (let s = 0; s < 2; s++) {
      const c = s * 6;
      expect(Array.from(outCholesky3d.subarray(c, c + 6))).toEqual([1, 0, 1, 0, 0, 1]);
    }
  });

  it('attenuates a splat far off the slice in a hidden dim beyond index 16', () => {
    const ndim = 20;
    const displayDims = new Uint32Array([0, 1, 2]);
    const continuousHiddenDims = new Uint32Array(Array.from({ length: ndim - 3 }, (_, k) => k + 3));

    const splatCount = 3;
    const fullPacked = identityPackedCholesky(ndim);
    const cholesky = new Float32Array(splatCount * fullPacked.length);
    for (let s = 0; s < splatCount; s++) cholesky.set(fullPacked, s * fullPacked.length);

    const positions = new Float32Array(splatCount * ndim); // start all on-slice
    // Splat 2 sits far away in dim 19 — a CONTINUOUS hidden dim past index 16, so
    // its distance is only computed correctly if the >16-D marginal is right.
    positions[2 * ndim + 19] = 10.0;

    const amplitudes = new Float32Array([1.0, 1.0, 1.0]);
    const colors = new Float32Array(splatCount * 3).fill(1.0);
    const discreteVisibility = new Uint8Array([1, 1, 1]);
    const slicePosition = new Float32Array(ndim);

    const outCenters3d = new Float32Array(splatCount * 3);
    const outCholesky3d = new Float32Array(splatCount * 6);
    const outAmplitudes = new Float32Array(splatCount);
    const outColors = new Float32Array(splatCount * 3);

    const count = project_gsplats_nd_to_3d(
      positions,
      cholesky,
      amplitudes,
      colors,
      discreteVisibility,
      slicePosition,
      continuousHiddenDims,
      displayDims,
      ndim,
      splatCount,
      3,
      1e-6,
      4.0,
      outCenters3d,
      outCholesky3d,
      outAmplitudes,
      outColors
    );

    // Only the two on-slice splats survive. If the >16-D distance were silently
    // zero (as under the overflow), the far splat would pass with attenuation 1.0
    // and count would be 3.
    expect(count).toBe(2);
    expect(outAmplitudes[0]).toBeCloseTo(1.0, 5);
    expect(outAmplitudes[1]).toBeCloseTo(1.0, 5);
  });

  // A 20-D identity Cholesky with ONE off-diagonal in the hidden region:
  // L[19,18] = 1.5 (dims 18,19 are hidden indices 15,16). The hidden 17×17
  // marginal is identity except a 2×2 block on (15,16):
  //   Σ = [[1, 1.5], [1.5, 1 + 1.5²]] = [[1, 1.5], [1.5, 3.25]]  (det = 1).
  const correlatedNdim = 20;
  const correlatedOffDiag = 1.5;
  const correlatedPacked = (): Float32Array => {
    const packed = identityPackedCholesky(correlatedNdim);
    packed[packedIdx(19, 18)] = correlatedOffDiag; // L[19,18]
    return packed;
  };

  it('computes the CORRELATED >16-D marginal Cholesky (pins the stride, not just overflow)', () => {
    // Direct computeMarginalCholesky over the 17 hidden dims 3..19.
    const hidden = new Uint32Array(Array.from({ length: correlatedNdim - 3 }, (_, k) => k + 3));
    expect(hidden.length).toBe(17);
    const subPacked = (hidden.length * (hidden.length + 1)) / 2; // 153
    const out = new Float32Array(subPacked);

    computeMarginalCholesky(correlatedPacked(), 0, hidden, hidden.length, out, 0);

    // Cholesky of the 2×2 block: L_S[15,15]=1, L_S[16,15]=1.5, L_S[16,16]=√(3.25-2.25)=1.
    // (hidden index 15 = dim 18, hidden index 16 = dim 19.)
    expect(out[packedIdx(15, 15)]).toBeCloseTo(1.0, 5);
    expect(out[packedIdx(16, 15)]).toBeCloseTo(1.5, 5);
    expect(out[packedIdx(16, 16)]).toBeCloseTo(1.0, 5);
    // Every other diagonal is the untouched identity, and the coupling does NOT
    // bleed into neighbouring cells. A wrong stride scrambles these to finite
    // garbage, so this fails on a stride regression even though the NaN test won't.
    expect(out[packedIdx(14, 14)]).toBeCloseTo(1.0, 5);
    expect(out[packedIdx(16, 14)]).toBeCloseTo(0.0, 5);
    expect(out[packedIdx(15, 14)]).toBeCloseTo(0.0, 5);
  });

  it('applies the CORRELATED >16-D attenuation to a splat offset in a hidden dim past index 16', () => {
    const ndim = correlatedNdim;
    const displayDims = new Uint32Array([0, 1, 2]);
    const continuousHiddenDims = new Uint32Array(Array.from({ length: ndim - 3 }, (_, k) => k + 3));

    // Offset ONLY in dim 18 (hidden index 15), the correlated column. With the
    // marginal above, D² = δ²·(Σ⁻¹)[15,15] = δ²·(3.25/det) = 3.25·δ² — distinct
    // from the identity backend's δ². δ = 0.5 → D² = 0.8125.
    const delta = 0.5;
    const packed = correlatedPacked();
    const positions = new Float32Array(ndim);
    positions[18] = delta;

    const truncate = 4.0;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);
    const d2 = 3.25 * delta * delta;
    const expectedAtten = Math.max(0.0, invOneMinusC * (Math.exp(-0.5 * d2) - shiftC));
    // Sanity: the correlated answer is clearly separated from the identity one.
    const identityAtten = Math.max(0.0, invOneMinusC * (Math.exp(-0.5 * delta * delta) - shiftC));
    expect(Math.abs(expectedAtten - identityAtten)).toBeGreaterThan(0.1);

    const outCenters3d = new Float32Array(3);
    const outCholesky3d = new Float32Array(6);
    const outAmplitudes = new Float32Array(1);
    const outColors = new Float32Array(3);

    const count = project_gsplats_nd_to_3d(
      positions,
      packed,
      new Float32Array([1.0]),
      new Float32Array(3).fill(1.0),
      new Uint8Array([1]),
      new Float32Array(ndim),
      continuousHiddenDims,
      displayDims,
      ndim,
      1,
      3,
      1e-6,
      truncate,
      outCenters3d,
      outCholesky3d,
      outAmplitudes,
      outColors
    );

    expect(count).toBe(1);
    expect(outAmplitudes[0]).toBeCloseTo(expectedAtten, 5);
  });

  it('legacy compute_gsplats_attenuation matches the correlated >16-D result', () => {
    const ndim = correlatedNdim;
    const hiddenDims = new Uint32Array(Array.from({ length: ndim - 3 }, (_, k) => k + 3));
    const delta = 0.5;
    const positions = new Float32Array(ndim);
    positions[18] = delta;

    const truncate = 4.0;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);
    const expectedAtten = Math.max(
      0.0,
      invOneMinusC * (Math.exp(-0.5 * 3.25 * delta * delta) - shiftC)
    );

    const visibility = new Uint8Array(1);
    const attenuation = new Float32Array(1);
    const count = compute_gsplats_attenuation(
      positions,
      correlatedPacked(),
      new Float32Array([1.0]),
      new Float32Array(ndim),
      hiddenDims,
      ndim,
      1,
      1e-6,
      truncate,
      visibility,
      attenuation
    );

    expect(count).toBe(1);
    expect(visibility[0]).toBe(1);
    expect(attenuation[0]).toBeCloseTo(expectedAtten, 5);
  });

  it('INTERLEAVE: a small subNdim=2 marginal after a >16-D one is uncorrupted', () => {
    // First a >16-D correlated marginal — this GROWS the module workspaces.
    const hidden = new Uint32Array(Array.from({ length: correlatedNdim - 3 }, (_, k) => k + 3));
    const bigOut = new Float32Array((hidden.length * (hidden.length + 1)) / 2);
    computeMarginalCholesky(correlatedPacked(), 0, hidden, hidden.length, bigOut, 0);

    // Then a small subNdim=2 marginal on an UNRELATED 3-D correlated factor
    // (the exact case from the earlier `should produce correct marginal` test).
    // If grow-then-reuse leaked stale 17-D data, this small result would be wrong.
    const small = new Float32Array([2, 1, 3, 0.5, 0.5, 4]); // L=[[2,0,0],[1,3,0],[0.5,0.5,4]]
    const smallOut = new Float32Array(3);
    computeMarginalCholesky(small, 0, new Uint32Array([0, 2]), 2, smallOut, 0);

    // Σ_S = [[4,1],[1,16.5]] → L_S = [2, 0.5, √16.25].
    expect(smallOut[0]).toBeCloseTo(2.0, 4);
    expect(smallOut[1]).toBeCloseTo(0.5, 4);
    expect(smallOut[2]).toBeCloseTo(Math.sqrt(16.25), 3);
  });
});
