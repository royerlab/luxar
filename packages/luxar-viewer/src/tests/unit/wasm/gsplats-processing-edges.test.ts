/**
 * Edge-case tests for WASM TypeScript-fallback gsplats-processing helpers.
 *
 * Closes wasm.md gap cluster:
 *   - [wasm.md G4][P5]  computeMarginalCholesky with displayDims=[0,2,4]
 *                       on a 5D correlated Cholesky — exercises its actual
 *                       cross-correlation logic.
 *   - [wasm.md G5][P5]  computeMarginalCholesky CHOLESKY_EPSILON degenerate
 *                       fallback (rank-deficient marginal → sqrt(EPS) clamp).
 *   - [wasm.md G19][P5] mahalanobis_distance ndim=16 (largest documented WASM
 *                       dim; full forward-sub accumulation).
 *
 * Pure math on typed arrays — no mocks. Floor-clamp behaviour at
 * CHOLESKY_EPSILON is a load-bearing invariant; future hardening will surface
 * here as an intentional contract change.
 */

import { describe, it, expect } from 'vitest';
import {
  computeMarginalCholesky,
  mahalanobis_distance,
} from '../../../wasm/typescript/gsplats-processing';

// Helper: pack a lower-triangular dense L into the packed format the helpers expect.
// L is given row by row (length = ndim*(ndim+1)/2 already).
function packLowerTri(rows: number[][]): Float32Array {
  const out: number[] = [];
  for (let i = 0; i < rows.length; i++) {
    for (let j = 0; j <= i; j++) {
      out.push(rows[i][j]);
    }
  }
  return new Float32Array(out);
}

describe('computeMarginalCholesky — non-sequential displayDims [wasm.md G4]', () => {
  it('[G4] keepDims=[0,2,4] on a 5D CORRELATED Cholesky reorders + reconstructs the marginal correctly', () => {
    // Construct a 5D L with non-zero off-diagonal at L[2,0] and L[4,2].
    // Σ_S[i,j] = Σ_k L[s_i,k]·L[s_j,k]. For S=[0,2,4]:
    //   Σ_S[0,0] = L[0,0]² = 1
    //   Σ_S[1,0] = L[2,0]·L[0,0] = 0.5 * 1 = 0.5
    //   Σ_S[1,1] = L[2,0]² + L[2,1]² + L[2,2]² = 0.25 + 0 + 1 = 1.25
    //   Σ_S[2,0] = L[4,0]·L[0,0] = 0
    //   Σ_S[2,1] = L[4,0]·L[2,0] + L[4,2]·L[2,2] = 0 + 0.3*1 = 0.3
    //   Σ_S[2,2] = L[4,0]² + L[4,1]² + L[4,2]² + L[4,3]² + L[4,4]² =
    //              0 + 0 + 0.09 + 0 + 1 = 1.09
    // Cholesky of this 3x3 matrix:
    //   L_S[0,0] = sqrt(1) = 1
    //   L_S[1,0] = 0.5 / 1 = 0.5
    //   L_S[1,1] = sqrt(1.25 - 0.25) = sqrt(1) = 1
    //   L_S[2,0] = 0
    //   L_S[2,1] = (0.3 - 0*0.5) / 1 = 0.3
    //   L_S[2,2] = sqrt(1.09 - 0 - 0.09) = sqrt(1) = 1
    const fullL = packLowerTri([[1], [0, 1], [0.5, 0, 1], [0, 0, 0, 1], [0, 0, 0.3, 0, 1]]);
    const keepDims = new Uint32Array([0, 2, 4]);
    const output = new Float32Array(6);

    computeMarginalCholesky(fullL, 0, keepDims, 3, output, 0);

    // Packed 3x3 lower-tri: [L00, L10, L11, L20, L21, L22].
    expect(output[0]).toBeCloseTo(1.0, 4);
    expect(output[1]).toBeCloseTo(0.5, 4);
    expect(output[2]).toBeCloseTo(1.0, 4);
    expect(output[3]).toBeCloseTo(0.0, 4);
    expect(output[4]).toBeCloseTo(0.3, 4);
    expect(output[5]).toBeCloseTo(1.0, 4);
  });

  it('[G4] correlated marginal L11 reflects the cross-correlation subtraction', () => {
    // A fully correlated L where row 2 has multiple non-zero entries. The
    // marginal for [0,2,4] must reconstruct Σ_S and re-factorize — NOT read raw
    // L elements (which would give L[2,2]=1 at slot 2).
    //   Σ_S[1,1] = L[2,0]² + L[2,1]² + L[2,2]² = 0.25 + 0.25 + 1 = 1.5
    //   L_S[1,1] = sqrt(1.5 - 0.5²) = sqrt(1.25) ≈ 1.118 (≠ the raw L[2,2]=1)
    const fullL2 = packLowerTri([[2], [0, 2], [0.5, 0.5, 1], [0, 0, 0, 1], [0.4, 0, 0.6, 0, 1]]);
    const margOut2 = new Float32Array(6);
    computeMarginalCholesky(fullL2, 0, new Uint32Array([0, 2, 4]), 3, margOut2, 0);

    expect(margOut2[2]).toBeCloseTo(Math.sqrt(1.25), 4);
  });
});

describe('computeMarginalCholesky — degenerate-variance floor [wasm.md G5]', () => {
  // The floor is SCALE-RELATIVE: a variance carries world-units², so an absolute
  // threshold would conflate "this axis has no extent" with "this scene uses
  // small units" and inflate genuinely tiny splats. The regularizer is anchored
  // to the largest diagonal of Σ_S (CHOLESKY_RELATIVE_EPSILON = 1e-12), with the
  // absolute CHOLESKY_EPSILON reserved for a SCALELESS (all-zero) Σ_S.
  it('[G5] scaleless marginal (all-zero variance) clamps to sqrt(EPSILON) rather than 0/NaN', () => {
    // Σ_S = [[0]] — no scale to be relative to, so the absolute backstop applies.
    const fullL = packLowerTri([[0], [0, 1]]);
    const keepDims = new Uint32Array([0]); // marginal of dim 0 only
    const output = new Float32Array(1);
    computeMarginalCholesky(fullL, 0, keepDims, 1, output, 0);

    // Floor: sqrt(1e-10) ≈ 1e-5. Not NaN, not 0.
    expect(output[0]).toBeCloseTo(Math.sqrt(1e-10), 8);
    expect(Number.isNaN(output[0])).toBe(false);
    expect(output[0]).toBeGreaterThan(0);
  });

  it('[G5] off-diagonal floor: degenerate diagonal → off-diag set to 0 (not NaN)', () => {
    // Σ = [[1e-14, 0], [0, 1]] — a condition number of 1e14, genuinely
    // rank-deficient at f32 precision. L = [[1e-7, 0], [0, 1]].
    const eps = 1e-7;
    const fullL = packLowerTri([[eps], [0, 1]]);
    const keepDims = new Uint32Array([0, 1]);
    const output = new Float32Array(3);
    computeMarginalCholesky(fullL, 0, keepDims, 2, output, 0);

    // Σ_S = [[eps², 0], [0, 1]], so max diagonal = 1 and the floor is 1e-12:
    //   L_S[0,0]: sum = 1e-14 < 1e-12 → sqrt(1e-12) = 1e-6.
    //   L_S[1,0]: diag = 1e-6 > 0 → sum / diag = 0 / 1e-6 = 0.
    //   L_S[1,1]: sum = 1 → sqrt(1) = 1.
    // Note the regularized diagonal is 1e-6 RELATIVE to the max axis, not the
    // absolute sqrt(1e-10) = 1e-5 an absolute floor produced — that older value
    // was scene-scale dependent and inflated nm-unit data 100x.
    expect(output[0]).toBeCloseTo(1e-6, 9);
    expect(output[1]).toBeCloseTo(0, 8);
    expect(output[2]).toBeCloseTo(1, 4);
    expect(Number.isNaN(output[1])).toBe(false);
  });

  it('[G5] the floor scales WITH the covariance (no absolute magnitude)', () => {
    // Same shape as above at two very different scene scales: the regularized
    // diagonal must stay the same FRACTION of the real axis, not a constant.
    const ratios = [1e-3, 1e3].map((scale) => {
      const fullL = packLowerTri([[1e-7 * scale], [0, 1 * scale]]);
      const output = new Float32Array(3);
      computeMarginalCholesky(fullL, 0, new Uint32Array([0, 1]), 2, output, 0);
      return output[0] / output[2];
    });

    expect(ratios[0]).toBeCloseTo(ratios[1], 9);
  });
});

describe('mahalanobis_distance — ndim=16 full-loop accumulation [wasm.md G19]', () => {
  it('[G19] identity 16D Cholesky: result equals ||diff||₂', () => {
    // L = I_16 → packed diagonal is 1, off-diagonals 0. Forward sub: y = diff.
    // ||y||₂ = sqrt(Σ diff²).
    const ndim = 16;
    const packedSize = (ndim * (ndim + 1)) / 2;
    const packedL = new Float32Array(packedSize);
    for (let i = 0; i < ndim; i++) {
      packedL[(i * (i + 1)) / 2 + i] = 1; // diagonal = 1
    }
    const diff = new Float32Array(ndim);
    for (let i = 0; i < ndim; i++) diff[i] = 0.25;

    const dist = mahalanobis_distance(diff, packedL, ndim);
    // ||diff|| = sqrt(16 * 0.0625) = sqrt(1) = 1.
    expect(dist).toBeCloseTo(1.0, 5);
  });

  it('[G19] identity 16D, all-zero diff returns 0 (no accumulator drift)', () => {
    const ndim = 16;
    const packedSize = (ndim * (ndim + 1)) / 2;
    const packedL = new Float32Array(packedSize);
    for (let i = 0; i < ndim; i++) packedL[(i * (i + 1)) / 2 + i] = 1;
    const diff = new Float32Array(ndim); // all zeros
    expect(mahalanobis_distance(diff, packedL, ndim)).toBe(0);
  });

  it('[G19] 16D, single nonzero diff slot at the LAST dim exercises full-loop pass', () => {
    // Pin loop-traversal contract for dim=15 — a mutation that capped the
    // loop at ndim-1 would miss this slot and produce 0 instead of 1.
    const ndim = 16;
    const packedSize = (ndim * (ndim + 1)) / 2;
    const packedL = new Float32Array(packedSize);
    for (let i = 0; i < ndim; i++) packedL[(i * (i + 1)) / 2 + i] = 1;
    const diff = new Float32Array(ndim);
    diff[15] = 1.0;
    expect(mahalanobis_distance(diff, packedL, ndim)).toBeCloseTo(1.0, 5);
  });
});
