/**
 * Edge-case tests for WASM TypeScript-fallback gsplats-processing helpers.
 *
 * Closes wasm.md gap cluster:
 *   - [wasm.md G4][P5]  extract_visible_cholesky_3d with displayDims=[0,2,4]
 *                       on a 5D correlated Cholesky — exercises
 *                       computeMarginalCholesky's actual cross-correlation logic.
 *   - [wasm.md G5][P5]  computeMarginalCholesky CHOLESKY_EPSILON degenerate
 *                       fallback (rank-deficient marginal → sqrt(EPS) clamp).
 *   - [wasm.md G19][P5] mahalanobis_distance ndim=16 (largest documented WASM
 *                       dim; full forward-sub accumulation).
 *   - [wasm.md G20][P5] compute_gsplats_attenuation truncate=0 (degenerate
 *                       shiftC=1 / invOneMinusC=Infinity), minAmplitude=0
 *                       (exact boundary), minAmplitude<0 (always visible).
 *   - [wasm.md G21][P5] extract_cholesky_submatrix keepDims.length=0 (no
 *                       output), keepDims=[d_n-1] (single highest dim).
 *
 * Pure math on typed arrays — no mocks. Floor-clamp behaviour at
 * CHOLESKY_EPSILON is a load-bearing invariant; future hardening will surface
 * here as an intentional contract change.
 */

import { describe, it, expect } from 'vitest';
import {
  computeMarginalCholesky,
  mahalanobis_distance,
  extract_cholesky_submatrix,
  extract_visible_cholesky_3d,
  compute_gsplats_attenuation,
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

describe('extract_visible_cholesky_3d — non-sequential displayDims [wasm.md G4]', () => {
  it('[G4] displayDims=[0,2,4] on a 5D CORRELATED Cholesky reorders + reconstructs the marginal correctly', () => {
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
    const fullL = packLowerTri([
      [1],
      [0, 1],
      [0.5, 0, 1],
      [0, 0, 0, 1],
      [0, 0, 0.3, 0, 1],
    ]);
    const visibility = new Uint8Array([1]);
    const displayDims = new Uint32Array([0, 2, 4]);
    const output = new Float32Array(6);

    extract_visible_cholesky_3d(fullL, visibility, displayDims, 5, 1, output);

    // Packed 3x3 lower-tri: [L00, L10, L11, L20, L21, L22].
    expect(output[0]).toBeCloseTo(1.0, 4);
    expect(output[1]).toBeCloseTo(0.5, 4);
    expect(output[2]).toBeCloseTo(1.0, 4);
    expect(output[3]).toBeCloseTo(0.0, 4);
    expect(output[4]).toBeCloseTo(0.3, 4);
    expect(output[5]).toBeCloseTo(1.0, 4);
  });

  it('[G4] marginal Cholesky DIFFERS from raw extraction for correlated covariances', () => {
    // Pin the audit-flagged regression: a mutant collapsing
    // extract_visible_cholesky_3d to extract_cholesky_submatrix would produce
    // a different result on this correlated 5D input.
    const fullL = packLowerTri([
      [1],
      [0, 1],
      [0.5, 0, 1],
      [0, 0, 0, 1],
      [0, 0, 0.3, 0, 1],
    ]);
    const visibility = new Uint8Array([1]);
    const displayDims = new Uint32Array([0, 2, 4]);
    const marginalOut = new Float32Array(6);
    const rawOut = new Float32Array(6);

    extract_visible_cholesky_3d(fullL, visibility, displayDims, 5, 1, marginalOut);
    extract_cholesky_submatrix(fullL, displayDims, 3, rawOut);

    // Marginal Cholesky at index 2 should be > 0 (cross-correlation contributes),
    // while raw extraction reads L[2,2]=1 → both are 1; the discriminating
    // slot is index 4 (sub L21 — captures off-diagonal contribution).
    // Marginal L21 ≈ 0.3 vs raw L[4,2] = 0.3 — these happen to coincide here,
    // but the L11 slot (index 2) differs: marginal L11 = 1 (from Σ_S[1,1]=1.25
    // minus cross-term 0.25 → sqrt(1)) while raw extraction reads L[2,2] = 1
    // also. The actual discriminator is L20 (slot 3): marginal = 0,
    // raw = L[4,0] = 0. Hmm — for this particular L, raw and marginal
    // happen to coincide because L has the same first column as the marginal
    // covariance allows. Let me use a stronger discriminator: assert the
    // INTERMEDIATE Σ_S reconstruction is correct by checking that
    // marginalOut, when squared back to Σ, recovers the right marginal.
    // For now, simpler check: marginalOut[2] (L11) reflects subtraction of
    // L[2,0]² = 0.25 from Σ_S[1,1] = 1.25, giving sqrt(1) = 1; raw gives
    // L[2,2] = 1. Same. Use a different correlated L for the discriminator.

    // Easier discriminator: a fully correlated L where row 2 has multiple
    // non-zero entries.
    const fullL2 = packLowerTri([
      [2],
      [0, 2],
      [0.5, 0.5, 1],
      [0, 0, 0, 1],
      [0.4, 0, 0.6, 0, 1],
    ]);
    const margOut2 = new Float32Array(6);
    const rawOut2 = new Float32Array(6);
    extract_visible_cholesky_3d(fullL2, visibility, displayDims, 5, 1, margOut2);
    extract_cholesky_submatrix(fullL2, displayDims, 3, rawOut2);

    // L_marginal[0,0] = sqrt(Σ_S[0,0]) = sqrt(L[0,0]²) = sqrt(4) = 2.
    // Raw extract reads packed L[0,0] = 2. SAME.
    // L_marginal[1,0] = Σ_S[1,0] / L_marg[0,0] = L[2,0]·L[0,0] / 2 = 0.5*2/2 = 0.5.
    // Raw reads L[2,0] = 0.5. SAME (because L00 is on diagonal).
    // L_marginal[1,1] = sqrt(Σ_S[1,1] - L_m[1,0]²)
    //   Σ_S[1,1] = L[2,0]² + L[2,1]² + L[2,2]² = 0.25 + 0.25 + 1 = 1.5
    //   L_m[1,1] = sqrt(1.5 - 0.25) = sqrt(1.25) ≈ 1.118
    // Raw reads L[2,2] = 1.
    // → DIFFER at slot 2 (L11). Pin this.
    expect(Math.abs(margOut2[2] - rawOut2[2])).toBeGreaterThan(0.05);
  });
});

describe('computeMarginalCholesky — CHOLESKY_EPSILON degenerate fallback [wasm.md G5]', () => {
  it('[G5] rank-deficient marginal (zero variance dim) clamps to sqrt(EPSILON) rather than 0/NaN', () => {
    // Construct a 2D L with zero variance in dim 0:
    //   L = [[0, 0], [0, 1]] → Σ = L·Lᵀ = [[0, 0], [0, 1]].
    // Marginal of dim 0: Σ_S[0,0] = 0 → Cholesky-Crout: sum = 0,
    //   sum > CHOLESKY_EPSILON (1e-10) is FALSE → output = sqrt(1e-10) ≈ 1e-5.
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
    // Construct a 2D Σ where Cholesky-Crout produces a near-zero diagonal:
    //   Σ = [[1e-12, 0], [0, 1]] — first diag is below epsilon.
    //   To make computeMarginalCholesky see this Σ, build L such that
    //   L·Lᵀ has that shape. L = [[sqrt(1e-12) ≈ 1e-6, 0], [0, 1]].
    //   Mit verbatim:
    const eps = 1e-7; // sqrt(1e-14) — well below CHOLESKY_EPSILON when squared.
    const fullL = packLowerTri([[eps], [0, 1]]);
    const keepDims = new Uint32Array([0, 1]);
    const output = new Float32Array(3);
    computeMarginalCholesky(fullL, 0, keepDims, 2, output, 0);

    // Σ_S = [[eps², 0], [0, 1]]. Cholesky-Crout:
    //   L_S[0,0]: sum = eps² (1e-14) < EPSILON (1e-10) → sqrt(EPS) ≈ 3.16e-6.
    //   L_S[1,0]: diag = L_S[0,0] = 3.16e-6 > EPSILON → sum / diag = 0/3.16e-6 = 0.
    //     But also diag check `> EPSILON` is true (3.16e-6 > 1e-10), so it's computed.
    //   L_S[1,1]: sum = 1, > EPS → sqrt(1) = 1.
    expect(output[0]).toBeCloseTo(Math.sqrt(1e-10), 8);
    expect(output[1]).toBeCloseTo(0, 8);
    expect(output[2]).toBeCloseTo(1, 4);
    expect(Number.isNaN(output[1])).toBe(false);
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

describe('compute_gsplats_attenuation — truncate / minAmplitude boundaries [wasm.md G20]', () => {
  it('[G20] truncate=0 makes shiftC=1, invOneMinusC=Infinity → splats far from slice get attenuation 0', () => {
    // shiftC = exp(0) = 1; invOneMinusC = 1/0 = Infinity.
    // For a splat NOT at the slice: D > 0, rawExp = exp(-0.5·D²) < 1,
    //   attenuation = max(0, Infinity * (rawExp - 1)) = max(0, -Infinity) = 0.
    const cholesky = packLowerTri([[1]]); // 1D identity Cholesky
    const positions = new Float32Array([3.0]); // splat 3σ away
    const amplitudes = new Float32Array([1.0]);
    const slicePos = new Float32Array([0.0]);
    const hiddenDims = new Uint32Array([0]);
    const visibility = new Uint8Array(1);
    const attenuation = new Float32Array(1);

    compute_gsplats_attenuation(
      positions,
      cholesky,
      amplitudes,
      slicePos,
      hiddenDims,
      1,
      1,
      0.01, // minAmplitude
      0.0, // truncate=0
      visibility,
      attenuation
    );
    expect(attenuation[0]).toBe(0);
    expect(visibility[0]).toBe(0);
  });

  it('[G20] truncate=0 splat exactly at slice (D=0) → Infinity * 0 = NaN attenuation (documented quirk)', () => {
    // Documents the edge: 0 * Infinity → NaN. A future hardening could
    // special-case truncate=0 to set attenuation=1 at D=0; pin today's quirk.
    const cholesky = packLowerTri([[1]]);
    const positions = new Float32Array([0.0]); // splat AT slice
    const amplitudes = new Float32Array([1.0]);
    const slicePos = new Float32Array([0.0]);
    const hiddenDims = new Uint32Array([0]);
    const visibility = new Uint8Array(1);
    const attenuation = new Float32Array(1);

    compute_gsplats_attenuation(
      positions,
      cholesky,
      amplitudes,
      slicePos,
      hiddenDims,
      1,
      1,
      0.01,
      0.0,
      visibility,
      attenuation
    );
    expect(Number.isNaN(attenuation[0])).toBe(true);
    // NaN amplitudes[i] * NaN = NaN; NaN >= 0.01 is false → hidden.
    expect(visibility[0]).toBe(0);
  });

  it('[G20] minAmplitude=0: any non-negative attenuation makes the splat visible (boundary on >=)', () => {
    // Strict `>= 0` boundary: with amplitude=1, attenuation=0, product=0,
    // 0 >= 0 → visible. A mutation flipping `>=` to `>` would fail here.
    const cholesky = packLowerTri([[1]]);
    const positions = new Float32Array([10.0]); // far away, atten will be 0
    const amplitudes = new Float32Array([1.0]);
    const slicePos = new Float32Array([0.0]);
    const hiddenDims = new Uint32Array([0]);
    const visibility = new Uint8Array(1);
    const attenuation = new Float32Array(1);

    compute_gsplats_attenuation(
      positions,
      cholesky,
      amplitudes,
      slicePos,
      hiddenDims,
      1,
      1,
      0.0, // minAmplitude=0
      3.0,
      visibility,
      attenuation
    );
    expect(attenuation[0]).toBe(0);
    expect(visibility[0]).toBe(1);
  });

  it('[G20] minAmplitude<0: every splat is visible (degenerate "always-show" mode)', () => {
    const cholesky = packLowerTri([[1]]);
    const positions = new Float32Array([100.0]); // very far
    const amplitudes = new Float32Array([1.0]);
    const slicePos = new Float32Array([0.0]);
    const hiddenDims = new Uint32Array([0]);
    const visibility = new Uint8Array(1);
    const attenuation = new Float32Array(1);

    compute_gsplats_attenuation(
      positions,
      cholesky,
      amplitudes,
      slicePos,
      hiddenDims,
      1,
      1,
      -1.0, // negative minAmplitude
      3.0,
      visibility,
      attenuation
    );
    expect(visibility[0]).toBe(1);
  });

  it('[G20] numHidden===0: no dims hidden → attenuation forced to 1 → visible', () => {
    // Pin the explicit `if (numHidden === 0)` short-circuit at L232.
    const cholesky = packLowerTri([[1]]);
    const positions = new Float32Array([5.0]); // value doesn't matter
    const amplitudes = new Float32Array([1.0]);
    const slicePos = new Float32Array([0.0]);
    const hiddenDims = new Uint32Array(0); // empty
    const visibility = new Uint8Array(1);
    const attenuation = new Float32Array(1);

    compute_gsplats_attenuation(
      positions,
      cholesky,
      amplitudes,
      slicePos,
      hiddenDims,
      1,
      1,
      0.01,
      3.0,
      visibility,
      attenuation
    );
    expect(attenuation[0]).toBe(1.0);
    expect(visibility[0]).toBe(1);
  });
});

describe('extract_cholesky_submatrix — keepDims boundaries [wasm.md G21]', () => {
  it('[G21] keepDims.length === 0: subPackedSize=0, output untouched (no iteration)', () => {
    const packed = new Float32Array([1, 2, 3, 4, 5, 6]); // arbitrary 3D L
    const keepDims = new Uint32Array(0);
    const output = new Float32Array(0);
    expect(() => extract_cholesky_submatrix(packed, keepDims, 0, output)).not.toThrow();
  });

  it('[G21] keepDims = [d_n-1] (single highest dim): output = L[d_n-1, d_n-1]', () => {
    // For ndim=4, packed = [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33].
    // packedIndex(3, 3) = 3*4/2 + 3 = 9 → packed[9] = L33.
    const packed = new Float32Array([
      0, 0, 0, 0, 0, 0, 0, 0, 0, 7.5, // L33 at index 9
    ]);
    const keepDims = new Uint32Array([3]);
    const output = new Float32Array(1);
    extract_cholesky_submatrix(packed, keepDims, 1, output);
    expect(output[0]).toBe(7.5);
  });

  it('[G21] keepDims = [0] (lowest dim): output = L[0, 0]', () => {
    const packed = new Float32Array([3.14, 1, 2, 1, 2, 3]);
    const keepDims = new Uint32Array([0]);
    const output = new Float32Array(1);
    extract_cholesky_submatrix(packed, keepDims, 1, output);
    // 3.14 is not exactly representable in Float32 → use closeness.
    expect(output[0]).toBeCloseTo(3.14, 5);
  });
});
