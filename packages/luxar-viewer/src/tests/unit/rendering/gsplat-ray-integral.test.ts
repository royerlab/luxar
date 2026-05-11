/**
 * Numeric verification of the ray-integral sigma formula.
 *
 * The Gaussian line-integral along a ray direction r has 1D std-dev:
 *   sigma_line(r) = 1 / sqrt(rᵀ Σ⁻¹ r)
 *
 * This test mirrors the shader formula in TypeScript so the
 * precision-based math is covered without a WebGL context.
 */
import { describe, it, expect } from 'vitest';

/**
 * 3x3 inverse of a symmetric positive-definite matrix via cofactor
 * expansion. Mirrors the closed-form inverse used in the GSplat
 * vertex shader.
 */
function invertSPD3(
  a: number,
  b: number,
  c: number,
  d: number,
  e: number,
  f: number
): { i00: number; i11: number; i22: number; i01: number; i02: number; i12: number } {
  const det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
  const invDet = 1 / det;
  return {
    i00: (d * f - e * e) * invDet,
    i11: (a * f - c * c) * invDet,
    i22: (a * d - b * b) * invDet,
    i01: -(b * f - c * e) * invDet,
    i02: (b * e - c * d) * invDet,
    i12: -(a * e - b * c) * invDet,
  };
}

/** Mirrors the shader formula `sigma = 1 / sqrt(rᵀ Σ⁻¹ r)`. */
function rayIntegralSigma(rayDir: [number, number, number], sigma: number[]): number {
  const [a, b, c, d, e, f] = sigma; // [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
  const inv = invertSPD3(a, b, c, d, e, f);
  const [rx, ry, rz] = rayDir;
  const prx = inv.i00 * rx + inv.i01 * ry + inv.i02 * rz;
  const pry = inv.i01 * rx + inv.i11 * ry + inv.i12 * rz;
  const prz = inv.i02 * rx + inv.i12 * ry + inv.i22 * rz;
  const quad = rx * prx + ry * pry + rz * prz;
  return 1 / Math.sqrt(quad);
}

describe('ray integral sigma', () => {
  it('isotropic Σ: precision-based and covariance-based formulas agree', () => {
    // Σ = σ² I → Σ⁻¹ = (1/σ²) I → 1/sqrt(rᵀΣ⁻¹r) = σ for any unit r
    const sigma = [4, 0, 0, 4, 0, 4]; // σ² = 4 → σ = 2
    expect(rayIntegralSigma([1, 0, 0], sigma)).toBeCloseTo(2);
    expect(rayIntegralSigma([0, 1, 0], sigma)).toBeCloseTo(2);
    expect(rayIntegralSigma([0, 0, 1], sigma)).toBeCloseTo(2);
    const oneOverRoot3 = 1 / Math.sqrt(3);
    expect(
      rayIntegralSigma([oneOverRoot3, oneOverRoot3, oneOverRoot3], sigma)
    ).toBeCloseTo(2);
  });

  it('diagonal anisotropic Σ along an eigenaxis: returns that axis std-dev', () => {
    // Σ = diag(1, 4, 9) → eigen std-devs (1, 2, 3) along x/y/z
    const sigma = [1, 0, 0, 4, 0, 9];
    expect(rayIntegralSigma([1, 0, 0], sigma)).toBeCloseTo(1);
    expect(rayIntegralSigma([0, 1, 0], sigma)).toBeCloseTo(2);
    expect(rayIntegralSigma([0, 0, 1], sigma)).toBeCloseTo(3);
  });

  it('rotated anisotropic Σ: precision and covariance disagree off-axis', () => {
    // Σ = R diag(1, 9, 1) Rᵀ where R is a 45° rotation in xy plane.
    // Rotated covariance:
    //   Σ = [[5, 4, 0], [4, 5, 0], [0, 0, 1]]
    const sigma = [5, 4, 0, 5, 0, 1];

    // Along the major axis [1,1,0]/√2: precision-based sigma should equal 3
    // (the major axis std-dev of the original eigenbasis).
    const sqrt2inv = 1 / Math.sqrt(2);
    expect(rayIntegralSigma([sqrt2inv, sqrt2inv, 0], sigma)).toBeCloseTo(3);
    // Along the minor axis [1,-1,0]/√2: precision-based sigma should equal 1.
    expect(rayIntegralSigma([sqrt2inv, -sqrt2inv, 0], sigma)).toBeCloseTo(1);

    // The covariance-based alternative would compute sqrt(rᵀΣr):
    //   along [1,1,0]/√2: sqrt(0.5*5 + 0.5*5 + 2*0.5*4) = sqrt(9) = 3 ✓ (luckily)
    //   along [1,-1,0]/√2: sqrt(0.5*5 + 0.5*5 - 2*0.5*4) = sqrt(1) = 1 ✓ (luckily)
    // For these eigen-aligned rays the two agree. Try a non-eigen ray:
    const r = [1, 0, 0]; // not an eigenvector of the rotated Σ
    const precSigma = rayIntegralSigma(r as [number, number, number], sigma);
    // rᵀ Σ r = 5; sqrt(5) ≈ 2.236
    const covSqrt = Math.sqrt(5);
    // rᵀ Σ⁻¹ r: Σ⁻¹ = R diag(1, 1/9, 1) Rᵀ. For r=(1,0,0) projected onto
    // major axis = sqrt2inv, minor = sqrt2inv. So rᵀΣ⁻¹r = 0.5*1 + 0.5/9
    // = 0.5 + 0.0556 = 0.5556. sigma = 1/sqrt(0.5556) ≈ 1.342
    expect(precSigma).toBeCloseTo(1 / Math.sqrt(0.5 + 0.5 / 9), 2);
    // The two formulas disagree (~2.24 vs ~1.34), covering the
    // precision-based behavior.
    expect(Math.abs(precSigma - covSqrt)).toBeGreaterThan(0.5);
  });
});
