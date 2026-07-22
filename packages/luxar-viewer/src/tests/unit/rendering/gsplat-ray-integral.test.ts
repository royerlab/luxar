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
    expect(rayIntegralSigma([1, 0, 0], sigma)).toBeCloseTo(2, 5);
    expect(rayIntegralSigma([0, 1, 0], sigma)).toBeCloseTo(2, 5);
    expect(rayIntegralSigma([0, 0, 1], sigma)).toBeCloseTo(2, 5);
    const oneOverRoot3 = 1 / Math.sqrt(3);
    expect(rayIntegralSigma([oneOverRoot3, oneOverRoot3, oneOverRoot3], sigma)).toBeCloseTo(2, 5);
  });

  it('diagonal anisotropic Σ along an eigenaxis: returns that axis std-dev', () => {
    // Σ = diag(1, 4, 9) → eigen std-devs (1, 2, 3) along x/y/z
    const sigma = [1, 0, 0, 4, 0, 9];
    expect(rayIntegralSigma([1, 0, 0], sigma)).toBeCloseTo(1, 5);
    expect(rayIntegralSigma([0, 1, 0], sigma)).toBeCloseTo(2, 5);
    expect(rayIntegralSigma([0, 0, 1], sigma)).toBeCloseTo(3, 5);
  });

  it('rotated anisotropic Σ: precision and covariance disagree off-axis', () => {
    // Σ = R diag(1, 9, 1) Rᵀ where R is a 45° rotation in xy plane.
    // Rotated covariance:
    //   Σ = [[5, 4, 0], [4, 5, 0], [0, 0, 1]]
    const sigma = [5, 4, 0, 5, 0, 1];

    // Along the major axis [1,1,0]/√2: precision-based sigma should equal 3
    // (the major axis std-dev of the original eigenbasis).
    const sqrt2inv = 1 / Math.sqrt(2);
    expect(rayIntegralSigma([sqrt2inv, sqrt2inv, 0], sigma)).toBeCloseTo(3, 5);
    // Along the minor axis [1,-1,0]/√2: precision-based sigma should equal 1.
    expect(rayIntegralSigma([sqrt2inv, -sqrt2inv, 0], sigma)).toBeCloseTo(1, 5);

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

/**
 * Float32 GPU emulation: fround every op and flush denormals to zero
 * (GPUs run FTZ). Mirrors the shader's TRACE-NORMALIZED inversion —
 * `Σn = Σ / (trace/3)`, floors on the O(1) normalized det/quadratic,
 * `sigmaRay = sqrt(s / quadN)` — and proves it is scale-free where the
 * old direct inversion (det in world-units⁶ with an absolute 1e-12
 * clamp) collapsed on tiny-unit scenes.
 */
const FLT_MIN_NORMAL = 1.17549435e-38;
function f32(x: number): number {
  const r = Math.fround(x);
  return Math.abs(r) < FLT_MIN_NORMAL ? 0 : r;
}

/** Mirrors the shader's normalized sum-mode block (shader-glsl.ts). */
function rayIntegralSigmaNormalizedF32(rayDir: [number, number, number], sigma: number[]): number {
  const [A, B, C, D, E, F] = sigma.map(f32);
  const sTrace = Math.max(f32(f32(A + f32(D + F)) * (1.0 / 3.0)), 1e-30);
  const invS = f32(1 / sTrace);
  const a = f32(A * invS);
  const b = f32(B * invS);
  const c = f32(C * invS);
  const d = f32(D * invS);
  const e = f32(E * invS);
  const f = f32(F * invS);
  const det = f32(a * f32(d * f - e * e) - b * f32(b * f - c * e) + c * f32(b * e - c * d));
  const invDet = f32(1 / Math.max(det, 1e-12));
  const i00 = f32(f32(d * f - e * e) * invDet);
  const i11 = f32(f32(a * f - c * c) * invDet);
  const i22 = f32(f32(a * d - b * b) * invDet);
  const i01 = f32(-f32(b * f - c * e) * invDet);
  const i02 = f32(f32(b * e - c * d) * invDet);
  const i12 = f32(-f32(a * e - b * c) * invDet);
  const [rx, ry, rz] = rayDir;
  const prx = f32(i00 * rx + i01 * ry + i02 * rz);
  const pry = f32(i01 * rx + i11 * ry + i12 * rz);
  const prz = f32(i02 * rx + i12 * ry + i22 * rz);
  const quad = Math.max(f32(rx * prx + ry * pry + rz * prz), 1e-8);
  return f32(f32(1 / Math.sqrt(quad)) * Math.sqrt(sTrace));
}

describe('ray integral sigma — trace-normalized inversion is scale-free (float32/FTZ)', () => {
  const sigmaBase = [5, 4, 0, 5, 0, 1]; // rotated anisotropic Σ from above
  const ray: [number, number, number] = [1, 0, 0];
  const expected = 1 / Math.sqrt(0.5 + 0.5 / 9); // exact result at scale 1

  it.each([1, 1e-6, 1e6])('scale factor %s: sigmaRay scales linearly with the scene', (s) => {
    // Scaling the scene by s scales Σ by s² and sigma_line by s.
    const sigmaScaled = sigmaBase.map((v) => v * s * s);
    const got = rayIntegralSigmaNormalizedF32(ray, sigmaScaled);
    expect(got / s).toBeCloseTo(expected, 3);
  });

  it('the un-normalized inversion collapses at scale 1e-6 under FTZ (why the shader normalizes)', () => {
    // det(Σ · 1e-12) ~ 1e-36 · det(Σ) ~ 1e-35 — still above FTZ here, but
    // realistic microscopy sigmas (~1e-7 units ⇒ variances ~1e-14) give
    // det ~ 1e-42, which flushes to zero: max(0, 1e-12) then produces a
    // garbage Σ⁻¹. Reproduce with a tiny-variance isotropic splat.
    const s = 1e-7;
    const sigmaTiny = [s * s, 0, 0, s * s, 0, s * s]; // det = s^6 = 1e-42 → FTZ 0
    const det = f32(f32(sigmaTiny[0] * sigmaTiny[3]) * sigmaTiny[5]);
    expect(det).toBe(0); // the underflow the normalization avoids
    // The normalized path still recovers sigma_line = s exactly.
    expect(rayIntegralSigmaNormalizedF32([0, 0, 1], sigmaTiny)).toBeCloseTo(s, 10);
  });
});
