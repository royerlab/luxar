/**
 * Sharpness radial LUT (#1352 PR-4) — the CPU reference and the texture
 * the volumetric line sum lanes sample (`_shared/line-integral-lut.ts`).
 *
 * The load-bearing property is the NO-SEAM contract: the β = 2 row must
 * equal the analytic radial the sum lanes used before the LUT,
 * (exp(−K·q²) − C)/(1 − C), so sampling the LUT unconditionally cannot
 * move a pixel at the default knob. The rest pins the texture's shape
 * guarantees (endpoints, monotonicity, knob-axis resolution, half-float
 * fidelity) that the shaders' texel-center UV math relies on.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  LINE_RADIAL_LUT_WIDTH,
  LINE_RADIAL_LUT_HEIGHT,
  lineSharpnessKnobToBeta,
  lineRadialAbel,
  lineRadialProfile,
  buildLineRadialLUTData,
  getLineRadialLUTTexture,
} from '../../../../rendering/materials/_shared/line-integral-lut';
import { FALLOFF_K, FALLOFF_FLOOR } from '../../../../rendering/materials/_shared/falloff';

describe('lineSharpnessKnobToBeta', () => {
  it('maps the knob to beta = 2^(6s − 2) — the shaders inline the same expression', () => {
    expect(lineSharpnessKnobToBeta(0)).toBeCloseTo(0.25, 12);
    expect(lineSharpnessKnobToBeta(0.5)).toBeCloseTo(2, 12);
    expect(lineSharpnessKnobToBeta(1)).toBeCloseTo(16, 12);
  });
});

describe('lineRadialAbel quadrature', () => {
  it('matches the closed-form Gaussian Abel transform at beta = 2', () => {
    // A_2(q) = sqrt(pi/K) · exp(−K·q²) — the one exactly integrable row.
    for (let i = 0; i <= 64; i++) {
      const q = i / 64;
      const want = Math.sqrt(Math.PI / FALLOFF_K) * Math.exp(-FALLOFF_K * q * q);
      expect(Math.abs(lineRadialAbel(q, 2) - want)).toBeLessThan(1e-6);
    }
  });
});

describe('lineRadialProfile (the shifted + normalized rows)', () => {
  it('beta = 2 row IS the analytic radial the sum lanes used pre-LUT (the no-seam contract)', () => {
    for (let i = 0; i <= 128; i++) {
      const q = i / 128;
      const want = Math.max(Math.exp(-FALLOFF_K * q * q) - FALLOFF_FLOOR, 0) / (1 - FALLOFF_FLOOR);
      expect(Math.abs(lineRadialProfile(q, 0.5) - want)).toBeLessThan(1e-4);
    }
  });

  it('hits the calibration endpoints exactly at every knob', () => {
    for (const s of [0, 0.25, 0.5, 0.75, 1]) {
      expect(lineRadialProfile(0, s)).toBeCloseTo(1, 10);
      expect(lineRadialProfile(1, s)).toBeCloseTo(0, 10);
      // Compact support: the shader's ClampToEdge must read 0 past q = 1.
      expect(lineRadialProfile(1.5, s)).toBe(0);
    }
  });
});

describe('buildLineRadialLUTData', () => {
  const data = buildLineRadialLUTData();

  it('every row is monotone non-increasing in q with exact 1 → 0 endpoints', () => {
    const W = LINE_RADIAL_LUT_WIDTH;
    for (let row = 0; row < LINE_RADIAL_LUT_HEIGHT; row++) {
      expect(data[row * W]).toBeCloseTo(1, 10);
      expect(data[row * W + W - 1]).toBeCloseTo(0, 10);
      for (let col = 1; col < W; col++) {
        expect(data[row * W + col]).toBeLessThanOrEqual(data[row * W + col - 1] + 1e-12);
      }
    }
  });

  it('128 q texels resolve the q axis: linear interpolation vs a 255-texel rebuild < 1e-3', () => {
    // The q-axis worst case is the β = 16 row's near-tophat shoulder (the
    // steepest curvature anywhere in the table). 1e-3 ≈ a quarter of an
    // 8-bit step — the filtered LUT is visually identical to the exact
    // profile at every knob.
    const W = LINE_RADIAL_LUT_WIDTH;
    const H = LINE_RADIAL_LUT_HEIGHT;
    const dense = buildLineRadialLUTData(2 * W - 1, H);
    let worst = 0;
    for (const row of [0, H - 1]) {
      for (let col = 0; col < W - 1; col++) {
        const interp = 0.5 * (data[row * W + col] + data[row * W + col + 1]);
        const exact = dense[row * (2 * W - 1) + 2 * col + 1];
        worst = Math.max(worst, Math.abs(interp - exact));
      }
    }
    expect(worst).toBeLessThan(1e-3);
  });

  it('64 knob rows resolve the knob axis: linear interpolation vs a 128-row rebuild < 1e-2', () => {
    // The shipped LUT relies on the sampler's LINEAR filter between knob
    // rows. Hold every midpoint of the shipped grid against a rebuild at
    // double the knob resolution (whose odd rows sit exactly on those
    // midpoints — the 2H−1 grid's even rows coincide with the H grid).
    const W = LINE_RADIAL_LUT_WIDTH;
    const dense = buildLineRadialLUTData(W, 2 * LINE_RADIAL_LUT_HEIGHT - 1);
    let worst = 0;
    for (let row = 0; row < LINE_RADIAL_LUT_HEIGHT - 1; row++) {
      for (let col = 0; col < W; col++) {
        const interp = 0.5 * (data[row * W + col] + data[(row + 1) * W + col]);
        const exact = dense[(2 * row + 1) * W + col];
        worst = Math.max(worst, Math.abs(interp - exact));
      }
    }
    expect(worst).toBeLessThan(1e-2);
  });
});

describe('getLineRadialLUTTexture', () => {
  it('is a cached 128×65 R16F linear-filtered clamp-to-edge singleton', () => {
    const tex = getLineRadialLUTTexture();
    expect(getLineRadialLUTTexture()).toBe(tex);
    expect(tex.image.width).toBe(LINE_RADIAL_LUT_WIDTH);
    expect(tex.image.height).toBe(LINE_RADIAL_LUT_HEIGHT);
    expect(tex.format).toBe(THREE.RedFormat);
    expect(tex.type).toBe(THREE.HalfFloatType);
    expect(tex.minFilter).toBe(THREE.LinearFilter);
    expect(tex.magFilter).toBe(THREE.LinearFilter);
    expect(tex.wrapS).toBe(THREE.ClampToEdgeWrapping);
    expect(tex.wrapT).toBe(THREE.ClampToEdgeWrapping);
  });

  it('half-float storage round-trips the reference within half-float precision', () => {
    const tex = getLineRadialLUTTexture();
    const stored = tex.image.data as Uint16Array;
    const f32 = buildLineRadialLUTData();
    let worst = 0;
    for (let i = 0; i < f32.length; i++) {
      worst = Math.max(worst, Math.abs(THREE.DataUtils.fromHalfFloat(stored[i]) - f32[i]));
    }
    // Values live in [0, 1]: half-float grid spacing tops out at 2^-11.
    expect(worst).toBeLessThan(5e-4);
  });

  it('GPU-filtered sampling at the DEFAULT knob reads the analytic beta = 2 radial (the no-seam contract)', () => {
    // Emulates exactly what the shaders + sampler do at s = 0.5 — v-axis
    // texel coordinate, bilinear row blend, u-axis blend — over the STORED
    // half-float values. This is the regression guard for the 64-row trap:
    // an even (H−1) grid has no s = 0.5 row, the sampler averages the
    // beta ~ 1.94 / 2.07 neighbours, and this test fails on the structural
    // assertion below before the numeric one drifts.
    const H = LINE_RADIAL_LUT_HEIGHT;
    const W = LINE_RADIAL_LUT_WIDTH;
    expect(((H - 1) * 0.5) % 1, 'the knob grid must CONTAIN s = 0.5').toBe(0);
    const tex = getLineRadialLUTTexture();
    const stored = tex.image.data as Uint16Array;
    const at = (row: number, col: number): number =>
      THREE.DataUtils.fromHalfFloat(stored[row * W + col]);
    // v = (0.5·(H−1) + 0.5)/H → texel coordinate v·H − 0.5 = 0.5·(H−1).
    const rowCoord = 0.5 * (H - 1);
    const r0 = Math.min(Math.floor(rowCoord), H - 2);
    const rf = rowCoord - r0;
    let worst = 0;
    for (let i = 0; i <= 256; i++) {
      const q = i / 256;
      const colCoord = q * (W - 1); // u·W − 0.5 with u = (q·(W−1) + 0.5)/W
      const c0 = Math.min(Math.floor(colCoord), W - 2);
      const cf = colCoord - c0;
      const sampleRow = (r: number): number => at(r, c0) * (1 - cf) + at(r, c0 + 1) * cf;
      const filtered = sampleRow(r0) * (1 - rf) + sampleRow(r0 + 1) * rf;
      const analytic =
        Math.max(Math.exp(-FALLOFF_K * q * q) - FALLOFF_FLOOR, 0) / (1 - FALLOFF_FLOOR);
      worst = Math.max(worst, Math.abs(filtered - analytic));
    }
    // Budget: half-float storage (~4.9e-4) + q-axis linear interp of the
    // Gaussian row (~7e-5). A 64-row grid reads ~1.8e-4 EXTRA from the
    // row blend and, more importantly, trips the structural assert above.
    expect(worst).toBeLessThan(8e-4);
  });
});
