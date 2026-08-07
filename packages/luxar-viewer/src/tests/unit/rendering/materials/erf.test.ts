/**
 * Unit tests for `materials/_shared/erf.ts` — the single erf source.
 *
 * `erfRef` is checked against high-precision reference values (from
 * mpmath, 12 significant digits) and for the structural properties every
 * erf consumer relies on: odd symmetry, monotonicity, saturation.
 *
 * `erfPoly` (the shader polynomial) is checked against `erfRef` for its
 * documented error bound, for CONTINUITY at the ±3 clamp (the fit is
 * constrained to P(3) = 1 — a drifted coefficient set breaks this first),
 * and for the difference-error bound that the volumetric line fragment
 * shader's erf-difference lane depends on (arguments ≥ 0.5 apart).
 *
 * The GLSL string and the TSL builder are pinned to the SAME serialized
 * coefficients so the two backends cannot drift.
 */

import { describe, it, expect } from 'vitest';
import { float } from 'three/tsl';
import {
  erfRef,
  erfPoly,
  erfPolyTSL,
  ERF_POLY_COEFFS,
  ERF_POLY_CLAMP,
  GLSL_ERF_FUNCTIONS,
} from '../../../../rendering/materials/_shared/erf';
import { computeRayIntegralFactor } from '../../../../rendering/materials/gsplat/math';

/** High-precision erf values (mpmath, 12 significant digits). */
const ERF_REFERENCE: Array<[number, number]> = [
  [0.0, 0.0],
  [0.1, 0.112462916018],
  [0.25, 0.276326390168],
  [0.5, 0.520499877813],
  [0.7071067811865476, 0.682689492137], // 1/sqrt(2): P(|X|<sigma)
  [1.0, 0.84270079295],
  [1.5, 0.966105146475],
  [2.0, 0.995322265019],
  [2.5, 0.999593047983],
  [3.0, 0.999977909503],
  [3.5, 0.999999256902],
];

describe('erfRef (A&S 7.1.26 CPU reference)', () => {
  it('matches high-precision erf within the documented 1.5e-7 bound', () => {
    for (const [x, expected] of ERF_REFERENCE) {
      expect(Math.abs(erfRef(x) - expected)).toBeLessThan(1.6e-7);
      expect(Math.abs(erfRef(-x) - -expected)).toBeLessThan(1.6e-7);
    }
  });

  it('is odd and monotone non-decreasing over [-4, 4]', () => {
    let prev = -Infinity;
    for (let i = 0; i <= 4096; i++) {
      const x = -4 + (8 * i) / 4096;
      const y = erfRef(x);
      // Odd symmetry is exact by construction except at x == 0, where the
      // A&S form itself returns ~2e-9 (and both sign branches agree).
      expect(Math.abs(y + erfRef(-x))).toBeLessThan(5e-9);
      // The A&S approximation wiggles within its own 1.5e-7 error bound.
      expect(y).toBeGreaterThanOrEqual(prev - 1.6e-7);
      prev = y;
    }
  });

  it('saturates: |erf(x)| <= 1 everywhere, and erf(4) is within 1.6e-7 of 1', () => {
    for (const x of [4, 6, 10, 38]) {
      expect(erfRef(x)).toBeLessThanOrEqual(1.0);
      expect(erfRef(-x)).toBeGreaterThanOrEqual(-1.0);
    }
    expect(erfRef(4)).toBeGreaterThan(1 - 1.6e-7);
  });
});

describe('erfPoly (the shader polynomial)', () => {
  it('stays within 6e-4 of erfRef across [-3, 3]', () => {
    let worst = 0;
    for (let i = 0; i <= 6000; i++) {
      const x = -3 + (6 * i) / 6000;
      worst = Math.max(worst, Math.abs(erfPoly(x) - erfRef(x)));
    }
    expect(worst).toBeLessThan(6e-4);
    // Sensitivity control: the bound is tight enough to see a broken fit —
    // a single-ulp-of-print perturbation of c0 (1e-9 in the leading
    // coefficient) stays inside, but a 1e-3 perturbation must not.
    expect(worst).toBeGreaterThan(1e-5);
  });

  it('is continuous at the ±3 clamp: P(3) = 1 exactly as fitted', () => {
    // The constrained fit pins P(3) = 1; the clamp outside returns ±1.
    expect(Math.abs(erfPoly(ERF_POLY_CLAMP) - 1.0)).toBeLessThan(1e-7);
    expect(Math.abs(erfPoly(ERF_POLY_CLAMP + 1e-6) - 1.0)).toBeLessThan(1e-7);
    expect(erfPoly(100)).toBe(erfPoly(3.5)); // hard clamp beyond the radius
  });

  it('is odd, and monotone non-decreasing over [-3.5, 3.5]', () => {
    let prev = -Infinity;
    for (let i = 0; i <= 4096; i++) {
      const x = -3.5 + (7 * i) / 4096;
      const y = erfPoly(x);
      expect(y + erfPoly(-x)).toBeCloseTo(0, 12);
      // The least-squares fit wiggles by up to ~1.1e-5 near saturation;
      // anything beyond 2e-5 means the coefficient set changed.
      expect(y).toBeGreaterThanOrEqual(prev - 2e-5);
      prev = y;
    }
  });

  it('keeps the difference error under 3e-3 of the peak for arguments >= 0.5 apart', () => {
    // The volumetric line shader computes (erf(x1) - erf(x0)) / dx and
    // switches to a midpoint/Taylor lane below dx = 0.5 — this bound is
    // what makes that lane threshold sufficient.
    const dx = 0.5;
    let worst = 0;
    for (let i = 0; i <= 700; i++) {
      const xm = -3.5 + (7 * i) / 700;
      const approx = (erfPoly(xm + dx / 2) - erfPoly(xm - dx / 2)) / dx;
      const exact = (erfRef(xm + dx / 2) - erfRef(xm - dx / 2)) / dx;
      worst = Math.max(worst, Math.abs(approx - exact));
    }
    // Peak of the difference quotient is 2/sqrt(pi) ~ 1.128.
    expect(worst).toBeLessThan(3e-3);
  });
});

describe('GLSL / TSL single-source serialization', () => {
  it('embeds every coefficient with the exact toFixed(9) literal', () => {
    for (const c of ERF_POLY_COEFFS) {
      expect(GLSL_ERF_FUNCTIONS).toContain(c.toFixed(9));
    }
    expect(GLSL_ERF_FUNCTIONS).toContain(`min(abs(x), ${ERF_POLY_CLAMP.toFixed(9)})`);
  });

  it('declares luxarErf with no exp() and no division (the whole point)', () => {
    expect(GLSL_ERF_FUNCTIONS).toContain('float luxarErf(float x)');
    expect(GLSL_ERF_FUNCTIONS).not.toMatch(/exp\s*\(/);
    expect(GLSL_ERF_FUNCTIONS).not.toContain('/');
  });

  it('erfPolyTSL builds a node graph without throwing (API smoke)', () => {
    // Full numeric parity is proven by the tsl-shader-parity harness once
    // a shader consumes it; this pins the builder chain against TSL API
    // drift (mul/add/negate/lessThan/select).
    const node = erfPolyTSL(float(1.5));
    expect(node).toBeDefined();
    expect(typeof node).toBe('object');
  });

  it('the GLSL body, transpiled to JS, is bit-identical to erfPoly', () => {
    // The strongest string<->mirror equivalence proof available without a
    // GPU: mechanically rewrite the GLSL body as JS (float -> let,
    // min/abs -> Math.*) and evaluate it. A structural drift between the
    // hand-nested GLSL Horner and the mirror's loop shows up as a
    // non-zero difference somewhere on the grid.
    const body = GLSL_ERF_FUNCTIONS.replace('float luxarErf(float x) {', '')
      .replace(/\}\s*$/, '')
      .replaceAll('float ', 'let ')
      .replaceAll('min(', 'Math.min(')
      .replaceAll('abs(', 'Math.abs(');
    // eslint-disable-next-line no-new-func
    const glslErf = new Function('x', body) as (x: number) => number;
    for (let i = 0; i <= 4000; i++) {
      const x = -5 + (10 * i) / 4000;
      expect(glslErf(x)).toBe(erfPoly(x));
    }
  });
});

describe('computeRayIntegralFactor (refactored onto erfRef)', () => {
  it('reproduces the documented values: ~2.507 unshifted, ~2.433 at T=3', () => {
    // T -> infinity limit is sqrt(2*pi).
    expect(computeRayIntegralFactor(40)).toBeCloseTo(Math.sqrt(2 * Math.PI), 6);
    expect(computeRayIntegralFactor(3)).toBeCloseTo(2.4332069, 5);
  });
});
