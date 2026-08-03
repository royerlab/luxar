/**
 * Guards for the shared Points/Lines sprite-falloff constants.
 *
 * Two things are pinned here:
 *
 * 1. **Serialization.** `FALLOFF_K` is interpolated into four hand-written GLSL3
 *    shader sources and emitted by TSL codegen. If its string form ever changes,
 *    every `src/tests/__codegen__/*.fragment.glsl.txt` snapshot silently
 *    rewrites. These tests fail first, and say why.
 * 2. **GLSL3-vs-TSL parity.** The codegen snapshots only cover the TSL backend.
 *    The hand-written GLSL3 sources are checked here against the same constant,
 *    so `K` can no longer drift between the two backends unnoticed.
 */

import { describe, expect, it } from 'vitest';

import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../../config/constants';
import {
  FALLOFF_FLOOR,
  FALLOFF_K,
  GAUSSIAN_EQUIVALENT_TRUNCATION,
  INV_ONE_MINUS_FALLOFF_FLOOR,
} from '../../../../rendering/materials/_shared/falloff';
import { LINE_FRAGMENT_SHADER } from '../../../../rendering/materials/line/shader-glsl';
import { POINT_FRAGMENT_SHADER } from '../../../../rendering/materials/point/shader-glsl';
import { LINE_PICK_FRAGMENT_SHADER } from '../../../../rendering/picking/line/shaders';
import { POINT_PICK_FRAGMENT_SHADER } from '../../../../rendering/picking/point/shaders';

describe('falloff constants', () => {
  it('serializes K exactly as the shader snapshots expect', () => {
    // The literal every __codegen__ snapshot and hand-written GLSL contains.
    // Raw Math.log(100) is 4.605170185988092 — interpolating THAT would
    // rewrite 16 snapshot files. The toFixed(7) in falloff.ts prevents it.
    expect(String(FALLOFF_K)).toBe('4.6051702');
    expect(String(FALLOFF_FLOOR)).toBe('0.01');
    expect(String(INV_ONE_MINUS_FALLOFF_FLOOR)).toBe('1.0101010101010102');
  });

  it('rounds K to a float32 no-op', () => {
    // 7 decimals is float32's significant-digit count, so the rounding that
    // keeps the shader text stable is invisible on the GPU.
    expect(Math.fround(FALLOFF_K)).toBe(Math.fround(Math.log(100)));
  });

  it('derives K from the floor rather than hardcoding it', () => {
    // K = ln(1 / floor): the profile reaches exactly `floor` at rho = 1,
    // which is what the shift then subtracts away.
    expect(Math.exp(-FALLOFF_K)).toBeCloseTo(FALLOFF_FLOOR, 7);
    expect(INV_ONE_MINUS_FALLOFF_FLOOR).toBeCloseTo(1 / (1 - FALLOFF_FLOOR), 12);
  });

  it('serializes as GLSL float literals, not int literals', () => {
    // `const float K = ${FALLOFF_K};` is only valid GLSL while the value
    // serializes with a decimal point. An integral value would emit
    // `const float K = 5;` — a type error the codegen snapshots would not
    // obviously flag. (Same hazard `${VOLUMETRIC_SERIES_C2_DIVISOR}.0` exists
    // for.)
    expect(String(FALLOFF_K)).toMatch(/^-?\d+\.\d/);
    expect(String(FALLOFF_FLOOR)).toMatch(/^-?\d+\.\d/);
  });

  it('exposes the Gaussian-equivalent truncation as a derived value', () => {
    // T* = sqrt(2K): the truncation radius at which a truncated Gaussian
    // coincides EXACTLY with this super-Gaussian at beta = 2.
    expect(GAUSSIAN_EQUIVALENT_TRUNCATION).toBe(Math.sqrt(2 * FALLOFF_K));
    expect(GAUSSIAN_EQUIVALENT_TRUNCATION).toBeCloseTo(3.0349, 4);
  });

  it('does NOT match the gsplat render default, and that is deliberate', () => {
    // Points/lines size their sprite to the 1% iso-contour; gsplats truncate
    // at a chosen sigma cutoff. The ~10% gap is intentional — asserting
    // equality here would encode a falsehood. What matters is that the two are
    // separate knobs and neither silently follows the other.
    expect(GAUSSIAN_EQUIVALENT_TRUNCATION).toBeGreaterThan(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
    const gapPercent =
      (100 * (GAUSSIAN_EQUIVALENT_TRUNCATION - GSPLAT_DEFAULT_TRUNCATION_RADIUS)) /
      GSPLAT_DEFAULT_TRUNCATION_RADIUS;
    expect(gapPercent).toBeCloseTo(10.36, 1);
  });
});

describe('gsplat truncation default', () => {
  it('matches the Python mirror', () => {
    // MIRROR: DEFAULT_TRUNCATION_RADIUS in
    // packages/luxar/src/luxar/typing_utils/constants.py must hold this value.
    // The two languages cannot share a symbol, so each side pins the literal
    // and names the other — same convention as ALPHA_CLAMP. A Python test
    // (typing_utils/tests/test_constants.py) pins that side.
    expect(GSPLAT_DEFAULT_TRUNCATION_RADIUS).toBe(2.75);
  });

  it('yields a well-conditioned shifted-Gaussian normalization', () => {
    // T sets the shift C = exp(-T^2/2) and the renormalization 1/(1-C);
    // a T large enough to drive C to 1 would divide by zero.
    const shiftC = Math.exp(-0.5 * GSPLAT_DEFAULT_TRUNCATION_RADIUS ** 2);
    expect(shiftC).toBeGreaterThan(0);
    expect(shiftC).toBeLessThan(1);
    expect(Number.isFinite(1 / (1 - shiftC))).toBe(true);
  });
});

describe('GLSL3 shaders emit the shared falloff constants', () => {
  // Build the expected literals from the same constant the TSL graph uses, so
  // a change on either side fails rather than diverging silently.
  const k = String(FALLOFF_K).replace('.', '\\.');
  const c = String(FALLOFF_FLOOR).replace('.', '\\.');

  const SOURCES: ReadonlyArray<readonly [string, string]> = [
    ['point visual', POINT_FRAGMENT_SHADER],
    ['line visual', LINE_FRAGMENT_SHADER],
    ['point pick', POINT_PICK_FRAGMENT_SHADER],
    ['line pick', LINE_PICK_FRAGMENT_SHADER],
  ];

  it.each(SOURCES)('%s shader declares K and C from the shared module', (_name, source) => {
    expect(source).toMatch(new RegExp(`float K = ${k};`));
    expect(source).toMatch(new RegExp(`float C = ${c};`));
    // The renormalization stays an in-shader expression (not an interpolated
    // literal) so the emitted text matches the historical snapshots exactly.
    expect(source).toMatch(/float INV_ONE_MINUS_C = 1\.0 \/ \(1\.0 - C\);/);
  });

  it.each(SOURCES)('%s shader contains no stray hardcoded ln(100)', (_name, source) => {
    // Catches a future edit that reintroduces the literal instead of importing.
    const stray = source.match(/4\.6051702/g) ?? [];
    expect(stray).toHaveLength(1); // the one interpolated occurrence
  });
});
