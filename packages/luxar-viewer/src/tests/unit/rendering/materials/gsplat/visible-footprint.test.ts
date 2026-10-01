/**
 * Visible-footprint quad tightening (#2944 B10).
 *
 * The gsplat quad used to circumscribe the T-sigma truncation ellipse for
 * every splat. The fragment ALSO discards below the visibility floor, which
 * binds first for a dim splat, so the vertex stage now shrinks the quad to
 * the ellipse that discard leaves (`gsplatVisibleMahalSq`) plus a 1 px
 * margin. These tests pin the exactness argument against a TS replica of the
 * fragment tests:
 *   - at/beyond the returned radius the fragment test fails, inside the
 *     exact (margin-free) boundary it passes;
 *   - the returned radius never excludes a passing fragment (conservative);
 *   - a splat culled outright has no passing fragment at all;
 *   - a 2D raster sweep: every pixel centre that passes the shader's
 *     Cholesky-based discard lies inside the tightened eigen-aligned quad.
 */
import { describe, it, expect } from 'vitest';
import {
  GLSL_GSPLAT_VISIBLE_FOOTPRINT,
  GSPLAT_FOOTPRINT_PEAK_MARGIN,
  GSPLAT_FOOTPRINT_PIXEL_MARGIN,
  GSPLAT_VISIBILITY_FLOOR,
  gsplatFootprintExtent,
  gsplatVisibleMahalSq,
} from '../../../../../rendering/materials/gsplat/math';
import {
  GSPLAT_FRAGMENT_SHADER,
  GSPLAT_VERTEX_SHADER,
} from '../../../../../rendering/materials/gsplat/shader-glsl';
import {
  GSPLAT_PICK_FRAGMENT_SHADER,
  GSPLAT_PICK_VERTEX_SHADER,
} from '../../../../../rendering/picking/gsplat/shaders';

const T = 3;
const T2 = T * T;
const C = Math.exp(-0.5 * T2);
const INV_ONE_MINUS_C = 1 / (1 - C);

/**
 * The fragment's two discards, as the shader evaluates them. `peakScale` is
 * vAmplitude2D · 1/(1−C) · alpha factor · max(gain, 1).
 */
function fragmentPasses(peakScale: number, mahalSq: number): boolean {
  if (mahalSq > T2) return false;
  const intensity = peakScale * Math.max(Math.exp(-0.5 * mahalSq) - C, 0);
  return !(intensity < GSPLAT_VISIBILITY_FLOOR);
}

/** The exact (margin-free) visible boundary: −2·ln(C + floor/P), capped at T². */
function exactBoundary(peakScale: number): number {
  return Math.min(T2, -2 * Math.log(C + GSPLAT_VISIBILITY_FLOOR / peakScale));
}

const PEAKS = Array.from({ length: 61 }, (_, i) => 10 ** (-4 + i * 0.1)); // 1e-4 .. 1e2

describe('gsplatVisibleMahalSq', () => {
  it('keeps (essentially) the full truncation radius for bright splats', () => {
    // The shifted Gaussian reaches 0 AT T, so the floor always binds a hair
    // inside T² — by < 0.02 for a unit peak, and not at all in the limit.
    expect(gsplatVisibleMahalSq(1, C, T2)).toBeGreaterThan(T2 - 0.02);
    expect(gsplatVisibleMahalSq(100, C, T2)).toBeGreaterThan(T2 - 1e-3);
    expect(gsplatVisibleMahalSq(Number.POSITIVE_INFINITY, C, T2)).toBe(T2);
  });

  it('shrinks the radius of dim splats', () => {
    // amplitude 1e-3 at T = 3: the visibility floor binds near 2.1 sigma.
    const r2 = gsplatVisibleMahalSq(1e-3 * INV_ONE_MINUS_C, C, T2);
    expect(r2).toBeGreaterThan(0);
    expect(r2).toBeLessThan(0.6 * T2);
  });

  it('fails the fragment test at and beyond the returned radius, passes inside the exact boundary', () => {
    for (const p of PEAKS) {
      const r2 = gsplatVisibleMahalSq(p, C, T2);
      if (r2 < 0) continue;
      // At the returned radius and beyond it, nothing survives (unless the
      // radius IS T², where the truncation test itself is the boundary).
      if (r2 < T2) {
        expect(fragmentPasses(p, r2), `P=${p} at R²`).toBe(false);
      }
      expect(fragmentPasses(p, r2 * (1 + 1e-6) + 1e-9), `P=${p} beyond R²`).toBe(false);
      // Just inside the exact boundary the fragment survives. (A splat the
      // margin keeps alive although even its peak fails has no inside.)
      const exact = exactBoundary(p);
      if (!(exact >= 0)) continue;
      expect(fragmentPasses(p, exact * (1 - 1e-6)), `P=${p} inside`).toBe(true);
    }
  });

  it('is conservative: never excludes a passing fragment, and overshoots by at most the margin', () => {
    const marginSlack = 2 * Math.log(GSPLAT_FOOTPRINT_PEAK_MARGIN) + 1e-9;
    for (const p of PEAKS) {
      const r2 = gsplatVisibleMahalSq(p, C, T2);
      for (let i = 0; i <= 400; i++) {
        const m = (i / 400) * T2;
        if (fragmentPasses(p, m)) {
          expect(r2, `P=${p} m=${m}`).toBeGreaterThanOrEqual(m);
        }
      }
      if (r2 >= 0 && exactBoundary(p) >= 0) {
        expect(r2 - exactBoundary(p)).toBeLessThanOrEqual(marginSlack);
        expect(r2).toBeGreaterThanOrEqual(exactBoundary(p));
      }
    }
  });

  it('culls a splat none of whose fragments can pass, and only such a splat', () => {
    for (const p of [0, -1, 1e-9, 1e-5, GSPLAT_VISIBILITY_FLOOR * 0.9]) {
      expect(gsplatVisibleMahalSq(p, C, T2), `P=${p}`).toBeLessThan(0);
      // Even the centre (the peak) fails.
      expect(fragmentPasses(p, 0), `P=${p} centre`).toBe(false);
    }
    // A splat whose centre just passes is never culled.
    const justVisible = GSPLAT_VISIBILITY_FLOOR / (1 - C);
    expect(fragmentPasses(justVisible, 0)).toBe(true);
    expect(gsplatVisibleMahalSq(justVisible, C, T2)).toBeGreaterThanOrEqual(0);
  });

  it('keeps the legacy quad for a NaN peak scale', () => {
    expect(gsplatVisibleMahalSq(Number.NaN, C, T2)).toBe(T2);
  });

  it('works when C underflows to 0 (large truncation radius)', () => {
    const bigT2 = 40 * 40;
    expect(gsplatVisibleMahalSq(1e30, 0, bigT2)).toBeCloseTo(-2 * Math.log(1e-34 / 1.05), 6);
    expect(gsplatVisibleMahalSq(1, 0, bigT2)).toBeLessThan(bigT2);
  });
});

describe('gsplatFootprintExtent', () => {
  it('never exceeds the legacy extent and covers the visible semi-axis plus the pixel margin', () => {
    expect(gsplatFootprintExtent(30, 100, T2)).toBe(30); // sqrt(9·100)+1 = 31 > 30
    expect(gsplatFootprintExtent(30, 100, 1)).toBe(10 + GSPLAT_FOOTPRINT_PIXEL_MARGIN);
    expect(gsplatFootprintExtent(0.5, 1e-6, 0)).toBe(0.5);
  });
});

/**
 * 2D raster sweep with the shader's own math: Cholesky-based Mahalanobis at
 * pixel centres (fragment), eigen-aligned rectangle (vertex). Every pixel
 * centre that passes the fragment tests keeps, within the tightened quad,
 * every part of its MSAA sample disc (radius √2/2 px) that the legacy quad
 * covered — so coverage of every surviving fragment is unchanged.
 */
describe('tightened quad covers every passing fragment (raster sweep)', () => {
  function cholesky2x2(s00: number, s10: number, s11: number): [number, number, number] {
    const l00 = Math.sqrt(Math.max(s00, 1e-6));
    const inv00 = 1 / l00;
    const l10 = s10 * inv00;
    const l11 = Math.sqrt(Math.max(s11 - l10 * l10, 1e-6));
    return [inv00, l10, 1 / l11];
  }

  function eigen(s00: number, s10: number, s11: number) {
    const trace = s00 + s11;
    const det = s00 * s11 - s10 * s10;
    const sqrtDisc = Math.sqrt(Math.max(trace * trace - 4 * det, 0));
    const l1 = Math.max(0.5 * (trace + sqrtDisc), 1e-6);
    const l2 = Math.max(0.5 * (trace - sqrtDisc), 1e-6);
    let major: [number, number];
    if (Math.abs(s10) > 1e-6) {
      const n = Math.hypot(l1 - s11, s10);
      major = [(l1 - s11) / n, s10 / n];
    } else {
      major = s00 >= s11 ? [1, 0] : [0, 1];
    }
    return { l1, l2, major, minor: [-major[1], major[0]] as [number, number] };
  }

  const CASES: Array<{ s00: number; s10: number; s11: number; amp: number }> = [
    { s00: 40, s10: 0, s11: 40, amp: 1 },
    { s00: 40, s10: 0, s11: 40, amp: 2e-3 },
    { s00: 120, s10: 70, s11: 60, amp: 5e-3 },
    { s00: 900, s10: -500, s11: 400, amp: 3e-4 },
    { s00: 0.3, s10: 0, s11: 0.3, amp: 0.02 }, // sub-pixel (dilated floor)
    { s00: 250, s10: 249, s11: 250.3, amp: 1e-2 }, // near-degenerate needle
  ];

  for (const { s00, s10, s11, amp } of CASES) {
    it(`Σ=[${s00},${s10};${s10},${s11}] amp=${amp}`, () => {
      const peak = amp * INV_ONE_MINUS_C;
      const [inv00, l10, inv11] = cholesky2x2(s00, s10, s11);
      const { l1, l2, major, minor } = eigen(s00, s10, s11);
      const r2 = gsplatVisibleMahalSq(peak, C, T2);
      expect(r2).toBeGreaterThanOrEqual(0);
      const legacy1 = T * Math.sqrt(l1);
      const legacy2 = T * Math.sqrt(l2);
      const e1 = gsplatFootprintExtent(legacy1, l1, r2);
      const e2 = gsplatFootprintExtent(legacy2, l2, r2);
      // Sub-pixel centre offset so pixel centres do not sit on the axes.
      const cx = 0.37;
      const cy = -0.21;
      const reach = Math.ceil(T * Math.sqrt(l1)) + 2;
      const sampleRadius = Math.SQRT1_2;
      let passing = 0;
      for (let py = -reach; py <= reach; py++) {
        for (let px = -reach; px <= reach; px++) {
          const dx = px - cx;
          const dy = py - cy;
          const y0 = dx * inv00;
          const y1 = (dy - l10 * y0) * inv11;
          if (!fragmentPasses(peak, y0 * y0 + y1 * y1)) continue;
          passing++;
          const u = Math.abs(dx * major[0] + dy * major[1]);
          const v = Math.abs(dx * minor[0] + dy * minor[1]);
          // What the legacy quad covered of this pixel's sample disc, the
          // tightened quad still covers (their intersection is the new quad).
          expect(
            Math.min(u + sampleRadius, legacy1),
            `pixel (${px},${py}) major`
          ).toBeLessThanOrEqual(e1);
          expect(
            Math.min(v + sampleRadius, legacy2),
            `pixel (${px},${py}) minor`
          ).toBeLessThanOrEqual(e2);
        }
      }
      expect(passing).toBeGreaterThan(0);
    });
  }
});

describe('shader sources stay wired to the footprint helper', () => {
  it('GLSL helper interpolates the shared constants', () => {
    expect(GLSL_GSPLAT_VISIBLE_FOOTPRINT).toContain(`* ${GSPLAT_FOOTPRINT_PEAK_MARGIN}`);
    expect(GLSL_GSPLAT_VISIBLE_FOOTPRINT).toContain('1e-4 / max(scaled, 1e-30)');
    expect(GLSL_GSPLAT_VISIBLE_FOOTPRINT).toContain('+ 1.0)');
  });

  it('visual and pick shaders use the helper and the same visibility floor', () => {
    for (const vs of [GSPLAT_VERTEX_SHADER, GSPLAT_PICK_VERTEX_SHADER]) {
      expect(vs).toContain('float gsplatVisibleMahalSq(');
      expect(vs).toContain('extent1 = gsplatFootprintExtent(extent1, lambda1, visibleMahalSq);');
      expect(vs).toContain('extent2 = gsplatFootprintExtent(extent2, lambda2, visibleMahalSq);');
    }
    expect(GSPLAT_FRAGMENT_SHADER).toContain('intensity * max(uIntensity, 1.0) < 1e-4) discard');
    expect(GSPLAT_PICK_FRAGMENT_SHADER).toContain('if (salience < 1e-4) discard');
  });
});
