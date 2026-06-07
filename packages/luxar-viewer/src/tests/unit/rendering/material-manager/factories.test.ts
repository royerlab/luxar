/**
 * Direct unit tests for the factory tables, backend resolution, and
 * cache-key helpers.
 *
 * Orchestrator tests (material-manager.test.ts) exercise these
 * indirectly through `getXMaterial` cache hits/misses; this file pins
 * the keying contract so a future "tighten the bucketing" change
 * trips a deliberate test update.
 */

import { describe, it, expect } from 'vitest';
import {
  resolveMaterialBackend,
  pointCacheKey,
  lineCacheKey,
  gsplatCacheKey,
  VISUAL_FACTORIES,
  PICKING_FACTORIES,
  MEGA_SHADER_FACTORIES,
  type PointMaterialProperties,
  type LineMaterialProperties,
  type GSplatMaterialProperties,
} from '../../../../rendering/material-manager/factories';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';

// Minimal RendererCapabilities for the resolveMaterialBackend tests —
// only `apiSurface` is read.
const caps = (apiSurface: 'webgl2' | 'webgpu'): RendererCapabilities =>
  ({ apiSurface }) as RendererCapabilities;

const basePoint: PointMaterialProperties = {
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 1.0,
  intensity: 1.0,
  offset: 0.0,
};

const baseLine: LineMaterialProperties = {
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 1.0,
  intensity: 1.0,
  offset: 0.0,
};

const baseGSplat: GSplatMaterialProperties = {
  blendingMode: 'additive',
  opacity: 1.0,
  gamma: 1.0,
  intensity: 1.0,
  offset: 0.0,
};

describe('resolveMaterialBackend', () => {
  it("returns 'tsl' for webgpu", () => {
    expect(resolveMaterialBackend(caps('webgpu'))).toBe('tsl');
  });

  it("returns 'glsl' for webgl2", () => {
    expect(resolveMaterialBackend(caps('webgl2'))).toBe('glsl');
  });

  it("defaults to 'glsl' when caps is null (unit-test default)", () => {
    expect(resolveMaterialBackend(null)).toBe('glsl');
  });
});

describe('VISUAL_FACTORIES / PICKING_FACTORIES / MEGA_SHADER_FACTORIES shape', () => {
  it('VISUAL_FACTORIES has one entry per geometry kind, each with both backends', () => {
    expect(Object.keys(VISUAL_FACTORIES).sort()).toEqual(['gsplat', 'line', 'point']);
    for (const kind of ['point', 'line', 'gsplat'] as const) {
      expect(typeof VISUAL_FACTORIES[kind].glsl).toBe('function');
      expect(typeof VISUAL_FACTORIES[kind].tsl).toBe('function');
      expect(VISUAL_FACTORIES[kind].glsl).not.toBe(VISUAL_FACTORIES[kind].tsl);
    }
  });

  it('PICKING_FACTORIES has the same shape as VISUAL_FACTORIES', () => {
    expect(Object.keys(PICKING_FACTORIES).sort()).toEqual(['gsplat', 'line', 'point']);
    for (const kind of ['point', 'line', 'gsplat'] as const) {
      expect(typeof PICKING_FACTORIES[kind].glsl).toBe('function');
      expect(typeof PICKING_FACTORIES[kind].tsl).toBe('function');
    }
  });

  it('MEGA_SHADER_FACTORIES exposes a flat {glsl, tsl} pair (no per-geometry split)', () => {
    expect(Object.keys(MEGA_SHADER_FACTORIES).sort()).toEqual(['glsl', 'tsl']);
    expect(typeof MEGA_SHADER_FACTORIES.glsl).toBe('function');
    expect(typeof MEGA_SHADER_FACTORIES.tsl).toBe('function');
  });
});

describe('pointCacheKey', () => {
  it('produces the same key for identical props + backend (deterministic)', () => {
    expect(pointCacheKey(basePoint, 'glsl')).toBe(pointCacheKey(basePoint, 'glsl'));
  });

  it('starts with `point_<backend>_<blendingMode>_…` for routing', () => {
    expect(pointCacheKey(basePoint, 'glsl')).toMatch(/^point_glsl_additive_/);
    expect(pointCacheKey(basePoint, 'tsl')).toMatch(/^point_tsl_additive_/);
  });

  it('encodes opacity into a 0–100 bucket (clamp [0,1] × 100, round)', () => {
    const half = pointCacheKey({ ...basePoint, opacity: 0.5 }, 'glsl');
    const halfPlus = pointCacheKey({ ...basePoint, opacity: 0.504 }, 'glsl');
    const halfMinus = pointCacheKey({ ...basePoint, opacity: 0.496 }, 'glsl');
    // All three round to the same 50 bucket
    expect(half).toBe(halfPlus);
    expect(half).toBe(halfMinus);
    // But 0.55 is a different bucket
    expect(half).not.toBe(pointCacheKey({ ...basePoint, opacity: 0.55 }, 'glsl'));
  });

  it('encodes the transparent flag t=1 for non-opaque, t=0 for opaque blending mode', () => {
    expect(pointCacheKey({ ...basePoint, blendingMode: 'additive' }, 'glsl')).toMatch(/_t1$/);
    expect(pointCacheKey({ ...basePoint, blendingMode: 'opaque' }, 'glsl')).toMatch(/_t0$/);
  });

  it('encodes radiusScale into a bucket suffix (default = 1000)', () => {
    const def = pointCacheKey(basePoint, 'glsl');
    expect(def).toMatch(/_r1000_/);
    const scaled = pointCacheKey({ ...basePoint, radiusScale: 0.5 }, 'glsl');
    expect(scaled).toMatch(/_r500_/);
  });

  it('clamps negative radiusScale to zero (Math.max(0, ...))', () => {
    const negative = pointCacheKey({ ...basePoint, radiusScale: -2.5 }, 'glsl');
    expect(negative).toMatch(/_r0_/);
  });
});

describe('lineCacheKey', () => {
  it('starts with `line_<backend>_<blendingMode>_…` (different prefix from points)', () => {
    expect(lineCacheKey(baseLine, 'glsl')).toMatch(/^line_glsl_additive_/);
  });

  it('omits radius/sharpness/truncation buckets (lines have no such props)', () => {
    const key = lineCacheKey(baseLine, 'glsl');
    expect(key).not.toMatch(/_r\d+/);
    expect(key).not.toMatch(/_s\d+/);
    expect(key).not.toMatch(/_tr\d+/);
  });

  it('encodes the transparent flag the same as pointCacheKey', () => {
    expect(lineCacheKey({ ...baseLine, blendingMode: 'opaque' }, 'glsl')).toMatch(/_t0$/);
    expect(lineCacheKey({ ...baseLine, blendingMode: 'additive' }, 'glsl')).toMatch(/_t1$/);
  });
});

describe('gsplatCacheKey', () => {
  it('starts with `gsplat_<backend>_<blendingMode>_…`', () => {
    expect(gsplatCacheKey(baseGSplat, 'glsl')).toMatch(/^gsplat_glsl_additive_/);
  });

  it('encodes truncationRadius into a tr<N> bucket; default 3.0 → tr30', () => {
    expect(gsplatCacheKey(baseGSplat, 'glsl')).toMatch(/_tr30_/);
    expect(gsplatCacheKey({ ...baseGSplat, truncationRadius: 2.5 }, 'glsl')).toMatch(/_tr25_/);
    expect(gsplatCacheKey({ ...baseGSplat, truncationRadius: 4.7 }, 'glsl')).toMatch(/_tr47_/);
  });

  it('uses the same opacity/gamma/intensity/offset/transparent buckets as point/line', () => {
    const key = gsplatCacheKey(baseGSplat, 'tsl');
    // opacity=1.0 → 100, gamma=1.0 → 100, intensity=1.0 → 100,
    // offset=0.0 → (0+10)*10 = 100
    expect(key).toContain('_o100_g100_i100_f100_');
  });
});

describe('cross-kind isolation', () => {
  it('point / line / gsplat keys never collide even with identical numeric buckets', () => {
    const p = pointCacheKey(basePoint, 'glsl');
    const l = lineCacheKey(baseLine, 'glsl');
    const g = gsplatCacheKey(baseGSplat, 'glsl');
    expect(p).not.toBe(l);
    expect(p).not.toBe(g);
    expect(l).not.toBe(g);
  });

  it('glsl vs tsl keys differ for the same geometry kind + props', () => {
    expect(pointCacheKey(basePoint, 'glsl')).not.toBe(pointCacheKey(basePoint, 'tsl'));
  });
});
