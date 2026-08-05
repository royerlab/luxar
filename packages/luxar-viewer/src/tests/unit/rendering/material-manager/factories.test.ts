/**
 * Direct unit tests for the factory tables and backend resolution.
 *
 * Orchestrator tests (material-manager.test.ts) exercise these
 * indirectly through the `getXMaterial` creation paths; this file pins
 * the table shapes and the caps → backend dispatch.
 *
 * No material kind is cached or keyed anymore: point, line, and gsplat
 * materials are all per node (each carries its own element texture),
 * so the historical `lineCacheKey` helper — and its keying contract —
 * is gone (it died with the lines texture-storage migration).
 */

import { describe, it, expect } from 'vitest';
import {
  resolveMaterialBackend,
  VISUAL_FACTORIES,
  PICKING_FACTORIES,
  MEGA_SHADER_FACTORIES,
} from '../../../../rendering/material-manager/factories';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';

// Minimal RendererCapabilities for the resolveMaterialBackend tests —
// only `apiSurface` is read.
const caps = (apiSurface: 'webgl2' | 'webgpu'): RendererCapabilities =>
  ({ apiSurface }) as RendererCapabilities;

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
    expect(Object.keys(VISUAL_FACTORIES).sort()).toEqual(['gsplat', 'line', 'mesh', 'point']);
    for (const kind of ['point', 'line', 'gsplat', 'mesh'] as const) {
      expect(typeof VISUAL_FACTORIES[kind].glsl).toBe('function');
      expect(typeof VISUAL_FACTORIES[kind].tsl).toBe('function');
      expect(VISUAL_FACTORIES[kind].glsl).not.toBe(VISUAL_FACTORIES[kind].tsl);
    }
  });

  it('PICKING_FACTORIES has one entry per geometry kind, each with both backends', () => {
    // The two tables agreed on three types through the material phase and now agree
    // on four: mesh keys its pick ids on `gl_VertexID` rather than an
    // element-texture texel (spec §6.5), which is why it needed its OWN pick pair
    // rather than reusing a sibling's — not why it could go without one.
    expect(Object.keys(PICKING_FACTORIES).sort()).toEqual(['gsplat', 'line', 'mesh', 'point']);
    for (const kind of ['point', 'line', 'gsplat', 'mesh'] as const) {
      expect(typeof PICKING_FACTORIES[kind].glsl).toBe('function');
      expect(typeof PICKING_FACTORIES[kind].tsl).toBe('function');
      expect(PICKING_FACTORIES[kind].glsl).not.toBe(PICKING_FACTORIES[kind].tsl);
    }
  });

  it('every VISUAL_FACTORIES kind has a matching PICKING_FACTORIES kind', () => {
    // The invariant the two assertions above only imply. Stated directly so adding a
    // fifth geometry type fails HERE — with a message naming the missing pick pair —
    // rather than by rendering an unpickable node in production.
    expect(Object.keys(PICKING_FACTORIES).sort()).toEqual(Object.keys(VISUAL_FACTORIES).sort());
  });

  it('MEGA_SHADER_FACTORIES exposes a flat {glsl, tsl} pair (no per-geometry split)', () => {
    expect(Object.keys(MEGA_SHADER_FACTORIES).sort()).toEqual(['glsl', 'tsl']);
    expect(typeof MEGA_SHADER_FACTORIES.glsl).toBe('function');
    expect(typeof MEGA_SHADER_FACTORIES.tsl).toBe('function');
  });
});
