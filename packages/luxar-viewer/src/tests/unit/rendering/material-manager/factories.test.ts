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

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  resolveMaterialBackend,
  VISUAL_FACTORIES,
  PICKING_FACTORIES,
  MEGA_SHADER_FACTORIES,
  GEOMETRY_KINDS,
} from '../../../../rendering/material-manager/factories';
import {
  loadTslMaterials,
  areTslMaterialsLoaded,
  resetTslMaterialsForTests,
} from '../../../../rendering/tsl/load';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';

// Every cell of these tables is a THUNK, not a constructor: the `tsl` ones
// resolve through the lazy `three/webgpu` boundary and do not exist until it
// has been awaited (issue #1679). So the assertions below must CALL the thunk —
// asserting `typeof cell === 'function'` would pass on the thunk itself and
// prove nothing about the class behind it.
beforeAll(async () => {
  await loadTslMaterials();
});

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
  it('VISUAL_FACTORIES has one entry per geometry kind plus the physical mesh family, each a distinct class', () => {
    expect(Object.keys(VISUAL_FACTORIES).sort()).toEqual([
      'gsplat',
      'line',
      'mesh',
      'meshPhysical',
      'point',
    ]);
    for (const kind of ['point', 'line', 'gsplat', 'mesh', 'meshPhysical'] as const) {
      const glsl = VISUAL_FACTORIES[kind].glsl();
      const tsl = VISUAL_FACTORIES[kind].tsl();
      expect(typeof glsl).toBe('function');
      expect(typeof tsl).toBe('function');
      expect(glsl).not.toBe(tsl);
      // Name the class, so a thunk that silently resolved to the WRONG backend
      // (the failure mode a `!==` check alone cannot see) fails here.
      expect(glsl.name).toMatch(/Material$/);
      expect(tsl.name).toMatch(/TSLMaterial$/);
    }
  });

  it('PICKING_FACTORIES has one entry per geometry kind, each with both backends', () => {
    // The two tables agreed on three types through the material phase and now agree
    // on four: mesh keys its pick ids on `gl_VertexID` rather than an
    // element-texture texel (spec §6.5), which is why it needed its OWN pick pair
    // rather than reusing a sibling's — not why it could go without one.
    expect(Object.keys(PICKING_FACTORIES).sort()).toEqual(['gsplat', 'line', 'mesh', 'point']);
    for (const kind of ['point', 'line', 'gsplat', 'mesh'] as const) {
      const glsl = PICKING_FACTORIES[kind].glsl();
      const tsl = PICKING_FACTORIES[kind].tsl();
      expect(typeof glsl).toBe('function');
      expect(typeof tsl).toBe('function');
      expect(glsl).not.toBe(tsl);
      expect(tsl.name).toMatch(/PickingTSLMaterial$/);
    }
  });

  it('every GEOMETRY kind has a matching PICKING_FACTORIES kind', () => {
    // The invariant the two assertions above only imply. Stated directly so adding a
    // fifth geometry type fails HERE — with a message naming the missing pick pair —
    // rather than by rendering an unpickable node in production.
    expect(Object.keys(PICKING_FACTORIES).sort()).toEqual([...GEOMETRY_KINDS].sort());
  });

  it('the physical mesh FAMILY has a visual pair but deliberately no pick pair', () => {
    // Picking renders geometry, not appearance: a physical mesh picks through the
    // house `mesh` pick material (spec MESH_PHYSICAL_MATERIALS_SPEC.md §3.2). A pick
    // entry here would be a second mesh pick shader to keep in sync for no gain.
    expect('meshPhysical' in PICKING_FACTORIES).toBe(false);
    expect(VISUAL_FACTORIES.meshPhysical.glsl().name).toBe('PhysicalMeshMaterial');
    expect(VISUAL_FACTORIES.meshPhysical.tsl().name).toBe('PhysicalMeshTSLMaterial');
    // And it is a different class from the house mesh material on both backends.
    expect(VISUAL_FACTORIES.meshPhysical.glsl()).not.toBe(VISUAL_FACTORIES.mesh.glsl());
    expect(VISUAL_FACTORIES.meshPhysical.tsl()).not.toBe(VISUAL_FACTORIES.mesh.tsl());
  });

  it('MEGA_SHADER_FACTORIES exposes a flat {glsl, tsl} pair (no per-geometry split)', () => {
    expect(Object.keys(MEGA_SHADER_FACTORIES).sort()).toEqual(['glsl', 'tsl']);
    expect(typeof MEGA_SHADER_FACTORIES.glsl()).toBe('function');
    expect(typeof MEGA_SHADER_FACTORIES.tsl()).toBe('function');
    expect(MEGA_SHADER_FACTORIES.glsl()).not.toBe(MEGA_SHADER_FACTORIES.tsl());
  });
});

describe('lazy TSL boundary', () => {
  // Restore the loaded state for any test file that shares this worker.
  afterAll(async () => {
    await loadTslMaterials();
  });

  it('a tsl thunk throws a directive error before the registry is loaded', () => {
    resetTslMaterialsForTests();
    expect(areTslMaterialsLoaded()).toBe(false);

    // The contract that matters: this must THROW, not quietly hand back the
    // GLSL class. A silent fallback would put a ShaderMaterial under
    // WebGPURenderer, which renders blank quads rather than failing — a
    // rendering bug wearing the costume of a wiring bug.
    expect(() => VISUAL_FACTORIES.point.tsl()).toThrow(/has not been.*loaded/s);
    expect(() => PICKING_FACTORIES.mesh.tsl()).toThrow(/loadTslMaterials/);
    expect(() => MEGA_SHADER_FACTORIES.tsl()).toThrow(/loadTslMaterials/);

    // The glsl side is unaffected — the WebGL path never touches the boundary.
    expect(typeof VISUAL_FACTORIES.point.glsl()).toBe('function');
  });

  it('loadTslMaterials is idempotent and returns the same registry object', async () => {
    resetTslMaterialsForTests();
    const [a, b] = await Promise.all([loadTslMaterials(), loadTslMaterials()]);
    expect(a).toBe(b);
    expect(await loadTslMaterials()).toBe(a);
    expect(areTslMaterialsLoaded()).toBe(true);
  });
});
