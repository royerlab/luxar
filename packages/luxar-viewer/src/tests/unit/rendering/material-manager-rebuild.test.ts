/**
 * Unit tests for MaterialManager.rebuildAfterContextRestore.
 *
 * The rebuild path is tested in isolation: we don't need a real WebGL
 * context. We populate the cache through the public API, call the
 * rebuild method, and assert that subsequent cache stats reflect a
 * fresh state and that the next access produces a new material
 * instance (not a stale cached one).
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  MaterialManager,
  __resetMaterialManagerForTests,
} from '../../../rendering/material-manager';

const POINT_PROPS = {
  opacity: 0.7,
  gamma: 1.5,
  intensity: 1.0,
  offset: 0.0,
  blendingMode: 'additive',
  hdrColors: false,
  hdrIntensityMultiplier: 1.0,
  pointSizeScale: 1.0,
  customColorMode: false,
} as const;

describe('MaterialManager.rebuildAfterContextRestore', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
  });

  it('clears every cache so cacheStats report 0 entries', () => {
    const mm = new MaterialManager();
    mm.getPointMaterial(POINT_PROPS);

    expect(mm.getCacheStats().cachedMaterials).toBeGreaterThan(0);
    mm.rebuildAfterContextRestore();
    const after = mm.getCacheStats();
    expect(after.cachedMaterials).toBe(0);
    expect(after.totalRegistered).toBe(0);
    expect(after.ownedMaterials).toBe(0);
  });

  it('produces a fresh material on next access (not a stale cached one)', () => {
    const mm = new MaterialManager();
    const before = mm.getPointMaterial(POINT_PROPS);
    mm.rebuildAfterContextRestore();
    const after = mm.getPointMaterial(POINT_PROPS);
    expect(after).not.toBe(before);
  });

  it('is idempotent — calling repeatedly is safe', () => {
    const mm = new MaterialManager();
    mm.getPointMaterial(POINT_PROPS);
    expect(() => {
      mm.rebuildAfterContextRestore();
      mm.rebuildAfterContextRestore();
      mm.rebuildAfterContextRestore();
    }).not.toThrow();
  });

  it('does NOT call material.dispose() on cached entries (avoids dead-context errors)', () => {
    // The reasoning is documented in the method's doc comment: pmndrs +
    // some THREE drivers throw when disposing programs from a now-dead
    // WebGL context. We test the behavior indirectly: the rebuild path
    // succeeds even on a brand-new manager whose materials were never
    // disposed (no explicit disposal of pre-existing cache happens).
    const mm = new MaterialManager();
    const m = mm.getPointMaterial(POINT_PROPS);
    let disposeCalls = 0;
    const origDispose = m.dispose.bind(m);
    m.dispose = () => {
      disposeCalls++;
      origDispose();
    };
    mm.rebuildAfterContextRestore();
    expect(disposeCalls).toBe(0);
  });

  it('still allows dispose() to run cleanly afterward (full teardown)', () => {
    const mm = new MaterialManager();
    mm.getPointMaterial(POINT_PROPS);
    mm.rebuildAfterContextRestore();
    expect(() => mm.dispose()).not.toThrow();
    expect(mm.getCacheStats().cachedMaterials).toBe(0);
  });
});
