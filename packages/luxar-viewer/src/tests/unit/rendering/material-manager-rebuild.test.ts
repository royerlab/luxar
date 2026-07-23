/**
 * Unit tests for MaterialManager.rebuildAfterContextRestore.
 *
 * The rebuild path is tested in isolation: we don't need a real WebGL
 * context. All three material kinds are PER NODE now (the line-material
 * LRU — the last cached kind — died with the lines texture-storage
 * migration), so the cache maps rebuild clears are permanently empty;
 * what the rebuild MUST NOT do is disturb the camera-update registry or
 * dispose live materials. We create materials through the public API,
 * call the rebuild method, and assert those invariants hold.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
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

// Line materials are per node too (each carries its own `uLineTex`),
// so the line cache — like the point/gsplat caches — stays empty.
const LINE_PROPS = {
  opacity: 0.7,
  gamma: 1.5,
  intensity: 1.0,
  offset: 0.0,
  blendingMode: 'additive',
} as const;

describe('MaterialManager.rebuildAfterContextRestore', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
  });

  it('keeps the caches empty and preserves the camera-update registry', () => {
    // rebuildAfterContextRestore drops the per-type allocation caches —
    // permanently empty in the per-node world, so the clear is a no-op —
    // but PRESERVES registeredMaterials / ownedMaterials so existing
    // visible scene materials keep receiving updateCameraParams()
    // across the restore.
    const mm = new MaterialManager();
    mm.getLineMaterial(LINE_PROPS); // per-node — registered only
    mm.getPointMaterial(POINT_PROPS); // per-node — registered only

    const before = mm.getCacheStats();
    expect(before.cachedMaterials).toBe(0); // nothing is cached anymore
    expect(before.totalRegistered).toBe(2);

    mm.rebuildAfterContextRestore();

    const after = mm.getCacheStats();
    expect(after.cachedMaterials).toBe(0); // still empty
    // Registry preserved — visible materials still tracked for camera updates.
    expect(after.totalRegistered).toBe(before.totalRegistered);
  });

  it('produces a fresh material on next access (per-node — never a stale instance)', () => {
    const mm = new MaterialManager();
    const before = mm.getLineMaterial(LINE_PROPS);
    mm.rebuildAfterContextRestore();
    const after = mm.getLineMaterial(LINE_PROPS);
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

  it('does NOT call material.dispose() on live materials (avoids dead-context errors)', () => {
    // The reasoning is documented in the method's doc comment: some
    // THREE drivers throw when disposing programs from a now-dead
    // WebGL context — and per-node materials are attached to visible
    // scene meshes, so rebuild must never dispose them anyway.
    const mm = new MaterialManager();
    const m = mm.getLineMaterial(LINE_PROPS);
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

  it('preserved registry materials still receive camera updates after restore', async () => {
    // Regression for the bug where clearing registeredMaterials in
    // rebuildAfterContextRestore() stranded existing visible materials —
    // their updateCameraParams stopped firing after restore.
    const mm = new MaterialManager();
    const material = mm.getPointMaterial(POINT_PROPS);

    // Snapshot current camera-uniform state.
    const updateSpy = (() => {
      const orig = material.updateCameraParams.bind(material);
      let count = 0;
      material.updateCameraParams = ((
        fov: number,
        resolution: THREE.Vector2,
        isOrtho?: boolean,
        nearCull?: number
      ) => {
        count++;
        orig(fov, resolution, isOrtho, nearCull);
      }) as typeof material.updateCameraParams;
      return () => count;
    })();

    mm.rebuildAfterContextRestore();
    mm.updateCameraParams(75 * (Math.PI / 180), new THREE.Vector2(1920, 1080), false);

    // The material is still in the registry, so updateCameraParams
    // reached it. Pre-fix this would be 0.
    expect(updateSpy()).toBeGreaterThan(0);
  });
});
