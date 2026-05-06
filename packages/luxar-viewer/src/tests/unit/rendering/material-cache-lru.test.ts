/**
 * Unit tests for the LRU bound on MaterialManager's caches. Each cache
 * uses an inline LRU pattern: `lruGet` re-inserts on hit (promote to
 * MRU); `lruSet` evicts the oldest entry when at capacity, disposes
 * the evicted material, and removes it from the registry.
 *
 * The cap is read live from `config.dataLoading.performance.materialCacheMaxSize`
 * so we mutate the config in tests and reset on each cycle.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  MaterialManager,
  __resetMaterialManagerForTests,
} from '../../../rendering/material-manager';
import { config } from '../../../config';

const baseProps = (over: Record<string, number | boolean | string> = {}) => ({
  opacity: 1.0,
  gamma: 1.0,
  intensity: 1.0,
  offset: 0.0,
  blendingMode: 'additive',
  hdrColors: false,
  hdrIntensityMultiplier: 1.0,
  pointSizeScale: 1.0,
  customColorMode: false,
  ...over,
});

describe('MaterialManager LRU eviction', () => {
  let originalCap: number;

  beforeEach(() => {
    __resetMaterialManagerForTests();
    originalCap = config.dataLoading.performance.materialCacheMaxSize;
  });

  afterEach(() => {
    config.dataLoading.performance.materialCacheMaxSize = originalCap;
  });

  it('caches entries up to the configured bound', () => {
    config.dataLoading.performance.materialCacheMaxSize = 3;
    const mm = new MaterialManager();
    mm.getPointMaterial(baseProps({ opacity: 0.1 }));
    mm.getPointMaterial(baseProps({ opacity: 0.2 }));
    mm.getPointMaterial(baseProps({ opacity: 0.3 }));
    expect(mm.getCacheStats().pointMaterials).toBe(3);
    expect(mm.getCacheStats().evictions).toBe(0);
  });

  it('evicts the LRU entry on insert past the bound', () => {
    config.dataLoading.performance.materialCacheMaxSize = 2;
    const mm = new MaterialManager();
    mm.getPointMaterial(baseProps({ opacity: 0.1 })); // A
    mm.getPointMaterial(baseProps({ opacity: 0.2 })); // B
    expect(mm.getCacheStats().pointMaterials).toBe(2);

    // Inserting a third evicts A (oldest insertion).
    mm.getPointMaterial(baseProps({ opacity: 0.3 })); // C
    const stats = mm.getCacheStats();
    expect(stats.pointMaterials).toBe(2);
    expect(stats.evictions).toBe(1);
    expect(stats.keys.some((k) => k.includes('o10'))).toBe(false); // A's bucket
    expect(stats.keys.some((k) => k.includes('o20'))).toBe(true); // B
    expect(stats.keys.some((k) => k.includes('o30'))).toBe(true); // C
  });

  it('promotes entries to MRU on get (so re-fetched entries survive eviction)', () => {
    config.dataLoading.performance.materialCacheMaxSize = 2;
    const mm = new MaterialManager();
    mm.getPointMaterial(baseProps({ opacity: 0.1 })); // A
    mm.getPointMaterial(baseProps({ opacity: 0.2 })); // B
    // Re-fetch A — promotes it to MRU. B is now the LRU.
    mm.getPointMaterial(baseProps({ opacity: 0.1 }));
    // Insert C — should evict B, not A.
    mm.getPointMaterial(baseProps({ opacity: 0.3 })); // C

    const stats = mm.getCacheStats();
    expect(stats.pointMaterials).toBe(2);
    expect(stats.keys.some((k) => k.includes('o10'))).toBe(true); // A survived
    expect(stats.keys.some((k) => k.includes('o20'))).toBe(false); // B evicted
    expect(stats.keys.some((k) => k.includes('o30'))).toBe(true); // C
  });

  it('disposes the evicted material on eviction', () => {
    config.dataLoading.performance.materialCacheMaxSize = 1;
    const mm = new MaterialManager();
    const a = mm.getPointMaterial(baseProps({ opacity: 0.1 }));
    let aDisposed = false;
    const origDispose = a.dispose.bind(a);
    a.dispose = () => {
      aDisposed = true;
      origDispose();
    };
    // Force an insert past the cap → A evicts.
    mm.getPointMaterial(baseProps({ opacity: 0.2 }));
    expect(aDisposed).toBe(true);
  });

  it('drops the evicted material from registeredMaterials', () => {
    config.dataLoading.performance.materialCacheMaxSize = 1;
    const mm = new MaterialManager();
    mm.getPointMaterial(baseProps({ opacity: 0.1 }));
    expect(mm.getCacheStats().totalRegistered).toBe(1);
    mm.getPointMaterial(baseProps({ opacity: 0.2 }));
    expect(mm.getCacheStats().totalRegistered).toBe(1);
  });

  it('disabled (maxSize=0) keeps unbounded behavior', () => {
    config.dataLoading.performance.materialCacheMaxSize = 0;
    const mm = new MaterialManager();
    for (let i = 0; i < 25; i++) {
      mm.getPointMaterial(baseProps({ opacity: i / 25 }));
    }
    expect(mm.getCacheStats().pointMaterials).toBe(25);
    expect(mm.getCacheStats().evictions).toBe(0);
  });

  it('eviction is per-cache-type — point cache does not evict line entries', () => {
    config.dataLoading.performance.materialCacheMaxSize = 1;
    const mm = new MaterialManager();
    mm.getPointMaterial(baseProps({ opacity: 0.1 }));
    mm.getLineMaterial(baseProps({ opacity: 0.1 }));
    expect(mm.getCacheStats().pointMaterials).toBe(1);
    expect(mm.getCacheStats().lineMaterials).toBe(1);
    // Insert a 2nd point — evicts only from the point cache.
    mm.getPointMaterial(baseProps({ opacity: 0.2 }));
    expect(mm.getCacheStats().pointMaterials).toBe(1);
    expect(mm.getCacheStats().lineMaterials).toBe(1);
  });

  it('getCacheStats exposes the configured maxSize', () => {
    config.dataLoading.performance.materialCacheMaxSize = 7;
    const mm = new MaterialManager();
    expect(mm.getCacheStats().maxSize).toBe(7);
  });
});
