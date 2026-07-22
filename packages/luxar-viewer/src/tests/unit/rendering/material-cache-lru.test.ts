/**
 * Unit tests for the LRU bound on MaterialManager's line cache (the
 * ONLY cached material kind — point and gsplat materials are per node
 * and never cached). The cache uses an inline LRU pattern: `lruGet`
 * re-inserts on hit (promote to MRU); `lruSet` evicts the oldest entry
 * when at capacity. Eviction is DEFER-DISPOSE: the entry leaves the
 * cache but stays registered for camera updates (shared materials may
 * still be attached to live meshes) and is only disposed at manager
 * dispose().
 *
 * The cap is read live from `config.dataLoading.performance.materialCacheMaxSize`
 * so we mutate the config in tests and reset on each cycle.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  MaterialManager,
  __resetMaterialManagerForTests,
} from '../../../rendering/material-manager';
import { config } from '../../../config';

const baseProps = (over: Record<string, number | boolean | string> = {}) =>
  ({
    opacity: 1.0,
    gamma: 1.0,
    intensity: 1.0,
    offset: 0.0,
    blendingMode: 'additive',
    ...over,
  }) as Parameters<MaterialManager['getLineMaterial']>[0];

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
    mm.getLineMaterial(baseProps({ opacity: 0.1 }));
    mm.getLineMaterial(baseProps({ opacity: 0.2 }));
    mm.getLineMaterial(baseProps({ opacity: 0.3 }));
    expect(mm.getCacheStats().lineMaterials).toBe(3);
    expect(mm.getCacheStats().evictions).toBe(0);
  });

  it('evicts the LRU entry on insert past the bound', () => {
    config.dataLoading.performance.materialCacheMaxSize = 2;
    const mm = new MaterialManager();
    mm.getLineMaterial(baseProps({ opacity: 0.1 })); // A
    mm.getLineMaterial(baseProps({ opacity: 0.2 })); // B
    expect(mm.getCacheStats().lineMaterials).toBe(2);

    // Inserting a third evicts A (oldest insertion).
    mm.getLineMaterial(baseProps({ opacity: 0.3 })); // C
    const stats = mm.getCacheStats();
    expect(stats.lineMaterials).toBe(2);
    expect(stats.evictions).toBe(1);
    expect(stats.keys.some((k) => k.includes('o10'))).toBe(false); // A's bucket
    expect(stats.keys.some((k) => k.includes('o20'))).toBe(true); // B
    expect(stats.keys.some((k) => k.includes('o30'))).toBe(true); // C
  });

  it('promotes entries to MRU on get (so re-fetched entries survive eviction)', () => {
    config.dataLoading.performance.materialCacheMaxSize = 2;
    const mm = new MaterialManager();
    mm.getLineMaterial(baseProps({ opacity: 0.1 })); // A
    mm.getLineMaterial(baseProps({ opacity: 0.2 })); // B
    // Re-fetch A — promotes it to MRU. B is now the LRU.
    mm.getLineMaterial(baseProps({ opacity: 0.1 }));
    // Insert C — should evict B, not A.
    mm.getLineMaterial(baseProps({ opacity: 0.3 })); // C

    const stats = mm.getCacheStats();
    expect(stats.lineMaterials).toBe(2);
    expect(stats.keys.some((k) => k.includes('o10'))).toBe(true); // A survived
    expect(stats.keys.some((k) => k.includes('o20'))).toBe(false); // B evicted
    expect(stats.keys.some((k) => k.includes('o30'))).toBe(true); // C
  });

  // Defer-dispose policy: cached materials are SHARED and attached
  // directly to live meshes (only the colormap / layers-panel paths
  // clone), so eviction must not dispose them — a disposed-but-rendered
  // material silently stops receiving updateCameraParams and renders
  // with stale resolution/FOV/nearCull after the next resize. Eviction
  // only drops the cache entry; the material stays registered (camera
  // updates keep flowing) and is disposed at manager dispose().
  it('does NOT dispose the evicted material at eviction time (may still be on a mesh)', () => {
    config.dataLoading.performance.materialCacheMaxSize = 1;
    const mm = new MaterialManager();
    const a = mm.getLineMaterial(baseProps({ opacity: 0.1 }));
    let aDisposed = false;
    const origDispose = a.dispose.bind(a);
    a.dispose = () => {
      aDisposed = true;
      origDispose();
    };
    // Force an insert past the cap → A evicts from the cache.
    mm.getLineMaterial(baseProps({ opacity: 0.2 }));
    expect(aDisposed).toBe(false);
    // Teardown still cleans it up.
    mm.dispose();
    expect(aDisposed).toBe(true);
  });

  it('keeps the evicted material registered for camera updates; cache stays bounded', () => {
    config.dataLoading.performance.materialCacheMaxSize = 1;
    const mm = new MaterialManager();
    const a = mm.getLineMaterial(baseProps({ opacity: 0.1 }));
    expect(mm.getCacheStats().totalRegistered).toBe(1);
    mm.getLineMaterial(baseProps({ opacity: 0.2 })); // evicts A from the cache
    const stats = mm.getCacheStats();
    expect(stats.lineMaterials).toBe(1); // cache bound holds
    expect(stats.evictions).toBe(1);
    expect(stats.totalRegistered).toBe(2); // A still registered

    // A (potentially still attached to a mesh) keeps receiving camera
    // updates — including nearCull.
    mm.updateCameraParams(0.9, new THREE.Vector2(640, 480), false, 0.33);
    expect(
      (a as unknown as { uniforms: { uResolution: { value: THREE.Vector2 } } }).uniforms.uResolution
        .value.x
    ).toBe(640);
  });

  it('disabled (maxSize=0) keeps unbounded behavior', () => {
    config.dataLoading.performance.materialCacheMaxSize = 0;
    const mm = new MaterialManager();
    for (let i = 0; i < 25; i++) {
      mm.getLineMaterial(baseProps({ opacity: i / 25 }));
    }
    expect(mm.getCacheStats().lineMaterials).toBe(25);
    expect(mm.getCacheStats().evictions).toBe(0);
  });

  it('per-node point materials never populate or evict the line cache', () => {
    // Point materials are PER NODE (uncached) since the texture-storage
    // migration: creating them must not touch the line cache, and the
    // point cache map must stay permanently empty.
    config.dataLoading.performance.materialCacheMaxSize = 1;
    const mm = new MaterialManager();
    mm.getLineMaterial(baseProps({ opacity: 0.1 }));
    mm.getPointMaterial(baseProps({ opacity: 0.1 }));
    mm.getPointMaterial(baseProps({ opacity: 0.2 }));
    const stats = mm.getCacheStats();
    expect(stats.lineMaterials).toBe(1); // line entry untouched
    expect(stats.pointMaterials).toBe(0); // per-node — never cached
    expect(stats.evictions).toBe(0); // per-node creates never evict
  });

  it('getCacheStats exposes the configured maxSize', () => {
    config.dataLoading.performance.materialCacheMaxSize = 7;
    const mm = new MaterialManager();
    expect(mm.getCacheStats().maxSize).toBe(7);
  });

  // [rendering.md/G12][P5] maxSize=1 + re-insert SAME key. With a unit
  // cap, every distinct insert evicts the prior; a cache HIT (same key)
  // must NOT trigger an eviction — it merely promotes the existing entry.
  // A mutant that ran the eviction loop on every getX call (regardless
  // of whether it was a hit or miss) would inflate the eviction count
  // and dispose the live material, breaking the hot-path.
  it('maxSize=1: repeated get of the SAME key is a cache hit with no eviction', () => {
    config.dataLoading.performance.materialCacheMaxSize = 1;
    const mm = new MaterialManager();
    const props = baseProps({ opacity: 0.42 });

    const m1 = mm.getLineMaterial(props);
    // Spy dispose to verify it does NOT fire on the cache hit.
    let m1Disposed = false;
    const origDispose = m1.dispose.bind(m1);
    m1.dispose = () => {
      m1Disposed = true;
      origDispose();
    };

    const m2 = mm.getLineMaterial(props);
    const m3 = mm.getLineMaterial(props);

    // Same key returns the same material instance (cache HIT path).
    expect(m2).toBe(m1);
    expect(m3).toBe(m1);
    // Dispose was never called on the live entry.
    expect(m1Disposed).toBe(false);
    // No evictions accumulated across the three hits.
    const stats = mm.getCacheStats();
    expect(stats.lineMaterials).toBe(1);
    expect(stats.evictions).toBe(0);
  });
});
