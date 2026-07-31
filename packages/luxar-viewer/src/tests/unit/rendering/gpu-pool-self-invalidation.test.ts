/**
 * GPUBufferPool self-invalidation on out-of-band geometry dispose.
 *
 * The scene-graph disposal helpers (`scene-disposal.ts`) are pure over a
 * THREE.Scene and hold NO pool reference, so a dataset switch / teardown
 * calls `geometry.dispose()` directly on meshes that still carry
 * pool-owned geometries (`userData.luxarPooled === true`). This bypasses
 * the pool's release/evict methods.
 *
 * Regression: without the pool's self-invalidating dispose listener the
 * disposed geometry stayed recorded in `activeBuffers`, so a later
 * `acquire*Geometry(sameNodeId, …)` handed the already-disposed geometry
 * back as reusable (use-after-dispose) and the resident-byte accounting
 * over-counted. This suite proves the fix for Points, Lines, AND GSplats.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import {
  disposeObjectTree,
  clearLoadedSceneContent,
} from '../../../scene/scene-manager/render-pipeline/scene-disposal';
import { getPointTexture } from '../../../rendering/point-geometry';
import { getLineTexture } from '../../../rendering/line-geometry';
import { getSplatTexture } from '../../../rendering/gsplat-geometry';

type GeomType = 'points' | 'lines' | 'gsplats';

interface TypeCase {
  type: GeomType;
  acquire: (pool: GPUBufferPool, nodeId: string, count: number) => THREE.InstancedBufferGeometry;
  /**
   * Sanity check that the geometry was constructed with its element
   * storage attached. Note this is NOT a disposal discriminator — the
   * storage's dispose listener frees the GPU texture but leaves the
   * `userData` reference in place — so liveness is carried by the
   * `luxarInvalidated`-unset assertion, and this only guards against a
   * caller ever getting back a storage-less geometry.
   */
  hasStorage: (geometry: THREE.BufferGeometry) => boolean;
}

const CASES: TypeCase[] = [
  {
    type: 'points',
    acquire: (pool, nodeId, count) => pool.acquirePointsGeometry(nodeId, count),
    hasStorage: (g) => getPointTexture(g) != null,
  },
  {
    type: 'lines',
    acquire: (pool, nodeId, count) => pool.acquireLinesGeometry(nodeId, count),
    hasStorage: (g) => getLineTexture(g) != null,
  },
  {
    type: 'gsplats',
    acquire: (pool, nodeId, count) => pool.acquireGSplatsGeometry(nodeId, count),
    hasStorage: (g) => getSplatTexture(g) != null,
  },
];

describe('GPUBufferPool self-invalidation on out-of-band dispose', () => {
  let pool: GPUBufferPool;

  beforeEach(() => {
    // Byte budget disabled so pooled buffers are retained (count-only) —
    // keeps the accounting assertions deterministic.
    pool = new GPUBufferPool(20, 300, 5, () => 0);
  });

  for (const { type, acquire, hasStorage } of CASES) {
    describe(`${type}`, () => {
      it('drops the active entry and re-acquire returns a fresh, non-disposed geometry', () => {
        const geometry = acquire(pool, `/${type}`, 100);
        expect(pool.getStats().activeBuffers).toBe(1);

        let selfDisposeEvents = 0;
        geometry.addEventListener('dispose', () => selfDisposeEvents++);

        // Out-of-band dispose (what scene-disposal does to the mesh).
        geometry.dispose();
        expect(selfDisposeEvents).toBe(1);
        expect(geometry.userData.luxarInvalidated).toBe(true);

        // The pool no longer counts the disposed geometry as active.
        expect(pool.getStats().activeBuffers).toBe(0);

        // Re-acquire for the SAME node must NOT return the disposed one.
        const reacquired = acquire(pool, `/${type}`, 50);
        expect(reacquired).not.toBe(geometry);

        // Liveness is carried by the flag: the pool stamps
        // `luxarInvalidated` on every geometry it disposes, so an unset
        // flag proves this one was never disposed. (Attaching a dispose
        // listener AFTER the fact and asserting 0 would be tautological —
        // three.js dispatches `dispose` synchronously.) `hasStorage` is a
        // secondary sanity check that the fresh geometry has its storage.
        expect(reacquired.userData.luxarInvalidated).toBeFalsy();
        expect(hasStorage(reacquired)).toBe(true);
      });

      it('resident-byte accounting drops back after the out-of-band dispose', () => {
        const geometry = acquire(pool, `/${type}`, 1000);
        const bytesActive = pool.getResidentBytes();
        expect(bytesActive).toBeGreaterThan(0);
        expect(pool.getStats().totalBytes).toBe(bytesActive);

        geometry.dispose();

        // Derived from activeBuffers membership: dropping the entry is the
        // whole correction, so residency falls to zero.
        expect(pool.getResidentBytes()).toBe(0);
        expect(pool.getStats().totalBytes).toBe(0);
        expect(pool.getStats().activeBytes).toBe(0);
      });

      it('multi-node: disposing ONE geometry leaves the OTHER node reusable', () => {
        // Guards against an over-broad clear()-style deletion: the listener
        // must delete only the entry that references THIS geometry.
        const survivor = acquire(pool, `/${type}/survivor`, 100);
        const victim = acquire(pool, `/${type}/victim`, 100);
        expect(pool.getStats().activeBuffers).toBe(2);

        victim.dispose();

        // Only the victim's entry is gone; the survivor's entry stays.
        expect(pool.getStats().activeBuffers).toBe(1);
        expect(pool.activeBuffers.get(`/${type}/survivor`)?.geometry).toBe(survivor);

        // The survivor re-acquires as the SAME geometry (in-place reuse),
        // while the victim re-acquires fresh.
        const survivorReuse = acquire(pool, `/${type}/survivor`, 80);
        expect(survivorReuse).toBe(survivor);
        const victimFresh = acquire(pool, `/${type}/victim`, 80);
        expect(victimFresh).not.toBe(victim);
        expect(victimFresh.userData.luxarInvalidated).toBeFalsy();
      });

      it('grow-then-dispose: disposing the OLD released geometry keeps the NEW active entry (identity match)', () => {
        // A grow is release + reacquire: the old geometry moves to a free
        // bucket and a NEW geometry becomes active under the SAME nodeId.
        // Disposing the OLD one out-of-band must match by IDENTITY (the old
        // geometry) — never by nodeId — so the NEW active entry survives.
        const oldGeom = acquire(pool, `/${type}/grow`, 1000); // capacity ~1500
        const newGeom = acquire(pool, `/${type}/grow`, 4000); // grow → fresh, old pooled
        expect(newGeom).not.toBe(oldGeom);
        expect(pool.activeBuffers.get(`/${type}/grow`)?.geometry).toBe(newGeom);

        oldGeom.dispose(); // dispose the RELEASED (pooled) old geometry

        // The new active entry for this nodeId is untouched…
        expect(pool.activeBuffers.get(`/${type}/grow`)?.geometry).toBe(newGeom);
        // …and a subsequent acquire returns it (in-place reuse), not a fresh one.
        expect(acquire(pool, `/${type}/grow`, 3000)).toBe(newGeom);
      });

      it('free-bucket zombie: a pooled geometry disposed out-of-band is NOT adopted', () => {
        // Exercises the adoptOrAllocate best-fit skip guard: a released
        // geometry (now in a free bucket) disposed out-of-band is flagged
        // invalidated and must not be adopted by a later acquire.
        const zombie = acquire(pool, `/${type}/z1`, 1000);
        // Release through the correct typed helper for this type.
        if (type === 'points') pool.releasePointsGeometry(`/${type}/z1`);
        else if (type === 'lines') pool.releaseLinesGeometry(`/${type}/z1`);
        else pool.releaseGSplatsGeometry(`/${type}/z1`);
        expect(pool.getStats().pooledBuffers).toBe(1);

        zombie.dispose(); // out-of-band dispose while pooled
        expect(zombie.userData.luxarInvalidated).toBe(true);

        // Same capacity would normally best-fit the pooled buffer; the guard
        // must skip the invalidated resident and allocate fresh instead.
        const adopted = acquire(pool, `/${type}/z2`, 1000);
        expect(adopted).not.toBe(zombie);
        expect(adopted.userData.luxarInvalidated).toBeFalsy();
        expect(hasStorage(adopted)).toBe(true);
      });
    });
  }

  it('a full scene teardown (disposeObjectTree) leaves no zombie active entries', () => {
    // Build a small scene graph: a group holding one mesh per geometry
    // type, each mesh carrying a pool-owned geometry.
    const scene = new THREE.Scene();
    const group = new THREE.Group();
    const geoms: THREE.InstancedBufferGeometry[] = [];
    for (const { type, acquire } of CASES) {
      const geom = acquire(pool, `/${type}`, 500);
      geoms.push(geom);
      group.add(new THREE.Mesh(geom, new THREE.MeshBasicMaterial()));
    }
    scene.add(group);
    expect(pool.getStats().activeBuffers).toBe(3);

    // Out-of-band teardown — pure over the scene, no pool reference.
    disposeObjectTree(group);

    // No zombie active entries survive; residency drops to zero.
    expect(pool.getStats().activeBuffers).toBe(0);
    expect(pool.activeBuffers.size).toBe(0);
    expect(pool.getResidentBytes()).toBe(0);
    for (const g of geoms) {
      expect(g.userData.luxarInvalidated).toBe(true);
    }

    // Re-acquire allocates fresh geometries (not the disposed ones).
    const allocsBefore = pool.getStats().allocations;
    for (let i = 0; i < CASES.length; i++) {
      const reacquired = CASES[i].acquire(pool, `/${CASES[i].type}`, 500);
      expect(reacquired).not.toBe(geoms[i]);
    }
    expect(pool.getStats().allocations).toBe(allocsBefore + CASES.length);
  });

  it('clearLoadedSceneContent teardown leaves no zombie active entries', () => {
    const scene = new THREE.Scene();
    const geoms: THREE.InstancedBufferGeometry[] = [];
    for (const { type, acquire } of CASES) {
      const geom = acquire(pool, `/${type}`, 400);
      geoms.push(geom);
      scene.add(new THREE.Mesh(geom, new THREE.MeshBasicMaterial()));
    }
    expect(pool.getStats().activeBuffers).toBe(3);

    const removed = clearLoadedSceneContent(scene);
    expect(removed).toBe(3);

    expect(pool.getStats().activeBuffers).toBe(0);
    expect(pool.getResidentBytes()).toBe(0);

    // A fresh acquire allocates rather than adopting a disposed zombie.
    const fresh = pool.acquirePointsGeometry('/points', 400);
    expect(geoms).not.toContain(fresh);
    expect(fresh.userData.luxarInvalidated).toBeUndefined();
  });

  it('normal release → evict → dispose path is unchanged (listener does not double-count)', () => {
    // evictionFrames=1 so the pooled buffer becomes evictable quickly.
    const evictPool = new GPUBufferPool(20, 1, 5, () => 0);
    const geometry = evictPool.acquirePointsGeometry('n', 1000);

    // Track the geometry's dispose to confirm it fires exactly once even
    // though both the eviction path AND the self-invalidation listener
    // are wired to it.
    let disposeEvents = 0;
    geometry.addEventListener('dispose', () => disposeEvents++);

    evictPool.releasePointsGeometry('n');
    expect(evictPool.getStats().pooledBuffers).toBe(1);
    expect(evictPool.getStats().activeBuffers).toBe(0);

    const evictionsBefore = evictPool.getStats().byType.points.evictions;

    // Advance past evictionFrames and sweep — the normal LRU path disposes
    // the pooled buffer (splices from the bucket BEFORE dispose, so the
    // self-invalidation listener finds nothing in activeBuffers → no-op).
    for (let i = 0; i < 3; i++) evictPool.beginFrame();
    evictPool.evictUnused();

    // The geometry was disposed exactly once by the normal eviction path.
    expect(disposeEvents).toBe(1);
    // Eviction counter advanced by exactly one — the listener did NOT
    // double-count (it never touches eviction stats, and by the time it
    // fires the buffer is already out of both the bucket and activeBuffers).
    expect(evictPool.getStats().byType.points.evictions).toBe(evictionsBefore + 1);
    expect(evictPool.getStats().pooledBuffers).toBe(0);
    expect(evictPool.getStats().activeBuffers).toBe(0);
    expect(evictPool.getResidentBytes()).toBe(0);
  });
});
