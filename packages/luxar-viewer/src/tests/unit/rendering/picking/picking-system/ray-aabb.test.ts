/**
 * Unit tests for the ray-AABB culling helpers in `picking-system/ray-aabb.ts`.
 *
 * These functions are pure (THREE objects in, booleans / Box3s out) and
 * fully jsdom-safe — no GL context needed.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  getOrComputeWorldBox,
  rayHitsAnyNode,
  invalidateBoxCache,
} from '../../../../../rendering/picking/picking-system/ray-aabb';
import type { PickNodeEntry } from '../../../../../rendering/picking/picking-system/registration';

/** Build a pick-node entry whose main mesh has a unit-cube AABB at the origin. */
function makeEntry(
  options: { translation?: THREE.Vector3; withBoundingBox?: boolean } = {}
): PickNodeEntry {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  if (options.withBoundingBox !== false) {
    geometry.computeBoundingBox();
  } else {
    geometry.boundingBox = null;
  }
  const main = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  if (options.translation) {
    main.position.copy(options.translation);
    main.updateMatrixWorld(true);
  } else {
    main.updateMatrixWorld(true);
  }
  // Pick node is unused by these helpers — the AABB comes from the main mesh's geometry.
  const pick = new THREE.Object3D();
  return { main, pick };
}

describe('getOrComputeWorldBox', () => {
  it('populates the cache on a miss and returns the world-space AABB', () => {
    const cache = new Map<number, THREE.Box3>();
    const entry = makeEntry({ translation: new THREE.Vector3(10, 0, 0) });

    const box = getOrComputeWorldBox(1, entry, cache);

    expect(box).not.toBeNull();
    expect(cache.size).toBe(1);
    expect(cache.get(1)).toBe(box); // same instance is stored
    // Verify it's the *world* AABB, not the local one (translation applied)
    expect(box!.min.x).toBeCloseTo(9.5, 5);
    expect(box!.max.x).toBeCloseTo(10.5, 5);
  });

  it('returns the cached AABB on a hit without recomputing', () => {
    const cache = new Map<number, THREE.Box3>();
    const entry = makeEntry();
    const first = getOrComputeWorldBox(1, entry, cache);

    // Mutate the geometry's bounding box: if the second call recomputed,
    // the cached instance would change. Cache hit must return the same
    // Box3 reference regardless.
    (entry.main as THREE.Mesh).geometry.boundingBox = new THREE.Box3(
      new THREE.Vector3(100, 100, 100),
      new THREE.Vector3(101, 101, 101)
    );

    const second = getOrComputeWorldBox(1, entry, cache);
    expect(second).toBe(first); // exact instance identity proves no recompute
  });

  it('returns null when the geometry has no bounding box', () => {
    const cache = new Map<number, THREE.Box3>();
    const entry = makeEntry({ withBoundingBox: false });
    expect(getOrComputeWorldBox(1, entry, cache)).toBeNull();
    expect(cache.size).toBe(0);
  });

  it('uses boundingBox verbatim — the footprint is already baked in by the producers', () => {
    // The three-geometry invariant: every producer bakes its rendered
    // footprint into boundingBox at creation/commit (points expand by the
    // disc radius, lines by max half-width, gsplats by maxRowNorm ×
    // truncation). So the cull must NOT add its own margin — it transforms
    // the local box to world space as-is. Here the geometry's box is a unit
    // cube at the origin; the world box matches it exactly.
    const cache = new Map<number, THREE.Box3>();
    const entry = makeEntry();

    const box = getOrComputeWorldBox(1, entry, cache);

    expect(box).not.toBeNull();
    expect(box!.min.x).toBeCloseTo(-0.5, 5);
    expect(box!.max.x).toBeCloseTo(0.5, 5);
    expect(box!.max.y).toBeCloseTo(0.5, 5);
  });

  it('does not add any margin regardless of userData (no footprint re-expansion)', () => {
    // Guard against re-introducing a cull-time margin: even if a geometry
    // carries radius-like userData, the cull ignores it — boundingBox is
    // the single source of truth for the footprint.
    const cache = new Map<number, THREE.Box3>();
    const entry = makeEntry();
    (entry.main as THREE.Mesh).geometry.userData = { radiusScale: 1.0, maxActualRadius: 50 };

    const box = getOrComputeWorldBox(1, entry, cache);

    expect(box).not.toBeNull();
    // Unchanged unit cube: no extra 50-unit margin.
    expect(box!.min.x).toBeCloseTo(-0.5, 5);
    expect(box!.max.x).toBeCloseTo(0.5, 5);
  });
});

describe('rayHitsAnyNode', () => {
  it('returns true on the first hit and short-circuits remaining nodes', () => {
    const cache = new Map<number, THREE.Box3>();
    const nodeMap = new Map<number, PickNodeEntry>();
    nodeMap.set(1, makeEntry({ translation: new THREE.Vector3(0, 0, 0) }));
    nodeMap.set(2, makeEntry({ translation: new THREE.Vector3(100, 0, 0) }));

    // Ray pointing through node 1's AABB at the origin (along -Z toward origin)
    const ray = new THREE.Ray(new THREE.Vector3(0, 0, 10), new THREE.Vector3(0, 0, -1));

    expect(rayHitsAnyNode(ray, nodeMap, cache)).toBe(true);
    // Short-circuit semantics: only the first node's box should have been cached.
    expect(cache.has(1)).toBe(true);
    expect(cache.has(2)).toBe(false);
  });

  it('returns false when the ray misses every registered AABB', () => {
    const cache = new Map<number, THREE.Box3>();
    const nodeMap = new Map<number, PickNodeEntry>();
    nodeMap.set(1, makeEntry({ translation: new THREE.Vector3(0, 0, 0) }));
    nodeMap.set(2, makeEntry({ translation: new THREE.Vector3(50, 0, 0) }));

    // Ray that misses both unit cubes (parallel +Y, far away in X)
    const ray = new THREE.Ray(new THREE.Vector3(1000, 0, 0), new THREE.Vector3(0, 1, 0));

    expect(rayHitsAnyNode(ray, nodeMap, cache)).toBe(false);
    // Both nodes were probed → both cached (we only short-circuit on hit)
    expect(cache.size).toBe(2);
  });

  it('treats nodes whose geometry has no bounding box as non-hits', () => {
    const cache = new Map<number, THREE.Box3>();
    const nodeMap = new Map<number, PickNodeEntry>();
    nodeMap.set(1, makeEntry({ withBoundingBox: false }));

    const ray = new THREE.Ray(new THREE.Vector3(0, 0, 10), new THREE.Vector3(0, 0, -1));
    expect(rayHitsAnyNode(ray, nodeMap, cache)).toBe(false);
  });
});

describe('invalidateBoxCache', () => {
  it('with a pickId, drops only that entry', () => {
    const cache = new Map<number, THREE.Box3>();
    cache.set(1, new THREE.Box3());
    cache.set(2, new THREE.Box3());

    invalidateBoxCache(cache, 1);

    expect(cache.has(1)).toBe(false);
    expect(cache.has(2)).toBe(true);
  });

  it('with no pickId, clears the entire cache', () => {
    const cache = new Map<number, THREE.Box3>();
    cache.set(1, new THREE.Box3());
    cache.set(2, new THREE.Box3());
    cache.set(3, new THREE.Box3());

    invalidateBoxCache(cache);

    expect(cache.size).toBe(0);
  });

  it('is a no-op for an unknown pickId', () => {
    const cache = new Map<number, THREE.Box3>();
    cache.set(1, new THREE.Box3());
    expect(() => invalidateBoxCache(cache, 9999)).not.toThrow();
    expect(cache.size).toBe(1);
  });

  // [rendering.md/G13][P5] pickId=0 is the background-sentinel in the
  // pick-buffer (no node maps to 0). The cache must not store an entry
  // for 0 in normal use, but the invalidate API should still treat 0 as
  // a regular id — i.e. dropping a stray 0-entry should work just as it
  // does for non-zero ids, and (mirroring the previous "unknown id"
  // contract) a no-op when no such entry exists.
  it('invalidateBoxCache(cache, 0) is a no-op when no 0-entry exists', () => {
    const cache = new Map<number, THREE.Box3>();
    cache.set(1, new THREE.Box3());
    cache.set(2, new THREE.Box3());
    expect(() => invalidateBoxCache(cache, 0)).not.toThrow();
    expect(cache.size).toBe(2);
    expect(cache.has(1)).toBe(true);
    expect(cache.has(2)).toBe(true);
  });

  it('invalidateBoxCache(cache, 0) drops the 0-entry when one is present', () => {
    // Defensive path: if a 0-entry somehow leaked in (e.g. a debugging
    // tool injected one), the invalidation API drops it the same way
    // as any other id. Pins the no-special-casing-of-0 contract.
    const cache = new Map<number, THREE.Box3>();
    cache.set(0, new THREE.Box3());
    cache.set(1, new THREE.Box3());
    invalidateBoxCache(cache, 0);
    expect(cache.size).toBe(1);
    expect(cache.has(0)).toBe(false);
    expect(cache.has(1)).toBe(true);
  });
});
