/**
 * Unit tests for SceneBoundsCache + computeBoundsFromMetadata.
 *
 * Extracted from SceneManager in Step 5 of the scene-folder layout
 * overhaul. These tests pin the cache-invalidation contract, the
 * lazy-recompute behaviour, and the metadata-scan that
 * `getSceneBoundsFromMetadata()` used to do inline on the class.
 *
 * nD→3D display-dim projection is tested separately in
 * `bounds-math.test.ts` (`projectBoundsToDisplayDims`).
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  SceneBoundsCache,
  computeBoundsFromMetadata,
  findPositionBoundsInScene,
} from '../../../../../scene/scene-manager/clipping/scene-bounds-cache';
import { sceneDimsManager } from '../../../../../scene/scene-dims-manager';

function makeSceneWithBounds(min: number[], max: number[]): THREE.Scene {
  const scene = new THREE.Scene();
  scene.userData = { ...scene.userData, positionBounds: { min, max } };
  return scene;
}

describe('SceneBoundsCache', () => {
  beforeEach(() => {
    // Default behaviour without dim init: getDims() returns null and
    // computeBoundsFromMetadata falls back to display dims [0,1,2].
    sceneDimsManager.reset();
  });

  it('returns null bounds before ensure() is called', () => {
    const cache = new SceneBoundsCache();
    expect(cache.getBounds()).toBeNull();
    expect(cache.getSphere()).toBeNull();
    expect(cache.getNearCull()).toBe(0.1);
  });

  it('computes bounds, sphere and near-cull on first ensure()', () => {
    const cache = new SceneBoundsCache();
    const scene = makeSceneWithBounds([0, 0, 0], [10, 20, 30]);

    cache.ensure(scene);

    const bounds = cache.getBounds();
    expect(bounds).not.toBeNull();
    expect(bounds!.min).toEqual({ x: 0, y: 0, z: 0 });
    expect(bounds!.max).toEqual({ x: 10, y: 20, z: 30 });

    const sphere = cache.getSphere();
    expect(sphere).not.toBeNull();
    expect(sphere!.radius).toBeGreaterThan(0);

    // Near-cull is ~0.1% of the diagonal length.
    const diagonal = Math.sqrt(10 ** 2 + 20 ** 2 + 30 ** 2);
    expect(cache.getNearCull()).toBeCloseTo(diagonal * 0.001, 5);
  });

  it('is idempotent on repeated ensure() calls (no recompute)', () => {
    const cache = new SceneBoundsCache();
    const scene = makeSceneWithBounds([0, 0, 0], [10, 10, 10]);

    cache.ensure(scene);
    const boundsFirst = cache.getBounds();

    // Mutate the scene metadata — second ensure() must NOT pick it up.
    scene.userData.positionBounds = { min: [-1000, -1000, -1000], max: [1000, 1000, 1000] };
    cache.ensure(scene);

    expect(cache.getBounds()).toBe(boundsFirst);
    expect(cache.getBounds()!.max.x).toBe(10);
  });

  it('invalidate() clears cached values; next ensure() recomputes', () => {
    const cache = new SceneBoundsCache();
    const scene = makeSceneWithBounds([0, 0, 0], [10, 10, 10]);

    cache.ensure(scene);
    expect(cache.getBounds()).not.toBeNull();

    cache.invalidate();
    expect(cache.getBounds()).toBeNull();
    expect(cache.getSphere()).toBeNull();
    expect(cache.getNearCull()).toBe(0.1);

    // After invalidate, ensure() should recompute from current metadata.
    scene.userData.positionBounds = { min: [0, 0, 0], max: [5, 5, 5] };
    cache.ensure(scene);
    expect(cache.getBounds()!.max.x).toBe(5);
  });

  it('leaves cache empty when scene has no positionBounds metadata', () => {
    const cache = new SceneBoundsCache();
    const scene = new THREE.Scene();
    cache.ensure(scene);
    expect(cache.getBounds()).toBeNull();
  });
});

describe('computeBoundsFromMetadata', () => {
  beforeEach(() => {
    sceneDimsManager.reset();
  });

  it('returns null when no positionBounds in scene graph', () => {
    expect(computeBoundsFromMetadata(new THREE.Scene())).toBeNull();
  });

  it('uses default display dims [0,1,2] when sceneDimsManager has no metadata', () => {
    const scene = makeSceneWithBounds([0, 0, 0], [10, 20, 30]);
    const bounds = computeBoundsFromMetadata(scene);
    expect(bounds).toEqual({
      min: { x: 0, y: 0, z: 0 },
      max: { x: 10, y: 20, z: 30 },
    });
  });
});

describe('findPositionBoundsInScene', () => {
  it('returns null when no userData.positionBounds anywhere', () => {
    expect(findPositionBoundsInScene(new THREE.Scene())).toBeNull();
  });

  it('finds bounds on the root scene itself', () => {
    const scene = makeSceneWithBounds([0, 0, 0], [1, 2, 3]);
    expect(findPositionBoundsInScene(scene)).toEqual({ min: [0, 0, 0], max: [1, 2, 3] });
  });

  it('finds bounds on a child node when root has none', () => {
    const scene = new THREE.Scene();
    const child = new THREE.Group();
    child.userData = { positionBounds: { min: [1, 1, 1], max: [9, 9, 9] } };
    scene.add(child);
    expect(findPositionBoundsInScene(scene)).toEqual({ min: [1, 1, 1], max: [9, 9, 9] });
  });

  it('returns first match (no merging) when bounds exist at multiple levels', () => {
    const scene = makeSceneWithBounds([0, 0, 0], [10, 10, 10]);
    const child = new THREE.Group();
    child.userData = { positionBounds: { min: [-100, -100, -100], max: [100, 100, 100] } };
    scene.add(child);
    // Root is visited first by traverse; result is root's bounds.
    expect(findPositionBoundsInScene(scene)).toEqual({ min: [0, 0, 0], max: [10, 10, 10] });
  });

  it('ignores userData entries that are not in {min, max} shape', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: 'wrong', max: 'shape' } };
    expect(findPositionBoundsInScene(scene)).toBeNull();
  });
});
