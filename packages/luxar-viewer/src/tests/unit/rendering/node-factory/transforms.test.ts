/**
 * Unit tests for `rendering/node-factory/transforms.ts` — the PARENTING
 * contract of `applyTransform`, as opposed to the matrix it installs.
 *
 * `applyTransform` turns `matrixAutoUpdate` OFF (so a directly-installed
 * matrix, shear included, is never recomposed from TRS), forces one synchronous
 * world-matrix update for creation-time consumers, and leaves the world matrix
 * dirty so on-demand updates can re-resolve it after parenting. The renderer's
 * `scene.updateMatrixWorld()` also cascades `force = true` through the whole
 * tree each frame; these tests pin both update paths.
 *
 * Every test applies the transform BEFORE `add()`, matching what
 * `load-scene-nodes.ts` actually does (`applyTransform(group, ...)` and only
 * then `parentThree.add(group)`). A test that parented first would resolve
 * against the real parent immediately and so could not detect a broken
 * cascade at all.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { applyTransform } from '../../../../rendering/node-factory/transforms';

/** Column-major translation matrix, the layout the loader feeds in. */
function translation(x: number, y: number, z: number): number[] {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

/** World-space origin of an object, read from its world matrix. */
function worldPosition(object: THREE.Object3D): number[] {
  const p = new THREE.Vector3();
  object.getWorldPosition(p);
  return [p.x, p.y, p.z];
}

describe('applyTransform — world-matrix propagation after parenting', () => {
  it('composes a parent transform applied before the child is added', () => {
    const scene = new THREE.Scene();
    const parent = new THREE.Group();
    const child = new THREE.Group();

    // The loader's order: transform, THEN parent. Reversing these two lines
    // makes this test pass even against the regression.
    applyTransform(parent, translation(10, 0, 0));
    applyTransform(child, translation(0, 5, 0));
    parent.add(child);
    scene.add(parent);

    scene.updateMatrixWorld();

    // Before the fix this read [0, 5, 0]: the child's forced update ran while
    // it was parentless, and nothing ever re-dirtied it.
    expect(worldPosition(child)).toEqual([10, 5, 0]);
  });

  it('composes through a three-level chain built transform-first', () => {
    const scene = new THREE.Scene();
    const grandparent = new THREE.Group();
    const parent = new THREE.Group();
    const child = new THREE.Group();

    applyTransform(grandparent, translation(1, 0, 0));
    applyTransform(parent, translation(0, 2, 0));
    applyTransform(child, translation(0, 0, 3));
    parent.add(child);
    grandparent.add(parent);
    scene.add(grandparent);

    scene.updateMatrixWorld();

    expect(worldPosition(child)).toEqual([1, 2, 3]);
  });

  it('propagates a parent transform applied after the child is attached', () => {
    const scene = new THREE.Scene();
    const parent = new THREE.Group();
    const child = new THREE.Group();

    applyTransform(child, translation(0, 5, 0));
    parent.add(child);
    scene.add(parent);
    scene.updateMatrixWorld();
    expect(worldPosition(child)).toEqual([0, 5, 0]);

    // A later parent transform must reach the already-settled child: the
    // parent's own recompute forces its subtree regardless of the child's flag.
    applyTransform(parent, translation(10, 0, 0));
    scene.updateMatrixWorld();

    expect(worldPosition(child)).toEqual([10, 5, 0]);
  });

  it('leaves a synchronously readable world matrix for creation-time consumers', () => {
    // Every create-*-node.ts copies `mesh.matrixWorld` into its pick node
    // immediately after applying the transform, with no frame in between.
    const object = new THREE.Group();
    applyTransform(object, translation(7, 8, 9));

    expect(Array.from(object.matrixWorld.elements)).toEqual(translation(7, 8, 9));
  });

  it('re-resolves bounds against a parent added after the transform', () => {
    const parent = new THREE.Group();
    parent.position.x = 100;
    parent.updateMatrixWorld(true);

    const child = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2));
    applyTransform(child, translation(0, 0, 0));
    parent.add(child);

    const bounds = new THREE.Box3().setFromObject(child);

    expect(bounds.min.toArray()).toEqual([99, -1, -1]);
    expect(bounds.max.toArray()).toEqual([101, 1, 1]);
  });
});
