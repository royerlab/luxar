import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import {
  EAGER_CHILD_LOAD_CONCURRENCY,
  loadChildrenConcurrently,
} from '../../../../../data/scene-loader/nodes/load-children-concurrently';
import type { SceneNode } from '../../../../../data/data-loader-types';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';

function makeChildren(count: number): SceneNode[] {
  return Array.from({ length: count }, (_, index) => ({
    path: `/group/child_${index}`,
    type: 'group',
    attrs: {},
    hasSpatialIndex: false,
    children: [],
  }));
}

function makeStubLoc() {
  return { resolve: vi.fn(() => makeStubLoc()) } as never;
}

describe('loadChildrenConcurrently', () => {
  it('keeps placeholders reachable from the real scene while their loads are in flight', async () => {
    const parent = new THREE.Group();
    const child = makeChildren(1)[0];
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const loadPromise = loadChildrenConcurrently(
      [child],
      parent,
      makeStubLoc(),
      makeTestNodeBuildCtx(),
      async (node, slot) => {
        const placeholder = new THREE.Group();
        placeholder.name = node.path;
        slot.add(placeholder);
        expect(parent.getObjectByName(node.path)).toBe(placeholder);
        await gate;
      }
    );

    await Promise.resolve();
    release();
    await loadPromise;

    expect(parent.children.map((object) => object.name)).toEqual([child.path]);
  });

  it('stops queued work, settles active loads, and removes slots after an unexpected error', async () => {
    const children = makeChildren(EAGER_CHILD_LOAD_CONCURRENCY + 2);
    const parent = new THREE.Group();
    const error = new Error('broken child');
    const started: string[] = [];

    await expect(
      loadChildrenConcurrently(
        children,
        parent,
        makeStubLoc(),
        makeTestNodeBuildCtx(),
        async (child, slot) => {
          started.push(child.path);
          if (child === children[0]) throw error;
          const object = new THREE.Group();
          object.name = child.path;
          slot.add(object);
        }
      )
    ).rejects.toBe(error);

    expect(started).toEqual(children.slice(0, EAGER_CHILD_LOAD_CONCURRENCY).map((c) => c.path));
    expect(parent.children.map((object) => object.name)).toEqual(
      children.slice(1, EAGER_CHILD_LOAD_CONCURRENCY).map((child) => child.path)
    );
    expect(parent.children.every((object) => object.name.length > 0)).toBe(true);
  });
});
