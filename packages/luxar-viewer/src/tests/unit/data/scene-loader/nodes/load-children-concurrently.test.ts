import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import {
  EAGER_CHILD_LOAD_CONCURRENCY,
  loadChildrenConcurrently,
  type LoadSceneChildren,
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

function makeLineChildren(count: number, nVertices: number): SceneNode[] {
  return Array.from({ length: count }, (_, index) => ({
    path: `/group/lines_${index}`,
    type: 'lines',
    attrs: { n_vertices: nVertices },
    hasSpatialIndex: true,
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

  it('reparents loaded objects without dispatching removal events', async () => {
    const parent = new THREE.Group();
    const child = makeChildren(1)[0];
    const removed = vi.fn();

    await loadChildrenConcurrently(
      [child],
      parent,
      makeStubLoc(),
      makeTestNodeBuildCtx(),
      async (node, slot) => {
        const object = new THREE.Group();
        object.name = node.path;
        object.addEventListener('removed', removed);
        slot.add(object);
      }
    );

    expect(removed).not.toHaveBeenCalled();
    expect(parent.children[0].parent).toBe(parent);
  });

  it('serializes million-vertex line siblings to bound their combined working set', async () => {
    const children = makeLineChildren(3, 1_000_000);
    const started: string[] = [];
    const releases: Array<() => void> = [];
    const loadPromise = loadChildrenConcurrently(
      children,
      new THREE.Group(),
      makeStubLoc(),
      makeTestNodeBuildCtx(),
      async (child) => {
        started.push(child.path);
        await new Promise<void>((resolve) => releases.push(resolve));
      }
    );

    await vi.waitFor(() => expect(started).toHaveLength(1));
    releases.shift()?.();
    await vi.waitFor(() => expect(started).toHaveLength(2));
    releases.shift()?.();
    await vi.waitFor(() => expect(started).toHaveLength(3));
    releases.shift()?.();
    await loadPromise;
  });

  it('keeps small line siblings at the existing eight-wide concurrency', async () => {
    const children = makeLineChildren(EAGER_CHILD_LOAD_CONCURRENCY + 1, 10_000);
    const started: string[] = [];
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const loadPromise = loadChildrenConcurrently(
      children,
      new THREE.Group(),
      makeStubLoc(),
      makeTestNodeBuildCtx(),
      async (child) => {
        started.push(child.path);
        await gate;
      }
    );

    await vi.waitFor(() => expect(started).toHaveLength(EAGER_CHILD_LOAD_CONCURRENCY));
    release();
    await loadPromise;
    expect(started).toEqual(children.map((child) => child.path));
  });

  it('releases a large child reservation when its load rejects', async () => {
    const child = makeLineChildren(1, 1_000_000)[0];
    const ctx = makeTestNodeBuildCtx();
    const error = new Error('broken line load');

    await expect(
      loadChildrenConcurrently([child], new THREE.Group(), makeStubLoc(), ctx, async () => {
        throw error;
      })
    ).rejects.toBe(error);

    const reloaded: string[] = [];
    await loadChildrenConcurrently([child], new THREE.Group(), makeStubLoc(), ctx, async (node) => {
      reloaded.push(node.path);
    });
    expect(reloaded).toEqual([child.path]);
  });

  it('shares the working-set budget across nested parent pools', async () => {
    const rootChildren = makeChildren(2);
    rootChildren[0].children = makeLineChildren(1, 1_000_000);
    rootChildren[1].children = makeLineChildren(1, 1_000_000).map((child) => ({
      ...child,
      path: '/group/lines_nested_1',
    }));
    const ctx = makeTestNodeBuildCtx();
    const started: string[] = [];
    const releases: Array<() => void> = [];
    const loadNode: LoadSceneChildren = async (node, parent, loc): Promise<void> => {
      if (node.type === 'group') {
        await loadChildrenConcurrently(node.children ?? [], parent, loc, ctx, loadNode);
        return;
      }
      started.push(node.path);
      await new Promise<void>((resolve) => releases.push(resolve));
    };

    const loadPromise = loadChildrenConcurrently(
      rootChildren,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadNode
    );

    await vi.waitFor(() => expect(started).toHaveLength(1));
    releases.shift()?.();
    await vi.waitFor(() => expect(started).toHaveLength(2));
    releases.shift()?.();
    await loadPromise;
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
