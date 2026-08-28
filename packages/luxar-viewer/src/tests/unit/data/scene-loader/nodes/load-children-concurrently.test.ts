import { afterEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import {
  EAGER_CHILD_LOAD_CONCURRENCY,
  loadChildrenConcurrently,
  type LoadSceneChildren,
} from '../../../../../data/scene-loader/nodes/load-children-concurrently';
import type { SceneNode } from '../../../../../data/data-loader-types';
import { log, Modules } from '../../../../../utils/log';
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

function makeLineChildren(
  count: number,
  nVertices: number,
  nSegments?: number,
  ndim?: number
): SceneNode[] {
  return Array.from({ length: count }, (_, index) => ({
    path: `/group/lines_${index}`,
    type: 'lines',
    attrs: {
      n_vertices: nVertices,
      ...(nSegments === undefined ? {} : { n_segments: nSegments }),
      ...(ndim === undefined ? {} : { ndim }),
    },
    hasSpatialIndex: true,
    children: [],
  }));
}

function makeGSplatChildren(count: number, nSplats: number): SceneNode[] {
  return Array.from({ length: count }, (_, index) => ({
    path: `/group/gsplats_${index}`,
    type: 'gsplats',
    attrs: { n_splats: nSplats },
    hasSpatialIndex: true,
    children: [],
  }));
}

function makeStubLoc() {
  return { resolve: vi.fn(() => makeStubLoc()) } as never;
}

const originalPerformanceMemory = Object.getOwnPropertyDescriptor(performance, 'memory');

function setHeapLimitBytes(bytes: number): void {
  Object.defineProperty(performance, 'memory', {
    configurable: true,
    value: { jsHeapSizeLimit: bytes },
  });
}

afterEach(() => {
  if (originalPerformanceMemory) {
    Object.defineProperty(performance, 'memory', originalPerformanceMemory);
  } else {
    delete (performance as Performance & { memory?: unknown }).memory;
  }
});

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

  it('scales the working-set budget down on a small measured heap', async () => {
    setHeapLimitBytes(512 * 1024 * 1024);
    const children = makeLineChildren(2, 300_000);
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
    await loadPromise;
  });

  it('caps eager line admission on a large measured heap', async () => {
    setHeapLimitBytes(8 * 1024 * 1024 * 1024);
    const children = makeLineChildren(9, 1_630_000, 1_280_000);
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

    await vi.waitFor(() => expect(started.length).toBeGreaterThan(0));
    expect(started).toHaveLength(1);
    release();
    await loadPromise;
    expect(started).toEqual(children.map((child) => child.path));
  });

  it('logs when a line child waits for working-set admission', async () => {
    const children = makeLineChildren(2, 1_000_000);
    const querySpy = vi.spyOn(log, 'query').mockImplementation(() => {});
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const loadPromise = loadChildrenConcurrently(
      children,
      new THREE.Group(),
      makeStubLoc(),
      makeTestNodeBuildCtx(),
      async () => gate
    );

    await vi.waitFor(() =>
      expect(querySpy).toHaveBeenCalledWith(
        Modules.SCENE_LOADER,
        expect.stringContaining(children[1].path)
      )
    );
    expect(querySpy).toHaveBeenCalledWith(
      Modules.SCENE_LOADER,
      expect.stringMatching(/charged .* active .* budget/)
    );
    release();
    await loadPromise;
  });

  it('warns when a lines node has no usable vertex count', async () => {
    const child = makeLineChildren(1, 1_000)[0];
    delete child.attrs.n_vertices;
    const warningSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});

    await loadChildrenConcurrently(
      [child],
      new THREE.Group(),
      makeStubLoc(),
      makeTestNodeBuildCtx(),
      async () => undefined
    );

    expect(warningSpy).toHaveBeenCalledWith(
      Modules.SCENE_LOADER,
      expect.stringContaining(child.path)
    );
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

  it('keeps large gsplat siblings at the existing eight-wide concurrency', async () => {
    const children = makeGSplatChildren(EAGER_CHILD_LOAD_CONCURRENCY + 1, 10_000_000);
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

  it('admits a small line sibling past a large waiter that does not fit yet', async () => {
    const children = [
      ...makeLineChildren(2, 500_000),
      { ...makeLineChildren(1, 10_000)[0], path: '/group/lines_small' },
    ];
    const started: string[] = [];
    const releases = new Map<string, () => void>();
    const loadPromise = loadChildrenConcurrently(
      children,
      new THREE.Group(),
      makeStubLoc(),
      makeTestNodeBuildCtx(),
      async (child) => {
        started.push(child.path);
        await new Promise<void>((resolve) => releases.set(child.path, resolve));
      }
    );

    await vi.waitFor(() => expect(started).toEqual([children[0].path, children[2].path]));
    releases.get(children[2].path)?.();
    expect(started).toEqual([children[0].path, children[2].path]);
    releases.get(children[0].path)?.();
    await vi.waitFor(() => expect(started).toHaveLength(3));
    releases.get(children[1].path)?.();
    await loadPromise;
  });

  it('charges declared segment pressure independently of vertex count', async () => {
    const children = makeLineChildren(2, 10_000, 1_500_000);
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
    await loadPromise;
  });

  it('includes uncapped nD position growth in the line working-set estimate', async () => {
    const children = makeLineChildren(2, 250_000, 250_000, 32);
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
    await loadPromise;
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

  it('does not start a memory-blocked sibling after an earlier load fails', async () => {
    const children = makeLineChildren(2, 1_000_000);
    const ctx = makeTestNodeBuildCtx();
    const error = new Error('broken first line load');
    const started: string[] = [];
    let rejectFirst!: () => void;
    const failureGate = new Promise<void>((resolve) => {
      rejectFirst = resolve;
    });
    const loadPromise = loadChildrenConcurrently(
      children,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      async (child) => {
        started.push(child.path);
        if (child === children[0]) {
          await failureGate;
          throw error;
        }
      }
    );
    const rejection = expect(loadPromise).rejects.toBe(error);

    await vi.waitFor(() => expect(started).toEqual([children[0].path]));
    rejectFirst();
    await rejection;
    expect(started).toEqual([children[0].path]);

    const reloaded: string[] = [];
    await loadChildrenConcurrently(
      [children[1]],
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      async (child) => {
        reloaded.push(child.path);
      }
    );
    expect(reloaded).toEqual([children[1].path]);
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
