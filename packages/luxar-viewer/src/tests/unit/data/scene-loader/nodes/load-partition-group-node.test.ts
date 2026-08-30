/**
 * Tests for ``loadPartitionGroupNode``.
 *
 * Strategy: the Partition loader recurses children through ``loadSceneNodes``
 * and registers validated child bounds for frustum-only selection. We mock
 * ``loadSceneNodes`` to attach a stub mesh per child, then assert on the
 * THREE-tree shape and registry entry.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

// Caller-injected recursion handle: the partition loader now takes
// loadSceneNodes as an explicit ``loadChildren`` parameter (mirrors
// load-lod-group-node) so tests can pass the mock directly.
const loadSceneNodesMock = vi.fn();

import { loadPartitionGroupNode } from '../../../../../data/scene-loader/nodes/load-partition-group-node';
import { EAGER_CHILD_LOAD_CONCURRENCY } from '../../../../../data/scene-loader/nodes/load-children-concurrently';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';
import { ROOT_ATTR_DOCS, rootAttributes } from '../../../../../types/zarr-documents';
import { log, Modules } from '../../../../../utils/log';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

beforeEach(() => {
  loadSceneNodesMock.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

function makePartNode(
  path: string,
  partType: 'points' | 'gsplats' = 'points',
  extraAttrs: Record<string, unknown> = {}
): SceneNode {
  return {
    path,
    type: partType,
    attrs: {
      type: partType,
      ...extraAttrs,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

function makePartitionGroupNode(
  children: SceneNode[],
  extraAttrs: Record<string, unknown> = {}
): SceneNode {
  return {
    path: '/partition',
    type: 'group',
    attrs: {
      type: 'group',
      kind: 'partition',
      display_type: 'points',
      max_elements: 1000,
      ...extraAttrs,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children,
  };
}

function makeCtx(): NodeBuildCtx {
  const nodeFactory = {
    applyTransform: vi.fn(),
  } as unknown as NodeBuildCtx['nodeFactory'];
  // `lodGroupRegistry` is simply left unset: `loadPartitionGroupNode` never
  // reads it (a partition group has no per-frame level selection to make), so
  // there is no registry path here either to exercise or to stub.
  return makeTestNodeBuildCtx({ nodeFactory });
}

function makeRegistryCtx(registerPartition: ReturnType<typeof vi.fn>): NodeBuildCtx {
  return makeTestNodeBuildCtx({
    nodeFactory: { applyTransform: vi.fn() } as unknown as NodeBuildCtx['nodeFactory'],
    lodGroupRegistry: { registerPartition } as unknown as NodeBuildCtx['lodGroupRegistry'],
  });
}

function makeStubLoc() {
  return { resolve: (_p: string) => ({ resolve: () => ({}) }) } as any;
}

/** Make the mock attach a child mesh to ``parentThree`` so the recursion
 *  produces a complete tree shape. */
function attachStubChildren() {
  loadSceneNodesMock.mockImplementation(async (child: SceneNode, parentThree: THREE.Object3D) => {
    const obj = new THREE.Group();
    obj.name = child.path;
    parentThree.add(obj);
  });
}

function pythonPartition(fixtureName: string): SceneNode {
  const fixture = path.resolve(__dirname, `../../../../../../tests/fixtures/${fixtureName}`);
  const attrs = (relativePath: string): SceneNode['attrs'] => {
    const docName = ROOT_ATTR_DOCS.find((candidate) =>
      existsSync(path.join(fixture, relativePath, candidate))
    );
    if (docName === undefined) throw new Error(`No attrs document for ${relativePath}`);
    const document = JSON.parse(readFileSync(path.join(fixture, relativePath, docName), 'utf8'));
    return rootAttributes(document, docName) as SceneNode['attrs'];
  };
  const partitionAttrs = attrs('tiles');
  const children = readdirSync(path.join(fixture, 'tiles'), { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith('part_'))
    .sort((left, right) => left.name.localeCompare(right.name, undefined, { numeric: true }))
    .map((entry) => {
      const relativePath = `tiles/${entry.name}`;
      const childAttrs = attrs(relativePath);
      return {
        path: `/${relativePath}`,
        type: childAttrs.type,
        attrs: childAttrs,
        hasSpatialIndex: false,
        children: [],
      } as SceneNode;
    });
  return {
    path: '/tiles',
    type: 'group',
    attrs: partitionAttrs,
    hasSpatialIndex: false,
    children,
  };
}

describe('loadPartitionGroupNode', () => {
  it('creates a THREE.Group with the wrapper path as its name', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const node = makePartitionGroupNode([
      makePartNode('/partition/part_0'),
      makePartNode('/partition/part_1'),
    ]);

    const parent = new THREE.Group();
    const wrapper = await loadPartitionGroupNode(
      node,
      parent,
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper).toBeInstanceOf(THREE.Group);
    expect(wrapper.name).toBe('/partition');
    expect(parent.children).toHaveLength(1);
    expect(parent.children[0]).toBe(wrapper);
  });

  it('recurses each child through loadSceneNodes', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const node = makePartitionGroupNode([
      makePartNode('/partition/part_0'),
      makePartNode('/partition/part_1'),
      makePartNode('/partition/part_2'),
    ]);

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(loadSceneNodesMock).toHaveBeenCalledTimes(3);
    // Each child mesh attached to the wrapper.
    expect(wrapper.children).toHaveLength(3);
    expect(wrapper.children.map((c) => c.name).sort()).toEqual([
      '/partition/part_0',
      '/partition/part_1',
      '/partition/part_2',
    ]);
  });

  it('registers validated part bounds by child_index for frustum selection', async () => {
    attachStubChildren();
    const registerPartition = vi.fn();
    const children = [
      makePartNode('/partition/part_1', 'points', {
        child_index: 1,
        position_bounds: { min: [10, 20], max: [11, 21] },
      }),
      makePartNode('/partition/part_0', 'points', {
        child_index: 0,
        position_bounds: { min: [0, 1], max: [2, 3] },
      }),
    ];

    const wrapper = await loadPartitionGroupNode(
      makePartitionGroupNode(children),
      new THREE.Group(),
      makeStubLoc(),
      makeRegistryCtx(registerPartition),
      loadSceneNodesMock
    );

    expect(registerPartition).toHaveBeenCalledOnce();
    const entry = registerPartition.mock.calls[0][0];
    expect(entry.path).toBe('/partition');
    expect(entry.groupObject).toBe(wrapper);
    expect(
      entry.children.map((child: { positionBounds: unknown }) => child.positionBounds)
    ).toEqual([
      { min: [0, 1], max: [2, 3] },
      { min: [10, 20], max: [11, 21] },
    ]);
    expect(
      entry.children.map((child: { object: THREE.Object3D }) => child.object.userData.partIndex)
    ).toEqual([0, 1]);
  });

  it('skips registry attachment when a part produces no scene object', async () => {
    const registerPartition = vi.fn();
    const children = [
      makePartNode('/partition/part_0', 'points', {
        child_index: 0,
        position_bounds: { min: [0, 0], max: [1, 1] },
      }),
      makePartNode('/partition/part_1', 'points', {
        child_index: 1,
        position_bounds: { min: [2, 2], max: [3, 3] },
      }),
    ];
    loadSceneNodesMock.mockImplementation(async (child: SceneNode, parentThree: THREE.Object3D) => {
      if (child === children[0]) {
        const object = new THREE.Group();
        object.name = child.path;
        parentThree.add(object);
      }
    });

    await loadPartitionGroupNode(
      makePartitionGroupNode(children),
      new THREE.Group(),
      makeStubLoc(),
      makeRegistryCtx(registerPartition),
      loadSceneNodesMock
    );

    expect(registerPartition).not.toHaveBeenCalled();
  });

  it('loads parts with bounded concurrency while preserving authored order and indices', async () => {
    const children = Array.from({ length: 10 }, (_, index) =>
      makePartNode(`/partition/part_${index}`, 'points', { child_index: index })
    );
    const releases: Array<() => void> = [];
    const gates = children.map(
      () =>
        new Promise<void>((resolve) => {
          releases.push(resolve);
        })
    );
    let active = 0;
    let maxActive = 0;
    let started = 0;
    loadSceneNodesMock.mockImplementation(async (child: SceneNode, parentThree: THREE.Object3D) => {
      const index = children.indexOf(child);
      started++;
      active++;
      maxActive = Math.max(maxActive, active);
      await gates[index];
      const object = new THREE.Group();
      object.name = child.path;
      parentThree.add(object);
      active--;
    });

    const loadPromise = loadPartitionGroupNode(
      makePartitionGroupNode(children),
      new THREE.Group(),
      makeStubLoc(),
      makeCtx(),
      loadSceneNodesMock
    );

    await Promise.resolve();
    const firstWave = started;
    for (let index = releases.length - 1; index >= 0; index--) releases[index]();
    const wrapper = await loadPromise;

    expect(firstWave).toBe(EAGER_CHILD_LOAD_CONCURRENCY);
    expect(maxActive).toBe(EAGER_CHILD_LOAD_CONCURRENCY);
    expect(wrapper.children.map((child) => child.name)).toEqual(
      children.map((child) => child.path)
    );
    expect(wrapper.children.map((child) => child.userData.partIndex)).toEqual(
      children.map((_, index) => index)
    );
  });

  it('stamps partition slots while their children are still loading', async () => {
    const child = makePartNode('/partition/part_0', 'points', { child_index: 7 });
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    loadSceneNodesMock.mockImplementation(async () => gate);
    const parent = new THREE.Group();

    const loadPromise = loadPartitionGroupNode(
      makePartitionGroupNode([child]),
      parent,
      makeStubLoc(),
      makeCtx(),
      loadSceneNodesMock
    );

    await Promise.resolve();
    expect(parent.children[0].children[0].userData.partIndex).toBe(7);
    release();
    await loadPromise;
  });

  it('keeps every object from a part contiguous and stamps each with the same part index', async () => {
    const children = [
      makePartNode('/partition/part_a', 'points', { child_index: 7 }),
      makePartNode('/partition/part_b', 'points', { child_index: 3 }),
    ];
    const releases: Array<() => void> = [];
    const gates = children.map(
      () =>
        new Promise<void>((resolve) => {
          releases.push(resolve);
        })
    );
    loadSceneNodesMock.mockImplementation(async (child: SceneNode, parentThree: THREE.Object3D) => {
      const index = children.indexOf(child);
      await gates[index];
      for (const suffix of ['first', 'second']) {
        const object = new THREE.Group();
        object.name = `${child.path}/${suffix}`;
        parentThree.add(object);
      }
    });

    const loadPromise = loadPartitionGroupNode(
      makePartitionGroupNode(children),
      new THREE.Group(),
      makeStubLoc(),
      makeCtx(),
      loadSceneNodesMock
    );
    await Promise.resolve();
    releases[1]();
    await Promise.resolve();
    releases[0]();
    const wrapper = await loadPromise;

    expect(wrapper.children.map((child) => child.name)).toEqual([
      '/partition/part_a/first',
      '/partition/part_a/second',
      '/partition/part_b/first',
      '/partition/part_b/second',
    ]);
    expect(wrapper.children.map((child) => child.userData.partIndex)).toEqual([7, 7, 3, 3]);
  });

  it('all children start visible before the first frustum evaluation', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const node = makePartitionGroupNode([
      makePartNode('/partition/part_0'),
      makePartNode('/partition/part_1'),
    ]);

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    for (const child of wrapper.children) {
      expect(child.visible).toBe(true);
    }
  });

  it('applies the wrapper transform when present', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1];
    const node = makePartitionGroupNode([makePartNode('/partition/part_0')], { transform });

    await loadPartitionGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(ctx.nodeFactory.applyTransform).toHaveBeenCalledWith(expect.any(THREE.Group), transform);
  });

  it('handles an empty-children Partition group without crashing', async () => {
    const ctx = makeCtx();
    const node = makePartitionGroupNode([]);

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.children).toHaveLength(0);
    expect(loadSceneNodesMock).not.toHaveBeenCalled();
  });

  it('stashes bsp_tree on the wrapper and tags each part with its child_index', async () => {
    // The depth-sort coordinator reads `userData.bspTree` off the wrapper and
    // `userData.partIndex` off each part object to map a `bsp_tree` leaf back
    // to its render mesh for stable BSP back-to-front ordering.
    attachStubChildren();
    const ctx = makeCtx();
    const bspTree = { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } };
    const parts = [
      makePartNode('/partition/part_0', 'points', {
        child_index: 0,
        position_bounds: { min: [-2, 0], max: [0.5, 1] },
      }),
      makePartNode('/partition/part_1', 'points', {
        child_index: 1,
        position_bounds: { min: [-0.5, 0], max: [2, 1] },
      }),
    ];
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toEqual(bspTree);
    expect(wrapper.children.map((c) => c.userData.partIndex)).toEqual([0, 1]);
  });

  it('keeps a nested bsp_tree produced by a native spatial partition', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const infoSpy = vi.spyOn(log, 'info').mockImplementation(() => {});
    const warningSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const bspTree = {
      axis: 0,
      split: 0,
      left: {
        axis: 0,
        split: -4,
        left: { part: 0 },
        right: { part: 1 },
      },
      right: {
        axis: 0,
        split: 4,
        left: { part: 2 },
        right: { part: 3 },
      },
    };
    const parts = [
      [-6.2, -5.8],
      [-2.2, -1.8],
      [1.8, 2.2],
      [5.8, 6.2],
    ].map(([min, max], childIndex) =>
      makePartNode(`/partition/part_${childIndex}`, 'points', {
        child_index: childIndex,
        position_bounds: { min: [min, -0.1, 0], max: [max, 0.1, 0] },
      })
    );
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toEqual(bspTree);
    expect(wrapper.children.map((child) => child.userData.partIndex)).toEqual([0, 1, 2, 3]);
    expect(warningSpy).not.toHaveBeenCalled();
    expect(infoSpy).not.toHaveBeenCalledWith(
      Modules.SCENE_LOADER,
      expect.stringContaining('without verifiable part bounds')
    );
  });

  it('drops a bsp_tree whose split plane is outside the overlap-tolerant center band', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const bspTree = { axis: 0, split: -10, left: { part: 0 }, right: { part: 1 } };
    const parts = [
      makePartNode('/partition/part_0', 'points', {
        child_index: 0,
        position_bounds: { min: [-2, 0], max: [0.5, 1] },
      }),
      makePartNode('/partition/part_1', 'points', {
        child_index: 1,
        position_bounds: { min: [-0.5, 0], max: [2, 1] },
      }),
    ];
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toBeUndefined();
  });

  it('validates bsp_tree leaves against child_index rather than child load order', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const bspTree = { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } };
    const parts = [
      makePartNode('/partition/part_1', 'points', {
        child_index: 1,
        position_bounds: { min: [-0.5, 0], max: [2, 1] },
      }),
      makePartNode('/partition/part_0', 'points', {
        child_index: 0,
        position_bounds: { min: [-2, 0], max: [0.5, 1] },
      }),
    ];
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toEqual(bspTree);
    expect(wrapper.children.map((c) => c.userData.partIndex)).toEqual([1, 0]);
  });

  it('keeps a valid 2D bsp_tree whose split uses the second stored axis', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const bspTree = { axis: 1, split: 0, left: { part: 0 }, right: { part: 1 } };
    const parts = [
      makePartNode('/partition/part_0', 'points', {
        position_bounds: { min: [0, -2], max: [1, 0.5] },
      }),
      makePartNode('/partition/part_1', 'points', {
        position_bounds: { min: [0, -0.5], max: [1, 2] },
      }),
    ];
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toEqual(bspTree);
  });

  it('keeps a valid nD bsp_tree whose split uses a displayed column above 2', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const bspTree = { axis: 3, split: 0, left: { part: 0 }, right: { part: 1 } };
    const parts = [
      makePartNode('/partition/part_0', 'points', {
        position_bounds: { min: [0, 0, 0, -2], max: [1, 1, 1, 0.5] },
      }),
      makePartNode('/partition/part_1', 'points', {
        position_bounds: { min: [0, 0, 0, -0.5], max: [1, 1, 1, 2] },
      }),
    ];
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toEqual(bspTree);
  });

  it('keeps a sparse grid bsp_tree using the measured per-axis overlap floor', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const bspTree = {
      axis: 0,
      split: 0,
      left: { axis: 0, split: -10, left: { part: 0 }, right: { part: 1 } },
      right: { axis: 0, split: 10, left: { part: 2 }, right: { part: 3 } },
    };
    const parts = [
      makePartNode('/partition/part_0', 'points', {
        position_bounds: { min: [-9], max: [-8] },
      }),
      makePartNode('/partition/part_1', 'points', {
        position_bounds: { min: [-7], max: [-6] },
      }),
      makePartNode('/partition/part_2', 'points', {
        position_bounds: { min: [8], max: [12] },
      }),
      makePartNode('/partition/part_3', 'points', {
        position_bounds: { min: [10], max: [14] },
      }),
    ];
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toEqual(bspTree);
  });

  it('drops a bsp_tree when a nested split is unsound', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const bspTree = {
      axis: 0,
      split: 0,
      left: { axis: 1, split: -10, left: { part: 0 }, right: { part: 1 } },
      right: { part: 2 },
    };
    const parts = [
      makePartNode('/partition/part_0', 'points', {
        position_bounds: { min: [-2, -2], max: [0.5, 0.5] },
      }),
      makePartNode('/partition/part_1', 'points', {
        position_bounds: { min: [-2, -0.5], max: [0.5, 2] },
      }),
      makePartNode('/partition/part_2', 'points', {
        position_bounds: { min: [-0.5, -2], max: [2, 2] },
      }),
    ];
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toBeUndefined();
  });

  it.each([
    ['duplicates a leaf label', { axis: 0, split: 0, left: { part: 0 }, right: { part: 0 } }, 2],
    ['uses an absent 2D axis', { axis: 2, split: 0, left: { part: 0 }, right: { part: 1 } }, 2],
    ['has no right subtree', { axis: 0, split: 0, left: { part: 0 } }, 2],
    ['omits a partition part', { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } }, 3],
  ])('drops a malformed bsp_tree that %s', async (_reason, bspTree, partCount) => {
    attachStubChildren();
    const ctx = makeCtx();
    const parts = [
      makePartNode('/partition/part_0', 'points', {
        position_bounds: { min: [-2, 0], max: [0.5, 1] },
      }),
      makePartNode('/partition/part_1', 'points', {
        position_bounds: { min: [-0.5, 0], max: [2, 1] },
      }),
    ];
    if (partCount === 3) {
      parts.push(
        makePartNode('/partition/part_2', 'points', {
          position_bounds: { min: [1.5, 0], max: [4, 1] },
        })
      );
    }
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toBeUndefined();
  });

  it('keeps a bsp_tree when part bounds are unavailable for validation', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const infoSpy = vi.spyOn(log, 'info').mockImplementation(() => {});
    const warningSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const bspTree = { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } };
    const parts = [
      makePartNode('/partition/part_0', 'points', {
        child_index: 0,
        position_bounds: { min: [-2, 0], max: [0.5, 1] },
      }),
      makePartNode('/partition/part_1', 'points', { child_index: 1 }),
    ];
    const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toEqual(bspTree);
    expect(infoSpy).toHaveBeenCalledWith(
      Modules.SCENE_LOADER,
      expect.stringContaining('without verifiable part bounds')
    );
    expect(warningSpy).not.toHaveBeenCalled();
  });

  it.each([
    ['has no right subtree', { axis: 0, split: 0, left: { part: 0 } }, 2],
    ['has a null right subtree', { axis: 0, split: 0, left: { part: 0 }, right: null }, 2],
    ['uses a non-integer axis', { axis: 0.5, split: 0, left: { part: 0 }, right: { part: 1 } }, 2],
    ['uses a negative axis', { axis: -1, split: 0, left: { part: 0 }, right: { part: 1 } }, 2],
    [
      'uses a non-finite split',
      { axis: 0, split: Infinity, left: { part: 0 }, right: { part: 1 } },
      2,
    ],
    ['duplicates a leaf label', { axis: 0, split: 0, left: { part: 0 }, right: { part: 0 } }, 2],
    ['omits a partition part', { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } }, 3],
    [
      'uses an out-of-range leaf label',
      { axis: 0, split: 0, left: { part: 0 }, right: { part: 7 } },
      2,
    ],
  ])(
    'drops a structurally malformed bsp_tree without part bounds that %s',
    async (_reason, bspTree, partCount) => {
      attachStubChildren();
      const ctx = makeCtx();
      const parts = Array.from({ length: partCount }, (_, partIndex) =>
        makePartNode(`/partition/part_${partIndex}`, 'points', { child_index: partIndex })
      );
      const node = makePartitionGroupNode(parts, { bsp_tree: bspTree });

      const wrapper = await loadPartitionGroupNode(
        node,
        new THREE.Group(),
        makeStubLoc(),
        ctx,
        loadSceneNodesMock
      );

      expect(wrapper.userData.bspTree).toBeUndefined();
    }
  );

  it('leaves bspTree undefined when the partition has no stored tree (fallback path)', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const node = makePartitionGroupNode([makePartNode('/partition/part_0')]);

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toBeUndefined();
    // Still tagged by load order (child_index absent → falls back to index).
    expect(wrapper.children[0].userData.partIndex).toBe(0);
  });

  it('drops a wrong-frame tree written by the Python scene compiler', async () => {
    attachStubChildren();

    const wrapper = await loadPartitionGroupNode(
      pythonPartition('test_partition_wrong_frame.luxar.zarr'),
      new THREE.Group(),
      makeStubLoc(),
      makeCtx(),
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toBeUndefined();
  });

  it('keeps a valid tree written by the Python scene compiler', async () => {
    attachStubChildren();
    const node = pythonPartition('test_partition_layer.luxar.zarr');
    const bspTree = node.attrs.bsp_tree;
    expect(bspTree).toBeDefined();

    const wrapper = await loadPartitionGroupNode(
      node,
      new THREE.Group(),
      makeStubLoc(),
      makeCtx(),
      loadSceneNodesMock
    );

    expect(wrapper.userData.bspTree).toEqual(bspTree);
  });
});
