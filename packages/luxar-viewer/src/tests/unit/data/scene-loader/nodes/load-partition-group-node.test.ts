/**
 * Tests for ``loadPartitionGroupNode``.
 *
 * Strategy: the Partition loader recurses children through ``loadSceneNodes``
 * but has no per-frame state to register (no LOD-style selector). We
 * mock ``loadSceneNodes`` to attach a stub mesh per child, then assert
 * on the THREE-tree shape.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

// Caller-injected recursion handle: the partition loader now takes
// loadSceneNodes as an explicit ``loadChildren`` parameter (mirrors
// load-lod-group-node) so tests can pass the mock directly.
const loadSceneNodesMock = vi.fn();

import { loadPartitionGroupNode } from '../../../../../data/scene-loader/nodes/load-partition-group-node';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';

beforeEach(() => {
  loadSceneNodesMock.mockReset();
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

  it('all children stay visible after load (no LOD-style selector)', async () => {
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
    // to its render mesh for exact back-to-front ordering.
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
    ['duplicates a leaf label', { axis: 0, split: 0, left: { part: 0 }, right: { part: 0 } }],
    ['uses an absent 2D axis', { axis: 2, split: 0, left: { part: 0 }, right: { part: 1 } }],
    ['has no right subtree', { axis: 0, split: 0, left: { part: 0 } }],
  ])('drops a malformed bsp_tree that %s', async (_reason, bspTree) => {
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

  it('drops a bsp_tree when part bounds are missing or malformed', async () => {
    attachStubChildren();
    const ctx = makeCtx();
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

    expect(wrapper.userData.bspTree).toBeUndefined();
  });

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
});
