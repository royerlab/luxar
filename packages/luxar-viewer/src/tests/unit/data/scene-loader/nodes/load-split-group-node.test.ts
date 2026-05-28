/**
 * Tests for ``loadSplitGroupNode``.
 *
 * Strategy: the Split loader recurses children through ``loadSceneNodes``
 * but has no per-frame state to register (no LOD-style selector). We
 * mock ``loadSceneNodes`` to attach a stub mesh per child, then assert
 * on the THREE-tree shape.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

const loadSceneNodesMock = vi.fn();
vi.mock('../../../../../data/scene-loader/nodes/load-scene-nodes', () => ({
  loadSceneNodes: (...args: unknown[]) => loadSceneNodesMock(...args),
}));

import { loadSplitGroupNode } from '../../../../../data/scene-loader/nodes/load-split-group-node';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';

beforeEach(() => {
  loadSceneNodesMock.mockReset();
});

function makePartNode(path: string, partType: 'points' | 'gsplats' = 'points'): SceneNode {
  return {
    path,
    type: partType,
    attrs: {
      type: partType,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

function makeSplitGroupNode(
  children: SceneNode[],
  extraAttrs: Record<string, unknown> = {}
): SceneNode {
  return {
    path: '/split',
    type: 'group',
    attrs: {
      type: 'group',
      kind: 'split',
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
  return {
    nodeFactory,
    lodGroupRegistry: null,
  } as unknown as NodeBuildCtx;
}

function makeStubLoc() {
  return { resolve: (_p: string) => ({ resolve: () => ({}) }) } as any;
}

/** Make the mock attach a child mesh to ``parentThree`` so the recursion
 *  produces a complete tree shape. */
function attachStubChildren() {
  loadSceneNodesMock.mockImplementation(
    async (child: SceneNode, parentThree: THREE.Object3D) => {
      const obj = new THREE.Group();
      obj.name = child.path;
      parentThree.add(obj);
    }
  );
}

describe('loadSplitGroupNode', () => {
  it('creates a THREE.Group with the wrapper path as its name', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const node = makeSplitGroupNode([
      makePartNode('/split/part_0'),
      makePartNode('/split/part_1'),
    ]);

    const parent = new THREE.Group();
    const wrapper = await loadSplitGroupNode(node, parent, makeStubLoc(), ctx);

    expect(wrapper).toBeInstanceOf(THREE.Group);
    expect(wrapper.name).toBe('/split');
    expect(parent.children).toHaveLength(1);
    expect(parent.children[0]).toBe(wrapper);
  });

  it('recurses each child through loadSceneNodes', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const node = makeSplitGroupNode([
      makePartNode('/split/part_0'),
      makePartNode('/split/part_1'),
      makePartNode('/split/part_2'),
    ]);

    const wrapper = await loadSplitGroupNode(node, new THREE.Group(), makeStubLoc(), ctx);

    expect(loadSceneNodesMock).toHaveBeenCalledTimes(3);
    // Each child mesh attached to the wrapper.
    expect(wrapper.children).toHaveLength(3);
    expect(wrapper.children.map((c) => c.name).sort()).toEqual([
      '/split/part_0',
      '/split/part_1',
      '/split/part_2',
    ]);
  });

  it('all children stay visible after load (no LOD-style selector)', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const node = makeSplitGroupNode([
      makePartNode('/split/part_0'),
      makePartNode('/split/part_1'),
    ]);

    const wrapper = await loadSplitGroupNode(node, new THREE.Group(), makeStubLoc(), ctx);

    for (const child of wrapper.children) {
      expect(child.visible).toBe(true);
    }
  });

  it('applies the wrapper transform when present', async () => {
    attachStubChildren();
    const ctx = makeCtx();
    const transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1];
    const node = makeSplitGroupNode(
      [makePartNode('/split/part_0')],
      { transform }
    );

    await loadSplitGroupNode(node, new THREE.Group(), makeStubLoc(), ctx);

    expect(ctx.nodeFactory.applyTransform).toHaveBeenCalledWith(
      expect.any(THREE.Group),
      transform
    );
  });

  it('handles an empty-children Split group without crashing', async () => {
    const ctx = makeCtx();
    const node = makeSplitGroupNode([]);

    const wrapper = await loadSplitGroupNode(node, new THREE.Group(), makeStubLoc(), ctx);

    expect(wrapper.children).toHaveLength(0);
    expect(loadSceneNodesMock).not.toHaveBeenCalled();
  });
});
