/**
 * A registry-activated partition part (`NodeBuildCtx.registerOnly`, B4) may
 * settle after its dataset was switched away: the activation is fire-and-forget
 * from the t+1 prefetch, and the loader registry outlives the dataset. A leaf
 * built for the dead dataset must not register its loader into the maps the
 * next dataset's passes sweep — it would be driven against the old store, or
 * overwrite the new dataset's loader at the same path. It is disposed instead.
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

const loaders = vi.hoisted(() => [] as Array<{ dispose: () => void }>);

vi.mock(
  import('../../../../../data/scene-loader/loaders/loader-factory'),
  async (importOriginal) => {
    const makeLoader = () => {
      const loader = { dispose: vi.fn() };
      loaders.push(loader);
      return loader as never;
    };
    return {
      ...(await importOriginal()),
      createPointsLoader: vi.fn(makeLoader),
      createLinesLoader: vi.fn(makeLoader),
      createGSplatsLoader: vi.fn(makeLoader),
      createMeshLoader: vi.fn(makeLoader),
    };
  }
);

import { loadPointsNode } from '../../../../../data/scene-loader/nodes/load-points-node';
import { loadLinesNode } from '../../../../../data/scene-loader/nodes/load-lines-node';
import { loadGSplatsNode } from '../../../../../data/scene-loader/nodes/load-gsplats-node';
import { loadMeshNode } from '../../../../../data/scene-loader/nodes/load-mesh-node';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';

type Load = (
  node: SceneNode,
  parent: THREE.Object3D,
  loc: never,
  ctx: NodeBuildCtx
) => Promise<unknown>;

const CASES: Array<[string, Load]> = [
  ['points', loadPointsNode as Load],
  ['lines', loadLinesNode as Load],
  ['gsplats', loadGSplatsNode as Load],
  ['mesh', loadMeshNode as Load],
];

function registeredPaths(ctx: NodeBuildCtx): string[] {
  return [
    ...ctx.registry.loaders.keys(),
    ...ctx.registry.linesLoaders.keys(),
    ...ctx.registry.gsplatLoaders.keys(),
    ...ctx.registry.meshLoaders.keys(),
  ];
}

describe('register-only leaves of a dead dataset register nothing', () => {
  it.fails.each(CASES)('%s', async (type, load) => {
    const node: SceneNode = {
      path: '/partition/part_0',
      type,
      attrs: { type, normal_dims: [0, 1, 2] } as unknown as SceneNode['attrs'],
      hasSpatialIndex: false,
      children: [],
    };
    const ctx = makeTestNodeBuildCtx({
      registerOnly: true,
      isDatasetLive: () => false,
      nodeFactory: {
        createEmptyPointsNode: vi.fn(() => new THREE.Mesh()),
        createEmptyLinesNode: vi.fn(() => new THREE.Mesh()),
        createEmptyGSplatsNode: vi.fn(() => new THREE.Mesh()),
        createEmptyMeshNode: vi.fn(() => new THREE.Mesh()),
      } as unknown as NodeBuildCtx['nodeFactory'],
    });
    loaders.length = 0;

    await load(node, new THREE.Group(), {} as never, ctx);

    expect(registeredPaths(ctx)).toEqual([]);
    expect(loaders).toHaveLength(1);
    expect(loaders[0].dispose).toHaveBeenCalledOnce();
  });
});
