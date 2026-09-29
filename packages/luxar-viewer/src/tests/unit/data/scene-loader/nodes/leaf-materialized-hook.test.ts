/**
 * Every leaf loader reports the placeholder it just attached through
 * `NodeBuildCtx.onLeafMaterialized` — synchronously, before any data is loaded
 * into it.
 *
 * The placeholders the four `load*NodeCheap` halves create are the ONLY place a
 * drawable leaf object comes into existence (initial load, lazy LOD levels,
 * registry-activated partition parts, subtree activations). A leaf built after
 * the Layers panel initialised is invisible to the panel otherwise, so it would
 * render its authored appearance instead of the layer's live state until the
 * next slider edit. Reporting it at attach time — before the first commit — is
 * what lets the owner restyle it before it is ever drawn.
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

vi.mock(
  import('../../../../../data/scene-loader/loaders/loader-factory'),
  async (importOriginal) => ({
    ...(await importOriginal()),
    createPointsLoader: vi.fn(() => ({ dispose: vi.fn() }) as never),
    createLinesLoader: vi.fn(() => ({ dispose: vi.fn() }) as never),
    createGSplatsLoader: vi.fn(() => ({ dispose: vi.fn() }) as never),
    createMeshLoader: vi.fn(() => ({ dispose: vi.fn() }) as never),
  })
);

import { loadPointsNodeCheap } from '../../../../../data/scene-loader/nodes/load-points-node';
import { loadLinesNodeCheap } from '../../../../../data/scene-loader/nodes/load-lines-node';
import { loadGSplatsNodeCheap } from '../../../../../data/scene-loader/nodes/load-gsplats-node';
import { loadMeshNodeCheap } from '../../../../../data/scene-loader/nodes/load-mesh-node';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';

type Cheap = (
  node: SceneNode,
  parent: THREE.Object3D,
  loc: never,
  ctx: NodeBuildCtx
) => Promise<{ placeholder: THREE.Mesh }>;

const CASES: Array<[string, Cheap, keyof NodeBuildCtx['nodeFactory']]> = [
  ['points', loadPointsNodeCheap as Cheap, 'createEmptyPointsNode'],
  ['lines', loadLinesNodeCheap as Cheap, 'createEmptyLinesNode'],
  ['gsplats', loadGSplatsNodeCheap as Cheap, 'createEmptyGSplatsNode'],
  ['mesh', loadMeshNodeCheap as Cheap, 'createEmptyMeshNode'],
];

describe('leaf loaders report each attached placeholder (onLeafMaterialized)', () => {
  it.each(CASES)(
    '%s: reported once, already attached, before the loader resolves',
    async (type, cheap, factoryMethod) => {
      const node: SceneNode = {
        path: `/layer/${type}`,
        type,
        attrs: { type } as SceneNode['attrs'],
        hasSpatialIndex: false,
        children: [],
      };
      const parent = new THREE.Group();
      const made = new THREE.Mesh();
      made.name = node.path;
      const seen: Array<{ path: string; object: THREE.Object3D; attached: boolean }> = [];
      const ctx = makeTestNodeBuildCtx({
        onLeafMaterialized: (path: string, object: THREE.Object3D) =>
          seen.push({ path, object, attached: object.parent === parent }),
      } as Partial<NodeBuildCtx>);
      vi.mocked(ctx.nodeFactory[factoryMethod] as (...args: unknown[]) => unknown).mockReturnValue(
        made
      );

      const { placeholder } = await cheap(node, parent, {} as never, ctx);

      expect(placeholder).toBe(made);
      expect(seen).toEqual([{ path: node.path, object: made, attached: true }]);
    }
  );

  it('a ctx with no hook still loads (the hook is optional)', async () => {
    const node: SceneNode = {
      path: '/p',
      type: 'points',
      attrs: { type: 'points' } as SceneNode['attrs'],
      hasSpatialIndex: false,
      children: [],
    };
    const ctx = makeTestNodeBuildCtx();
    vi.mocked(ctx.nodeFactory.createEmptyPointsNode).mockReturnValue(new THREE.Mesh() as never);
    await expect(
      loadPointsNodeCheap(node, new THREE.Group(), {} as never, ctx)
    ).resolves.toBeDefined();
  });
});
