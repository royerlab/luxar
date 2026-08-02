/**
 * `loadSceneNodes` dispatch guards.
 *
 * The geometry-kind branch indexes a descriptor table with `node.type`, which
 * `build-scene-graph.ts` copies verbatim out of the store's `.zattrs`. That
 * makes it an untrusted string, so the lookup must not reach `Object.prototype`
 * — see `geometryDescriptorFor`.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { loadSceneNodes } from '../../../../../data/scene-loader/nodes/load-scene-nodes';
import type { SceneNode } from '../../../../../data/data-loader-types';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';

function makeCtx(): NodeBuildCtx {
  return { nodeFactory: { applyTransform: vi.fn() } } as unknown as NodeBuildCtx;
}

describe('loadSceneNodes — unrecognized node.type', () => {
  // Every one of these resolves truthy through the prototype chain of an
  // object literal, so a bare `TABLE[node.type]` returns a non-descriptor.
  // Calling `.loadNode` on it throws a TypeError, and `loadLeafNode` re-throws
  // anything that is not a LoaderError — which would sink the whole scene load,
  // not just this node.
  it.each(['constructor', 'toString', 'valueOf', 'hasOwnProperty', '__proto__', 'isPrototypeOf'])(
    'treats type=%s as a non-geometry node instead of throwing',
    async (badType) => {
      const node = {
        path: '/bad',
        type: badType,
        attrs: {},
        children: [],
      } as unknown as SceneNode;
      const parent = new THREE.Group();
      const loc = { resolve: () => loc } as never;

      await expect(loadSceneNodes(node, parent, loc, makeCtx())).resolves.toBeUndefined();
    }
  );

  it('still recurses into a plain group with an unknown type', async () => {
    const child = { path: '/g/c', type: 'wat', attrs: {}, children: [] } as unknown as SceneNode;
    const node = {
      path: '/g',
      type: 'group',
      attrs: {},
      children: [child],
    } as unknown as SceneNode;
    const parent = new THREE.Group();
    const loc = { resolve: () => loc } as never;

    await loadSceneNodes(node, parent, loc, makeCtx());

    // The group is materialised and named even though its child is unknown.
    expect(parent.children.map((c) => c.name)).toContain('/g');
  });
});
