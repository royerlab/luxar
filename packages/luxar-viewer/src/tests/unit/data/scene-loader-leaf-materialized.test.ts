/**
 * `SceneLoader` forwards every leaf its loaders materialise to the listener the
 * app connected (`setLeafMaterializedListener`), together with the scene graph
 * the leaf belongs to — so an owner holding state for a DIFFERENT graph (a
 * panel not yet re-initialised after a dataset switch) can tell and decline.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { SceneLoader } from '../../../data/scene-loader';
import type { NodeBuildCtx } from '../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../data/data-loader-types';

function ctxOf(loader: SceneLoader): NodeBuildCtx {
  return (loader as unknown as { makeNodeBuildCtx(): NodeBuildCtx }).makeNodeBuildCtx();
}

describe('SceneLoader — leaf-materialized listener', () => {
  let loader: SceneLoader | null = null;
  afterEach(async () => {
    await loader?.dispose();
    loader = null;
  });

  it('forwards (sceneGraph, path, object) from the build ctx to the listener', () => {
    loader = new SceneLoader();
    const graph = { path: '/', type: 'scene', attrs: {}, children: [] } as unknown as SceneNode;
    (loader as unknown as { _sceneGraph: SceneNode })._sceneGraph = graph;
    const listener = vi.fn();
    loader.setLeafMaterializedListener(listener);
    const leaf = new THREE.Mesh();

    ctxOf(loader).onLeafMaterialized?.('/a/b', leaf);

    expect(listener).toHaveBeenCalledExactlyOnceWith(graph, '/a/b', leaf);
  });

  it('is a no-op with no listener, and after the listener is cleared', () => {
    loader = new SceneLoader();
    const listener = vi.fn();
    const ctx = ctxOf(loader);
    expect(() => ctx.onLeafMaterialized?.('/a', new THREE.Mesh())).not.toThrow();
    loader.setLeafMaterializedListener(listener);
    loader.setLeafMaterializedListener(null);
    ctx.onLeafMaterialized?.('/a', new THREE.Mesh());
    expect(listener).not.toHaveBeenCalled();
  });
});
