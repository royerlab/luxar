import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import { LayerApplyEngine } from '../../../../ui/layers/layer-apply';
import { SceneNodeIndex } from '../../../../data/scene-loader/view-state/scene-node-index';
import { LayerStateManager } from '../../../../ui/layers/layer-state';
import type { SceneNode } from '../../../../data/data-loader-types';

describe('LayerApplyEngine visibility', () => {
  it('stamps user-controlled visibility separately from the Three.js display flag', () => {
    const root = new THREE.Group();
    const layer = new THREE.Group();
    layer.name = '/layer';
    root.add(layer);
    const requestRender = vi.fn();
    const requestReprocess = vi.fn();
    const engine = new LayerApplyEngine({
      getRootGroup: () => root,
      getSceneNodeIndex: () => null,
      state: {} as never,
      requestRender,
      requestReprocess,
    });

    engine.applyVisibility(layer.name, false);

    expect(layer.visible).toBe(false);
    expect(layer.userData.layerVisible).toBe(false);
    expect(requestRender).toHaveBeenCalledTimes(1);
    expect(requestReprocess).not.toHaveBeenCalled();

    engine.applyVisibility(layer.name, true);

    expect(layer.visible).toBe(true);
    expect(layer.userData.layerVisible).toBe(true);
    expect(requestReprocess).toHaveBeenCalledOnce();
    expect(requestReprocess).toHaveBeenCalledWith([layer.name]);

    engine.applyVisibility(layer.name, true);
    expect(requestReprocess).toHaveBeenCalledOnce();
  });
});

it('reprocesses Brightness on 2D and remapped 3D gsplats, but not standard 3D', () => {
  const leaves = [
    { path: '/two', type: 'gsplats', attrs: { layer: true, ndim: 2 } },
    { path: '/three', type: 'gsplats', attrs: { layer: true, ndim: 3 } },
  ] as unknown as SceneNode[];
  const graph = { path: '/', type: 'scene', attrs: {}, children: leaves } as SceneNode;
  const state = new LayerStateManager();
  state.initFromSceneGraph(graph);
  const requestReprocess = vi.fn();
  let displayDims = [0, 1, 2];
  const engine = new LayerApplyEngine({
    getRootGroup: () => null,
    getSceneNodeIndex: () => new SceneNodeIndex(graph),
    getDisplayDims: () => displayDims,
    state,
    requestRender: vi.fn(),
    requestReprocess,
  });
  engine.applyBrightness(state.getLayer('/two')!);
  expect(requestReprocess).toHaveBeenLastCalledWith(['/two']);
  requestReprocess.mockClear();
  engine.applyBrightness(state.getLayer('/three')!);
  expect(requestReprocess).not.toHaveBeenCalled();
  displayDims = [2, 1, 0];
  engine.applyBrightness(state.getLayer('/three')!);
  expect(requestReprocess).toHaveBeenLastCalledWith(['/three']);
});
