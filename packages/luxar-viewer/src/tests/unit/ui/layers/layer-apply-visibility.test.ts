import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import { LayerApplyEngine } from '../../../../ui/layers/layer-apply';

describe('LayerApplyEngine visibility', () => {
  it('stamps user-controlled visibility separately from the Three.js display flag', () => {
    const root = new THREE.Group();
    const layer = new THREE.Group();
    layer.name = '/layer';
    root.add(layer);
    const requestRender = vi.fn();
    const engine = new LayerApplyEngine({
      getRootGroup: () => root,
      getSceneGraph: () => null,
      state: {} as never,
      requestRender,
    });

    engine.applyVisibility(layer.name, false);

    expect(layer.visible).toBe(false);
    expect(layer.userData.layerVisible).toBe(false);
    expect(requestRender).toHaveBeenCalledTimes(1);
  });
});
