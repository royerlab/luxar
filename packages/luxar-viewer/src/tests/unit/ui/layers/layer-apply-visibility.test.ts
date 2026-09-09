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
    const requestReprocess = vi.fn();
    const engine = new LayerApplyEngine({
      getRootGroup: () => root,
      getSceneGraph: () => null,
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
