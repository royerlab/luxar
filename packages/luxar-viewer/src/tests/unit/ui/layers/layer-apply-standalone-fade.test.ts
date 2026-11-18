import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { SceneNodeIndex } from '../../../../data/scene-loader/view-state/scene-node-index';
import type { SceneNode } from '../../../../data/data-loader-types';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { applyStandaloneLadderFade } from '../../../../scene/lod-fade';
import { getLadderEnergyStampVersion } from '../../../../data/scene-loader/commit/stamp-view-version';
import { LayerApplyEngine } from '../../../../ui/layers/layer-apply';
import { LayerStateManager } from '../../../../ui/layers/layer-state';

describe('standalone ladder fade after layer reset', () => {
  it.each([
    { authored: 'additive', edited: 'normal', beforeReset: 1, afterReset: 2 },
    { authored: 'normal', edited: 'additive', beforeReset: 2, afterReset: 1 },
  ] as const)(
    'restores $authored after editing to $edited',
    ({ authored, edited, beforeReset, afterReset }) => {
      const graph = {
        path: '',
        type: 'scene',
        attrs: {},
        children: [
          {
            path: '/leaf',
            type: 'gsplats',
            attrs: { layer: true, blending_mode: authored },
            children: [],
          },
        ],
      } as unknown as SceneNode;
      const root = new THREE.Group();
      const material = new GSplatMaterial({ blendingMode: authored });
      const leaf = new THREE.Mesh(new THREE.BufferGeometry(), material);
      leaf.name = '/leaf';
      leaf.userData.nodeType = 'gsplats';
      leaf.userData._layerMaterialCloned = true;
      leaf.userData.committedEnergyFraction = 0.5;
      root.add(leaf);
      const state = new LayerStateManager();
      const index = new SceneNodeIndex(graph);
      state.initFromSceneGraph(graph, index);
      const engine = new LayerApplyEngine({
        getRootGroup: () => root,
        getSceneNodeIndex: () => index,
        state,
        requestRender: () => {},
        requestReprocess: () => {},
      });
      const version = getLadderEnergyStampVersion();

      applyStandaloneLadderFade(root, true, undefined, version);
      state.setBlendingMode('/leaf', edited);
      engine.applyBlendingMode(state.getLayer('/leaf')!);
      applyStandaloneLadderFade(root, true, undefined, version);
      expect(material.getOpacity()).toBe(beforeReset);

      state.initFromSceneGraph(graph, index);
      engine.applyColormap(state.getLayer('/leaf')!);
      expect(material.userData.blendingMode).toBe(authored);
      expect(applyStandaloneLadderFade(root, true, undefined, version)).toBe(true);
      expect(material.getOpacity()).toBe(afterReset);
    }
  );
});
