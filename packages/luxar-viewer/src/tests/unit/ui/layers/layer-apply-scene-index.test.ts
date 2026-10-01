/**
 * A Layers-panel slider tick over a many-part layer must compose every part
 * from the loader's `SceneNodeIndex`, never by descending the scene graph.
 *
 * `collectAncestorNodes` resolves each level with a linear `children.find`, so
 * one fan-out over a P-part wrapper costs ~P²/2 path comparisons per tick. The
 * index holds each node's root→node chain, so the tick is O(P · depth).
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { LayerApplyEngine, type LayerApplyEngineDeps } from '../../../../ui/layers/layer-apply';
import { LayerStateManager } from '../../../../ui/layers/layer-state';
import { SceneNodeIndex } from '../../../../data/scene-loader/view-state/scene-node-index';
import type { SceneNode } from '../../../../data/data-loader-types';

const PARTS = 64;

function opacityMaterial(writes: number[]): Record<string, unknown> {
  return {
    userData: { blendingMode: 'additive' },
    uniforms: { uOpacity: { value: 1.0 } },
    defines: {},
    updateGamma: () => {},
    updateOpacity: (v: number) => writes.push(v),
    updateAbsorption: () => {},
    updateIntensity: () => {},
    updateOffset: () => {},
    updateScalarRange: () => {},
    applyBlendingMode: () => {},
  };
}

function partitionHarness() {
  const rootGroup = new THREE.Group();
  const writes: number[][] = [];
  const parts: SceneNode[] = [];
  for (let i = 0; i < PARTS; i++) {
    const path = `/layer/part_${i}`;
    const partWrites: number[] = [];
    writes.push(partWrites);
    const obj = new THREE.Points(
      new THREE.BufferGeometry(),
      opacityMaterial(partWrites) as unknown as THREE.Material
    );
    obj.name = path;
    // Skip clone-on-first-use so the stub is the material the tick writes to.
    obj.userData._layerMaterialCloned = true;
    rootGroup.add(obj);
    parts.push({
      path,
      type: 'points',
      attrs: { opacity: 0.5 },
      children: [],
    } as unknown as SceneNode);
  }
  const layer = {
    path: '/layer',
    type: 'group',
    attrs: { layer: true, kind: 'partition' },
    children: parts,
  } as unknown as SceneNode;
  const graph = { path: '/', type: 'scene', attrs: {}, children: [layer] } as unknown as SceneNode;

  const state = new LayerStateManager();
  state.initFromSceneGraph(graph);
  const index = new SceneNodeIndex(graph);

  // Count every path search from now on. A search by path starts at the root,
  // so it reads the root's children; collecting the layer's own leaves is a
  // downward visit from the layer node and never touches the root. Building
  // the state and the index above walked the graph once each, which is the
  // whole O(N) budget.
  let walks = 0;
  const rootChildren = graph.children;
  Object.defineProperty(graph, 'children', {
    get() {
      walks++;
      return rootChildren;
    },
  });

  const deps = {
    getRootGroup: () => rootGroup,
    // Both ports: the bare graph (what the engine read before the index port
    // existed) and the index, so a descent through either is counted.
    getSceneGraph: () => graph,
    getSceneNodeIndex: () => index,
    state,
    requestRender: () => {},
    requestReprocess: () => {},
  };
  const engine = new LayerApplyEngine(deps as unknown as LayerApplyEngineDeps);
  return { engine, state, writes, walks: () => walks };
}

describe('LayerApplyEngine — scene-node index', () => {
  it.fails('composes a slider tick over a partition from the index without descending the graph', () => {
    const h = partitionHarness();
    const layer = h.state.getLayer('/layer')!;
    layer.opacity = 0.4;
    h.engine.applyOpacity(layer);
    // 0.4 (live layer) × 0.5 (each part's authored opacity).
    for (const partWrites of h.writes) expect(partWrites.at(-1)).toBeCloseTo(0.2, 10);
    expect(h.walks()).toBe(0);
  });
});
