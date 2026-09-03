/**
 * `applyLayerOrder` must COMPOSE the cross-layer draw order, not assign it.
 *
 * A group layer's control fans out to every data descendant
 * (`getAffectedDataLeaves` returns all of them, nested layers included). The
 * first implementation wrote the edited layer's own `layerOrder` straight onto
 * each leaf, which silently overrode the authored order of a nested leaf that
 * is ITSELF a layer with its own — the exact opposite of nearest-setter-wins,
 * and a change the user could not see or undo without a reload.
 *
 * `blending_mode` already had this shape solved; these tests pin that
 * `layer_order` now follows it.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { LayerApplyEngine } from '../../../../ui/layers/layer-apply';
import { LayerStateManager } from '../../../../ui/layers/layer-state';
import type { SceneNode } from '../../../../data/data-loader-types';

/** A gsplats mesh the engine can find by scene-graph path. */
function leafMesh(name: string): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.Material());
  mesh.name = name;
  mesh.userData.nodeType = 'gsplats';
  mesh.userData.attrs = {};
  return mesh;
}

/**
 * A plain group layer containing a leaf that is ALSO a layer. Both may author
 * an order, which is what makes nearest-setter-wins observable.
 */
function nestedLayerHarness(groupOrder?: number, leafOrder?: number) {
  const mesh = leafMesh('/g/inner');
  const rootGroup = new THREE.Group();
  rootGroup.add(mesh);

  const graph = {
    path: '',
    type: 'scene',
    attrs: {},
    hasSpatialIndex: false,
    children: [
      {
        path: '/g',
        type: 'group',
        hasSpatialIndex: false,
        attrs:
          groupOrder === undefined ? { layer: true } : { layer: true, layer_order: groupOrder },
        children: [
          {
            path: '/g/inner',
            type: 'gsplats',
            hasSpatialIndex: true,
            attrs:
              leafOrder === undefined ? { layer: true } : { layer: true, layer_order: leafOrder },
            children: [],
          },
        ],
      },
    ],
  } as unknown as SceneNode;

  const state = new LayerStateManager();
  state.initFromSceneGraph(graph);
  const engine = new LayerApplyEngine({
    getRootGroup: () => rootGroup,
    getSceneGraph: () => graph,
    state,
    requestRender: () => {},
  });
  return { engine, state, mesh };
}

const orderOf = (mesh: THREE.Mesh): unknown =>
  (mesh.userData as { layerOrder?: unknown }).layerOrder;

describe('applyLayerOrder — composition, not assignment', () => {
  it("does NOT clobber a nested layer leaf's own authored order", () => {
    const { engine, state, mesh } = nestedLayerHarness(20, 5);

    // The user edits the GROUP's order. The nested leaf authored 5 of its own.
    engine.applyLayerOrder(state.getLayer('/g')!);

    // Nearest-setter-wins: the leaf's 5 must survive.
    expect(orderOf(mesh)).toBe(5);
  });

  it("propagates the group's order to a leaf that authored none", () => {
    const { engine, state, mesh } = nestedLayerHarness(20, undefined);

    engine.applyLayerOrder(state.getLayer('/g')!);

    expect(orderOf(mesh)).toBe(20);
  });

  it('lets the nested leaf layer set its own order', () => {
    const { engine, state, mesh } = nestedLayerHarness(20, 5);

    state.setLayerOrder('/g/inner', 7);
    engine.applyLayerOrder(state.getLayer('/g/inner')!);

    expect(orderOf(mesh)).toBe(7);
  });

  // Clearing must reach the mesh as absent, not as a stale number — the
  // renderer's "authored" test is `!== undefined`.
  it('clearing every order in the chain leaves the leaf unset', () => {
    const { engine, state, mesh } = nestedLayerHarness(20, 5);

    state.setLayerOrder('/g/inner', undefined);
    state.setLayerOrder('/g', undefined);
    engine.applyLayerOrder(state.getLayer('/g')!);

    expect(orderOf(mesh)).toBeUndefined();
  });

  it('does not mutate the loaded leaf attrs record', () => {
    const { engine, state, mesh } = nestedLayerHarness(20, undefined);
    const attrs = mesh.userData.attrs;

    expect(() => engine.applyLayerOrder(state.getLayer('/g')!)).not.toThrow();
    expect(orderOf(mesh)).toBe(20);
    expect(mesh.userData.attrs).toBe(attrs);
    expect(attrs).toEqual({});
  });
});
