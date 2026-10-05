/**
 * Geometry-behaviour row `colormapSupport`: the Layers panel offers a colormap
 * exactly when the node's material can feed one.
 *
 * Two places answer "can this take a colormap?": the renderer
 * (`supportsScalarColormap`, which reads the geometry's scalar stamp — or, for
 * gsplats, needs none, since amplitude IS the scalar) and the panel
 * (`LayerInfo.supportsColormap`, which reads the scene node's attrs). They
 * drifted: a bare gsplats leaf with no `has_scalars` attr was refused a
 * palette its own `kind=lod` wrapper offered, while the material would have
 * rendered it. The probe asks both, for a leaf and for its wrapper, with and
 * without scalars, and requires one answer.
 */
import * as THREE from 'three';
import { expect } from 'vitest';

import type { SceneNode } from '../../../data/data-loader-types';
import { supportsScalarColormap } from '../../../rendering/material-colormap-helpers';
import type { GeometryTypeName } from '../../../types/format-contract';
import { LayerStateManager } from '../../../ui/layers/layer-state';
import { defineBehaviourConformance } from '../../_conformance/define-behaviour-conformance';

/** The panel's answer for a scene whose one layer is `layerNode`. */
function panelOffersColormap(layerNode: SceneNode): boolean {
  const manager = new LayerStateManager();
  manager.initFromSceneGraph({
    path: '',
    type: 'scene',
    hasSpatialIndex: false,
    attrs: {},
    children: [layerNode],
  });
  const [layer] = manager.getLayers();
  return layer.supportsColormap;
}

function leaf(
  type: GeometryTypeName,
  path: string,
  hasScalars: boolean,
  layer: boolean
): SceneNode {
  return {
    path,
    type,
    hasSpatialIndex: false,
    attrs: { ...(layer ? { layer: true } : {}), ...(hasScalars ? { has_scalars: true } : {}) },
  };
}

/** The renderer's answer for a committed geometry carrying (or not) scalars. */
function materialTakesColormap(type: GeometryTypeName, hasScalars: boolean): boolean {
  const geometry = new THREE.BufferGeometry();
  geometry.userData.hasScalars = hasScalars;
  return supportsScalarColormap(type, geometry);
}

defineBehaviourConformance('colormapSupport', {
  holds(type) {
    let offeredAtAll = false;
    for (const hasScalars of [false, true]) {
      const expected = materialTakesColormap(type, hasScalars);
      offeredAtAll ||= expected;
      const bare = panelOffersColormap(leaf(type, 'n', hasScalars, true));
      const wrapped = panelOffersColormap({
        path: 'lod',
        type: 'group',
        hasSpatialIndex: false,
        attrs: { layer: true, kind: 'lod', display_type: type },
        children: [leaf(type, 'lod/l0', hasScalars, false)],
      });
      const scalars = hasScalars ? 'with scalars' : 'without scalars';
      expect(bare, `${type} leaf ${scalars}: panel vs material`).toBe(expected);
      expect(wrapped, `${type} kind=lod wrapper ${scalars}: panel vs material`).toBe(expected);
    }
    expect(offeredAtAll, `${type} can never take a colormap`).toBe(true);
  },
});
