/**
 * Tests for colormap support in LayerStateManager.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { LayerStateManager } from '../../../../ui/layers/layer-state';
import type { SceneNode } from '../../../../data/data-loader-types';

/** Helper to build a minimal scene graph with layer nodes */
function makeSceneGraph(layers: Partial<SceneNode['attrs']>[]): SceneNode {
  return {
    path: '',
    type: 'scene',
    hasSpatialIndex: false,
    attrs: {},
    children: layers.map((attrs, i) => ({
      path: `layer_${i}`,
      type: 'gsplats',
      hasSpatialIndex: false,
      attrs: {
        layer: true,
        ...attrs,
      },
    })),
  };
}

describe('LayerStateManager colormap support', () => {
  let mgr: LayerStateManager;

  beforeEach(() => {
    mgr = new LayerStateManager();
  });

  it('reads colormap from node attrs', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ colormap: 'viridis' }]));
    const layer = mgr.getLayers()[0];
    expect(layer.colormap).toBe('viridis');
  });

  it('colormap is undefined when not set', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    const layer = mgr.getLayers()[0];
    expect(layer.colormap).toBeUndefined();
  });

  it('supportsColormap is false for bare gsplats (no scalars, no colormap attr)', () => {
    // Regression for MED-29: a gsplats layer with only positions/amplitudes/
    // cholesky_factors has no scalar field to drive a colormap. The UI must
    // NOT offer a colormap dropdown that would no-op when the user picks an
    // entry. supportsColormap was previously hard-true for gsplats; now it
    // gates on the same has_scalars / colormap attr that Points uses.
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    const layer = mgr.getLayers()[0];
    expect(layer.supportsColormap).toBe(false);
  });

  it('supportsColormap is true for gsplats with has_scalars', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ has_scalars: true }]));
    const layer = mgr.getLayers()[0];
    expect(layer.supportsColormap).toBe(true);
  });

  it('supportsColormap is true when colormap is set', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ colormap: 'green' }]));
    const layer = mgr.getLayers()[0];
    expect(layer.supportsColormap).toBe(true);
  });

  it('supportsColormap is true when has_scalars', () => {
    const root: SceneNode = {
      path: '',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [
        {
          path: 'pts',
          type: 'points',
          hasSpatialIndex: false,
          attrs: { layer: true, has_scalars: true, colormap: 'inferno' },
        },
      ],
    };
    mgr.initFromSceneGraph(root);
    const layer = mgr.getLayers()[0];
    expect(layer.supportsColormap).toBe(true);
  });

  it('reads scalarDataRange from amplitude_data_range for gsplats', () => {
    mgr.initFromSceneGraph(
      makeSceneGraph([{ amplitude_data_range: [0.1, 0.9] as [number, number] }])
    );
    const layer = mgr.getLayers()[0];
    expect(layer.scalarDataRange).toEqual([0.1, 0.9]);
  });

  it('reads scalarDataRange from scalar_data_range for points', () => {
    const root: SceneNode = {
      path: '',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [
        {
          path: 'pts',
          type: 'points',
          hasSpatialIndex: false,
          attrs: {
            layer: true,
            has_scalars: true,
            scalar_data_range: [0, 100] as [number, number],
            colormap: 'viridis',
          },
        },
      ],
    };
    mgr.initFromSceneGraph(root);
    const layer = mgr.getLayers()[0];
    expect(layer.scalarDataRange).toEqual([0, 100]);
  });

  it('setColormap updates the layer colormap', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ colormap: 'green' }]));
    const layer = mgr.getLayers()[0];
    expect(layer.colormap).toBe('green');

    mgr.setColormap(layer.path, 'magenta');
    expect(mgr.getLayer(layer.path)!.colormap).toBe('magenta');
  });

  it('setColormap notifies listeners', () => {
    mgr.initFromSceneGraph(makeSceneGraph([{ colormap: 'green' }]));

    let notified = false;
    mgr.onChange(() => {
      notified = true;
    });

    mgr.setColormap('layer_0', 'cyan');
    expect(notified).toBe(true);
  });

  it('uses scalar_data_range for display range when available', () => {
    const root: SceneNode = {
      path: '',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [
        {
          path: 'pts',
          type: 'points',
          hasSpatialIndex: false,
          attrs: {
            layer: true,
            scalar_data_range: [10, 50] as [number, number],
            colormap: 'viridis',
          },
        },
      ],
    };
    mgr.initFromSceneGraph(root);
    const layer = mgr.getLayers()[0];
    // Display range should default to scalar_data_range
    expect(layer.dataMin).toBe(10);
    expect(layer.dataMax).toBe(50);
  });
});
