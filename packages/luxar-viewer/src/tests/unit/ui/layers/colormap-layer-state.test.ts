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

  it('supportsColormap is true for a bare gsplats leaf, as for the same data under a wrapper', () => {
    // A gsplat's amplitude IS its scalar (`supportsScalarColormap('gsplats')` is
    // unconditionally true), so a palette picked on a bare leaf really renders.
    // The leaf used to answer false while a kind=lod wrapper over the very same
    // leaf answered true: the dropdown appeared or not depending on topology.
    mgr.initFromSceneGraph(makeSceneGraph([{}]));
    expect(mgr.getLayers()[0].supportsColormap).toBe(true);

    mgr.initFromSceneGraph({
      path: '',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [
        {
          path: 'lod',
          type: 'group',
          hasSpatialIndex: false,
          attrs: { layer: true, kind: 'lod', display_type: 'gsplats' },
          children: [{ path: 'lod/l0', type: 'gsplats', hasSpatialIndex: false, attrs: {} }],
        },
      ],
    });
    expect(mgr.getLayers()[0].supportsColormap).toBe(true);
  });

  it('carries the authored LUT of a custom palette, own or inherited', () => {
    // The compiler stores every non-builtin palette as `colormap: 'custom'` plus a
    // sibling `colormap_lut`, which the loader hands over as `customLutBytes`. The
    // panel needs the bytes to offer and draw that palette (dropdown, menu, legend).
    const own = new Uint8Array(768).fill(7);
    const inherited = new Uint8Array(768).fill(9);
    mgr.initFromSceneGraph({
      path: '',
      type: 'scene',
      hasSpatialIndex: false,
      attrs: {},
      children: [
        {
          path: 'own',
          type: 'gsplats',
          hasSpatialIndex: false,
          attrs: { layer: true, colormap: 'custom', customLutBytes: own },
        },
        {
          path: 'palette',
          type: 'group',
          hasSpatialIndex: false,
          attrs: { colormap: 'custom', customLutBytes: inherited },
          children: [
            { path: 'palette/gs', type: 'gsplats', hasSpatialIndex: false, attrs: { layer: true } },
          ],
        },
        {
          // A kind=partition wrapper: the palette lives on its parts.
          path: 'tiles',
          type: 'group',
          hasSpatialIndex: false,
          attrs: { layer: true, kind: 'partition', display_type: 'gsplats' },
          children: [
            {
              path: 'tiles/part_0',
              type: 'gsplats',
              hasSpatialIndex: false,
              attrs: { colormap: 'custom', customLutBytes: own },
            },
          ],
        },
        {
          path: 'builtin',
          type: 'gsplats',
          hasSpatialIndex: false,
          attrs: { layer: true, colormap: 'viridis' },
        },
      ],
    });
    expect(mgr.getLayer('own')!.customLut).toBe(own);
    expect(mgr.getLayer('tiles')!.colormap).toBe('custom');
    expect(mgr.getLayer('tiles')!.customLut).toBe(own);
    expect(mgr.getLayer('palette/gs')!.colormap).toBe('custom');
    expect(mgr.getLayer('palette/gs')!.customLut).toBe(inherited);
    expect(mgr.getLayer('builtin')!.customLut).toBeUndefined();
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
