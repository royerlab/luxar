/**
 * Unit tests for ColormapLegend component
 *
 * Verifies the legend's reactive lifecycle: subscription to layer state,
 * hash-based skip on identical updates, and DOM structure of legend entries.
 *
 * [ui.md/C4 / Phase F] The hand-rolled `LayerStateManager` mock that
 * previously lived here has been replaced with the REAL
 * `LayerStateManager`, populated from a synthesised SceneNode tree. A
 * regression where the legend reads `intensity/offset` instead of
 * `displayMin/Max` from a real LayerStateManager would now fail here —
 * the field values come from the production `initFromSceneGraph` walk,
 * not from test-author fiction.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../themes/theme-manager', () => ({
  ThemeManager: {
    getInstance: () => ({
      onChange: vi.fn(() => vi.fn()),
    }),
  },
}));

import { ColormapLegend } from '../../../ui/colormap-legend';
import { LayerStateManager, type LayerInfo } from '../../../ui/layers/layer-state';
import type { SceneNode } from '../../../data/data-loader-types';
import { MESH_DEFAULTS } from '../../../rendering/materials/mesh/appearance';

function makeLayer(overrides: Partial<LayerInfo> = {}): LayerInfo {
  return {
    path: 'group/channel_0',
    name: 'channel_0',
    type: 'points',
    visible: true,
    opacity: 1,
    absorption: 1,
    // Mesh appearance defaults. Present on every LayerInfo (the field set is uniform
    // across types) and inert for a points layer, exactly like `absorption` above.
    ambient: MESH_DEFAULTS.ambient,
    shadeExponent: MESH_DEFAULTS.shadeExponent,
    alphaCutoff: MESH_DEFAULTS.alphaCutoff,
    displayMin: 0,
    displayMax: 1,
    dataMin: 0,
    dataMax: 1,
    gamma: 1,
    blendingMode: 'additive',
    blendingModeExplicit: true,
    selected: false,
    colormap: 'viridis',
    supportsColormap: true,
    scalarWindow: true,
    ...overrides,
  };
}

/**
 * Build a synthesised root SceneNode whose `initFromSceneGraph` walk
 * produces the requested `LayerInfo[]`. The real manager derives the
 * layer `name` from the last path segment, so we route the test's
 * declared name through `path: \`layer/\${l.name}\`` to keep the
 * test-side name override intact under the real walk.
 */
function toSceneGraph(layers: LayerInfo[]): SceneNode {
  return {
    path: '',
    type: 'scene',
    attrs: {},
    hasSpatialIndex: false,
    children: layers.map((l) => ({
      path: `layer/${l.name}`,
      type: l.type,
      attrs: {
        layer: true,
        visible: l.visible,
        opacity: l.opacity,
        gamma: l.gamma,
        blending_mode: l.blendingMode,
        // The real LayerStateManager derives displayMin/Max from the SCALAR
        // range when the layer is colormapped and intensity=1 / offset=0
        // (defaults) — a direct-colour layer instead starts at the identity
        // [0, 1]. The legend only ever renders colormapped layers, so feed the
        // desired range in as scalar_data_range and the derived LayerInfo
        // fields come out matching the test's expectation.
        scalar_data_range: [l.displayMin, l.displayMax],
        colormap: l.colormap,
        has_scalars: l.supportsColormap,
      },
      hasSpatialIndex: true,
    })),
  };
}

/**
 * Wrap a real `LayerStateManager` with the test-only `setLayers` /
 * `callbackCount` accessors the existing tests use. `initFromSceneGraph`
 * is a one-time setup call in production and does NOT notify listeners;
 * the wrapper invokes the private `notify` after a re-init so the
 * legend's `onChange` callback fires (production never needs this
 * because it subscribes AFTER init).
 */
function makeLayerState(initialLayers: LayerInfo[] = []) {
  const mgr = new LayerStateManager();
  mgr.initFromSceneGraph(toSceneGraph(initialLayers));

  return Object.assign(mgr, {
    setLayers(next: LayerInfo[]) {
      mgr.initFromSceneGraph(toSceneGraph(next));
      (mgr as unknown as { notify: () => void }).notify();
    },
    callbackCount: (): number => (mgr as unknown as { listeners: Set<unknown> }).listeners.size,
  });
}

describe('ColormapLegend', () => {
  let layerState: ReturnType<typeof makeLayerState>;
  let legend: ColormapLegend;

  beforeEach(() => {
    document.body.innerHTML = '';
    layerState = makeLayerState();
    legend = new ColormapLegend({ layerState });
  });

  afterEach(() => {
    legend.dispose();
  });

  describe('lifecycle', () => {
    it('subscribes to layer state on construction', () => {
      expect(layerState.callbackCount()).toBe(1);
    });

    it('unsubscribes on dispose', () => {
      legend.dispose();
      expect(layerState.callbackCount()).toBe(0);
      // Reassign so afterEach's dispose() is a no-op
      legend = new ColormapLegend({ layerState });
    });

    it('starts hidden (no visible class until show())', () => {
      expect(legend.isVisible()).toBe(false);
    });

    it('attaches element to DOM and applies visible class on show()', () => {
      legend.show();
      const el = legend.getElement();
      expect(el.parentNode).toBe(document.body);
      expect(legend.isVisible()).toBe(true);
    });
  });

  describe('updateLegend (hidden)', () => {
    it('does nothing when component is hidden', () => {
      layerState.setLayers([makeLayer()]);
      const el = legend.getElement();
      expect(el.children.length).toBe(0);
    });
  });

  describe('updateLegend (visible)', () => {
    beforeEach(() => {
      legend.show();
    });

    it('shows hint when no visible-with-colormap layers', () => {
      const el = legend.getElement();
      const hint = el.querySelector('.luxar-colormap-legend__hint');
      expect(hint?.textContent).toBe('No colormaps active');
    });

    it('skips invisible layers and layers without colormap', () => {
      layerState.setLayers([
        makeLayer({ visible: false }),
        makeLayer({ path: 'no-cmap', name: 'no-cmap', colormap: undefined }),
      ]);
      const el = legend.getElement();
      const hint = el.querySelector('.luxar-colormap-legend__hint');
      expect(hint?.textContent).toBe('No colormaps active');
    });

    it('creates one entry per visible-with-colormap layer', () => {
      layerState.setLayers([
        makeLayer({ path: 'a', name: 'A', colormap: 'viridis' }),
        makeLayer({ path: 'b', name: 'B', colormap: 'plasma' }),
      ]);
      const entries = legend.getElement().querySelectorAll('.luxar-colormap-legend__entry');
      expect(entries.length).toBe(2);
      expect(entries[0].querySelector('.luxar-colormap-legend__name')?.textContent).toBe('A');
      expect(entries[1].querySelector('.luxar-colormap-legend__name')?.textContent).toBe('B');
    });

    it('renders a 120x12 gradient canvas per entry', () => {
      layerState.setLayers([makeLayer()]);
      const canvas = legend
        .getElement()
        .querySelector<HTMLCanvasElement>('.luxar-colormap-legend__gradient');
      expect(canvas).toBeTruthy();
      expect(canvas?.width).toBe(120);
      expect(canvas?.height).toBe(12);
    });

    it('formats integer min/max as plain integer string', () => {
      layerState.setLayers([makeLayer({ displayMin: 0, displayMax: 100 })]);
      const labels = legend.getElement().querySelectorAll('.luxar-colormap-legend__labels span');
      expect(labels[0].textContent).toBe('0');
      expect(labels[1].textContent).toBe('100');
    });

    it('formats fractional min/max with two decimals', () => {
      layerState.setLayers([makeLayer({ displayMin: 0.5, displayMax: 12.345 })]);
      const labels = legend.getElement().querySelectorAll('.luxar-colormap-legend__labels span');
      expect(labels[0].textContent).toBe('0.50');
      expect(labels[1].textContent).toBe('12.35');
    });

    it('formats very small numbers in exponential', () => {
      layerState.setLayers([makeLayer({ displayMin: 0.001, displayMax: 0.005 })]);
      const labels = legend.getElement().querySelectorAll('.luxar-colormap-legend__labels span');
      expect(labels[0].textContent).toMatch(/e/);
      expect(labels[1].textContent).toMatch(/e/);
    });

    it('skips rebuild when layer hash is unchanged', () => {
      const layer = makeLayer();
      layerState.setLayers([layer]);
      const firstCanvas = legend
        .getElement()
        .querySelector<HTMLCanvasElement>('.luxar-colormap-legend__gradient');
      // Trigger another change with the same layers — the hash should match
      // and the DOM should NOT be rebuilt (same canvas instance retained).
      layerState.setLayers([layer]);
      const secondCanvas = legend
        .getElement()
        .querySelector<HTMLCanvasElement>('.luxar-colormap-legend__gradient');
      expect(secondCanvas).toBe(firstCanvas);
    });

    it('forces a rebuild when displayMin/Max changes', () => {
      layerState.setLayers([makeLayer({ displayMax: 1 })]);
      const firstCanvas = legend
        .getElement()
        .querySelector<HTMLCanvasElement>('.luxar-colormap-legend__gradient');
      layerState.setLayers([makeLayer({ displayMax: 2 })]);
      const secondCanvas = legend
        .getElement()
        .querySelector<HTMLCanvasElement>('.luxar-colormap-legend__gradient');
      expect(secondCanvas).not.toBe(firstCanvas);
    });

    it('show() forces a rebuild via the hash sentinel', () => {
      layerState.setLayers([makeLayer()]);
      const firstCanvas = legend
        .getElement()
        .querySelector<HTMLCanvasElement>('.luxar-colormap-legend__gradient');
      legend.hide();
      legend.show();
      const secondCanvas = legend
        .getElement()
        .querySelector<HTMLCanvasElement>('.luxar-colormap-legend__gradient');
      expect(secondCanvas).not.toBe(firstCanvas);
    });

    it('falls back to gray fill for unknown colormap', () => {
      // W5 strengthening (P2): canvas-pixel inspection isn't reliable in
      // jsdom (no GL/2d render), but we can verify (a) the entry was still
      // created, (b) canvas dimensions match the spec, (c) the layer's
      // name and min/max labels rendered, and (d) the gradient does NOT
      // get re-built when the unknown colormap is re-applied with the
      // same min/max (hash sentinel still works on the fallback path).
      layerState.setLayers([
        makeLayer({
          colormap: 'nonexistent-colormap',
          name: 'mystery',
          displayMin: 0,
          displayMax: 42,
        }),
      ]);

      const el = legend.getElement();
      const entries = el.querySelectorAll('.luxar-colormap-legend__entry');
      expect(entries.length).toBe(1);

      const canvas = entries[0].querySelector<HTMLCanvasElement>(
        '.luxar-colormap-legend__gradient'
      );
      expect(canvas).toBeTruthy();
      expect(canvas?.width).toBe(120);
      expect(canvas?.height).toBe(12);

      const nameEl = entries[0].querySelector('.luxar-colormap-legend__name');
      expect(nameEl?.textContent).toBe('mystery');

      const labels = entries[0].querySelectorAll('.luxar-colormap-legend__labels span');
      expect(labels[0].textContent).toBe('0');
      expect(labels[1].textContent).toBe('42');

      // The fallback path must still respect the hash-skip optimisation.
      layerState.setLayers([
        makeLayer({
          colormap: 'nonexistent-colormap',
          name: 'mystery',
          displayMin: 0,
          displayMax: 42,
        }),
      ]);
      const canvasAfterIdempotentUpdate = el.querySelector<HTMLCanvasElement>(
        '.luxar-colormap-legend__gradient'
      );
      expect(canvasAfterIdempotentUpdate).toBe(canvas);
    });
  });
});
