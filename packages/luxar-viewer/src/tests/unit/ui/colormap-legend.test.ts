/**
 * Unit tests for ColormapLegend component
 *
 * Verifies the legend's reactive lifecycle: subscription to layer state,
 * hash-based skip on identical updates, and DOM structure of legend entries.
 *
 * AUDIT NOTE (ui.md C5): the tests below construct a hand-rolled
 * LayerStateManager mock instead of a real instance. LayerStateManager is
 * a sibling module in the same subpackage (with dedicated tests under
 * `layers/layer-state.test.ts`), not a trust boundary. A regression
 * where the legend reads `intensity/offset` instead of `displayMin/Max`
 * from a real LayerStateManager would be invisible to this file because
 * the mock returns whatever shape the test author wrote. Follow-up:
 * construct a real LayerStateManager with stub source data.
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
import type { LayerInfo, LayerStateManager } from '../../../ui/layers/layer-state';

function makeLayer(overrides: Partial<LayerInfo> = {}): LayerInfo {
  return {
    path: 'group/channel_0',
    name: 'channel_0',
    type: 'points',
    visible: true,
    opacity: 1,
    displayMin: 0,
    displayMax: 1,
    dataMin: 0,
    dataMax: 1,
    gamma: 1,
    blendingMode: 'additive',
    selected: false,
    colormap: 'viridis',
    supportsColormap: true,
    ...overrides,
  };
}

function makeLayerState(initialLayers: LayerInfo[] = []) {
  let layers = initialLayers;
  const callbacks: Array<() => void> = [];

  const state = {
    getLayers: vi.fn(() => layers),
    onChange: vi.fn((cb: () => void) => {
      callbacks.push(cb);
      return () => {
        const idx = callbacks.indexOf(cb);
        if (idx >= 0) callbacks.splice(idx, 1);
      };
    }),
    setLayers(next: LayerInfo[]) {
      layers = next;
      callbacks.forEach((cb) => cb());
    },
    callbackCount: () => callbacks.length,
  };

  return state as unknown as LayerStateManager & {
    setLayers: (next: LayerInfo[]) => void;
    callbackCount: () => number;
  };
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

      const canvas = entries[0].querySelector<HTMLCanvasElement>('.luxar-colormap-legend__gradient');
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
