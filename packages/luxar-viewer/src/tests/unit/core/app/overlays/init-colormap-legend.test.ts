/**
 * Unit tests for core/app/overlays/init-colormap-legend.ts (G9).
 *
 * `initColormapLegend` (re-)builds the legend overlay. Three early-
 * return paths + the success path:
 *
 *   - When previous is set, dispose it first (always).
 *   - When layersPanel is undefined, return undefined (no legend to
 *     render).
 *   - When ColormapLegend construction throws (no DOM in headless /
 *     test environments), return undefined.
 *   - Otherwise: construct legend, wire into inputHandler, return it.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const mocks = vi.hoisted(() => ({
  legendDispose: vi.fn(),
  ColormapLegendCtor: vi.fn(),
}));

vi.mock('../../../../../ui/colormap-legend', () => ({
  ColormapLegend: mocks.ColormapLegendCtor,
}));

import { initColormapLegend } from '../../../../../core/app/overlays/init-colormap-legend';

beforeEach(() => {
  mocks.ColormapLegendCtor.mockReset();
  mocks.legendDispose.mockReset();
  mocks.ColormapLegendCtor.mockImplementation(() => ({
    dispose: mocks.legendDispose,
  }));
});

function makePorts(
  opts: {
    previous?: unknown;
    layersPanel?: unknown;
  } = {}
) {
  return {
    previous: opts.previous as never,
    layersPanel: opts.layersPanel as never,
    inputHandler: {
      setColormapLegend: vi.fn(),
    } as never,
  };
}

describe('initColormapLegend', () => {
  describe('layersPanel guard', () => {
    it('returns undefined when layersPanel is undefined', () => {
      const ports = makePorts({ layersPanel: undefined });
      const result = initColormapLegend(ports);

      expect(result).toBeUndefined();
      // ColormapLegend was NOT constructed when there's nothing to legend.
      expect(mocks.ColormapLegendCtor).not.toHaveBeenCalled();
    });

    it('still disposes the previous legend even when layersPanel is undefined', () => {
      // Reload path: dataset switch may transition from "had layers" to
      // "no layers"; the existing legend must still go away.
      const previousDispose = vi.fn();
      const previous = { dispose: previousDispose };
      const ports = makePorts({ previous, layersPanel: undefined });

      const result = initColormapLegend(ports);

      expect(result).toBeUndefined();
      expect(previousDispose).toHaveBeenCalledOnce();
    });
  });

  describe('success path', () => {
    it('constructs ColormapLegend with the layersPanel.layerState', () => {
      const layerState = { id: 'layer-state' };
      const ports = makePorts({ layersPanel: { layerState } });

      const result = initColormapLegend(ports);

      expect(mocks.ColormapLegendCtor).toHaveBeenCalledExactlyOnceWith({
        layerState,
      });
      expect(result).toBe(mocks.ColormapLegendCtor.mock.results[0].value);
    });

    it('wires the new legend into inputHandler.setColormapLegend', () => {
      const ports = makePorts({ layersPanel: { layerState: {} } });
      const legend = initColormapLegend(ports);

      const input = ports.inputHandler as unknown as {
        setColormapLegend: ReturnType<typeof vi.fn>;
      };
      expect(input.setColormapLegend).toHaveBeenCalledExactlyOnceWith(legend);
    });

    it('disposes previous legend BEFORE constructing the new one', () => {
      const order: string[] = [];
      const previous = { dispose: vi.fn(() => order.push('disposePrev')) };
      mocks.ColormapLegendCtor.mockImplementation(() => {
        order.push('construct');
        return { dispose: mocks.legendDispose };
      });
      const ports = makePorts({ previous, layersPanel: { layerState: {} } });

      initColormapLegend(ports);

      expect(order).toEqual(['disposePrev', 'construct']);
    });
  });

  describe('DOM-unavailable path (test/headless environments)', () => {
    it('returns undefined when ColormapLegend constructor throws (no DOM)', () => {
      mocks.ColormapLegendCtor.mockImplementation(() => {
        throw new Error('no document');
      });
      const ports = makePorts({ layersPanel: { layerState: {} } });

      const result = initColormapLegend(ports);

      expect(result).toBeUndefined();
      // input handler is NOT wired with a stale undefined ref.
      const input = ports.inputHandler as unknown as {
        setColormapLegend: ReturnType<typeof vi.fn>;
      };
      expect(input.setColormapLegend).not.toHaveBeenCalled();
    });

    it('still disposes previous when ColormapLegend construction throws', () => {
      const previousDispose = vi.fn();
      const previous = { dispose: previousDispose };
      mocks.ColormapLegendCtor.mockImplementation(() => {
        throw new Error('no document');
      });
      const ports = makePorts({ previous, layersPanel: { layerState: {} } });

      const result = initColormapLegend(ports);

      expect(result).toBeUndefined();
      expect(previousDispose).toHaveBeenCalledOnce();
    });
  });
});
