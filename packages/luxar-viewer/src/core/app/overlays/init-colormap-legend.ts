import { ColormapLegend } from '../../../ui/colormap-legend';
import type { LayersPanel } from '../../../ui/layers';
import type { InputHandler } from '../../../input/input-handler';

/**
 * Build a fresh {@link ColormapLegend} for the current layers panel and
 * wire it through the input handler so the keyboard toggle keeps working.
 * Disposes the previous instance when present. Returns `undefined` when
 * no LayersPanel is set (no layer state to render), or when DOM is not
 * available (test/headless environments).
 */
export interface InitColormapLegendPorts {
  previous: ColormapLegend | undefined;
  layersPanel: LayersPanel | undefined;
  inputHandler: InputHandler;
}

export function initColormapLegend(ports: InitColormapLegendPorts): ColormapLegend | undefined {
  if (ports.previous) {
    ports.previous.dispose();
  }

  if (!ports.layersPanel) return undefined;

  try {
    const legend = new ColormapLegend({
      layerState: ports.layersPanel.layerState,
    });

    ports.inputHandler.setColormapLegend(legend);
    return legend;
  } catch {
    // ColormapLegend requires DOM; may fail in test environments
    return undefined;
  }
}
