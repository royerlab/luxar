/**
 * Colormap Legend Overlay
 *
 * Displays a compact legend in the bottom-right corner showing each visible
 * layer's colormap gradient, name, and data range. Toggled with the J key.
 *
 * Updates reactively when layer state changes (colormap, visibility, range).
 */

import { UIComponent } from './overlay-widgets/ui-component';
import type { LayerStateManager, LayerInfo } from './layers/layer-state';
import { BUILTIN_COLORMAPS } from '../rendering/colormap-data';

/**
 * Construction options for {@link ColormapLegend}.
 */
export interface ColormapLegendConfig {
  /** Source of the layer list, colormaps, ranges, and change notifications the legend reflects. */
  layerState: LayerStateManager;
}

/**
 * Draw a colormap gradient onto a canvas element.
 */
function drawColormapGradient(canvas: HTMLCanvasElement, colormapName: string): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const lut = BUILTIN_COLORMAPS[colormapName];
  if (!lut) {
    // Unknown colormap — draw gray
    ctx.fillStyle = '#888';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const w = canvas.width;
  const h = canvas.height;
  for (let x = 0; x < w; x++) {
    const i = Math.floor((x / w) * 255) * 3;
    ctx.fillStyle = `rgb(${lut[i]},${lut[i + 1]},${lut[i + 2]})`;
    ctx.fillRect(x, 0, 1, h);
  }
}

/**
 * Bottom-right overlay listing each visible layer's colormap gradient, name, and
 * data range (toggled with the J key). Subscribes to the {@link LayerStateManager}
 * and rebuilds its entries reactively, but only re-renders the DOM when the
 * visible layers, their colormaps, or their data ranges (displayMin/displayMax)
 * actually change (guarded by a cheap content hash over those fields).
 */
export class ColormapLegend extends UIComponent<ColormapLegendConfig> {
  // `declare` skips the implicit `= undefined` initializer. With
  // useDefineForClassFields=true, a regular field declaration would run
  // AFTER super(), which would wipe the subscription set up by
  // `attachEventListeners` (called from UIComponent's constructor).
  declare private unsubscribe?: () => void;
  private lastHash = '';

  protected getClassName(): string {
    return 'luxar-colormap-legend';
  }

  protected render(): HTMLElement {
    const el = document.createElement('div');
    el.className = this.getClassName();
    return el;
  }

  protected attachEventListeners(): void {
    // Subscribe to layer state changes
    this.unsubscribe = this.config.layerState.onChange(() => {
      this.updateLegend();
    });
  }

  /**
   * Rebuild the legend entries.
   * Only rebuilds DOM if the set of visible layers+colormaps has changed.
   */
  updateLegend(): void {
    if (!this.isVisible()) return;

    const layers = this.config.layerState.getLayers();
    const visibleWithColormap = layers.filter((l) => l.visible && l.colormap);

    // Cheap hash to skip redundant rebuilds
    const hash = visibleWithColormap
      .map((l) => `${l.path}:${l.colormap}:${l.displayMin}:${l.displayMax}`)
      .join('|');
    if (hash === this.lastHash) return;
    this.lastHash = hash;

    // Rebuild
    this.element.innerHTML = '';

    if (visibleWithColormap.length === 0) {
      // Show hint so user knows the legend is active but no colormaps are set
      const hint = document.createElement('div');
      hint.className = `${this.getClassName()}__hint`;
      hint.textContent = 'No colormaps active';
      this.element.appendChild(hint);
      return;
    }

    for (const layer of visibleWithColormap) {
      const entry = this.createEntry(layer);
      this.element.appendChild(entry);
    }
  }

  private createEntry(layer: LayerInfo): HTMLElement {
    const entry = document.createElement('div');
    entry.className = `${this.getClassName()}__entry`;

    // Layer name
    const nameEl = document.createElement('div');
    nameEl.className = `${this.getClassName()}__name`;
    nameEl.textContent = layer.name;
    entry.appendChild(nameEl);

    // Gradient bar
    const barRow = document.createElement('div');
    barRow.className = `${this.getClassName()}__bar-row`;

    const canvas = document.createElement('canvas');
    canvas.className = `${this.getClassName()}__gradient`;
    canvas.width = 120;
    canvas.height = 12;
    drawColormapGradient(canvas, layer.colormap!);
    barRow.appendChild(canvas);
    entry.appendChild(barRow);

    // Min/max labels
    const labelsEl = document.createElement('div');
    labelsEl.className = `${this.getClassName()}__labels`;

    const minEl = document.createElement('span');
    minEl.textContent = formatRange(layer.displayMin);
    const maxEl = document.createElement('span');
    maxEl.textContent = formatRange(layer.displayMax);

    labelsEl.appendChild(minEl);
    labelsEl.appendChild(maxEl);
    entry.appendChild(labelsEl);

    return entry;
  }

  /**
   * Force a full update (call after show()).
   */
  show(): void {
    super.show();
    this.lastHash = '\0'; // Sentinel that never matches a real hash (forces rebuild)
    this.updateLegend();
  }

  protected onDispose(): void {
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = undefined;
    }
  }
}

function formatRange(value: number): string {
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) < 0.01) return value.toExponential(1);
  return value.toFixed(2);
}
