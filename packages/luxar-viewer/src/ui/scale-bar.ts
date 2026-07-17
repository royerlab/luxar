/**
 * Scale Bar Overlay Component
 *
 * Renders a physical scale bar in the viewer using the `unit` metadata
 * from Dimensions. Computes the bar width from camera distance and FOV
 * at the orbit target depth (accurate at scene center).
 *
 * For perspective projection, the bar represents physical distance at the
 * camera's focus target. For orthographic projection, it is exact everywhere
 * since visible height is independent of depth.
 */

import { UIComponent } from './overlay-widgets/ui-component';
import { ControlsManager } from '../controls/controls-manager';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import { type LuxarCamera, isPerspectiveCamera } from '../utils/camera-utils';

export interface ScaleBarConfig {
  /**
   * Live camera accessor, NOT a captured instance: the camera is the one
   * component the scene manager REPLACES at runtime (perspective ↔ ortho
   * swap in camera-mode.ts). A captured reference goes stale after the
   * first V-key mode switch, freezing the scale bar on the abandoned
   * camera (ortho zoom then never updates the label).
   */
  getCamera: () => LuxarCamera;
  controls: ControlsManager;
  canvas: HTMLCanvasElement;
  targetWidthPx: number;
  position: 'bottom-left' | 'bottom-right';
}

/**
 * Pick a "nice" round number (1, 2, 5 × 10^n) closest to the raw value.
 * These are the standard scale bar increments used in microscopy.
 */
export function computeNiceValue(raw: number): number {
  if (raw <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const residual = raw / magnitude;
  if (residual < 1.5) return magnitude;
  if (residual < 3.5) return 2 * magnitude;
  if (residual < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/**
 * Format a scale value for display.
 * Since computeNiceValue only returns {1, 2, 5} × 10^n, results are always clean.
 * Uses parseFloat to strip trailing zeros (e.g., "0.50" → "0.5").
 */
export function formatScaleValue(value: number): string {
  if (value >= 1 && Number.isInteger(value)) return String(value);
  return parseFloat(value.toPrecision(2)).toString();
}

export class ScaleBar extends UIComponent<ScaleBarConfig> {
  // NOTE: Do NOT declare barElement/labelElement as class fields here.
  // With useDefineForClassFields (TypeScript default), field initializers run
  // AFTER super() returns, which would overwrite values set in render() with undefined.
  // Instead, we query the DOM in update() — the tree is only 3 nodes, so it's trivial.
  private lastBarWidthPx = 0;
  private lastLabelText = '';

  protected getClassName(): string {
    return 'luxar-scale-bar';
  }

  protected render(): HTMLElement {
    const el = document.createElement('div');
    el.className = this.getClassName();

    // Position modifier
    if (this.config.position === 'bottom-right') {
      el.classList.add(`${this.getClassName()}--bottom-right`);
    }

    const bar = document.createElement('div');
    bar.className = `${this.getClassName()}__bar`;

    const label = document.createElement('div');
    label.className = `${this.getClassName()}__label`;

    el.appendChild(bar);
    el.appendChild(label);

    return el;
  }

  /**
   * Update the scale bar based on current camera state.
   * Called per-frame from the animation loop.
   */
  update(): void {
    if (!this.isVisible()) return;

    const { getCamera, controls, canvas, targetWidthPx } = this.config;
    const camera = getCamera();

    // Compute world-units-per-pixel at the orbit target depth
    const target = controls.getFocusTarget();
    const d = camera.position.distanceTo(target);
    if (d === 0) return;

    const visibleHeight = isPerspectiveCamera(camera)
      ? 2 * d * Math.tan((camera.fov * Math.PI) / 360)
      : (camera.top - camera.bottom) / camera.zoom;
    const canvasHeight = canvas.clientHeight;
    if (canvasHeight === 0) return;

    const worldPerPx = visibleHeight / canvasHeight;

    // Pick a nice round number that fits near the target width
    const rawValue = worldPerPx * targetWidthPx;
    const niceValue = computeNiceValue(rawValue);
    const barWidthPx = niceValue / worldPerPx;

    // Get unit from the first displayed spatial dimension
    const unit = this.getDisplayUnit();

    // Build label
    const labelText = unit ? `${formatScaleValue(niceValue)} ${unit}` : formatScaleValue(niceValue);

    // Query child elements from our small DOM tree (3 nodes — negligible cost)
    const barEl = this.element.querySelector(`.${this.getClassName()}__bar`) as HTMLElement;
    const labelEl = this.element.querySelector(`.${this.getClassName()}__label`) as HTMLElement;
    if (!barEl || !labelEl) return;

    // Only update DOM if values changed (avoid layout thrashing)
    if (barWidthPx !== this.lastBarWidthPx) {
      barEl.style.width = `${Math.round(barWidthPx)}px`;
      this.lastBarWidthPx = barWidthPx;
    }
    if (labelText !== this.lastLabelText) {
      labelEl.textContent = labelText;
      this.lastLabelText = labelText;
    }
  }

  private getDisplayUnit(): string {
    const units = sceneDimsManager.getDimensionUnits();
    const dims = sceneDimsManager.getDims();
    if (!units || !dims || !dims.displayed) return '';

    // Use the unit from the first displayed dimension
    const firstDisplayedIdx = dims.displayed[0];
    if (firstDisplayedIdx !== undefined && firstDisplayedIdx < units.length) {
      return units[firstDisplayedIdx];
    }
    return '';
  }
}
