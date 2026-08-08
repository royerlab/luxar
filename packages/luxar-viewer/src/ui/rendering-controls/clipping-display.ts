/**
 * Dynamic clipping display controller for the rendering-controls panel.
 *
 * When dynamic clipping is ON, the near/far sliders should:
 *   - be greyed out and pointer-disabled (read-only),
 *   - show the live camera near/far values, refreshed via a throttled RAF loop.
 *
 * When dynamic clipping is OFF the sliders go back to normal.
 *
 * Owns the RAF id so callers can fully tear down the loop on dispose.
 */

import type { Controller } from '../gui';
import type { RenderingSettings } from '../../config';
import type { SceneManager } from '../../scene/scene-manager';

const CLIPPING_DISPLAY_THROTTLE_MS = 100;

/**
 * Relative drift at which the readout is refreshed, matching the 0.1% gate
 * `updateDynamicFromCache` uses before it touches the camera at all.
 *
 * RELATIVE, not absolute: near/far are scene-scaled quantities, and the
 * absolute thresholds this replaced (1e-4 for near, 0.1 for far) were tuned
 * for a scene roughly 100 world units across. On a micron-scale scene every
 * real change falls under them, so the sliders froze on their defaults and
 * stopped being the live readout dynamic clipping advertises. Matching the
 * camera's own gate also means the display can never be the coarser of the
 * two — anything the camera bothered to change, this shows.
 */
const CLIPPING_DISPLAY_REL_EPSILON = 0.001;

export interface ClippingDisplayContext {
  sceneManager: SceneManager;
  settings: RenderingSettings;
  /**
   * Live reference to the rendering controllers map. The near/far controllers
   * may not exist when this controller is constructed (they're added by the
   * camera-setup module), so we look them up on every call.
   */
  getNearPlane: () => Controller | undefined;
  getFarPlane: () => Controller | undefined;
}

/**
 * Whether the displayed value has drifted far enough from the live camera
 * value to be worth a re-render.
 *
 * A non-finite camera value is never mirrored into the settings — the readout
 * keeps its last good value rather than showing (and stamping in) NaN.
 */
function hasDrifted(shown: number, current: number): boolean {
  if (!Number.isFinite(current)) return false;
  const magnitude = Math.abs(current);
  if (magnitude === 0) return shown !== 0;
  return Math.abs(shown - current) > magnitude * CLIPPING_DISPLAY_REL_EPSILON;
}

export class ClippingDisplay {
  private rafId: number | null = null;
  private lastUpdate: number = 0;

  constructor(private readonly context: ClippingDisplayContext) {}

  /**
   * Apply enabled state. When `dynamicEnabled` is true the near/far sliders
   * are visually muted and a RAF loop pulls live camera values into them.
   * When false, the loop is cancelled and the sliders return to normal.
   */
  setDynamicEnabled(dynamicEnabled: boolean): void {
    const opacity = dynamicEnabled ? '0.5' : '1.0';
    const pointerEvents = dynamicEnabled ? 'none' : 'auto';

    const nearPlane = this.context.getNearPlane();
    if (nearPlane) {
      const container = nearPlane.domElement.closest('.luxar-gui__controller');
      if (container instanceof HTMLElement) {
        container.style.opacity = opacity;
        container.style.pointerEvents = pointerEvents;
      }
    }

    const farPlane = this.context.getFarPlane();
    if (farPlane) {
      const container = farPlane.domElement.closest('.luxar-gui__controller');
      if (container instanceof HTMLElement) {
        container.style.opacity = opacity;
        container.style.pointerEvents = pointerEvents;
      }
    }

    this.cancelRAF();

    if (dynamicEnabled) {
      this.refreshDisplays();
      this.scheduleRAF();
    }
  }

  /** Update slider displays from live camera near/far without firing onChange. */
  refreshDisplays(): void {
    const camera = this.context.sceneManager.camera;
    if (!camera) return;

    const nearPlane = this.context.getNearPlane();
    if (nearPlane) {
      const currentNear = camera.near;
      if (hasDrifted(this.context.settings.near, currentNear)) {
        this.context.settings.near = currentNear;
        nearPlane.updateDisplay();
      }
    }

    const farPlane = this.context.getFarPlane();
    if (farPlane) {
      const currentFar = camera.far;
      if (hasDrifted(this.context.settings.far, currentFar)) {
        this.context.settings.far = currentFar;
        farPlane.updateDisplay();
      }
    }
  }

  /** Cancel the RAF loop. Safe to call multiple times. */
  dispose(): void {
    this.cancelRAF();
  }

  private cancelRAF(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  private scheduleRAF(): void {
    const tick = (timestamp: number): void => {
      if (timestamp - this.lastUpdate >= CLIPPING_DISPLAY_THROTTLE_MS) {
        this.refreshDisplays();
        this.lastUpdate = timestamp;
      }
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }
}
