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
      if (Math.abs(this.context.settings.near - currentNear) > 0.0001) {
        this.context.settings.near = currentNear;
        nearPlane.updateDisplay();
      }
    }

    const farPlane = this.context.getFarPlane();
    if (farPlane) {
      const currentFar = camera.far;
      if (Math.abs(this.context.settings.far - currentFar) > 0.1) {
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
