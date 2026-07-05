/**
 * Resolution Indicator Component
 *
 * A subtle visual indicator that appears when the adaptive DPR system
 * has reduced rendering resolution to maintain smooth frame rates.
 *
 * Shows a scaling icon and "Resolution Scaled to X% to maintain Y fps" text.
 * Automatically shows/hides based on the AdaptiveDPRManager state.
 */

import { log, Modules } from '../utils/log';
import { getViewerContainer } from '../utils/viewer-container';

/**
 * Manages the resolution indicator UI
 */
export class ResolutionIndicator {
  private element: HTMLDivElement | null = null;
  private isVisible: boolean = false;
  private hideTimeout: ReturnType<typeof setTimeout> | null = null;
  private autoHideTimeout: ReturnType<typeof setTimeout> | null = null;
  private hasShownForCurrentMode: boolean = false;
  private targetFPS: number = 60; // Default, will be set from config

  /** Duration to show the indicator before auto-hiding (ms) */
  private static readonly AUTO_HIDE_DELAY = 4000;

  /**
   * Create the resolution indicator
   * The indicator is created lazily when first shown
   */
  constructor() {
    // Element created lazily on first show
  }

  /**
   * Create the DOM element if it doesn't exist
   */
  private ensureElement(): HTMLDivElement {
    if (this.element) {
      return this.element;
    }

    this.element = document.createElement('div');
    // luxar-glass-surface: opt into the frosted/liquid-glass material so the
    // indicator matches the other floating panels under those themes.
    this.element.className =
      'luxar-resolution-indicator luxar-resolution-indicator--hidden luxar-glass-surface';
    this.element.innerHTML = `
      <span class="luxar-resolution-indicator__icon">&#x21C5;</span>
      <span class="luxar-resolution-indicator__text"></span>
    `;

    // Hide initially
    this.element.style.display = 'none';

    getViewerContainer().appendChild(this.element);

    return this.element;
  }

  /**
   * Set the target FPS for display purposes
   *
   * @param fps - Target FPS from config
   */
  setTargetFPS(fps: number): void {
    this.targetFPS = fps;
  }

  /**
   * Show the indicator with the current resolution scale
   *
   * Only shows once per reduced resolution mode activation. After showing for 4 seconds,
   * the indicator auto-hides. Call reset() when exiting reduced resolution mode to allow
   * the indicator to show again on next activation. Further show() calls while the
   * toast is still visible refresh the displayed percentage in place (no re-show,
   * no timer extension) so a multi-step DPR walk doesn't freeze the first value.
   *
   * @param scale - Current resolution as a fraction of native DPR (0..1]
   */
  show(scale?: number): void {
    // Only show once per mode activation. While the toast from this
    // activation is still on screen, later calls refresh the text in
    // place (the adaptive caller streams every DPR step); after it has
    // auto-hidden, they are ignored entirely until reset() re-arms.
    if (this.hasShownForCurrentMode) {
      if (this.isVisible) {
        this.updateText(scale);
      }
      return;
    }

    // Clear any pending hide timeout
    if (this.hideTimeout) {
      clearTimeout(this.hideTimeout);
      this.hideTimeout = null;
    }

    const element = this.ensureElement();

    // Update text display
    this.updateText(scale);

    // Show with animation
    element.style.display = 'flex';
    element.classList.remove('luxar-resolution-indicator--hidden');

    this.isVisible = true;
    this.hasShownForCurrentMode = true;
    const autoHideSeconds = ResolutionIndicator.AUTO_HIDE_DELAY / 1000;
    log.info(
      Modules.ADAPTIVE_DPR,
      `Resolution indicator shown (will auto-hide in ${autoHideSeconds}s)`
    );

    // Auto-hide after AUTO_HIDE_DELAY
    this.autoHideTimeout = setTimeout(() => {
      this.hide();
      this.autoHideTimeout = null;
    }, ResolutionIndicator.AUTO_HIDE_DELAY);
  }

  /**
   * Hide the indicator with animation
   */
  hide(): void {
    if (!this.isVisible || !this.element) {
      return;
    }

    // Clear any pending timeouts
    if (this.hideTimeout) {
      clearTimeout(this.hideTimeout);
      this.hideTimeout = null;
    }
    if (this.autoHideTimeout) {
      clearTimeout(this.autoHideTimeout);
      this.autoHideTimeout = null;
    }

    // Start hide animation
    this.element.classList.add('luxar-resolution-indicator--hidden');

    // Remove from DOM after animation completes
    this.hideTimeout = setTimeout(() => {
      if (this.element) {
        this.element.style.display = 'none';
      }
      this.hideTimeout = null;
    }, 300); // Match animation duration

    this.isVisible = false;
    log.info(Modules.ADAPTIVE_DPR, 'Resolution indicator hidden');
  }

  /**
   * Reset the indicator state when exiting reduced resolution mode
   *
   * This allows the indicator to show again on the next reduced resolution mode activation.
   * Should be called when reduced resolution mode is turned OFF.
   */
  reset(): void {
    this.hasShownForCurrentMode = false;
    // Also hide if still visible
    if (this.isVisible) {
      this.hide();
    }
  }

  /**
   * Update the text display
   *
   * @param scale - Current resolution as a fraction of native DPR (0..1]
   */
  private updateText(scale?: number): void {
    if (!this.element) return;

    const textElement = this.element.querySelector('.luxar-resolution-indicator__text');
    if (textElement) {
      if (scale !== undefined) {
        const percentage = (scale * 100).toFixed(0);
        textElement.textContent = `Resolution Scaled to ${percentage}% to maintain ${this.targetFPS} fps`;
      } else {
        textElement.textContent = `Resolution Scaled to maintain ${this.targetFPS} fps`;
      }
    }
  }

  /**
   * Check if the indicator is currently visible
   */
  getIsVisible(): boolean {
    return this.isVisible;
  }

  /**
   * Dispose of the indicator and clean up resources
   */
  dispose(): void {
    if (this.hideTimeout) {
      clearTimeout(this.hideTimeout);
      this.hideTimeout = null;
    }

    if (this.autoHideTimeout) {
      clearTimeout(this.autoHideTimeout);
      this.autoHideTimeout = null;
    }

    if (this.element) {
      this.element.remove();
      this.element = null;
    }

    this.isVisible = false;
    this.hasShownForCurrentMode = false;
    log.info(Modules.ADAPTIVE_DPR, 'Resolution indicator disposed');
  }
}
