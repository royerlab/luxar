/**
 * Low Power Mode Indicator Component
 *
 * A subtle visual indicator that appears when the adaptive DPR system
 * has reduced rendering resolution to maintain smooth frame rates.
 *
 * Shows a lightning bolt icon and "Low Power Mode" text with the current
 * DPR value. Automatically shows/hides based on the AdaptiveDPRManager state.
 */

import { log, Modules } from '../../utils/log';

/**
 * Manages the low power mode indicator UI
 */
export class LowPowerIndicator {
  private element: HTMLDivElement | null = null;
  private isVisible: boolean = false;
  private hideTimeout: ReturnType<typeof setTimeout> | null = null;
  private autoHideTimeout: ReturnType<typeof setTimeout> | null = null;
  private hasShownForCurrentMode: boolean = false;

  /** Duration to show the indicator before auto-hiding (ms) */
  private static readonly AUTO_HIDE_DELAY = 4000;

  /**
   * Create the low power indicator
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
    this.element.className = 'luxar-low-power-indicator luxar-low-power-indicator--hidden';
    this.element.innerHTML = `
      <span class="luxar-low-power-indicator__icon">⚡</span>
      <span class="luxar-low-power-indicator__text">Low Power Mode</span>
      <span class="luxar-low-power-indicator__dpr"></span>
    `;

    // Hide initially
    this.element.style.display = 'none';

    document.body.appendChild(this.element);

    return this.element;
  }

  /**
   * Show the indicator with optional DPR value
   *
   * Only shows once per low power mode activation. After showing for 4 seconds,
   * the indicator auto-hides. Call reset() when exiting low power mode to allow
   * the indicator to show again on next activation.
   *
   * @param dpr - Current device pixel ratio to display
   */
  show(dpr?: number): void {
    // Only show once per mode activation
    if (this.hasShownForCurrentMode) {
      return;
    }

    if (this.isVisible) {
      // Just update DPR if already visible
      this.updateDPR(dpr);
      return;
    }

    // Clear any pending hide timeout
    if (this.hideTimeout) {
      clearTimeout(this.hideTimeout);
      this.hideTimeout = null;
    }

    const element = this.ensureElement();

    // Update DPR display
    this.updateDPR(dpr);

    // Show with animation
    element.style.display = 'flex';
    element.classList.remove('luxar-low-power-indicator--hidden');

    this.isVisible = true;
    this.hasShownForCurrentMode = true;
    log.info(Modules.ADAPTIVE_DPR, 'Low power indicator shown (will auto-hide in 4s)');

    // Auto-hide after 4 seconds
    this.autoHideTimeout = setTimeout(() => {
      this.hide();
      this.autoHideTimeout = null;
    }, LowPowerIndicator.AUTO_HIDE_DELAY);
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
    this.element.classList.add('luxar-low-power-indicator--hidden');

    // Remove from DOM after animation completes
    this.hideTimeout = setTimeout(() => {
      if (this.element) {
        this.element.style.display = 'none';
      }
      this.hideTimeout = null;
    }, 300); // Match animation duration

    this.isVisible = false;
    log.info(Modules.ADAPTIVE_DPR, 'Low power indicator hidden');
  }

  /**
   * Reset the indicator state when exiting low power mode
   *
   * This allows the indicator to show again on the next low power mode activation.
   * Should be called when low power mode is turned OFF.
   */
  reset(): void {
    this.hasShownForCurrentMode = false;
    // Also hide if still visible
    if (this.isVisible) {
      this.hide();
    }
  }

  /**
   * Update the DPR display value
   *
   * @param dpr - Current device pixel ratio
   */
  updateDPR(dpr?: number): void {
    if (!this.element) return;

    const dprElement = this.element.querySelector('.luxar-low-power-indicator__dpr');
    if (dprElement) {
      dprElement.textContent = dpr !== undefined ? `(${(dpr * 100).toFixed(0)}%)` : '';
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
    log.info(Modules.ADAPTIVE_DPR, 'Low power indicator disposed');
  }
}
