/**
 * Focus / outside-click management for the rendering-controls panel.
 *
 * Two responsibilities:
 *  - When the panel opens, register a deferred mousedown listener on
 *    `document` (capture phase) that blurs any GUI input when the user
 *    clicks outside the panel — keeping focus on the canvas so keyboard
 *    shortcuts keep working.
 *  - When the panel closes, blur the active GUI input and refocus the
 *    canvas; tear down the deferred listener and any pending timer.
 *
 * The class owns both the timer and the listener so a `dispose()` is
 * always sufficient to leave no document-level listeners behind.
 */

const SETUP_DELAY_MS = 100;

export interface FocusManagerContext {
  /** The GUI's root DOM element — used to test if a click is inside the panel. */
  panel: HTMLElement;
  /** The canvas to refocus on hide / outside-click. */
  canvas: HTMLElement;
}

export class FocusManager {
  private setupTimer: ReturnType<typeof setTimeout> | null = null;
  private outsideClickHandler?: (e: MouseEvent) => void;

  constructor(private readonly context: FocusManagerContext) {}

  /**
   * Schedule the outside-click handler. Deferred so the click that
   * opened the panel doesn't immediately trigger the handler.
   */
  onPanelShown(): void {
    if (this.setupTimer !== null) {
      clearTimeout(this.setupTimer);
    }
    this.setupTimer = setTimeout(() => {
      this.setupTimer = null;
      this.installOutsideClickHandler();
    }, SETUP_DELAY_MS);
  }

  /**
   * Blur the active element, tear down the outside-click handler, and
   * return focus to the canvas. Idempotent.
   */
  onPanelHidden(): void {
    const active = document.activeElement;
    if (active instanceof HTMLElement && typeof active.blur === 'function') {
      active.blur();
    }
    this.cancelPendingSetup();
    this.removeOutsideClickHandler();
    this.context.canvas.focus();
  }

  /** Tear down everything. Always safe to call. */
  dispose(): void {
    this.cancelPendingSetup();
    this.removeOutsideClickHandler();
  }

  private installOutsideClickHandler(): void {
    if (this.outsideClickHandler) return;

    this.outsideClickHandler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (this.context.panel.contains(target)) return;

      const active = document.activeElement;
      if (
        active instanceof HTMLElement &&
        typeof active.blur === 'function' &&
        this.context.panel.contains(active)
      ) {
        active.blur();
        this.context.canvas.focus();
      }
    };

    document.addEventListener('mousedown', this.outsideClickHandler, true);
  }

  private cancelPendingSetup(): void {
    if (this.setupTimer !== null) {
      clearTimeout(this.setupTimer);
      this.setupTimer = null;
    }
  }

  private removeOutsideClickHandler(): void {
    if (this.outsideClickHandler) {
      document.removeEventListener('mousedown', this.outsideClickHandler, true);
      this.outsideClickHandler = undefined;
    }
  }
}
