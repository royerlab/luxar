/**
 * The "Tap to enable sound" gate (`SOUND_SPEC.md` §4.4).
 *
 * Browsers refuse to start an `AudioContext` without a user gesture on the
 * page. A kiosk launches Chrome with `--autoplay-policy=no-user-gesture-required`
 * and never sees this; everywhere else, when the context is still `suspended`
 * after load, the engine shows this minimal overlay and the first pointer or
 * key event anywhere dismisses it and resumes the context.
 *
 * @module audio/autoplay-gate
 */

/** The tap-to-enable overlay; `show()` / `hide()` are idempotent. */
export class AutoplayGate {
  private element: HTMLElement | null = null;
  private readonly onGesture = (): void => {
    if (!this.element) return;
    this.hide();
    this.onTap();
  };

  constructor(
    private readonly container: () => HTMLElement,
    private readonly onTap: () => void
  ) {}

  /** True while the overlay is on screen. */
  get visible(): boolean {
    return this.element !== null;
  }

  /** Mount the overlay and arm the one-shot dismissal listeners. */
  show(): void {
    if (this.element || typeof document === 'undefined') return;
    const el = document.createElement('div');
    el.className = 'luxar-audio-gate';
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', 'Tap to enable sound');
    el.innerHTML =
      '<div class="luxar-audio-gate__card">' +
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z"/><path d="M15.5 9a4 4 0 0 1 0 6"/><path d="M18 6.5a7.5 7.5 0 0 1 0 11"/></svg>' +
      '<span>Tap to enable sound</span>' +
      '</div>';
    this.container().appendChild(el);
    this.element = el;
    // Capture phase so a tap on the canvas (which stops propagation for its
    // own controls) still counts, and `once` so the listeners self-clean.
    document.addEventListener('pointerdown', this.onGesture, { capture: true, once: true });
    document.addEventListener('keydown', this.onGesture, { capture: true, once: true });
  }

  /** Remove the overlay (and the listeners, if they have not fired). */
  hide(): void {
    if (!this.element) return;
    this.element.remove();
    this.element = null;
    document.removeEventListener('pointerdown', this.onGesture, { capture: true });
    document.removeEventListener('keydown', this.onGesture, { capture: true });
  }

  dispose(): void {
    this.hide();
  }
}
