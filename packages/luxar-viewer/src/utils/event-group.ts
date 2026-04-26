/**
 * EventGroup — register a batch of event listeners (or arbitrary cleanup
 * callbacks) and tear them down with a single `dispose()` call.
 *
 * The viewer used to track listeners through bound-handler fields:
 *
 * ```ts
 * private boundFocusHandler: (() => void) | null = null;
 * // ...
 * this.boundFocusHandler = this.handleFocus.bind(this);
 * window.addEventListener('focus', this.boundFocusHandler);
 * // ...
 * if (this.boundFocusHandler) {
 *   window.removeEventListener('focus', this.boundFocusHandler);
 *   this.boundFocusHandler = null;
 * }
 * ```
 *
 * That pattern is repeated for every listener and is easy to get wrong (a
 * mismatched `removeEventListener`, a forgotten cleanup branch). EventGroup
 * captures the cleanup at registration time so dispose is one call:
 *
 * ```ts
 * private events = new EventGroup();
 *
 * setup() {
 *   this.events.on(window, 'focus', this.handleFocus);
 *   this.events.on(document, 'visibilitychange', this.handleVisibility);
 *   this.events.add(() => this.observer.disconnect());
 * }
 *
 * dispose() {
 *   this.events.dispose();   // removes everything in reverse order
 * }
 * ```
 *
 * Calling `dispose()` more than once is a no-op (cleanup callbacks are
 * popped off the stack as they run).
 */

export class EventGroup {
  private cleanups: Array<() => void> = [];

  /**
   * Register a DOM event listener. Returns a function that unregisters this
   * specific listener early if needed (most callers ignore it and rely on
   * group-level dispose()).
   */
  on<K extends keyof WindowEventMap>(
    target: Window,
    type: K,
    listener: (this: Window, ev: WindowEventMap[K]) => unknown,
    options?: AddEventListenerOptions | boolean
  ): () => void;
  on<K extends keyof DocumentEventMap>(
    target: Document,
    type: K,
    listener: (this: Document, ev: DocumentEventMap[K]) => unknown,
    options?: AddEventListenerOptions | boolean
  ): () => void;
  on<K extends keyof HTMLElementEventMap>(
    target: HTMLElement,
    type: K,
    listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => unknown,
    options?: AddEventListenerOptions | boolean
  ): () => void;
  on(
    target: EventTarget,
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: AddEventListenerOptions | boolean
  ): () => void;
  on(
    target: EventTarget,
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: AddEventListenerOptions | boolean
  ): () => void {
    // Only pass `options` when actually provided. Some hand-rolled mocks
    // (and a small number of older browsers) compare argument arity, and
    // we want a forwarded call equivalent to writing
    // `target.addEventListener(type, listener)` directly.
    if (options === undefined) {
      target.addEventListener(type, listener);
    } else {
      target.addEventListener(type, listener, options);
    }
    let removed = false;
    const cleanup = (): void => {
      if (removed) return;
      removed = true;
      if (options === undefined) {
        target.removeEventListener(type, listener);
      } else {
        target.removeEventListener(type, listener, options);
      }
    };
    this.cleanups.push(cleanup);
    return cleanup;
  }

  /**
   * Register an arbitrary cleanup callback (e.g. `observer.disconnect()`,
   * `cancelAnimationFrame(handle)`, vendor-library `unsubscribe()`). Runs in
   * LIFO order with the listeners on dispose().
   */
  add(cleanup: () => void): void {
    this.cleanups.push(cleanup);
  }

  /**
   * Run every registered cleanup in reverse order and clear the group.
   * Idempotent: a second dispose() does nothing.
   */
  dispose(): void {
    while (this.cleanups.length) {
      const cleanup = this.cleanups.pop();
      try {
        cleanup?.();
      } catch (err) {
        // A failing cleanup must not stop the rest from running. Log via
        // platform console — not the project log utility — to avoid pulling
        // in a dependency that itself relies on event-group cleanup.
        console.error('[EventGroup] cleanup threw:', err);
      }
    }
  }

  /** Number of registered cleanups still pending. Useful for tests. */
  get size(): number {
    return this.cleanups.length;
  }
}
