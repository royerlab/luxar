/**
 * One-shot async initializer with retry-on-failure semantics.
 *
 * Each spatial-index loader wraps its `initialize()` method in the same
 * guard: cache the in-flight promise so concurrent callers all await the
 * same work, but null it back out on rejection so the next caller can
 * retry from scratch. That pattern was duplicated four times across the
 * three loaders. This helper centralizes it.
 *
 * @example
 * ```ts
 * private _onceInit = new OnceInit();
 *
 * async loadGSplats(...) {
 *   await this._onceInit.ensure(() => this.initialize());
 *   // ...
 * }
 * ```
 *
 * @module data/loaders/once-init
 */

export class OnceInit {
  private inFlight: Promise<void> | null = null;

  /**
   * Run `initFn` once. Concurrent calls share the in-flight promise. If
   * `initFn` rejects, the cached promise is cleared so a subsequent call
   * starts fresh — callers can retry without constructing a new instance.
   *
   * @param initFn - The work to run at most once. Returning a rejected
   *   promise allows the next call to retry; resolved means "done forever".
   */
  async ensure(initFn: () => Promise<void>): Promise<void> {
    if (!this.inFlight) {
      this.inFlight = initFn().catch((err) => {
        this.inFlight = null;
        throw err;
      });
    }
    await this.inFlight;
  }

  /** True once `ensure` has resolved. */
  get isInitialized(): boolean {
    return this.inFlight !== null;
  }

  /**
   * Force a fresh initialization next time. Mainly useful in tests; in
   * production callers should let the retry-on-failure path handle reset.
   */
  reset(): void {
    this.inFlight = null;
  }
}
