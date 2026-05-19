/**
 * Data loading network configuration
 */
export interface DataLoadingNetworkConfig {
  timeoutMs: number;
  /**
   * Dedicated short budget for the L2 cache-validation HEAD probe. On flaky
   * networks the validation must NOT block scene loading for the full
   * `timeoutMs` — failing fast lets cached data render quickly.
   *
   * **Trade-off**: lower values fail faster (good — render from cached
   * data while the network is slow). Higher values tolerate slower
   * networks but block first-paint until the validation completes or
   * times out. Default is `5000` (5 s).
   *
   * **3G / Edge / high-latency**: real-world 3G round-trip + server
   * processing can exceed 5 s, which would cause spurious validation
   * timeouts and force re-fetches of otherwise-valid cached data. If you
   * target slow networks, raise this to `>=8000` (8 s).
   *
   * The validation layer logs a warning when this drops below 3 s (the
   * "almost certainly broken" floor); 5 s is the broadband-tuned
   * default and does not warn.
   */
  validationTimeoutMs: number;
  maxConcurrent: number;
  retryAttempts: number;
}
