/**
 * Widths of the global fetch gate (`utils/fetch-concurrency.ts`).
 *
 * The caps must never exceed what the browser actually puts on the wire,
 * because `fetchWithRetry` starts each attempt's header timer when the gate
 * admits it: a gate wider than the socket pool parks the surplus in the
 * browser's own queue with the timer already running.
 */
export interface FetchGateConfig {
  /**
   * Data-lane width for an HTTP/1.1 or not-yet-identified origin, and the cap
   * on chunk responses in flight across every data path. 24 keeps enough HTTP/2
   * streams ready to fill ordinary broadband while bounding a representative
   * 500 KiB chunk wave to about 12 MiB.
   */
  maxChunkFetches: number;
  /**
   * Data-lane width for an origin resource timing shows negotiated h2/h3 (the
   * global ceiling of the data lane). Measured on hosted HTTP/2 (100 ms RTT,
   * 25 Mbps): at 24 the lane sat pinned full before first frame; at 96 first
   * frame improved 14% and settle 15-21% (h2afva).
   */
  maxMultiplexedChunkFetches: number;
  /** Metadata-lane width (root `zarr.json` / `.zattrs` probes). */
  maxMetadataFetches: number;
  /**
   * Data leases one origin seen over plain `http:` may hold. With
   * {@link http1MaxMetadataFetches} it should equal the six sockets every
   * current browser opens per HTTP/1.1 origin, so an admitted request is a
   * dispatched request.
   */
  http1MaxChunkFetches: number;
  /** Metadata leases one plain-`http:` origin may hold (its own two sockets). */
  http1MaxMetadataFetches: number;
  /** Largest fraction of a lane `speculative` requests may occupy (at least one slot). */
  speculativeShare: number;
}

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
  /** The fetch gate's lane widths (see {@link FetchGateConfig}). */
  fetchGate: FetchGateConfig;
}
