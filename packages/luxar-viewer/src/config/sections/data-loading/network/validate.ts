import type { AppConfig } from '../../../types';

/**
 * Validate data-loading network configuration
 */
export function validateDataLoadingNetwork(
  config: AppConfig,
  errors: string[],
  warnings: string[]
): void {
  const network = config.dataLoading.network;

  // Network validation: reject NaN (comparisons with NaN are always
  // false, so `<= 0` accepts it), Infinity, and non-integers where
  // integer semantics are required.
  if (!Number.isFinite(network.timeoutMs) || network.timeoutMs <= 0) {
    errors.push(
      `Invalid network timeout: ${network.timeoutMs} ms (must be a finite positive number)`
    );
  }
  // validationTimeoutMs is the per-request total budget for cache
  // validation in fetchWithRetry; 0 / negative / NaN / Infinity all
  // produce surprising abort/retry behavior, so reject up front.
  if (!Number.isFinite(network.validationTimeoutMs) || network.validationTimeoutMs <= 0) {
    errors.push(
      `Invalid validation timeout: ${network.validationTimeoutMs} ms (must be a finite positive number)`
    );
  } else if (network.validationTimeoutMs < 3000) {
    // Soft warning, not a hard error. The documented default (5 s) is
    // a fail-fast budget tuned for broadband; values under 3 s are
    // almost always too aggressive — every round-trip including DNS,
    // TLS, and server processing must complete in that window or the
    // validation aborts and forces a re-fetch of otherwise-valid
    // cached data. For 3G / Edge / high-latency targets, raise to
    // >=8000 instead. See
    // `DataLoadingNetworkConfig.validationTimeoutMs` JSDoc.
    warnings.push(
      `Very low cache validation timeout: ${network.validationTimeoutMs} ms ` +
        '(values <3000 ms cause spurious validation aborts; consider 5000 ms default ' +
        'or >=8000 ms for 3G/Edge targets)'
    );
  }
  if (!Number.isInteger(network.maxConcurrent) || network.maxConcurrent <= 0) {
    errors.push(
      `Invalid max concurrent requests: ${network.maxConcurrent} (must be a positive integer)`
    );
  }
  if (!Number.isInteger(network.retryAttempts) || network.retryAttempts < 0) {
    errors.push(
      `Invalid retry attempts: ${network.retryAttempts} (must be a non-negative integer)`
    );
  }
}
