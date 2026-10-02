import type { AppConfig } from '../../../types';
import type { FetchGateConfig } from './types';

/** Browsers open six sockets per HTTP/1.1 origin. */
const HTTP1_SOCKETS_PER_ORIGIN = 6;

/**
 * Validate the fetch-gate widths: positive integer lanes, a multiplexed lane
 * no narrower than the default one, and a speculative share in (0, 1]. Warns
 * when the HTTP/1.1 caps would not fill (or would overrun) an origin's sockets.
 */
function validateFetchGate(gate: FetchGateConfig, errors: string[], warnings: string[]): void {
  const lanes = [
    'maxChunkFetches',
    'maxMultiplexedChunkFetches',
    'maxMetadataFetches',
    'http1MaxChunkFetches',
    'http1MaxMetadataFetches',
  ] as const;
  for (const lane of lanes) {
    if (!Number.isInteger(gate[lane]) || gate[lane] <= 0) {
      errors.push(`Invalid fetchGate.${lane}: ${gate[lane]} (must be a positive integer)`);
    }
  }
  if (gate.maxMultiplexedChunkFetches < gate.maxChunkFetches) {
    errors.push(
      `Invalid fetchGate.maxMultiplexedChunkFetches: ${gate.maxMultiplexedChunkFetches} ` +
        `(must be at least maxChunkFetches, ${gate.maxChunkFetches})`
    );
  }
  if (!(gate.speculativeShare > 0 && gate.speculativeShare <= 1)) {
    errors.push(`Invalid fetchGate.speculativeShare: ${gate.speculativeShare} (must be in (0, 1])`);
  }
  const http1 = gate.http1MaxChunkFetches + gate.http1MaxMetadataFetches;
  if (http1 !== HTTP1_SOCKETS_PER_ORIGIN) {
    warnings.push(
      `fetchGate HTTP/1.1 caps total ${http1} leases per origin; browsers open ` +
        `${HTTP1_SOCKETS_PER_ORIGIN} sockets, so admitted requests may wait in the browser ` +
        'queue with their header timers running (or leave sockets idle)'
    );
  }
}

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
  validateFetchGate(network.fetchGate, errors, warnings);
}
