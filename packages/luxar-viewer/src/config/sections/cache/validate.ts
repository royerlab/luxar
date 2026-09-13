import type { AppConfig } from '../../types';

/**
 * Validate cache configuration
 *
 * Cache size + OPFS-timeout + external-TTL validation. NaN / Infinity /
 * negative values would cascade into the cache layer sizing logic and
 * surface as cryptic OOMs or zero-budget caches.
 */
export function validateCache(config: AppConfig, errors: string[], _warnings: string[]): void {
  const cache = config.cache;
  if (!cache) return;

  const l0 = cache.l0MaxSizeMB;
  if (!Number.isFinite(l0) || l0 <= 0) {
    errors.push(`Invalid cache.l0MaxSizeMB: ${l0} (must be a finite positive number)`);
  }
  const slice = cache.sliceCacheMaxSizeMB;
  if (!Number.isFinite(slice) || slice <= 0) {
    errors.push(`Invalid cache.sliceCacheMaxSizeMB: ${slice} (must be a finite positive number)`);
  }
  const l1 = cache.l1MaxSizeMB;
  if (!Number.isFinite(l1) || l1 <= 0) {
    errors.push(`Invalid cache.l1MaxSizeMB: ${l1} (must be a finite positive number)`);
  } else if (l1 < 10) {
    // SegmentedLRUCache reserves a 10MB metadata floor; below that
    // the chunks segment becomes zero bytes and every chunk write
    // is silently rejected. Reject the config rather than ship a
    // cache that secretly stores nothing.
    errors.push(
      `Invalid cache.l1MaxSizeMB: ${l1} (must be ≥ 10 — SegmentedLRUCache's metadata floor)`
    );
  }
  const l2 = cache.l2MaxSizeMB;
  if (!Number.isFinite(l2) || l2 <= 0) {
    errors.push(`Invalid cache.l2MaxSizeMB: ${l2} (must be a finite positive number)`);
  }

  // R1: opfsOperationTimeoutMs gates every OPFS read/write via withTimeout.
  // 0 / negative / NaN cause immediate timeout on every op; Infinity disables
  // the safety net entirely.
  const opfsTimeout = cache.opfsOperationTimeoutMs;
  if (!Number.isFinite(opfsTimeout) || opfsTimeout <= 0) {
    errors.push(
      `Invalid cache.opfsOperationTimeoutMs: ${opfsTimeout} (must be a finite positive number; 10000 = 10s recommended)`
    );
  }

  // R1: opfsTimeoutTripThreshold arms the L2 circuit breaker. It counts
  // CONSECUTIVE timeouts, so it must be a whole number >= 1; a fractional
  // value could never be reached exactly and 0/negative would trip the
  // breaker before any evidence of a stall.
  const tripThreshold = cache.opfsTimeoutTripThreshold;
  if (!Number.isInteger(tripThreshold) || tripThreshold < 1) {
    errors.push(
      `Invalid cache.opfsTimeoutTripThreshold: ${tripThreshold} (must be an integer >= 1; 3 recommended)`
    );
  }

  const readConcurrency = cache.opfsReadConcurrency;
  if (!Number.isInteger(readConcurrency) || readConcurrency < 1) {
    errors.push(
      `Invalid cache.opfsReadConcurrency: ${readConcurrency} (must be an integer >= 1; 64 recommended)`
    );
  }

  // R1: the background L2 write queue caps. Concurrency 0/negative would stall
  // all persistence; a 0/negative depth would drop every write. NaN passes
  // `<= 0` (always false), so reject it explicitly.
  const writeConcurrency = cache.opfsWriteConcurrency;
  if (!Number.isFinite(writeConcurrency) || writeConcurrency <= 0) {
    errors.push(
      `Invalid cache.opfsWriteConcurrency: ${writeConcurrency} (must be a finite positive number; 4 recommended)`
    );
  }
  const writeQueueMax = cache.opfsWriteQueueMax;
  if (!Number.isFinite(writeQueueMax) || writeQueueMax <= 0) {
    errors.push(
      `Invalid cache.opfsWriteQueueMax: ${writeQueueMax} (must be a finite positive number; 16384 recommended)`
    );
  }

  // R1: externalDatasetTtlMs is allowed to be null (no TTL — content-hash
  // validation only). Anything else must be a finite positive number.
  // NaN passes `> 0` checks (always false), so reject it explicitly.
  const externalTtl = cache.externalDatasetTtlMs;
  if (externalTtl !== null && (!Number.isFinite(externalTtl) || externalTtl <= 0)) {
    errors.push(
      `Invalid cache.externalDatasetTtlMs: ${externalTtl} (must be null or a finite positive number)`
    );
  }
}
