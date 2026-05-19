import type { AppConfig } from '../../types';

/**
 * Validate data loading configuration (including cache + performance + spatial + network + memory)
 *
 * Cache validation lives here for now; sub-section split will happen in Step 3.
 */
export function validateDataLoading(
  config: AppConfig,
  errors: string[],
  warnings: string[]
): void {
  const { dataLoading } = config;

  // Network validation: reject NaN (comparisons with NaN are always
  // false, so `<= 0` accepts it), Infinity, and non-integers where
  // integer semantics are required.
  if (!Number.isFinite(dataLoading.network.timeoutMs) || dataLoading.network.timeoutMs <= 0) {
    errors.push(
      `Invalid network timeout: ${dataLoading.network.timeoutMs} ms (must be a finite positive number)`
    );
  }
  // validationTimeoutMs is the per-request total budget for cache
  // validation in fetchWithRetry; 0 / negative / NaN / Infinity all
  // produce surprising abort/retry behavior, so reject up front.
  if (
    !Number.isFinite(dataLoading.network.validationTimeoutMs) ||
    dataLoading.network.validationTimeoutMs <= 0
  ) {
    errors.push(
      `Invalid validation timeout: ${dataLoading.network.validationTimeoutMs} ms (must be a finite positive number)`
    );
  } else if (dataLoading.network.validationTimeoutMs < 3000) {
    // Soft warning, not a hard error. The documented default (5 s) is
    // a fail-fast budget tuned for broadband; values under 3 s are
    // almost always too aggressive — every round-trip including DNS,
    // TLS, and server processing must complete in that window or the
    // validation aborts and forces a re-fetch of otherwise-valid
    // cached data. For 3G / Edge / high-latency targets, raise to
    // >=8000 instead. See
    // `DataLoadingNetworkConfig.validationTimeoutMs` JSDoc.
    warnings.push(
      `Very low cache validation timeout: ${dataLoading.network.validationTimeoutMs} ms ` +
        '(values <3000 ms cause spurious validation aborts; consider 5000 ms default ' +
        'or >=8000 ms for 3G/Edge targets)'
    );
  }
  if (
    !Number.isInteger(dataLoading.network.maxConcurrent) ||
    dataLoading.network.maxConcurrent <= 0
  ) {
    errors.push(
      `Invalid max concurrent requests: ${dataLoading.network.maxConcurrent} (must be a positive integer)`
    );
  }
  if (
    !Number.isInteger(dataLoading.network.retryAttempts) ||
    dataLoading.network.retryAttempts < 0
  ) {
    errors.push(
      `Invalid retry attempts: ${dataLoading.network.retryAttempts} (must be a non-negative integer)`
    );
  }

  // Memory validation: reject NaN — `NaN <= 0` is always false, so a
  // bare `<= 0 || > 1` check would let NaN through.
  const targetHeap = dataLoading.memory.targetHeapUsage;
  if (!Number.isFinite(targetHeap) || targetHeap <= 0 || targetHeap > 1) {
    errors.push(`Invalid target heap usage: ${targetHeap} (must be a finite number in (0, 1])`);
  }
  const minCache = dataLoading.memory.minCacheMB;
  if (!Number.isFinite(minCache) || minCache <= 0) {
    errors.push(`Invalid min cache size: ${minCache} MB (must be a finite positive number)`);
  }

  // Spatial validation: same NaN hardening.
  if (dataLoading.spatial) {
    const tol = dataLoading.spatial.defaultTolerance;
    if (!Number.isFinite(tol) || tol <= 0) {
      errors.push(`Invalid spatial default tolerance: ${tol} (must be a finite positive number)`);
    }
    const maxR = dataLoading.spatial.defaultMaxRadius;
    if (!Number.isFinite(maxR) || maxR <= 0) {
      errors.push(`Invalid spatial default max radius: ${maxR} (must be a finite positive number)`);
    }
  }

  // Cache size validation: NaN / Infinity / negative values would
  // cascade into the cache layer sizing logic and surface as cryptic
  // OOMs or zero-budget caches.
  const cache = config.cache;
  if (cache) {
    const l0 = cache.l0MaxSizeMB;
    if (!Number.isFinite(l0) || l0 <= 0) {
      errors.push(`Invalid cache.l0MaxSizeMB: ${l0} (must be a finite positive number)`);
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

  // Worker timeouts: 0 disables; otherwise must be a finite positive number
  // (we don't restrict the upper bound — long-running fits can legitimately
  // exceed any "sane" ceiling).
  const visTimeout = dataLoading.performance.workerVisibilityTimeoutMs;
  if (!Number.isFinite(visTimeout) || visTimeout < 0) {
    errors.push(
      `Invalid workerVisibilityTimeoutMs: ${visTimeout} (must be ≥ 0; 0 disables timeout)`
    );
  }
  const projTimeout = dataLoading.performance.workerProjectionTimeoutMs;
  if (!Number.isFinite(projTimeout) || projTimeout < 0) {
    errors.push(
      `Invalid workerProjectionTimeoutMs: ${projTimeout} (must be ≥ 0; 0 disables timeout)`
    );
  }
  // Init timeout: must be a finite positive number; 0 disables, but the
  // intent is the opposite of per-call timeouts — without an init guard
  // a blocked worker chunk hangs the page indefinitely. We allow 0 only
  // for tests that need to disable it.
  const initTimeout = dataLoading.performance.workerInitTimeoutMs;
  if (!Number.isFinite(initTimeout) || initTimeout < 0) {
    errors.push(
      `Invalid workerInitTimeoutMs: ${initTimeout} (must be ≥ 0; 0 disables, but the guard is recommended)`
    );
  }
}
