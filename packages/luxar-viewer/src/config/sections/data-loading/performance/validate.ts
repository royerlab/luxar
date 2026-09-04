import type { AppConfig } from '../../../types';

/**
 * Validate data-loading performance (worker timeouts) configuration
 */
export function validateDataLoadingPerformance(
  config: AppConfig,
  errors: string[],
  _warnings: string[]
): void {
  const performance = config.dataLoading.performance;

  // Worker timeouts: 0 disables; otherwise must be a finite positive number
  // (we don't restrict the upper bound — long-running fits can legitimately
  // exceed any "sane" ceiling).
  const projTimeout = performance.workerProjectionTimeoutMs;
  if (!Number.isFinite(projTimeout) || projTimeout < 0) {
    errors.push(
      `Invalid workerProjectionTimeoutMs: ${projTimeout} (must be ≥ 0; 0 disables timeout)`
    );
  }
  // Init timeout: must be a finite positive number; 0 disables, but the
  // intent is the opposite of per-call timeouts — without an init guard
  // a blocked worker chunk hangs the page indefinitely. We allow 0 only
  // for tests that need to disable it.
  const initTimeout = performance.workerInitTimeoutMs;
  if (!Number.isFinite(initTimeout) || initTimeout < 0) {
    errors.push(
      `Invalid workerInitTimeoutMs: ${initTimeout} (must be ≥ 0; 0 disables, but the guard is recommended)`
    );
  }
}
