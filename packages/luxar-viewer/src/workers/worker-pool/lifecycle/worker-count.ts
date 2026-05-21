/**
 * Resolve the configured worker count, capped by hardware concurrency.
 * Pure helper lifted from `WorkerPool.getConfiguredWorkerCount`.
 *
 * - `configCount = 0`: Auto mode, uses `(hardwareConcurrency - 1)`.
 * - `configCount > 0`: Uses that number, capped at `(hardwareConcurrency - 1)`.
 *
 * When `navigator.hardwareConcurrency` is unavailable, falls back to 4.
 * Always leaves at least one core for the main thread (rendering + UI).
 */
export function getConfiguredWorkerCount(configCount: number): number {
  const hardwareConcurrency =
    typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;

  // Leave one core for main thread (rendering, UI)
  const maxWorkers = Math.max(1, hardwareConcurrency - 1);

  // 0 = auto mode: use all available cores minus one
  if (configCount <= 0) {
    return maxWorkers;
  }

  // Otherwise use config value, capped at max
  return Math.min(configCount, maxWorkers);
}
