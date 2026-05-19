import type { AppConfig } from '../../../types';

/**
 * Validate data-loading memory configuration
 */
export function validateDataLoadingMemory(
  config: AppConfig,
  errors: string[],
  _warnings: string[]
): void {
  const memory = config.dataLoading.memory;

  // Memory validation: reject NaN — `NaN <= 0` is always false, so a
  // bare `<= 0 || > 1` check would let NaN through.
  const targetHeap = memory.targetHeapUsage;
  if (!Number.isFinite(targetHeap) || targetHeap <= 0 || targetHeap > 1) {
    errors.push(`Invalid target heap usage: ${targetHeap} (must be a finite number in (0, 1])`);
  }
  const minCache = memory.minCacheMB;
  if (!Number.isFinite(minCache) || minCache <= 0) {
    errors.push(`Invalid min cache size: ${minCache} MB (must be a finite positive number)`);
  }
}
