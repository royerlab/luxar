/**
 * Resolve the configured worker count, capped by hardware concurrency and,
 * on a phone or tablet, by a fixed mobile ceiling.
 * Pure helper lifted from `WorkerPool.getConfiguredWorkerCount`.
 *
 * - `configCount = 0`: Auto mode, uses `(hardwareConcurrency - 1)`.
 * - `configCount > 0`: Uses that number, capped at `(hardwareConcurrency - 1)`.
 *
 * When `navigator.hardwareConcurrency` is unavailable, falls back to 4.
 * Always leaves at least one core for the main thread (rendering + UI).
 *
 * Mobile: every data worker instantiates its own WASM linear memory and grows
 * its own accumulator buffers, and on iOS each realm counts against the tab's
 * jetsam budget alongside the dedicated sort worker. `MOBILE_MAX_WORKERS`
 * bounds the pool there regardless of the reported core count (an iPhone
 * reports 6; a flagship Android up to 8).
 */

import { getInputProfile } from '../../../utils/input-capabilities';

/** Data-worker pool ceiling on a mobile device class. */
export const MOBILE_MAX_WORKERS = 3;

export function getConfiguredWorkerCount(configCount: number): number {
  const hardwareConcurrency =
    typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;

  // Leave one core for main thread (rendering, UI); on mobile also respect the
  // fixed ceiling.
  const deviceCap = getInputProfile().deviceClass === 'mobile' ? MOBILE_MAX_WORKERS : Infinity;
  const maxWorkers = Math.max(1, Math.min(hardwareConcurrency - 1, deviceCap));

  // 0 = auto mode: use all available cores minus one
  if (configCount <= 0) {
    return maxWorkers;
  }

  // Otherwise use config value, capped at max
  return Math.min(configCount, maxWorkers);
}
