import type { AppConfig } from '../../types';
import { validateDataLoadingNetwork } from './network/validate';
import { validateDataLoadingMemory } from './memory/validate';
import { validateDataLoadingPerformance } from './performance/validate';

/**
 * Validate data loading configuration (composes sub-section validators
 * for the parts that have already been extracted, plus inline blocks
 * for the rest until they're split in Step 3).
 */
export function validateDataLoading(
  config: AppConfig,
  errors: string[],
  warnings: string[]
): void {
  const { dataLoading } = config;

  validateDataLoadingNetwork(config, errors, warnings);
  validateDataLoadingMemory(config, errors, warnings);

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

  validateDataLoadingPerformance(config, errors, warnings);
}
