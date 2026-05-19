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

  validateDataLoadingPerformance(config, errors, warnings);
}
