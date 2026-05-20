import type { AppConfig } from '../../types';
import { validateDataLoadingNetwork } from './network/validate';
import { validateDataLoadingMemory } from './memory/validate';
import { validateDataLoadingPerformance } from './performance/validate';

/**
 * Validate data loading configuration. Composes the sub-section
 * validators (network, memory, performance) and inlines the trivial
 * spatial-section checks that don't warrant their own validate.ts.
 */
export function validateDataLoading(
  config: AppConfig,
  errors: string[],
  warnings: string[]
): void {
  const { dataLoading } = config;

  validateDataLoadingNetwork(config, errors, warnings);
  validateDataLoadingMemory(config, errors, warnings);

  // Spatial validation: NaN hardening (spatial is non-optional in DataLoadingConfig).
  const tol = dataLoading.spatial.defaultTolerance;
  if (!Number.isFinite(tol) || tol <= 0) {
    errors.push(`Invalid spatial default tolerance: ${tol} (must be a finite positive number)`);
  }
  const maxR = dataLoading.spatial.defaultMaxRadius;
  if (!Number.isFinite(maxR) || maxR <= 0) {
    errors.push(`Invalid spatial default max radius: ${maxR} (must be a finite positive number)`);
  }

  validateDataLoadingPerformance(config, errors, warnings);
}
