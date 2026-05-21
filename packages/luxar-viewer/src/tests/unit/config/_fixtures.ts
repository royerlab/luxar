/**
 * Shared fixtures + helpers for per-section validator tests.
 *
 * The top-level `validateConfig` dispatcher wraps every per-section validator
 * with the void-returning `(cfg, errors, warnings)` contract; per-section tests
 * use `invokeValidator()` so assertions keep the same `{valid, errors, warnings}`
 * shape as dispatcher-level tests.
 */

import { config } from '../../../config';
import type { AppConfig } from '../../../config/types';
import type { ValidationResult } from '../../../config/validation';

/** Deep clone the production config for safe mutation. */
export function cloneConfig(): AppConfig {
  return JSON.parse(JSON.stringify(config)) as AppConfig;
}

/**
 * Invoke a single per-section validator with the production config (or a
 * mutated clone) and return the {valid, errors, warnings} shape that the
 * top-level dispatcher produces.
 */
export function invokeValidator(
  validator: (cfg: AppConfig, errors: string[], warnings: string[]) => void,
  cfg: AppConfig = cloneConfig()
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  validator(cfg, errors, warnings);
  return { valid: errors.length === 0, errors, warnings };
}
