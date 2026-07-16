/**
 * Configuration Validation Module
 *
 * Provides runtime validation for configuration values to catch errors early
 * and ensure configuration consistency.
 */

import type { AppConfig } from './types';
import { log, Modules } from '../utils/log';
import { validateCamera } from './sections/camera/validate';
import { validateScene } from './sections/scene/validate';
import { validateWebGL } from './sections/webgl/validate';
import { validateInput } from './sections/input/validate';
import { validateControls } from './sections/controls/validate';
import {
  validateRendering,
  validateBloomConsistency,
} from './sections/rendering-controls/validate';
import { validateDataLoading } from './sections/data-loading/validate';
import { validateCache } from './sections/cache/validate';
import { validateAdaptiveDPR } from './sections/adaptive-dpr/validate';
import { validateDepthSort } from './sections/depth-sort/validate';

/**
 * Validation result interface
 */
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Validates the entire application configuration
 */
export function validateConfig(config: AppConfig): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Validate camera configuration
  validateCamera(config, errors, warnings);

  // Validate rendering configuration
  validateRendering(config, errors, warnings);

  // Validate bloom configuration consistency
  validateBloomConsistency(config, errors, warnings);

  // Validate control configuration (ConfigRange consistency)
  validateControls(config, errors, warnings);

  // Validate data loading configuration
  validateDataLoading(config, errors, warnings);

  // Validate cache configuration
  validateCache(config, errors, warnings);

  // Validate scene configuration
  validateScene(config, errors, warnings);

  // Validate input configuration
  validateInput(config, errors, warnings);

  // Validate WebGL configuration
  validateWebGL(config, errors, warnings);

  // Validate adaptive DPR control-loop configuration
  validateAdaptiveDPR(config, errors, warnings);

  // Validate depth-sort scheduling configuration
  validateDepthSort(config, errors, warnings);

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Log validation results
 */
export function logValidationResults(result: ValidationResult): void {
  if (result.valid) {
    log.success(Modules.LUXAR, 'Configuration validation passed');
  } else {
    log.error(Modules.LUXAR, `Configuration validation failed with ${result.errors.length} errors`);
    result.errors.forEach((error) => {
      log.error(Modules.LUXAR, `  ${error}`);
    });
  }

  if (result.warnings.length > 0) {
    log.warning(Modules.LUXAR, `Configuration has ${result.warnings.length} warnings`);
    result.warnings.forEach((warning) => {
      log.warning(Modules.LUXAR, `  ${warning}`);
    });
  }
}

/**
 * Validate configuration at runtime with automatic logging
 */
export function validateAndLog(config: AppConfig): boolean {
  const result = validateConfig(config);
  logValidationResults(result);
  return result.valid;
}
