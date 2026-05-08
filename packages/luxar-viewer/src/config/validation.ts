/**
 * Configuration Validation Module
 *
 * Provides runtime validation for configuration values to catch errors early
 * and ensure configuration consistency.
 */

import type { AppConfig } from './types';
import { log, Modules } from '../utils/log';

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

  // Validate scene configuration
  validateScene(config, errors, warnings);

  // Validate input configuration
  validateInput(config, errors, warnings);

  // Validate WebGL configuration
  validateWebGL(config, errors, warnings);

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate camera configuration
 */
function validateCamera(config: AppConfig, errors: string[], warnings: string[]): void {
  const { camera } = config;
  const defaults = config.renderingControls.defaults;

  // FOV validation (lives in renderingControls.defaults)
  if (defaults.fov < 1 || defaults.fov > 180) {
    errors.push(`Invalid camera FOV: ${defaults.fov} (must be between 1 and 180)`);
  }

  // Near/far plane validation (lives in renderingControls.defaults)
  if (defaults.near <= 0) {
    errors.push(`Invalid camera near plane: ${defaults.near} (must be > 0)`);
  }
  if (defaults.far <= defaults.near) {
    errors.push(
      `Invalid camera far plane: ${defaults.far} (must be > near plane ${defaults.near})`
    );
  }

  // FOV min/max validation
  if (camera.fovMin >= camera.fovMax) {
    errors.push(`Invalid FOV limits: min ${camera.fovMin} >= max ${camera.fovMax}`);
  }

  // FOV sensitivity
  if (camera.fovSensitivity <= 0 || camera.fovSensitivity > 1) {
    warnings.push(`Unusual FOV sensitivity: ${camera.fovSensitivity} (typical range 0.01-0.2)`);
  }
}

/**
 * Validate rendering configuration
 */
function validateRendering(config: AppConfig, errors: string[], _warnings: string[]): void {
  const defaults = config.renderingControls.defaults;

  // Global EOG validation (lives in renderingControls.defaults)
  if (defaults.exposure < -5 || defaults.exposure > 5) {
    errors.push(`Invalid exposure: ${defaults.exposure} (must be -5 to 5)`);
  }
  if (defaults.globalOffset < -1 || defaults.globalOffset > 1) {
    errors.push(`Invalid globalOffset: ${defaults.globalOffset} (must be -1 to 1)`);
  }
  if (defaults.globalGamma < 0.1 || defaults.globalGamma > 10) {
    errors.push(`Invalid globalGamma: ${defaults.globalGamma} (must be 0.1 to 10)`);
  }
}

/**
 * Validate bloom configuration consistency
 */
function validateBloomConsistency(config: AppConfig, _errors: string[], warnings: string[]): void {
  const bloom = config.renderingControls.defaults;

  // Check bloom value ranges
  if (bloom.bloomStrength < 0 || bloom.bloomStrength > 10) {
    warnings.push(`Unusual bloom.bloomStrength: ${bloom.bloomStrength} (typical range 0-2)`);
  }
  if (bloom.bloomRadius < 0 || bloom.bloomRadius > 10) {
    warnings.push(`Unusual bloom.bloomRadius: ${bloom.bloomRadius} (typical range 0-2)`);
  }
  if (bloom.bloomThreshold < 0 || bloom.bloomThreshold > 1) {
    warnings.push(`Invalid bloom.bloomThreshold: ${bloom.bloomThreshold} (must be 0-1)`);
  }
  if (bloom.bloomLevels < 1 || bloom.bloomLevels > 12) {
    warnings.push(`Invalid bloom.bloomLevels: ${bloom.bloomLevels} (must be 1-12)`);
  }

  // No more duplication to check - single source of truth!
}

/**
 * Validate control configuration (ConfigRange consistency)
 */
function validateControls(config: AppConfig, errors: string[], _warnings: string[]): void {
  const { controls } = config;

  // Validate all ConfigRange objects: min < max and min <= default <= max
  const ranges: Array<{ name: string; range: { min: number; max: number; default: number } }> = [
    { name: 'fly.movement.speed', range: controls.fly.movement.speed },
    { name: 'fly.movement.acceleration', range: controls.fly.movement.acceleration },
    { name: 'fly.movement.damping', range: controls.fly.movement.damping },
    { name: 'fly.rotation.speed', range: controls.fly.rotation.speed },
    { name: 'fly.rotation.damping', range: controls.fly.rotation.damping },
    { name: 'orbit.autoRotate.speed', range: controls.orbit.autoRotate.speed },
    { name: 'orbit.zoom.speed', range: controls.orbit.zoom.speed },
    { name: 'orbit.damping.factor', range: controls.orbit.damping.factor },
  ];

  for (const { name, range } of ranges) {
    if (range.min >= range.max) {
      errors.push(`Invalid controls.${name}: min (${range.min}) >= max (${range.max})`);
    }
    if (range.default < range.min || range.default > range.max) {
      errors.push(
        `Invalid controls.${name}: default (${range.default}) outside [${range.min}, ${range.max}]`
      );
    }
  }
}

/**
 * Validate data loading configuration
 */
function validateDataLoading(config: AppConfig, errors: string[], _warnings: string[]): void {
  const { dataLoading } = config;

  // Network validation. Phase 14 hygiene: tighten to reject NaN
  // (comparisons with NaN are always false, so `<= 0` accepts it),
  // Infinity, and non-integers where integer semantics are required.
  if (
    !Number.isFinite(dataLoading.network.timeoutMs) ||
    dataLoading.network.timeoutMs <= 0
  ) {
    errors.push(
      `Invalid network timeout: ${dataLoading.network.timeoutMs} ms (must be a finite positive number)`
    );
  }
  // Phase 13.12: validationTimeoutMs is the per-request total budget
  // for cache validation in fetchWithRetry; 0 / negative / NaN /
  // Infinity all produce surprising abort/retry behavior, so reject
  // up front.
  if (
    !Number.isFinite(dataLoading.network.validationTimeoutMs) ||
    dataLoading.network.validationTimeoutMs <= 0
  ) {
    errors.push(
      `Invalid validation timeout: ${dataLoading.network.validationTimeoutMs} ms (must be a finite positive number)`
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

  // Memory validation
  if (dataLoading.memory.targetHeapUsage <= 0 || dataLoading.memory.targetHeapUsage > 1) {
    errors.push(`Invalid target heap usage: ${dataLoading.memory.targetHeapUsage} (must be 0-1)`);
  }
  if (dataLoading.memory.minCacheMB <= 0) {
    errors.push(`Invalid min cache size: ${dataLoading.memory.minCacheMB} MB (must be > 0)`);
  }

  // Spatial validation (new)
  if (dataLoading.spatial) {
    if (dataLoading.spatial.defaultTolerance <= 0) {
      errors.push(
        `Invalid spatial default tolerance: ${dataLoading.spatial.defaultTolerance} (must be > 0)`
      );
    }
    if (dataLoading.spatial.defaultMaxRadius <= 0) {
      errors.push(
        `Invalid spatial default max radius: ${dataLoading.spatial.defaultMaxRadius} (must be > 0)`
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

/**
 * Validate scene configuration
 */
function validateScene(config: AppConfig, errors: string[], _warnings: string[]): void {
  const { scene } = config;

  // Background color validation
  if (scene.backgroundColor < 0 || scene.backgroundColor > 0xffffff) {
    errors.push(
      `Invalid scene background color: ${scene.backgroundColor} (must be valid hex color)`
    );
  }

  // Fit ratio validation
  if (scene.defaultFitRatio <= 0 || scene.defaultFitRatio > 1) {
    errors.push(`Invalid scene fit ratio: ${scene.defaultFitRatio} (must be between 0 and 1)`);
  }
}

/**
 * Validate input configuration
 */
function validateInput(config: AppConfig, _errors: string[], warnings: string[]): void {
  const { input } = config;

  // Sensitivity validation
  if (input.defaultSensitivity <= 0 || input.defaultSensitivity > 1) {
    warnings.push(
      `Unusual input sensitivity: ${input.defaultSensitivity} (typical range 0.01-0.5)`
    );
  }
}

/**
 * Validate WebGL configuration
 */
function validateWebGL(config: AppConfig, errors: string[], warnings: string[]): void {
  const { webgl } = config;

  // Validate power preference
  const validPowerPreferences = ['high-performance', 'low-power', 'default'];
  if (!validPowerPreferences.includes(webgl.context.powerPreference)) {
    errors.push(
      `Invalid WebGL powerPreference: ${webgl.context.powerPreference} (must be one of: ${validPowerPreferences.join(', ')})`
    );
  }

  // Validate precision
  const validPrecisions = ['highp', 'mediump', 'lowp'];
  if (!validPrecisions.includes(webgl.renderer.precision)) {
    errors.push(
      `Invalid WebGL precision: ${webgl.renderer.precision} (must be one of: ${validPrecisions.join(', ')})`
    );
  }

  // Validate MSAA samples
  const validSamples = [0, 2, 4, 8];
  if (!validSamples.includes(webgl.renderTarget.samples)) {
    warnings.push(
      `Unusual MSAA samples: ${webgl.renderTarget.samples} (typical values: ${validSamples.join(', ')})`
    );
  }

  // Validate color space
  const validColorSpaces = ['srgb', 'display-p3', 'rec2020'];
  if (!validColorSpaces.includes(webgl.context.colorSpace)) {
    warnings.push(
      `Unusual color space: ${webgl.context.colorSpace} (typical values: ${validColorSpaces.join(', ')})`
    );
  }
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
