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
  const { shader } = config;
  const defaults = config.renderingControls.defaults;

  // HDR multiplier validation (lives in renderingControls.defaults)
  if (defaults.hdrMultiplier < 0) {
    errors.push(`Invalid HDR multiplier: ${defaults.hdrMultiplier} (must be >= 0)`);
  }
  if (shader.points.baseAlpha < 0 || shader.points.baseAlpha > 1) {
    errors.push(`Invalid base alpha: ${shader.points.baseAlpha} (must be between 0 and 1)`);
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

  // Network validation
  if (dataLoading.network.timeoutMs <= 0) {
    errors.push(`Invalid network timeout: ${dataLoading.network.timeoutMs} ms (must be > 0)`);
  }
  if (dataLoading.network.maxConcurrent <= 0) {
    errors.push(
      `Invalid max concurrent requests: ${dataLoading.network.maxConcurrent} (must be > 0)`
    );
  }
  if (dataLoading.network.retryAttempts < 0) {
    errors.push(`Invalid retry attempts: ${dataLoading.network.retryAttempts} (must be >= 0)`);
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
