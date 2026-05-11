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

  // Each numeric check uses Number.isFinite() to reject NaN/Infinity.
  // Bare `<` / `>` against NaN is always false, so naked range
  // checks would let NaN pass silently.

  if (!Number.isFinite(defaults.fov) || defaults.fov < 1 || defaults.fov > 180) {
    errors.push(`Invalid camera FOV: ${defaults.fov} (must be a finite number 1..180)`);
  }

  if (!Number.isFinite(defaults.near) || defaults.near <= 0) {
    errors.push(`Invalid camera near plane: ${defaults.near} (must be a finite positive number)`);
  }
  if (!Number.isFinite(defaults.far) || defaults.far <= defaults.near) {
    errors.push(
      `Invalid camera far plane: ${defaults.far} (must be a finite number > near plane ${defaults.near})`
    );
  }

  if (
    !Number.isFinite(camera.fovMin) ||
    !Number.isFinite(camera.fovMax) ||
    camera.fovMin >= camera.fovMax
  ) {
    errors.push(`Invalid FOV limits: min ${camera.fovMin} >= max ${camera.fovMax}`);
  }

  if (
    !Number.isFinite(camera.fovSensitivity) ||
    camera.fovSensitivity <= 0 ||
    camera.fovSensitivity > 1
  ) {
    warnings.push(`Unusual FOV sensitivity: ${camera.fovSensitivity} (typical range 0.01-0.2)`);
  }
}

/**
 * Validate rendering configuration
 */
function validateRendering(config: AppConfig, errors: string[], _warnings: string[]): void {
  const defaults = config.renderingControls.defaults;

  // NaN/Infinity hardening for the global EOG values.
  if (!Number.isFinite(defaults.exposure) || defaults.exposure < -5 || defaults.exposure > 5) {
    errors.push(`Invalid exposure: ${defaults.exposure} (must be a finite number -5..5)`);
  }
  if (
    !Number.isFinite(defaults.globalOffset) ||
    defaults.globalOffset < -1 ||
    defaults.globalOffset > 1
  ) {
    errors.push(`Invalid globalOffset: ${defaults.globalOffset} (must be a finite number -1..1)`);
  }
  if (
    !Number.isFinite(defaults.globalGamma) ||
    defaults.globalGamma < 0.1 ||
    defaults.globalGamma > 10
  ) {
    errors.push(`Invalid globalGamma: ${defaults.globalGamma} (must be a finite number 0.1..10)`);
  }
}

/**
 * Validate bloom configuration consistency.
 *
 * NaN passes bare numeric comparisons (NaN < 0 is false, NaN > 10 is
 * false) so we use Number.isFinite explicitly. bloomLevels is also
 * required to be a positive integer in [1, 12].
 */
function validateBloomConsistency(config: AppConfig, errors: string[], warnings: string[]): void {
  const bloom = config.renderingControls.defaults;

  if (!Number.isFinite(bloom.bloomStrength)) {
    errors.push(`Invalid bloom.bloomStrength: ${bloom.bloomStrength} (must be finite)`);
  } else if (bloom.bloomStrength < 0 || bloom.bloomStrength > 10) {
    warnings.push(`Unusual bloom.bloomStrength: ${bloom.bloomStrength} (typical range 0-2)`);
  }

  if (!Number.isFinite(bloom.bloomRadius)) {
    errors.push(`Invalid bloom.bloomRadius: ${bloom.bloomRadius} (must be finite)`);
  } else if (bloom.bloomRadius < 0 || bloom.bloomRadius > 10) {
    warnings.push(`Unusual bloom.bloomRadius: ${bloom.bloomRadius} (typical range 0-2)`);
  }

  if (!Number.isFinite(bloom.bloomThreshold)) {
    errors.push(`Invalid bloom.bloomThreshold: ${bloom.bloomThreshold} (must be finite)`);
  } else if (bloom.bloomThreshold < 0 || bloom.bloomThreshold > 1) {
    errors.push(`Invalid bloom.bloomThreshold: ${bloom.bloomThreshold} (must be 0-1)`);
  }

  if (!Number.isInteger(bloom.bloomLevels) || bloom.bloomLevels < 1 || bloom.bloomLevels > 12) {
    errors.push(
      `Invalid bloom.bloomLevels: ${bloom.bloomLevels} (must be an integer in 1-12)`
    );
  }
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
    // NaN check on every range field — without this, any of
    // {min, max, default} could be NaN and silently pass.
    if (
      !Number.isFinite(range.min) ||
      !Number.isFinite(range.max) ||
      !Number.isFinite(range.default)
    ) {
      errors.push(
        `Invalid controls.${name}: non-finite values (min=${range.min}, max=${range.max}, default=${range.default})`
      );
      continue;
    }
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

  // Network validation: reject NaN (comparisons with NaN are always
  // false, so `<= 0` accepts it), Infinity, and non-integers where
  // integer semantics are required.
  if (
    !Number.isFinite(dataLoading.network.timeoutMs) ||
    dataLoading.network.timeoutMs <= 0
  ) {
    errors.push(
      `Invalid network timeout: ${dataLoading.network.timeoutMs} ms (must be a finite positive number)`
    );
  }
  // validationTimeoutMs is the per-request total budget for cache
  // validation in fetchWithRetry; 0 / negative / NaN / Infinity all
  // produce surprising abort/retry behavior, so reject up front.
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

  // Memory validation: reject NaN — `NaN <= 0` is always false, so a
  // bare `<= 0 || > 1` check would let NaN through.
  const targetHeap = dataLoading.memory.targetHeapUsage;
  if (!Number.isFinite(targetHeap) || targetHeap <= 0 || targetHeap > 1) {
    errors.push(
      `Invalid target heap usage: ${targetHeap} (must be a finite number in (0, 1])`
    );
  }
  const minCache = dataLoading.memory.minCacheMB;
  if (!Number.isFinite(minCache) || minCache <= 0) {
    errors.push(`Invalid min cache size: ${minCache} MB (must be a finite positive number)`);
  }

  // Spatial validation: same NaN hardening.
  if (dataLoading.spatial) {
    const tol = dataLoading.spatial.defaultTolerance;
    if (!Number.isFinite(tol) || tol <= 0) {
      errors.push(
        `Invalid spatial default tolerance: ${tol} (must be a finite positive number)`
      );
    }
    const maxR = dataLoading.spatial.defaultMaxRadius;
    if (!Number.isFinite(maxR) || maxR <= 0) {
      errors.push(
        `Invalid spatial default max radius: ${maxR} (must be a finite positive number)`
      );
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

  // `NaN < 0` is always false, so a bare range check would let NaN
  // through. Also require an integer color value.
  if (
    !Number.isInteger(scene.backgroundColor) ||
    scene.backgroundColor < 0 ||
    scene.backgroundColor > 0xffffff
  ) {
    errors.push(
      `Invalid scene background color: ${scene.backgroundColor} (must be an integer 0..0xffffff)`
    );
  }

  // Fit ratio validation. Same NaN hardening.
  if (
    !Number.isFinite(scene.defaultFitRatio) ||
    scene.defaultFitRatio <= 0 ||
    scene.defaultFitRatio > 1
  ) {
    errors.push(
      `Invalid scene fit ratio: ${scene.defaultFitRatio} (must be a finite number in (0, 1])`
    );
  }
}

/**
 * Validate input configuration
 */
function validateInput(config: AppConfig, errors: string[], warnings: string[]): void {
  const { input } = config;

  // NaN/Infinity hardening.
  if (!Number.isFinite(input.defaultSensitivity)) {
    errors.push(
      `Invalid input.defaultSensitivity: ${input.defaultSensitivity} (must be a finite number)`
    );
    return;
  }

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
