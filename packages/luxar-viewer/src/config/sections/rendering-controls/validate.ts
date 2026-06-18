import type { AppConfig } from '../../types';

/**
 * Validate rendering configuration
 */
export function validateRendering(config: AppConfig, errors: string[], _warnings: string[]): void {
  const defaults = config.renderingControls.defaults;

  // NaN/Infinity hardening for the global EOG values.
  if (!Number.isFinite(defaults.exposure) || defaults.exposure < -10 || defaults.exposure > 10) {
    errors.push(`Invalid exposure: ${defaults.exposure} (must be a finite number -10..10)`);
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
export function validateBloomConsistency(
  config: AppConfig,
  errors: string[],
  warnings: string[]
): void {
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
    errors.push(`Invalid bloom.bloomLevels: ${bloom.bloomLevels} (must be an integer in 1-12)`);
  }
}
