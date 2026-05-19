import type { AppConfig } from '../../types';

/**
 * Validate input configuration
 */
export function validateInput(config: AppConfig, errors: string[], warnings: string[]): void {
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
