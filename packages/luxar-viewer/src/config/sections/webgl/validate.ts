import type { AppConfig } from '../../types';

/**
 * Validate WebGL configuration
 */
export function validateWebGL(config: AppConfig, errors: string[], warnings: string[]): void {
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
}
