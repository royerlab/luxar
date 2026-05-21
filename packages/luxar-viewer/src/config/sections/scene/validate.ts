import type { AppConfig } from '../../types';

/**
 * Validate scene configuration
 */
export function validateScene(config: AppConfig, errors: string[], _warnings: string[]): void {
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
