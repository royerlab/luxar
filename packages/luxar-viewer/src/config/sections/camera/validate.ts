import type { AppConfig } from '../../types';

/**
 * Validate camera configuration
 */
export function validateCamera(config: AppConfig, errors: string[], warnings: string[]): void {
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
